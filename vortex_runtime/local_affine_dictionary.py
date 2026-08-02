from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Iterable, Literal, Mapping

import torch
from torch import nn

from vortex_runtime.capsule_quantization import fake_quantize_columns


DictionaryMode = Literal["exact", "dictionary"]


@dataclass(frozen=True)
class LocalClusterStats:
    cluster_index: int
    vectors: int
    numerical_rank: int
    compiled_rank: int
    input_reconstruction_relative_error: float
    output_reconstruction_relative_error: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class LocalDictionaryStats:
    vectors: int
    clusters: int
    requested_local_rank: int
    active_rank_maximum: int
    stored_response_columns: int
    routing_centroid_columns: int
    training_input_reconstruction_relative_error: float
    training_output_reconstruction_relative_error: float
    cluster_counts: tuple[int, ...]
    cluster_stats: tuple[LocalClusterStats, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "cluster_stats": [item.to_dict() for item in self.cluster_stats],
        }


@dataclass(frozen=True)
class LocalDictionaryQuantizationStats:
    bits: int
    tensors: int
    elements: int
    logical_payload_bytes: int
    scale_bytes: int
    logical_total_bytes: int
    maximum_absolute_error: float
    maximum_relative_l2_error: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class LocalAffineDictionary:
    input_centroids: torch.Tensor
    output_centroids: torch.Tensor
    input_bases: torch.Tensor
    output_images: torch.Tensor
    ranks: torch.Tensor
    cluster_counts: torch.Tensor

    @property
    def clusters(self) -> int:
        return int(self.input_centroids.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.input_centroids.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.output_centroids.shape[1])

    @property
    def local_rank_limit(self) -> int:
        return int(self.input_bases.shape[2])

    @property
    def stored_response_columns(self) -> int:
        return self.clusters + int(self.ranks.sum().item())

    @property
    def active_response_columns(self) -> int:
        return 1 + int(self.ranks.max().item())

    @property
    def routing_centroid_columns(self) -> int:
        return self.clusters

    def validate(self) -> None:
        k = self.clusters
        if self.output_centroids.shape[0] != k:
            raise ValueError("input and output centroid counts differ")
        if self.input_bases.shape[:2] != (k, self.in_features):
            raise ValueError("input basis shape is inconsistent")
        if self.output_images.shape[:2] != (k, self.out_features):
            raise ValueError("output image shape is inconsistent")
        if self.input_bases.shape[2] != self.output_images.shape[2]:
            raise ValueError("input and output local rank limits differ")
        if self.ranks.shape != (k,) or self.cluster_counts.shape != (k,):
            raise ValueError("rank/count metadata shape is inconsistent")
        if torch.any(self.ranks < 0) or torch.any(
            self.ranks > self.local_rank_limit
        ):
            raise ValueError("local rank metadata is out of range")

    def route(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError("dictionary input dimension mismatch")
        flat = x.reshape(-1, self.in_features)
        centroids = self.input_centroids.to(
            device=x.device,
            dtype=x.dtype,
        )
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2.0 * flat @ centroids.T
            + centroids.square().sum(dim=1).unsqueeze(0)
        )
        return torch.argmin(distances, dim=1)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        self.validate()
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        assignments = self.route(x)
        output = torch.empty(
            (flat.shape[0], self.out_features),
            device=x.device,
            dtype=x.dtype,
        )
        for cluster in range(self.clusters):
            mask = assignments == cluster
            if not bool(mask.any()):
                continue
            center = self.input_centroids[cluster].to(
                device=x.device,
                dtype=x.dtype,
            )
            output_center = self.output_centroids[cluster].to(
                device=x.device,
                dtype=x.dtype,
            )
            rows = flat[mask]
            rank = int(self.ranks[cluster].item())
            if rank == 0:
                output[mask] = output_center
                continue
            basis = self.input_bases[cluster, :, :rank].to(
                device=x.device,
                dtype=x.dtype,
            )
            image = self.output_images[cluster, :, :rank].to(
                device=x.device,
                dtype=x.dtype,
            )
            coordinates = (rows - center) @ basis
            output[mask] = output_center + coordinates @ image.T
        return output.reshape(*original_shape, self.out_features)


class LocalAffineDictionaryLinearModule(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.exact = linear
        self.dictionary: LocalAffineDictionary | None = None
        self.mode: DictionaryMode = "exact"

    def set_mode(self, mode: DictionaryMode) -> None:
        if mode not in {"exact", "dictionary"}:
            raise ValueError(f"unsupported dictionary mode: {mode}")
        self.mode = mode

    def configure_dictionary(self, dictionary: LocalAffineDictionary) -> None:
        dictionary.validate()
        if dictionary.in_features != self.exact.in_features:
            raise ValueError("dictionary input dimension does not match module")
        if dictionary.out_features != self.exact.out_features:
            raise ValueError("dictionary output dimension does not match module")
        self.dictionary = dictionary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "exact":
            return self.exact(x)
        if self.dictionary is None:
            raise RuntimeError("local affine dictionary is not configured")
        return self.dictionary.apply(x)


def _relative_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(reference - estimate)
    denominator = torch.linalg.vector_norm(reference)
    return float((numerator / torch.clamp(denominator, min=1e-12)).item())


def deterministic_kmeans(
    vectors: torch.Tensor,
    *,
    clusters: int,
    iterations: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("k-means vectors must be a non-empty matrix")
    if clusters <= 0 or clusters > vectors.shape[0]:
        raise ValueError("cluster count must be in [1, number of vectors]")
    if iterations <= 0:
        raise ValueError("k-means iterations must be positive")

    data = vectors.detach().to("cpu", torch.float32)
    first = int(torch.argmax(data.square().sum(dim=1)).item())
    center_indices = [first]
    nearest = (data - data[first]).square().sum(dim=1)
    for _ in range(1, clusters):
        index = int(torch.argmax(nearest).item())
        center_indices.append(index)
        distance = (data - data[index]).square().sum(dim=1)
        nearest = torch.minimum(nearest, distance)
    centroids = data[center_indices].clone()

    assignments = torch.zeros(data.shape[0], dtype=torch.long)
    for _ in range(iterations):
        distances = (
            data.square().sum(dim=1, keepdim=True)
            - 2.0 * data @ centroids.T
            + centroids.square().sum(dim=1).unsqueeze(0)
        )
        assignments = torch.argmin(distances, dim=1)
        minimum_distances = distances.gather(1, assignments[:, None]).squeeze(1)
        updated = centroids.clone()
        used_replacements: set[int] = set()
        for cluster in range(clusters):
            mask = assignments == cluster
            if bool(mask.any()):
                updated[cluster] = data[mask].mean(dim=0)
                continue
            ordered = torch.argsort(minimum_distances, descending=True)
            replacement = next(
                int(index.item())
                for index in ordered
                if int(index.item()) not in used_replacements
            )
            used_replacements.add(replacement)
            updated[cluster] = data[replacement]
        if torch.equal(updated, centroids):
            break
        centroids = updated

    distances = (
        data.square().sum(dim=1, keepdim=True)
        - 2.0 * data @ centroids.T
        + centroids.square().sum(dim=1).unsqueeze(0)
    )
    assignments = torch.argmin(distances, dim=1)
    return centroids.contiguous(), assignments.contiguous()


def build_local_affine_dictionary(
    *,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    clusters: int,
    local_rank: int,
    rank_rtol: float = 1e-6,
    kmeans_iterations: int = 12,
) -> tuple[LocalAffineDictionary, LocalDictionaryStats]:
    if input_tensor.shape[:-1] != output_tensor.shape[:-1]:
        raise ValueError("input and output leading dimensions must match")
    if local_rank < 0:
        raise ValueError("local rank must be non-negative")
    if rank_rtol < 0:
        raise ValueError("rank tolerance must be non-negative")

    inputs = input_tensor.detach().to("cpu", torch.float32).reshape(
        -1,
        input_tensor.shape[-1],
    )
    outputs = output_tensor.detach().to("cpu", torch.float32).reshape(
        -1,
        output_tensor.shape[-1],
    )
    centroids, assignments = deterministic_kmeans(
        inputs,
        clusters=clusters,
        iterations=kmeans_iterations,
    )

    output_centroids = torch.empty(
        (clusters, outputs.shape[1]),
        dtype=torch.float32,
    )
    input_bases = torch.zeros(
        (clusters, inputs.shape[1], local_rank),
        dtype=torch.float32,
    )
    output_images = torch.zeros(
        (clusters, outputs.shape[1], local_rank),
        dtype=torch.float32,
    )
    ranks = torch.zeros(clusters, dtype=torch.long)
    counts = torch.zeros(clusters, dtype=torch.long)
    cluster_stats: list[LocalClusterStats] = []

    training_estimate = torch.empty_like(outputs)
    for cluster in range(clusters):
        mask = assignments == cluster
        cluster_inputs = inputs[mask]
        cluster_outputs = outputs[mask]
        count = int(cluster_inputs.shape[0])
        if count == 0:
            raise RuntimeError("deterministic k-means produced an empty cluster")
        counts[cluster] = count
        input_center = cluster_inputs.mean(dim=0)
        output_center = cluster_outputs.mean(dim=0)
        centroids[cluster] = input_center
        output_centroids[cluster] = output_center
        input_residual = cluster_inputs - input_center
        output_residual = cluster_outputs - output_center

        if local_rank == 0 or count == 1:
            numerical_rank = 0
            compiled_rank = 0
            estimate = output_center.expand_as(cluster_outputs)
            input_estimate = input_center.expand_as(cluster_inputs)
        else:
            _, singular_values, vh = torch.linalg.svd(
                input_residual,
                full_matrices=False,
            )
            threshold = (
                0.0
                if singular_values.numel() == 0
                else float(singular_values[0].item()) * rank_rtol
            )
            numerical_rank = int(
                torch.count_nonzero(singular_values > threshold).item()
            )
            compiled_rank = min(local_rank, numerical_rank)
            if compiled_rank == 0:
                estimate = output_center.expand_as(cluster_outputs)
                input_estimate = input_center.expand_as(cluster_inputs)
            else:
                basis = vh[:compiled_rank].T.contiguous()
                coordinates = input_residual @ basis
                image = torch.linalg.lstsq(
                    coordinates,
                    output_residual,
                ).solution.T.contiguous()
                input_bases[cluster, :, :compiled_rank] = basis
                output_images[cluster, :, :compiled_rank] = image
                ranks[cluster] = compiled_rank
                input_estimate = input_center + coordinates @ basis.T
                estimate = output_center + coordinates @ image.T

        training_estimate[mask] = estimate
        cluster_stats.append(
            LocalClusterStats(
                cluster_index=cluster,
                vectors=count,
                numerical_rank=numerical_rank,
                compiled_rank=compiled_rank,
                input_reconstruction_relative_error=_relative_error(
                    cluster_inputs,
                    input_estimate,
                ),
                output_reconstruction_relative_error=_relative_error(
                    cluster_outputs,
                    estimate,
                ),
            )
        )

    dictionary = LocalAffineDictionary(
        input_centroids=centroids.contiguous(),
        output_centroids=output_centroids.contiguous(),
        input_bases=input_bases.contiguous(),
        output_images=output_images.contiguous(),
        ranks=ranks.contiguous(),
        cluster_counts=counts.contiguous(),
    )
    dictionary.validate()
    routed_estimate = dictionary.apply(inputs)
    stats = LocalDictionaryStats(
        vectors=int(inputs.shape[0]),
        clusters=clusters,
        requested_local_rank=local_rank,
        active_rank_maximum=dictionary.active_response_columns,
        stored_response_columns=dictionary.stored_response_columns,
        routing_centroid_columns=dictionary.routing_centroid_columns,
        training_input_reconstruction_relative_error=max(
            item.input_reconstruction_relative_error for item in cluster_stats
        ),
        training_output_reconstruction_relative_error=_relative_error(
            outputs,
            routed_estimate,
        ),
        cluster_counts=tuple(int(value) for value in counts.tolist()),
        cluster_stats=tuple(cluster_stats),
    )
    return dictionary, stats


def quantize_local_affine_dictionary(
    dictionary: LocalAffineDictionary,
    *,
    bits: int,
    scale_bits: int = 16,
) -> LocalDictionaryQuantizationStats:
    dictionary.validate()
    tensor_stats = []

    input_centroids, stats = fake_quantize_columns(
        dictionary.input_centroids.T,
        bits=bits,
        scale_bits=scale_bits,
    )
    dictionary.input_centroids = input_centroids.T.contiguous()
    tensor_stats.append(stats)
    output_centroids, stats = fake_quantize_columns(
        dictionary.output_centroids.T,
        bits=bits,
        scale_bits=scale_bits,
    )
    dictionary.output_centroids = output_centroids.T.contiguous()
    tensor_stats.append(stats)

    for cluster in range(dictionary.clusters):
        rank = int(dictionary.ranks[cluster].item())
        if rank == 0:
            continue
        basis, stats = fake_quantize_columns(
            dictionary.input_bases[cluster, :, :rank],
            bits=bits,
            scale_bits=scale_bits,
        )
        dictionary.input_bases[cluster, :, :rank] = basis
        tensor_stats.append(stats)
        image, stats = fake_quantize_columns(
            dictionary.output_images[cluster, :, :rank],
            bits=bits,
            scale_bits=scale_bits,
        )
        dictionary.output_images[cluster, :, :rank] = image
        tensor_stats.append(stats)

    elements = sum(item.elements for item in tensor_stats)
    payload = sum(item.logical_payload_bytes for item in tensor_stats)
    scales = sum(item.scale_bytes for item in tensor_stats)
    return LocalDictionaryQuantizationStats(
        bits=bits,
        tensors=len(tensor_stats),
        elements=elements,
        logical_payload_bytes=payload,
        scale_bytes=scales,
        logical_total_bytes=payload + scales,
        maximum_absolute_error=max(
            item.maximum_absolute_error for item in tensor_stats
        ),
        maximum_relative_l2_error=max(
            item.relative_l2_error for item in tensor_stats
        ),
    )


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_with_local_affine_dictionary_modules(
    model: nn.Module,
    *,
    suffixes: Iterable[str],
) -> dict[str, LocalAffineDictionaryLinearModule]:
    selected = tuple(suffixes)
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(suffix) for suffix in selected
        ):
            matches.append((name, module))

    replacements: dict[str, LocalAffineDictionaryLinearModule] = {}
    for name, linear in matches:
        parent, attribute = _resolve_parent(model, name)
        wrapper = LocalAffineDictionaryLinearModule(linear)
        setattr(parent, attribute, wrapper)
        replacements[name] = wrapper
    return replacements


def build_local_affine_dictionaries(
    replacements: Mapping[str, LocalAffineDictionaryLinearModule],
    *,
    captured_inputs: Mapping[str, torch.Tensor],
    captured_outputs: Mapping[str, torch.Tensor],
    clusters: int,
    local_rank: int,
    rank_rtol: float = 1e-6,
    kmeans_iterations: int = 12,
) -> dict[str, LocalDictionaryStats]:
    if not replacements:
        raise ValueError("at least one dictionary module is required")
    result: dict[str, LocalDictionaryStats] = {}
    for name, module in replacements.items():
        if name not in captured_inputs or name not in captured_outputs:
            raise RuntimeError(f"missing exact dictionary capture for {name}")
        dictionary, stats = build_local_affine_dictionary(
            input_tensor=captured_inputs[name],
            output_tensor=captured_outputs[name],
            clusters=clusters,
            local_rank=local_rank,
            rank_rtol=rank_rtol,
            kmeans_iterations=kmeans_iterations,
        )
        module.configure_dictionary(dictionary)
        result[name] = stats
    return result
