from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch


class PrefixBisimulationError(RuntimeError):
    """Fail-closed violation of the prefix-state witness contract."""


@dataclass(frozen=True)
class PrefixRNGState:
    mode: str
    generator_state: bytes = b""
    seed: int | None = None
    temperature: float = 1.0
    top_k: int = 0

    @staticmethod
    def greedy() -> "PrefixRNGState":
        return PrefixRNGState(mode="greedy")

    @staticmethod
    def sampled(
        seed: int,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> "PrefixRNGState":
        if not torch.isfinite(torch.tensor(float(temperature))) or temperature <= 0:
            raise PrefixBisimulationError("temperature must be finite and positive")
        if int(top_k) < 0:
            raise PrefixBisimulationError("top_k must be nonnegative")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        state = generator.get_state().detach().contiguous().cpu().numpy().tobytes()
        return PrefixRNGState(
            mode="sample",
            generator_state=state,
            seed=int(seed),
            temperature=float(temperature),
            top_k=int(top_k),
        )

    def digest(self) -> str:
        payload = {
            "mode": self.mode,
            "state_sha256": hashlib.sha256(self.generator_state).hexdigest(),
            "seed": self.seed,
            "temperature": self.temperature,
            "top_k": self.top_k,
        }
        return sha256_json(payload)


@dataclass(frozen=True)
class PrefixCompiledState:
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    rng_state: PrefixRNGState
    step: int

    @property
    def prefix_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.generated_ids

    @property
    def position(self) -> int:
        return len(self.prefix_ids)

    def digest(self) -> str:
        payload = {
            "prompt_ids": list(self.prompt_ids),
            "generated_ids": list(self.generated_ids),
            "rng": {
                **asdict(self.rng_state),
                "generator_state": hashlib.sha256(
                    self.rng_state.generator_state
                ).hexdigest(),
            },
            "step": self.step,
        }
        return sha256_json(payload)


@dataclass
class ReplayObservation:
    logits: torch.Tensor
    past_key_values: Any
    target_calls: int
    replayed_token_positions: int
    cache_sha256: str
    cache_bytes: int
    logits_sha256: str


@dataclass
class PrefixDecodeTrace:
    step: int
    prefix_sha256: str
    prefix_length: int
    selected_token: int
    rng_before_sha256: str
    rng_after_sha256: str
    logits_sha256: str
    cache_sha256: str
    cache_bytes: int
    target_calls: int
    replayed_token_positions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def first_byte_mismatch(left: bytes, right: bytes) -> dict[str, int] | None:
    if len(left) != len(right):
        return {"offset": -1, "left_length": len(left), "right_length": len(right)}
    for offset, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return {
                "offset": int(offset),
                "left_byte": int(lhs),
                "right_byte": int(rhs),
                "left_length": len(left),
                "right_length": len(right),
            }
    return None


def _flatten_cache(value: Any, *, path: str = "cache") -> list[tuple[str, torch.Tensor]]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if hasattr(value, "to_legacy_cache"):
        return _flatten_cache(value.to_legacy_cache(), path=path)
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        pairs: list[tuple[str, torch.Tensor]] = []
        for index, tensor in enumerate(value.key_cache):
            if isinstance(tensor, torch.Tensor):
                pairs.append((f"{path}.key.{index}", tensor))
        for index, tensor in enumerate(value.value_cache):
            if isinstance(tensor, torch.Tensor):
                pairs.append((f"{path}.value.{index}", tensor))
        return pairs
    if isinstance(value, (tuple, list)):
        pairs = []
        for index, item in enumerate(value):
            pairs.extend(_flatten_cache(item, path=f"{path}.{index}"))
        return pairs
    raise PrefixBisimulationError(
        f"unsupported cache object at {path}: {type(value).__name__}"
    )


def cache_manifest(cache: Any) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": len(tensor_bytes(tensor)),
            "sha256": tensor_sha256(tensor),
        }
        for path, tensor in _flatten_cache(cache)
    ]


def cache_sha256(cache: Any) -> str:
    digest = hashlib.sha256()
    for row in cache_manifest(cache):
        digest.update(row["path"].encode("utf-8"))
        digest.update(json.dumps(row["shape"], separators=(",", ":")).encode())
        digest.update(row["dtype"].encode("utf-8"))
        digest.update(bytes.fromhex(row["sha256"]))
    return digest.hexdigest()


def cache_nbytes(cache: Any) -> int:
    return sum(int(row["bytes"]) for row in cache_manifest(cache))


def initialize_prefix_state(
    prompt_ids: Sequence[int], rng_state: PrefixRNGState
) -> PrefixCompiledState:
    ids = tuple(int(value) for value in prompt_ids)
    if not ids:
        raise PrefixBisimulationError("prompt token sequence must be nonempty")
    if any(value < 0 for value in ids):
        raise PrefixBisimulationError("token IDs must be nonnegative")
    return PrefixCompiledState(ids, (), rng_state, 0)


def append_exact_token(
    state: PrefixCompiledState,
    token: int,
    next_rng_state: PrefixRNGState,
) -> PrefixCompiledState:
    if int(token) < 0:
        raise PrefixBisimulationError("selected token must be nonnegative")
    if state.step != len(state.generated_ids):
        raise PrefixBisimulationError("prefix state step invariant failed")
    return PrefixCompiledState(
        state.prompt_ids,
        state.generated_ids + (int(token),),
        next_rng_state,
        state.step + 1,
    )


def _generator_from_bytes(state: bytes) -> torch.Generator:
    if not state:
        raise PrefixBisimulationError("sample RNG state is empty")
    generator = torch.Generator(device="cpu")
    tensor = torch.frombuffer(bytearray(state), dtype=torch.uint8).clone()
    generator.set_state(tensor)
    return generator


def _generator_bytes(generator: torch.Generator) -> bytes:
    return generator.get_state().detach().contiguous().cpu().numpy().tobytes()


def select_token(
    logits: torch.Tensor, rng_state: PrefixRNGState
) -> tuple[int, PrefixRNGState]:
    values = logits.detach().contiguous().cpu()
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or values.numel() == 0:
        raise PrefixBisimulationError("next-token logits must be a nonempty vector")
    if not torch.isfinite(values.float()).all():
        raise PrefixBisimulationError("next-token logits contain nonfinite values")

    if rng_state.mode == "greedy":
        token = int(torch.argmax(values).item())
        return token, rng_state
    if rng_state.mode != "sample":
        raise PrefixBisimulationError(f"unsupported RNG mode: {rng_state.mode}")
    if rng_state.temperature <= 0:
        raise PrefixBisimulationError("temperature must be positive")

    scores = values.float() / float(rng_state.temperature)
    generator = _generator_from_bytes(rng_state.generator_state)
    if rng_state.top_k > 0:
        count = min(int(rng_state.top_k), int(scores.numel()))
        top_values, top_indices = torch.topk(scores, k=count, dim=0)
        probabilities = torch.softmax(top_values, dim=0)
        local = int(torch.multinomial(probabilities, 1, generator=generator).item())
        token = int(top_indices[local].item())
    else:
        probabilities = torch.softmax(scores, dim=0)
        token = int(torch.multinomial(probabilities, 1, generator=generator).item())
    next_state = PrefixRNGState(
        mode="sample",
        generator_state=_generator_bytes(generator),
        seed=rng_state.seed,
        temperature=rng_state.temperature,
        top_k=rng_state.top_k,
    )
    return token, next_state


def replay_prefix(model: Any, state: PrefixCompiledState) -> ReplayObservation:
    if state.step != len(state.generated_ids):
        raise PrefixBisimulationError("prefix state step invariant failed")
    input_ids = torch.tensor([state.prompt_ids], dtype=torch.long)
    target_calls = 0
    replayed_positions = int(input_ids.shape[1])
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        target_calls += 1
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        for token in state.generated_ids:
            output = model(
                input_ids=torch.tensor([[int(token)]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            target_calls += 1
            replayed_positions += 1
            past = output.past_key_values
            logits = output.logits[:, -1, :]
    detached_logits = logits.detach().contiguous().cpu()
    return ReplayObservation(
        logits=detached_logits,
        past_key_values=past,
        target_calls=target_calls,
        replayed_token_positions=replayed_positions,
        cache_sha256=cache_sha256(past),
        cache_bytes=cache_nbytes(past),
        logits_sha256=tensor_sha256(detached_logits),
    )


def decode_step_by_replay(
    model: Any, state: PrefixCompiledState
) -> tuple[int, PrefixCompiledState, PrefixDecodeTrace, ReplayObservation]:
    observation = replay_prefix(model, state)
    token, next_rng = select_token(observation.logits, state.rng_state)
    next_state = append_exact_token(state, token, next_rng)
    trace = PrefixDecodeTrace(
        step=state.step,
        prefix_sha256=state.digest(),
        prefix_length=state.position,
        selected_token=token,
        rng_before_sha256=state.rng_state.digest(),
        rng_after_sha256=next_rng.digest(),
        logits_sha256=observation.logits_sha256,
        cache_sha256=observation.cache_sha256,
        cache_bytes=observation.cache_bytes,
        target_calls=observation.target_calls,
        replayed_token_positions=observation.replayed_token_positions,
    )
    return token, next_state, trace, observation
