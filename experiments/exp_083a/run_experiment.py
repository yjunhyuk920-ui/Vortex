#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterator

import numpy as np

from vortex_runtime.causal_residual_atlas_gate import (
    derive_first_decode_gate,
    select_favorable_page,
)
from vortex_runtime.causal_residual_atlas_oracle import (
    canonical_prefix_basis,
    canonical_sha256,
    deterministic_core,
    native_anchored_pages,
    summarize_gate,
    target_to_candidate_kls,
)
from vortex_runtime.fractal_oracle import (
    registered_teacher_forcing_tokens,
    validate_prompt_and_trace_ids,
)


ROOT = Path(__file__).resolve().parents[2]


class CandidateBatchControlError(RuntimeError):
    """Raised when vectorized page branches change the frozen dense input."""


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    import torch

    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        text=True,
    ).strip()


def write_checksums(output: Path) -> None:
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            entries.append(
                f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
            )
    (output / "checksums.sha256").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def peak_rss_bytes() -> int | None:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.PeakWorkingSetSize) if ok else None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except (ImportError, OSError):
        return None


@contextmanager
def default_dtype(dtype: Any) -> Iterator[None]:
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def validate_registered_inputs(
    config: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    registered = config["registered_inputs"]
    prompts_path = ROOT / registered["prompts_path"]
    trace_path = ROOT / registered["target_trace_path"]
    authority_path = ROOT / config["preregistration"]["authority"]
    lock_path = ROOT / config["runtime"]["requirements_lock"]
    inherited_lock = ROOT / "experiments/exp_076/requirements.lock.json"
    exp076_config_path = ROOT / config["checkpoint"]["exp076_manifest"]
    for path in (
        prompts_path,
        trace_path,
        authority_path,
        lock_path,
        inherited_lock,
        exp076_config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    expected_hashes = {
        prompts_path: registered["prompts_sha256"],
        trace_path: registered["target_trace_sha256"],
        authority_path: config["preregistration"]["authority_sha256"],
        inherited_lock: json.loads(lock_path.read_text(encoding="utf-8"))[
            "inherited_file_sha256"
        ],
    }
    actual_hashes = {}
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        actual_hashes[path.relative_to(ROOT).as_posix()] = actual
        if actual != expected:
            raise ValueError(
                f"registered SHA mismatch for {path}: {actual} != {expected}"
            )

    from experiments.exp_076.run_experiment import validate_checkpoint_manifest

    exp076_config = json.loads(exp076_config_path.read_text(encoding="utf-8"))
    checkpoint_rows = validate_checkpoint_manifest(model_dir, exp076_config)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    traces = load_rows(trace_path)
    join_audit = validate_prompt_and_trace_ids(prompts, traces)
    evaluation = prompts.get("evaluation")
    if not isinstance(evaluation, list):
        raise ValueError("evaluation prompt population is missing")
    if len(evaluation) != int(registered["evaluation_prompts"]):
        raise ValueError("evaluation prompt count mismatch")
    required_families = tuple(registered["required_families"])
    family_counts = {
        family: sum(row.get("family") == family for row in evaluation)
        for family in required_families
    }
    expected_per_family = int(registered["prompts_per_family"])
    if any(value != expected_per_family for value in family_counts.values()):
        raise ValueError(f"evaluation family count mismatch: {family_counts}")

    frozen = derive_first_decode_gate()
    gate = config["gate"]
    comparisons = {
        "rank": (gate["rank"], frozen.rank),
        "page_columns": (gate["page_columns"], frozen.page_columns),
        "teacher_position_index": (
            gate["teacher_position_index"],
            frozen.teacher_position_index,
        ),
        "layer_indices": (tuple(gate["layer_indices"]), frozen.layer_indices),
        "expected_token_states": (
            gate["expected_token_states"],
            frozen.token_states,
        ),
        "expected_projection_branches": (
            gate["expected_projection_branches"],
            frozen.projection_branches,
        ),
        "expected_oracle_page_candidates": (
            gate["expected_oracle_page_candidates"],
            frozen.oracle_page_candidates,
        ),
        "required_token_successes": (
            gate["required_token_successes"],
            frozen.required_token_successes,
        ),
        "maximum_token_failures": (
            gate["maximum_token_failures"],
            frozen.maximum_token_failures,
        ),
    }
    mismatches = {
        key: values for key, values in comparisons.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"config drift from preregistered Gate: {mismatches}")
    if str(gate["minimum_coverage"]) != str(frozen.minimum_coverage):
        raise ValueError("minimum coverage drift")
    if tuple(gate["projection_order"]) != ("q_proj", "down_proj"):
        raise ValueError("projection ordering drift")
    if not gate["stop_on_first_valid_scientific_failure"]:
        raise ValueError("scientific stop rule must remain enabled")

    traces_by_id = {str(row["prompt_id"]): row for row in traces}
    return {
        "prompts_path": prompts_path,
        "trace_path": trace_path,
        "evaluation": evaluation,
        "traces_by_id": traces_by_id,
        "join_audit": join_audit,
        "checkpoint_rows": checkpoint_rows,
        "actual_hashes": actual_hashes,
        "requirements_lock_sha256": sha256_file(lock_path),
        "family_counts": family_counts,
    }


def load_target(model_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = Qwen3_5Config.from_pretrained(model_dir, local_files_only=True)
    text_config = copy.deepcopy(config.text_config)
    text_config._attn_implementation = "eager"
    text_config.dtype = torch.bfloat16
    with default_dtype(torch.bfloat16):
        target = Qwen3_5ForCausalLM(text_config)
    weight_path = model_dir / "model.safetensors-00001-of-00001.safetensors"
    state: dict[str, Any] = {}
    target_text_parameters = 0
    embedding_parameters = 0
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if not name.startswith("model.language_model."):
                continue
            tensor = handle.get_tensor(name)
            mapped = "model." + name.removeprefix("model.language_model.")
            state[mapped] = tensor
            target_text_parameters += tensor.numel()
            if name == "model.language_model.embed_tokens.weight":
                embedding_parameters = tensor.numel()
                state["lm_head.weight"] = tensor
    missing, unexpected = target.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"target state mismatch: missing={missing}, unexpected={unexpected}"
        )
    target.tie_weights()
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    layer_index = 11
    layer = target.model.layers[layer_index]
    if layer.layer_type != "full_attention":
        raise ValueError("registered layer 11 is not full_attention")
    projection_shapes = {
        "q_proj": list(layer.self_attn.q_proj.weight.shape),
        "down_proj": list(layer.mlp.down_proj.weight.shape),
    }
    return target, tokenizer, {
        "target_text_parameters": target_text_parameters,
        "tied_lm_head_parameters": embedding_parameters,
        "num_hidden_layers": len(target.model.layers),
        "layer_11_type": layer.layer_type,
        "projection_shapes": projection_shapes,
    }


def tokenize_prompt(tokenizer: Any, prompt: str, max_tokens: int = 192) -> Any:
    import torch

    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor([token_ids], dtype=torch.long)
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    if token_ids.shape[1] > max_tokens:
        raise ValueError(f"prompt exceeds max token count: {token_ids.shape[1]}")
    return token_ids.to(dtype=torch.long, device="cpu")


class ProjectionRecorder:
    def __init__(self, name: str, module: Any) -> None:
        self.name = name
        self.module = module
        self.mode = "idle"
        self.prefix_input: Any | None = None
        self.prefix_output: Any | None = None
        self.current_input: Any | None = None
        self.current_output: Any | None = None
        self.patch_values: Any | None = None
        self.patch_audit: dict[str, Any] | None = None
        self.handle = module.register_forward_hook(self._hook)

    def close(self) -> None:
        self.handle.remove()

    def set_mode(self, mode: str, patch_values: Any | None = None) -> None:
        if mode not in {"idle", "prefix", "current", "patch"}:
            raise ValueError(f"unknown recorder mode: {mode}")
        self.mode = mode
        self.patch_values = patch_values
        if mode == "patch":
            self.patch_audit = None

    def _hook(self, module: Any, arguments: tuple[Any, ...], output: Any) -> Any:
        import torch

        if self.mode == "idle":
            return None
        if not arguments:
            raise CandidateBatchControlError(f"{self.name}: missing module input")
        current_input = arguments[0].detach()
        native_output = output.detach()
        if self.mode == "prefix":
            if current_input.ndim != 3 or current_input.shape[0] != 1:
                raise CandidateBatchControlError(
                    f"{self.name}: malformed prefix input shape"
                )
            self.prefix_input = current_input[0].cpu().clone()
            self.prefix_output = native_output[0].cpu().clone()
            return None
        if self.mode == "current":
            if current_input.shape[0:2] != (1, 1):
                raise CandidateBatchControlError(
                    f"{self.name}: malformed current input shape"
                )
            self.current_input = current_input[0, 0].cpu().clone()
            self.current_output = native_output[0, 0].cpu().clone()
            return None

        if self.patch_values is None:
            raise CandidateBatchControlError(f"{self.name}: patch is missing")
        if self.current_input is None or self.current_output is None:
            raise CandidateBatchControlError(
                f"{self.name}: baseline current state is missing"
            )
        expected_input = self.current_input.reshape(1, 1, -1).expand_as(
            current_input
        )
        expected_output = self.current_output.reshape(1, 1, -1).expand_as(
            native_output
        )
        input_equal = bool(torch.equal(current_input.cpu(), expected_input))
        output_equal = bool(torch.equal(native_output.cpu(), expected_output))
        patch_shape_equal = tuple(self.patch_values.shape) == tuple(output.shape)
        self.patch_audit = {
            "candidate_batch_size": int(current_input.shape[0]),
            "current_input_bitwise_equal": input_equal,
            "native_output_bitwise_equal": output_equal,
            "patch_shape_equal": patch_shape_equal,
        }
        if not input_equal:
            raise CandidateBatchControlError(
                f"{self.name}: candidate batch changed current input"
            )
        if not output_equal:
            raise CandidateBatchControlError(
                f"{self.name}: candidate batch changed native output"
            )
        if not patch_shape_equal:
            raise CandidateBatchControlError(
                f"{self.name}: candidate patch shape mismatch"
            )
        return self.patch_values


def clone_repeated_cache(cache: Any, batch_size: int) -> Any:
    import torch

    if batch_size <= 0:
        raise ValueError("candidate batch size must be positive")
    candidate_cache = copy.deepcopy(cache)
    candidate_cache.reorder_cache(
        torch.zeros(batch_size, dtype=torch.long, device="cpu")
    )
    return candidate_cache


@dataclass
class ExecutionCounters:
    model_forward_calls: int = 0
    logical_batch_rows: int = 0
    transformer_token_positions: int = 0

    def record(self, *, batch_size: int, sequence_length: int) -> None:
        self.model_forward_calls += 1
        self.logical_batch_rows += batch_size
        self.transformer_token_positions += batch_size * sequence_length


def evaluate_projection(
    *,
    target: Any,
    prefix_cache: Any,
    recorder: ProjectionRecorder,
    prompt: dict[str, Any],
    prompt_length: int,
    first_target_token: int,
    baseline_logits: Any,
    config: dict[str, Any],
    counters: ExecutionCounters,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
]:
    import torch

    if any(
        value is None
        for value in (
            recorder.prefix_input,
            recorder.prefix_output,
            recorder.current_input,
            recorder.current_output,
        )
    ):
        raise ValueError(f"{recorder.name}: missing captured state")
    gate = config["gate"]
    prefix_input = recorder.prefix_input.to(torch.float64).numpy()
    prefix_output = recorder.prefix_output.to(torch.float64).numpy()
    current_input = recorder.current_input.to(torch.float64).numpy()
    native_output = recorder.current_output.to(torch.float64).numpy()
    weight = recorder.module.weight.detach().to(torch.float64).numpy()
    if not all(
        np.all(np.isfinite(value))
        for value in (
            prefix_input,
            prefix_output,
            current_input,
            native_output,
            weight,
        )
    ):
        raise ValueError(f"{recorder.name}: non-finite projection state")

    basis_result = canonical_prefix_basis(
        prefix_input,
        maximum_rank=int(gate["rank"]),
    )
    page_result = native_anchored_pages(
        weight,
        basis_result.basis,
        current_input,
        native_output,
        page_columns=int(gate["page_columns"]),
    )
    page_count = int(page_result.page_outputs.shape[0])
    expected_pages = int(gate["pages_per_projection"][recorder.name])
    if page_count != expected_pages:
        raise ValueError(
            f"{recorder.name}: page count {page_count} != {expected_pages}"
        )

    basis_row = {
        "prompt_id": prompt["id"],
        "family": prompt["family"],
        "layer_index": int(gate["layer_indices"][0]),
        "projection": recorder.name,
        "prompt_length": prompt_length,
        "prefix_input_shape": list(prefix_input.shape),
        "prefix_image_shape": list(prefix_output.shape),
        "prefix_input_sha256": sha256_array(prefix_input),
        "prefix_image_sha256": sha256_array(prefix_output),
        "basis_sha256": sha256_array(basis_result.basis),
        "singular_values_sha256": sha256_array(
            basis_result.singular_values
        ),
        "numerical_cutoff": basis_result.numerical_cutoff,
        "numerical_rank": basis_result.numerical_rank,
        "retained_rank": basis_result.retained_rank,
        "current_input_sha256": sha256_array(current_input),
        "native_output_sha256": sha256_array(native_output),
        "input_residual_l2": float(
            np.linalg.norm(page_result.input_residual)
        ),
        "residual_image_l2": float(
            np.linalg.norm(page_result.residual_image)
        ),
    }

    all_page_difference = float(
        np.max(np.abs(page_result.all_page_output - native_output))
    )
    patch_rows = np.concatenate(
        [
            page_result.page_outputs,
            native_output[None, :],
            page_result.all_page_output[None, :],
        ],
        axis=0,
    )
    patch_tensor = torch.from_numpy(patch_rows).to(
        dtype=recorder.current_output.dtype,
        device="cpu",
    )[:, None, :]
    batch_size = int(patch_tensor.shape[0])
    token_batch = torch.full(
        (batch_size, 1),
        int(first_target_token),
        dtype=torch.long,
        device="cpu",
    )
    candidate_cache = clone_repeated_cache(prefix_cache, batch_size)
    recorder.set_mode("patch", patch_tensor)
    started = time.perf_counter_ns()
    control_failures: list[str] = []
    try:
        with torch.inference_mode():
            candidate_hidden = target.model(
                input_ids=token_batch,
                past_key_values=candidate_cache,
                use_cache=True,
            ).last_hidden_state[:, -1, :]
            candidate_logits = target.lm_head(candidate_hidden).detach()
        counters.record(batch_size=batch_size, sequence_length=1)
    except CandidateBatchControlError as error:
        control_failures.append(str(error))
        return (
            basis_row,
            [],
            None,
            [
                {
                    "prompt_id": prompt["id"],
                    "family": prompt["family"],
                    "projection": recorder.name,
                    "control": "candidate_batch_identity",
                    "pass": False,
                    "detail": str(error),
                }
            ],
            control_failures,
            {
                "prompt_id": prompt["id"],
                "projection": recorder.name,
                "candidate_batch_wall_ns": time.perf_counter_ns() - started,
            },
        )
    finally:
        recorder.set_mode("idle")

    branch_wall_ns = time.perf_counter_ns() - started
    if not bool(torch.isfinite(candidate_logits.float()).all()):
        raise ValueError(f"{recorder.name}: candidate logits are non-finite")
    target_np = baseline_logits.detach().float().numpy()
    candidate_np = candidate_logits.detach().float().numpy()
    page_logits = candidate_np[:page_count]
    control_logits = candidate_np[page_count:]
    page_kls = target_to_candidate_kls(target_np, page_logits)
    control_kls = target_to_candidate_kls(target_np, control_logits)
    target_top1 = int(np.argmax(target_np))
    page_top1 = np.argmax(page_logits, axis=1)
    dense_top1 = int(np.argmax(control_logits[0]))
    all_page_top1 = int(np.argmax(control_logits[1]))
    patch_audit = recorder.patch_audit or {}

    control_rows = [
        {
            "prompt_id": prompt["id"],
            "family": prompt["family"],
            "projection": recorder.name,
            "control": "candidate_batch_current_input_identity",
            "pass": bool(patch_audit.get("current_input_bitwise_equal")),
        },
        {
            "prompt_id": prompt["id"],
            "family": prompt["family"],
            "projection": recorder.name,
            "control": "candidate_batch_native_output_identity",
            "pass": bool(patch_audit.get("native_output_bitwise_equal")),
        },
        {
            "prompt_id": prompt["id"],
            "family": prompt["family"],
            "projection": recorder.name,
            "control": "dense_patch_top1_identity",
            "pass": dense_top1 == target_top1,
            "target_top1": target_top1,
            "candidate_top1": dense_top1,
            "target_to_candidate_kl": float(control_kls[0]),
            "candidate_logits_sha256": tensor_sha256(
                candidate_logits[page_count]
            ),
        },
        {
            "prompt_id": prompt["id"],
            "family": prompt["family"],
            "projection": recorder.name,
            "control": "all_page_top1_identity",
            "pass": all_page_top1 == target_top1,
            "target_top1": target_top1,
            "candidate_top1": all_page_top1,
            "target_to_candidate_kl": float(control_kls[1]),
            "float64_output_max_abs_difference": all_page_difference,
            "candidate_logits_sha256": tensor_sha256(
                candidate_logits[page_count + 1]
            ),
        },
    ]
    for row in control_rows:
        if not row["pass"]:
            control_failures.append(
                f"{prompt['id']}:{recorder.name}:{row['control']}"
            )
    if control_failures:
        return (
            basis_row,
            [],
            None,
            control_rows,
            control_failures,
            {
                "prompt_id": prompt["id"],
                "projection": recorder.name,
                "candidate_batch_wall_ns": branch_wall_ns,
            },
        )

    page_rows: list[dict[str, Any]] = []
    page_columns = int(gate["page_columns"])
    input_width = int(weight.shape[1])
    for page_index in range(page_count):
        page_start = page_index * page_columns
        page_stop = min(input_width, page_start + page_columns)
        page_rows.append(
            {
                "prompt_id": prompt["id"],
                "family": prompt["family"],
                "layer_index": int(gate["layer_indices"][0]),
                "projection": recorder.name,
                "page_index": page_index,
                "page_start": page_start,
                "page_stop": page_stop,
                "target_top1": target_top1,
                "candidate_top1": int(page_top1[page_index]),
                "top1_match": bool(page_top1[page_index] == target_top1),
                "target_to_candidate_kl": float(page_kls[page_index]),
                "input_residual_l2": basis_row["input_residual_l2"],
                "exact_unread_error_l2": float(
                    page_result.exact_unread_error_norms[page_index]
                ),
                "certified_unread_radius": float(
                    page_result.certified_unread_radii[page_index]
                ),
                "page_frobenius_bound": float(
                    page_result.block_frobenius_bounds[page_index]
                ),
                "page_input_residual_l2": float(
                    page_result.block_residual_norms[page_index]
                ),
                "candidate_output_sha256": tensor_sha256(
                    patch_tensor[page_index, 0]
                ),
                "candidate_logits_sha256": tensor_sha256(
                    candidate_logits[page_index]
                ),
            }
        )

    choice = select_favorable_page(
        target_np,
        page_logits,
        page_indices=tuple(range(page_count)),
    )
    selected = page_rows[choice.page_index]
    if choice.target_to_candidate_kl != selected["target_to_candidate_kl"]:
        if not np.isclose(
            choice.target_to_candidate_kl,
            selected["target_to_candidate_kl"],
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("independent page-choice KL mismatch")
    branch_row = {
        "prompt_id": prompt["id"],
        "family": prompt["family"],
        "layer_index": int(gate["layer_indices"][0]),
        "projection": recorder.name,
        "prompt_length": prompt_length,
        "prompt_rank": basis_result.retained_rank,
        "page_count": page_count,
        "selected_page_index": choice.page_index,
        "target_top1": choice.target_top1,
        "candidate_top1": choice.candidate_top1,
        "top1_match": choice.top1_match,
        "selected_kl": selected["target_to_candidate_kl"],
        "input_residual_l2": selected["input_residual_l2"],
        "exact_unread_error_l2": selected["exact_unread_error_l2"],
        "certified_unread_radius": selected["certified_unread_radius"],
        "selected_candidate_logits_sha256": selected[
            "candidate_logits_sha256"
        ],
    }
    timing = {
        "prompt_id": prompt["id"],
        "projection": recorder.name,
        "candidate_batch_size": batch_size,
        "candidate_batch_wall_ns": branch_wall_ns,
    }
    del candidate_cache, candidate_hidden, candidate_logits, patch_tensor
    gc.collect()
    return (
        basis_row,
        page_rows,
        branch_row,
        control_rows,
        control_failures,
        timing,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    import torch
    import transformers
    import tokenizers
    import huggingface_hub
    import safetensors

    config_path = Path(arguments.config).resolve()
    model_dir = Path(arguments.model_dir).resolve()
    output = Path(arguments.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
    }
    for name, actual in expected_versions.items():
        registered = str(config["runtime"].get(name, actual))
        if actual != registered:
            raise ValueError(
                f"runtime version mismatch for {name}: {actual} != {registered}"
            )

    run_wall_started = time.perf_counter_ns()
    run_cpu_started = time.process_time_ns()
    torch.set_num_threads(int(config["runtime"]["torch_num_threads"]))
    torch.use_deterministic_algorithms(True)
    inputs = validate_registered_inputs(config, model_dir)
    target, tokenizer, parameter_audit = load_target(model_dir)
    layer = target.model.layers[int(config["gate"]["layer_indices"][0])]
    recorders = {
        "q_proj": ProjectionRecorder("q_proj", layer.self_attn.q_proj),
        "down_proj": ProjectionRecorder("down_proj", layer.mlp.down_proj),
    }

    basis_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    control_failures: list[str] = []
    leakage_failures: list[str] = []
    malformed_state_count = 0
    counters = ExecutionCounters()
    baseline_trace_mismatches = 0
    scientific_failure_seen = False
    log_lines: list[str] = []

    try:
        for prompt in inputs["evaluation"]:
            prompt_id = str(prompt["id"])
            trace = inputs["traces_by_id"][prompt_id]
            conditioning, expected_tokens = registered_teacher_forcing_tokens(
                trace,
                token_count=2,
            )
            if len(conditioning) != 1 or len(expected_tokens) != 2:
                raise ValueError("teacher position-one trace extraction failed")
            prefix_ids = tokenize_prompt(tokenizer, str(prompt["prompt"]))
            prompt_length = int(prefix_ids.shape[1])
            prefix_token_sha256 = tensor_sha256(prefix_ids)
            prefix_trace_control = (
                prompt_length == int(trace["prefix_token_count"])
                and prefix_token_sha256 == trace["prefix_token_sha256"]
            )
            control_rows.append(
                {
                    "prompt_id": prompt_id,
                    "family": prompt["family"],
                    "control": "prefix_token_trace_identity",
                    "pass": prefix_trace_control,
                    "prompt_length": prompt_length,
                    "prefix_token_sha256": prefix_token_sha256,
                }
            )
            if not prefix_trace_control:
                control_failures.append(
                    f"{prompt_id}:prefix_token_trace_identity"
                )
                break

            for recorder in recorders.values():
                recorder.set_mode("prefix")
            prefill_started = time.perf_counter_ns()
            with torch.inference_mode():
                prefix_output = target.model(
                    input_ids=prefix_ids,
                    use_cache=True,
                )
                prefix_logits = target.lm_head(
                    prefix_output.last_hidden_state[:, -1, :]
                ).detach()[0]
            counters.record(batch_size=1, sequence_length=prompt_length)
            for recorder in recorders.values():
                recorder.set_mode("idle")
            prefix_top1 = int(prefix_logits.float().argmax().item())
            prefix_expected = int(expected_tokens[0])
            prefix_pass = prefix_top1 == prefix_expected
            baseline_trace_mismatches += int(not prefix_pass)
            control_rows.append(
                {
                    "prompt_id": prompt_id,
                    "family": prompt["family"],
                    "control": "teacher_position_zero_top1_trace",
                    "pass": prefix_pass,
                    "expected_top1": prefix_expected,
                    "observed_top1": prefix_top1,
                    "logits_sha256": tensor_sha256(prefix_logits),
                }
            )
            if not prefix_pass:
                control_failures.append(
                    f"{prompt_id}:teacher_position_zero_top1_trace"
                )
                break

            for recorder in recorders.values():
                recorder.set_mode("current")
            baseline_cache = copy.deepcopy(prefix_output.past_key_values)
            current_ids = torch.tensor(
                [[int(conditioning[0])]],
                dtype=torch.long,
                device="cpu",
            )
            baseline_started = time.perf_counter_ns()
            with torch.inference_mode():
                baseline_hidden = target.model(
                    input_ids=current_ids,
                    past_key_values=baseline_cache,
                    use_cache=True,
                ).last_hidden_state[:, -1, :]
                baseline_logits = target.lm_head(baseline_hidden).detach()[0]
            counters.record(batch_size=1, sequence_length=1)
            for recorder in recorders.values():
                recorder.set_mode("idle")
            baseline_top1 = int(baseline_logits.float().argmax().item())
            baseline_expected = int(expected_tokens[1])
            baseline_pass = baseline_top1 == baseline_expected
            baseline_trace_mismatches += int(not baseline_pass)
            control_rows.append(
                {
                    "prompt_id": prompt_id,
                    "family": prompt["family"],
                    "control": "teacher_position_one_top1_trace",
                    "pass": baseline_pass,
                    "expected_top1": baseline_expected,
                    "observed_top1": baseline_top1,
                    "logits_sha256": tensor_sha256(baseline_logits),
                }
            )
            timing_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prefill_wall_ns": baseline_started - prefill_started,
                    "baseline_decode_wall_ns": (
                        time.perf_counter_ns() - baseline_started
                    ),
                }
            )
            if not baseline_pass:
                control_failures.append(
                    f"{prompt_id}:teacher_position_one_top1_trace"
                )
                break

            prompt_branch_rows: list[dict[str, Any]] = []
            evaluated_projections: list[str] = []
            for projection in config["gate"]["projection_order"]:
                recorder = recorders[projection]
                if recorder.prefix_input is None:
                    malformed_state_count += 1
                    break
                if int(recorder.prefix_input.shape[0]) != prompt_length:
                    leakage_failures.append(
                        f"{prompt_id}:{projection}:prefix_row_count"
                    )
                    break
                (
                    basis_row,
                    projection_pages,
                    branch_row,
                    projection_controls,
                    projection_control_failures,
                    timing,
                ) = evaluate_projection(
                    target=target,
                    prefix_cache=prefix_output.past_key_values,
                    recorder=recorder,
                    prompt=prompt,
                    prompt_length=prompt_length,
                    first_target_token=int(conditioning[0]),
                    baseline_logits=baseline_logits,
                    config=config,
                    counters=counters,
                )
                basis_rows.append(basis_row)
                page_rows.extend(projection_pages)
                control_rows.extend(projection_controls)
                control_failures.extend(projection_control_failures)
                timing_rows.append(timing)
                if projection_control_failures or branch_row is None:
                    break
                branch_rows.append(branch_row)
                prompt_branch_rows.append(branch_row)
                evaluated_projections.append(projection)
                line = json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "projection": projection,
                        "selected_page": branch_row["selected_page_index"],
                        "top1_match": branch_row["top1_match"],
                        "selected_kl": branch_row["selected_kl"],
                    },
                    sort_keys=True,
                )
                print(line, flush=True)
                log_lines.append(line)
                if not branch_row["top1_match"]:
                    scientific_failure_seen = True
                    break

            if control_failures or leakage_failures or malformed_state_count:
                break
            token_success = (
                len(prompt_branch_rows) == 2
                and all(row["top1_match"] for row in prompt_branch_rows)
            )
            token_rows.append(
                {
                    "prompt_id": prompt_id,
                    "family": prompt["family"],
                    "teacher_position_index": 1,
                    "evaluated_projections": evaluated_projections,
                    "complete_projection_pair": len(prompt_branch_rows) == 2,
                    "success": token_success,
                    "failure_projection": next(
                        (
                            row["projection"]
                            for row in prompt_branch_rows
                            if not row["top1_match"]
                        ),
                        None,
                    ),
                }
            )
            if scientific_failure_seen:
                break
            del prefix_output, baseline_cache, baseline_hidden, baseline_logits
            gc.collect()
    finally:
        for recorder in recorders.values():
            recorder.close()

    execution_complete = (
        not control_failures
        and not leakage_failures
        and malformed_state_count == 0
        and len(token_rows) == int(config["gate"]["expected_token_states"])
        and len(branch_rows)
        == int(config["gate"]["expected_projection_branches"])
    )
    gate_summary = summarize_gate(
        branch_rows,
        token_rows,
        expected_token_states=int(config["gate"]["expected_token_states"]),
        expected_projection_branches=int(
            config["gate"]["expected_projection_branches"]
        ),
        expected_families=tuple(
            config["registered_inputs"]["required_families"]
        ),
        prompts_per_family=int(
            config["registered_inputs"]["prompts_per_family"]
        ),
        maximum_mean_kl=float(config["gate"]["maximum_mean_kl"]),
        maximum_p95_kl=float(config["gate"]["maximum_p95_kl"]),
        control_failures=control_failures,
        leakage_failures=leakage_failures,
        malformed_state_count=malformed_state_count,
        execution_complete=execution_complete,
    )
    registered_core = {
        "checkpoint_model_id": config["checkpoint"]["model_id"],
        "checkpoint_revision": config["checkpoint"]["revision"],
        "weight_sha256": config["checkpoint"]["weight_sha256"],
        "prompts_sha256": config["registered_inputs"]["prompts_sha256"],
        "target_trace_sha256": config["registered_inputs"][
            "target_trace_sha256"
        ],
        "preregistration_authority_sha256": config["preregistration"][
            "authority_sha256"
        ],
        "teacher_position_index": config["gate"]["teacher_position_index"],
        "rank": config["gate"]["rank"],
        "page_columns": config["gate"]["page_columns"],
        "layer_indices": config["gate"]["layer_indices"],
        "projection_order": config["gate"]["projection_order"],
    }
    core = deterministic_core(
        registered_inputs=registered_core,
        basis_rows=basis_rows,
        page_rows=page_rows,
        branch_rows=branch_rows,
        token_rows=token_rows,
        control_rows=control_rows,
        gate=gate_summary,
    )
    core_hash = canonical_sha256(core)
    environment = {
        "platform": platform.platform(),
        **expected_versions,
        "torch_num_threads": torch.get_num_threads(),
        "source_commit": git_commit(),
        "config_sha256": sha256_file(config_path),
        "requirements_lock_sha256": inputs["requirements_lock_sha256"],
    }
    elapsed_wall = time.perf_counter_ns() - run_wall_started
    elapsed_cpu = time.process_time_ns() - run_cpu_started
    summary = {
        "experiment": "EXP-083A",
        "name": config["name"],
        "phase": ["C-small-real-checkpoint-favorable-oracle-falsification"],
        "evidence_level": "E1",
        "authoritative_decision": gate_summary["decision"],
        "REGISTERED_EXTERNAL_INPUTS": registered_core,
        "MEASURED": {
            "evaluated_token_states": len(token_rows),
            "evaluated_projection_branches": len(branch_rows),
            "oracle_page_candidates_evaluated": len(page_rows),
            "control_rows": len(control_rows),
            "control_failures": len(control_failures),
            "leakage_failures": len(leakage_failures),
            "malformed_state_count": malformed_state_count,
            "baseline_trace_mismatches": baseline_trace_mismatches,
            "model_forward_calls": counters.model_forward_calls,
            "physical_decoder_layer_calls": (
                counters.model_forward_calls
                * int(parameter_audit["num_hidden_layers"])
            ),
            "logical_batch_row_layer_evaluations": (
                counters.logical_batch_rows
                * int(parameter_audit["num_hidden_layers"])
            ),
            "transformer_token_positions": counters.transformer_token_positions,
            "wall_ns": elapsed_wall,
            "cpu_ns": elapsed_cpu,
            "peak_rss_bytes": peak_rss_bytes(),
            "deterministic_core_sha256": core_hash,
        },
        "DERIVED": {
            "parameter_audit": parameter_audit,
            "input_join_audit": inputs["join_audit"],
            "family_counts": inputs["family_counts"],
            "gate": gate_summary,
            "frozen_gate": derive_first_decode_gate().to_dict(),
        },
        "UNVERIFIED": [
            "causal page selector without exact-reference logits",
            "legal WQ image constructed only from committed prefix pairs",
            "outward-rounded native numerical certificate",
            "simultaneous multi-projection and multi-layer composition",
            "actual fail-closed dense-operation replacement at E2",
            "physical SSD PCIe GPU traffic latency and peak VRAM",
            "medium large 122B and dense 405B scaling",
            "E2 through E7",
        ],
        "claim_boundary": {
            "checkpoint": "UNCHANGED_PINNED_OFFICIAL_BF16_PAYLOAD",
            "center": "NON_DEPLOYABLE_NATIVE_ANCHORED_EXACT_REFERENCE",
            "selector": "NON_DEPLOYABLE_EXHAUSTIVE_PAGE_ORACLE",
            "future_tokens": "NONE_USED_TO_BUILD_BASIS_OR_CANDIDATE",
            "operation_replacement": "ONE_PROJECTION_PATCHED_AT_A_TIME_WITH_ALL_OTHER_WORK_DENSE",
            "physical_speed": "NOT_MEASURED",
            "target_server": "NO_COMMAND_EXECUTED",
        },
        "provenance": environment,
    }

    dump_rows(output / "raw/basis_rows.jsonl", basis_rows)
    dump_rows(output / "raw/page_rows.jsonl", page_rows)
    dump_rows(output / "raw/branch_rows.jsonl", branch_rows)
    dump_rows(output / "raw/token_rows.jsonl", token_rows)
    dump(output / "raw/control_rows.json", control_rows)
    dump(output / "raw/checkpoint_manifest.json", inputs["checkpoint_rows"])
    dump(
        output / "raw/input_audit.json",
        {
            "file_hashes": inputs["actual_hashes"],
            "join_audit": inputs["join_audit"],
            "family_counts": inputs["family_counts"],
        },
    )
    dump_rows(output / "raw/timing_rows.jsonl", timing_rows)
    dump(output / "processed/gate.json", gate_summary)
    dump(output / "processed/deterministic_core.json", core)
    dump(output / "artifacts/environment.json", environment)
    (output / "artifacts/contract.txt").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        "\n".join(log_lines) + ("\n" if log_lines else ""),
        encoding="utf-8",
        newline="\n",
    )
    dump(output / "summary.json", summary)
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
