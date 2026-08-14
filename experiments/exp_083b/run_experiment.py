#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterator

import numpy as np

from vortex_runtime.causal_residual_atlas_legal_execution import (
    bound_legal_projection_candidate,
    compile_stored_pair_certificate,
    propagate_last_down_to_final_hidden,
    vector_norm_upper,
)
from vortex_runtime.causal_residual_atlas_legal_gate import (
    array_frobenius_norm_upper,
    certify_top1_from_row_norms,
    derive_legal_pair_outward_gate,
    matrix_row_norm_uppers,
    native_linear_l2_rounding_bound,
    native_linear_per_row_rounding_bounds,
    select_max_residual_energy_page,
    summarize_legal_pair_outward_gate,
    verified_cholesky_spectral_upper_bound,
)
from vortex_runtime.causal_residual_atlas_oracle import (
    canonical_sha256,
    target_to_candidate_kls,
)


ROOT = Path(__file__).resolve().parents[2]


class BranchPointReached(RuntimeError):
    """Private signal that stops the model before current dense down_proj."""


class GateControlError(RuntimeError):
    """A malformed or non-reproducible model state invalidated the run."""


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
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
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
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        ok = kernel32.K32GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.PeakWorkingSetSize) if ok else None
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
    authority_path = ROOT / config["preregistration"]["authority"]
    lock_path = ROOT / config["runtime"]["requirements_lock"]
    inherited_lock = ROOT / "experiments/exp_076/requirements.lock.json"
    exp076_config_path = ROOT / config["checkpoint"]["exp076_manifest"]
    for path in (
        prompts_path,
        authority_path,
        lock_path,
        inherited_lock,
        exp076_config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    expected_hashes = {
        prompts_path: registered["prompts_sha256"],
        authority_path: config["preregistration"]["authority_sha256"],
        inherited_lock: json.loads(lock_path.read_text(encoding="utf-8"))[
            "inherited_file_sha256"
        ],
    }
    actual_hashes: dict[str, str] = {}
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
    evaluation = prompts.get("evaluation")
    if not isinstance(evaluation, list):
        raise ValueError("evaluation prompt population is missing")
    if len(evaluation) != int(registered["evaluation_prompts"]):
        raise ValueError("evaluation prompt count mismatch")
    prompt_ids = [str(row.get("id")) for row in evaluation]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("evaluation prompt IDs are not unique")
    if any(not isinstance(row.get("prompt"), str) for row in evaluation):
        raise ValueError("evaluation prompt text is malformed")
    required_families = tuple(registered["required_families"])
    family_counts = {
        family: sum(row.get("family") == family for row in evaluation)
        for family in required_families
    }
    expected_per_family = int(registered["prompts_per_family"])
    if any(value != expected_per_family for value in family_counts.values()):
        raise ValueError(f"evaluation family count mismatch: {family_counts}")

    frozen = derive_legal_pair_outward_gate()
    gate = config["gate"]
    comparisons = {
        "rank": (gate["rank"], frozen.rank),
        "page_columns": (gate["page_columns"], frozen.page_columns),
        "teacher_position_index": (
            gate["teacher_position_index"],
            frozen.teacher_position_index,
        ),
        "layer_index": (gate["layer_index"], frozen.layer_index),
        "projection": (gate["projection"], frozen.projection),
        "expected_token_states": (
            gate["expected_token_states"],
            frozen.evaluation_prompts,
        ),
        "required_token_successes": (
            gate["required_token_successes"],
            frozen.required_token_successes,
        ),
        "required_family_successes": (
            gate["required_family_successes"],
            frozen.required_family_successes,
        ),
        "selector": (gate["selector"], frozen.selector),
        "declared_output": (gate["declared_output"], frozen.declared_output),
    }
    mismatches = {
        key: values for key, values in comparisons.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"config drift from preregistered Gate: {mismatches}")
    if str(gate["minimum_coverage"]) != str(frozen.minimum_coverage):
        raise ValueError("minimum coverage drift")
    if float(gate["maximum_mean_kl"]) != float(frozen.maximum_mean_kl):
        raise ValueError("mean KL threshold drift")
    if float(gate["maximum_p95_kl"]) != float(frozen.maximum_p95_kl):
        raise ValueError("p95 KL threshold drift")

    return {
        "evaluation": evaluation,
        "checkpoint_rows": checkpoint_rows,
        "actual_hashes": actual_hashes,
        "requirements_lock_sha256": sha256_file(lock_path),
        "family_counts": family_counts,
    }


def load_target(model_dir: Path, *, layer_index: int) -> tuple[Any, Any, dict[str, Any]]:
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
    layer = target.model.layers[layer_index]
    if layer.layer_type != "full_attention":
        raise ValueError(f"registered layer {layer_index} is not full_attention")
    return target, tokenizer, {
        "target_text_parameters": target_text_parameters,
        "tied_lm_head_parameters": embedding_parameters,
        "num_hidden_layers": len(target.model.layers),
        "registered_layer_type": layer.layer_type,
        "down_projection_shape": list(layer.mlp.down_proj.weight.shape),
        "lm_head_shape": list(target.lm_head.weight.shape),
        "final_norm_shape": list(target.model.norm.weight.shape),
        "rms_norm_epsilon": float(target.model.norm.eps),
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


class PrefixPairRecorder:
    def __init__(self, module: Any) -> None:
        self.active = False
        self.prefix_input: Any | None = None
        self.prefix_output: Any | None = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: Any, arguments: tuple[Any, ...], output: Any) -> None:
        if not self.active:
            return None
        if not arguments or arguments[0].ndim != 3 or arguments[0].shape[0] != 1:
            raise GateControlError("malformed prefix down-projection input")
        self.prefix_input = arguments[0][0].detach().cpu().clone()
        self.prefix_output = output[0].detach().cpu().clone()
        return None

    def start(self) -> None:
        self.prefix_input = None
        self.prefix_output = None
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.handle.remove()


class DenseCompletionRecorder:
    def __init__(self, down_module: Any, final_norm: Any) -> None:
        self.active = False
        self.current_input: Any | None = None
        self.current_output: Any | None = None
        self.pre_norm: Any | None = None
        self.handles = [
            down_module.register_forward_hook(self._down_hook),
            final_norm.register_forward_pre_hook(self._norm_hook),
        ]

    def _down_hook(self, module: Any, arguments: tuple[Any, ...], output: Any) -> None:
        if not self.active:
            return None
        if not arguments or arguments[0].shape[0:2] != (1, 1):
            raise GateControlError("malformed dense current down state")
        self.current_input = arguments[0][0, 0].detach().cpu().clone()
        self.current_output = output[0, 0].detach().cpu().clone()
        return None

    def _norm_hook(self, module: Any, arguments: tuple[Any, ...]) -> None:
        if not self.active:
            return None
        if not arguments or arguments[0].shape[0:2] != (1, 1):
            raise GateControlError("malformed dense final-norm state")
        self.pre_norm = arguments[0][0, 0].detach().cpu().clone()
        return None

    def start(self) -> None:
        self.current_input = None
        self.current_output = None
        self.pre_norm = None
        self.active = True

    def stop(self) -> None:
        self.active = False

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if any(
            value is None
            for value in (self.current_input, self.current_output, self.pre_norm)
        ):
            raise GateControlError("dense completion recorder is incomplete")
        return tuple(
            value.float().numpy().astype(np.float64, copy=False)
            for value in (self.current_input, self.current_output, self.pre_norm)
        )  # type: ignore[return-value]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def capture_branch_point(
    *,
    target: Any,
    prefix_cache: Any,
    first_token: int,
    layer_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stop before the current down weight is read and return legal state."""

    import torch

    layer = target.model.layers[layer_index]
    captures: dict[str, Any] = {}

    def residual_hook(module: Any, arguments: tuple[Any, ...]) -> None:
        if not arguments or arguments[0].shape[0:2] != (1, 1):
            raise GateControlError("malformed post-attention residual")
        captures["residual_branch"] = arguments[0][0, 0].detach().cpu().clone()

    def down_hook(module: Any, arguments: tuple[Any, ...]) -> None:
        if not arguments or arguments[0].shape[0:2] != (1, 1):
            raise GateControlError("malformed current down input")
        captures["current_input"] = arguments[0][0, 0].detach().cpu().clone()
        raise BranchPointReached()

    handles = [
        layer.post_attention_layernorm.register_forward_pre_hook(residual_hook),
        layer.mlp.down_proj.register_forward_pre_hook(down_hook),
    ]
    current_ids = torch.tensor([[first_token]], dtype=torch.long, device="cpu")
    reached = False
    try:
        with torch.inference_mode():
            target.model(
                input_ids=current_ids,
                past_key_values=copy.deepcopy(prefix_cache),
                use_cache=True,
            )
    except BranchPointReached:
        reached = True
    finally:
        for handle in handles:
            handle.remove()
    if not reached or set(captures) != {"residual_branch", "current_input"}:
        raise GateControlError("registered current branch point was not reached")
    return tuple(
        captures[key].float().numpy().astype(np.float64, copy=False)
        for key in ("current_input", "residual_branch")
    )  # type: ignore[return-value]


def build_legal_query_state(
    *,
    stored_basis: np.ndarray,
    current_input: np.ndarray,
    page_columns: int,
) -> dict[str, Any]:
    """Compute causal coordinates/residual and freeze the target-free page."""

    import torch

    basis = torch.from_numpy(np.asarray(stored_basis, dtype=np.float32)).to(
        torch.bfloat16
    )
    current = torch.from_numpy(np.asarray(current_input, dtype=np.float32)).to(
        torch.bfloat16
    )
    with torch.inference_mode():
        coordinates = basis.T @ current
        basis_application = basis @ coordinates
        residual = current - basis_application
    residual_array = residual.float().numpy().astype(np.float64, copy=False)
    selected_page = select_max_residual_energy_page(
        residual_array,
        page_columns=page_columns,
    )
    return {
        "coordinates": coordinates.float().numpy().astype(np.float64, copy=False),
        "basis_application": (
            basis_application.float().numpy().astype(np.float64, copy=False)
        ),
        "residual": residual_array,
        "selected_page": selected_page,
    }


def build_selected_page_candidate(
    *,
    stored_image: np.ndarray,
    coordinates: np.ndarray,
    residual: np.ndarray,
    residual_branch: np.ndarray,
    selected_weight_page: np.ndarray,
    selected_page: int,
    page_columns: int,
    final_norm: Any,
    lm_head: Any,
) -> dict[str, np.ndarray]:
    """Execute exactly one page and the candidate-only declared suffix."""

    import torch

    image = torch.from_numpy(np.asarray(stored_image, dtype=np.float32)).to(
        torch.bfloat16
    )
    coordinate_tensor = torch.from_numpy(
        np.asarray(coordinates, dtype=np.float32)
    ).to(torch.bfloat16)
    residual_tensor = torch.from_numpy(np.asarray(residual, dtype=np.float32)).to(
        torch.bfloat16
    )
    residual_branch_tensor = torch.from_numpy(
        np.asarray(residual_branch, dtype=np.float32)
    ).to(torch.bfloat16)
    page = torch.from_numpy(
        np.asarray(selected_weight_page, dtype=np.float32)
    ).to(torch.bfloat16)
    start = selected_page * page_columns
    page_input = residual_tensor[start : start + page.shape[1]]
    with torch.inference_mode():
        image_application = image @ coordinate_tensor
        page_application = page @ page_input
        candidate_down = image_application + page_application
        candidate_pre_norm = residual_branch_tensor + candidate_down
        candidate_hidden = final_norm(candidate_pre_norm[None, None, :])[0, 0]
        candidate_logits = lm_head(candidate_hidden)
    return {
        "stored_image_application": (
            image_application.float().numpy().astype(np.float64, copy=False)
        ),
        "selected_page_application": (
            page_application.float().numpy().astype(np.float64, copy=False)
        ),
        "candidate_down": (
            candidate_down.float().numpy().astype(np.float64, copy=False)
        ),
        "candidate_pre_norm": (
            candidate_pre_norm.float().numpy().astype(np.float64, copy=False)
        ),
        "candidate_hidden": (
            candidate_hidden.float().numpy().astype(np.float64, copy=False)
        ),
        "candidate_logits": (
            candidate_logits.float().numpy().astype(np.float64, copy=False)
        ),
    }


def run_dense_completion(
    *,
    target: Any,
    prefix_cache: Any,
    first_token: int,
    recorder: DenseCompletionRecorder,
) -> dict[str, Any]:
    import torch

    recorder.start()
    current_ids = torch.tensor([[first_token]], dtype=torch.long, device="cpu")
    try:
        with torch.inference_mode():
            hidden = target.model(
                input_ids=current_ids,
                past_key_values=copy.deepcopy(prefix_cache),
                use_cache=True,
            ).last_hidden_state[0, -1]
            logits = target.lm_head(hidden).detach()
    finally:
        recorder.stop()
    current_input, current_output, pre_norm = recorder.arrays()
    return {
        "current_input": current_input,
        "native_down": current_output,
        "native_pre_norm": pre_norm,
        "native_hidden": hidden.float().numpy().astype(np.float64, copy=False),
        "native_logits": logits.float().numpy().astype(np.float64, copy=False),
        "native_logits_tensor_sha256": tensor_sha256(logits),
    }


def static_lm_row_norms(weight: Any, *, chunk_rows: int = 4096) -> np.ndarray:
    rows = int(weight.shape[0])
    result = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        chunk = weight[start:stop].detach().float().numpy().astype(np.float64)
        result[start:stop] = matrix_row_norm_uppers(chunk)
    return result


def exact_rmsnorm(value: np.ndarray, gain: np.ndarray, epsilon: float) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    return vector / math.sqrt(float(np.mean(vector * vector)) + epsilon) * gain


def append_control(
    rows: list[dict[str, Any]],
    failures: list[str],
    *,
    prompt_id: str,
    name: str,
    passed: bool,
    measured: float | int | str | None = None,
    bound: float | int | str | None = None,
) -> None:
    row: dict[str, Any] = {
        "prompt_id": prompt_id,
        "control": name,
        "pass": bool(passed),
    }
    if measured is not None:
        row["measured"] = measured
    if bound is not None:
        row["bound"] = bound
    rows.append(row)
    if not passed:
        failures.append(f"{prompt_id}:{name}")


def deterministic_core(
    *,
    registered_inputs: dict[str, Any],
    static_certificate: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    certificate_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "registered_inputs": registered_inputs,
        "static_certificate": static_certificate,
        "pair_rows": pair_rows,
        "certificate_rows": certificate_rows,
        "token_rows": token_rows,
        "control_rows": control_rows,
        "gate": gate,
    }


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
    actual_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
    }
    for name, actual in actual_versions.items():
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
    layer_index = int(config["gate"]["layer_index"])
    target, tokenizer, parameter_audit = load_target(
        model_dir,
        layer_index=layer_index,
    )
    expected_shapes = {
        "down_projection_shape": [
            int(config["gate"]["projection_output_rows"]),
            int(config["gate"]["projection_input_width"]),
        ],
        "lm_head_shape": [
            int(config["gate"]["vocabulary_rows"]),
            int(config["gate"]["hidden_size"]),
        ],
        "final_norm_shape": [int(config["gate"]["hidden_size"])],
    }
    shape_mismatches = {
        key: (parameter_audit[key], expected)
        for key, expected in expected_shapes.items()
        if parameter_audit[key] != expected
    }
    if shape_mismatches:
        raise ValueError(f"registered parameter shape mismatch: {shape_mismatches}")

    layer = target.model.layers[layer_index]
    down_module = layer.mlp.down_proj
    down_weight = (
        down_module.weight.detach().float().numpy().astype(np.float64)
    )
    static_started = time.perf_counter_ns()
    spectral = verified_cholesky_spectral_upper_bound(down_weight)
    lm_row_norms = static_lm_row_norms(target.lm_head.weight)
    static_wall_ns = time.perf_counter_ns() - static_started
    gain = (
        1.0 + target.model.norm.weight.detach().float().numpy().astype(np.float64)
    )
    gain_abs_max = float(np.max(np.abs(gain)))
    epsilon = float(target.model.norm.eps)
    static_certificate = {
        "spectral": spectral.to_dict(),
        "lm_head_row_norms_sha256": sha256_array(lm_row_norms),
        "lm_head_rows": int(lm_row_norms.size),
        "lm_head_minimum_row_norm_upper": float(np.min(lm_row_norms)),
        "lm_head_maximum_row_norm_upper": float(np.max(lm_row_norms)),
        "final_norm_gain_abs_max": gain_abs_max,
        "final_norm_epsilon": epsilon,
        "registered_spectral_compile_operations": int(
            config["registered_accounting"]["spectral_compile_operations"]
        ),
    }
    (output / "raw").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "raw/static_arrays.npz",
        lm_head_row_norm_uppers=lm_row_norms,
        final_norm_gain=gain,
    )

    prefix_recorder = PrefixPairRecorder(down_module)
    dense_recorder = DenseCompletionRecorder(down_module, target.model.norm)
    pair_rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    control_failures: list[str] = []
    leakage_failures: list[str] = []
    malformed_state_count = 0
    log_lines: list[str] = []
    full_model_forward_calls = 0
    partial_model_forward_calls = 0
    evaluated_prompt_token_positions = 0

    try:
        for prompt in inputs["evaluation"]:
            prompt_started = time.perf_counter_ns()
            prompt_id = str(prompt["id"])
            family = str(prompt["family"])
            prefix_ids = tokenize_prompt(tokenizer, str(prompt["prompt"]))
            prompt_length = int(prefix_ids.shape[1])
            prefix_token_sha = tensor_sha256(prefix_ids)

            prefix_recorder.start()
            prefill_started = time.perf_counter_ns()
            try:
                with torch.inference_mode():
                    prefix_output = target.model(
                        input_ids=prefix_ids,
                        use_cache=True,
                    )
                    prefix_logits_tensor = target.lm_head(
                        prefix_output.last_hidden_state[:, -1, :]
                    ).detach()[0]
            finally:
                prefix_recorder.stop()
            full_model_forward_calls += 1
            evaluated_prompt_token_positions += prompt_length
            if prefix_recorder.prefix_input is None or prefix_recorder.prefix_output is None:
                raise GateControlError(f"{prompt_id}: prefix pair capture missing")
            prefix_inputs = (
                prefix_recorder.prefix_input.float().numpy().astype(np.float64)
            )
            prefix_images = (
                prefix_recorder.prefix_output.float().numpy().astype(np.float64)
            )
            first_token = int(prefix_logits_tensor.float().argmax().item())
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="prefix_pair_row_count",
                passed=(
                    prefix_inputs.shape[0] == prompt_length
                    and prefix_images.shape[0] == prompt_length
                ),
                measured=int(prefix_inputs.shape[0]),
                bound=prompt_length,
            )
            if control_failures:
                break

            pair_started = time.perf_counter_ns()
            pair = compile_stored_pair_certificate(
                prefix_inputs,
                prefix_images,
                maximum_rank=int(config["gate"]["rank"]),
                operator_norm_upper=spectral.operator_norm_upper,
                frobenius_norm_upper=spectral.frobenius_norm_upper,
            )
            exact_prefix_images = prefix_inputs @ down_weight.T
            prefix_error_norms = np.linalg.norm(
                exact_prefix_images - prefix_images,
                axis=1,
            )
            prefix_envelope_pass = bool(
                np.all(
                    prefix_error_norms
                    <= np.asarray(pair.prefix_native_error_bounds)
                )
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="prefix_native_arithmetic_envelope",
                passed=prefix_envelope_pass,
                measured=float(np.max(prefix_error_norms)),
                bound=float(np.max(pair.prefix_native_error_bounds)),
            )
            pair_row = {
                "prompt_id": prompt_id,
                "family": family,
                "prompt_length": prompt_length,
                "prefix_token_sha256": prefix_token_sha,
                "prompt_last_greedy_token": first_token,
                "prefix_logits_sha256": tensor_sha256(prefix_logits_tensor),
                "prefix_input_sha256": sha256_array(prefix_inputs),
                "prefix_image_sha256": sha256_array(prefix_images),
                "stored_basis_sha256": sha256_array(pair.stored_basis),
                "stored_image_sha256": sha256_array(pair.stored_image),
                "coefficient_map_sha256": sha256_array(pair.coefficient_map),
                "rank": pair.rank,
                "accepted_prefix_indices": list(pair.accepted_prefix_indices),
                "accepted_residual_norm_lowers": list(
                    pair.accepted_residual_norm_lowers
                ),
                "accepted_threshold_uppers": list(
                    pair.accepted_threshold_uppers
                ),
                "prefix_image_error_frobenius": (
                    pair.prefix_image_error_frobenius
                ),
                "coefficient_operator_upper": pair.coefficient_operator_upper,
                "stored_basis_relation_error_operator": (
                    pair.stored_basis_relation_error_operator
                ),
                "stored_image_relation_error_operator": (
                    pair.stored_image_relation_error_operator
                ),
                "pair_image_defect_operator": pair.pair_image_defect_operator,
                "maximum_observed_prefix_native_error": float(
                    np.max(prefix_error_norms)
                ),
                "pair_compile_wall_ns": time.perf_counter_ns() - pair_started,
                "array_file": f"raw/prompt_arrays/{prompt_id}.npz",
            }
            pair_rows.append(pair_row)
            if control_failures:
                break

            current_input, residual_branch = capture_branch_point(
                target=target,
                prefix_cache=prefix_output.past_key_values,
                first_token=first_token,
                layer_index=layer_index,
            )
            partial_model_forward_calls += 1
            if pair.rank != int(config["gate"]["rank"]):
                dense_first = run_dense_completion(
                    target=target,
                    prefix_cache=prefix_output.past_key_values,
                    first_token=first_token,
                    recorder=dense_recorder,
                )
                dense_replay = run_dense_completion(
                    target=target,
                    prefix_cache=prefix_output.past_key_values,
                    first_token=first_token,
                    recorder=dense_recorder,
                )
                full_model_forward_calls += 2
                replay_equal = bool(
                    np.array_equal(
                        dense_first["native_logits"],
                        dense_replay["native_logits"],
                    )
                )
                append_control(
                    control_rows,
                    control_failures,
                    prompt_id=prompt_id,
                    name="dense_completion_replay",
                    passed=replay_equal,
                )
                token_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "family": family,
                        "success": False,
                        "certificate_resolved": False,
                        "fallback_executed": True,
                        "dense_completion_matches_baseline": replay_equal,
                        "false_accept": False,
                        "scientific_failure": not control_failures,
                        "failure_reason": "pair_rank_below_16",
                        "target_to_candidate_kl": None,
                    }
                )
                break

            query = build_legal_query_state(
                stored_basis=pair.stored_basis,
                current_input=current_input,
                page_columns=int(config["gate"]["page_columns"]),
            )
            selected_page = int(query["selected_page"])
            page_columns = int(config["gate"]["page_columns"])
            page_start = selected_page * page_columns
            selected_weight_page = np.ascontiguousarray(
                down_weight[:, page_start : page_start + page_columns]
            )
            candidate_started = time.perf_counter_ns()
            candidate = build_selected_page_candidate(
                stored_image=pair.stored_image,
                coordinates=query["coordinates"],
                residual=query["residual"],
                residual_branch=residual_branch,
                selected_weight_page=selected_weight_page,
                selected_page=selected_page,
                page_columns=page_columns,
                final_norm=target.model.norm,
                lm_head=target.lm_head,
            )
            projection_bounds = bound_legal_projection_candidate(
                stored_basis=pair.stored_basis,
                stored_image=pair.stored_image,
                pair_image_defect=pair.pair_image_defect_operator,
                operator_norm_upper=spectral.operator_norm_upper,
                frobenius_norm_upper=spectral.frobenius_norm_upper,
                current_input=current_input,
                coordinates=query["coordinates"],
                basis_application=query["basis_application"],
                residual=query["residual"],
                selected_page=selected_page,
                page_columns=page_columns,
                selected_weight_page=selected_weight_page,
                stored_image_application=candidate["stored_image_application"],
                selected_page_application=candidate[
                    "selected_page_application"
                ],
            )
            final_bounds = propagate_last_down_to_final_hidden(
                residual_branch=residual_branch,
                candidate_down=candidate["candidate_down"],
                candidate_pre_norm=candidate["candidate_pre_norm"],
                projection_radius=projection_bounds.projection_output_radius,
                gain_abs_max=gain_abs_max,
                epsilon=epsilon,
            )
            candidate_hidden_norm = vector_norm_upper(candidate["candidate_hidden"])
            target_hidden_norm_upper = math.nextafter(
                candidate_hidden_norm + final_bounds.final_hidden_radius,
                math.inf,
            )
            candidate_logit_rounding = native_linear_per_row_rounding_bounds(
                lm_row_norms,
                input_norm_upper=candidate_hidden_norm,
                columns=int(config["gate"]["hidden_size"]),
            )
            target_logit_rounding = native_linear_per_row_rounding_bounds(
                lm_row_norms,
                input_norm_upper=target_hidden_norm_upper,
                columns=int(config["gate"]["hidden_size"]),
            )
            two_path_logit_rounding = np.nextafter(
                candidate_logit_rounding + target_logit_rounding,
                np.inf,
            )
            certificate = certify_top1_from_row_norms(
                candidate["candidate_logits"],
                hidden_radius=final_bounds.final_hidden_radius,
                row_norm_uppers=lm_row_norms,
                per_logit_rounding_uppers=two_path_logit_rounding,
            )
            verdict_frozen_ns = time.perf_counter_ns()

            dense_first = run_dense_completion(
                target=target,
                prefix_cache=prefix_output.past_key_values,
                first_token=first_token,
                recorder=dense_recorder,
            )
            dense_replay = run_dense_completion(
                target=target,
                prefix_cache=prefix_output.past_key_values,
                first_token=first_token,
                recorder=dense_recorder,
            )
            full_model_forward_calls += 2
            native_top1 = int(np.argmax(dense_first["native_logits"]))
            candidate_top1 = int(np.argmax(candidate["candidate_logits"]))
            dense_completion_matches = bool(
                all(
                    np.array_equal(dense_first[key], dense_replay[key])
                    for key in (
                        "current_input",
                        "native_down",
                        "native_pre_norm",
                        "native_hidden",
                        "native_logits",
                    )
                )
            )

            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="branch_current_input_identity",
                passed=bool(
                    np.array_equal(current_input, dense_first["current_input"])
                    and np.array_equal(
                        current_input,
                        dense_replay["current_input"],
                    )
                ),
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="dense_completion_replay",
                passed=dense_completion_matches,
            )
            native_down_exact = down_weight @ current_input
            native_down_error = float(
                np.linalg.norm(dense_first["native_down"] - native_down_exact)
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="current_native_down_arithmetic_envelope",
                passed=(
                    native_down_error
                    <= projection_bounds.native_dense_rounding_radius
                ),
                measured=native_down_error,
                bound=projection_bounds.native_dense_rounding_radius,
            )
            actual_pair_defect = float(
                np.linalg.norm(
                    down_weight @ pair.stored_basis - pair.stored_image,
                    ord=2,
                )
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="pair_image_defect_envelope",
                passed=actual_pair_defect <= pair.pair_image_defect_operator,
                measured=actual_pair_defect,
                bound=pair.pair_image_defect_operator,
            )

            basis_frobenius = array_frobenius_norm_upper(pair.stored_basis)
            coordinate_rounding_bound = native_linear_l2_rounding_bound(
                input_norm_upper=vector_norm_upper(current_input),
                frobenius_norm_upper=basis_frobenius,
                operator_norm_upper=basis_frobenius,
                rows=pair.rank,
                columns=current_input.size,
            )
            coordinate_error = float(
                np.linalg.norm(
                    query["coordinates"]
                    - pair.stored_basis.T @ current_input
                )
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="coordinate_arithmetic_envelope",
                passed=coordinate_error <= coordinate_rounding_bound,
                measured=coordinate_error,
                bound=coordinate_rounding_bound,
            )
            candidate_step_controls = (
                (
                    "basis_apply_arithmetic_envelope",
                    float(
                        np.linalg.norm(
                            query["basis_application"]
                            - pair.stored_basis @ query["coordinates"]
                        )
                    ),
                    projection_bounds.basis_apply_rounding_radius,
                ),
                (
                    "residual_subtraction_arithmetic_envelope",
                    float(
                        np.linalg.norm(
                            query["residual"]
                            - (current_input - query["basis_application"])
                        )
                    ),
                    projection_bounds.residual_subtraction_rounding_radius,
                ),
                (
                    "stored_image_apply_arithmetic_envelope",
                    float(
                        np.linalg.norm(
                            candidate["stored_image_application"]
                            - pair.stored_image @ query["coordinates"]
                        )
                    ),
                    projection_bounds.stored_image_apply_rounding_radius,
                ),
                (
                    "selected_page_apply_arithmetic_envelope",
                    float(
                        np.linalg.norm(
                            candidate["selected_page_application"]
                            - selected_weight_page
                            @ query["residual"][
                                page_start : page_start
                                + selected_weight_page.shape[1]
                            ]
                        )
                    ),
                    projection_bounds.selected_page_apply_rounding_radius,
                ),
                (
                    "candidate_down_add_arithmetic_envelope",
                    float(
                        np.linalg.norm(
                            candidate["candidate_down"]
                            - (
                                candidate["stored_image_application"]
                                + candidate["selected_page_application"]
                            )
                        )
                    ),
                    projection_bounds.candidate_addition_rounding_radius,
                ),
                (
                    "candidate_residual_add_arithmetic_envelope",
                    float(
                        np.linalg.norm(
                            candidate["candidate_pre_norm"]
                            - (residual_branch + candidate["candidate_down"])
                        )
                    ),
                    final_bounds.candidate_residual_addition_radius,
                ),
            )
            for name, measured, bound in candidate_step_controls:
                append_control(
                    control_rows,
                    control_failures,
                    prompt_id=prompt_id,
                    name=name,
                    passed=measured <= bound,
                    measured=measured,
                    bound=bound,
                )

            actual_projection_error = float(
                np.linalg.norm(
                    candidate["candidate_down"] - dense_first["native_down"]
                )
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="projection_output_radius",
                passed=(
                    actual_projection_error
                    <= projection_bounds.projection_output_radius
                ),
                measured=actual_projection_error,
                bound=projection_bounds.projection_output_radius,
            )
            target_add_error = float(
                np.linalg.norm(
                    dense_first["native_pre_norm"]
                    - (residual_branch + dense_first["native_down"])
                )
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="target_residual_add_arithmetic_envelope",
                passed=(
                    target_add_error
                    <= final_bounds.target_residual_addition_radius
                ),
                measured=target_add_error,
                bound=final_bounds.target_residual_addition_radius,
            )
            candidate_norm_error = float(
                np.linalg.norm(
                    candidate["candidate_hidden"]
                    - exact_rmsnorm(candidate["candidate_pre_norm"], gain, epsilon)
                )
            )
            target_norm_error = float(
                np.linalg.norm(
                    dense_first["native_hidden"]
                    - exact_rmsnorm(dense_first["native_pre_norm"], gain, epsilon)
                )
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="candidate_rmsnorm_implementation_envelope",
                passed=(
                    candidate_norm_error
                    <= final_bounds.candidate_rmsnorm_implementation_radius
                ),
                measured=candidate_norm_error,
                bound=final_bounds.candidate_rmsnorm_implementation_radius,
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="target_rmsnorm_implementation_envelope",
                passed=(
                    target_norm_error
                    <= final_bounds.target_rmsnorm_implementation_radius
                ),
                measured=target_norm_error,
                bound=final_bounds.target_rmsnorm_implementation_radius,
            )
            actual_hidden_error = float(
                np.linalg.norm(
                    candidate["candidate_hidden"] - dense_first["native_hidden"]
                )
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="final_hidden_radius",
                passed=actual_hidden_error <= final_bounds.final_hidden_radius,
                measured=actual_hidden_error,
                bound=final_bounds.final_hidden_radius,
            )
            logit_error_uppers = np.nextafter(
                np.nextafter(
                    lm_row_norms * final_bounds.final_hidden_radius,
                    np.inf,
                )
                + two_path_logit_rounding,
                np.inf,
            )
            actual_logit_errors = np.abs(
                dense_first["native_logits"] - candidate["candidate_logits"]
            )
            append_control(
                control_rows,
                control_failures,
                prompt_id=prompt_id,
                name="full_vocabulary_logit_envelope",
                passed=bool(np.all(actual_logit_errors <= logit_error_uppers)),
                measured=float(np.max(actual_logit_errors)),
                bound=float(np.max(logit_error_uppers)),
            )

            false_accept = bool(
                certificate.certified and certificate.winner != native_top1
            )
            if false_accept:
                control_failures.append(f"{prompt_id}:false_accept")
            target_kl = float(
                target_to_candidate_kls(
                    dense_first["native_logits"],
                    candidate["candidate_logits"][None, :],
                )[0]
            )
            fallback_executed = not certificate.certified
            prompt_control_failures = [
                value
                for value in control_failures
                if value.startswith(f"{prompt_id}:")
            ]
            success = bool(
                certificate.certified
                and certificate.winner == native_top1
                and not prompt_control_failures
                and dense_completion_matches
            )
            scientific_failure = bool(
                not prompt_control_failures and not certificate.certified
            )
            token_row = {
                "prompt_id": prompt_id,
                "family": family,
                "teacher_position_index": int(
                    config["gate"]["teacher_position_index"]
                ),
                "success": success,
                "certificate_resolved": bool(certificate.certified),
                "certified_winner": (
                    int(certificate.winner) if certificate.certified else None
                ),
                "candidate_winner": candidate_top1,
                "native_winner": native_top1,
                "fallback_executed": fallback_executed,
                "dense_completion_matches_baseline": dense_completion_matches,
                "false_accept": false_accept,
                "scientific_failure": scientific_failure,
                "failure_reason": (
                    None if success else "strict_final_margin_unresolved"
                ),
                "target_to_candidate_kl": target_kl,
            }
            token_rows.append(token_row)
            certificate_row = {
                "prompt_id": prompt_id,
                "selected_page": selected_page,
                "selected_page_start": page_start,
                "selected_page_width": int(selected_weight_page.shape[1]),
                "residual_page_energies": [
                    float(
                        np.sum(
                            query["residual"][start : start + page_columns]
                            ** 2
                        )
                    )
                    for start in range(0, query["residual"].size, page_columns)
                ],
                "projection_bounds": projection_bounds.to_dict(),
                "final_bounds": final_bounds.to_dict(),
                "candidate_hidden_norm_upper": candidate_hidden_norm,
                "target_hidden_norm_upper": target_hidden_norm_upper,
                "candidate_winner": candidate_top1,
                "certified_winner": int(certificate.winner),
                "certificate_resolved": bool(certificate.certified),
                "minimum_margin_lower": certificate.minimum_margin_lower,
                "maximum_two_path_logit_rounding": float(
                    np.max(two_path_logit_rounding)
                ),
                "verdict_frozen_before_dense_completion": True,
                "verdict_frozen_monotonic_ns": verdict_frozen_ns,
                "dense_completion_finished_monotonic_ns": time.perf_counter_ns(),
                "actual_pair_defect": actual_pair_defect,
                "actual_projection_error": actual_projection_error,
                "actual_hidden_error": actual_hidden_error,
                "maximum_actual_logit_error": float(
                    np.max(actual_logit_errors)
                ),
                "native_winner": native_top1,
                "target_to_candidate_kl": target_kl,
                "array_file": f"raw/prompt_arrays/{prompt_id}.npz",
            }
            certificate_rows.append(certificate_row)

            array_path = output / "raw/prompt_arrays" / f"{prompt_id}.npz"
            array_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                array_path,
                prefix_inputs=prefix_inputs,
                prefix_images=prefix_images,
                stored_basis=pair.stored_basis,
                stored_image=pair.stored_image,
                coefficient_map=pair.coefficient_map,
                current_input=current_input,
                residual_branch=residual_branch,
                coordinates=query["coordinates"],
                basis_application=query["basis_application"],
                input_residual=query["residual"],
                selected_weight_page=selected_weight_page,
                stored_image_application=candidate[
                    "stored_image_application"
                ],
                selected_page_application=candidate[
                    "selected_page_application"
                ],
                candidate_down=candidate["candidate_down"],
                candidate_pre_norm=candidate["candidate_pre_norm"],
                candidate_hidden=candidate["candidate_hidden"],
                candidate_logits=candidate["candidate_logits"],
                native_down=dense_first["native_down"],
                native_pre_norm=dense_first["native_pre_norm"],
                native_hidden=dense_first["native_hidden"],
                native_logits=dense_first["native_logits"],
                replay_native_down=dense_replay["native_down"],
                replay_native_pre_norm=dense_replay["native_pre_norm"],
                replay_native_hidden=dense_replay["native_hidden"],
                replay_native_logits=dense_replay["native_logits"],
                candidate_logit_rounding=candidate_logit_rounding,
                target_logit_rounding=target_logit_rounding,
                logit_error_uppers=logit_error_uppers,
            )
            timing_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prefill_wall_ns": pair_started - prefill_started,
                    "pair_compile_wall_ns": pair_row["pair_compile_wall_ns"],
                    "candidate_and_certificate_wall_ns": (
                        verdict_frozen_ns - candidate_started
                    ),
                    "total_prompt_wall_ns": (
                        time.perf_counter_ns() - prompt_started
                    ),
                }
            )
            line = json.dumps(
                {
                    "prompt_id": prompt_id,
                    "rank": pair.rank,
                    "selected_page": selected_page,
                    "certificate_resolved": certificate.certified,
                    "minimum_margin_lower": certificate.minimum_margin_lower,
                    "native_winner": native_top1,
                    "kl": target_kl,
                },
                sort_keys=True,
            )
            print(line, flush=True)
            log_lines.append(line)
            if control_failures or leakage_failures or scientific_failure:
                break
            del prefix_output, prefix_logits_tensor
            gc.collect()
    finally:
        prefix_recorder.close()
        dense_recorder.close()

    execution_complete = bool(
        not control_failures
        and not leakage_failures
        and malformed_state_count == 0
        and len(token_rows) == int(config["gate"]["expected_token_states"])
    )
    gate_summary = summarize_legal_pair_outward_gate(
        token_rows,
        expected_token_states=int(config["gate"]["expected_token_states"]),
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
        "preregistration_authority_sha256": config["preregistration"][
            "authority_sha256"
        ],
        "preregistration_commit": config["preregistration"]["commit"],
        "teacher_position_index": config["gate"]["teacher_position_index"],
        "rank": config["gate"]["rank"],
        "page_columns": config["gate"]["page_columns"],
        "layer_index": config["gate"]["layer_index"],
        "projection": config["gate"]["projection"],
        "selector": config["gate"]["selector"],
        "spectral_proof": config["gate"]["spectral_proof"],
    }
    core = deterministic_core(
        registered_inputs=registered_core,
        static_certificate=static_certificate,
        pair_rows=pair_rows,
        certificate_rows=certificate_rows,
        token_rows=token_rows,
        control_rows=control_rows,
        gate=gate_summary,
    )
    core_hash = canonical_sha256(core)
    environment = {
        "platform": platform.platform(),
        **actual_versions,
        "torch_num_threads": torch.get_num_threads(),
        "source_commit": git_commit(),
        "config_sha256": sha256_file(config_path),
        "requirements_lock_sha256": inputs["requirements_lock_sha256"],
    }
    elapsed_wall = time.perf_counter_ns() - run_wall_started
    elapsed_cpu = time.process_time_ns() - run_cpu_started
    summary = {
        "experiment": "EXP-083B",
        "name": config["name"],
        "phase": ["C-small-real-checkpoint-legal-certificate-falsification"],
        "evidence_level": "E1",
        "authoritative_decision": gate_summary["decision"],
        "REGISTERED_EXTERNAL_INPUTS": registered_core,
        "MEASURED": {
            "evaluated_token_states": len(token_rows),
            "pair_rows": len(pair_rows),
            "certificate_rows": len(certificate_rows),
            "control_rows": len(control_rows),
            "control_failures": len(control_failures),
            "leakage_failures": len(leakage_failures),
            "malformed_state_count": malformed_state_count,
            "full_model_forward_calls": full_model_forward_calls,
            "partial_model_forward_calls": partial_model_forward_calls,
            "evaluated_prompt_token_positions": (
                evaluated_prompt_token_positions
            ),
            "static_compile_wall_ns": static_wall_ns,
            "wall_ns": elapsed_wall,
            "cpu_ns": elapsed_cpu,
            "peak_rss_bytes": peak_rss_bytes(),
            "deterministic_core_sha256": core_hash,
        },
        "DERIVED": {
            "parameter_audit": parameter_audit,
            "input_file_hashes": inputs["actual_hashes"],
            "family_counts": inputs["family_counts"],
            "static_certificate": static_certificate,
            "gate": gate_summary,
            "frozen_gate": derive_legal_pair_outward_gate().to_dict(),
        },
        "UNVERIFIED": [
            "simultaneous multi-projection and multi-layer composition",
            "actual fail-closed dense-operation replacement at E2",
            "complete-generation quality preservation",
            "physical SSD PCIe GPU traffic latency and peak VRAM",
            "122B and dense 405B scaling",
            "E2 through E7",
        ],
        "claim_boundary": {
            "checkpoint": "UNCHANGED_PINNED_OFFICIAL_BF16_PAYLOAD",
            "pair_source": "COMMITTED_PREFIX_INPUT_IMAGE_PAIRS_ONLY",
            "selector": "CURRENT_RESIDUAL_ENERGY_ONLY",
            "native_current_output_before_verdict": "PROHIBITED_AND_NOT_EXECUTED",
            "operation_replacement": "ONE_LAST_LAYER_DOWN_PROJECTION_E1_BRANCH",
            "physical_speed": "NOT_MEASURED",
            "target_server": "NO_COMMAND_EXECUTED",
        },
        "provenance": environment,
    }

    dump_rows(output / "raw/pair_rows.jsonl", pair_rows)
    dump_rows(output / "raw/certificate_rows.jsonl", certificate_rows)
    dump_rows(output / "raw/token_rows.jsonl", token_rows)
    dump(output / "raw/control_rows.json", control_rows)
    dump_rows(output / "raw/timing_rows.jsonl", timing_rows)
    dump(output / "raw/static_certificate.json", static_certificate)
    dump(output / "raw/checkpoint_manifest.json", inputs["checkpoint_rows"])
    dump(
        output / "raw/input_audit.json",
        {
            "file_hashes": inputs["actual_hashes"],
            "family_counts": inputs["family_counts"],
        },
    )
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
