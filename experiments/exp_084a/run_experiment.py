#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np

from experiments.exp_083b.run_experiment import load_target, tokenize_prompt
from vortex_runtime.causal_bilinear_rank_gate import (
    CausalBilinearGateError,
    FactorizedModularSpan,
    PRIMES,
    exact_coefficients_at_pivots,
    factor_pair_sha256,
    first_exact_residual_coordinate,
    frozen_side_residual,
    summarize_gate,
    two_pass_mgs_basis,
    verify_fingerprint_witness,
    witness_strings,
)
from vortex_runtime.causal_residual_atlas_oracle import canonical_sha256
from vortex_runtime.fractal_oracle import validate_prompt_and_trace_ids


ROOT = Path(__file__).resolve().parents[2]


class GateControlError(RuntimeError):
    """A frozen input or native replay control invalidated the run."""


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def validate_source_freeze(config: dict[str, Any]) -> str:
    implementation = str(config["source_freeze"]["implementation_commit"])
    if implementation == "PENDING_SOURCE_FREEZE":
        raise GateControlError("source freeze has not been committed")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()
    if status:
        raise GateControlError("tracked worktree must be clean before model execution")
    protected = [str(value) for value in config["source_freeze"]["protected_paths"]]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", implementation, "HEAD", "--", *protected],
        cwd=ROOT,
        text=True,
    ).strip()
    if changed:
        raise GateControlError(f"protected source changed after freeze: {changed}")
    return implementation


def append_control(
    rows: list[dict[str, Any]],
    failures: list[str],
    *,
    prompt_id: str,
    name: str,
    passed: bool,
    measured: Any | None = None,
) -> None:
    row = {"prompt_id": prompt_id, "control": name, "passed": bool(passed)}
    if measured is not None:
        row["measured"] = measured
    rows.append(row)
    if not passed:
        failures.append(f"{prompt_id}:{name}")


class LastDownRecorder:
    def __init__(self, layer: Any, final_norm: Any) -> None:
        self.active = False
        self.values: dict[str, Any] = {}
        self.handles = [
            layer.post_attention_layernorm.register_forward_pre_hook(
                self._residual_hook
            ),
            layer.mlp.down_proj.register_forward_hook(self._down_hook),
            final_norm.register_forward_pre_hook(self._norm_hook),
        ]

    def _residual_hook(self, module: Any, arguments: tuple[Any, ...]) -> None:
        if self.active:
            self._store("residual", arguments[0])

    def _down_hook(
        self, module: Any, arguments: tuple[Any, ...], output: Any
    ) -> None:
        if self.active:
            self._store("x", arguments[0])
            self._store("y", output)

    def _norm_hook(self, module: Any, arguments: tuple[Any, ...]) -> None:
        if self.active:
            self._store("pre_norm", arguments[0])

    def _store(self, name: str, value: Any) -> None:
        if value.ndim != 3 or int(value.shape[0]) != 1:
            raise GateControlError(f"malformed recorder tensor: {name}")
        self.values[name] = value.detach().cpu().clone()

    def start(self) -> None:
        self.values = {}
        self.active = True

    def stop(self) -> dict[str, Any]:
        self.active = False
        if set(self.values) != {"x", "y", "residual", "pre_norm"}:
            raise GateControlError(f"incomplete last-down capture: {set(self.values)}")
        lengths = {int(value.shape[1]) for value in self.values.values()}
        if len(lengths) != 1:
            raise GateControlError("last-down capture sequence mismatch")
        return self.values

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def winner_competitor(logits: Any) -> tuple[int, int]:
    import torch

    row = logits.detach().reshape(-1)
    winner = int(torch.argmax(row).item())
    masked = row.clone()
    masked[winner] = float("-inf")
    competitor = int(torch.argmax(masked).item())
    if winner == competitor:
        raise GateControlError("winner and competitor must differ")
    return winner, competitor


def decision_direction(
    *,
    residual: Any,
    native_y: Any,
    winner: int,
    competitor: int,
    final_norm: Any,
    lm_head: Any,
) -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as functional

    with torch.enable_grad():
        y = native_y.detach().clone().requires_grad_(True)
        pre_norm = residual.detach() + y
        hidden = final_norm(pre_norm[None, None, :])[0, 0]
        ids = torch.tensor([winner, competitor], dtype=torch.long, device="cpu")
        selected_weight = lm_head.weight.detach().index_select(0, ids)
        selected_logits = functional.linear(hidden, selected_weight)
        margin = selected_logits[0] - selected_logits[1]
        gradient = torch.autograd.grad(margin, y, retain_graph=False)[0]
    return pre_norm.detach(), hidden.detach(), gradient.detach()


def capture_forward(
    *,
    target: Any,
    recorder: LastDownRecorder,
    input_ids: Any,
    past_key_values: Any | None = None,
) -> tuple[Any, dict[str, Any], Any]:
    import torch

    recorder.start()
    try:
        with torch.inference_mode():
            kwargs = {"input_ids": input_ids, "use_cache": True}
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            output = target.model(**kwargs)
    finally:
        captures = recorder.stop()
    return output, captures, output.last_hidden_state.detach()


def prompt_state(
    *,
    prompt: dict[str, Any],
    trace: dict[str, Any],
    target: Any,
    tokenizer: Any,
    recorder: LastDownRecorder,
    side_rank: int,
    control_rows: list[dict[str, Any]],
    control_failures: list[str],
) -> dict[str, Any]:
    import torch

    prompt_id = str(prompt["id"])
    prefix_ids = tokenize_prompt(tokenizer, str(prompt["prompt"]), max_tokens=192)
    prefix_sha = tensor_sha256(prefix_ids)
    append_control(
        control_rows,
        control_failures,
        prompt_id=prompt_id,
        name="prefix_token_trace_identity",
        passed=(
            int(prefix_ids.shape[1]) == int(trace["prefix_token_count"])
            and prefix_sha == str(trace["prefix_token_sha256"])
        ),
        measured={"length": int(prefix_ids.shape[1]), "sha256": prefix_sha},
    )
    if control_failures:
        raise GateControlError(control_failures[-1])

    length = int(prefix_ids.shape[1])
    x_sources: list[np.ndarray] = []
    v_sources: list[np.ndarray] = []
    basis_positions: list[int] = []
    q_basis = np.empty((int(target.model.layers[-1].mlp.down_proj.in_features), 0), dtype=np.float32)
    p_basis = np.empty((int(target.model.layers[-1].mlp.down_proj.out_features), 0), dtype=np.float32)
    cache: Any | None = None
    first_token: int | None = None
    sequential_suffix_match = True
    for position in range(length):
        current_ids = prefix_ids[:, position : position + 1]
        output, captures, hidden = capture_forward(
            target=target,
            recorder=recorder,
            input_ids=current_ids,
            past_key_values=cache,
        )
        cache = output.past_key_values
        basis_open = q_basis.shape[1] < side_rank or p_basis.shape[1] < side_rank
        if basis_open or position == length - 1:
            with torch.inference_mode():
                logits = target.lm_head(hidden[:, -1, :]).detach()[0]
            winner, competitor = winner_competitor(logits)
            if position == length - 1:
                first_token = winner
            if basis_open:
                replay_pre, replay_hidden, gradient = decision_direction(
                    residual=captures["residual"][0, 0],
                    native_y=captures["y"][0, 0],
                    winner=winner,
                    competitor=competitor,
                    final_norm=target.model.norm,
                    lm_head=target.lm_head,
                )
                sequential_suffix_match &= bool(
                    torch.equal(replay_pre, captures["pre_norm"][0, 0])
                    and torch.equal(replay_hidden, hidden[0, 0])
                )
                x_sources.append(
                    captures["x"][0, 0]
                    .float()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                v_sources.append(
                    gradient.float().numpy().astype(np.float32, copy=False)
                )
                basis_positions.append(position)
                q_basis = two_pass_mgs_basis(
                    np.stack(x_sources), maximum_rank=side_rank
                )
                p_basis = two_pass_mgs_basis(
                    np.stack(v_sources), maximum_rank=side_rank
                )
            del logits
    append_control(
        control_rows,
        control_failures,
        prompt_id=prompt_id,
        name="sequential_prefix_suffix_identity",
        passed=sequential_suffix_match,
    )
    x_rows = np.ascontiguousarray(np.stack(x_sources), dtype=np.float32)
    v_rows = np.ascontiguousarray(np.stack(v_sources), dtype=np.float32)
    append_control(
        control_rows,
        control_failures,
        prompt_id=prompt_id,
        name="prompt_primal_basis_rank",
        passed=(q_basis.shape[1] == side_rank),
        measured=int(q_basis.shape[1]),
    )
    append_control(
        control_rows,
        control_failures,
        prompt_id=prompt_id,
        name="prompt_dual_basis_rank",
        passed=(p_basis.shape[1] == side_rank),
        measured=int(p_basis.shape[1]),
    )
    append_control(
        control_rows,
        control_failures,
        prompt_id=prompt_id,
        name="prompt_basis_current_decode_exclusion",
        passed=bool(basis_positions and max(basis_positions) < length),
        measured=basis_positions,
    )
    if first_token is None:
        raise GateControlError("sequential prompt did not produce a final token")
    append_control(
        control_rows,
        control_failures,
        prompt_id=prompt_id,
        name="prompt_top1_exp076_trace_identity",
        passed=(first_token == int(trace["first_target_token"])),
        measured={"observed": first_token, "expected": int(trace["first_target_token"])},
    )
    if control_failures:
        raise GateControlError(control_failures[-1])
    row = {
        "prompt_id": prompt_id,
        "family": prompt["family"],
        "split": prompt["split"],
        "prompt_length": length,
        "prefix_token_sha256": prefix_sha,
        "prompt_last_greedy_token": first_token,
        "prompt_x_sha256": sha256_array(x_rows),
        "prompt_v_sha256": sha256_array(v_rows),
        "q_basis_sha256": sha256_array(q_basis),
        "p_basis_sha256": sha256_array(p_basis),
        "q_rank": int(q_basis.shape[1]),
        "p_rank": int(p_basis.shape[1]),
        "basis_source_positions": basis_positions,
        "prompt_execution": "sequential_single_token_committed_prefix",
    }
    return {
        "row": row,
        "cache": cache,
        "token": first_token,
        "q_basis": q_basis,
        "p_basis": p_basis,
    }


def decode_query(
    *,
    prompt_id: str,
    position: int,
    token: int,
    cache: Any,
    q_basis: np.ndarray,
    p_basis: np.ndarray,
    target: Any,
    recorder: LastDownRecorder,
    control_rows: list[dict[str, Any]],
    control_failures: list[str],
) -> dict[str, Any]:
    import torch

    input_ids = torch.tensor([[token]], dtype=torch.long, device="cpu")
    output, captures, hidden = capture_forward(
        target=target,
        recorder=recorder,
        input_ids=input_ids,
        past_key_values=cache,
    )
    with torch.inference_mode():
        logits = target.lm_head(hidden[:, -1, :]).detach()[0]
    winner, competitor = winner_competitor(logits)
    replay_pre, replay_hidden, gradient = decision_direction(
        residual=captures["residual"][0, 0],
        native_y=captures["y"][0, 0],
        winner=winner,
        competitor=competitor,
        final_norm=target.model.norm,
        lm_head=target.lm_head,
    )
    suffix_match = bool(
        torch.equal(replay_pre, captures["pre_norm"][0, 0])
        and torch.equal(replay_hidden, hidden[0, 0])
    )
    append_control(
        control_rows,
        control_failures,
        prompt_id=prompt_id,
        name=f"decode_{position}_suffix_identity",
        passed=suffix_match,
    )
    x = captures["x"][0, 0].float().numpy().astype(np.float32, copy=False)
    v = gradient.float().numpy().astype(np.float32, copy=False)
    u = frozen_side_residual(x, q_basis)
    r = frozen_side_residual(v, p_basis)
    finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(r)))
    append_control(
        control_rows,
        control_failures,
        prompt_id=prompt_id,
        name=f"decode_{position}_finite_query",
        passed=finite,
    )
    if control_failures:
        raise GateControlError(control_failures[-1])
    return {
        "position": position,
        "input_token": token,
        "winner": winner,
        "competitor": competitor,
        "native_logits_sha256": tensor_sha256(logits),
        "x_sha256": sha256_array(x),
        "v_sha256": sha256_array(v),
        "r": np.ascontiguousarray(r),
        "u": np.ascontiguousarray(u),
        "pair_sha256": factor_pair_sha256(r, u),
        "cache": output.past_key_values,
        "next_token": winner,
    }


def save_pair(output: Path, row_id: str, left: np.ndarray, right: np.ndarray) -> str:
    relative = f"raw/query_arrays/{row_id}.npz"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, r=left, u=right)
    return relative


def deterministic_core(
    *,
    source: dict[str, Any],
    registered_inputs: dict[str, Any],
    prompt_rows: list[dict[str, Any]],
    build_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": source,
        "registered_inputs": registered_inputs,
        "prompt_rows": prompt_rows,
        "build_rows": build_rows,
        "evaluation_rows": evaluation_rows,
        "control_rows": control_rows,
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
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
    config = load(config_path)
    implementation_commit = validate_source_freeze(config)
    actual_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
    }
    for name, actual in actual_versions.items():
        if str(config["runtime"][name]) != str(actual):
            raise GateControlError(f"runtime mismatch {name}: {actual}")
    authority_path = ROOT / config["preregistration"]["authority"]
    prompt_path = ROOT / config["registered_inputs"]["prompts_path"]
    trace_path = ROOT / config["registered_inputs"]["target_trace_path"]
    weight_path = model_dir / config["checkpoint"]["weight_file"]
    for path, expected in (
        (authority_path, config["preregistration"]["authority_sha256"]),
        (prompt_path, config["registered_inputs"]["prompts_sha256"]),
        (trace_path, config["registered_inputs"]["target_trace_sha256"]),
        (weight_path, config["checkpoint"]["weight_sha256"]),
    ):
        observed = sha256_file(path)
        if observed != expected:
            raise GateControlError(f"registered SHA mismatch: {path}: {observed}")

    prompts = load(prompt_path)
    traces = load_rows(trace_path)
    joined = validate_prompt_and_trace_ids(prompts, traces)
    prompt_rows_input: list[dict[str, Any]] = []
    for split in ("build", "evaluation"):
        prompt_rows_input.extend({**row, "split": split} for row in prompts[split])
    prompt_by_id = {str(row["id"]): row for row in prompt_rows_input}
    trace_by_id = {str(row["prompt_id"]): row for row in traces}
    expected_families = sorted(str(value) for value in config["registered_inputs"]["required_families"])
    if sorted(joined["families"]) != expected_families:
        raise GateControlError("registered family set mismatch")

    output.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter_ns()
    torch.set_num_threads(int(config["runtime"]["torch_num_threads"]))
    torch.use_deterministic_algorithms(True)
    target, tokenizer, parameter_audit = load_target(
        model_dir, layer_index=int(config["gate"]["layer_index"])
    )
    expected_shape = [
        int(config["gate"]["projection_output_rows"]),
        int(config["gate"]["projection_input_width"]),
    ]
    if parameter_audit["down_projection_shape"] != expected_shape:
        raise GateControlError("registered down projection shape mismatch")
    layer = target.model.layers[int(config["gate"]["layer_index"])]
    recorder = LastDownRecorder(layer, target.model.norm)

    control_rows: list[dict[str, Any]] = []
    control_failures: list[str] = []
    observed_prompt_rows: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    log_lines: list[str] = []
    basis_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    exact_pivots: list[tuple[int, int]] = []
    primes = tuple(int(value) for value in config["gate"]["primes"])
    if primes != PRIMES:
        raise GateControlError("registered prime order mismatch")
    build_spans = {
        prime: FactorizedModularSpan(
            int(config["gate"]["projection_output_rows"]),
            int(config["gate"]["projection_input_width"]),
            prime,
        )
        for prime in primes
    }
    oracle_spans = {
        prime: FactorizedModularSpan(
            int(config["gate"]["projection_output_rows"]),
            int(config["gate"]["projection_input_width"]),
            prime,
        )
        for prime in primes
    }
    stop_reason: str | None = None

    try:
        for prompt in prompts["build"]:
            registered = {**prompt, "split": "build"}
            prompt_id = str(registered["id"])
            state = prompt_state(
                prompt=registered,
                trace=trace_by_id[prompt_id],
                target=target,
                tokenizer=tokenizer,
                recorder=recorder,
                side_rank=int(config["gate"]["side_rank"]),
                control_rows=control_rows,
                control_failures=control_failures,
            )
            observed_prompt_rows.append(state["row"])
            for position in range(1, int(config["gate"]["build_decode_positions"]) + 1):
                query = decode_query(
                    prompt_id=prompt_id,
                    position=position,
                    token=int(state["token"]),
                    cache=state["cache"],
                    q_basis=state["q_basis"],
                    p_basis=state["p_basis"],
                    target=target,
                    recorder=recorder,
                    control_rows=control_rows,
                    control_failures=control_failures,
                )
                pair = (query.pop("r"), query.pop("u"))
                state["cache"] = query.pop("cache")
                state["token"] = query.pop("next_token")
                modular_increments: dict[str, bool] = {}
                modular_ranks: dict[str, int] = {}
                for prime, span in build_spans.items():
                    incremented, _ = span.add(*pair)
                    modular_increments[str(prime)] = incremented
                    modular_ranks[str(prime)] = span.rank
                coefficients = exact_coefficients_at_pivots(
                    basis_pairs, exact_pivots, pair
                )
                residual_coordinate = first_exact_residual_coordinate(
                    basis_pairs, pair, coefficients
                )
                accepted = bool(
                    residual_coordinate is not None
                    and len(basis_pairs) < int(config["gate"]["ledger_dimension"])
                )
                if accepted:
                    basis_pairs.append(pair)
                    exact_pivots.append(residual_coordinate)  # type: ignore[arg-type]
                row_id = f"{prompt_id}_p{position}"
                array_file = save_pair(output, row_id, *pair)
                row = {
                    **query,
                    "row_id": row_id,
                    "prompt_id": prompt_id,
                    "family": registered["family"],
                    "split": "build",
                    "array_file": array_file,
                    "ledger_accepted": accepted,
                    "ledger_dimension_after": len(basis_pairs),
                    "exact_residual_coordinate": (
                        list(residual_coordinate) if residual_coordinate is not None else None
                    ),
                    "exact_coefficients": witness_strings(coefficients),
                    "modular_increments": modular_increments,
                    "modular_ranks": modular_ranks,
                }
                build_rows.append(row)
                log_lines.append(json.dumps({"row_id": row_id, "ledger_dimension": len(basis_pairs)}))

        append_control(
            control_rows,
            control_failures,
            prompt_id="build_population",
            name="registered_build_row_count",
            passed=(len(build_rows) == int(config["gate"]["expected_build_rows"])),
            measured=len(build_rows),
        )
        append_control(
            control_rows,
            control_failures,
            prompt_id="build_population",
            name="ledger_dimension_bound",
            passed=(len(basis_pairs) <= int(config["gate"]["ledger_dimension"])),
            measured=len(basis_pairs),
        )
        for span in build_spans.values():
            span.validate()
        if control_failures:
            raise GateControlError(control_failures[-1])

        for prompt in prompts["evaluation"]:
            registered = {**prompt, "split": "evaluation"}
            prompt_id = str(registered["id"])
            state = prompt_state(
                prompt=registered,
                trace=trace_by_id[prompt_id],
                target=target,
                tokenizer=tokenizer,
                recorder=recorder,
                side_rank=int(config["gate"]["side_rank"]),
                control_rows=control_rows,
                control_failures=control_failures,
            )
            observed_prompt_rows.append(state["row"])
            for position in range(1, int(config["gate"]["evaluation_decode_positions"]) + 1):
                query = decode_query(
                    prompt_id=prompt_id,
                    position=position,
                    token=int(state["token"]),
                    cache=state["cache"],
                    q_basis=state["q_basis"],
                    p_basis=state["p_basis"],
                    target=target,
                    recorder=recorder,
                    control_rows=control_rows,
                    control_failures=control_failures,
                )
                pair = (query.pop("r"), query.pop("u"))
                state["cache"] = query.pop("cache")
                state["token"] = query.pop("next_token")
                coefficients = exact_coefficients_at_pivots(
                    basis_pairs, exact_pivots, pair
                )
                residual_coordinate = first_exact_residual_coordinate(
                    basis_pairs, pair, coefficients
                )
                exact_hit = residual_coordinate is None
                fingerprint_fields: list[int] = []
                fingerprint_pass = True
                if exact_hit:
                    for prime in primes:
                        try:
                            passed = verify_fingerprint_witness(
                                basis_pairs,
                                pair,
                                coefficients,
                                prime=prime,
                                seeds=config["gate"]["fingerprint_seeds"],
                            )
                        except CausalBilinearGateError:
                            continue
                        fingerprint_fields.append(prime)
                        fingerprint_pass &= passed
                    fingerprint_pass &= bool(fingerprint_fields)
                    append_control(
                        control_rows,
                        control_failures,
                        prompt_id=prompt_id,
                        name=f"decode_{position}_exact_hit_fingerprints",
                        passed=fingerprint_pass,
                        measured=fingerprint_fields,
                    )
                membership_ranks: dict[str, int] = {}
                for prime in primes:
                    span = FactorizedModularSpan(
                        int(config["gate"]["projection_output_rows"]),
                        int(config["gate"]["projection_input_width"]),
                        prime,
                    )
                    for basis_pair in basis_pairs:
                        span.add(*basis_pair)
                    span.add(*pair)
                    membership_ranks[str(prime)] = span.rank
                oracle_increments: dict[str, bool] = {}
                oracle_ranks: dict[str, int] = {}
                for prime, span in oracle_spans.items():
                    incremented, _ = span.add(*pair)
                    oracle_increments[str(prime)] = incremented
                    oracle_ranks[str(prime)] = span.rank
                row_id = f"{prompt_id}_p{position}"
                array_file = save_pair(output, row_id, *pair)
                row = {
                    **query,
                    "row_id": row_id,
                    "prompt_id": prompt_id,
                    "family": registered["family"],
                    "split": "evaluation",
                    "array_file": array_file,
                    "exact_hit": exact_hit,
                    "exact_residual_coordinate": (
                        list(residual_coordinate) if residual_coordinate is not None else None
                    ),
                    "exact_coefficients": witness_strings(coefficients),
                    "fingerprint_fields": fingerprint_fields,
                    "membership_modular_ranks": membership_ranks,
                    "oracle_modular_increments": oracle_increments,
                    "oracle_modular_ranks": oracle_ranks,
                }
                evaluation_rows.append(row)
                misses = sum(not item["exact_hit"] for item in evaluation_rows)
                oracle_rank = max(span.rank for span in oracle_spans.values())
                log_lines.append(json.dumps({"row_id": row_id, "exact_hit": exact_hit, "misses": misses, "oracle_rank": oracle_rank}))
                if misses > int(config["gate"]["maximum_exact_misses"]):
                    stop_reason = "fifth_exact_miss"
                    break
                if oracle_rank >= int(config["gate"]["oracle_rejection_rank"]):
                    stop_reason = "oracle_rank_28"
                    break
            if stop_reason:
                break
    except GateControlError:
        stop_reason = "control_failure"
    finally:
        recorder.close()

    for span in oracle_spans.values():
        span.validate()
    oracle_rank = max((span.rank for span in oracle_spans.values()), default=0)
    execution_complete = len(evaluation_rows) == int(config["gate"]["expected_evaluation_rows"])
    gate = summarize_gate(
        evaluation_rows,
        control_failures=control_failures,
        execution_complete=execution_complete,
        oracle_rank_lower_bound=oracle_rank,
        expected_rows=int(config["gate"]["expected_evaluation_rows"]),
        maximum_misses=int(config["gate"]["maximum_exact_misses"]),
        minimum_family_hits=int(config["gate"]["minimum_family_hits"]),
    )
    gate.update(
        {
            "stop_reason": stop_reason,
            "ledger_dimension": len(basis_pairs),
            "ledger_pivots": [list(value) for value in exact_pivots],
            "build_modular_ranks": {
                str(prime): span.rank for prime, span in build_spans.items()
            },
            "oracle_modular_ranks": {
                str(prime): span.rank for prime, span in oracle_spans.items()
            },
            "target_future_token_reads": 0,
        }
    )
    source = {
        "implementation_commit": implementation_commit,
        "execution_commit": git_commit(),
        "config_sha256": sha256_file(config_path),
        "protected_paths": config["source_freeze"]["protected_paths"],
    }
    registered_inputs = {
        "authority_sha256": sha256_file(authority_path),
        "prompts_sha256": sha256_file(prompt_path),
        "target_trace_sha256": sha256_file(trace_path),
        "weight_sha256": sha256_file(weight_path),
        "joined_prompt_trace": joined,
        "parameter_audit": parameter_audit,
    }
    core = deterministic_core(
        source=source,
        registered_inputs=registered_inputs,
        prompt_rows=observed_prompt_rows,
        build_rows=build_rows,
        evaluation_rows=evaluation_rows,
        control_rows=control_rows,
        gate=gate,
    )
    core_sha = canonical_sha256(core)
    summary = {
        "experiment": config["experiment"],
        "decision": gate["decision"],
        "evidence_ceiling": config["evidence_ceiling"],
        "deterministic_core_sha256": core_sha,
        "gate": gate,
        "source": source,
        "registered_inputs": registered_inputs,
        "runtime": actual_versions,
        "run_wall_ns": time.perf_counter_ns() - run_started,
    }
    dump(output / "config.snapshot.json", config)
    dump_rows(output / "raw/prompt_rows.jsonl", observed_prompt_rows)
    dump_rows(output / "raw/build_rows.jsonl", build_rows)
    dump_rows(output / "raw/evaluation_rows.jsonl", evaluation_rows)
    dump_rows(output / "raw/control_rows.jsonl", control_rows)
    dump(output / "raw/deterministic_core.json", core)
    dump(output / "summary.json", summary)
    dump(
        output / "result.json",
        {
            "experiment": config["experiment"],
            "decision": gate["decision"],
            "deterministic_core_sha256": core_sha,
            "control_failures": control_failures,
            "stop_reason": stop_reason,
        },
    )
    (output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8", newline="\n")
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
