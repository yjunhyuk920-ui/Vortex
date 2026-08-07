#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import types
from typing import Any, Iterator

from vortex_runtime.fractal_oracle import (
    aggregate_quality_rows,
    canonical_sha256,
    family_aggregates,
    gate_decision,
    quality_gate,
    selected_channel_count,
    selected_parameter_fraction,
    validate_prompt_and_trace_ids,
)


ROOT = Path(__file__).resolve().parents[2]


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


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def write_checksums(output: Path) -> None:
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            entries.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(entries) + "\n", encoding="utf-8", newline="\n"
    )


@contextmanager
def default_dtype(dtype: Any) -> Iterator[None]:
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def validate_registered_inputs(config: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    prompts_path = ROOT / config["registered_inputs"]["prompts_path"]
    traces_path = ROOT / config["registered_inputs"]["target_trace_path"]
    lock_path = ROOT / config["runtime"]["requirements_lock"]
    inherited_lock = ROOT / "experiments/exp_076/requirements.lock.json"
    exp076_config_path = ROOT / config["checkpoint"]["exp076_manifest"]
    for path in (prompts_path, traces_path, lock_path, inherited_lock, exp076_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = {
        prompts_path: config["registered_inputs"]["prompts_sha256"],
        traces_path: config["registered_inputs"]["target_trace_sha256"],
        inherited_lock: json.loads(lock_path.read_text(encoding="utf-8"))[
            "inherited_file_sha256"
        ],
    }
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"registered SHA mismatch for {path}: {actual} != {digest}")
    exp076_config = json.loads(exp076_config_path.read_text(encoding="utf-8"))
    from experiments.exp_076.run_experiment import validate_checkpoint_manifest

    checkpoint_rows = validate_checkpoint_manifest(model_dir, exp076_config)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    traces = load_rows(traces_path)
    join_audit = validate_prompt_and_trace_ids(prompts, traces)
    required = set(config["registered_inputs"]["required_families"])
    if set(join_audit["families"]) != required:
        raise ValueError("registered prompt family set mismatch")
    return {
        "prompts_path": prompts_path,
        "traces_path": traces_path,
        "prompts": prompts,
        "traces": traces,
        "join_audit": join_audit,
        "checkpoint_rows": checkpoint_rows,
        "requirements_lock_sha256": sha256_file(lock_path),
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
        raise ValueError(f"target state mismatch: missing={missing}, unexpected={unexpected}")
    target.tie_weights()
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mlp_rows = []
    for layer_index, layer in enumerate(target.model.layers):
        mlp = layer.mlp
        hidden = int(mlp.hidden_size)
        intermediate = int(mlp.intermediate_size)
        mlp_rows.append(
            {
                "layer_index": layer_index,
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "gate_parameters": int(mlp.gate_proj.weight.numel()),
                "up_parameters": int(mlp.up_proj.weight.numel()),
                "down_parameters": int(mlp.down_proj.weight.numel()),
                "total_parameters": 3 * hidden * intermediate,
            }
        )
    return target, tokenizer, {
        "target_text_parameters": target_text_parameters,
        "tied_lm_head_parameters": embedding_parameters,
        "mlp_parameters": sum(row["total_parameters"] for row in mlp_rows),
        "mlp_shapes": mlp_rows,
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


class OracleTracker:
    def __init__(self, fraction: float, valid_token_mask: Any) -> None:
        self.fraction = fraction
        self.valid_token_mask = valid_token_mask.to(dtype=__import__("torch").bool)
        self.batch_size = int(valid_token_mask.shape[0])
        self.selected_channels = [0] * self.batch_size
        self.total_channels = [0] * self.batch_size
        self.squared_error = [0.0] * self.batch_size
        self.squared_target = [0.0] * self.batch_size
        self.calls = 0

    def record(self, *, selected: int, total: int, candidate: Any, target: Any) -> None:
        delta = candidate.float() - target.float()
        if candidate.ndim != 3 or candidate.shape[:2] != self.valid_token_mask.shape:
            raise ValueError("oracle tracker batch/sequence shape mismatch")
        mask = self.valid_token_mask.to(device=candidate.device).unsqueeze(-1)
        errors = (delta.square() * mask).sum(dim=(1, 2)).tolist()
        targets = (target.float().square() * mask).sum(dim=(1, 2)).tolist()
        positions = self.valid_token_mask.sum(dim=1).tolist()
        for index in range(self.batch_size):
            self.selected_channels[index] += int(positions[index]) * selected
            self.total_channels[index] += int(positions[index]) * total
            self.squared_error[index] += float(errors[index])
            self.squared_target[index] += float(targets[index])
        self.calls += 1

    def case_summary(self, index: int) -> dict[str, Any]:
        return {
            "mlp_calls": self.calls,
            "selected_channels": self.selected_channels[index],
            "total_channels": self.total_channels[index],
            "realized_channel_fraction": self.selected_channels[index]
            / self.total_channels[index],
            "mlp_relative_l2": math.sqrt(
                self.squared_error[index] / max(self.squared_target[index], 1e-30)
            ),
        }


@contextmanager
def activation_norm_oracle(target: Any, fraction: float, tracker: OracleTracker) -> Iterator[None]:
    import torch

    originals: list[tuple[Any, Any]] = []
    for layer in target.model.layers:
        mlp = layer.mlp
        original = mlp.forward
        down_norm = mlp.down_proj.weight.detach().float().square().sum(dim=0).sqrt()
        keep = selected_channel_count(int(mlp.intermediate_size), fraction)

        def oracle_forward(self: Any, x: Any, *, _norm=down_norm, _keep=keep) -> Any:
            intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            scores = intermediate.detach().float().abs() * _norm
            indices = torch.topk(scores, k=_keep, dim=-1, sorted=False).indices
            mask = torch.zeros_like(intermediate, dtype=torch.bool)
            mask.scatter_(-1, indices, True)
            candidate = self.down_proj(intermediate.masked_fill(~mask, 0))
            with torch.no_grad():
                full = self.down_proj(intermediate)
                tracker.record(
                    selected=_keep,
                    total=int(intermediate.shape[-1]),
                    candidate=candidate,
                    target=full,
                )
            return candidate

        originals.append((mlp, original))
        mlp.forward = types.MethodType(oracle_forward, mlp)
    try:
        yield
    finally:
        for mlp, original in originals:
            mlp.forward = original


def logits_sha256(logits: Any) -> str:
    value = logits.detach().contiguous().view(dtype=__import__("torch").uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def compare_logits(target_logits: Any, candidate_logits: Any) -> dict[str, Any]:
    import torch

    target = target_logits.float()
    candidate = candidate_logits.float()
    target_log_probs = torch.log_softmax(target, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate, dim=-1)
    target_probs = target_log_probs.exp()
    kls = (target_probs * (target_log_probs - candidate_log_probs)).sum(dim=-1)
    target_top = target.argmax(dim=-1)
    candidate_top = candidate.argmax(dim=-1)
    return {
        "token_count": int(target.shape[0]),
        "top1_matches": int((target_top == candidate_top).sum().item()),
        "target_top_tokens": [int(value) for value in target_top.tolist()],
        "candidate_top_tokens": [int(value) for value in candidate_top.tolist()],
        "token_kls": [max(0.0, float(value)) for value in kls.tolist()],
    }


def select_prediction_logits(logits: Any, prompt_length: int, token_count: int) -> Any:
    start = prompt_length - 1
    stop = start + token_count
    selected = logits[0, start:stop, :]
    if selected.shape[0] != token_count:
        raise ValueError("model output lacks registered prediction positions")
    return selected


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
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_started = time.perf_counter_ns()
    torch.set_num_threads(int(config["runtime"]["torch_num_threads"]))
    torch.use_deterministic_algorithms(True)
    inputs = validate_registered_inputs(config, model_dir)
    target, tokenizer, parameter_audit = load_target(model_dir)

    prompts_by_id = {
        row["id"]: {**row, "split": split}
        for split in ("build", "evaluation")
        for row in inputs["prompts"][split]
    }
    traces_by_id = {row["prompt_id"]: row for row in inputs["traces"]}
    token_count = int(config["registered_inputs"]["tokens_per_prompt"])
    fractions = [float(value) for value in config["oracle"]["fractions"]]
    baseline_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    log_lines: list[str] = []
    baseline_mismatches = 0

    ordered_ids = [
        row["id"]
        for split in ("build", "evaluation")
        for row in inputs["prompts"][split]
    ]
    prepared: list[dict[str, Any]] = []
    for prompt_id in ordered_ids:
        prompt = prompts_by_id[prompt_id]
        trace = traces_by_id[prompt_id]
        continuation = [int(trace["first_target_token"]), *map(int, trace["target_verification_tokens"])]
        continuation = continuation[:token_count]
        if len(continuation) != token_count:
            raise ValueError(f"short registered target trace for {prompt_id}")
        prefix_ids = tokenize_prompt(tokenizer, prompt["prompt"])
        teacher_input = torch.cat(
            [
                prefix_ids,
                torch.tensor([continuation[:-1]], dtype=torch.long),
            ],
            dim=1,
        )
        prepared.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "continuation": continuation,
                "prefix_length": int(prefix_ids.shape[1]),
                "teacher_input": teacher_input[0],
            }
        )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer provides no padding or EOS token")
    max_teacher_length = max(int(row["teacher_input"].shape[0]) for row in prepared)
    batch_input = torch.full(
        (len(prepared), max_teacher_length), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros_like(batch_input)
    for index, row in enumerate(prepared):
        length = int(row["teacher_input"].shape[0])
        batch_input[index, :length] = row["teacher_input"]
        attention_mask[index, :length] = 1

    with torch.inference_mode():
        baseline_output = target(
            input_ids=batch_input, attention_mask=attention_mask, use_cache=False
        )
    baseline_logits_by_id: dict[str, Any] = {}
    for index, row in enumerate(prepared):
        prompt_id = row["prompt_id"]
        prompt = row["prompt"]
        continuation = row["continuation"]
        baseline_logits = select_prediction_logits(
            baseline_output.logits[index : index + 1],
            row["prefix_length"],
            token_count,
        ).detach().clone()
        baseline_logits_by_id[prompt_id] = baseline_logits
        baseline_top = [int(value) for value in baseline_logits.argmax(dim=-1).tolist()]
        mismatch_count = sum(
            observed != expected
            for observed, expected in zip(baseline_top, continuation)
        )
        baseline_mismatches += mismatch_count
        baseline_rows.append(
            {
                "prompt_id": prompt_id,
                "split": prompt["split"],
                "family": prompt["family"],
                "prefix_token_count": row["prefix_length"],
                "teacher_token_count": token_count,
                "registered_target_tokens": continuation,
                "baseline_top_tokens": baseline_top,
                "trace_mismatches": mismatch_count,
                "baseline_logits_sha256": logits_sha256(baseline_logits),
            }
        )
    del baseline_output

    for fraction in fractions:
        tracker = OracleTracker(fraction, attention_mask)
        started = time.perf_counter_ns()
        with activation_norm_oracle(target, fraction, tracker), torch.inference_mode():
            candidate_output = target(
                input_ids=batch_input, attention_mask=attention_mask, use_cache=False
            )
        batch_wall_ns = time.perf_counter_ns() - started
        for index, prepared_row in enumerate(prepared):
            prompt_id = prepared_row["prompt_id"]
            prompt = prepared_row["prompt"]
            candidate_logits = select_prediction_logits(
                candidate_output.logits[index : index + 1],
                prepared_row["prefix_length"],
                token_count,
            )
            baseline_logits = baseline_logits_by_id[prompt_id]
            comparison = compare_logits(baseline_logits, candidate_logits)
            tracker_summary = tracker.case_summary(index)
            row = {
                "prompt_id": prompt_id,
                "split": prompt["split"],
                "family": prompt["family"],
                "fraction": fraction,
                **comparison,
                **tracker_summary,
                "candidate_logits_sha256": logits_sha256(candidate_logits),
                "fraction_batch_wall_ns": batch_wall_ns,
            }
            case_rows.append(row)
            line = json.dumps(
                {
                    "prompt_id": prompt_id,
                    "split": prompt["split"],
                    "fraction": fraction,
                    "top1_matches": comparison["top1_matches"],
                    "token_count": token_count,
                    "mean_kl": sum(comparison["token_kls"]) / token_count,
                    "mlp_relative_l2": tracker_summary["mlp_relative_l2"],
                },
                sort_keys=True,
            )
            print(line, flush=True)
            log_lines.append(line)

    fraction_aggregates: dict[str, Any] = {}
    for fraction in fractions:
        key = format(fraction, ".12g")
        subset = [row for row in case_rows if row["fraction"] == fraction]
        evaluation = [row for row in subset if row["split"] == "evaluation"]
        fraction_aggregates[key] = {
            "build": aggregate_quality_rows(
                [row for row in subset if row["split"] == "build"]
            ),
            "evaluation": aggregate_quality_rows(evaluation),
            "evaluation_families": family_aggregates(evaluation),
        }

    gate_fraction = float(config["oracle"]["gate_fraction"])
    gate_key = format(gate_fraction, ".12g")
    gate_population = fraction_aggregates[gate_key]["evaluation"]
    gate_families = fraction_aggregates[gate_key]["evaluation_families"]
    first_shape = parameter_audit["mlp_shapes"][0]
    selected = selected_channel_count(first_shape["intermediate_size"], gate_fraction)
    realized_parameter_fraction = selected_parameter_fraction(
        hidden_size=first_shape["hidden_size"],
        intermediate_size=first_shape["intermediate_size"],
        selected_channels=selected,
    )
    gates = quality_gate(
        population=gate_population,
        families=gate_families,
        baseline_mismatches=baseline_mismatches,
        selected_fraction=realized_parameter_fraction,
        max_fraction=float(config["quality_gate"]["max_mlp_parameter_fraction"]),
        min_top1_agreement=float(config["quality_gate"]["min_top1_agreement"]),
        min_family_top1_agreement=float(
            config["quality_gate"]["min_each_family_top1_agreement"]
        ),
        max_mean_kl=float(
            config["quality_gate"]["max_mean_target_to_candidate_kl"]
        ),
        max_p95_kl=float(
            config["quality_gate"]["max_p95_target_to_candidate_kl"]
        ),
    )
    decision = gate_decision(gates)
    deterministic_core = {
        "baseline_trace_mismatches": baseline_mismatches,
        "fraction_aggregates": fraction_aggregates,
        "gate_fraction": gate_fraction,
        "realized_mlp_parameter_fraction": realized_parameter_fraction,
        "gates": gates,
        "decision": decision,
        "input_join_sha256": inputs["join_audit"]["joined_core_sha256"],
        "token_core": [
            {
                "prompt_id": row["prompt_id"],
                "fraction": row["fraction"],
                "target_top_tokens": row["target_top_tokens"],
                "candidate_top_tokens": row["candidate_top_tokens"],
                "token_kls": row["token_kls"],
            }
            for row in case_rows
        ],
    }
    core_hash = canonical_sha256(deterministic_core)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "source_commit": git_commit(),
        "config_sha256": sha256_file(config_path),
        "requirements_lock_sha256": inputs["requirements_lock_sha256"],
    }
    summary = {
        "experiment": "EXP-077A",
        "name": config["name"],
        "phase": ["C-small-real-checkpoint-favorable-oracle-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "REGISTERED_EXTERNAL_INPUTS": {
            "checkpoint": {
                "model_id": config["checkpoint"]["model_id"],
                "revision": config["checkpoint"]["revision"],
                "weight_sha256": config["checkpoint"]["weight_sha256"],
            },
            "prompts_sha256": config["registered_inputs"]["prompts_sha256"],
            "target_trace_sha256": config["registered_inputs"][
                "target_trace_sha256"
            ],
            "tokens_per_prompt": token_count,
        },
        "MEASURED": {
            "case_count": len(case_rows),
            "baseline_case_count": len(baseline_rows),
            "teacher_forced_token_count_per_fraction": len(baseline_rows)
            * token_count,
            "baseline_trace_mismatches": baseline_mismatches,
            "wall_ns": time.perf_counter_ns() - run_started,
            "deterministic_core_sha256": core_hash,
        },
        "DERIVED": {
            "parameter_audit": parameter_audit,
            "gate_fraction": gate_fraction,
            "gate_selected_channels_per_mlp": selected,
            "realized_mlp_parameter_fraction": realized_parameter_fraction,
            "fraction_aggregates": fraction_aggregates,
            "gate": gates,
            "decision": decision,
            "projection": config["projection"],
        },
        "UNVERIFIED": [
            "causal selector that avoids full gate/up computation",
            "selector metadata and runtime cost",
            "attention DeltaNet and LM-head fracturing",
            "autoregressive generation quality after divergent tokens",
            "ground-truth benchmark quality",
            "physical sparse kernels traffic latency and VRAM",
            "35B 122B MoE scaling and expert locality",
            "arbitrary dense 405B transfer",
            "E2 through E7",
        ],
        "claim_boundary": {
            "checkpoint": "UNCHANGED_PINNED_OFFICIAL_BF16_PAYLOAD",
            "selector": "NON_DEPLOYABLE_ORACLE_USING_FULL_POST_SILU_INTERMEDIATE_ACTIVATIONS",
            "teacher_forcing": "EXACT_REGISTERED_TARGET_PREFIX_ONLY",
            "operation_replacement": "MLP_OUTPUT_REPLACED_IN_REFERENCE_FORWARD_BUT_SELECTOR_COST_GRANTED_FREE",
            "quality": "TARGET_LOGIT_DISTRIBUTION_AGREEMENT_NOT_GROUND_TRUTH_TASK_QUALITY",
            "physical_speed": "NOT_MEASURED",
            "target_server": "NO_COMMAND_EXECUTED",
        },
        "provenance": environment,
    }
    dump_rows(output / "raw/baseline_rows.jsonl", baseline_rows)
    dump_rows(output / "raw/case_rows.jsonl", case_rows)
    dump(output / "raw/checkpoint_manifest.json", inputs["checkpoint_rows"])
    dump(output / "raw/input_audit.json", inputs["join_audit"])
    dump(output / "raw/mlp_shapes.json", parameter_audit["mlp_shapes"])
    dump(output / "processed/aggregate.json", fraction_aggregates)
    dump(output / "artifacts/environment.json", environment)
    (output / "artifacts/contract.txt").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
    )
    dump(output / "summary.json", summary)
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
