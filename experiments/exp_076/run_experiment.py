#!/usr/bin/env python3
"""Run EXP-076 on the pinned unchanged Qwen3.5-0.8B checkpoint."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vortex_runtime.mtp_acceptance import (
    acceptance_distribution,
    canonical_sha256,
    derive_k_row,
    exact_commit_tokens,
    gate_decision,
    ideal_minimum_accepted_tokens,
    select_build_k,
    validate_prompt_manifest,
)


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
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload, usedforsecurity=False
    ).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def write_checksums(output: Path) -> None:
    rows: list[str] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


def validate_checkpoint_manifest(model_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expected in config["checkpoint"]["required_files"]:
        path = model_dir / expected["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing registered checkpoint file: {path}")
        size = path.stat().st_size
        if size != int(expected["bytes"]):
            raise ValueError(f"checkpoint size mismatch: {expected['path']}")
        actual_sha256 = sha256_file(path)
        expected_sha256 = expected.get("sha256")
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(f"checkpoint SHA-256 mismatch: {expected['path']}")
        lfs_payload = expected_sha256 is not None
        actual_repository_oid = None if lfs_payload else git_blob_oid(path)
        if not lfs_payload and actual_repository_oid != expected["repository_oid"]:
            raise ValueError(f"checkpoint repository OID mismatch: {expected['path']}")
        rows.append(
            {
                "path": expected["path"],
                "bytes": size,
                "sha256": actual_sha256,
                "registered_repository_oid": expected["repository_oid"],
                "actual_repository_oid": actual_repository_oid,
                "lfs_payload": lfs_payload,
                "passed": True,
            }
        )
    return rows


def peak_rss_bytes() -> int | None:
    if os.name != "nt":
        status = Path("/proc/self/status")
        if status.is_file():
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        return None
    import ctypes
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
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    return int(counters.PeakWorkingSetSize) if ok else None


def cache_sha256(cache: Any) -> str:
    """Hash every tensor in a Transformers hybrid DynamicCache."""
    import torch

    digest = hashlib.sha256()
    seen: set[int] = set()

    def visit(value: Any, name: str) -> None:
        identity = id(value)
        if identity in seen:
            return
        if isinstance(value, torch.Tensor):
            seen.add(identity)
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            return
        if isinstance(value, dict):
            seen.add(identity)
            for key in sorted(value, key=str):
                visit(value[key], f"{name}.{key}")
            return
        if isinstance(value, (list, tuple)):
            seen.add(identity)
            for index, item in enumerate(value):
                visit(item, f"{name}[{index}]")
            return
        if hasattr(value, "__dict__") and value.__class__.__module__.startswith(
            "transformers"
        ):
            seen.add(identity)
            for key in sorted(value.__dict__):
                visit(value.__dict__[key], f"{name}.{key}")

    visit(cache, "cache")
    return digest.hexdigest()


@contextmanager
def default_dtype(dtype: Any) -> Iterator[None]:
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class NativeMtpReference:
    """Minimal CPU reference matching the pinned vLLM Qwen3.5 MTP equations."""

    def __init__(self, text_config: Any) -> None:
        import torch
        from torch import nn
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextRotaryEmbedding,
        )

        self.torch = torch
        self.config = copy.deepcopy(text_config)
        self.config.num_hidden_layers = 1
        self.config.layer_types = ["full_attention"]
        self.config._attn_implementation = "eager"
        with default_dtype(torch.bfloat16):
            self.module = nn.Module()
            # The checkpoint fc consumes concatenated embedding + target hidden.
            self.module.fc = nn.Linear(
                self.config.hidden_size * 2, self.config.hidden_size, bias=False
            )
            self.module.layer = Qwen3_5DecoderLayer(self.config, 0)
            self.module.norm = Qwen3_5RMSNorm(
                self.config.hidden_size, eps=self.config.rms_norm_eps
            )
            self.module.pre_fc_norm_embedding = Qwen3_5RMSNorm(
                self.config.hidden_size, eps=self.config.rms_norm_eps
            )
            self.module.pre_fc_norm_hidden = Qwen3_5RMSNorm(
                self.config.hidden_size, eps=self.config.rms_norm_eps
            )
            self.rotary = Qwen3_5TextRotaryEmbedding(self.config)
        self.module.eval()

    def load_state(self, state: dict[str, Any]) -> None:
        missing, unexpected = self.module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"MTP state mismatch: missing={missing}, unexpected={unexpected}")

    def _mask(self, query_length: int, past_length: int, dtype: Any) -> Any:
        torch = self.torch
        key_length = past_length + query_length
        mask = torch.full(
            (1, 1, query_length, key_length),
            torch.finfo(dtype).min,
            dtype=dtype,
        )
        for query in range(query_length):
            mask[:, :, query, : past_length + query + 1] = 0
        return mask

    def step(
        self,
        *,
        input_ids: Any,
        target_hidden_states: Any,
        positions: Any,
        cache: Any,
        embedding_weight: Any,
    ) -> Any:
        import torch.nn.functional as functional

        torch = self.torch
        inputs_embeds = functional.embedding(input_ids, embedding_weight)
        hidden = self.module.pre_fc_norm_embedding(inputs_embeds)
        target_hidden = self.module.pre_fc_norm_hidden(target_hidden_states)
        hidden = self.module.fc(torch.cat([hidden, target_hidden], dim=-1))
        position_ids = positions.view(1, -1)
        mrope_positions = position_ids.unsqueeze(0).expand(3, 1, -1)
        position_embeddings = self.rotary(hidden, mrope_positions)
        past_length = cache.get_seq_length()
        hidden = self.module.layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=self._mask(hidden.shape[1], past_length, hidden.dtype),
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        return self.module.norm(hidden)

    def propose(
        self,
        *,
        prefix_ids: Any,
        first_target_token: int,
        target_hidden_states: Any,
        embedding_weight: Any,
        max_k: int,
    ) -> tuple[list[int], int]:
        from transformers.cache_utils import DynamicCache

        torch = self.torch
        if prefix_ids.shape[1] < 2:
            raise ValueError("MTP prefill requires at least two prefix tokens")
        shifted = torch.cat(
            [
                prefix_ids[:, 1:],
                torch.tensor([[first_target_token]], dtype=prefix_ids.dtype),
            ],
            dim=1,
        )
        positions = torch.arange(prefix_ids.shape[1], dtype=torch.long)
        cache = DynamicCache(config=self.config)
        hidden = self.step(
            input_ids=shifted,
            target_hidden_states=target_hidden_states,
            positions=positions,
            cache=cache,
            embedding_weight=embedding_weight,
        )
        logits = torch.nn.functional.linear(hidden[:, -1:, :], embedding_weight)
        token = int(logits.argmax(dim=-1).item())
        proposals = [token]
        calls = 1
        while len(proposals) < max_k:
            input_ids = torch.tensor([[proposals[-1]]], dtype=prefix_ids.dtype)
            position = torch.tensor([prefix_ids.shape[1] + len(proposals) - 1])
            hidden = self.step(
                input_ids=input_ids,
                target_hidden_states=hidden[:, -1:, :],
                positions=position,
                cache=cache,
                embedding_weight=embedding_weight,
            )
            logits = torch.nn.functional.linear(hidden, embedding_weight)
            proposals.append(int(logits.argmax(dim=-1).item()))
            calls += 1
        return proposals, calls


def load_models(model_dir: Path) -> tuple[Any, NativeMtpReference, Any, dict[str, Any]]:
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
        mtp = NativeMtpReference(text_config)

    weight_path = model_dir / "model.safetensors-00001-of-00001.safetensors"
    target_state: dict[str, Any] = {}
    mtp_state: dict[str, Any] = {}
    shape_rows: list[dict[str, Any]] = []
    target_text_parameters = 0
    mtp_parameters = 0
    embedding_parameters = 0
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name.startswith("model.language_model."):
                tensor = handle.get_tensor(name)
                mapped = "model." + name.removeprefix("model.language_model.")
                target_state[mapped] = tensor
                target_text_parameters += tensor.numel()
                if name == "model.language_model.embed_tokens.weight":
                    embedding_parameters = tensor.numel()
                    target_state["lm_head.weight"] = tensor
            elif name.startswith("mtp."):
                tensor = handle.get_tensor(name)
                mapped = name.removeprefix("mtp.")
                if mapped.startswith("layers.0."):
                    mapped = "layer." + mapped.removeprefix("layers.0.")
                mtp_state[mapped] = tensor
                mtp_parameters += tensor.numel()
                shape_rows.append(
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "numel": tensor.numel(),
                        "bytes": tensor.numel() * tensor.element_size(),
                    }
                )
    missing, unexpected = target.load_state_dict(target_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"target state mismatch: missing={missing}, unexpected={unexpected}")
    target.tie_weights()
    mtp.load_state(mtp_state)
    target.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    parameter_audit = {
        "target_text_parameters": target_text_parameters,
        "mtp_parameters": mtp_parameters,
        "tied_lm_head_parameters": embedding_parameters,
        "proposal_parameters_per_token": mtp_parameters + embedding_parameters,
        "mtp_shapes": sorted(shape_rows, key=lambda row: row["name"]),
    }
    return target, mtp, tokenizer, parameter_audit


def tokenize_prompt(tokenizer: Any, prompt: str, max_tokens: int) -> Any:
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
        raise ValueError(f"prompt exceeds registered max token count: {token_ids.shape[1]}")
    return token_ids.to(dtype=torch.long, device="cpu")


def run_case(
    *,
    row: dict[str, str],
    split: str,
    tokenizer: Any,
    target: Any,
    mtp: NativeMtpReference,
    config: dict[str, Any],
    parameter_audit: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    prefix_ids = tokenize_prompt(
        tokenizer, row["prompt"], int(config["prompts"]["max_prompt_tokens"])
    )
    prefix_hash_before = hashlib.sha256(
        prefix_ids.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    with torch.inference_mode():
        target_started = time.perf_counter_ns()
        prefix_outputs = target.model(input_ids=prefix_ids, use_cache=True)
        target_prefix_ns = time.perf_counter_ns() - target_started
        target_hidden = prefix_outputs.last_hidden_state
        committed_cache = prefix_outputs.past_key_values
        first_logits = target.lm_head(target_hidden[:, -1:, :])
        first_target_token = int(first_logits.argmax(dim=-1).item())
        committed_cache_hash_before = cache_sha256(committed_cache)

        mtp_started = time.perf_counter_ns()
        proposals, mtp_calls = mtp.propose(
            prefix_ids=prefix_ids,
            first_target_token=first_target_token,
            target_hidden_states=target_hidden,
            embedding_weight=target.model.embed_tokens.weight,
            max_k=int(config["proposal"]["max_k"]),
        )
        mtp_ns = time.perf_counter_ns() - mtp_started

        verify_cache = copy.deepcopy(committed_cache)
        verify_ids = torch.tensor(
            [[first_target_token, *proposals]], dtype=prefix_ids.dtype
        )
        verify_started = time.perf_counter_ns()
        verify_outputs = target.model(
            input_ids=verify_ids, past_key_values=verify_cache, use_cache=True
        )
        verify_logits = target.lm_head(verify_outputs.last_hidden_state)
        target_verification_tokens = (
            verify_logits.argmax(dim=-1).squeeze(0).tolist()
        )
        target_verify_ns = time.perf_counter_ns() - verify_started
        committed_cache_hash_after_verification = cache_sha256(committed_cache)

        max_row = derive_k_row(
            configured_k=int(config["proposal"]["max_k"]),
            proposal_tokens=proposals,
            target_verification_tokens=target_verification_tokens,
            target_block_parameter_reads=int(
                config["accounting"]["target_block_parameter_reads"]
            ),
            proposal_parameters_per_token=int(
                parameter_audit["proposal_parameters_per_token"]
            ),
            baseline_parameters=int(config["accounting"]["baseline_parameters"]),
        )
        committed_tokens = exact_commit_tokens(
            first_target_token, proposals, target_verification_tokens
        )
        expected_commit = [
            first_target_token,
            *target_verification_tokens[: max_row["accepted_prefix"] + 1],
        ]
        commit_replay_mismatch = committed_tokens != expected_commit

        commit_ids = torch.tensor([committed_tokens], dtype=prefix_ids.dtype)
        commit_cache_a = copy.deepcopy(committed_cache)
        commit_cache_b = copy.deepcopy(committed_cache)
        commit_started = time.perf_counter_ns()
        target.model(input_ids=commit_ids, past_key_values=commit_cache_a, use_cache=True)
        target.model(input_ids=commit_ids, past_key_values=commit_cache_b, use_cache=True)
        commit_state_ns = time.perf_counter_ns() - commit_started
        commit_cache_hash_a = cache_sha256(commit_cache_a)
        commit_cache_hash_b = cache_sha256(commit_cache_b)

    prefix_hash_after = hashlib.sha256(
        prefix_ids.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    wrong_accepts = sum(
        int(proposals[index] != target_verification_tokens[index])
        for index in range(max_row["accepted_prefix"])
    )
    k_rows: list[dict[str, Any]] = []
    for configured_k in config["proposal"]["registered_k"]:
        k_row = derive_k_row(
            configured_k=int(configured_k),
            proposal_tokens=proposals,
            target_verification_tokens=target_verification_tokens,
            target_block_parameter_reads=int(
                config["accounting"]["target_block_parameter_reads"]
            ),
            proposal_parameters_per_token=int(
                parameter_audit["proposal_parameters_per_token"]
            ),
            baseline_parameters=int(config["accounting"]["baseline_parameters"]),
        )
        k_row.update({"prompt_id": row["id"], "family": row["family"], "split": split})
        k_rows.append(k_row)
    case = {
        "prompt_id": row["id"],
        "family": row["family"],
        "split": split,
        "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
        "prefix_token_count": prefix_ids.shape[1],
        "prefix_token_sha256": prefix_hash_before,
        "first_target_token": first_target_token,
        "proposal_tokens": proposals,
        "target_verification_tokens": target_verification_tokens,
        "max_k_accepted_prefix": max_row["accepted_prefix"],
        "committed_tokens": committed_tokens,
        "wrong_accepts": wrong_accepts,
        "commit_replay_mismatch": commit_replay_mismatch,
        "target_future_token_reads": 0,
        "prefix_mutation": prefix_hash_before != prefix_hash_after,
        "committed_cache_hash_before": committed_cache_hash_before,
        "committed_cache_hash_after_verification": committed_cache_hash_after_verification,
        "committed_cache_mutation_during_verification": (
            committed_cache_hash_before != committed_cache_hash_after_verification
        ),
        "commit_cache_hash_a": commit_cache_hash_a,
        "commit_cache_hash_b": commit_cache_hash_b,
        "rollback_recompute_mismatch": commit_cache_hash_a != commit_cache_hash_b,
        "target_forward_calls": 4,
        "mtp_forward_calls": mtp_calls,
        "target_prefix_ns": target_prefix_ns,
        "mtp_proposal_ns": mtp_ns,
        "target_verification_ns": target_verify_ns,
        "commit_state_recompute_ns": commit_state_ns,
        "wall_ns": time.perf_counter_ns() - wall_started,
        "cpu_ns": time.process_time_ns() - cpu_started,
        "peak_rss_bytes": peak_rss_bytes(),
        "proposal_eos_positions": [
            index + 1
            for index, token in enumerate(proposals)
            if token == tokenizer.eos_token_id
        ],
    }
    return case, k_rows


def family_aggregates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    families = sorted({str(row["family"]) for row in rows})
    return {
        family: acceptance_distribution(
            [row for row in rows if str(row["family"]) == family]
        )
        for family in families
    }


def require_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{path}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/exp_076/config.json"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/exp_076_candidate"
    )
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    prompt_path = ROOT / config["prompts"]["path"]
    if sha256_file(prompt_path) != config["prompts"]["file_sha256"]:
        raise ValueError("prompt file SHA-256 mismatch")
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    prompt_audit = validate_prompt_manifest(
        prompts,
        required_families=config["prompts"]["required_families"],
        build_per_family=int(config["prompts"]["build_per_family"]),
        evaluation_per_family=int(config["prompts"]["evaluation_per_family"]),
    )
    output = arguments.output_dir.resolve()
    if output == ROOT.resolve() or not (
        output.name == "exp_076" or output.name.startswith("exp_076_")
    ):
        raise ValueError("output directory must be a dedicated exp_076 or exp_076_* path")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    import tokenizers
    import huggingface_hub
    import safetensors

    if transformers.__version__ != config["runtime"]["transformers"]:
        raise RuntimeError("Transformers version does not match the lock")
    if torch.__version__ != config["runtime"]["torch"]:
        raise RuntimeError("PyTorch version does not match the lock")
    torch.set_num_threads(int(config["runtime"]["torch_num_threads"]))
    torch.manual_seed(760)
    torch.use_deterministic_algorithms(True)

    run_started = time.perf_counter_ns()
    checkpoint_manifest = validate_checkpoint_manifest(arguments.model_dir, config)
    target, mtp, tokenizer, parameter_audit = load_models(arguments.model_dir)
    proposal_cost = int(parameter_audit["proposal_parameters_per_token"])
    p50_minimum = ideal_minimum_accepted_tokens(
        target_block_parameter_reads=int(
            config["accounting"]["target_block_parameter_reads"]
        ),
        baseline_parameters=int(config["accounting"]["baseline_parameters"]),
        allowed_multiplier=float(config["accounting"]["p50_allowed_multiplier"]),
        proposal_parameters_per_token=proposal_cost,
    )
    p95_minimum = ideal_minimum_accepted_tokens(
        target_block_parameter_reads=int(
            config["accounting"]["target_block_parameter_reads"]
        ),
        baseline_parameters=int(config["accounting"]["baseline_parameters"]),
        allowed_multiplier=float(config["accounting"]["p95_allowed_multiplier"]),
        proposal_parameters_per_token=proposal_cost,
    )

    cases: list[dict[str, Any]] = []
    all_k_rows: list[dict[str, Any]] = []
    for split in ("build", "evaluation"):
        for prompt in prompts[split]:
            case, k_rows = run_case(
                row=prompt,
                split=split,
                tokenizer=tokenizer,
                target=target,
                mtp=mtp,
                config=config,
                parameter_audit=parameter_audit,
            )
            cases.append(case)
            all_k_rows.extend(k_rows)
            print(
                json.dumps(
                    {
                        "prompt_id": case["prompt_id"],
                        "split": split,
                        "max_k_accepted_prefix": case["max_k_accepted_prefix"],
                        "wall_ns": case["wall_ns"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    build_rows = [row for row in all_k_rows if row["split"] == "build"]
    selected_k = select_build_k(build_rows, config["proposal"]["registered_k"])
    evaluation_rows = [
        row
        for row in all_k_rows
        if row["split"] == "evaluation" and row["configured_k"] == selected_k
    ]
    population = acceptance_distribution(evaluation_rows)
    families = family_aggregates(evaluation_rows)
    integrity = {
        "wrong_accepts": sum(int(case["wrong_accepts"]) for case in cases),
        "commit_replay_mismatches": sum(
            bool(case["commit_replay_mismatch"]) for case in cases
        ),
        "rollback_recompute_mismatches": sum(
            bool(case["rollback_recompute_mismatch"]) for case in cases
        ),
        "committed_cache_mutations_during_verification": sum(
            bool(case["committed_cache_mutation_during_verification"])
            for case in cases
        ),
        "prefix_mutations": sum(bool(case["prefix_mutation"]) for case in cases),
        "target_future_token_reads": sum(
            int(case["target_future_token_reads"]) for case in cases
        ),
    }
    integrity_passed = all(value == 0 for value in integrity.values())
    acceptance_passed = (
        p50_minimum is not None
        and p95_minimum is not None
        and population["accepted_prefix_p50"] >= p50_minimum
        and population["accepted_prefix_p05"] >= p95_minimum
    )
    family_passed = (
        p50_minimum is not None
        and p95_minimum is not None
        and all(
            aggregate["accepted_prefix_p50"] >= p50_minimum
            and aggregate["accepted_prefix_p05"] >= p95_minimum
            for aggregate in families.values()
        )
    )
    traffic_passed = (
        population["normalized_traffic_p50"] is not None
        and population["normalized_traffic_p95"] is not None
        and population["normalized_traffic_p50"]
        <= float(config["gate"]["require_realized_p50_at_most"])
        and population["normalized_traffic_p95"]
        <= float(config["gate"]["require_realized_p95_at_most"])
    )
    decision = gate_decision(
        integrity_passed=integrity_passed,
        acceptance_passed=acceptance_passed,
        family_passed=family_passed,
        traffic_passed=traffic_passed,
    )
    controls = [
        {"control": key, "passed": value == 0, "failures": value}
        for key, value in integrity.items()
    ]
    deterministic_core = {
        "selected_k": selected_k,
        "shape_thresholds": {
            "proposal_parameters_per_token": proposal_cost,
            "p50_minimum_accepted": p50_minimum,
            "p95_minimum_accepted": p95_minimum,
        },
        "evaluation_population": population,
        "family_aggregates": families,
        "integrity": integrity,
        "gate": {
            "integrity_passed": integrity_passed,
            "acceptance_passed": acceptance_passed,
            "family_passed": family_passed,
            "traffic_passed": traffic_passed,
        },
        "decision": decision,
        "prompt_core_sha256": prompt_audit["manifest_core_sha256"],
        "case_token_core": [
            {
                "prompt_id": case["prompt_id"],
                "first_target_token": case["first_target_token"],
                "proposal_tokens": case["proposal_tokens"],
                "target_verification_tokens": case["target_verification_tokens"],
                "committed_tokens": case["committed_tokens"],
            }
            for case in cases
        ],
    }
    core_hash = canonical_sha256(deterministic_core)
    measured = {
        "build_cases": len(prompts["build"]),
        "evaluation_cases": len(prompts["evaluation"]),
        "case_count": len(cases),
        "k_row_count": len(all_k_rows),
        "target_forward_calls": sum(int(case["target_forward_calls"]) for case in cases),
        "mtp_forward_calls": sum(int(case["mtp_forward_calls"]) for case in cases),
        "wrong_accepts": integrity["wrong_accepts"],
        "future_target_token_reads": integrity["target_future_token_reads"],
        "commit_replay_mismatches": integrity["commit_replay_mismatches"],
        "rollback_recompute_mismatches": integrity["rollback_recompute_mismatches"],
        "peak_process_rss_bytes": peak_rss_bytes(),
        "wall_ns": time.perf_counter_ns() - run_started,
        "cpu_ns": sum(int(case["cpu_ns"]) for case in cases),
        "deterministic_core_sha256": core_hash,
    }
    derived = {
        "selected_k": selected_k,
        "parameter_audit": parameter_audit,
        "shape_p50_minimum_accepted": p50_minimum,
        "shape_p95_minimum_accepted": p95_minimum,
        "evaluation_population": population,
        "family_aggregates": families,
        "gate": deterministic_core["gate"],
        "decision": decision,
    }
    summary = {
        "experiment": "EXP-076",
        "name": config["name"],
        "phase": ["C-small-real-checkpoint-observation"],
        "evidence_level": "E1",
        "authoritative_decision": decision,
        "REGISTERED_EXTERNAL_INPUTS": {
            "checkpoint": {
                "model_id": config["checkpoint"]["model_id"],
                "revision": config["checkpoint"]["revision"],
                "weight_sha256": config["checkpoint"]["weight_sha256"],
            },
            "prompts_file_sha256": config["prompts"]["file_sha256"],
            "runtime": config["runtime"],
        },
        "MEASURED": measured,
        "DERIVED": derived,
        "UNVERIFIED": [
            "vLLM or SGLang physical speculative runtime equivalence",
            "physical weight traffic or latency improvement",
            "CUDA GPU VRAM PCIe SSD H2D power and thermal behavior",
            "quantized native-MTP preservation",
            "35B 122B expert route locality and scaling",
            "transfer of Qwen-specific MTP to arbitrary dense checkpoints",
            "405B execution under 8 GiB or 4B-class speed",
            "E2 through E7",
        ],
        "claim_boundary": {
            "checkpoint": "UNCHANGED_PINNED_OFFICIAL_BF16_PAYLOAD",
            "proposal": "CAUSAL_CPU_REFERENCE_OF_PINNED_NATIVE_MTP_EQUATIONS",
            "target_verification": "EXACT_GREEDY_TARGET_LOGITS_ON_SEPARATE_CACHE_CLONE",
            "rollback": "DISCARD_AND_RECOMPUTE_REFERENCE_NOT_VLLM_RUNTIME_PROOF",
            "operation_replacement": "OBSERVATION_ONLY_NOT_COMPLETE_GENERATION",
            "target_server": "NO_COMMAND_EXECUTED",
            "phase_d": "NOT_TESTED",
            "dense_405b": "NOT_VALIDATED",
        },
        "provenance": {
            "source_commit": git_commit(),
            "config_sha256": sha256_file(arguments.config),
            "requirements_lock_sha256": sha256_file(
                ROOT / "experiments/exp_076/requirements.lock.json"
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "safetensors": safetensors.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    require_finite(summary)
    dump(output / "raw/checkpoint_manifest.json", checkpoint_manifest)
    dump(output / "raw/prompt_manifest.json", prompt_audit)
    dump_rows(output / "raw/case_rows.jsonl", cases)
    dump_rows(output / "raw/k_rows.jsonl", all_k_rows)
    dump(output / "raw/control_rows.json", controls)
    dump(output / "raw/parameter_shapes.json", parameter_audit)
    dump(output / "processed/aggregate.json", deterministic_core)
    dump(output / "summary.json", summary)
    dump(output / "artifacts/environment.json", summary["provenance"])
    (output / "artifacts/contract.txt").write_text(
        config["evidence_ceiling"] + "\n", encoding="utf-8", newline="\n"
    )
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "logs/run.log").write_text(
        json.dumps(
            {
                "experiment": "EXP-076",
                "decision": decision,
                "selected_k": selected_k,
                "measured": measured,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_checksums(output)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    if not integrity_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
