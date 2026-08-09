#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vortex_runtime.causal_residual_atlas_oracle import (
    canonical_sha256,
    deterministic_core,
    summarize_gate,
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(
                f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
            )
    (output / "checksums.sha256").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify_checksums(output: Path) -> None:
    checksum_path = output / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError("missing checksums.sha256")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        if relative in expected:
            raise ValueError(f"duplicate checksum row: {relative}")
        expected[relative] = digest
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if set(expected) != actual_files:
        raise ValueError(
            f"checksum file-set mismatch: expected={set(expected)} actual={actual_files}"
        )
    for relative, digest in expected.items():
        actual = sha256_file(output / relative)
        if actual != digest:
            raise ValueError(
                f"checksum mismatch for {relative}: {actual} != {digest}"
            )


def verify_page_choices(
    page_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in page_rows:
        key = (str(row["prompt_id"]), str(row["projection"]))
        groups.setdefault(key, []).append(row)
    if len(groups) != len(branch_rows):
        raise ValueError("page/branch group count mismatch")
    for branch in branch_rows:
        key = (str(branch["prompt_id"]), str(branch["projection"]))
        rows = sorted(groups.get(key, []), key=lambda row: row["page_index"])
        if [row["page_index"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-contiguous page indices for {key}")
        if len(rows) != int(branch["page_count"]):
            raise ValueError(f"page count mismatch for {key}")
        matching = [row for row in rows if row["top1_match"]]
        selected = min(
            matching or rows,
            key=lambda row: (
                float(row["target_to_candidate_kl"]),
                int(row["page_index"]),
            ),
        )
        if int(branch["selected_page_index"]) != int(selected["page_index"]):
            raise ValueError(f"selected page mismatch for {key}")
        if bool(branch["top1_match"]) != bool(selected["top1_match"]):
            raise ValueError(f"selected top1 verdict mismatch for {key}")
        if float(branch["selected_kl"]) != float(
            selected["target_to_candidate_kl"]
        ):
            raise ValueError(f"selected KL mismatch for {key}")


def verify_token_rows(
    branch_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in branch_rows:
        groups.setdefault(str(row["prompt_id"]), []).append(row)
    for token in token_rows:
        prompt_id = str(token["prompt_id"])
        rows = groups.get(prompt_id, [])
        projections = [str(row["projection"]) for row in rows]
        expected_success = len(rows) == 2 and all(
            bool(row["top1_match"]) for row in rows
        )
        if list(token["evaluated_projections"]) != projections:
            raise ValueError(f"token projection ordering mismatch for {prompt_id}")
        if bool(token["success"]) != expected_success:
            raise ValueError(f"token success mismatch for {prompt_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-report", action="store_true")
    arguments = parser.parse_args()
    output = Path(arguments.output_dir).resolve()
    verify_checksums(output)

    summary = load(output / "summary.json")
    config = json.loads(
        (output / "artifacts/contract.txt").read_text(encoding="utf-8")
    )
    basis_rows = load_rows(output / "raw/basis_rows.jsonl")
    page_rows = load_rows(output / "raw/page_rows.jsonl")
    branch_rows = load_rows(output / "raw/branch_rows.jsonl")
    token_rows = load_rows(output / "raw/token_rows.jsonl")
    control_rows = load(output / "raw/control_rows.json")
    stored_gate = load(output / "processed/gate.json")
    verify_page_choices(page_rows, branch_rows)
    verify_token_rows(branch_rows, token_rows)

    observed_control_failures = sum(
        not bool(row.get("pass")) for row in control_rows
    )
    if observed_control_failures != len(stored_gate["control_failures"]):
        raise ValueError("control failure count mismatch")
    recomputed_gate = summarize_gate(
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
        control_failures=tuple(stored_gate["control_failures"]),
        leakage_failures=tuple(stored_gate["leakage_failures"]),
        malformed_state_count=int(stored_gate["malformed_state_count"]),
        execution_complete=bool(stored_gate["execution_complete"]),
    )
    if recomputed_gate != stored_gate:
        raise ValueError("recomputed Gate differs from stored Gate")
    if summary["authoritative_decision"] != stored_gate["decision"]:
        raise ValueError("summary decision differs from processed Gate")

    core = deterministic_core(
        registered_inputs=summary["REGISTERED_EXTERNAL_INPUTS"],
        basis_rows=basis_rows,
        page_rows=page_rows,
        branch_rows=branch_rows,
        token_rows=token_rows,
        control_rows=control_rows,
        gate=stored_gate,
    )
    stored_core = load(output / "processed/deterministic_core.json")
    if core != stored_core:
        raise ValueError("rebuilt deterministic core differs from stored core")
    core_hash = canonical_sha256(core)
    if core_hash != summary["MEASURED"]["deterministic_core_sha256"]:
        raise ValueError("deterministic core SHA mismatch")
    if len(page_rows) != summary["MEASURED"][
        "oracle_page_candidates_evaluated"
    ]:
        raise ValueError("page row count mismatch")
    if len(branch_rows) != summary["MEASURED"][
        "evaluated_projection_branches"
    ]:
        raise ValueError("branch row count mismatch")
    if len(token_rows) != summary["MEASURED"]["evaluated_token_states"]:
        raise ValueError("token row count mismatch")

    report = {
        "experiment": "EXP-083A",
        "verification": "PASS",
        "authoritative_decision": stored_gate["decision"],
        "deterministic_core_sha256": core_hash,
        "checks": [
            "bundle file checksums",
            "top1-first minimum-KL page selection",
            "token conjunction",
            "control failure count",
            "Gate aggregation and decision",
            "deterministic core payload and SHA-256",
            "summary population counts",
        ],
    }
    if arguments.write_report:
        dump(output / "artifacts/independent_verification.json", report)
        write_checksums(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
