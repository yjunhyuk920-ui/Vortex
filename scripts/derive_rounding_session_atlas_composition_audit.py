#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vortex_runtime.rounding_session_atlas_composition import derive_composition_audit


NATIVE_PATH = ROOT / "results/e0_native_exact_shortcut_frontier/summary.json"
HYPERBLOCK_PATH = ROOT / "results/exp_080a/summary.json"
EXPECTED_NATIVE_BLOB_SHA1 = "f8bbf7181ef1e71c3f7999094ad97e3082b6b2f1"
EXPECTED_HYPERBLOCK_BLOB_SHA1 = "83e278b76d4348754b728ac3e0fb908441855d37"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def load_pinned(path: Path, expected_blob_sha1: str) -> dict[str, Any]:
    actual = git_blob_sha1(path)
    if actual != expected_blob_sha1:
        raise ValueError(
            f"source Git blob mismatch for {path}: {actual} != {expected_blob_sha1}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_output(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.json"
    summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checksum = sha256_file(summary)
    (output_dir / "checksums.sha256").write_text(
        f"{checksum}  summary.json\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/e0_rounding_session_atlas_composition_audit",
    )
    args = parser.parse_args()

    native = load_pinned(NATIVE_PATH, EXPECTED_NATIVE_BLOB_SHA1)
    hyperblock = load_pinned(HYPERBLOCK_PATH, EXPECTED_HYPERBLOCK_BLOB_SHA1)
    payload = derive_composition_audit(native, hyperblock)
    payload["sources"] = {
        "native_exact_shortcut": {
            "path": NATIVE_PATH.relative_to(ROOT).as_posix(),
            "git_blob_sha1": EXPECTED_NATIVE_BLOB_SHA1,
        },
        "hyperblock": {
            "path": HYPERBLOCK_PATH.relative_to(ROOT).as_posix(),
            "git_blob_sha1": EXPECTED_HYPERBLOCK_BLOB_SHA1,
        },
    }
    write_output(args.output_dir, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
