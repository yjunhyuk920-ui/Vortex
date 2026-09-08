from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vortex_runtime.direct_global_producer_gate import derive_direct_global_producer_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be absent or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = derive_direct_global_producer_audit()
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    (args.output_dir / "checksums.sha256").write_text(
        f"{digest}  summary.json\n", encoding="utf-8"
    )
    print(summary)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
