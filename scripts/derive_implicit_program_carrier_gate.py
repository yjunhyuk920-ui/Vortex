from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vortex_runtime.implicit_program_carrier_gate import build_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(build_audit(), indent=2, sort_keys=True) + "\n").encode()
    summary = output / "summary.json"
    summary.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (output / "checksums.sha256").write_text(f"{digest}  summary.json\n", encoding="utf-8")
    print(summary)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
