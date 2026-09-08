from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vortex_runtime.boolean_oracle_parity_lift_gate import audit_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = args.output_dir / "summary.json"
    summary.write_text(
        json.dumps(audit_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    (args.output_dir / "checksums.sha256").write_text(
        f"{digest}  summary.json\n", encoding="utf-8"
    )
    print(summary)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
