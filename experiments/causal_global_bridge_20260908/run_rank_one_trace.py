from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_rank_one_trace import RankOneTraceConfig, trace_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--right-bits", type=int, default=8)
    parser.add_argument("--output-rows", type=int, default=4)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260908)
    args = parser.parse_args()
    payload = trace_audit(
        RankOneTraceConfig(
            right_bits=args.right_bits,
            output_rows=args.output_rows,
            queries=args.queries,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
