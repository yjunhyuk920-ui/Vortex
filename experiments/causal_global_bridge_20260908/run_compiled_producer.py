from __future__ import annotations

import argparse
import json
from pathlib import Path

from basis_column_producer import compiled_producer_audit
from causal_kv_exposure import ExposureConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=224)
    parser.add_argument("--value-size", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260908)
    parser.add_argument("--tokens", type=int, nargs="+", default=[223, 0, 17, 91, 3, 44])
    args = parser.parse_args()
    payload = compiled_producer_audit(
        ExposureConfig(
            hidden_size=args.hidden_size,
            value_size=args.value_size,
            head_dim=args.head_dim,
            layers=args.layers,
            seed=args.seed,
        ),
        list(args.tokens),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
