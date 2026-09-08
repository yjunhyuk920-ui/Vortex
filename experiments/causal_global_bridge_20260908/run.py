from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_kv_exposure import ExposureConfig, write_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--value-size", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260908)
    args = parser.parse_args()
    payload = write_audit(
        args.output_dir,
        ExposureConfig(
            hidden_size=args.hidden_size,
            value_size=args.value_size,
            head_dim=args.head_dim,
            layers=args.layers,
            seed=args.seed,
        ),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
