from __future__ import annotations

import json

from vortex_runtime.causal_residual_atlas_gate import derive_first_decode_gate


def main() -> None:
    print(json.dumps(derive_first_decode_gate().to_dict(), indent=2))


if __name__ == "__main__":
    main()
