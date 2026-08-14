"""Print the deterministic E0 post-Atlas source accounting."""
from __future__ import annotations

import json

from vortex_runtime.post_atlas_causal_source import (
    derive_post_atlas_resource_audit,
)


def main() -> None:
    print(json.dumps(derive_post_atlas_resource_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
