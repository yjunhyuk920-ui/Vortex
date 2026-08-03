from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sanitize(value: Any) -> Any:
    """Convert non-finite experiment values to JSON null without hiding failure.

    Recurrence divergence is experimental evidence, not an infrastructure error.
    The underlying computation is left untouched; only serialization changes.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


_original_dumps = json.dumps


def _safe_dumps(value: Any, *args: Any, **kwargs: Any) -> str:
    kwargs["allow_nan"] = False
    return _original_dumps(_sanitize(value), *args, **kwargs)


json.dumps = _safe_dumps  # type: ignore[assignment]

from scripts import run_prompt_hankel_decision_program as runner  # noqa: E402


if __name__ == "__main__":
    runner.main()
