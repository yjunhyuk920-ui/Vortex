from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vortex_runtime.quadratic_bf16_residual_fp32_context import (  # noqa: E402
    install_fp32_context_semantics,
)

# Install the correction before importing the frozen runner. fit_library() keeps
# its globals in the base runtime module, so this changes only the predicate
# semantics and leaves checkpoint, population, program language and Gates fixed.
install_fp32_context_semantics()

from experiments.exp_087a.run_experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
