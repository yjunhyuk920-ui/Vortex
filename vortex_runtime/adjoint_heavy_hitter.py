"""Branch-local compatibility exports for nonlinear heavy-hitter experiments.

The nonlinear experiment was originally prototyped on top of a sibling branch.
Keeping this tiny compatibility module makes old experiment imports reproducible
while the implementation itself lives in ``nonlinear_heavy_hitter``.
"""

from vortex_runtime.nonlinear_heavy_hitter import (
    replace_llama_mlp_with_count_allocation,
    uniform_neuron_allocation,
)

__all__ = [
    "replace_llama_mlp_with_count_allocation",
    "uniform_neuron_allocation",
]
