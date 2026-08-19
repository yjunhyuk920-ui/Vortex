# Exact whole-SwiGLU global compiler — current handoff

The active branch implements and preregisters EXP-085A. It compiles the
complete function `Wd[silu(Wg x) * (Wu x)]` into an exact functional DAG,
measures strict and deliberately favorable algebraic operation counts over all
SwiGLU layers of the pinned SmolLM2-135M checkpoint, and probes a joint gate/up
lowering against the frozen native BF16 ABI.

No full executor is authorized unless the favorable whole-function structural
fraction reaches the fixed `1.185185185%` target. A negative result is the
intended cheap-kill outcome and must be committed rather than rescued with a
backend, tile, rank, prompt, or tolerance sweep.
