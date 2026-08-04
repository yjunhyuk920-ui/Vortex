# EXP-064 — Real-Q4 Exact Output-Row Prototype Gate

This Gate measures the row-space dual of EXP-057 on the unchanged pinned TinyStories-1M/3M/8M checkpoints.

Each Q4 integer output row is tested as:

- an exact identical-row group;
- an exact sign-canonical row group;
- an exact prototype plus sparse integer delta.

Only the integer dot result may be shared. Every output retains its own row scale and bias operation. Prototype weights, residual values and column indexes, mappings, accumulator copies/signs, activation reads and static bytes are charged. A structured plan is deployable only when both logical operations and logical query bytes improve over dense execution.

The experiment reuses the frozen EXP-057 Q4 integer checksums. It observes weight structure only: Q4 model-output preservation, physical kernels, actual Transformer operation replacement, 405B execution, 8 GiB VRAM and target hardware are not tested.

```bash
bash experiments/exp_064/reproduce.sh
```
