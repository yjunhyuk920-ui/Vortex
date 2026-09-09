# Two-query native residual lift — auxiliary result, no core admission

An explicit finite decoder recovers exact binary counts/parity from TWO native
BF16 projection queries. The first may round away low bits; the second uses
paid row-private bias input bits computed from the first observed output.
The coefficient matrix stays fixed. [PROOF.md](PROOF.md) proves the declared
FP32-intermediate/BF16-store lemma for arbitrary B,v with n<=16384 and admitted
scales. Independent review agrees within that ABI; the primary checked its
integer bounds and the direction of the reduction.

This does not produce native effects cheaply. It spends two full augmented
matvecs, and its structured embedding of the registered405B shapes has only
1.0876666104GiB of variable binary source, which the global8GiB advice could hold.
Therefore this is neither a fast core nor a target-excluding lower bound.
The three materially new universal principles remain unconstructed this round.

## Actual evidence and failures

Ordered FP32-tree implementation:106 finite controls, including n8192/n16384,
independent scales and the single-query collision256 versus257. Both initially
store BF16 word4380; the second paid query recovers their different counts/parity.
Canonical implementation results.json SHA256 476e04d9db6ed182feab6c4e7bb00de76a12b010765eea77fd9cc18a3b5c9bcf.

Primary integer audit enumerates2,097,280 first count/significand pairs and16,768
residual/significand pairs. Maximum observed first residual63; the proof bound64
is universal in the declared scope and does not follow merely from the sample.
Nine invalid-domain cases are refused, and negative zero decodes as zero.

Actual CPU torch2.8.0+cpu operator audit:24 calls in12 linear/mv cases at n257 and
n16384. Both output words match the declared arithmetic formula, both calls'
fixed coefficient hash remains unchanged, and the CPU RNG state stays identical.
torch_cpu_results.json SHA256 fa922f93f258a1595a2226a63f2d6c19f32a826723c88f0839886c654dffcd7e.
This is not a full HF model, sampled generation, KV, legal-hidden-state, CUDA,
405B,8GiB or latency experiment. The larger native-model expansion was stopped
after the scale audit; only this small operator ABI check ran.

Interrupted Terra-requested work left incomplete source and no result. Its
agent handle disappeared, and a targeted process query found no matching Python
run before primary resumed. Primary preserved partial_source_v1.py and actually
reproduced wrong bias width, uint8 oracle wrap and out-of-range alpha acceptance.
ACTUAL_RED.json preserves failure fields; traceback paths are redacted and the
raw local record is hash-bound. Primary corrected all three and executed the
finished program. No interrupted agent result was counted as a pass.

The first Sol-requested review also produced no file before its handle vanished.
A resumed Sol/high request completed INDEPENDENT_REVIEW.md. Requested settings
are recorded; actual model metadata was not exposed. No recursive delegation,
GPU or agent Git. The primary owns proof, cost and final scope judgment.

## Costs and remaining obligations

[ACCOUNTING.json](ACCOUNTING.json) charges compilation, source and validation
reads, augmented storage, both projections, bias/ratio work, state, verification
and recovery/fallback limitations. The implementation's m*n source-copy counter
is not a total traffic counter: two domain validations add4mn binary source
reads, and Boolean temporaries/verification also cost work. Full physical CPU
allocator peak and a hardware latency upper bound remain unclosed.

O1-O5 OPEN/O6 PARTIAL; THEORY_STATUS=NOT_ESTABLISHED,
HARDWARE_STATUS=NOT_TESTED, CORE_ADMISSION=false. User has no405B hardware.
No universal constructor/native causal state/cost obligation was closed.

Do not extend n, add more scalar seeds, or build a larger HF fixture as primary
work. The missing object is still the finite arbitrary-checkpoint global
compiler/address/read/native decoder with a target-feasible complete cost.
This reduction pays for that missing native decoder twice; it cannot replace it.
