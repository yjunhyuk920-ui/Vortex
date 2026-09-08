# Causal/global producer frontier report — 2026-09-08

## Decision

```text
ESTABLISH_STANDARD_HF_CAUSAL_DENSE_INFORMATION_EXPOSURE
ESTABLISH_RESTRICTED_EXACT_NO_DENSE_BASIS_COLUMN_PRODUCER
ESTABLISH_BOUNDED_STANDARD_CAUSAL_MULTI_RIGHT_FACTOR_COMPILER
KEEP_AREA5400_NATIVE_32_QUERY_TRANSFER_OPEN
KEEP_GLOBAL_NONLINEAR_ARBITRARY_CHECKPOINT_PRODUCER_OPEN
REJECT_LITERAL_DYNAMIC_COLUMN_DELTA_SUMMARY
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
```

This round does not establish the VORTEX mission.  It does solve the strongest
missing construction inside the selected causal bridge and then continues to
Principles B/C instead of stopping at a lower-bound artifact.

## 1. Frozen principles

The preregistration compared three materially different principles:

1. **A — legal-causal dense-effect exposure compiler.**  Construct an ordinary
   HF Llama checkpoint/prefix whose required successor state exposes arbitrary
   checkpoint matrix information.
2. **B — global nonlinear cross-matrix producer.**  Compile the whole checkpoint
   into globally mixed words and provide a finite exact address/decoder.
3. **C — paid dynamic exact causal summary.**  Carry an exact sufficient state
   whose update/query law is subdense for every legal future continuation.

A was selected first because it had an explicit finite constructor rather than
an assumed global decoder.

## 2. Principle A1 — exact native KV exposure

The implementation uses the real `transformers.LlamaForCausalLM`, BF16, CPU,
`batch=1`, legal `input_ids`, `use_cache=True` and the returned
`DynamicCache`.  No custom fake Transformer or future state is supplied.

For each declared binary source matrix `W_l`, basis token `j` remains a basis
direction at the input of layer `l`.  `v_proj` contains `W_l`, q/k/o and the MLP
are zero, and residual state is unchanged.  The native value cache therefore
obeys, in the tested finite-word path,

```text
flatten(Vcache[l, position, :, :])[i] != 0
    iff W_l[i, token_id[position]] == 1.
```

The checker also records the exact nonzero BF16 word and compares it to the
native input-RMSNorm basis word, so this is stronger than a numerical
zero/nonzero comparison.

### Square registered control

```text
hidden                 32
layers                  3
source/cache coords   3072
mismatches               0
full/incremental cache   bit equal
checkpoint               unchanged
RNG                      unchanged
```

### Native GQA extension

```text
v_proj shape          32 x 224
head_dim                     2
Q heads                    112
KV heads                    16
source/cache coords       7168
mismatches                   0
full/incremental cache       bit equal
checkpoint                   unchanged
```

The GQA extension is post-preregistration and is labeled as such.

## 3. Principle A2 — a real restricted producer

The exposure family itself is not merely a lower-bound witness.  A finite
producer was constructed:

```text
Compile(W_l):
    pack every binary v_proj column into 64-bit words
    store each layer's exact BF16 RMSNorm basis word
    store a source-independent native RoPE zero-key template

Query(token=j, cache):
    read packed column j from every layer
    decode 1 bits to the exact stored BF16 word
    append exact values to DynamicCache
    append exact native signed-zero keys from the RoPE template
    return exact zero lm-head logits
```

The first implementation failed exact cache equality because native RoPE turns
mathematical zero q/k into position-dependent **signed zero words**.  That
failure was retained as a debugging fact; adding the paid source-independent
RoPE template closed the exact bit contract.

For `2 x (32 x 224)` GQA sources:

```text
native public logits/K/V bit exact               yes
compiled runtime RNG unchanged                    yes
dense v_proj calls                                  0
runtime source payload                         128 bits/token
native variable-v_proj full-read             229376 bits/token
payload/full-read                          0.0005580357142857143
variable-v_proj read removed               99.94419642857143%
```

This satisfies a >=90% local route only for the declared binary/basis family.
It is not evidence that arbitrary native finite-word weights have such a lossless
representation.

## 4. Principle A3 — causal multi-right-factor compiler

The most important scope advance is that causal states need not remain confined
to a single basis-column query.

Use signed basis input tokens, zero q/k, identity v/o and zero MLP.  A zero query
marker attends uniformly to its prefix.  The first desired right factor writes
one signed token per coordinate.  To change a coordinate on the next desired
factor, write two copies of its new sign; if unchanged, write two zero markers.
In exact algebra this replaces the old cumulative sign by the new sign.

An arbitrary binary matrix is placed in the first lm-head rows.  At query
markers, native final hidden coordinates were checked to have the requested
sign and one common BF16 magnitude.  The lm-head outputs signed row sums.  A
row with Hamming weight `h` and signed sum `z` has

```text
count(W_row AND u) = (z + h) / 2,
```

so its GF(2) parity is exact once `z` is decoded.  The decoder uses the BF16
output word plus known `h mod 2`; same-parity codebook collisions are fatal and
are counted explicitly.

### 32-query bounded native result

```text
right bits                                      8
output rows                                     4
independently chosen right factors             32
legal sequence length                         536
query sign/common-magnitude failures             0
full-vs-incremental query-logit mismatches        0
same-parity BF16 codebook collisions              0
actual source-row parity mismatches                0
random left-mask scalar mismatches                 0
all binary lm-head rows checked                  8192
all-row decode mismatches                           0
RNG unchanged                                      yes
```

### Wider controls

```text
25 x 108 source, 2 right factors:
  sign/common-magnitude failures    0
  native full/incremental mismatch  0
  row/left-mask decode mismatch     0

25 x 216 source, 4 right factors (area = 5400):
  sequence length                 1516
  sign/common-magnitude failures     0
  native full/incremental mismatch   0
  same-parity codebook collisions    0
  row/left-mask decode mismatch       0
```

The symbolic sequence constructor accepts any finite number of right factors.
Nevertheless, the area-5400 **32-query native trace was not executed**, and the
repository does not promote small-width 32-query evidence plus four-query
area-5400 evidence into a native theorem by interpolation.

The global-advice issue is independent: even a successful causal reachability
lift does not justify assigning only a proportional fraction of 8 GiB to one
source block when globally nonlinear cells may mix matrices.

## 5. Principle B — global nonlinear producer remains open

No object of the required form was obtained:

```text
E(checkpoint_all) -> G
A(current_input,state,prior_words) -> finite addresses
D(returned_words, input, state) -> exact native logits/KV/RNG successor
```

The round explicitly checked and rejected promotion of the following as B:

- the new column producer, because basis-column addressing is a restricted
  query family and binary source alphabet;
- full binary bitset/popcount scanning, because arbitrary right factors still
  read the complete binary matrix, and lifting to Q4/BF16 requires all native
  source planes;
- succinct Boolean MatVec data structures, because the best fast upper bound
  is randomized/errorful and Boolean-semiring, while the deterministic known
  route is also Boolean-semiring and does not supply native ordered arithmetic.

Current relevant literature checked in this round:

- D. Chakraborty, L. Kamma, K. G. Larsen, *Tight Cell Probe Bounds for Succinct
  Boolean Matrix-Vector Multiplication*, STOC 2018, arXiv:1711.04467.  Theorem
  1.1 is randomized and succeeds with probability at least `1-1/n` for Boolean
  semiring MatVec; the paper explicitly contrasts it with a slower deterministic
  Boolean predecessor.
- Young Kun Ko, *Lower Bounds for Linear Operators*, ECCC TR25-155 / arXiv
  2509.02730.  This strengthens static nonlinear-preprocessing lower bounds; it
  is not an upper construction.
- Young Kun Ko, *An Omega((log n/log log n)^2) Cell-Probe Lower Bound for
  Dynamic Boolean Data Structures*, ECCC TR26-047 (2026).  It strengthens the
  dynamic Boolean lower-bound side and likewise does not instantiate the native
  producer.

Therefore B remains a genuine algorithmic hole, not a named primitive.

## 6. Principle C — literal dynamic column update is closed

The new legal causal trace can choose successive right factors with Hamming
distance `n`.  A summary that stores the current exact dense result `y=W s`
would update by

```text
y' = y + W(s' - s).
```

For the sign representation, each changed coordinate contributes one full
source column.  At Hamming distance `n`, the literal update reads/combines all
columns and is exactly the response-column/time-axis family already rejected by
the repository.  No renaming changes that cost.

Consequently a successful C must contain a new subdense aggregate for an
arbitrary dense set of changed columns.  That aggregate is itself a static
matrix-vector producer of B.  No finite arbitrary-checkpoint update law avoiding
this reduction was constructed here.

## 7. Costs and scope

The native exposure/trace experiments are CPU E1 controls.  They do not measure
or establish:

- a public 405B checkpoint;
- CUDA;
- one GPU <=8 GiB;
- PCIe/SSD/HBM placement;
- same-machine native 4B Q4 p50/p95;
- TTFT;
- arbitrary HF masks/backends;
- arbitrary BF16/Q4/FP32 matrices;
- global nonlinear advice locality;
- whole-model constructor/storage/latency closure.

The restricted producer pays its packed source, RoPE template, value decoding
and materialized KV bits.  Its strong traffic number is intentionally scoped to
the variable binary `v_proj` source and is not a whole-model latency claim.

## 8. O1--O6

```text
O1 OPEN     no arbitrary-public-checkpoint finite native producer constructor
O2 OPEN     no complete arbitrary-checkpoint loading/prefill/decode program
O3 OPEN     exact native transition only for declared restricted E1 families
O4 OPEN     no sufficient whole-model paid upper bound
O5 OPEN     target hardware/405B/native4BQ4/TTFT not executed
O6 PARTIAL  code, raw E1 results and focused tests are independently replayable
```

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, and
`CORE_ADMISSION=false` remain unchanged.

## 9. Validation

Focused suites at the final round state were executed as:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
& '.\experiments\native_global_transition_20260908\.venv\Scripts\python.exe' `
  -m unittest `
  experiments.causal_global_bridge_20260908.test_causal_kv_exposure `
  experiments.causal_global_bridge_20260908.test_basis_column_producer `
  experiments.causal_global_bridge_20260908.test_causal_rank_one_trace -v
```

Because the experiment directory is not a Python package, the actual replay was
run from that directory with the three module basenames. Observed result:

```text
14/14 PASS
```

The related nonlinear-router/degree/joint-geometry regression passed `28/28`.
`scripts/run_validation.py` exited `0`. Six canonical result checksums matched.

A whole tests-directory attempt did not pass: unittest discovery reached 220
tests but 15 modules could not import `pytest` in the native-global venv; a
second pytest collection using the system pytest was blocked by pre-existing
duplicate module basenames, missing SciPy and Windows-incompatible `resource`/
`libm` dependencies. No whole-suite PASS is claimed. See `VALIDATION.md`.

No old 3510-file native-global replay was rerun because those artifacts were not
modified.

## 10. Next exact frontier

The next constructive object is no longer “some causal shortcut.”  It is:

```text
an exact zero-error finite-word matrix-vector data structure / producer
for arbitrary checkpoint matrices,
with globally paid <=8 GiB advice,
whose runtime source-dependent traffic and arithmetic close the registered
native-4B-Q4 line,
and whose decoder preserves native ordered finite-word results.
```

If pursued dynamically, it must also survive arbitrary dense changes between
successive legal causal right factors; otherwise the new sign-delta compiler
reduces it back to full response-column transport.
