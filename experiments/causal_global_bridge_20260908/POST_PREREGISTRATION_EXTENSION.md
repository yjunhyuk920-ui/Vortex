# Post-preregistration extension

The frozen `PREREGISTRATION.md` selected Principle A as a square `v_proj`
causal-exposure bridge and required continuation to Principles B/C if A did not
itself produce the mission executor.  The following additions were discovered
only after the frozen A implementation had passed its first focused replay.
They are therefore reported as extensions, not retroactively inserted into the
preregistration.

## A1. Native rectangular GQA exposure

The square construction was generalized to ordinary grouped-query attention.
For even `head_dim`, `value_size = num_key_value_heads * head_dim` and
`hidden_size = num_attention_heads * head_dim`, with the native requirement
`num_attention_heads % num_key_value_heads == 0`, the same basis-token proof
exposes an arbitrary binary rectangular `v_proj[value_size, hidden_size]` in
`DynamicCache.values`.

The pinned native control `32 x 224` uses:

```text
hidden_size             224
value_size               32
head_dim                   2
num_attention_heads      112
num_key_value_heads       16
source bits             7168
full-vocab cache checks 7168
mismatches                  0
```

This shape is useful because the earlier local nonlinear router cover gate is
already capacity-inconclusive at `32 x 224` under its favorable target probe
budget.  It still does not transfer the independently selectable hard query
tuple to one causal trace.

An odd-head exploratory `25 x 225` config was discarded after discovering that
native RoPE broadcasts a one-dimensional head into a two-coordinate rotated
key.  The executable extension therefore requires an even native RoPE head
dimension and does not use that edge configuration as scientific evidence.

## A2. Exact restricted basis-column producer

The bridge family itself admits a finite exact producer.  Its encoder packs the
binary `v_proj` source by columns into 64-bit words and stores the exact BF16
RMSNorm scale word plus a source-independent native RoPE signed-zero template.
At runtime token `j` addresses only packed column `j`, emits the exact value
cache, emits the exact signed-zero key cache, and returns the family-fixed zero
logits without calling an original dense kernel.

For two `32 x 224` GQA layers:

```text
source binary bits                              14336
compiled padded source bits                     28672
compiled auxiliary bits                         14368
runtime source payload bits/token                 128
native variable-v_proj full-read bits/token     229376
payload/full-read                         0.0005580357
variable-v_proj read fraction removed      0.9994419643
dense v_proj calls in compiled runtime                0
native logits/K/V bit equality                       PASS
compiled runtime RNG unchanged                         yes
```

This is a real finite encoder/address/decoder and a complete transition for the
declared restricted ABI.  It is **not** a mission constructor because binary
weights, basis-token hidden states, zero q/k/o/MLP paths and zero lm-head are
essential to it.  The result is retained because it demonstrates exactly why a
causal exposure lower-bound bridge must not silently assume that every exposed
dense source is expensive to query.

## A3. Standard-causal multi-right-factor compiler

To attack the remaining reachability gap without a custom attention mask, a
second ordinary Llama family uses signed basis embeddings and zero q/k:

```text
token(+,j) -> +e_j
token(-,j) -> -e_j
query marker -> 0
v_proj = I
o_proj = I
q_proj = k_proj = MLP = 0
```

The first block writes one sign token per coordinate.  Between desired query
vectors, an unchanged coordinate writes two zero markers; a flipped coordinate
writes two copies of the new sign.  Algebraically the cumulative signed-basis
sum changes from the old sign to the new sign.  A zero query marker therefore
receives the desired independently chosen right factor through ordinary causal
attention.  A binary source matrix is placed in the first rows of `lm_head`.
The final query hidden has one common BF16 magnitude and the requested signs;
the logits encode signed row sums.  Knowing each source row's Hamming weight
turns those signed sums into GF(2) row parities and hence into any requested
left-mask rank-one parity.

The construction explicitly checks finite-word behavior rather than assuming
real-arithmetic cancellation.

Executed controls:

```text
right bits  output rows  queries  seq len  common-sign failures  logit mismatch
8           4            32       536      0                     0
108         25           2        326      0                     0
216         25           4        1516     0                     0
```

For the 8-bit/32-query run every one of the `32 * 2^8 = 8192` binary lm-head
rows was also checked through the native BF16 linear operator:

```text
same-parity BF16 codebook collisions   0
row parity decode mismatches            0
all binary row decode mismatches         0 / 8192
random left-mask scalar mismatches       0
full-vs-incremental query logits         0 mismatches
RNG                                      unchanged
```

The `25 x 216` variable lm-head block has exactly 5400 source bits, matching one
of the first area-5400 shapes from the local nonlinear-router frontier.  Four
independent right factors were executed there with zero observed mismatch and
zero same-parity BF16 codebook collision.

The full 32-query area-5400 native trace was **not executed**.  Therefore the
repository still does not state that the earlier area-5400 32-query physical
union theorem has been fully transferred to one pinned native Transformer
trace.  What is new is an explicit finite standard-causal query-sequence
compiler, native 32-query evidence at small width, and native area-5400 evidence
at four queries.  The remaining lift is sharply localized to long-trace native
finite-word preservation plus the global-advice model.

## Principle B continuation: global nonlinear producer

No arbitrary-checkpoint global encoder/address/decoder was constructed.

Three concrete checks prevent promoting a placeholder:

1. The exact restricted basis-column producer above works only because the
   causal right factor is a coordinate and the source alphabet is binary.
   General right factors require a genuine matrix-vector data structure.
2. Bit-packing plus popcount is exact for binary sign queries, but an arbitrary
   Q4/BF16 matrix requires all native bit planes.  It therefore returns to a
   full source read and is excluded by the fixed mission.
3. The best relevant succinct Boolean MatVec literature does not provide the
   missing primitive.  Chakraborty--Kamma--Larsen (STOC 2018,
   arXiv:1711.04467) gives a randomized Boolean-semiring upper bound with
   per-query error and explicitly contrasts it with a slower deterministic
   Boolean-semiring predecessor.  Neither is a zero-error native finite-word
   Q4/BF16/FP32 producer.  Young Kun Ko's 2025 static nonlinear-preprocessing
   lower bound and 2026 dynamic Boolean lower bound improve lower-bound
   technology but do not supply an upper construction.

The global-advice issue also remains real: an 8 GiB representation may mix
information across matrices.  The earlier proportional per-block allocation
cannot simply be promoted to a whole-checkpoint direct sum.

## Principle C continuation: paid dynamic exact summary

The new causal compiler makes one important dynamic-summary escape route
decisively narrower.  If a summary stores an exact current product `y = W s`,
then changing right factor from `s` to `s'` gives

```text
y' = y + W (s' - s).
```

In the sign encoding, every changed bit contributes a signed multiple of one
full source column.  The legal compiler permits two consecutive desired query
vectors with Hamming distance `n`, so literal column-delta maintenance touches
all `n` columns and performs a full dense effect.  This is the already-closed
response-column transport family, now with an explicit legal causal adversary.

Any successful dynamic summary must therefore aggregate an arbitrary dense set
of changed columns subdensely.  That missing operation is again an exact static
matrix-vector producer of Principle B, not a free temporal update rule.

No finite arbitrary-checkpoint `Z_{t+1}=Update(Z_t,input_t)` law satisfying the
mission costs was found in this round.

## Post-persistence Principle-B lift gate

After the first causal/global result had been committed and remotely verified,
the deterministic Larsen--Williams succinct Boolean-semiring MatVec structure
was revisited only at the unresolved algebraic lift boundary. The exact result
is in `BOOLEAN_MATVEC_LIFT_RESULT.md`.

Two routes are now closed:

1. A deterministic adaptive black-box conversion from complete Boolean `Mv`
   answers to `F2` `Mv` needs at least `|supp(v)|` Boolean products on an exact
   deletion-witness family. Full support needs `n` calls.
2. Even arbitrary nonlinear row/query feature maps that try to make **one**
   Boolean product equal GF(2) inner product require exactly `2^d-1` Boolean
   features, by an exact Boolean-rank/rectangle proof.

These results strengthen F-052 without changing the claim boundary: direct GF(2)
data structures, direct nonlinear access to globally mixed checkpoint advice and
native finite-word producers remain open. No Boolean OR witness is promoted as
a numerical source.
