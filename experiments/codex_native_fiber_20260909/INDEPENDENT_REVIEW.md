# Independent review: native finite-fiber obstruction

## Verdict

**The local theorem is established in the declared finite-word model, after three wording corrections.** The constructed set contains `256^(n-1)` distinct BF16 inputs that all produce the same FP32 output `[2,1,...,1]`. Therefore the declared native map is noninjective despite the real matrix having full rank. Bijective input and output relabelings preserve that noninjectivity, so they cannot turn this map into an injective full-coordinate copy/permutation.

The corrections are:

1. “Full-coordinate copy/permutation” must mean an **injective** map, such as a permutation that uses every source coordinate exactly once. Noninjectivity alone does not reject copies that drop or duplicate coordinates.
2. The witness proves that one fiber has **at least** `256^(n-1)` elements. It does not prove that the entire fiber has exactly that size, because inputs outside the constructed set were not classified.
3. The `8(n-1)` statement is a cardinality lower bound for a finite, injectively coded side state. Calling it stored bits requires a fixed-length binary representation, or an equivalent worst-case distinguishability model.

This is a bounded obstruction for one native projection operator. It does not close contract O3, any other O1–O6 obligation, or the fixed mission. That matches the preregistration’s own limits (`native_fiber_audit/PREREGISTRATION.md:8-11,40-49`) and the contract’s requirement that all essential obligations close before a theory is accepted (`vortex-independent-audit/docs/CONSTRUCTIVE_THEORY_CONTRACT.md:21-32,74-86`).

## Formal fiber argument

Let `F : X -> Y` be the native finite map. Let `e : X' -> X` and `d : Y -> Y'` be bijections, and define

`G = d o F o e`.

For every `y in Y`,

`G^(-1)({d(y)}) = e^(-1)(F^(-1)({y}))`.

Because `e` is a bijection, these two fibers have the same cardinality. Thus all fiber sizes, and in particular injectivity or noninjectivity, are invariant under bijective input and output relabeling. Finiteness is not needed for this identity; it is needed for the later finite counting statement.

If `P` is a full-coordinate permutation, then `P` is injective. If `F` has two distinct inputs with the same output, every `d o F o e` also has such a pair and therefore cannot equal `P`.

This result permits arbitrary global cross-wire encodings as encodings. It only says that a bijective cross-wire encoding cannot make this particular noninjective map into an injective permutation without adding information or changing the operator.

## Witness proof

Fix `2 <= n <= 16384`, put `k = ceil(log2(n))`, and put

`delta = 2^(-(34+k))`.

Let `W = I + 11^T`, so `W` has diagonal entries `2` and off-diagonal entries `1`. The matrix determinant lemma gives

`det(W) = det(I) * (1 + 1^T 1) = n+1`,

so `W` is full rank over the reals. Its entries `1` and `2` are exactly representable in BF16.

For the input family, set `x_0=1` and `x_j=a_j*delta` for `j>0`, with each `a_j` independently chosen from `{0,...,255}`. Since `1 <= k <= 14`, `delta` ranges from `2^-35` through `2^-48`, well inside the BF16 normal range. Every nonzero integer `a_j <= 255` has at most eight binary significant bits, so `a_j*delta` is exactly representable in BF16. The map from the coefficient tuple to the input vector is injective; hence there are exactly `256^(n-1)` distinct constructed inputs.

The separately rounded products are exact in FP32. Multiplication by `1` changes nothing, and multiplication by `2` only shifts the exponent. In output row `0`, the large product is `2`. In output row `i>0`, the large product contributed by `x_0` is `1`. Every other product is a nonnegative integer multiple of `delta`.

Consider a tree node whose leaves do not include the large product. Its exact value is `B*delta` for a nonnegative integer `B`. Across the whole tiny part of row `0`,

`B <= 255(n-1)`,

and across the whole tiny part of row `i>0`, where the diagonal product contributes one extra `a_i*delta`,

`B <= 255n <= 255*16384 = 4,177,920 < 2^22`.

Every subtree has a no-larger coefficient. A scaled integer below `2^22` is exactly representable with FP32’s 24-bit precision, and all nonzero values here remain normal. Induction from the leaves therefore shows that every tiny-only subtree is computed exactly as an integer multiple of `delta`. Padded `+0` leaves do not alter this result.

For every tiny-only sibling attached along the unique large-leaf path,

`0 <= T <= 255n*delta <= 255*2^k*2^(-(34+k)) < 2^-26`.

At `1`, the midpoint to the next larger FP32 number is `1 + 2^-24`; at `2`, it is `2 + 2^-23`. Therefore round-to-nearest, ties-to-even gives

`RNE_FP32(1+T)=1` and `RNE_FP32(2+T)=2`.

After each large-path addition the node is again exactly `1` or `2`, so the same bound applies at the next level. This proves the required before-and-after tree induction rather than relying on one exact total. Consequently every constructed input yields `[2,1,...,1]` as FP32 words. An optional BF16 RNE store preserves those values because `1` and `2` are exact BF16 values.

The fiber containing `[2,1,...,1]` therefore has at least `256^(n-1)` elements, and the native map is noninjective for every declared `n`.

## Later derived binary corollary

This corollary was supplied after the frozen `I+11^T` preregistration. It is a proof-only consequence, not a replacement for the preregistered witness, controls, or experiments.

Define the binary matrix `B` by making row `0` all ones and, for each `i>0`, making row `i` all ones except for `B_ii=0`. Then

`row_0 - row_i = e_i` for every `i>0`.

Over the reals, replacing each row `i>0` by `row_i-row_0` leaves row `0` unchanged and produces `-e_i`. Expansion in column `0` gives

`det(B)=(-1)^(n-1)`.

Over `GF(2)`, the same row operations produce `e_i` and the determinant is `1`. Thus `B` has full rank both over the reals and over `GF(2)` for every declared `n`.

All entries are unchanged binary BF16 weights. Every row has the first-column anchor `B_i0*x_0=1`. On the same coefficient box, row `0` has tiny coefficient `sum_(j>0) a_j <=255(n-1)`, while row `i>0` omits `a_i` and has an even smaller coefficient. The tiny-only subtree exactness proof above applies unchanged, and every large-path sibling is below `2^-26`, hence below the positive half-ulp at `1`. Every constructed input therefore maps to the FP32 all-ones vector, with the optional BF16 store also exact.

This gives a direct binary-weight example where algebraic invertibility over `GF(2)` and the reals does not imply injectivity of the declared native BF16/FP32 operator. Consequently independent bijective word encodings—and, by the general fiber theorem, arbitrary global bijective encodings—cannot turn this native map into an injective GF(2)-style identity/copy. The statement still requires “identity/copy” to mean an injective full-coordinate map. It gives no HF activation reachability result and no impossibility result for side state, nonbijective continuation representations, rank-deficient copies, nonlinear producers, or global execution state.

## Side-state counting consequence

Let `A` be the constructed collision set, so `|A|=256^(n-1)`. Suppose an extended output `(F(x),s(x))` is injective on `A`. Since `F` is constant on `A`, `s` alone must take at least `|A|` distinct values. If `s` is stored in a fixed `b`-bit field, then

`2^b >= 256^(n-1) = 2^(8(n-1))`,

and hence `b >= 8(n-1)`.

This counts information needed by an injective reversible lift on this collision set. It is not a lower bound on native KV-cache memory, total executor memory, average variable-length storage, or a continuation representation allowed to merge states that are proven continuation-equivalent.

## Counterexamples and exact exclusions

- A rank-deficient copy can itself be noninjective. For example, `(u,v) -> (u,u)` drops `v`. The fiber theorem does not generically reject such copies; full fiber cardinalities would have to match, not merely the fact of noninjectivity.
- Adding side state defeats the local obstruction. The map `x -> (F(x),x)` is injective, at the stated information cost on the witness set.
- A nonbijective representation equipped with a proved exact continuation relation is outside the conjugacy theorem.
- A different native operator, including a nonlinear producer, is outside the witness proof.
- Global cross-wire bijections are covered by the invariant, but only their attempted conversion of this `F` into an injective full-coordinate permutation is ruled out. They are not rejected as a general design family.
- The proof gives no reachability result. It does not show that the witness inputs occur as activations in any legal trace of an unchanged Hugging Face 405B checkpoint.
- The arithmetic proof applies only to separate FP32 RNE products followed by the declared fixed padded balanced FP32 RNE addition tree and FP32 observation, with the optional final BF16 store. A CUDA/HF kernel with a different accumulation order, fusion rule, precision, ABI, or observation boundary is a different map and requires a new lifting proof.
- The collision is at one projection output. It does not by itself show equality of logits, RNG consumption, successor state, or KV state for all continuations.

The previously rejected reconstruction, free-rounding witness, row replay, full-bitset scan, response catalog, free decoder, global-advice-per-matrix split, and sparse-delta families are neither reopened nor reevaluated by this result. This is not three new execution principles and is not an overall impossibility theorem.

## Contract and cost audit

The preregistration identifies the operator, finite-word ABI, scale range, and test controls (`PREREGISTRATION.md:20-39`). It does not yet provide the explicit execution schedule and upper bounds required by O4. For an unpruned explicit reference at dimension `n`, a direct accounting starts with `n^2` products and `n(2^k-1)` tree additions. At `n=16384`, that is `268,435,456` products and `268,419,072` additions. Explicit BF16 storage for `W` alone is `2n^2 = 536,870,912` bytes (512 MiB); materializing all FP32 product leaves would be `4n^2 = 1,073,741,824` bytes (1 GiB). A streaming or implicit implementation can reduce storage, but it needs its own exact schedule, address work, scratch bound, and verification cost. The planned two-nonzero pruned evaluator also needs the preregistered proof that skipping zero-only subtrees reproduces the declared tree on that restricted domain (`PREREGISTRATION.md:34-39`). No experiment was performed in this review.

More importantly, the witness has no 405B execution algorithm to cost. The files supply no full upper bounds for CPU work, host RAM, SSD storage/read traffic, PCIe movement, HBM traffic, GPU allocation and fragmentation, compilation, address generation, encoding/decoding, initialization, verification/repair/rollback, KV/cache, synchronization, or fallback. They also provide no baseline lower bound or coupled p50/p95 ratio proof. Those omissions are material under the contract’s cost rules (`CONSTRUCTIVE_THEORY_CONTRACT.md:42-52`) and fixed target (`CONSTRUCTIVE_THEORY_CONTRACT.md:13-19`).

The fixed mission therefore remains: every public unchanged dense Hugging Face 405B-class checkpoint; batch one; exact native output/logits and RNG consumption; exact successor/KV correspondence for every legal continuation; one GPU with peak allocation at most 8 GiB; and the frozen same-machine latency/TTFT requirements, with all CPU, RAM, SSD, PCIe, HBM, GPU, compilation, addressing, decoding, initialization, verification, repair, and fallback costs charged. No target hardware exists for this review, so `HARDWARE_STATUS=NOT_TESTED`. No hidden remote computation, confidentiality change, evaluation change, checkpoint modification, or free fallback follows from the theorem.

Overall `THEORY_STATUS` remains `NOT_ESTABLISHED`: the review proves a local no-go lemma in the declared model and narrows one O3 assumption, but supplies no uniform constructor, causal full-model program, continuation proof, complete budget, target-scale latency closure, or mission-wide reproducible artifact set. The most direct next theoretical requirement is exactly the preregistration’s stated one: a native operator/side-state construction that does not assume real full rank implies native bijectivity (`PREREGISTRATION.md:44-49`).

## Evidence and review boundary

This was a static independent proof review of only:

- `native_fiber_audit/PREREGISTRATION.md`
- `vortex-independent-audit/docs/CONSTRUCTIVE_THEORY_CONTRACT.md`

No Git operation, GPU run, configuration change, experiment, or external message was used. The requested reviewer selection “Sol/high” was not independently verifiable from execution metadata available to this review, so no claim is made that this selection actually executed.
