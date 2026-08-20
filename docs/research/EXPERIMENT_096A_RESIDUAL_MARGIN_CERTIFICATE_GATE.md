# EXP-096A — Online Residual Direct-Margin Certificate Gate

## 1. Question

EXP-095A showed a large but incomplete signal: projecting the exact deep residual into the complete 128-row online residual span reduced untouched-holdout true-token rank to median `6` and top-16 coverage to about `64.3%`, yet p95 remained above `1,000`.

That Gate optimized full-logit L2 error, which is not the objective required by greedy decoding. EXP-096A asks the cheaper and stronger token-decision question:

> Given the same online residual bank, does there exist one finite FP64 coefficient vector per position that makes the exact target token provably larger than every competing token, after charging FP64 roundoff?

This is a target-seeing capacity oracle. It does not claim a causal coefficient generator.

## 2. Frozen source

The public-checkpoint and online source are inherited without tuning:

```text
DEV-W       HuggingFaceTB/SmolLM2-135M
revision    93efa2f097d58c2a74874c7e644dbc9b0cee75a2
reference   transformers==4.46.3 LlamaForCausalLM, eager BF16 CPU
block       K=128
seed        prompt_suffix_cycle_16
fine        one complete official guessed-context sweep
coarse      official first decoder layer + official final RMSNorm
head        rowwise-symmetric INT4, BF16 scales
population  build English/Korean/code; untouched holdout math/JSON/repetition
```

For guessed block `g`:

```text
B_j = float64(F_j(g)) - float64(G_j(g)),  j=0..127.
```

`B` is complete and hashed before the delayed official target continuation begins.

## 3. Direct token-margin program

At delayed true position `i`, let `c` be the official shallow/Q4 coarse logits and `y` the official exact target token. A coefficient word vector

```text
a in FP64^128
```

produces

```text
s(v) = c(v) + sum_j a_j B_j(v).
```

The exact target decision requires

```text
s(y) > s(v) for every v != y.
```

The Gate strengthens this to a unique represented margin

```text
s(y) - s(v) >= 2^-20.
```

The compiler minimizes the L1 norm of `a`. Write `a=p-n`, `p,n>=0`. For every competitor `v`:

```text
-[B(:,y)-B(:,v)]^T p
+[B(:,y)-B(:,v)]^T n
<= c(y)-c(v)-2^-20.
```

The objective is

```text
min sum_j (p_j+n_j).
```

No coefficient magnitude bound is imposed. The positive L1 objective keeps every optimum finite when the problem is feasible. Coefficients are charged as 64-bit words.

## 4. Active-set HiGHS solve

The frozen solver is SciPy HiGHS dual simplex with primal and dual feasibility tolerances `1e-9`.

The first active set is the union of the top 64 competitors under:

1. coarse logits;
2. position-aligned residual scores;
3. the EXP-095A full-row-span L2 projection.

After each solve, every vocabulary row is scanned. The 64 smallest certified lower margins not already active are added. At most 24 active-set iterations run. If unresolved, the full vocabulary constraint matrix is solved once.

- Infeasibility of an active subset proves infeasibility of the complete frozen LP under the same represented inputs.
- A solver error, iteration-limit result, missing full fallback, or uncertified feasible result is an integrity failure, not a scientific rejection.

## 5. FP64 finite-word certificate

The LP solution itself is not trusted as an exact-real proof. The deployed object is the returned finite FP64 coefficient word vector.

For each score word:

```text
fl(s(v)) = fl(c(v) + sum_j a_j B_j(v)).
```

Let

```text
A(v) = |c(v)| + sum_j |a_j B_j(v)|
u = 2^-53
n = 2K + 4
gamma_n = n*u / (1-n*u).
```

The score-word error enclosure is

```text
|fl(s(v))-s_exact_fp64words(v)| <= gamma_n A(v).
```

For competitor `v`, the certified margin lower bound is

```text
L(v) = fl(s(y))-fl(s(v))
       - gamma_n[A(y)+A(v)].
```

A token certificate passes only when

```text
min_{v != y} L(v) > 0
```

and the explicit stable argmax is `y`. Thus every accepted certificate excludes the complete vocabulary under the declared finite-word arithmetic model.

## 6. Why this is new

- EXP-093A ranked the target inside immutable guessed-sweep rows. EXP-096A synthesizes a new score vector from the complete current-block residual bank.
- EXP-094A used one position-aligned residual. EXP-096A can combine all 128.
- EXP-095A minimized L2 residual error. EXP-096A directly solves the all-competitor token-margin inequalities.
- EXP-092A tested a small linear extension of internal activation inputs. EXP-096A operates after the full deep sweep at the accepted token-decision boundary and does not require exact hidden reconstruction.
- This is not a causal runtime: target token `y` and delayed target coarse state are oracle inputs to coefficient synthesis.

## 7. Frozen controls

The Gate is invalid unless all controls pass:

1. pinned checkpoint/runtime identities;
2. six unique prompt hashes;
3. online bank complete before delayed target;
4. exactly 128 fine and 128 incremental target positions per case;
5. first position has a certificate because guessed and true first inputs coincide;
6. stable lowest-ID tie rules are independently checked;
7. a synthetic separable positive LP is certified;
8. a synthetic identical-column negative LP is certified infeasible;
9. an FP64 one-bit coefficient perturbation changes the coefficient digest;
10. every feasible LP passes the complete-vocabulary interval scan;
11. every infeasible result is produced by HiGHS status `2` on an explicit active or full constraint set;
12. no NaN, infinity, missing coefficient row, unresolved solver status, or target-order violation occurs.

## 8. Resource equation

The EXP-095A favorable target hot ledger is inherited. EXP-096A adds one coefficient block:

```text
K positions * K coefficients * 8 bytes
= 128 * 128 * 8
= 131,072 bytes.
```

Runtime score construction, if a causal coefficient generator were later found, requires

```text
K * V = 128 * 128,256 = 16,416,768
```

residual MACs per token plus a complete vocabulary argmax and margin scan. This is roughly `0.0040535%` of 405B parameter-equivalent scalar work, excluding the still-unreduced fine sweep.

One fine sweep remains `100%` of dense target arithmetic. At a perfect 128-token release its logical checkpoint traffic is `0.78125%` raw or `0.619070788%` under the inherited favorable lossless ratio.

## 9. Frozen decision

Promotion requires:

```text
all 384 build positions certified
all 384 untouched holdout positions certified
all integrity controls pass
```

Decision:

```text
PROMOTE_RESIDUAL_MARGIN_CERTIFICATE_TO_CAUSAL_COEFFICIENT_GATE
```

Otherwise, if at least one explicit LP is certified infeasible:

```text
REJECT_BOUNDED_BLOCK_RESIDUAL_MARGIN_CERTIFICATE_AS_EXACT_SOURCE
```

Any unresolved numerical or provenance condition yields:

```text
INVALID_RESIDUAL_MARGIN_CERTIFICATE_CONTROL_FAILURE
```

## 10. Stop rule

On rejection, do not sweep the margin, solver, L1 objective, active-set size, coefficient precision, block length, seed, shallow depth, prompt subset, or residual bank. Do not add target-specific nonlinear features and call them the same source.

Reopening requires a new information dependency: branch-generated residual rows, a checkpoint-derived nonlinear coefficient program available before target continuation, a sound token certificate that does not rely on this linear bank, or an exact symbolic transition that reduces fine dense arithmetic.

## 11. Claim boundary

A pass establishes only that the current online residual bank has finite-word token-margin capacity under a target-seeing coefficient oracle. A causal coefficient generator, target-independent certificate, exact prefix-state commitment cost, TARGET-W execution, physical 8-GiB allocation, CUDA/SASS, fine arithmetic reduction, and same-machine 4B-class p50/p95 remain `NOT TESTED`.
