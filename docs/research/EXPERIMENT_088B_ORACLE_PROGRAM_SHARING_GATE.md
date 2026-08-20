# EXP-088B — Oracle cross-layer exact program-sharing Gate

## 1. Question

Does a fixed public checkpoint contain a **checkpoint-static, cross-layer, cross-page, nonseparable exact transition microprogram** that can be selected on build states and reused unchanged on multiple unseen causal states while removing enough checkpoint pages and dense arithmetic to justify an executor implementation?

This is the cheapest favorable oracle Gate. It does not claim that the oracle is deployable.

## 2. Mechanism fingerprint

```text
DEV-W: HuggingFaceTB/SmolLM2-135M
revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
reference: official transformers 4.46.3 LlamaForCausalLM, eager BF16 CPU
window: complete adjacent decoder layers 14 and 15
checkpoint page: contiguous 64 x 64 tile of a selected nn.Linear weight
page family: deterministic round-robin union of tiles across every linear projection in both layers
group count: 8
program: 8-bit retained-family mask
search: exhaustive all 2^8 programs, Gray-code checkpoint mutation
state exactness: final pair hidden plus complete post-step K/V for both layers, raw-byte equality
selection: minimum retained bytes in the exact-program intersection of three build states
holdout: one frozen program, no selector, three distinct untouched states
```

The partition is generated from checkpoint tensor layout only. It never observes activations, prompts, reference outputs, state IDs, or hashes.

## 3. Exact transition

For causal state `s`, let the official two-layer transition be

\[
E_W(s) = (h_{16}, K'_{14}, V'_{14}, K'_{15}, V'_{15}).
\]

The static plan partitions every linear checkpoint tile in the window into disjoint families

\[
\mathcal G = \{G_0,\ldots,G_7\}.
\]

A program mask `m` retains families whose bits are one and replaces every omitted checkpoint tile by exact zero before executing the unchanged official two-layer graph:

\[
E_m(s)=E_{W\odot M_m}(s).
\]

Exactness is declared only when the raw bytes of all five successor tensors match:

\[
E_m(s) \equiv_{\mathrm{bytes}} E_W(s).
\]

No equality is required at an individual page, projection, attention block, MLP, or first-layer boundary. Therefore a program may depend on cancellation and compensation across pages, projections, nonlinearities, residuals, attention, and both layers. Exhaustive enumeration includes masks that remove several page families even when every single-family removal fails.

## 4. Program sharing

For each build state `s`, the oracle measures the exact set

\[
A_s=\{m\in\{0,1\}^8:E_m(s)=E_W(s)\}.
\]

The selected program is

\[
m^*=\arg\min_{m\in\bigcap_{s\in B}A_s} B_{\rm pair}(m),
\qquad
B_{\rm pair}(m)=\sum_{j:m_j=1}|G_j|.
\]

Only build states participate in this selection. After `m*` is frozen, holdout reference transitions are captured and the same mask is applied to every holdout state. There is no runtime selector, state lookup, state hash, output table, target literal, future generated token, verification fallback, or per-state program choice.

## 5. Why this is a favorable upper bound

The Gate grants the proposed route more than a deployable executor would receive:

- exhaustive reference-output access on build states;
- all oracle-search dense forwards are offline;
- page metadata, mask decoding, addressing, materialization, and scheduling are free;
- a future sparse kernel is assumed able to skip omitted tiles while preserving the zero-masked graph exactly;
- the measured retained fraction is optimistically projected across the full 405B checkpoint;
- non-linear state traffic and program bytes are free in that projection.

A negative result therefore kills the frozen page-mask language before backend work. It does not prove that every richer exact decompiler language is impossible.

## 6. Resource equations

For one-token execution, a retained tile with `n` BF16 weights contributes `2n` checkpoint bytes and `n` linear MACs. The oracle language therefore has the same retained fraction for selected-window linear cold bytes and dense MACs:

\[
r(m)=\frac{B_{\rm pair}(m)}{B_{\rm pair}(\mathbf 1)}.
\]

The deliberately optimistic 405B projection is

\[
B_{405}^{\rm optimistic}(m)=810\times10^9\,r(m)\;\text{bytes/token}.
\]

The registered hot-byte target requires

\[
r(m)\le \frac{100\times10^6}{810\times10^9}
=1.2345679\times10^{-4}.
\]

The experiment uses `r <= 0.25` only as a **promotion-to-next-Gate signal**, not as final-target success. A real 405B executor is not authorized by a 25% result.

## 7. Preregistered gates

Integrity requires all of the following:

1. the official public checkpoint and pinned runtime load;
2. all six input hidden-state hashes are distinct and build/holdout are disjoint;
3. replaying the unmodified two-layer window is bit-exact for every state;
4. a one-bit successor perturbation is detected;
5. all 256 build programs are evaluated;
6. the checkpoint bytes are restored exactly after Gray-code mutation;
7. every page family spans both selected layers.

Program-sharing promotion requires:

1. a non-full program is exact on all three build states;
2. the same frozen program is exact on all three untouched holdout states;
3. retained selected-window linear bytes are at most 25%;
4. every integrity control passes.

Direct final-target evidence additionally requires the optimistic 405B hot-byte projection to be at most 100 MB/token. Even then, TARGET-W and target hardware remain `NOT TESTED` until measured.

## 8. Controls

- **Full mask:** must reproduce all five transition tensors exactly.
- **Empty mask:** recorded as a negative control; a genuine exact identity is not silently discarded.
- **Bit flip:** the comparator must detect a one-bit successor change.
- **Duplicate exclusion:** equal input-state hashes invalidate the split.
- **Restoration:** selected-window checkpoint hash must match before and after enumeration.
- **Nonseparable synthetic positive control:** two residual scalar layers are constructed so that deleting either page alone fails but deleting both pages succeeds. This proves the search language and implementation can detect joint cancellation.

## 9. Stop rule

If the minimum build-shared program is the full mask, fails untouched holdout exactness, or retains more than 25%, reject this page-mask microprogram as a 405B core. Do not sweep tile sizes, family counts, layer pairs, prompts, mask order, or thresholds.

Reopening requires a materially richer static instruction—such as exact page replacement or fused computation—that changes the information source rather than merely making the mask finer.

## 10. Claim boundary

A pass would justify a complete-layer, 128-consecutive-transition, existing-ISA executor Gate. It would not establish 405B feasibility, 8-GiB residency, target latency, or universal arbitrary-checkpoint success. A rejection applies only to the frozen exact page-mask language.
