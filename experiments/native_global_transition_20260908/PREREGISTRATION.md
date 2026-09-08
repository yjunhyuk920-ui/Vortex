# Native full-transition constructor continuation — 2026-09-08

## Final target and current state
Parent35e0a2eb1d9d64b20d17bd6e0c4afba07d3d70d3 (PR148 output-envelope).
Concurrent PR147/378fcd584330d17f9d144f64ccbc01721903ea6a is separately preserved.
Fixed mission/CTC-2026-09-05 and O1-O6 were reread at the actual PR147 checkout.
No output-envelope packet/seed/tensor sweep or correlation positional scan follows.

For every unchanged public dense405B checkpoint, legal input/prefix/context/RNG,
the intended constructor must produce an exact whole causal executor with original
observer and successor-state semantics, peak GPU<=8GiB, same-machine4BQ4 ratios
p50<=1.2,p95<=1.5 and existing TTFT. All preparation/storage/arithmetic/transfers/
verification/repair costs included. The final theorem is NOT established.

## Three flows compared before actual full HF execution

### A — Global native relations with shared functional elimination (selected first)
Input: checkpoint+frozen reference implementation/config+explicit input/state ABI.
Represent each native operation as a typed functional relation z=op(args) with
fixed attributes and alias/effect semantics; pin actual kernel/library identities.
Roots are *all* returned logits and all K/V, positions, and effect/RNG state.
Construct in trace order, backward-mark live roots, substitute eliminated variables
as references into a hash-consed DAG. Never expand a relation into a Cartesian
truth table. Deterministic sequence/serialization and last-use reclamation are
explicit; no 'find a good order' routine. Exact duplicate deterministic operations
may share a value only when alias/mutation/effect guards make it legal.

Equation: exists intermediates AND_v R_v equals the graph of the decoded function.
Preserving exact op implementations/order among remaining dependent/effectful
nodes yields output and successor bit equality under the guarded trace semantics.
Native kernels are NOT free primitives: each retained matrix contraction is paid
by its actual coefficient bytes and dimensions; backend service remains a parameter.

Sufficient structural description bound O(V+E+constantbytes), query schedule one
evaluation per remaining node, not an exponential relation table. This controls
description scope, not the cost of its opaque matrix functions. Complete latency
bound requires sum actual op service bounds+state+transport+allocator costs.
>=10x route: compile away >=90% of paid contractions/weight consumption across
the original body. This property is to be measured/proved, NOT assumed. Strongest
counterexample: every contraction contributes to an exposed logit or future KV.
Cheapest decisive test: exact full-HF captured graph; count removed *paid work*,
not nodes. If this rule retains body work, no larger symbolic search is justified.

### B — Persistent native causal chart, not per-call reconstruction
Try a checkpoint/config-generated chart of the existing rotary KV transition;
the candidate removes recurring coordinate operations by storing values in the
producer chart and rewriting consumers. Required equations for every legal event:
 E(T(s,u,r))=T'(E(s),u,r), O'(E(s))=O(s), and equal next RNG bytes.
First finite test: generate actual rotary coefficients at frozen positions and
test forward/inverse *native* relation; algebraic real inverses are not sufficient.
If inverse transport loses bits, that chart is rejected rather than repaired by
an unspecified oracle. Continue with a fully explicit invertible storage chart
generated from observed state schema: exact bit layout, metadata/offsets, loader,
native consumers and future updates. This latter bridge proves a state API, NOT
a saving unless decode/reconstruction and the original body genuinely disappear.

Paid equation: build_E+load+init_E+sum(step_T'+observer+codec+allocation+transport).
>=10x requires body projection work to disappear in the new chart, not only a KV
copy/rotation. Strongest counterexample: original nonlinearity/projection is still
called, or inverse native rotations collide. Cheapest tests: actual rotary pairs,
actual sampled legal HF continuations, complete logit/KV/RNG comparisons and full
operation budget. No synthetic conjugated program is substituted for the checkpoint.

### C — Query-time bidirectional native constraint propagation
Rather than compile-time variable elimination or a persistent chart, keep domains
for live words and propagate exact native forward/inverse fibers from known input
and observer requirements. A finite worklist visits a node when a domain shrinks;
sets are finite word sets, not real intervals assumed to encode signed-zero/NaNs.
The invariant retains all reference assignments. Singleton roots can be committed;
otherwise ordered native forward evaluation completes all roots with paid work.
No bound oracle or supplied desired output. Store domains with an explicit size
cap and native expression references; cap overflow does not imply a cheap decode.

Resource bound for explicit w-bit sets: O(V*2^w)bits and O(E*2^w*iterations)work;
expression form O(V+E)description but complete forward work may remain. >=10x
requires propagation to eliminate expensive gates before their original products,
not merely confirm results afterward. Strongest case: unconstrained logits/KV
leave no backward information; each nontrivial output must be computed. Cheapest
decisive test: consumer/root domain inventory on the same actual HF graph, then
restricted exact-word rule implementation only where it removes a paid dependency.

These are three different proposed information flows, NOT three admitted new
principles. Known graph/compiler/invariant tools may be used; no novelty claim.
If A fails, proceed to B and C constructions within this session, not stop at A.

## Frozen implementation and test population
Isolated Python3.12 environment. torch2.8.0 CPU, transformers4.55.4, NumPy2.2.6,
requests2.32.4,safetensors0.6.2; transitive versions recorded after install.
Original SmolLM2-135M revision93efa2f097d58c2a74874c7e644dbc9b0cee75a2.
Load untouched safetensors at checkpoint dtype; no trust_remote_code or adapter.
Reference is pinned CPU HF eval default attention unless explicitly marked otherwise.
One thread, deterministic settings. Capture metadata/alias/stride and all root types.
No conversion of matmul into a different reduction to make the test pass.

Legal input token-ID prefixes (no tokenizer/content claim):
 [1,42,73,11], [1,314,271,18], [2,77,29,3].
HF generate do_sample=True,top_k=0,top_p=1,temperature=1,max_new_tokens=4;
seeds260908,260909,260910. Each execution samples from its own logits/RNG.
No target token sequence fed to the candidate. EOS stopping is not overridden.
Capture original RNG state before/after every step, all KV bytes/metadata, full
returned logits and separately the untruncated prefill interface.
No claim that this finite population proves all continuations or target latency.

Compilation/capture failures, fixes and additional regression checks are logged.
Trace-shape specialization must be recorded; non-strict capture alone is not a
universal proof. Explicitly guarded pure tensor op semantics enable local induction;
coverage of Python data-dependent branches, every legal length and all CUDA kernels
is a distinct OPEN lemma. Any original dense function retained is fully charged.

## Termination/schedule/acceptance
Finite graph capture for the supported reference, deterministic elimination and
finite replay plus explicit shape/dtype/layout guards. Unknown operators/effects
are rejected or retained with costs, not declared free or correct by convention.
Runtime outside admitted shape/semantic guards is NOT a target success.
Whole-model O1-O6 cannot close solely from these runs. Full405B/8GiB/GPU/4BQ4/TTFT
remain NOT TESTED; O5 stays OPEN absent sufficient upper/coupled ratio evidence.
No reference performance claim from a library node count or favorable lower bound.

User-requested session_finish has actually been called twice and returned HELD.
After this newly attached instruction and substantive verification, call it again
before final and process returned in-scope work. No exact five-minute claim.
