# Causal/global producer frontier — preregistration

Date: 2026-09-08
Parent scientific commit: `2bd2bb8d27f0da7178c10211461bc787aadbe0da`

The fixed VORTEX mission and O1-O6 are unchanged. The preceding nonlinear-word
round closes local bounded-word routing under a strong independently-selectable
query interface, but explicitly leaves the bridge to legal batch-1 causal
execution open. This round does not reopen that local router.

## Intended final theorem

The desired final theorem is still a uniform finite constructor and causal
executor for every checkpoint in mission scope. It must map

```text
(checkpoint bytes, legal input, exact causal state, RNG state)
    -> exact logits + exact required successor state + exact RNG successor
```

while evaluating/reading less than ten percent of the original dense effect as
a core-entry condition and ultimately closing the same-machine native-4B-Q4
latency target with all costs paid.

No principle below is admitted merely because an interface can be written.

## Principle A — legal-causal dense-effect exposure compiler

Reverse the assumption that the abstract dense queries used by information
Gates may have nothing to do with a legal Hugging Face causal state transition.

Finite constructor for a bounded architecture family:

```text
input:  layer count L, hidden size d, binary matrices W_l in {0,1}^{dxd}
output: ordinary LlamaForCausalLM checkpoint C(W_0,...,W_{L-1})
```

Use `d` legal token IDs whose embeddings are standard basis vectors. Zero the
attention output and MLP paths so residual hidden state remains the same basis
token at every layer. Put `W_l` in layer `l`'s native `v_proj`. Keep the native
RMSNorm and cache implementation unchanged.

Stored representation: the ordinary checkpoint tensors. No response table is
stored by the constructor.

Runtime address rule: a legal token ID `j` selects embedding `e_j` through the
normal HF embedding lookup. No checkpoint-derived selector is supplied.

Executable query: run the unmodified HF forward with `use_cache=True`.

Exact native equation to prove for the binary subclass:

```text
Vcache[l, position t, output i] != 0  <=>  W_l[i, token_t] = 1.
```

The nonzero scalar is whatever BF16 value native RMSNorm produces on a one-hot
embedding; it need not be assumed or algebraically replaced. Each `v_proj` row
has only one nonzero input coordinate, so binary zero versus one remains
distinguishable without an associativity assumption.

Termination: ordinary finite HF prefill/decode.

Paid bounds: this is a reachability construction, not a proposed fast executor.
Its constructor writes `L*d^2` variable binary `v_proj` entries plus fixed model
tensors. Native reference forward and full state costs are paid in the bounded
test. No target-scale speed claim is assigned.

Potential >=90% route if causal queries were always basis-column queries: a
column-addressed lossless layout could read only the required output column
instead of contracting the full matrix. The strongest counterexample to making
this a universal VORTEX producer is an arbitrary later-layer dense hidden state;
the basis invariant is deliberately created by this special checkpoint and is
not a property of arbitrary checkpoints.

Cheapest decisive test: construct actual transformers `LlamaForCausalLM` BF16
models locally, compare complete native `DynamicCache` values for full prefill
and incremental continuation against the embedded binary matrices, and verify
RNG plus checkpoint hashes. If this fails, the abstract-to-causal bridge is not
established even in the declared family.

Disposition before execution: **selected as the strongest finite missing
construction that can be completed without inventing a producer oracle.** It
is a bridge/obstruction primitive, not core admission.

## Principle B — globally mixed nonlinear checkpoint producer

Reverse the local-direct-sum premise. Encode the complete checkpoint jointly,
so one runtime word may mix coefficients from many matrices/layers and one
causal query may reuse that word across all required outputs/state.

Required constructor:

```text
E(checkpoint) -> global encoded store G
A(input,state,prior returned words) -> next address
D(input,state,returned words) -> exact logits/KV/RNG successor
```

Stored representation and addressing must be finite and explicitly charged.
Termination requires a fixed finite probe/computation bound for every legal
continuation. Exactness is literal native finite-word state equality.

The >=90% route is direct only if total runtime cold/global payload and decoder
work are <=10% of the original dense effect while `G` and the original
checkpoint remain within the paid storage/memory schedule. Global mixing is the
one known way this principle could evade the preceding local cover Gate.

Strongest adversary: independent arbitrary dense matrices across every layer,
with causal inputs chosen so no checkpoint-local redundancy can be assumed.

Cheapest falsifier: demand an explicit encoder/address/decoder at two growing
sizes. A literal response catalog, exhaustive continuation table, per-matrix
local code, or query-time full contraction is immediate rejection.

Disposition: **OPEN interface, not promoted.** No finite generic encoder or
decoder has been constructed. Treating `E/A/D` as solved would violate CTC.

## Principle C — paid dynamic causal effect state

Reverse the premise that checkpoint information must be re-queried from a
static representation every token. Maintain an exact evolving state `Z_t` that
contains future-useful checkpoint effects generated by earlier legal work:

```text
(output_t, Z_{t+1}) = Q(Z_t, input_t, causal_state_t)
Z_0 = Build(checkpoint)
```

`Z_t` is distinct from raw KV serialization: it qualifies only if its finite
update/query algorithm produces later dense effects without replaying the
original dense matrices.

Stored representation: checkpoint plus `Z_t`; peak GPU, CPU RAM and persistent
storage all count.

Address rule: explicit fields in `Z_t`, never a free future-token selector.

Exactness equation: inductive equality of exposed logits, required native state
and RNG after every legal continuation prefix.

Termination: finite Build, Query and Update programs.

The >=90% route requires Build amortization to be valid under TTFT/short-session
rules and each later Query+Update to move/compute <=10% original dense effect.

Strongest adversary: a cold legal session followed by causally reachable hidden
states that introduce new independent checkpoint queries each token.

Cheapest falsifier: if the first useful `Z_t` is merely a set of previously
observed `W x` values, an adversarial new hidden direction forces another full
`W x`; that is the prior reuse/frontier family and is rejected. A candidate must
specify a different sufficient statistic.

Disposition: **no new sufficient statistic constructed.** Raw KV, observed
response caches and growing linear-span bases are excluded as prior families.

## Selection and execution order

A is executed first because it has a complete finite constructor and decides a
real causal-information gap. B remains the highest-upside *producer* class, but
there is no admissible algorithm to execute yet. C currently collapses to
rejected response/span caching when made explicit.

After A, use its exact scope to sharpen B's required causal query family. Do not
claim A itself accelerates inference. If A succeeds, the next missing object is
still a global causal producer, not another abstract local query code.

Initial status:

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
```

405B, CUDA, PCIe/SSD target scheduling, physical <=8 GiB GPU allocation,
native4BQ4 p50/p95 and TTFT are not executed in this bounded round.
