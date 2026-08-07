# EXP-074 — Weight-Stationary MTP/Expert Block Budget Gate

## Question

Can checkpoint-native causal proposals plus expert-stationary block verification
make Qwen3.5-122B-A10B behave like a 1B-class model on the measured 8 GiB target,
before any checkpoint is downloaded or a physical runtime is built?

This is a bounded Phase-A/B accounting Gate. It evaluates the user's proposed
122B surrogate and a transferable block-streaming primitive. It is not a claim
that the MoE surrogate proves the fixed arbitrary dense 405B mission.

## New information source and reopening premise

EXP-048/050 rejected target-only iterative proposals and fixed external drafts.
This candidate is different only in the following registered ways:

1. Qwen3.5 declares a checkpoint-native MTP component, causally coupled to the
   target rather than a fixed target-independent draft;
2. the model router reveals the exact active expert set after each layer input
   exists;
3. a verification block can group positions by expert and read each required
   weight page once per block.

These facts permit a cheap accounting Gate. They do not establish long accepted
prefixes, router locality, or universality. MTP and MoE are not guaranteed for an
arbitrary dense checkpoint.

## Candidate execution contract

```text
exact committed prefix
  -> checkpoint-native causal MTP proposes K tokens
  -> unchanged target verifies the K-token causal block
  -> each layer groups positions by routed expert
  -> cold expert page is loaded once for all positions using it
  -> exact longest matching prefix is committed
  -> first mismatch receives exact correction; later state is discarded
```

Any missing page, invalid route, numerical-contract failure, or unavailable
verifier triggers exact sequential completion or abort. Proposal tokens are
never committed without target verification.

## Registered target facts

Official Qwen material declares:

```text
total parameters                 122B
activated parameters/token       10B
layers                            48
experts/layer                     256
routed experts/token              8
shared experts/token              1
hidden size                    3,072
expert intermediate            1,024
MTP hidden layers                  1
Ollama Q4_K_M artifact            81 GB decimal
```

Sources are pinned by URL in `experiments/exp_074/config.json`. Values are
registered inputs, not locally measured checkpoint facts.

EXP-073 measured the target envelope:

```text
VRAM                            8 GiB class
host RAM                        23.4983 GiB
root free                       97.6183 GiB
```

No EXP-074 remote command, model download, inference, or storage allocation is
authorized.

## Reference accounting

Each Qwen expert is modeled as gate/up/down projections:

```text
P_expert/layer = 3 * hidden_size * expert_intermediate
P_expert/model = P_expert/layer * layer_count
```

The declared single-token active expert parameters are the eight routed plus
one shared expert. Remaining declared active parameters are charged as
non-expert active work.

For a verification block with routed-expert union `U(K)`:

```text
P_block(K) = P_nonexpert_active
             + P_expert/model * (U(K) + shared_experts)

F(K) = (P_block(K) / accepted_tokens + P_draft/token) / P_1B
```

Three route-union models are registered:

```text
fixed                       U(K) = 8
independent uniform expected U(K) = 256 * (1 - (1 - 8/256)^K)
maximally distinct           U(K) = min(256, 8K)
```

`fixed` is an optimistic oracle ceiling, not a measured route distribution.
The other two are reference controls, not claims about Qwen routing.

## Success and rejection thresholds

The target-equivalent traffic conditions are:

```text
p50 F(K) <= 1.2
p95 F(K) <= 1.5
```

The narrow MTP-1 plus paging candidate survives only if the standard one-token
speculation block meets the p50 bound even under fixed routes and zero-cost
drafting. Failure rejects that narrow mechanism.

The broader weight-stationary block hypothesis remains eligible only for a
small-checkpoint causal-proposal and router-trace Gate if its zero-cost,
fixed-route optimistic minimum is no more than 16 accepted tokens. This is a
necessary condition, not promotion.

## Strongest falsifications

- MTP-1 fails even under zero-cost proposal and perfect expert reuse;
- a realistic proposal cost consumes the 1B baseline allowance;
- expert union growth moves the required block toward 100 tokens;
- p50/p95 accepted-prefix distributions do not close the fully charged bound;
- batched arithmetic, rollback, KV, selector, and page costs erase traffic gain;
- the measured GPU is compute-bound rather than weight-bandwidth-bound;
- the mechanism remains Qwen/MoE-specific and cannot promote the dense mission.

## Stop rule

Until this Gate is interpreted, prohibit:

```text
122B or 35B checkpoint download
server inference or mutation
CUDA/page scheduler implementation
MTP backend implementation
claiming 1B-class latency from logical traffic
claiming a Qwen MoE result validates dense 405B
```

If MTP-1 fails but the optimistic long-block ceiling survives, record `REVISE`
and require a separate small-checkpoint proposal-length Gate before any MoE
checkpoint or scheduler work.

## Claim boundary

The result ceiling is E1 reference accounting. Model routing, proposal
acceptance, exact floating behavior, physical traffic, latency, VRAM runtime,
405B execution, and E4–E7 remain **NOT TESTED**.
