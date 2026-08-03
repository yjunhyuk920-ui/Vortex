# VORTEX Architecture

## Mission boundary

VORTEX is a runtime, not a retrained model. It ingests an unmodified supported Hugging Face dense checkpoint and automatically constructs runtime metadata.

## Target execution stack

```text
Hugging Face checkpoint
        |
        v
checkpoint/shard inspector
        |
        +--> automatic checksummed runtime-format compiler
        |
        v
causal token executor
        |
        +--> causal proposal, continuous block state, or exact baseline
        +--> exact/probabilistic certificate or exact target verification
        +--> certified commit OR declared exact/safe correction
        |
        v
memory scheduler
        - VRAM hot state <=8 GiB
        - RAM/SSD cold state
        - charged transfers, solver, proposal, verification, rejection, fallback
        |
        v
original-model-compatible output contract
```

## Core requirement

The primary runtime must causally reduce or amortize original 405B weight traffic and arithmetic on unseen prompts. A component is core only when it participates in that measured cost reduction.

## Mandatory interfaces

### Proposal or solver

Produces candidate tokens, soft block states, an operation result, interval, or execution capsule without future generated tokens. Reports every target stream, draft/solver operation, state byte, construction cost, rejected position, and numerical fallback.

### Certificate or verifier

Declares one explicit contract:

- deterministic exact;
- deterministic top-1;
- bounded-logit error;
- probabilistic top-1 with declared union-accounted `delta`;
- exact target block verification with longest-prefix plus first-mismatch correction.

Probabilistic certification is not deterministic exactness. Future-token oracle controls are not deployable.

### Fallback/correction

When certification, proposal matching, or numerical solving fails, execute the exact omitted target work required by the reference contract. No silent approximation and no uncharged fallback/correction stream.

### Memory virtualization

Separate GPU hot metadata, KV cache, work buffers, proposal/soft-state history, verification state, fallback tile, RAM cache, and SSD format.

### Evidence

Every run emits phase/evidence/provenance, exact revisions, target/draft/solver pass counts, logical/physical bytes where measured, proposal acceptance, rejected positions, fallback/correction, numerical failures, memory, timing distribution, and output agreement.

## Component classification

### Auxiliary accepted

- safetensors discovery/slice access;
- compact40/aligned64 mmap pointer VM;
- atomic/checksummed format builder;
- bounded exact decision-index compiler;
- exact finite-horizon suffix DAG as body compression;
- CPTC causal certificate, metadata fault rejection, and exact fallback at E1;
- `vortex_runtime/block_verify.py` exact longest-prefix plus correction verifier at E1.

These are not the core cost-reduction principle.

### Rejected core families

See `FAILED_APPROACHES.md`, including:

- global/oracle-tight/stratified range-based CPTC;
- hard target-only Jacobi under the tested stream accounting;
- sequential partial-layer self-draft with repeated target LM-head evaluation;
- proposal-tree expansion from the failed B3 source.

### Active core research

`EXP-049 — Anderson-Accelerated Continuous Block Fixed-Point Gate`.

## Closed CPTC architecture

EXP-047/047R correctness passed at E1, while the exact realized range oracle evaluated 100% at median and p90. Range-based CPTC is auxiliary only; no C3 rescue or GPU backend is justified.

## Closed EXP-048 architecture

### Retained verifier

```text
exact committed prefix + proposed block
        |
        v
one exact causal target block pass
        |
        v
compare proposal and target left to right
        |
        +--> commit every exact matching token
        +--> commit exact target token at first mismatch
        +--> discard all later positions/state
```

MEASURED verifier result:

```text
9 reference tests passed
B1/B2/B3 committed-output mismatches 0
future information in deployable B3 0
```

### B1 perfect future oracle

```text
96 exact tokens / one target pass
logical target-equivalent fraction 1.0416667%
future information true
deployable false
```

This proves the verifier arithmetic is sufficient if a nearly perfect long proposal already exists.

### Rejected proposal sources

B2 hard Jacobi:

```text
p50 58 target passes / 32 exact tokens
p50 fraction 181.25%
p90 fraction 193.75%
```

B3 partial-layer self-draft:

```text
p50 committed tokens / target verification 1
maximum proposal matching prefix 1
minimum fully accounted fraction 1333.463%
p90 fully accounted fraction 2893.843%
```

Decision:

- retain the exact block verifier;
- reject B2/B3 as core proposal mechanisms;
- do not continue B4 tree expansion from failed B3;
- complete real operation replacement remains NOT TESTED.

## Active EXP-049 architecture

### Continuous future block state

Represent `K` unknown future positions as soft token embeddings `Z`.

```text
exact prefix embeddings || soft future embeddings Z
        |
        v
unmodified target model, full causal batched pass
        |
        v
aligned future logits L(Z)
        |
        v
top-k sparse softmax projection through original token embedding
        |
        v
F(Z), residual R(Z)=F(Z)-Z
```

No target weight is modified. The deployable path may use only the exact current prefix, fixed initialization metadata, previous solver iterates, and current target outputs.

### S0 — hard Jacobi control

Reuse EXP-048 B2 and charge every target pass.

### S1 — damped Picard

```text
Z_next = (1 - lambda) Z + lambda F(Z)
```

Top-k, temperature, damping, block length, and iteration count are pre-registered.

### S2 — bounded Anderson acceleration

Maintain a small history of states/residuals, solve the residual least-squares system in float64, regularize and clip coefficients, reject NaN/Inf/ill-conditioning, and fail closed to S1.

### S3 — exact future-state oracle

Use exact future embeddings only to validate alignment, hardening, and upper-bound behavior. S3 is non-deployable and excluded from causal aggregates.

### S4 — adversarial triangular models

Construct causal finite models where position `i` reveals only a transformation of the exact predecessor. These test the worst-case one-new-exact-position-per-target-round barrier and prevent average-case extrapolation into a universal arbitrary-model claim.

### Hardening and exact commit

After the pre-registered solver iterations:

```text
soft state Z
   -> hard token proposal
   -> retained exact block verifier
   -> exact longest prefix + correction
```

All solver target streams, exact verification streams, projection operations/bytes, Anderson history, rejected positions, and corrections are charged.

## Resource equations

```text
M_total = M_hot + M_kv + M_work + M_soft_block
          + M_anderson_history + M_verify + M_fallback <= 8 GiB

S_target_equiv/token =
    (S_solver_target
     + S_exact_verify
     + S_target_correction
     + normalized_projection_and_solver_cost)
    / exact_committed_tokens

T_token >= max(B_total / effective_bandwidth,
               C_total / effective_throughput,
               serial_latency_floor)
```

PROJECTED same-bit traffic comparison:

```text
405B Q4 stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required target-equivalent stream fraction: 1.185185%
```

Dynamic exact-commit requirement before other overhead:

```text
required_committed_tokens = ceil(total_target_equivalent_streams / 0.01185185)
2 streams ->169
3 ->254
4 ->338
5 ->422
6 ->507
```

These are projections, not measured target performance.

## Theoretical claim boundary

EXP-049 may not claim universal arbitrary-model acceleration without resolving the causal triangular lower-bound question. A proof that black-box target-only rounds can guarantee at most one new exact position in the worst case rejects the universal mechanism even if average checkpoint prompts improve.

## Safety rule

No optimized path commits outside its declared exact verifier or certificate. Invalid metadata, numerical failure, ill-conditioned Anderson solve, proposal mismatch, absent proof, or corrupt state triggers exact correction/fallback or abort.
