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
        +--> causal proposal or block proposal
        +--> exact/probabilistic certificate or exact target verification
        +--> certified commit OR declared exact/safe fallback
        |
        v
memory scheduler
        - VRAM hot state <=8 GiB
        - RAM/SSD cold state
        - charged transfers, proposal, verification, rejection, fallback
        |
        v
original-model-compatible output contract
```

## Core requirement

The primary runtime must causally reduce or amortize original 405B weight traffic and arithmetic on unseen prompts. A component is core only when it participates in that measured cost reduction.

## Mandatory interfaces

### Proposal

Produces candidate tokens, an operation result, interval, or execution capsule without future generated tokens. Reports every weight stream, operation, state byte, construction cost, and rejected proposal.

### Certificate or verifier

Declares one explicit contract:

- deterministic exact;
- deterministic top-1;
- bounded-logit error;
- probabilistic top-1 with declared union-accounted `delta`;
- exact target block verification with longest-prefix commit.

Probabilistic certification is not deterministic exactness. Future-token oracle controls are not deployable.

### Fallback/correction

When certification or proposal matching fails, execute the exact omitted target work required by the reference contract. No silent approximation and no uncharged fallback/correction stream.

### Memory virtualization

Separate GPU hot metadata, KV cache, work buffers, proposal state, verification state, fallback tile, RAM cache, and SSD format.

### Evidence

Every run emits phase/evidence/provenance, exact revisions, forward/layer/tile/pass counts, logical/physical bytes where measured, proposal acceptance, rejected positions, fallback/correction, wrong accepts, memory, timing distribution, and output agreement.

## Component classification

### Auxiliary accepted

- safetensors discovery/slice access;
- compact40/aligned64 mmap pointer VM;
- atomic/checksummed format builder;
- bounded exact decision-index compiler;
- exact finite-horizon suffix DAG as body compression;
- `vortex_runtime/cptc.py` and EXP-047R audit code as E1 certificate/fallback references.

These are not the core cost-reduction principle.

### Rejected core families

See `FAILED_APPROACHES.md`, including global/oracle-tight/stratified range-based CPTC.

### Active core research

`EXP-048 — Causal Block Verification Amortization Gate`.

## Closed CPTC architecture

EXP-047 and EXP-047R insertion point:

```text
linear pair margin partitioned by input dimension
causal random contribution sampling
range-based finite-population interval
accept sign or complete exact work
```

Correctness passed at E1. Savings failed.

Authoritative EXP-047R MEASURED result:

```text
18 current-token states from three pinned trained checkpoints
C1 exact-state oracle median 100%
C1 oracle p90 100%
C2 median/p90 100%
C2 best 254/256 = 99.21875%
wrong accepts 0
bound violations 0
```

Decision:

- retain certificate/fallback machinery only as auxiliary safety infrastructure;
- reject range-based CPTC as the primary executor;
- do not implement C3 as an EXP-047R rescue;
- do not build a CPTC GPU backend from these results.

## Active EXP-048 architecture

### Stage B0 — sequential exact baseline

Run exact greedy generation and record one target-equivalent full stream per generated token.

### Stage B1 — exact block-verification oracle

Given a proposed block, execute the exact target over all proposed positions with a causal mask, compare tokens left to right, and commit only the matching prefix plus the exact correction token. Perfect future proposals are allowed only as a non-deployable upper bound and accounting test.

### Stage B2 — Jacobi control

Reuse the existing exact Jacobi decoder. Charge every full target pass and failed iteration. It is a control, not the active mechanism.

### Stage B3 — causal partial-layer self-draft

```text
exact committed prefix/KV
        |
        v
same-checkpoint early-layer draft, no training/adapter
        |
        v
K-token causal proposal
        |
        v
one exact full-target teacher-forced block verification
        |
        +--> longest exact prefix commit
        +--> exact target correction at first mismatch
```

All partial-layer streams, proposal steps, output-head work, exact verification, rejected scored positions, KV rebuilds, and corrections are charged.

### Stage B4 — bounded proposal tree

Forbidden until B3 survives its early rejection Gate. Every expanded node and target-scored position must be charged; no future/reference routing.

## Resource equations

```text
M_total = M_hot + M_kv + M_work + M_proposal + M_verify + M_fallback <= 8 GiB

S_target_equiv/token =
    (S_target_verify
     + S_target_correction
     + S_partial_draft)
    / accepted_tokens

T_token >= max(B_total / effective_bandwidth,
               C_total / effective_throughput,
               serial_latency_floor)
```

PROJECTED same-bit traffic comparison:

```text
405B Q4 stream: 188.592821 GiB
1.2x 4B Q4 allowance: 2.235174 GiB/token
required target-equivalent stream fraction: 1.185185%
zero-cost perfect-proposal minimum: 85 accepted tokens/full target pass
```

The 85-token value is a projection, not measured performance. Phase-D hardware terms remain NOT TESTED.

## Safety rule

No optimized path commits outside its declared exact verifier or certificate. Invalid metadata, numerical failure, proposal mismatch, absent proof, or corrupt state triggers exact correction/fallback or abort.
