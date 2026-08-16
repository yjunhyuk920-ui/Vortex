# VORTEX Research Handoff

Last updated: 2026-08-16 Asia/Seoul

## 1. Authority and provenance

This is the canonical cross-session handoff for VORTEX. Conversation history, reconstructed local trees, local-only commits, bundles, and unapplied patches are not authoritative project history.

Authority order:

1. actual remote commits in `yjunhyuk920-ui/Vortex`;
2. machine-readable results and checksums contained by those commits;
3. root governance documents in those commits;
4. this handoff;
5. chat/local-only artifacts.

Verified base when this handoff branch was created:

```text
repository   yjunhyuk920-ui/Vortex
base branch  research/exp-082a-differential-spanning-tree
base SHA     03737a8e4aa76bd3c23a0c38478a9079c27e649d
base message research: reject free nonlinear recursive lift
```

The previously reported SHAs below were reconstructed local work and must never be cited as remote Vortex commits:

```text
6153a73c6b3d17dbf000a0c6592ae49ee323ce9a
a8a9cfc32b6f2faa046094d5794ffe0c26a28f34
f34124884fc6cc74518150eb62c3d1e3c71ca581
```

Their scientific content can only become project authority after reapplication, validation, and a new remote commit.

## 2. Non-negotiable objective

Build a runtime replacement for arbitrary publicly released, unmodified Hugging Face dense autoregressive Transformers, including a real 405B-class model, with:

```text
no retraining / fine-tuning / LoRA / distillation
single GPU
peak GPU VRAM <= 8 GiB
batch size 1
same-machine native 4B Q4 baseline
p50 warm time/token <= 1.2x baseline
p95 warm time/token <= 1.5x baseline
CPU RAM, SSD, PCIe, preprocessing, artifacts, verification, misses and fallback charged
reproducible code, checkpoint revisions, commands and hashes
```

The target is not satisfied by a theorem, toy model, projected resource equation, throughput batching, or local-only patch.

## 3. Output contracts

### EXACT-E

For a fixed checkpoint, tokenizer, numerical ABI, prompt, generation settings, sampling algorithm and RNG tape:

\[
Tokens(E,W,p,r)=Tokens(Ref,W,p,r).
\]

Greedy equality and fixed-RNG sampling equality are evaluated separately. Q4-surrogate exactness is not BF16/FP16 target exactness.

### QUALITY-Q

QUALITY-Q is separate. Perplexity, benchmark, generation-quality and representation thresholds must be preregistered before results. QUALITY-Q results must never be reported as EXACT-E.

## 4. Claim scopes

Every result must be labelled as one of:

- `ALL-WEIGHTS`
- `PUBLIC-FAMILY`
- `SINGLE-CHECKPOINT`
- `TOY-CONSTRUCTION`

An ALL-WEIGHTS lower bound does not automatically prove a claim about one pretrained checkpoint. A structure measured in one checkpoint does not automatically generalize.

## 5. Current final state

```text
actual 405B end-to-end execution                  NOT TESTED
peak VRAM <= 8 GiB                                NOT TESTED
same-machine native 4B p50/p95 gate               NOT TESTED
final exact token/sampling suite                  NOT TESTED
completed executor core                           NONE
TARGET_PASS                                       false
E6/E7                                             not achieved
```

A valid executor must produce both the next token and an exact or behaviorally bisimilar successor state while avoiding almost all dense-projection information movement and arithmetic on the online path.

## 6. Remote-authoritative frontier through base SHA 03737a8

The verified branch already contains repeated negative evidence for ordinary exact low rank, displacement structure, exact sparsity/zero skipping, cached K/KV reuse, output-row/prototype reuse, exact Kronecker/tensor structure, temporal span replay, local pattern tables, terminal-only differential trees, legal Causal Residual Atlas, trace-built bilinear span ledgers, exact-field algebraic branching and nearby variants.

Do not reopen a rejected mechanism by changing ranks, block sizes, prompts, mode order, primes, tree shapes, or names unless a new information source or measured fact invalidates the original rejection premise.

The verified frontier at `03737a8` leaves a paid, lossless, sub-dense source for the Bilinear Cross Residual `r^T W u` unresolved. No executor core survives on that remote base.

## 7. Local-only research after the verified base

Several later research rounds existed only in reconstructed/local trees. Their results are useful hypotheses and evidence, but are not remote-authoritative until re-applied and revalidated on this branch or a descendant.

Important local-only conclusions to preserve as research leads, not remote facts:

- proof-carrying causal quotient alone is a verifier, not a transition source;
- unrestricted `2x3 -> 7-bit -> 2 adaptive bit probes` nonlinear seed was reported rejected;
- a block-local `25x108 -> 50 x 64-bit -> 2 reads` source was reported to require far more cells under a certificate-style lower bound;
- global mixed-cell/direct-sum and entropy/Mailman/SubSpec compositions were reported insufficient under their frozen models;
- an exact three-lane signed-Q4 integer packed-dot primitive was constructed, but it is Q4-integer-only, expands representation, and is not BF16/FP16 target exactness or a full core;
- ROOT-ESCAPE auditing corrected the claim that `84.375 contributions/operation` is a universal law; it is only a dense-equivalent normalization;
- cold-backed ALL-WEIGHTS inner-product work reported a logical tradeoff of the form `H + p(w + ceil(log2 S)) >= n`, but that does not settle one fixed public pretrained checkpoint;
- a one-probe selector construction showed that checkpoint entropy alone does not imply large per-token reads.

These results must be individually reintroduced with code/results if they are needed for a future proof or implementation.

## 8. Correct current problem

For a fixed public checkpoint `W*` and exact numerical ABI `R`, the remaining constructive problem is the explicit online state-transition function:

\[
F_{W^*,R}:(prompt,KV,RNG)\mapsto(next\ token,successor\ state).
\]

The objective is not merely to compress static checkpoint entropy. It is to construct an existing-ISA, checkpoint-specific compiler/executor that computes this continuous autoregressive transition under the full 8 GiB and same-machine 4B latency ledger, or to produce a correctly scoped lower bound for that explicit fixed function.

A static next-token lookup without successor state is not an executor core.

## 9. Research-session Git rule

Every meaningful research round must produce at least one meaningful Git commit. When write access exists, the same round must push the branch and verify that the remote branch contains the commit.

Required discipline:

```text
verify remote + authentication
-> verify base branch/SHA
-> create research branch
-> read AGENTS.md + this handoff + canonical state docs
-> implement and validate
-> save raw evidence + hashes
-> update research state + one next experiment
-> commit before reporting
-> push
-> verify remote SHA
-> create/update PR and record CI
```

Rules:

- no direct push to `main`;
- no force push;
- no rewriting another research branch;
- no empty commit used only to satisfy policy;
- local-only SHA/bundle/patch must never be reported as a remote commit;
- failed mechanisms are committed with reproducer, evidence and decision;
- tests not run are never reported as passed;
- actual branch, SHA, push state, PR and CI state are reported explicitly.

## 10. Single next constructive experiment

After this handoff is remotely committed, the only promoted constructive experiment is:

`FIXED-PUBLIC-DYNAMIC-EXECUTOR-G1`

Minimum gate:

```text
actual public small Llama-family checkpoint loaded via official Hugging Face path       PASS
official reference forward and 128-token generation                                    PASS
checkpoint-specific artifact/compiler generated automatically                           PASS
one complete Transformer layer replaced                                                  PASS
128 consecutive decode transitions compared                                              PASS
successor hidden/KV exact or formally bisimilar                                          PASS
existing CPU/GPU ISA only                                                                PASS
preprocessing + artifact + hot/cold bytes fully charged                                  PASS
physical bytes or measured latency lower than official reference                         PASS
remote Git commit and branch verification                                                 PASS
```

The experiment must record exact package versions, checkpoint revision/hash, dtype/numerical ABI, reference commands, raw outputs, token sequence, successor state, bytes, preprocessing, artifact size and measured latency where available.

If the mechanism fails, commit the failure and close its mechanism fingerprint. Do not branch into a family of nearby parameter sweeps before the gate is decided.

## 11. Startup instruction for the next session

Read, in order:

1. `AGENTS.md`
2. `docs/research/VORTEX_RESEARCH_HANDOFF.md`
3. `RESEARCH_STATE.md`
4. `FAILED_APPROACHES.md`
5. `ASSUMPTION_REGISTER.md`
6. `VALIDATION_MATRIX.md`
7. `NEXT_EXPERIMENT.md`

Then verify the actual branch HEAD, PR and CI before doing research. Conversation memory is not authoritative.
