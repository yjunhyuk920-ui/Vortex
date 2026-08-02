# Experiment 007 — causal selector rejection for rank-32 block repair

Evidence level: **E1 real-model falsification**

## Fixed target

The project target remains an unmodified 405B-class dense Hugging Face model on an 8 GiB GPU with original-quality acceptance and native-4B-class warm-decode wall-clock, without user training or manual model-specific preparation.

This experiment rejects one mechanism family. It does not reduce the target and does not claim that all possible runtimes are impossible.

## Family under test

```text
rank-32 O/down session capsule
+ one fixed exact residual-tile set shared across a proposed block
+ exact causal-prefix commitment
```

The corrected resource equations are:

```text
traffic/token = B_hot + rho * B_cold / A
compute/token = C_hot + rho * C_cold
```

For selectors that run a proposal-margin backward pass, selector compute is charged in addition to the hot forward path.

## Exact-future upper bound

An optimistic oracle using exact continuation tokens and teacher-forced gradients found:

```text
selected exact bytes:       8 MiB
repair fraction:          0.190642%
zero-repair prefix:          1 token
repaired prefix:             2 tokens
incremental prefix:          1 token
traffic efficiency E:     1049.087891
projected traffic:          2.014943 GiB/token
projected compute:          5.143432 GFLOP/token
```

This proves only that a useful set exists after future continuation information is revealed.

Source:

- `results/tinyllama_1_1b_block_shared_combined_gate.json`

## Causal selector 1 — residual-energy ranking

Selector inputs:

- approximate autoregressive activation residual energy;
- precomputed exact weight-tile Frobenius norms.

Result:

```text
zero-repair prefix:           1 token
compute-bound tile count:   683
selected bytes:       44,761,088
repaired prefix:              1 token
incremental prefix:           0 tokens
```

Decision: reject residual-energy ranking for this family.

Source:

- `results/tinyllama_1_1b_block_shared_residual_selector.json`

## Causal selector 2 — proposal-token signed adjoint

Selector inputs:

- hot path's own proposed continuation;
- proposed-token versus competitor margins;
- teacher-forced gradients on the hot proposal.

This was still an optimistic diagnostic because every managed exact weight tile was scanned when computing signed contributions.

Selector backward compute was charged:

```text
hot forward:        3.531515 GFLOP/token
selector backward:  4.059997 GFLOP/token
combined hot cost:  7.591513 GFLOP/token
exact-repair bytes allowed on TinyLlama: about 22.54 MiB
```

Result:

```text
zero-repair prefix:          1 token
1–64 selected tiles:         1 token
128 selected tiles:          0 tokens
compute-bound 360 tiles:     0 tokens
```

Decision: reject proposal-token signed-adjoint ranking for this family.

Source:

- `results/tinyllama_1_1b_block_shared_proposal_adjoint_oracle.json`

## Causal selector 3 — proposal-margin metadata bound

Selector inputs:

- hot proposed continuation;
- proposed-token margin gradients;
- input-residual energy;
- one precomputed Frobenius-norm scalar per exact tile.

The Cauchy score is:

```text
||W_tile||_F * ||G_row||_F * ||R_col||_F
```

Runtime ranking metadata for the tested modules was 84,480 bytes. Selector backward compute was charged.

Result:

```text
zero-repair prefix:          1 token
1–64 selected tiles:         1 token
128 selected tiles:          0 tokens
compute-bound 360 tiles:     0 tokens
```

Decision: reject this metadata-bound selector for the tested family.

Source:

- `results/tinyllama_1_1b_block_shared_margin_bound_selector.json`

## Causal selector 4 — exact prompt-prefill compiler

This experiment used information legally available before continuation:

- actual user prompt tokens;
- exact model logits and argmax decisions during prompt prefill;
- gradients from the last 16 prompt positions;
- exact weights streamed during automatic cold session compilation.

It used no continuation token or continuation gradient when selecting tiles.

Result:

```text
prompt tokens:               45
prefill decision rows:       16
zero-repair prefix:           1 token
1–683 selected tiles:         1 token
full 1.384 GB O/down repair: 64 tokens, but resource gates fail
```

Decision: reject the tested causal prefill signed-adjoint compiler.

Source:

- `results/tinyllama_1_1b_prefill_compiled_adjoint_oracle.json`

## Combined conclusion

```text
exact-future selectors passing: 1
causal selectors tested:        4
causal selectors passing:       0
```

The only passing tile set required exact future continuation tokens and teacher-forced continuation gradients. Four selectors restricted to information available before or during real generation could not increase the exact causal prefix, even though two were optimistic full-weight-scan diagnostics.

Therefore:

```text
REJECT rank-32 O/down capsule + fixed block-shared exact-tile repair
as the main VORTEX execution family.
```

Do not add more single-proposal scoring heuristics to this family without new evidence that changes the information boundary.

## Next architecture question

The next experiment asks whether the hot representation still preserves the exact token inside a small candidate set when top-1 is wrong.

```text
exact and hot logits are evaluated on the same authoritative exact prefix
measure exact-token rank under hot logits
measure coverage at K = 1,2,4,8,16,32,64,128,256
```

If the exact token remains in a small top-K set, a multi-hypothesis uncertainty certificate may be worth testing. If it falls outside top-32 frequently or at the first divergence, the rank-32 hot representation itself is rejected rather than only its selector.

Command:

```bash
python scripts/run_hot_candidate_coverage.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cpu \
  --tokens 32 \
  --max-rank 32
```
