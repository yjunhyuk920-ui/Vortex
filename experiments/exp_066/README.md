# EXP-066 — Exact Tensor-Train / MPO Bond-Rank Gate

Status: closed, E1.

```text
REJECT_REAL_Q4_TT_MPO_BOND_RANK_AS_CORE_RETAIN_MPO_CERTIFIER_AUXILIARY
```

This is a bounded lower-bound screen, not an MPO runtime implementation. It reuses checksum-verified EXP-065 Kronecker ranks and EXP-058 full-matrix ranks for the same Q4 tensors, applies exact adjacent TT-rank inequalities, and computes deliberately favorable classical MPO costs.

Run:

```bash
bash experiments/exp_066/reproduce.sh
```

The reproduction writes `results/exp_066_candidate/`. The committed authority is `results/exp_066/summary.json`; hashes and provenance for large deterministic raw JSONL files are in `results/exp_066/evidence_manifest.json`.

Authoritative result:

```text
operation p50 3.8941%  PASS
operation p90 6.7788%  PASS
storage p50  11.0524% FAIL against 10%
storage p90  22.9883% PASS
```

Because unresolved ranks and omitted implementation costs can only increase the lower bound, no exact core reconstruction, kernel, or runtime integration is authorized. The next Gate must change execution class.
