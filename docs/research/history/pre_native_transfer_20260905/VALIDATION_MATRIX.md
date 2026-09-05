# Current validation snapshot

Date: 2026-09-05. This table adds the new auxiliary Gate; it does not promote or
reverse earlier core results. The
[complete previous matrix](docs/research/history/pre_global_decoder_20260905/VALIDATION_MATRIX.md)
is preserved as the original blob. Prior test counts remain historical.

| Claim | Phase A | Phase B | Phase C/D | Decision |
|---|---|---|---|---|
| Two adaptive bit probes for all 2x2 rank-one parities require five stored bits | proved by exact kernel obstruction | integer kernel/rank controls pass | not a checkpoint/hardware claim | scoped E1 auxiliary |
| Five-bit positive decoder | explicit construction | 144 queries pass | NOT TESTED | no physical speedup claim |
| Positive GF(2) successor-state control | explicit nontrivial transition | 8,192 transitions, zero mismatch | NOT a Transformer | control only |
| Arbitrary seven-bit 2x3 synthesis | formula specified | UNKNOWN/timeout, positive grammar SAT | NOT TESTED | INCONCLUSIVE |
| Two-probe grammar formula | branching semantics specified | 351,232 exhaustive cases pass | not a runtime | auxiliary |
| Binary/parity implies exact native numeric output | invalid implication | equal-parity/different-sum witness | no native lift | REJECT IMPLICATION |
| Reassociation preserves native FP32 reduction | false without ABI proof | BF16-term counterexample passes | target CUDA NOT TESTED | preserve declared ABI |
| General nonlinear word-probe impossibility | NOT ESTABLISHED | scope guard refuses extension | NOT TESTED | no claim |
| Complete 405B / physical <=8 GiB / native-4B p50,p95 | objective unchanged | no proxy acceptance | NOT TESTED | NOT ACHIEVED |

Reproduction and evidence: [research note](docs/research/E0_GLOBAL_DECODER_KERNEL_AUDIT.md),
[result JSON](results/e0_global_decoder/summary.json),
[focused test log](results/e0_global_decoder/local_tests.txt).
18 pytest tests and 18 unittest tests passed locally. Full-repository regression
was NOT run. No existing executable module was edited and no Actions job was
dispatched. `LOCAL_VALIDATION_PASS` applies only to the declared auxiliary scope.

## Governance-only amendment — CTC-2026-09-05

[Constructive theory acceptance](docs/CONSTRUCTIVE_THEORY_CONTRACT.md) adds a
separate O1-O6 proof track; it does not upgrade the empirical rows above.
`THEORY_STATUS=NOT_ESTABLISHED`; `HARDWARE_STATUS=NOT_TESTED`.
Document validation checks policy/link consistency only. Historical 18-test
counts above belong to the previous research run and were not rerun here.
