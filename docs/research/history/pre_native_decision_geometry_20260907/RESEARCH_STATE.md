# Research state — 2026-09-07

Fixed mission / CTC-2026-09-05 unchanged.
[Constructive conditional residual/state source](experiments/residual_absorption_20260907/REPORT.md).
[Scoped obligations, failures and assumptions](experiments/residual_absorption_20260907/LEDGER.md).

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false

Exact L1 envelopes and native endpoint guards can erase whole residual branches
without reading their matrices. Live K/V are still generated. If all token/layer
residuals certify over the context bound, token-indexed KV/logit tables preserve
complete declared state. No V^T histories or Q^h input responses are enumerated.
Coverage is restrictive:4/8 synthetic models globally qualify, all high-scale
favorable controls already context-blind in their original finite arithmetic.
Ordinary-scale controls have no native residual identities. Midscale native
identities are missed by conservative certificates; these are different failures.

256 guarded/reference steps,8192 logit values;128 table steps,4096 logit values;
raw KV and own RNG match. No-RoPE custom reference, declared exp ABI assumption;
not a general HF/CUDA equivalence theorem or a formal proof-assistant result.
Query192B does not include cold preparation or state traffic. Derived target
geometry tables96.20GiB and current build~8.12e14 MAC. Universal budgets OPEN.

19 tests;455 run files and2 cost/witness files regenerate exactly. Source capsule
and manifest are persistent; full raw data/development logs are in user archive.
[Prior state unchanged](docs/research/history/pre_residual_absorption_20260907/RESEARCH_STATE.md).
