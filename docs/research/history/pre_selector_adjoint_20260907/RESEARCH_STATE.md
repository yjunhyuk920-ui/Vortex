# Research state — 2026-09-07

Fixed mission / CTC-2026-09-05 unchanged.
[Current bounded source, obligations, decisions and cost](experiments/dense_residue_20260907/docs/REPORT_KO.md).

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false

Constructed dense integer correction by interval/residue lift, plus a finite serialized bitplane source from W and current input. No sparse correction-support assumption or free residue oracle. Strict integer BF16/ternary input and positive-row guard; exact FP32 intermediate bound; final BF16 RNE. This is not general native floating arithmetic or a Transformer executor.

18 matrices288queries19968coordinates match exact integer/C references.91 queries have nonzero correction at every output.12 nonconstant matrices have full modular rank. Dense128 source51.60–51.70% BF16, but117.94–118.16% of signed-bitpacked original payload. Small-residual and constant-row controls are separately labelled. Every residual plane is read. The coefficient-recovery proof is not a universal memory-probe lower bound.

Residue-only state fails ReLU/SiLU witnesses. Restricted syndrome-only decoding cannot distinguish an arbitrary ternary error cube with the tested short code; this does not constrain decoders with extra input-dependent computation.

16 tests/111 generated science files reproduce to a frozen manifest. Code, preregistration, expected hash, proof and summary are stored directly; raw inputs/programs/traces and local history/logs are in the user ZIP and regenerate, not all embedded in Git. Full mission and target measurements remain open.
[Prior state unchanged](docs/research/history/pre_dense_residue_20260907/RESEARCH_STATE.md).
