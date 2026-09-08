# Research state — 2026-09-08

Fixed mission and CTC unchanged. [Current bounded construction](experiments/native_row_frontier_20260908/REPORT.md).
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
CORE_ADMISSION=false
FULL_MISSION_O1_O6=OPEN
THREE_QUALIFYING_NEW_PRINCIPLES=false
README_CURRENT=true

HARDWARE_STATUS concerns target hardware; limited CURRENT CPU reference-kernel timing was measured and is separately scoped.

Constructed original-BF16 row edit source, exact subtree-signature metric D=4H+12A, Prim/preorder <=2 metric-tour guarantee, serialized C query reusing one native heap across rows. Finite BF16 input vectors unrestricted within declared fixed-balanced/canonical-NaN ABI. No input bank/previous input/good-output oracle/refusal-on-poor-structure. Original information preserved in source. All360queries across18matrices return80640coordinates exactly;378Fraction checks. Unit suite14includes all65280finite scalar inputs at4weights, not all vector inputs.

512coherent source7748-9032B,1%corrupt31204-31556B,random1048740-1048876B vs524288B original. FPwork1.26-1.36%,8.14-8.21%,~99.86%. Countermodel to generic speedup remains: random data consumes nearly all coefficients plus addresses. Native heap logical traffic and constructor O(m^2 N) are charged. Each512case counts268173312 signature-word comparisons. Fullbody/coherence not established in real models.

Post-hoc CPU batch1 handwritten-C median ratios: coherent0.0486-0.0505;1%corrupt0.1292-0.1301;random1.5880-1.6186. Warm/resident source, no compile/load, no optimized baseline/GPU/4B claim. 147sciencefiles regenerate to e3018ea424d0e3e2b6b91f20bc3f579eabc33e3f1df6635762d2ad17081eec42. CPU time is preserved observation, not deterministic replay target.

Projection local numeric proof only; HF fullcausal state/KV/RNG/targetcost obligations OPEN. Prior local native_cover is not silently committed in this round. Previous deferred-state results preserved unchanged.
[Prior state](docs/research/history/pre_native_row_frontier_20260908/RESEARCH_STATE.md).
