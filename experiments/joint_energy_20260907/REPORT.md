# Joint moments and deterministic certificates — 2026-09-07

Parent: `e5d7c9ecde971def1245beb1edf2e84c12443dba`, verified draft PR #130.
This is bounded auxiliary E1 research, not a qualifying 405B engine. The fixed mission and CTC-2026-09-05 are unchanged.

## Actual construction

For the declared integer projection, set `P=a*b^T`, retain the exact residual `R=W-P`, and write `S[k,j]=sum_i i^k R[i,j]`. A query computes the moments `c_k=S[k,:]x` from the serialized source and the current input. This is not previous-token reuse, a truth table, a free source, or a perfect selector.

Four exact moments recover a candidate residual with at most two nonzeros. For positions i,j and amplitudes A,B:

```
d = c0*c2-c1*c1 = A*B*(i-j)^2
sigma = (c0*c3-c1*c2)/d = i+j
p = (c1*c3-c2*c2)/d = i*j
A = (c1-j*c0)/(i-j); B=c0-A
```

The implementation handles zero/one entries, cancellation, exact division, isqrt, addresses, and moment checks. It never accepts a candidate just because four moments match.

**Energy mode** stores `G=R^T R` and R, then checks `E=x^T G x-2*ehat^T R x+||ehat||^2=||Rx-ehat||^2`. Only two R rows are used, but the dense Gram is fully charged. An adversarial signed residual `[1,-4,6,-4,1]` has four zero moments; the actual certificate rejects it with E=70.

**Positive-annihilator mode** automatically takes `a_i=min_j W_ij`, `b_j=1`, proves R>=0 at construction, and checks x>=0 at query. No Gram or residual file is stored in this mode. For candidate support J, `q(t)=product_(j in J)(t-j)^2`, so `D=sum_i e_i q(i)` is nonnegative and is zero exactly when all residual entries outside J vanish. For two positions:

```
D = c4-2*sigma*c3+(sigma*sigma+2*p)*c2-2*sigma*p*c1+p*p*c0
```

Together with the recovered amplitudes this is a deterministic exact certificate, not a probabilistic hash. The positive vectors `[1,0,6,0,1]` and `[0,4,0,4,0]` share four moments but not five; the certificate rejects the false candidate with D=24. Without nonnegativity, five moments are unsafe: `[1,-5,10,-10,5,-1]` is a nonzero zero-moment witness.

These use established moment/Prony, polynomial and sum-of-squares ideas; no field-level novelty is claimed. Three proposed premise changes were compared, but `THREE_QUALIFYING_NEW_PRINCIPLES=false`. A cheap general post-rounding moment generator was not constructed.

## Exact limitations and paid budget

When R>=0 and every x_j>0, `(Rx)_i>0` exactly for nonzero rows of R. Success of this s<=2 positive representation then requires rank(W)<=3. Thus rank>=4 matrices with strictly positive dense inputs cannot succeed through this particular path. Full-rank successful controls use sparse inputs; dense-input successful controls deliberately have low rank. Splitting signed x into positive/negative parts can make each part's residual dense even when Rx=0.

For the explicit int64 Gram format, storage/read is `4*n*(n+1)` bytes and the ratio to original BF16 coefficient bytes is `2*(n+1)/m`. Illustrative square/gate/up/fused/down dimensions yield Gram-only ratios 200.01%, 61.54%, 30.77%, 650.01%; these are derived format counts, not GPU or actual checkpoint measurements. Preparing Gram costs O(m*n^2) products. Residual storage is 8mn bytes in addition to the preserved original 2mn bytes.

Positive-mode payload reads are `8m+48n` bytes **only if the stored moment coefficients fit int64**; construction refuses overflow. At large dimensions i^4 moments may exceed int64, so this expression is not a proven target-scale storage bound. Query output construction, original input checks, metadata, prototype selection, integer arithmetic and all preprocessing remain charged. Python big-integer operations are not constant-cost machine instructions. The small constructor takes W in RAM; it is not a tested 405B out-of-core loader.

The native bridge is only exactly BF16-representable integers, FP32 products/leftfold starting +0 and final BF16 RNE. The checked bound `max_i sum_j abs(W_ij) * max_j abs(x_j) <= 2^24` ensures all native intermediates are exact integers. Outside this domain the query refuses. The native witness `[2^24,1,-2^24] dot [1,1,1]` is 1 algebraically but 0 in the declared FP32 order and is refused. Arbitrary BF16 exponents, NaN/Inf, signed-zero ABI, CUDA order, full Transformer consumers, KV and RNG are not closed.

All mission O1-O6 remain OPEN. The local constructor/decoder/certificate proof only discharges limited sub-obligations. No full theory, core admission, public checkpoint inference, hardware speed or broad impossibility theorem is asserted.

## All observed outcomes

| Frozen group | Queries | Exact returns | Scope |
|---|---:|---:|---|
| Energy: planted small residual, dense full-column-rank W | 16 | 16 | Deliberately favorable, not an LLM distribution |
| Energy: unstructured dense | 32 | 0 | No output; all unresolved |
| Positive: 64x64 dense full-rank W | 5 | 3 | Two sparse inputs and zero; positive dense unresolved, negative refused |
| Positive: planted rank<=3 | 8 | 8 | Deliberately low-complexity control |
| Positive: unstructured dense positive | 8 | 0 | No output; all unresolved |
| Aliasing/native-guard witnesses | 2 | 0 | Expected refusal |

Total: 71 queries, 27 exact returns, 44 unresolved/unsupported. All 2,368 returned coordinates agree in both exact integer and native-reference BF16 output. No dense fallback is executed. Zero mismatches does not mean 100% execution coverage.

21 local tests pass. Two runs match all 63 manifest-listed result/packet files, and the manifest itself also matches (64 files including the manifest). Restoring the archive and running again independently reproduced those files and passed 21 tests. Counts are reference validation, not inference throughput evidence.

Initial hypotheses/inputs were frozen before results. A duplicate-key report-assembly bug stopped the first run before its first query; it was fixed without changing frozen inputs. The positive-mode extension has a separate dated preregistration amendment. Full original inputs, raw observations, actual serialized packets, logs and hashes are retained.

## Restore and reproduce

The eight `part-*.b64` files losslessly contain **78 research files**, including `src/source.py`, `src/run.py`, tests, both preregistrations, full input matrices, the complete Korean proof/cost report, raw results and every serialized packet. Archival compression is not the proposed inference codec. No restored code is executed automatically.

```
python experiments/joint_energy_20260907/restore.py --out /tmp/vortex-joint-energy
cd /tmp/vortex-joint-energy
python -m unittest discover -s tests -v
python src/run.py --out results/reproduced
```

Compare SHA256 values in `results/run1/manifest.json` with `results/reproduced`. The detailed O1-O6 ledger, dependency list, proofs and accounting are in restored `docs/REPORT_KO.md`; the original local validation is `results/validation.json`. The complete archive SHA256 is `a910186456c5b5a0dca60d2ec3ef05e7a518a41055ede7520c1d9a8368da7c7e` and is checked before writing anything. Existing differing files are never overwritten.

## Handoff and next construction

Do not expand Gram matrices, increase the Prony sparsity parameter, or port this fixed path to GPU without a new information source defeating the dense-rank/sign/native gates. The primary task remains a uniform cheap native causal constructor/query/state algorithm and a sufficient whole-target budget. Cheap acceptance on planted controls is not a route to that theorem.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`. Remote persistence must be verified by reading the actual branch/commit; it is not proof acceptance. No Actions, full-repository tests, main writes, merges or force-pushes are requested.

Primary background consulted: Bennett et al., *Matrix Multiplication Verification Using Coding Theory*, arXiv:2309.16176v2 (coding verification under explicit assumptions, not this native executor); NVIDIA *Floating Point and IEEE 754* documentation (execution-order scope). A 2026 sparse multiplication abstract was screened, not reproduced. All claims about this algorithm are derived here and scoped to its declared arithmetic.
