# Correlated-input SwiGLU: constructive normal form and native/cost gate

2026-09-07. Parent: PR #135, `d9e19c5ab060488c8788acbcfabc4d0933870aa3`.
Full Korean proof, all sources, tests, preregistration and raw summaries are in the archival capsule. This is not a complete executor or an academic novelty claim.

## Constructed result
For real F(x)=D[phi(Gx)*Ux], phi(t)=t/(1+exp(-t)), group gate rows by exact equality up to sign only. With representative g, construct A_g=sum d_j u_j^T over both signs and B_g=sum d_j u_j^T over negative rows. Then

    F(x)=sum_g phi(g.x)(A_g x) + Q(x)
    Q(x)=-sum_g (g.x)(B_g x).

The constructor, rational coefficients, serialized format and interpreter exist; original G/U/D are absent from the transformed interpreter. It is an exact-real formula with a high-precision numerical sanity interpreter, NOT a bit-exact native executor. No ideal exact-exp primitive is counted as free.

A Taylor/Vandermonde proof shows: for nonzero pairwise non-antipodal real gates, a sum of these linear-amplitude SiLU ridges is a finite-degree polynomial on all real inputs only when every amplitude vanishes. This closes the independent-input gap within this explicit real representation grammar. It is NOT a universal read/time lower bound, a finite-word impossibility result, or a proof that independent ridge evaluation is required.

24 generic registered cases retain all192 groups;12 engineered correlated cases eliminate all nonlinear groups. Exact rational jets:11520 matching coefficients. High-precision real checks:576 outputs, not native equivalence.

## Native witness and finite-domain distinction
G=(1,1,1), U=(1,2,-3), D=(1,1,1) defines an identically zero real function. At x=1, the declared BF16/FP32 ABI instead gives1/256. Independently certified activation187/256 and products187/256,187/128,-35/16 establish it. C and NumPy agree for1025 inputs;736 are nonzero despite analytic cancellation. Never use this real rewrite as an admitted native path.

Conversely a degree4 rational polynomial exactly interpolates five native inputs; it fails off-grid. Thus the real-domain theorem cannot exclude all finite-native representations. Constructing an explicit multidimensional grid costs Q**n native evaluations in the direct implementation.

The second candidate, multiplicative exponential coordinates, has128 exact rational outputs but visits all exponent entries and retains projection work. Arbitrary-precision powers are charged, not single scalar operations.

## Costs and scope
Materialized generic amplitudes require f*o*n coefficients versus f*(2*n+o) original coefficients. At n=o=16384,f=53248 this is14293651161088 versus2617245696, a5461.333... ratio. Factoring the outer products avoids expansion but the explicit query returns to original work. This is a cost of the constructed formats, not a lower bound on every algorithm. Constructor RAM/precision, source reads, serialization and future native repair remain charged.

THEORY_STATUS=NOT_ESTABLISHED; CORE_ADMISSION=false; HARDWARE_STATUS=NOT_TESTED; O1-O6=OPEN; THREE_QUALIFYING_NEW_PRINCIPLES=false.
No pretrained checkpoint bytes were acquired: DNS failure and an unavailable web API were recorded, not bypassed. No public-model execution, Transformer KV/RNG, CUDA,405B,8GiB,4BQ4 latency/TTFT, Actions or full repository suite was run.

## Evidence and reproducibility
23 local tests pass.152 generated files plus their manifest reproduce byte-exactly. Capsule restoration, C rebuild, tests and replay were independently repeated locally. The first capsule replay omitted --lib and produced null C fields; the documented invocation below restores the original hashes. No scientific outputs were altered.

The capsule contains25 UTF-8 text files, including all source/tests/full Korean report and the original manifest digest. Per-case raw traces are regenerated against that digest; they are not all embedded remotely. The full user ZIP preserves original and exploratory data. XZ/base64 is archival encoding, NOT an inference codec or compression result.

Preserved deviations: initial native.so shadowed native.py before results; rename to libwitness.so; initial four-triple exploratory group is kept separately from the registered one-triple run; actual power-call accounting replaces an optimistic schedule. No exploratory samples are added to confirmatory counts.

```bash
python experiments/correlated_ridge_20260907/restore.py --out /tmp/vortex-correlated-ridge
cd /tmp/vortex-correlated-ridge
cc -O2 -std=c11 -fPIC -shared -fno-fast-math -ffp-contract=off src/native.c -lm -o src/libwitness.so
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python src/run.py --out regenerated/run --lib src/libwitness.so
python src/verify.py regenerated/run
```

See [ledger](LEDGER.md) for scoped obligations and next action. Prior root policies and evidence remain intact; replaced entrypoints are preserved by their exact old Git blobs under docs/research/history/pre_correlated_ridge_20260907. Remote completion requires branch/commit/file read-back, not this text.
