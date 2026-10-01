# Paid causal Krylov feedback construction: registered finite check

Date: 2026-10-01. Analysis and source reading preceded this plan. No candidate
numerical results have been inspected. Parent owns root ledgers and publication.
Original/dirty replay files remain untouched.

## Fixed final theorem and obligations

The final target is arbitrary released unmodified dense HF 405B, batch one,
original exposed outputs/RNG/required continuation state, one GPU total peak
<=8 GiB, same-machine native 4B Q4 p50<=1.2x and p95<=1.5x and unchanged TTFT.
All costs count. O1-O6 remain OPEN. This bounded task addresses a constructive
O2/O3/O4 counterexample to the assumption that independently queried matrices
are unavoidable, not an HF lift or complete resource bound.

## Three distinct principles compared before implementation

1. Temporal reachable coordinates with a small nonlinear feedback port. Build
   K=[b,Ab,...] by finite linear algebra on actual source A; retain z with s=Kz.
   A companion update and transported observers avoid repeated dense A reads.
   Exactness: AK=KJ, b=Ke0, cK=d. Construction is polynomial and fully paid.
   >=10x route: one constructor plus T*O(n) instead of T*O(n^2), with a finite
   break-even horizon. Native closure and narrow feedback/observer rank are
   essential, not given for Transformers. Selected for one bounded source test.
2. Whole-transition joint-relation compilation. Exact native relations preserve
   arbitrary rounding. Current literal constructor enumerates joint boundaries;
   input substitution restores original dense work. Prior whole-program report
   controls; do not repeat table/variable-order experiments.
3. Native suffix elimination. A paid narrow invariant might delete history.
   Existing prefix-norm/suffix source retains wide intervals or dense bounds;
   no new interval source is present. Prior suffix/deferred-state reports control;
   do not expand positive controls or suffix lengths.

These are a comparison of the current frontier, not three new qualifying cores.
Known Krylov/control realization algebra is not claimed novel.

## Registered primitive, scope and exactness

All arithmetic state is F2^n, n<=64. Source A is arbitrary binary, b,c1,c2,co
are arbitrary binary vectors. The original supplied native program implements
A*s and ci*s using dense NONZERO BF16 coefficients 1+Aij or 1+cij, sequential
FP32 multiply/add from +0, BF16 store, subtract popcount(s), then integer parity.
Sums are integers <=128; all FP32/BF16 operations are exact. This includes a
parity operation in the ORIGINAL program: it is not ordinary HF MatVec.

Original transition:
 g=(c1*s mod2) AND (c2*s mod2)
 s'=A*s XOR b*(a XOR g)
 y=co*s' mod2
 exposed logits=[BF16(y), BF16(1-y)]
 one uint32 LCG draw; Bernoulli threshold is 1/4 for y=0 and 3/4 for y=1.

Initial state s=0 and supplied RNG. Inputs are arbitrary external bits or the
candidate's own prior sampled output. Candidate keeps z and exactly the RNG.
Original-state readback is supported through Kz and fully charged separately;
calling it on every token is not hidden in the fast observer-only interface.

Native C reference and independently written Python packed source/compiler are
compared. No state/input enumeration, cached answers, future states, checkpoint
or GPU run. Numerical scope cannot be lifted to HF by name.

## Frozen checks and rejection thresholds

Deterministically generate sources with Python Random seed 20261001. No source
selection/retry for favorable Krylov rank. Cases: random n=32, random n=64,
identity n=64, and random n=64 with b=0. Register 512 externally supplied bits
per case, then 512 own-output feedback steps from fresh zero state/RNG.

Require exact every-step original state (through separately charged decode),
logit bits, token, RNG word and input agreement; source-only serialized runtime;
AK=KJ, b=Ke0 and three transported observer identities by direct checks.
Include one deliberate bad recurrence coefficient to require a mismatch witness
when r>0; no finding of such a witness is assumed for a dead/nonobservable source.

Record full source bytes, hot/cold encoding bytes, constructor operations and
word reads/writes, per-step operations, explicit decode operations, causal traces,
source hashes, environment and reproducibility. Operation inventory is not timing.
One random source also receives a gate-removal ablation to show the nonlinear
feedback actually changes reachable state; report the result without filtering.

A second symbolic/native witness checks that ordinary BF16 rounding is not the
field law: BF16(256+1)=256, versus the exact integer 257. The native program with
parity above avoids this only on its declared <=128 sums.

No scale/latency/GPU/HF claim follows; stop this test after checking construction.
Continue toward the necessary native innovation relation rather than larger toys.
