# Validation matrix — 2026-09-08

[Report/replay](experiments/deferred_state_20260908/REPORT.md).
[Prior matrix unchanged](docs/research/history/pre_deferred_state_20260908/VALIDATION_MATRIX.md).

| Item | Evidence and exact scope |
|---|---|
| Source | Actual BF16 coefficient/norm generation; frozen CPU unary ABI; original G/U/D files retained |
| Lazy runtime | Exact expression+sound native range; same one-word Bernoulli RNG rule; state-bit-only history deletion |
| Initial corpus |18syntheticmodels/36runs/4608steps/86016statecoordinates;0token,RNG,statebound/exactstate mismatch |
| Debt |F+Z+P=T; exact flushF+Z=T; tiny deferred128allreplayed; moderate/wide127/128body afterflush |
| Tail |Example perdecision bodycount p50=0,p95=6,max9; not measured time/token |
| Ablation |Tiny nonlinear contribution changes0/14336statecoordinates; moderate36;wide12743 |
| Suffix |Verified invariant B then native endpoint transport;12initial modelsfirsttested16;6wide refuse B |
| Continuation |12recovered states/384steps originalbodyC vs Python agree; not acceleration |
| Active follow-on |6labeledposthocmodels,1536lazy steps; nonlinear effect14224/14336;5suffix successesat16,onefails through128 |
| Suffix cost |Fullweightread/2xlinearproducts perintervalstep;attemptedlengths total255 plus2invariantpasses |
| Memory |1703424Bshared unary/index payload; pending arrays8*n*P plus objects; rawweights remain; tracemallocpartial/nondeterministic |
| Scope |Contraction with2nativefixedpoints; same-token/non-equivalentstate;literalappend-onlyprefix distinction only |
| Replay |17tests/152deterministicfiles;manifest49098e4052a39dfa942c980696d838161f81b32bc87a4df1a57cffae61a43c21;freshZIP/C rebuild match |
| Full mission |O1-O6OPEN;CORE_ADMISSION=false;THEORY_STATUS=NOT_ESTABLISHED;HARDWARE_STATUS=NOT_TESTED |
| HF/KV/categorical sampler/CUDA/405B/8GiB/4BQ4/TTFT |NOT TESTED / NOT CONSTRUCTED |
| Actions/fullrepositorysuite |NOT RUN |

Originalprereg andlaterfollow-on/activeplans separated. Independent C covers arithmetic/order,not independently computed SiLU/sigmoid. Exact-force/replay is charged safety behavior, not target success. Source/fullKoreanreport/frozenmanifest are in checked capsule; rawscience/logs in userZIP and deterministic science regenerates.
