# Primary response to independent review

The Sol/high review is a static review of historical v2/v3 and the then-current
source; its findings are retained rather than retroactively rewritten as a v5
review. Primary inspected each finding and made the following bounded updates.

- Executed-source binding: v5's manifest hashes hf_bridge.py, scalar_producer.py
  and PREREGISTRATION.md before the native run. It records invocation, OS/CPU
  capability, relevant nonsecret thread/ISA variables, Python optimization and
  Torch build. The package manifest also hashes all retained code/results.
- Generation and prefill logits: v4 actually executed the additional checks;
  v5 repeats them with the source binding. V3's combined field never proved
  generation logits; only its recorded probabilities/samples/RNG receive credit.
- Cache state: v5 adds storage offsets/sizes, tensor view bits, tuple/list tags,
  qualified cache types and pairwise internal storage aliases. Machine fields
  now say serialized fields, not every possible state observable. Arbitrary
  caller-owned aliases, external cache mutations and complete HF API equivalence
  remain unproved.
- Prefill: K/V word comparison remains explicitly named as such. Prefill logits
  are separately checked. Prefill/incremental strides are not asserted equal.
- Norm validation: replace Python assert with an explicit ValueError, preserving
  validation under Python optimization. No fixture/threshold alteration.
- Work: the report includes whole source residency/hash/initialization,
  transpose and native compile work, Python representation/temporary overhead,
  separate cache reads/writes and cumulative quadratic copying, input/output,
  sampler/RNG, validation, abort/restart and target movement terms. Physical
  allocator/device/latency upper bounds remain OPEN rather than omitted as free.
- Recovery: no transactional guarantee was implemented. A partial cache after
  failure is invalid; a restart needs a paid reconstruction. This cannot close
  the mission's recovery obligation.
- Causal scope: prior V is stored but does not affect the fixture's zero logits.
  Nontrivial attention-dependent output preservation remains O2/O3 work.
- Runtime guard: the monkeypatch blocks Linear/model forwards only. Static
  inspection of the actual compiled path finds no functional linear/matmul
  escape; the monkeypatch alone is not a general proof against every dense API.

The final v5 run passed its explicit checks. No review finding was resolved by
model agreement or by silently promoting a finite trace to a general proof.
O1-O5 OPEN, O6 PARTIAL; target hardware NOT_TESTED.
