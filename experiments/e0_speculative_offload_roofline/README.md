# E0 speculative-offload dual-roofline audit

This deterministic model-free audit computes the most favorable simultaneous
I/O and dense-arithmetic lower bound for verifying speculative candidate nodes
with an exact offloaded 405B target.

It intentionally grants perfect transfer/compute overlap and every omitted
runtime cost for free. A failed inequality is decisive; a passing row is not a
runtime or hardware claim.

Run:

```bash
python experiments/e0_speculative_offload_roofline/run_audit.py \
  --config experiments/e0_speculative_offload_roofline/config.json \
  --output /tmp/summary.json
```
