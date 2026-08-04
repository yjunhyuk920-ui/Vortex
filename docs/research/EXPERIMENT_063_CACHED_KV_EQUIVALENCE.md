# EXP-063 — Pinned Causal Exact Cached-KV Equivalence Reuse Gate

Authority: workflow `30846082964`, source `979bde3a23b76270f740740fbf511c7f90900a7c`, merge `488fa0e3785885bbcea25681aae55bb361fa0f84`, artifact `8868770832`, ZIP SHA-256 `b900a7019d8527d6f67d0eb412bb2fb7a0331188d84cd74444ca10762a105a14`.

MEASURED: 3 models; 18 cases; 1,152 forwards; 147,456 group rows; exact K duplicates 0; exact KV duplicates 0; mismatches 0; warm p50/p90 work 100.021%/100.027%; bytes 106.263%/119.401%; peak RSS 1,087,700 KiB.

Decision:

```text
REJECT_CAUSAL_EXACT_KV_EQUIVALENCE_REUSE_AS_CORE_RETAIN_KV_AUXILIARY
```

Exact cached-KV equivalence reuse is rejected for this population. Physical kernels, 405B, 8 GiB and target hardware were not tested.
