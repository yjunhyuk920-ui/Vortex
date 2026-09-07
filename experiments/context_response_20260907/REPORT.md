# Native context-response source and deferred zero tags — 2026-09-07

Parent: PR #134, `e3f8b3e466075c7fdedcac8c25a2ef7dfab5472a`.
Fixed mission and CTC-2026-09-05 unchanged. This is a bounded attention-state/source lemma,
not a generic dense405B executor or a claim of three new qualifying principles.

`THEORY_STATUS=NOT_ESTABLISHED`, `HARDWARE_STATUS=NOT_TESTED`, `CORE_ADMISSION=false`,
`FULL_MISSION_O1_O6=OPEN`, `THREE_QUALIFYING_NEW_PRINCIPLES=false`.

## Constructive result and exact scope

Compared fixed-feature response aggregation, eager all-query response updates, and a
checked native-zero index. The whole-work gate was applied before a large backend:
all dense projections remain. Only a bounded source/numerical reference was built.

For the declared d=16 BF16 sign-code keys (components +/-1) and queries (+/-512),
QK/4 = 2048 - 256*Hamming(code_q,code_k). When a matching key exists, stable FP32
exp weights are 1 on matches and exactly +0 elsewhere, not approximated small values.
The index appends checked KV, retains original positions/values and a sign-AND tree,
and visits only matching leaves of the ORIGINAL balanced FP32 PV reduction tree.
Deferred zero-sign masks compose by AND; they materialize before real additions or
at the root. This preserves -0 and the original grouping rather than compacting a mean.
Generator, actual serialized state, query addresses, guards, load verification,
state reconstruction, and counters are fully specified in the capsule sources/report.

Finite BF16 values, width<=64, power-of-two capacity<=2^20, matching query required;
separate FP32 RNE operations, balanced reduction with +0 padding, BF16 output.
Other geometries are rejected without modifying them. This is not arbitrary HF/RoPE,
CUDA/SFU/FlashAttention, full Transformer/RNG or full-mission state equivalence.

## Evidence and paid costs

32 contexts /256 static queries /4096 coordinates, plus384 online append/query steps
/6144 coordinates: FP32 and BF16 mismatch0. Raw KV reconstructs in original order.
16 generic or perturbed keys are rejected, not counted as successful execution.
16 unit tests pass, including independent scalar-struct oracle and all65280 finite
BF16 values for the zero-tag rule. Same selected values at positions [0,1,2,4] vs
[0,2,3,4] yield native0.25 vs0.5, so unordered bucket mean is not exact.

At T1024,d=dv16, favourable alphabet256 groups have query logical-byte p50
6.9641%/7.7942%, p95 12.5851%/11.2961% of one raw BF16 KV read. Actual state files
57880B vs65536B: not tenfold state compression. All-equal keys cost646.60%.
Initial eager-zero implementation costs17.12%/17.92% p50 on the SAME inputs and
is preserved with its complete results. Init+append+query on128/256-step streams
cost78.32%/62.99%; a stated cold build/save/reload+8-query workload costs3.42x/3.44x.
Byte accounting includes directory shifts/search, positions, sign metadata, control
records and explicit temporary operands; NOT measured DRAM/GPU traffic, latency or
complete Python heap/target memory. Eight-query quantiles are descriptive only.

Using the cited PR134 bf16-mp8 dimensions, unchanged projection work is
403747897344 MAC. Even free QK/PV leaves95.9798% of counted MAC at context4096.
Attention-only10x in that optimistic MAC model requires T>=880101. This is a scoped
coverage gate, NOT a measured latency lower bound or universal impossibility theorem.

Native response identity matrices show exact linear features can need rank>=T;
state counting also need not imply full-state reads per query. The actual index
retrieves one row on that same identity family. Do not conflate storage and traffic.

## Reproduction and archival integrity

The capsule contains20 full UTF-8 files: all sources/tests/preregistrations, full
Korean proof/cost report, both-stage summaries and manifest anchors. XZ/base64 is
archival encoding only. Original binary arrays/state/per-query traces are in the
user ZIP; they are regenerated to the original manifest anchors, not all embedded.
Both stages regenerate241 files each with byte-identical manifests. Restoring this
capsule, rerunning16 tests and regenerating BOTH stages also matched those anchors.
No Actions, full-repository suite, public-model, GPU/405B/native4BQ4/TTFT test was run.

```bash
python experiments/context_response_20260907/restore.py --out /tmp/vortex-response
cd /tmp/vortex-response
python -m unittest discover -s tests -v
python src/run.py --out regenerated/final
python src/replay_eager.py --out regenerated/eager
python - <<'PY'
import hashlib,json
from pathlib import Path
expected=json.loads(Path('results/expected_manifest_hashes.json').read_text())
actual={s:hashlib.sha256((Path('regenerated')/s/'MANIFEST.json').read_bytes()).hexdigest() for s in expected}
assert actual==expected,(actual,expected)
print('Both 241-file generated manifests match original anchors')
PY
```

## Obligations and next decision

A bounded append/query/state correspondence and native zero-tag algorithm are
constructed. Full O1-O6 remain OPEN. Do not expand code-key tables, feature rank,
context length or this attention backend as a proxy for dense-projection elimination.
A new core direction needs a concrete current-input-dependent source for the costly
projections, including construction/state/numeric/traffic costs and a credible
whole-work>=10x path. This lemma rejects neither all encodings nor the mission.
README/state/next/validation are current; exact prior root blobs are retained under
`docs/research/history/pre_context_response_20260907/`. Other policies/evidence/root
ledgers are preserved; this report and its full capsule report are the new addendum.
Remote persistence is established only by commit/ref/PR/content read-back.
