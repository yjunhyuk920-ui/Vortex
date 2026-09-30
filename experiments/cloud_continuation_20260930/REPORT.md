# Cloud continuation: executable baseline and control-classification repair

## Outcome

The previous native experiment can now be inspected and its 14 control tests run
in this independent Linux CPU environment. All 3,510 frozen canonical files retain
their original comparison hashes. The pinned 269,060,552-byte public checkpoint was
downloaded and matched SHA256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
This is not a new full-generation replay, speedup, native-ABI portability theorem
or target hardware measurement.

An actual checked-in EXP100A runner defect was corrected: an empty frontier after
a bounded resource-filtered search is no longer automatically a corrupt control.
Missing, unexpected and mismatched search identities still fail closed. This
implements the substance of the previously stored repair workflow, strengthening
its count-only population check to exact key-set and result-identity checks.
The focused suite passed 16 tests, including five new edge-case controls.

The recorded EXP100A metadata contains all 50 expected searches, one resource-empty
direct frontier and no oracle-empty frontier. No recorded direct or favorable
oracle block passed the joint 10x gate. This audit does **not** re-execute searches,
revalidate the complete factorization catalogue, change the old INVALID decision,
or authorize a scientific promotion. The historical result bytes remain unchanged.

THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
HANDOFF_STATUS=IN_PROGRESS
CORE_ADMISSION=false
THREE_QUALIFYING_NEW_PRINCIPLES=false
README_CURRENT=true

## Why the repair matters and what it does not solve

The previous head contains an unapplied classification patch in
`.github/workflows/repair-exp100a-control-classification.yml`, but its runner still
uses `empty_direct_shape_search` as an integrity error. The repeated "repair then
rerun" frontier can therefore be mistaken for untested positive FMM evidence.
Separating search coverage from resource exhaustion prevents that mistake.

The native numerical producer is still missing. Correct bookkeeping supplies no
cheap causal block, no native-repair selector and no finite-word transform closure.
The old recorded failure also does not prove that every rectangular multiplication
algorithm fails. Its catalogue, beam, workspace representation and oracle grants
remain the exact scope of that bounded result.

## Reconciled research frontier

- Ordered coefficient-subtree sharing was already implemented, including mirrored
  signs, in `experiments/signed_orbit_20260907/REPORT.md`; row traversal and ancestor
  repairs also exist in `experiments/native_row_frontier_20260908/REPORT.md`.
  Ordinary subtree hashing is not a new native producer.
- EXP096A's target-seeing capacity result must be read with
  `docs/research/EXP_097A_LATEST_RESULT.md`, which reports lack of specificity.
  No causal coefficient generator follows from the older promotion label.
- `docs/research/EXP_099A_LATEST_RESULT.md` observes sparse disagreement with an
  exact-real proxy using a perfect official-match selector. Its implementation
  computes the official answer before deciding a lock; it is not a cheap online
  certificate. A generic FP32 bound certifies only roughly 44–46%, not 99.94%.
- PR149 graph/state/demand constructors preserve their bounded native contract but
  retain every dominant matrix MAC. Enlarging their graphs is not the next core.
- Globally nonlinear bounded-word numerical data structures remain unclosed.
  Existing linear/semiring/table rejections are not a universal impossibility
  theorem; no unproved selector or compression assumption is admitted here.

O1–O5 remain open. O6 has a stronger local audit/control path for this limited
record, not a completed mission-wide proof. A new core round still needs a concrete
finite native producer and a genuinely different three-way comparison. No such
qualifying trio is claimed in this supporting round.

## Environment and paid work

Linux x86_64, Python 3.12.14, visible CPU affinity 9, visible RAM about 9.7 GiB,
no CUDA device. Exact package inventory is `requirements.observed.txt`; the core
versions are torch 2.8.0+cpu, transformers 4.55.4, numpy 2.2.6 and safetensors 0.6.2.
The earlier reference is Windows/Python 3.12.10. Transitive dependencies differ;
bitwise cross-platform full-generation equivalence is not inferred.

Dependencies and model files were fetched into ignored local environments/caches.
No user desktop, target GPU or hosted Actions experiment was used for this round.
The changed EXP100A runner was compiled, the 16 focused tests passed in 0.28 s and
the unchanged 14 transition tests passed in 0.912 s on the observed runs. Those
durations describe tests only, never LLM inference latency. Full repository suite,
405B, 8-GiB peak, TTFT and same-machine native-4B latency are NOT TESTED.

## Reproduction

From repository root, in the recorded environment:

```sh
python -m pytest -q tests/exp_100a
python experiments/cloud_continuation_20260930/audit_exp100a.py
cd experiments/native_global_transition_20260908
python -m unittest -v test_transition
```

`results/exp100a_metadata_audit.json` is a new metadata-only audit; it explicitly
names the immutable historical source SHA256. `results/canonical_integrity.json`
records frozen-file verification; that verification did not execute the model.
Original weights, raw historical outputs, search config and scientific thresholds
were not modified. New experiments must preserve these scope distinctions.
