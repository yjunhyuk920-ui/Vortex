# VORTEX

A research runtime for public, unmodified Hugging Face dense Transformers.
The fixed mission is a real dense 405B, executor replacement only, original
output/RNG and required successor state, one GPU with total peak <=8 GiB,
and same-machine native 4B Q4 warm p50 <=1.2x and p95 <=1.5x.
The final target has not been achieved.

## Constructive-theory-first governance — 2026-09-05

Read [AGENTS.md](AGENTS.md), [the canonical mission](MISSION_AND_WORKING_PRINCIPLES.md)
and [Constructive Theory Contract](docs/CONSTRUCTIVE_THEORY_CONTRACT.md) first.
The primary deliverable is a finite execution algorithm with original-contract
proofs and sufficient full-cost upper bounds that close the final theorem.
Organize work by O1-O6 and compare three materially different principles before
concentrating on the strongest credible >=10x construction. A small lemma,
rejection, test pass or commit is not scientific completion.

Theory acceptance (`VERIFIED_IN_MODEL`), actual hardware acceptance (`E7_VERIFIED`)
and repository persistence (`REMOTE_COMMIT_VERIFIED`) are independent. This
policy update establishes no new theory, model run or performance result.
See [the decision record](docs/governance/CTC_20260905_DECISION.md) and
[the theorem-first next work](NEXT_EXPERIMENT.md).

## Branch and evidence scope

This is a policy-only update to main, based at
`fc780350f1284f4aea3146fa1d79f6b4ed046142`. It does not merge the unmerged
experimental lineage or its code/results. Main's unchanged scientific ledgers
remain historical branch evidence, not the newest research-wide frontier.
Before research, resolve the actual remote branches, PRs and committed evidence.

The latest branch inspected for this update was
[PR #119](https://github.com/yjunhyuk920-ui/Vortex/pull/119), pinned at
`3b6eb2bc7bddeb00e658a0d265ff9b751461ea55`. Its scoped nonlinear-decoder result
is E1 auxiliary, not an implemented 405B fast executor. Its tests were not rerun
for this governance-only change. The updated research branch retains its own
scientific entry points and evidence.

Older main prototype descriptions, exact reported test counts and command
examples are retained without alteration in
[the original main README](docs/governance/history/pre_ctc_20260905/main/README.md).
They are historical evidence, not fresh validation or full-goal achievement.

## Local-first work and basic setup

```text
LOCAL_RESEARCH -> LOCAL_VALIDATION_PASS -> COMMIT_PUSHED -> REMOTE_COMMIT_VERIFIED
GITHUB_ACTIONS_REQUIRED=false
GITHUB_REEXECUTION_REQUIRED=false
```

Research and applicable validation run locally. Actions is only used when
explicitly requested; branch protection is not bypassed. Persist complete
evidence to a research branch and read back the remote SHA. Do not directly push
main, force-push, or merge unrelated experiments. Missing hardware remains
NOT TESTED. Review README and affected ledgers on every meaningful round.

Existing prototype setup (not a claim these runtime checks were run in this edit):

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
python scripts/run_validation.py
```

`vortex_runtime/` contains prototype runtime code, `experiments/` experiments,
`tests/` tests, `results/` evidence, and `docs/` contracts/research/handoffs.
Unchanged runtime and experimental trees are not promoted by the policy update.
The new governance validation is scoped to Markdown, links, policy consistency
and source/content hashes; no public-checkpoint or hardware benchmark ran.

```text
THEORY_STATUS=NOT_ESTABLISHED
HARDWARE_STATUS=NOT_TESTED
README_CURRENT=true
README_UPDATED=constructive theory contract; branch/evidence scope; next work; local-first policy
```
