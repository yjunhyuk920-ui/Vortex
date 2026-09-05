# VORTEX repository commit and handoff mandate

This mandate has the same authority as `AGENTS.md`, `MISSION_AND_WORKING_PRINCIPLES.md`, the proof-first contract, and the research-efficiency contract.

## 1. Completion definition

A research round is complete only when:

```text
meaningful change exists
local/sandbox research is finished
local validation is recorded
source/result/docs are committed
commit is present on a real remote VORTEX research branch
remote SHA is read back and verified
required ledgers and README are current
provenance is truthful
```

A local-only result is useful research but not a completed repository handoff.

Only a pushed commit whose remote SHA is read back may be classified:

```text
REMOTE_COMMIT_VERIFIED
```

## 2. Local research and validation are authoritative

The default path is:

```text
LOCAL_RESEARCH
-> LOCAL_VALIDATION_PASS
-> COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Use the local/sandbox environment for:

- derivations and exact arithmetic;
- cost/roofline calculations;
- tensor/circuit/program/combinatorial search;
- prototypes and rapid repair;
- exhaustive/random/adversarial controls;
- unit/property/regression tests;
- available public-checkpoint execution;
- raw/processed evidence, logs, and checksums.

The locally executed validation is the scientific validation for the round when the required experiment is runnable there.

If some target-hardware fact cannot be measured locally, record it as `NOT TESTED`. Do not replace it with a free grant or unrelated hardware measurement.

## 3. GitHub is commit-and-handoff only after local validation

Once `LOCAL_VALIDATION_PASS` is established, GitHub is used for persistence and cross-session handoff:

- commit/push source, configs, dependency pins;
- commit raw/processed results, logs, checksums;
- update decisions, failures, state, next experiment, architecture, reproducibility as applicable;
- update README;
- read back the remote branch head and commit SHA.

Do **not** rerun the already validated experiment merely because it was pushed to GitHub.

Default policy:

```text
GITHUB_ACTIONS_REQUIRED=false
GITHUB_REEXECUTION_REQUIRED=false
CLEAN_ROOM_RERUN_REQUIRED=false
HOSTED_CHECKPOINT_RERUN_REQUIRED=false
```

GitHub Actions may be used only when the user explicitly asks for an additional hosted run. Existing automatic CI may run, but it is not a mandatory scientific approval Gate after local validation.

The following workflow is therefore prohibited as a normal requirement:

```text
LOCAL_VALIDATION_PASS
-> GitHub Actions executes the same science again
-> wait for hosted result
-> only then call the round complete
```

Use instead:

```text
LOCAL_VALIDATION_PASS
-> commit/push validated state
-> remote SHA read-back
-> REMOTE_COMMIT_VERIFIED
```

## 4. Connector-first repository writes

For persistence, use the available writer in this order:

1. connected GitHub connector/app;
2. configured local MCP Git writer;
3. authenticated Git/GitHub CLI;
4. local commit + patch/bundle only when all remote writers fail.

A local DNS or shell Git failure does not prove the connected writer is unavailable.

## 5. Infrastructure failure is not scientific evidence

Missing GPU, package, DNS, checkpoint cache, or target hardware is an infrastructure fact.

- continue all local theory/search/testing that remains possible;
- record unavailable target measurements as `NOT TESTED`;
- never fabricate a run;
- never convert non-execution into a scientific rejection.

## 6. Branch and history policy

- Never push directly to `main`/default branch.
- Never force-push or rewrite another research branch.
- Start from a verified remote branch/commit SHA.
- Use descriptive `research/*` branches.
- Preserve prior positive and negative evidence.
- Do not commit secrets, credentials, private checkpoint data, or large upstream weights.

## 7. Startup sequence

Before new research:

1. verify repository, active branch, remote head SHA, latest open PR, and write access;
2. read `AGENTS.md`;
3. read `MISSION_AND_WORKING_PRINCIPLES.md`;
4. read `README.md` and check freshness;
5. read `docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`;
6. read `docs/research/VORTEX_RESEARCH_HANDOFF.md`;
7. read root research ledgers required by `AGENTS.md`;
8. inspect active experiment source/config/result/checksum files;
9. identify closed mechanism families;
10. start with the cheapest local Gate.

Conversation memory is not authoritative.

## 8. Commit contents

After local validation, the commit should include every artifact needed to understand and resume the round, as applicable:

```text
source
config/dependency pins
focused tests
raw evidence
processed result
logs
checksums
scientific decision
claim boundary
README
canonical ledgers
```

A smaller documentation/governance round may use one atomic commit.

Negative research is committed when it changes reusable project state.

## 9. README freshness

Review `README.md` during every meaningful repository round. Update it in the same commit when any of these changed:

- mission/acceptance target;
- mandatory workflow/governance;
- latest authoritative completed Gate;
- active frontier;
- quick start/dependencies;
- repository layout;
- advertised capability/status/test count.

Final report uses one of:

```text
README_CURRENT=true
README_UPDATED=<sections>
```

or:

```text
README_CURRENT=true
README_UNCHANGED_REASON=<specific reason>
```

## 10. Validation before commit

Run every validation available in the actual local environment:

- focused unit/property tests;
- independent reference comparison where applicable;
- deterministic regeneration;
- syntax/compile checks;
- checksum verification;
- diff/link review;
- full repository tests when feasible;
- public checkpoint execution when the round requires it and the environment supports it.

For documentation-only governance changes, validate the changed Markdown and cross-file consistency.

## 11. Remote verification after commit

After the final push:

1. read the branch from the remote;
2. verify its head SHA equals the produced commit SHA;
3. fetch commit metadata or changed files;
4. update/read the PR if one exists;
5. read back README and mission contract when changed.

No GitHub Actions run is required by this step.

## 12. Final-report table

Every repository-changing final report begins with:

| Field | Actual value |
|---|---|
| Repository | |
| Base branch/SHA | |
| Working branch | |
| Commit SHA | |
| Commit message | |
| Local validation | PASS/FAIL/NOT TESTED |
| Remote SHA verified | PASS/FAIL |
| Pull request | number/state or none |
| GitHub Actions | not required / explicitly requested state |
| README | updated/current |
| Uncommitted remainder | none or exact description |

## 13. Fail-closed completion Gate

```text
HAS_MEANINGFUL_CHANGE=true
LOCAL_VALIDATION_RECORDED=true
HAS_COMMIT=true
REMOTE_CONTAINS_COMMIT=true
RESEARCH_STATE_CURRENT=true
NEXT_GATE_CURRENT=true
README_CURRENT=true
PROVENANCE_TRUTHFUL=true
```

When all are true, the normal round is complete even if no GitHub workflow ran.

If every remote writer fails:

```text
BLOCKED_REMOTE_WRITE
LOCAL_COMMIT_ONLY
ROUND_NOT_COMPLETE
```

## 14. Status vocabulary

```text
LOCAL_RESEARCH
LOCAL_VALIDATION_PASS
LOCAL_VALIDATION_FAILED
COMMIT_PUSHED
REMOTE_COMMIT_VERIFIED
BLOCKED_REMOTE_WRITE
NOT_TESTED
```

`WORKFLOW_RUNNING` and `WORKFLOW_FAILED` are reserved for user-requested exceptional GitHub Actions runs.

## Reality-first execution

Every core Gate has one authoritative `REAL_EXECUTOR_ONLY` arm. Future target tokens/states, perfect selectors, free runtime work, unmeasured compression, or hidden/remote compute cannot satisfy promotion. Charge the implemented finite-word causal path and mark missing target measurements `NOT TESTED`.
