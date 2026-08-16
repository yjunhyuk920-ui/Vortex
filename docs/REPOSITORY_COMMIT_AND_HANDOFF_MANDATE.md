# VORTEX repository commit and handoff mandate

This mandate has the same authority as `AGENTS.md`, the proof-first contract, and the research-efficiency contract. It applies to every AI or human research session that changes VORTEX.

## 1. Completion definition

Research is not complete merely because code, equations, logs, a local bundle, or a patch appeared in a chat or temporary workspace.

A repository-changing round is complete only when:

```text
meaningful change exists
local or connector commit exists
commit is present on a real remote VORTEX branch
remote SHA is read back and verified
required state/handoff documents are updated
validation evidence is recorded
the worktree or connector change set has no unreported remainder
```

A local-only commit is classified as:

```text
LOCAL_ONLY
REMOTE_RESEARCH_NOT_RECORDED
ROUND_NOT_COMPLETE
```

Only a commit verified on the remote branch may be classified as:

```text
REMOTE_COMMIT_VERIFIED
```

## 2. Connector-first repository access

Do not make local shell networking a prerequisite when an authenticated GitHub connector or repository writer is available.

Use this order:

1. connected GitHub connector/app;
2. configured local MCP Git writer;
3. authenticated Git and GitHub CLI;
4. local commit plus clean patch/bundle only when all remote writers actually fail.

A failure of local `git fetch`, `curl`, DNS, `gh`, or a local tunnel does not prove that the GitHub connector is unavailable. Discover and invoke the connector before reporting `GITHUB_WRITER_NOT_EXPOSED`.

When the connector exposes branch, file, commit, PR, comment, workflow, or status operations, use those operations directly.

## 3. Infrastructure failure is not a session stop condition

Missing GPU, package, outbound DNS, checkpoint cache, or local writer is an infrastructure fact, not a scientific rejection.

Do not end the entire round at the first environment gate. Instead:

- perform all repository reads/writes through available connectors;
- commit pinned dependency and CI workflows for hosted execution;
- commit reproducible environment probes and fail-closed runners;
- perform theory, code review, artifact-format work, and static validation that do not require the missing device;
- record device-only metrics as `NOT TESTED`;
- leave a real remote commit and one executable next gate.

Never fabricate a run. Never use honest non-execution as a reason to leave the repository unchanged when meaningful repository work can still be completed.

## 4. Branch and history policy

- Never push directly to `main` or the default branch.
- Never force-push or rewrite another research branch.
- Start from a verified remote branch or commit SHA.
- Create a descriptive `research/*` branch.
- Preserve prior negative evidence and provenance.
- Do not cite reconstructed local SHAs as remote history.
- Do not delete failed evidence merely to make the tree clean.
- Do not commit secrets, private checkpoint data, credentials, or large upstream model weights.

## 5. Required startup sequence

Before proposing a new mechanism:

1. verify repository, base branch, base SHA, open PRs, and write permissions;
2. read `AGENTS.md`;
3. read `docs/research/VORTEX_RESEARCH_HANDOFF.md`;
4. read all root research ledgers required by `AGENTS.md`;
5. inspect the active experiment, tests, workflow, raw results, and PR discussion;
6. identify whether the proposed mechanism is already closed;
7. freeze the cheapest decisive gate;
8. create or select the actual remote research branch.

Conversation memory and a reconstructed bundle are not authoritative.

## 6. Required commit sequence

Use atomic commits when the round contains distinct phases. A typical sequence is:

```text
docs: update research contract and handoff
test: establish official reference environment
feat: implement checkpoint-specific executor mechanism
research: record exactness and physical resource verdict
```

A smaller round may use one meaningful commit. An empty commit, whitespace-only change, or unrelated edit does not satisfy the mandate.

Negative research must also be committed. A correct rejection prevents future sessions from repeating a closed mechanism.

## 7. Mandatory state updates

Update the canonical files affected by the result, including as applicable:

```text
RESEARCH_STATE.md
NEXT_EXPERIMENT.md
DECISION_LOG.md
FAILED_APPROACHES.md
FAILED_APPROACHES_RECENT.md
ASSUMPTION_REGISTER.md
VALIDATION_MATRIX.md
ARCHITECTURE.md
HARDWARE_VALIDATION_PLAN.md
REPRODUCIBILITY.md
docs/research/VORTEX_RESEARCH_HANDOFF.md
```

Do not mechanically touch every file. Update every canonical ledger whose truth changed.

`NEXT_EXPERIMENT.md` must contain one highest-information next gate rather than a list of loosely related possibilities.

## 8. Validation before commit

Run every validation available in the actual environment:

- focused unit/property tests;
- independent reference comparison;
- deterministic result regeneration;
- syntax/compile checks;
- generated-artifact checksum verification;
- diff/format validation;
- full repository tests when feasible.

Record tests that were not run. An expected fail-closed gate must document its expected exit code and the evidence required to make it pass.

## 9. Remote verification after commit

After every final commit:

1. read the branch from the remote;
2. verify that its head SHA equals the produced commit SHA;
3. fetch the commit metadata or changed files;
4. create or update a PR when appropriate;
5. inspect CI/check status when a workflow should run.

Do not use the words `committed`, `pushed`, `PR created`, `CI running`, or `merged` before the corresponding remote object is read back.

## 10. Final-report Git table

Every repository-changing round must begin its final report with:

| Field | Actual value |
|---|---|
| Repository | |
| Base branch | |
| Base SHA | |
| Working branch | |
| Commit SHA(s) | |
| Commit message(s) | |
| Remote SHA verified | PASS/FAIL |
| Pull request | number/state or none |
| CI/checks | actual state |
| Uncommitted remainder | none or exact description |

Then list changed files, validations, result hashes, scientific verdict, and `NOT TESTED` items.

## 11. Fail-closed end-of-round gate

Before a user-facing progress report, evaluate:

```text
HAS_MEANINGFUL_CHANGE
HAS_COMMIT
REMOTE_CONTAINS_COMMIT
RESEARCH_STATE_CURRENT
NEXT_GATE_CURRENT
VALIDATION_RECORDED
PROVENANCE_TRUTHFUL
```

When remote writing is available, all must be true.

If every remote writer actually fails, preserve the local commit, patch, bundle, base SHA, exact errors, hashes, and apply verification, then report:

```text
BLOCKED_REMOTE_WRITE
LOCAL_COMMIT_ONLY
ROUND_NOT_COMPLETE
```

Do not report a completed research round in that state.

## 12. Current constructive focus

The active focus after the cold-backed audit is not another generic lower bound or static selector. It is an actual fixed-public dynamic executor with a nontrivial successor state. Read `docs/research/FIXED_PUBLIC_DYNAMIC_EXECUTOR_DIRECTIVE.md` and `docs/research/VORTEX_RESEARCH_HANDOFF.md` before beginning that work.
