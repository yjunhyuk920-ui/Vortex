# VORTEX repository commit and handoff mandate

This mandate has the same authority as `AGENTS.md`, `MISSION_AND_WORKING_PRINCIPLES.md`, the proof-first contract, and the research-efficiency contract. It applies to every AI or human research session that changes VORTEX.

## 1. Completion definition

Research is not complete merely because code, equations, logs, a sandbox result, a local bundle, or a patch appeared in a chat or temporary workspace.

A repository-changing round is complete only when:

```text
meaningful change exists
sandbox/reference validation is recorded
local or connector commit exists
commit is present on a real remote VORTEX branch
remote SHA is read back and verified
required state/handoff documents are current
README freshness is checked
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

## 2. Sandbox-first research, GitHub-final evidence

Do not use GitHub Actions as the default inner research loop when the ordinary sandbox can perform the work.

Use the sandbox first for:

- equations and exact arithmetic;
- cost/roofline calculations;
- combinatorial, tensor, circuit, or program search;
- prototypes and rapid repair;
- exhaustive, randomized, and adversarial controls;
- focused unit/property tests;
- small locally available checkpoint experiments.

Apply the cheapest decisive sandbox Gate before opening or extending a remote experiment:

```text
SANDBOX_RESEARCH
-> SANDBOX_GATE
```

Only a survivor, a decisive reusable rejection, or meaningful reusable infrastructure is promoted:

```text
SOURCE_COMMIT_PUSHED
-> WORKFLOW_RUNNING        # only when hosted reproduction adds independent value
-> RESULT_COMMIT_PUSHED
-> REMOTE_COMMIT_VERIFIED
```

Hosted workflows are appropriate for clean-room reproduction, pinned public-checkpoint access, long deterministic runs, and immutable artifact/checksum production. They are not a substitute for rapid local reasoning.

Do not create a commit/workflow for every nearby parameter or repair iteration. Do not wait silently for a workflow: report `WORKFLOW_RUNNING` with branch, source SHA, run ID, and independent remaining work.

Sandbox results are not discarded. A meaningful negative conclusion must be included in the next appropriate remote research record, but an obviously rejected nearby variant does not require its own full PR and Actions round.

## 3. Connector-first repository access

Do not make local shell networking a prerequisite when an authenticated GitHub connector or repository writer is available.

Use this order:

1. connected GitHub connector/app;
2. configured local MCP Git writer;
3. authenticated Git and GitHub CLI;
4. local commit plus clean patch/bundle only when all remote writers actually fail.

A failure of local `git fetch`, `curl`, DNS, `gh`, or a tunnel does not prove the GitHub connector is unavailable. Discover and invoke the connector before reporting `GITHUB_WRITER_NOT_EXPOSED`.

## 4. Infrastructure failure is not a session stop condition

Missing GPU, package, outbound DNS, checkpoint cache, or local writer is an infrastructure fact, not a scientific rejection.

Do not end the entire round at the first environment Gate. Instead:

- continue all theory, sandbox search, static validation, and code review that remain possible;
- use available connectors for repository reads/writes;
- commit pinned dependencies and hosted workflows only when hosted execution is actually needed;
- record device-only metrics as `NOT TESTED`;
- leave an executable next Gate when meaningful repository work exists.

Never fabricate a run. Never treat non-execution as scientific evidence.

## 5. Branch and history policy

- Never push directly to `main` or the default branch.
- Never force-push or rewrite another research branch.
- Start from a verified remote branch or commit SHA.
- Create or use a descriptive `research/*` branch.
- Preserve prior negative evidence and provenance.
- Do not cite reconstructed local SHAs as remote history.
- Do not delete failed evidence merely to make the tree clean.
- Do not commit secrets, private checkpoint data, credentials, or large upstream model weights.

## 6. Required startup sequence

Before proposing a new mechanism:

1. verify repository, base branch, base SHA, active branch, PRs, workflow state, and write permissions;
2. read `AGENTS.md`;
3. read `MISSION_AND_WORKING_PRINCIPLES.md`;
4. read `README.md` and assess whether it is current;
5. read `docs/research/VORTEX_RESEARCH_HANDOFF.md`;
6. read all root research ledgers required by `AGENTS.md`;
7. inspect the active experiment, tests, workflow, raw results, and PR discussion;
8. identify whether the proposed mechanism is already closed;
9. perform the cheapest sandbox Gate;
10. freeze a remote experiment only for the survivor or reusable result.

Conversation memory and a reconstructed bundle are not authoritative.

## 7. Required commit sequence

Use atomic commits when the round contains distinct phases. A typical sequence is:

```text
docs: update mission/workflow/handoff
test: establish independent reference and controls
feat: implement the surviving executor mechanism
research: record exactness and resource verdict
```

A smaller round may use one meaningful commit. An empty commit, whitespace-only change, or unrelated edit does not satisfy the mandate.

Negative research must also be committed when it changes the reusable project state.

## 8. Mandatory state and README updates

Update every canonical file whose truth changed, including as applicable:

```text
README.md
MISSION_AND_WORKING_PRINCIPLES.md
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

Do not mechanically touch every file.

`NEXT_EXPERIMENT.md` must contain one highest-information next Gate rather than a loose list.

### README freshness is mandatory

Review `README.md` during every meaningful repository round. Update it in the same round if any of these changed:

- fixed mission or acceptance target;
- mandatory workflow or governance;
- latest authoritative completed Gate;
- active frontier, branch, or PR;
- quick-start command or dependency;
- repository layout;
- an advertised capability, status, or numeric test count.

Do not leave stale active-experiment text or unregenerated test counts.

The final report must contain one of:

```text
README_CURRENT=true
README_UPDATED=<sections or commit>
```

or:

```text
README_CURRENT=true
README_UNCHANGED_REASON=<specific reason>
```

`README_CURRENT=false` prevents round completion when a remote writer is available.

## 9. Validation before commit

Run every validation available in the actual environment:

- focused unit/property tests;
- independent reference comparison;
- deterministic result regeneration;
- syntax/compile checks;
- generated-artifact checksum verification;
- diff and link review;
- full repository tests when feasible.

For documentation-only governance changes, validate Markdown structure, internal paths, required status tokens, and the remote diff. Record tests that were not run.

## 10. Remote verification after commit

After every final commit:

1. read the branch from the remote;
2. verify that its head SHA equals the produced commit SHA;
3. fetch commit metadata or changed files;
4. create or update a PR when appropriate;
5. inspect CI/check status when a workflow should run;
6. read back `README.md` and the mission contract when either changed.

Do not say `committed`, `pushed`, `PR created`, `CI running`, or `merged` before reading the corresponding remote object.

## 11. Final-report Git table

Every repository-changing final report begins with:

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
| README | updated/current or exact reason |
| Uncommitted remainder | none or exact description |

Then list changed files, validations, result hashes, scientific verdict, and `NOT TESTED` items.

## 12. Fail-closed end-of-round Gate

Before a user-facing completion report, evaluate:

```text
HAS_MEANINGFUL_CHANGE
HAS_COMMIT
REMOTE_CONTAINS_COMMIT
RESEARCH_STATE_CURRENT
NEXT_GATE_CURRENT
VALIDATION_RECORDED
README_CURRENT
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

## 13. Status vocabulary

Use as applicable:

```text
SANDBOX_RESEARCH
SANDBOX_GATE
SOURCE_COMMIT_PUSHED
WORKFLOW_RUNNING
WORKFLOW_FAILED
RESULT_COMMIT_PUSHED
REMOTE_COMMIT_VERIFIED
BLOCKED_REMOTE_WRITE
NOT_TESTED
```

The current constructive focus is always obtained from the latest remote PR, authoritative results, `RESEARCH_STATE.md`, `NEXT_EXPERIMENT.md`, and `docs/research/VORTEX_RESEARCH_HANDOFF.md`; do not hard-code a long-lived frontier here.

<!-- REALITY-FIRST-EXECUTION:START -->
## Reality-first authoritative execution

Every new core Gate has one authoritative arm: `REAL_EXECUTOR_ONLY`.
Future target tokens or hidden states, perfect selectors, free `N/A`, free transforms,
free metadata/workspace, free repair/fallback, unmeasured compression, and peak
throughput presented as sustained throughput are forbidden from satisfying a
promotion threshold. Synthetic or target-seeing calculations may appear only as
non-authoritative debugging diagnostics.

The authoritative arm must execute a finite-word causal path and charge candidate
generation, every target position, verification, mismatch repair, rollback,
fallback, transforms, packing, metadata, storage/host/device bytes, KV/cache,
workspace, fragmentation, and measured wall time. Missing quantities remain
`NOT TESTED`; they are never replaced by an ideal grant.

Normative detail: [`docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md`](docs/research/REALITY_FIRST_EXECUTION_CONTRACT.md).
<!-- REALITY-FIRST-EXECUTION:END -->
