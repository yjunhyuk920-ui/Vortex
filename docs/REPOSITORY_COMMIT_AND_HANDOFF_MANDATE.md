# VORTEX Repository Commit and Handoff Mandate

Version: CTC-2026-09-05.

This mandate covers durable recording, not scientific achievement. [Constructive Theory Contract](CONSTRUCTIVE_THEORY_CONTRACT.md) separately defines theory and target-hardware acceptance.

## Required path

```text
LOCAL_RESEARCH -> LOCAL_VALIDATION_PASS -> COMMIT_PUSHED -> REMOTE_COMMIT_VERIFIED
GITHUB_ACTIONS_REQUIRED=false
GITHUB_REEXECUTION_REQUIRED=false
```

Do research and available validation locally. Freeze hypotheses/thresholds/input hashes before results and retain preregistration. Persist source, configs/pins, raw and processed evidence, logs, checksums, decisions, claim boundaries and affected canonical ledgers after validation. GitHub is persistence and handoff, not a required second laboratory.

A local-only commit, chat response, ZIP, reconstructed tree or unapplied patch is not a completed repository handoff. All available remote writers must actually be tried before reporting them unavailable. Prefer the connected GitHub writer, then configured local MCP Git, then authenticated Git/CLI. A failed shell DNS lookup is not a failed GitHub connector.

## Branch/history safety

Start from a verified remote SHA and use descriptive `research/*` branches. Never directly push main, force-push, rewrite others' history, bypass branch protection or merge unrelated draft experiments. A policy-only PR may target main without importing unmerged research. Preserve prior positive/negative evidence and exact pre-edit policy blobs when consolidating long histories. No secrets or upstream checkpoint weights in commits.

## Checks and verification

Run locally applicable syntax/reference/unit/property/regeneration/checksum checks. Documentation-only work needs Markdown/link/contract-consistency review, not fictitious runtime execution. Record full-suite/hardware checks as NOT TESTED when not performed.

Review README every meaningful round; update changed governance, frontier, evidence, setup or repository map in the same change. Update only ledgers whose truth changed. A policy-only decision may be preserved in a linked governance record while scientific logs remain untouched.

After pushing, read the remote branch head, compare it to the produced commit, inspect commit metadata/diff or changed-file hashes, inspect the PR when present, and read back changed mission/README entry points. After a merge, verify the target branch/merge commit too. No separate hosted rerun is required. Explicitly requested Actions and existing branch protection remain separate concerns.

## Persistence gate

```text
HAS_MEANINGFUL_CHANGE=true
LOCAL_VALIDATION_RECORDED=true
HAS_COMMIT=true
REMOTE_CONTAINS_COMMIT=true
AFFECTED_LEDGERS_CURRENT=true
README_CURRENT=true
PROVENANCE_TRUTHFUL=true
```

These facts establish only `HANDOFF_STATUS=REMOTE_COMMIT_VERIFIED`. They do not establish a complete theory or E7, and must not be used as an automatic scientific stopping rule. Preserve incomplete evidence truthfully.

When remote writing fails after actual attempts, report `HANDOFF_STATUS=BLOCKED_REMOTE_WRITE`, `LOCAL_COMMIT_ONLY` where true, and `HANDOFF_NOT_COMPLETE`. Do not invent a commit or remote link. A snapshot cannot contain its own final commit hash; record post-commit verification in the actual tool evidence/PR or a later receipt rather than a self-referential claim.

Report what changed and its scientific meaning first, then branch/base SHA, commit/PR, local validation scope, remote verification, README freshness and any remainder. Use `THEORY_STATUS`, `HARDWARE_STATUS` and `HANDOFF_STATUS` independently.
