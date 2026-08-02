# VORTEX work-session protocol

Last updated: 2026-08-03 (Asia/Seoul)

This protocol is mandatory for every AI or human session that performs meaningful VORTEX work. It exists so that progress, failures, evidence, and next actions survive chat boundaries without relying on conversational memory.

## Startup contract

Before proposing architecture changes or editing code, read in this order:

1. `AGENTS.md`
2. `docs/PROOF_FIRST_CONTRACT.md`
3. `docs/WORK_SESSION_PROTOCOL.md`
4. `docs/RESEARCH_PROGRESS_LEDGER.md`
5. `docs/SESSION_HANDOFF.md`
6. `docs/ROADMAP.md`
7. the active experiment document and workflow
8. the latest open research PR, workflow run, comments, and committed result JSON

Then verify the active branch, head commit, PR state, CI state, and the most recent machine-readable evidence. Never assume a previously mentioned run is still current.

## Mandatory work loop

For every meaningful hypothesis or implementation step:

1. State the measurable hypothesis.
2. Write the correctness or declared-quality contract.
3. Derive memory, traffic, compute, and amortization equations at the 405B target.
4. Define explicit promotion and rejection thresholds.
5. Implement the smallest real-operation falsification.
6. Add tests and an isolated workflow that references only files present on its branch.
7. Run the workflow and inspect actual logs, not only status badges.
8. Commit raw evidence as JSON when the workflow succeeds.
9. Record both positive and negative results in `docs/RESEARCH_PROGRESS_LEDGER.md`.
10. Update `docs/SESSION_HANDOFF.md` with the exact current frontier and next executable step.

Failed hypotheses are permanent project data. Do not delete, soften, or rewrite a negative result as partial success.

## Mandatory response-completion rule

Before giving the user a progress or completion answer after meaningful repository work, update the repository first.

At minimum, the repository update must record:

- timestamp in Asia/Seoul;
- active branch and PR number;
- latest head commit;
- workflow run and conclusion, or that it is still running;
- implemented files and equations;
- measured results and evidence level;
- rejected assumptions and why they failed;
- the one next decisive experiment;
- exact commands or workflow needed to continue.

The durable locations are:

- chronological research results: `docs/RESEARCH_PROGRESS_LEDGER.md`;
- current state and immediate continuation: `docs/SESSION_HANDOFF.md`;
- experiment-specific derivation and thresholds: `docs/EXPERIMENT_*.md`;
- raw reproducible metrics: `results/*.json`.

A chat answer must not claim that progress was recorded unless the Git commit actually exists. If repository writing fails, report that failure explicitly.

## Branch and workflow isolation

Research branches may be based on different rejected or active candidates. A workflow must never assume sibling-branch files exist.

Each experiment workflow must:

- list its required test files explicitly;
- verify each required path exists before running pytest;
- avoid importing helper modules that exist only in a sibling branch unless those helpers are copied or the branch base contains them;
- checkout the intended branch head;
- commit evidence only after all correctness checks and measurements succeed;
- use a branch-specific concurrency group.

A missing file is an infrastructure failure, not experimental evidence. Fix it and rerun before interpreting the candidate.

## Evidence and communication

Use the E0–E4 evidence scale from `docs/PROOF_FIRST_CONTRACT.md`.

- Do not describe E1/E2 results as proving the 405B objective.
- Separate exact measurements from 405B projections.
- State all proxy hardware assumptions.
- Distinguish a component optimization from a complete runtime path.
- Never hide full exact gate/up, teacher gradients, future target tokens, or fallback streams from the accounting.

## Session shutdown checklist

A session is not complete until all applicable items are done:

- tests/workflow inspected;
- raw evidence committed or failure logged;
- experiment PR updated or closed with a factual decision;
- research ledger updated;
- session handoff updated;
- next decisive command documented;
- user-facing answer matches the committed repository state.
