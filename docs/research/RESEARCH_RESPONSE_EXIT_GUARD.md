# VORTEX Research Response-Exit Guard

This contract closes the gap between a **research-round exit** and merely ending a chat response while the round is still open. It is normative whenever the user explicitly asks to start, continue, or pursue the fixed VORTEX mission.

## 1. Research-turn requests enter closed-loop execution

Requests equivalent to:

```text
연구 시작해
계속 진행해
다음 연구 진행해
목표달성을 위한 추론 시작해
```

mean: continue `IDEATE -> PRIOR -> TEST -> FAIL: IDEATE AGAIN / PASS: NEXT DECISIVE VALIDATION` inside the active research round. They do **not** mean “find one promising next idea and report it.”

## 2. Non-terminal states cannot be presented as completed research

The following states are explicitly non-terminal:

```text
PARTIAL_PROGRESS
PROMISING_SIGNAL
IDEALIZED_PASS
LITERATURE_LEAD
NEXT_CANDIDATE_IDENTIFIED
GO
CHEAP_KILL_ONLY
PARTIAL_FAMILY_CLOSURE
ROUND_EXIT_REASON=NONE
```

When any of these is the best current state, the required action is `CONTINUE_RESEARCH`, not a completed-round report.

In particular:

\[
\boxed{\text{IDEALIZED\_PASS} \neq \text{VALIDATED\_SURVIVOR}}
\]

An ideal/asymptotic/free-cost envelope that crosses a threshold is only a `PROMISING_SIGNAL` until the registered finite, exact, fully charged Gate also passes.

## 3. Response-exit guard

For an explicit research-turn request, if:

```text
ROUND_EXIT_REASON=NONE
```

then the assistant must not voluntarily end the research turn merely because it has:

- found a promising paper;
- derived a favorable ideal equation;
- identified a `GO` or `CHEAP_KILL_ONLY` candidate;
- closed one or several mechanism families;
- identified the next candidate batch;
- produced an intermediate experiment number or handoff.

Instead it must continue with the next required Gate or return to ideation after failure.

The only scientific round-exit states remain:

```text
VALIDATED_SURVIVOR
FULL_DESIGN_SPACE_STRUCTURAL_CLOSURE
```

## 4. Execution interruption is not research completion

A tool/runtime/platform limit may make it impossible to continue execution in the same assistant turn. That is an interruption, not a scientific result.

Record it as:

```text
EXECUTION_INTERRUPTED
ROUND_EXIT_REASON=NONE
ROUND_ACTION=RESUME_FROM_EXACT_CHECKPOINT
INTERRUPTION_REASON=<concrete external/runtime/tool reason>
RESUME_CHECKPOINT=<exact branch/commit/result/Gate state>
NEXT_REQUIRED_ACTION=<the first action to execute on resume>
```

Do not relabel an interruption as `VALIDATED_SURVIVOR`, `STRUCTURAL_CLOSURE`, `PARTIAL_FAMILY_CLOSURE`, or “research complete.”

Do not invent an interruption to shorten a research turn. Use it only when an actual execution/tool/runtime boundary prevents further work.

## 5. Mandatory continuation after promising signals

A favorable intermediate signal must immediately be followed by its first non-ideal decisive Gate.

Example:

```text
IDEALIZED_PASS
-> instantiate finite target-size schedule
-> charge transforms/additions/bytes/workspace/native ABI
-> TEST
-> FAIL: preserve evidence, generate new principle batch, continue
-> PASS: continue to the next mandatory Gate
```

A statement like “the ideal envelope passes, so the next EXP will test the finite case” is forbidden as a research-turn endpoint.

## 6. Final reporting boundary

A completed scientific round report may be emitted only when it can truthfully state one of:

```text
ROUND_EXIT_REASON=VALIDATED_SURVIVOR
```

or

```text
ROUND_EXIT_REASON=FULL_DESIGN_SPACE_STRUCTURAL_CLOSURE
```

If execution was externally interrupted, the response must clearly say that the round remains open and provide the exact resume checkpoint. It must not describe the interruption response as the final research result.

## 7. Integration

This guard strengthens, and does not weaken:

- `REAL_EXECUTOR_ONLY`;
- pre-result prior screening;
- immediate testing of `GO` / `CHEAP_KILL_ONLY`;
- same-round return to ideation after failure;
- `PARTIAL_FAMILY_CLOSURE -> CONTINUE_IDEATION`;
- exact output/successor-state requirements;
- full resource accounting;
- local validation before GitHub persistence.

Compact form:

\[
\boxed{
\text{ROUND\_EXIT\_REASON=NONE}
\Rightarrow
\text{CONTINUE\_RESEARCH}
}
\]

unless a real execution boundary forces:

\[
\boxed{
\text{EXECUTION\_INTERRUPTED}
\Rightarrow
\text{ROUND remains OPEN and resumes from an exact checkpoint.}
}
\]
