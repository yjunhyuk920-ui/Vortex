# VORTEX Research Prioritization Contract

This contract prevents VORTEX research from turning into a mechanical sequence of invented experiment topics. It is normative for every new core-research round.

## 1. No homework-mode research

Do not create an experiment merely because the previous experiment finished.
Do not implement one of three ideas merely because the process says to choose one.
Before code, checkpoint execution, or a new EXP number, make an explicit technical prior judgment about whether the mechanism has a credible chance to change the dominant cost equation.

The required question is not only:

> Can this mechanism be made to work?

It is:

> If it works, does it remove enough of the actual dominant 405B cost to matter, and is there a checkpoint-independent reason to expect it to survive the arbitrary-dense requirement?

## 2. Mandatory pre-implementation intuition screen

For each of the three materially different principles, write the following before observing the experiment result:

```text
PRIOR=<HIGH|MEDIUM|LOW>
WHY_IT_MIGHT_WORK=<mechanistic reason>
WHY_IT_MIGHT_FAIL=<strongest known counterargument>
DOMINANT_TERM_CHANGED=<which term in the real cost equation disappears or shrinks>
MAX_IMPACT_IF_TRUE=<best fully charged reduction that could plausibly follow>
ARBITRARY_CHECKPOINT_ARGUMENT=<why this is not model/prompt luck>
CHEAPEST_KILL=<cheapest decisive falsification>
FALSIFICATION_COST=<LOW|MEDIUM|HIGH>
IMPLEMENTATION_COST=<LOW|MEDIUM|HIGH>
DECISION=<GO|NO_GO|CHEAP_KILL_ONLY>
```

`PRIOR` is an informed engineering/research judgment, not a fabricated measured probability. Use prior failures, resource equations, algebraic structure, information availability, and adversarial examples to set it.

## 3. Research-value ordering

Use the heuristic

\[
\boxed{
\text{Research Value}
\propto
\frac{P(\text{works})\times \text{impact if true}}
{\text{cost to falsify}}
}
\]

only as an ordering principle. Do not present the probability term as measured unless it actually is measured.

Prefer candidates that combine:

- a nontrivial reason to work;
- a path to removing a dominant term rather than shaving an auxiliary term;
- a credible `>=10x` core reduction route;
- an arbitrary-checkpoint argument;
- a cheap decisive falsification.

## 4. Mandatory NO-GO conditions before implementation

A candidate is `NO_GO` before coding when any of the following is already clear:

- even perfect success cannot provide the first required `10x` core reduction;
- it only improves an auxiliary term while the dominant term remains effectively unchanged;
- a simple legal arbitrary-dense counterexample already defeats the universal claim;
- it requires future target information, free work, hidden compute, unmeasured compression, or another non-realistic grant;
- it depends on an empirical low-rank/sparse/repetitive property with no reason to hold for arbitrary public dense checkpoints;
- it is a parameter sweep, relabeling, or nearby variant of a closed family;
- the cheapest resource equation already exceeds 8 GiB or the p50/p95 target by a decisive margin.

Do not build 500 lines of code to reconfirm a five-line decisive bound.

## 5. `CHEAP_KILL_ONLY` is an action state, never a stopping state

A mechanism with a LOW prior may still be worth testing when both are true:

1. success would radically change the governing equation, e.g. eliminate an original dense sweep, make one weight scan serve many exact causal states/tokens, or reduce `r` by an order of magnitude; and
2. the decisive falsification is very cheap.

In that case `CHEAP_KILL_ONLY` means **run the cheapest Gate now in the same research round**. It does not mean “save this idea as the next experiment and stop.”

```text
CHEAP_KILL_ONLY
-> execute CHEAPEST_KILL immediately
-> FAIL: record the failure premise and return to principle generation
-> PASS: promote only to a validation survivor and continue to the next required Gate
```

Do not build a backend, kernel, public-checkpoint suite, or large infrastructure until the cheap Gate survives. But also do not postpone the cheap Gate merely by assigning a new EXP number.

This is the moonshot exception: low probability can be acceptable when impact is transformative and falsification is inexpensive.

## 6. `GO` is also not a stopping state

`GO` means that a candidate is worth implementation/validation. It is not evidence that the mechanism works.

```text
GO
-> implement the minimum decisive mechanism
-> run the frozen local validation
-> FAIL: record the failure premise and return to principle generation
-> PASS: mark the candidate VALIDATED_SURVIVOR for that Gate
```

A research report must never present `GO`, `CHEAP_KILL_ONLY`, a literature lead, or a promising equation as the completed scientific result of the round.

## 7. Impact means changing the dominant equation

The minimum accounting frame remains

\[
T_{\rm token}\ge\max\left(\frac{S_c}{BA},\;r\frac{N}{A}\frac{2P}{F}\right).
\]

A high-impact mechanism must credibly change at least one dominant quantity:

- reduce exact checkpoint bytes `S_c` or the number of times they must be moved;
- increase actually committed `A` without exploding `N/A`;
- reduce `N/A` toward one;
- reduce retained exact dense arithmetic `r` by a large factor;
- eliminate a whole verification/state-reconstruction phase rather than merely optimize its constant.

A mechanism that predictably yields only a small constant-factor improvement receives low priority even if it is likely to work.

## 8. Arbitrary-checkpoint prior

Before implementation ask:

> Why should this survive a deliberately adversarial dense checkpoint?

If the answer is only “real LLMs may happen to be structured,” classify the mechanism as restricted/empirical rather than a universal core unless the project goal is explicitly changed.

Construct the strongest legal counterexample you can think of before the positive experiment. If that counterexample is already decisive, stop that candidate and return to ideation.

## 9. Required round selection table

Every core round starts with a comparison table equivalent to:

| Principle | Prior | Why it may work | Strongest failure argument | Max impact | Arbitrary-checkpoint basis | Cheapest kill | Falsification cost | Decision |
|---|---|---|---|---|---|---|---|---|
| A | | | | | | | | |
| B | | | | | | | | |
| C | | | | | | | | |

`NO_GO` is documented briefly and receives no EXP implementation merely to fill a research sequence. `CHEAP_KILL_ONLY` immediately executes its minimal falsification. `GO` immediately enters minimum decisive implementation/validation.

## 10. Three principles are a minimum batch, not a stopping condition

The requirement to invent three materially different principles defines the **minimum batch size for one search iteration**. It is not permission to end a core research round after producing three bad ideas.

If all three candidates are `NO_GO`, or if all candidates that entered validation fail, do not create a new EXP merely to preserve sequence continuity. Instead:

```text
3 new principles
-> prior screen
-> NO_GO candidates discarded
-> CHEAP_KILL_ONLY candidates tested immediately
-> GO candidates minimally validated immediately
-> no validated survivor
-> extract the common failure premise
-> invert or remove that premise
-> generate a new batch of 3 materially different principles
-> repeat
```

A new batch counts as new only when it materially changes at least one of:

- information source;
- computation order;
- verification unit;
- state representation;
- weight-access dependency;
- causal schedule;
- cross-token or cross-layer sharing mechanism.

A rename, threshold change, rank/tile/block sweep, nearby decomposition, or another member of the same decisively closed mechanism family does **not** count as a new principle.

When an entire batch fails, the next batch should be generated from the **common reason for failure**, not by random topic substitution. The objective is to invert the shared hidden premise that killed the previous batch.

## 11. Mandatory validation-return loop

Every failed Gate returns control to ideation in the same research round.

\[
\boxed{
\text{IDEATE}
\rightarrow
\text{PRIOR SCREEN}
\rightarrow
\text{TEST}
\rightarrow
\begin{cases}
\text{FAIL}\rightarrow\text{IDEATE AGAIN}\\
\text{PASS}\rightarrow\text{NEXT DECISIVE VALIDATION}
\end{cases}
}
\]

The following behavior is forbidden:

```text
find CHEAP_KILL_ONLY
-> describe why it might work
-> assign the actual test to EXP-(n+1)
-> end the current research round
```

The required behavior is:

```text
find CHEAP_KILL_ONLY
-> run CHEAPEST_KILL now
-> if it fails, generate the next batch now
-> continue until a real round-exit condition exists
```

Likewise, a failed `GO` implementation does not justify ending the round. Extract the failure premise and return to the principle-generation loop.

## 12. The only core-round exit conditions

A core research round may end only with one of these two recorded states:

### `VALIDATED_SURVIVOR`

At least one candidate has actually passed the frozen decisive Gate for the current rung with the required exactness and fully charged accounting. A prior-screen pass, literature lead, asymptotic possibility, or unexecuted cheapest Gate is not sufficient.

The final report must identify:

```text
ROUND_EXIT_REASON=VALIDATED_SURVIVOR
SURVIVOR=<mechanism>
SURVIVOR_GATE=<actual Gate passed>
SURVIVOR_EVIDENCE=<result/log/checksum paths>
NEXT_UNTESTED_RUNG=<next validation boundary>
```

### `STRUCTURAL_CLOSURE`

A separately recorded theorem, finite lower bound, information argument, or exhaustive structural result closes the remaining admissible design space under the fixed mission/constraints.

The final report must identify:

```text
ROUND_EXIT_REASON=STRUCTURAL_CLOSURE
CLOSED_DESIGN_SPACE=<precise class closed>
CLOSURE_EVIDENCE=<proof/result paths>
```

“Three ideas failed,” “we found a promising paper,” “we found a CHEAP_KILL_ONLY candidate,” or “the next EXP will test it” are **not** valid round-exit reasons.

## 13. Pre-result timestamp/provenance

The prior judgment must be written before the authoritative result is observed. When committed together with the result after local validation, the experiment document must clearly mark the prior section as `PRE_RESULT_PRIOR` and state that it was frozen before the result run.

This prevents hindsight from turning every result into something that supposedly looked obvious afterward.

## 14. Integration with existing VORTEX rules

This contract does not weaken:

- `REAL_EXECUTOR_ONLY`;
- complete resource accounting;
- exact token/successor-state requirements;
- local research and local validation;
- GitHub commit/push/read-back only after local validation;
- README freshness;
- permanent closure of decisively rejected mechanism families.

It adds a stricter closed loop before repository handoff:

\[
\boxed{
\textbf{Think first about whether the idea is worth testing.}
\rightarrow
\textbf{Test every GO/CHEAP\_KILL\_ONLY candidate now.}
\rightarrow
\textbf{On failure, immediately return to ideation.}
\rightarrow
\textbf{Exit only on VALIDATED\_SURVIVOR or STRUCTURAL\_CLOSURE.}
}
\]