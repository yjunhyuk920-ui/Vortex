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

## 5. `CHEAP_KILL_ONLY` exception

A mechanism with a LOW prior may still be worth testing when both are true:

1. success would radically change the governing equation, e.g. eliminate an original dense sweep, make one weight scan serve many exact causal states/tokens, or reduce `r` by an order of magnitude; and
2. the decisive falsification is very cheap.

In that case run only the cheapest Gate first. Do not build a backend, kernel, public-checkpoint suite, or large infrastructure until the cheap Gate survives.

This is the moonshot exception: low probability can be acceptable when impact is transformative and falsification is inexpensive.

## 6. Impact means changing the dominant equation

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

## 7. Arbitrary-checkpoint prior

Before implementation ask:

> Why should this survive a deliberately adversarial dense checkpoint?

If the answer is only “real LLMs may happen to be structured,” classify the mechanism as restricted/empirical rather than a universal core unless the project goal is explicitly changed.

Construct the strongest legal counterexample you can think of before the positive experiment. If that counterexample is already decisive, stop.

## 8. Required round selection table

Every core round starts with a comparison table equivalent to:

| Principle | Prior | Why it may work | Strongest failure argument | Max impact | Arbitrary-checkpoint basis | Cheapest kill | Falsification cost | Decision |
|---|---|---|---|---|---|---|---|---|
| A | | | | | | | | |
| B | | | | | | | | |
| C | | | | | | | | |

Only `GO` proceeds to implementation. `CHEAP_KILL_ONLY` receives only the minimal falsification. `NO_GO` is documented briefly and receives no EXP implementation merely to fill a research sequence.

## 9. Pre-result timestamp/provenance

The prior judgment must be written before the authoritative result is observed. When committed together with the result after local validation, the experiment document must clearly mark the prior section as `PRE_RESULT_PRIOR` and state that it was frozen before the result run.

This prevents hindsight from turning every result into something that supposedly looked obvious afterward.

## 10. Integration with existing VORTEX rules

This contract does not weaken:

- `REAL_EXECUTOR_ONLY`;
- complete resource accounting;
- exact token/successor-state requirements;
- local research and local validation;
- GitHub commit/push/read-back only after local validation;
- README freshness;
- permanent closure of decisively rejected mechanism families.

It adds a stricter rule before all of them:

\[
\boxed{
\textbf{Think first about whether the idea is worth testing.}
\rightarrow
\textbf{Kill low-value ideas before implementation.}
\rightarrow
\textbf{Spend experiments only on high-value survivors.}
}
\]
