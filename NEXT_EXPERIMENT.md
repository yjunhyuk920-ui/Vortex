# Next Experiment — EXP-108A

## Finite Strassen-Calculus Direct-Sum Extraction Gate

### PRE_RESULT_PRIOR

```text
PRIOR=LOW
WHY_IT_MIGHT_WORK=2026 asymptotic-rank speedup theorems prove that nontrivial direct-sum degenerations can outperform naive use of border rank
WHY_IT_MIGHT_FAIL=the theorem is asymptotic and may require tensor powers, extraction multiplicity, coefficient growth or constants far beyond target size
DOMINANT_TERM_CHANGED=exact bilinear rank plus the exactification overhead that destroyed naive rank-46 APA
MAX_IMPACT_IF_TRUE=effective model-weighted arithmetic <=10%, enabling a real finite-word kernel Gate
ARBITRARY_CHECKPOINT_ARGUMENT=tensor identities are independent of checkpoint values
CHEAPEST_KILL=construct one finite h<=target schedule and fully count it before kernel/checkpoint work
FALSIFICATION_COST=MEDIUM
IMPLEMENTATION_COST=HIGH
DECISION=CHEAP_KILL_ONLY
```

### Required output

One explicit finite exact arithmetic schedule for target-relevant rectangular projection shapes. A symbolic asymptotic exponent or existence theorem does not count.

The schedule must state:

```text
base degeneration/direct-sum tensors
finite tensor power h
which direct-sum components are extracted
number and shapes of exact matrix products
all scalar multiplications/additions
coefficient and word growth
interpolation/extraction implementation
transforms, packing, output writes
peak workspace
native finite-word exactness/repair
```

### First promotion Gate

Across the complete registered non-embedding 405B projection inventory at one real block length:

```text
model-weighted exact arithmetic fraction <= 10%
peak workspace <= 8 GiB
zero exactness/control failure
no future target state
no free transform/interpolation/coefficient work
```

Passing this Gate authorizes only a finite-word kernel and causal-source Gate. It does not establish final latency.

### Stop rule and repeated-batch rule

If no explicit finite schedule crosses `10%` before positive runtime costs:

1. reject the instantiated direct-sum extraction, not all asymptotic rank theory;
2. do not tune tensor power, epsilon, degree or block size nearby merely to keep the EXP sequence moving;
3. extract the common failure premise—most likely finite exactification overhead or unavailable causal block—and generate another batch of three principles that materially changes information source, computation order, verification unit, state representation, weight-access dependency, causal schedule, or cross-token/cross-layer sharing.

No GitHub Actions rerun is required after local validation.
