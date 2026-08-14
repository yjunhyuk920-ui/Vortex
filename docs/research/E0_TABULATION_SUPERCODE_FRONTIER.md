# E0 Tabulation / Segre-Supercode Frontier

## Verdict

```text
REDUCE NONADAPTIVE XOR TABULATION TO THE SPARSE FUNCTIONAL DICTIONARY
REJECT RANDOM ATOMS AS A RANK-ONE ALIGNMENT ARGUMENT
KEEP A SEGRE-ALIGNED SPARSE SUPERCODE OPEN BUT UNCONSTRUCTED
KEEP ADAPTIVE NONLINEAR FINITE-WORD DECODERS OUTSIDE THIS THEOREM
NO SURVIVING CANDIDATE
```

No checkpoint, model, backend, download, server, or hardware action occurred.
This is a scoped algebraic reduction plus a capacity diagnostic, not a target
lower bound and not an implementation.

## Elementary explanation

Imagine storing clever yes/no facts about a very large switchboard. For each
question, a fixed card tells us which stored facts to read, and we XOR their
yes/no answers.

The stored facts may look nonlinear and clever. But if their XOR must equal
one exact parity question for **every** possible switchboard, all nonlinear
pieces selected by that card must cancel. What remains is exactly the XOR of
some ordinary linear switch groups. Thus this kind of table is not a new
machine; it is the already open sparse dictionary in disguise.

A narrower possibility survives: choose the linear groups with exceptional
structure so that every required rank-one pattern is the XOR of very few
groups. There are numerically enough short group selections in small block
parameters, but no construction makes those selections land on the required
patterns. It is like having enough possible key rings without knowing how to
cut keys for these particular locks.

## 1. Linearization lemma

Let the binary checkpoint be `W in F_2^D`. A compiler stores arbitrary Boolean
cells

```text
z_j(W) in F_2.
```

For every query mask `q`, a fixed nonadaptive recovery set `S(q)` must satisfy

```text
XOR(j in S(q), z_j(W)) = <q,W>              for every W.       (1)
```

Expand each cell in its unique algebraic normal form:

```text
z_j(W) = a_j,empty XOR XOR_i a_j,{i} W_i
         XOR XOR_|A|>=2 a_j,A product(i in A,W_i).             (2)
```

Equation (1) is a linear polynomial for every input. By uniqueness of (2),
the selected constant terms cancel, every selected degree-two-or-higher term
cancels, and the selected degree-one coefficients obey

```text
q = XOR(j in S(q), g_j),
g_j[i] = a_j,{i}.                                             (3)
```

Replacing every stored bit by the linear form `<g_j,W>` therefore preserves
the same recovery sets and exact answers. Arbitrary nonlinear preprocessing
provides no advantage for this precise decoder class.

The executable control uses genuinely quadratic cells in dimension three;
their nonlinear monomials cancel and the extracted degree-one atoms recover
all declared queries. This control checks the implementation of the proof. It
is not evidence from a toy search.

The lemma also applies independently to each output bit of an XOR-only word
decoder. It does **not** apply when addresses depend on previously read data,
when the decoder uses arbitrary word operations, or when the requested output
has native BF16/FP32 accumulation semantics.

## 2. The remaining structured object

After the reduction, a local binary `b x b` block has ambient dimension

```text
d = b^2
```

and exactly

```text
Q_b = 1 + (2^b-1)^2                                      (4)
```

distinct rank-at-most-one masks. Define a **Segre-aligned sparse supercode**
here to mean an implicit atom family `g_1,...,g_S in F_2^d` for which every
mask in (4) is the XOR of at most `t` atoms. The name is descriptive; it does
not assert that a published construction has this property.

Give this object every favorable concession:

- allocate the complete model-wide 8 GiB advice proportionally to each block;
- store one exact bit per atom;
- make atom descriptions, lookup, arithmetic, and physical addressing free;
- test only one binary query, not a native 32-query causal batch.

Then

```text
S = floor((403,747,897,344 + 68,719,476,736)
          / 403,747,897,344 * d).
```

A necessary condition is

```text
sum(k=0..t, C(S,k)) >= Q_b.                                (5)
```

The left side merely counts possible subsets. Collisions can only reduce the
actual number of represented masks, so (5) is necessary but not sufficient.

## 3. Capacity does not reject the object

The first block side whose minimum counting weight happens to fit the
registered `8/675` coefficient fraction is

```text
b                                      23
d=b^2                                 529
favorable atoms S                     619
minimum t not rejected by (5)           6
t/d                                  6/529 = 1.1342155...%
registered fraction                  8/675 = 1.1851851...%
```

This boundary is not monotone at every adjacent side: `b=24` needs seven
selections and gives `7/576`, which is slightly above the fraction, while
`b=32` needs eight selections and gives `8/1024`.

The result means only that name counting cannot prove impossibility. It does
not exhibit even one 619-atom family covering the 23-by-23 rank-one set, much
less an implicit layout across all registered matrices.

## 4. Random atoms are not the missing construction

For independent uniform atoms, the XOR of any fixed nonempty subset is uniform
in `F_2^d`. At `b=23`, a union bound gives

```text
Pr[a fixed nonzero query has a representation of weight <=6]
    <= (sum(k=1..6,C(619,k))) / 2^529
    < 2^-482.
```

Even after multiplying by every nonzero local rank-one mask, the expected
number hit is below `2^-436`. A generic high-girth or random dictionary thus
does not explain alignment. Any positive construction must deliberately place
its short subset sums on the binary Segre set while remaining near-source in
space and locally decodable.

This does not prove no such structured family exists. Indeed, equation (5)
shows why a pure counting rejection cannot work at these parameters.

## 5. Exact open obligations

The surviving object must still supply all of the following before it is a
candidate:

1. an explicit or succinct atom generator fitting the complete persistent
   state budget;
2. a proof that every legal rank-one query has a target-small representation;
3. a sub-dense decomposer that returns physical cell addresses without a
   metadata scan;
4. one small shared physical union for a 32-query causal batch;
5. a lift from GF(2) parity to the checkpoint's native bounded-word values and
   reference accumulation order;
6. the complete 405B cost equation, including state, KV, scheduling,
   verification, repair, and fallback.

Until those exist, this is an accurately isolated algebraic frontier and not
`OMEGA` runtime architecture.

## 6. Relationship to prior work

The sparse linear interface is the non-systematic data-structure model audited
from Ramamoorthy and Rashtchian, *Equivalence of Systematic Linear Data
Structures and Matrix Rigidity*, ITCS 2020:
https://doi.org/10.4230/LIPIcs.ITCS.2020.35.

The earlier project authority already rejects literal global answer books,
direct local rank-one tables, fixed linear right inverses, and counting-only
promotion:

- `docs/research/E0_SPARSE_FUNCTIONAL_DICTIONARY_FRONTIER.md`
- `docs/research/E0_FIXED_LINEAR_FUNCTIONAL_DECODER_ACTIVITY_BOUND.md`
- `docs/research/E0_JOINT_BATCH_COSET_GEOMETRY.md`

The ANF lemma above closes only the new claim that arbitrary nonlinear stored
bits rescue a fixed-address XOR decoder. It deliberately leaves the general
adaptive finite-word ticket open.

## 7. Reproduction

```powershell
$env:PYTHONPATH = "."
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_tabulation_supercode_frontier.py `
  --output-dir results\e0_tabulation_supercode_frontier

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_tabulation_supercode_frontier -v
```
