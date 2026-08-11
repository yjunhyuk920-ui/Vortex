# E0 Extension-Field Adaptive-Support Gate

## Verdict

```text
REJECT_BLOCK_LOCAL_EXTENSION_FIELD_LINEAR_CELLS
REJECT_DETERMINISTIC_ADAPTIVE_ADDRESS_ESCAPE
REJECT_ARBITRARY_EXACT_POSTPROCESSING_ESCAPE
KEEP_NONLINEAR_STORED_CELLS_AND_CROSS_BLOCK_MIXING OPEN
NO SURVIVING CANDIDATE
```

This is a theorem and integer-calculator Gate. No model, checkpoint, backend,
download, server, GPU, storage device, or hardware action occurred.

## Elementary explanation

Suppose one large matrix column is packed into one very large finite-field
letter. The proposed shortcut stores many exact linear summaries of those
letters. For a new question it may:

1. choose a summary using the question;
2. read it;
3. choose the next address using the value just read;
4. repeat; and
5. apply any exact final function to the values.

That looks stronger than a fixed XOR dictionary. There is nevertheless one
unavoidable execution: run it on an all-zero matrix. Every linear summary it
reads is zero, so the adaptive choices produce one definite short address
list. Any matrix change invisible to that list produces exactly the same zero
transcript. Exactness therefore forces the requested answer to be a linear
combination of the summaries on that list.

One list of `j` field summaries can cover at most `2^j` binary directions.
There are only `C(S,j)` lists of size `j`. Consequently `t` reads can cover at
most

```text
sum_(j=0)^t C(S,j) 2^j
```

directions, while a length-`k` input has `2^k` directions. At the registered
hidden size `k=16,384`, proportional storage gives `S=19,172` summaries. The
exact first radius whose raw count can reach every direction is `3,421`.

```text
logical target                                      194 summaries
all four Q4 traffic lanes granted to this plane      776 summaries
exact counting minimum                             3,421 summaries
```

Thus even the four-lane traffic grant misses by `4.4085x`; the normal logical
work allowance misses by `17.6340x`. This rejects this mechanism, not every
possible finite-word data structure.

## 1. Exact extension-field query

Let `E=GF(2^m)`. Pack an `m x k` binary matrix as

```text
beta = (beta_1,...,beta_k) in E^k.
```

Using trace-dual bases, every binary rank-one query has the exact form

```text
q_(alpha,u)(beta) = Trace(alpha * sum_j u_j beta_j),
alpha != 0, u in GF(2)^k.                              (1)
```

The candidate stores `S` arbitrary `E`-linear cells

```text
y_s(beta) = <g_s,beta>,  g_s in E^k.                   (2)
```

The decoder is allowed deterministic query-dependent addresses, addresses
depending on all earlier returned field values, repeated probes, and arbitrary
exact Boolean post-processing.

## 2. Adaptivity collapses on the zero transcript

Fix `(alpha,u)` and run the decoder on `beta=0`. Every value from (2) is zero.
Let `J` be the distinct cells on this path, with `|J|<=t`.

For every perturbation `h` satisfying

```text
<g_s,h> = 0 for all s in J,                             (3)
```

induction over the adaptive steps shows that the decoder sees the same zeros,
chooses the same next addresses, and returns the same bit. Exactness at `0`
and `h` therefore gives

```text
intersection_(s in J) ker(g_s) subset ker(q_(alpha,u)). (4)
```

For linear maps over a finite field, (4) means that the queried binary
functional lies in the binary row span of the probed field cells. The binary
coordinate functionals of one field cell are precisely

```text
h -> Trace(c <g_s,h>), c in E.
```

Hence

```text
alpha*u in span_E{g_s:s in J}.
```

Scaling by nonzero `alpha` proves the necessary sparse representation

```text
u in span_E{g_s:s in J}.                                (5)
```

This is why changing the addresses after each read does not rescue linear
stored cells. The proof uses one legal transcript rather than assuming that
the addresses were fixed in advance.

## 3. Extension coefficients cannot pack extra binary directions

For a fixed support `J`, put

```text
V_J = span_E{g_s:s in J} subset E^k.
```

Any set of binary vectors independent over `GF(2)` remains independent over
`E`: a nonzero binary determinant remains nonzero after extending the field.
Therefore

```text
dim_GF(2)(V_J intersect GF(2)^k)
    <= dim_E(V_J)
    <= |J|.                                               (6)
```

It follows that one `j`-cell support contains at most `2^j` binary directions.
Taking a union over every support of size at most `t` gives the necessary
condition

```text
sum_(j=0)^t C(S,j) 2^j >= 2^k.                           (7)
```

The implementation evaluates (7) with exact integers. A complete enumeration
of every zero-, one-, and two-generator subspace of `GF(4)^3` independently
reaches the exact intersection maxima `1,2,4`, respectively. That finite
control checks the implementation; equation (6) is the general proof.

## 4. Registered exact counts

The proportional one-plane grant converts the complete binary source plus all
8 GiB of favorable hot state into field cells:

```text
lambda = (403,747,897,344 + 68,719,476,736)
         / 403,747,897,344
       = 112645/96261
       = 1.170203924746... .                              (8)
```

| binary direction `k` | cells `S` | logical target | four-lane target | exact minimum `t` | minimum fraction |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,198 | 12 | 48 | 216 | 21.093750% |
| 16,384 | 19,172 | 194 | 776 | 3,421 | 20.880127% |
| 53,248 | 62,311 | 631 | 2,524 | 11,111 | 20.866511% |
| 128,256 | 150,085 | 1,520 | 6,080 | 26,759 | 20.863741% |

For `k=16,384`, equation (7) at the logical target is strictly below
`2^1,753`, versus the required `2^16,384`. Even the four-lane traffic grant is
below `2^5,457`. The failure is not a rounding artifact near the boundary.

## 5. Arbitrary block-local storage allocation also fails

Proportional allocation is not required for the aggregate result. For block
`i`, define

```text
N_i = m_i k_i       source bits
A_i = m_i S_i       stored field-cell bits
T_i = m_i t_i       probed field-cell bits.
```

The standard exact-support relaxation is

```text
sum_(j=0)^t C(S,j)2^j <= (2eS/t)^t,
```

so every block obeys

```text
N_i <= T_i log2(2e A_i/T_i).                             (9)
```

The perspective `T log(A/T)` is concave. Summing (9), while allowing any
distribution of block-local storage, gives

```text
1 <= p log2(2e lambda/p),
p = sum_i T_i / sum_i N_i,
lambda = sum_i A_i / sum_i N_i.                          (10)
```

Two deliberately excessive storage grants still fail:

| grant to one binary plane | `lambda` | relaxed minimum `p` | four-lane target | miss factor |
|---|---:|---:|---:|---:|
| all packed-Q4 bits + 8 GiB | 4.170204 | 13.535470% | 4.740741% | 2.8551x |
| all 551.22 GB DFloat11 bits + 8 GiB | 11.092267 | 10.988948% | 4.740741% | 2.3180x |

The second rejection has an integer-only check. Since `e<3` and
`lambda<12`, at `p=32/675`:

```text
2e lambda/p < 2*3*12/(32/675) < 2^11,
p log2(2e lambda/p) < 11*(32/675) = 352/675 < 1.          (11)
```

Inverting (10), the four-lane target would require at least

```text
lambda >= p 2^(1/p)/(2e)
       = 19,515.208649...
```

or about `895.764 TiB` merely to encode the registered one-bit surrogate.
This is before native numerical work, addresses, KV/state, and output logic.

## 6. Exact claim boundary

This Gate rejects a meaningful proposed mechanism:

- every stored block cell is an arbitrary extension-field linear form;
- query coefficients and addresses may be chosen freely;
- addresses may depend on previously returned values;
- final exact post-processing may be arbitrary;
- storage may be allocated arbitrarily among block-local dictionaries.

It does **not** reject:

- source-dependent nonlinear stored cells;
- one cell mixing data from otherwise independent matrix blocks;
- an arbitrary nonlinear hot-advice partition of the source;
- randomized error or approximate answers;
- an as-yet-unconstructed native arithmetic identity not reducible to these
  linear cells.

Those exclusions are now the next mechanism boundary. Reintroducing fixed or
adaptive extension-field linear summaries under another name is closed.

## 7. Reproduction

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
.deps\exp076-venv\Scripts\python.exe `
  scripts\derive_extension_field_adaptive_support_gate.py `
  --output-dir results\e0_extension_field_adaptive_support_gate

.deps\exp076-venv\Scripts\python.exe -m unittest `
  tests.test_extension_field_adaptive_support_gate -v
```

Authority:

```text
results/e0_extension_field_adaptive_support_gate/summary.json
results/e0_extension_field_adaptive_support_gate/checksums.sha256
```
