# EXP-072A — Self-Contained Exact Q4 DAG Information-Capacity Gate

## Question

Can a self-contained exact arithmetic-DAG artifact replace an arbitrary Q4 dense checkpoint while fitting the VORTEX static/hot-state budgets, without consulting the original checkpoint during a query?

EXP-072 proposed deterministic synthesis of non-contiguous shared linear forms. Before any synthesizer, beam search, model download, or kernel is built, EXP-072A applies the cheaper information-capacity Gate required by the proof-first contract.

The only allowed outcomes are:

```text
PROMOTE_SELF_CONTAINED_EXACT_Q4_DAG_TO_BOUNDED_SYNTHESIS
REJECT_SELF_CONTAINED_EXACT_Q4_DAG_AS_UNIVERSAL_CORE_RETAIN_RESTRICTED_SYNTHESIS_AUXILIARY
```

## Declared interface

The candidate artifact and a fixed interpreter receive only an activation vector at query time. They must produce the exact integer linear map for every activation. The artifact may contain opcodes, operand IDs, constants, output maps, tile maps, scales, or other metadata, but it may not read the original Q4 coefficients or another checkpoint-derived cold store during the query.

This restriction isolates the self-contained DAG proposed by EXP-072. A design that consults the original checkpoint or another lossless cold representation is a different online data-structure/execution class; EXP-071 did not prove that broader class impossible.

## E0 candidate-efficiency scorecard

1. **Target-scale upside** — A positive DAG could in principle share arithmetic across outputs. To fit the fixed 8 GiB hot-state cap using only a self-contained artifact, however, it must encode 405,849,243,648 Q4 coefficients in at most 68,719,476,736 bits: `4.2327%` of packed Q4. The former EXP-072 `10%` static median Gate is already too weak for the final hot-state cap, and the final target-equivalent traffic fraction is approximately `1.185185%` before overhead.
2. **Mechanism novelty** — Non-contiguous shared linear forms are broader than EXP-067 whole-row reuse and EXP-070 contiguous local patterns. They do not introduce an external information source: the exact circuit still determines the complete matrix.
3. **Evidence basis** — Q4 has a small coefficient alphabet and structured controls admit compact circuits. No population or theorem evidence currently shows that arbitrary public dense checkpoints have descriptions below the required fraction.
4. **Scaling reason** — The number of Q4 maps grows as `16^P`, while the 8 GiB allowance is fixed. The universal information-capacity gap grows linearly with parameter count rather than improving toward 405B.
5. **Universality** — Any public unmodified dense checkpoint is in scope. A checkpoint containing an incompressible Q4 matrix is therefore a valid adversarial target.
6. **Correctness closure** — Exact equality for every integer activation is required. Equality on basis vectors uniquely recovers every coefficient, so two different matrices cannot share one self-contained artifact.
7. **Resource closure** — Count the entire artifact. Omitting row scales, biases, interpreter code, workspace, and alignment is deliberately favorable to the candidate. Querying external weight-derived information is forbidden in this Gate and must be charged in a separate execution class.
8. **Cheapest falsification** — A counting/injectivity argument precedes synthesis: compare the number of Q4 matrices with the number of artifacts allowed by a bit cap, and validate the finite-domain reduction exhaustively on tiny matrices.

## Information-capacity statement

For `P` independently selectable coefficients from an alphabet of size `A`, there are `N = A^P` distinct linear maps. If the artifact plus fixed interpreter is exact for every activation, evaluating it on the standard basis recovers the matrix. The matrix-to-artifact mapping must therefore be injective.

There are at most `2^(B+1)-1` binary strings of length at most `B`. A universal artifact cap must satisfy:

```text
2^(B+1) - 1 >= A^P
```

For Q4, `A=16`, so the worst-case maximum artifact length is at least `4P` bits. This lower bound ignores scales, biases, opcodes, alignment, interpreter state, and workspace; it is favorable to the candidate.

## Registered target arithmetic

```text
registered parameters                 405,849,243,648
Q4 information                        1,623,396,974,592 bits
packed Q4 size                         188.98828125 GiB
8 GiB hot-state allowance              68,719,476,736 bits
hot allowance / packed Q4              approximately 4.2327%
former EXP-072 static median threshold 10%
final target-equivalent fraction       1.185185185%
```

## Reference and controls

- exact integer matrix-vector evaluation;
- standard-basis signatures for every matrix over registered tiny domains;
- exhaustive proof that distinct matrices have distinct exact linear maps;
- boundary checks around the minimum universal maximum artifact length;
- structured matrices as positive controls for per-instance compressibility;
- one-coefficient mutation changes the basis signature;
- deterministic reruns produce identical evidence hashes.

The structured controls may show that some matrices are compact. They may not promote a universal arbitrary-checkpoint claim.

## Promotion Gate

Promotion to bounded real-weight synthesis requires all of the following:

```text
zero finite-domain enumeration/signature/control mismatch
a universal self-contained artifact cap <=8 GiB for the registered 405B Q4 population
a universal static fraction <=4.2327% before scales, bias, workspace, and runtime overhead
an explicit route from that cap toward 1.185185% fully charged target-equivalent traffic
```

If injectivity forces the worst-case artifact to retain the packed Q4 information content, the self-contained exact DAG is rejected as a universal core without implementing the synthesizer.

## Stop rule

Before this Gate survives, prohibit:

```text
pair-frequency CSE over downloaded checkpoints
beam/SAT/SMT circuit synthesis
model-wide circuit transcoding
floating-point replay-order work
CUDA circuit kernels
405B implementation work
```

After rejection, a bounded synthesizer may be retained only as an auxiliary optimization for explicitly restricted structured matrices. Reopening the primary track requires a materially different query-time information source or execution dependency, with original/cold-data probes fully charged.

## Claim boundary

Phase A/B information-capacity and finite-domain reference evidence, ceiling E1. The Gate does not prove that every real checkpoint is incompressible and does not establish an online probe, traffic, latency, or hardware lower bound. Cold checkpoint access, actual Transformer operation replacement, 405B execution, 8 GiB GPU behavior, CUDA, PCIe, SSD, TTFT, and tokens/second remain **NOT TESTED**.
