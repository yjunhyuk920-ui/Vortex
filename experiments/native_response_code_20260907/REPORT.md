# Native response-code source — 2026-09-07

Baseline: yjunhyuk920-ui/Vortex PR135, d9e19c5ab060488c8788acbcfabc4d0933870aa3.
The prior correlated-jet investigation was a local artifact, not a new remote commit.

**Finite-domain whole-SwiGLU replacement constructed; not a 405B executor.**
THEORY_STATUS=NOT_ESTABLISHED; HARDWARE_STATUS=NOT_TESTED; CORE_ADMISSION=false;
THREE_QUALIFYING_NEW_PRINCIPLES=false; FULL_MISSION_O1_O6=OPEN.

The unchanged mission is arbitrary public unmodified HF dense405B, batch1, one GPU
with total peak<=8GiB, original output/RNG/required successor state, same-machine
native4BQ4 p50<=1.2/p95<=1.5 and existing TTFT. All preparation, storage, movement,
addresses, computation, state and verification count. No weight/input modification,
training, hidden remote compute or free fallback. No success is claimed by commit.

## Constructed execution and exactness

Prior smooth derivative arguments do not cover native finite-word programs. We
compared three explicit finite-native representations: (A) two-input-side XOR response
factorization, (B) Boolean Mobius output-word coefficients, and (C) ordered constant
response intervals. These are bounded comparisons using known mathematics, not a
claim of three new goal-qualifying principles. Their exhaustive source-generation
cost is checked before any large backend is admitted.

Split the actual current input into a,b. For L left tuples, R right tuples and m
BF16 output words, define the binary matrix

    T[a,(b,j,t)] = bit t of the original native output word j.

The constructor actually runs the original computation on all pairs and performs
exact GF(2) elimination, selecting original pivot rows. It creates T=B*C exactly.
The loaded-file runtime evaluates

    output_word_j(a,b) = XOR over k with B[a,k]=1 of C_word[k,b,j].

This is XOR of the complete stored output bits, not floating-point addition of
approximate answers. The constructor, actual domain search, address calculation,
serialization, SHA-256 verification, Python and independent C evaluators are supplied.
The C evaluator owns neither original weights nor an activation table and performs
no floating-point arithmetic. Original synthetic weight files remain preserved.
Unknown input bits are rejected, not quantized and not counted as supported cases.

Proof: Gaussian elimination preserves T=B*C over GF(2); evaluating its indicated
row/column gives every output bit. Full finite-domain reconstruction establishes
the upper rank, and separately verified nonsingular binary pivot minors establish
the lower rank. A pure block with the same final output can replace that block in
an unchanged suffix, provided all future calls satisfy the domain and no eliminated
intermediate is observable. The all-future domain, whole Transformer state/RNG and
actual HF/CUDA ABI obligations have not been established here.

## Numeric ABI and preregistered data

BF16 weights and inputs; distinct FP32 products; balanced FP32 addition trees;
BF16 stores after projection, SiLU, gated product and down projection. No FMA.
SiLU is a frozen torch2.10.0+cpu BF16 one-dimensional primitive table, not a proof
of correctly rounded mathematical exp. On all826 encountered gate values, singleton
SiLU and table-generation batch results agree. Independent scalar arithmetic shares
only this declared primitive; it is not an independently implemented exp library.

n=m=4, h=32, Q=2,4,8,16, seeds7,19; also h=256,Q=8,seeds7,19. Nonzero dense
weights are PCG64-generated integers +/-1..7 divided by8. All supported tuples are
executed. Q16 includes +0/-0 as separate bit patterns. These are synthetic domains,
not actual pretrained weights or quantization of arbitrary model activations.

10 SwiGLU fixtures:148000 queries/592000 output coordinates, zero Python/C mismatches.
544 input vectors also match independent scalar struct-based FP32/BF16 calculations.
Identity-word, signed-zero and zero response controls are separately labeled, not
model acceleration results. The first run and second run reproduce all171 files.

## Paid size and query findings

For h32, original G/U/D total768 bytes. Both seeds have the following results:

|Q|Original calls to construct|Exact rank|Program bytes|Program/original|
|---:|---:|---:|---:|---:|
|2|16|4|208|0.270833|
|4|256|16|2160|2.8125|
|8|4096|64|33368|43.447917|
|16|65536|225|468328|609.802083|

High rank is NOT a per-query full-read lower bound. A full-rank matrix can select
an original row with B=I. In these generic fixtures every query chooses one pivot
atom, but the response payload is large. Do not misreport the storage failure as
a universal query-time lower bound.

For h256,Q8: original6144 bytes, program33368 bytes; logical references p50=124,
p95=140 bytes, or2.02%/2.28% of original coefficient bytes. This is NOT measured
DRAM traffic or GPU latency. It requires4096 original block evaluations to construct.
The remaining runtime contains domain comparisons, integer addresses, bit scans and
XOR updates; a small payload alone is not total runtime performance.

The reference byte count is:40-byte header,2n input bytes,2 bytes per actual domain
comparison,ceil(r/8) left code bytes,2mk atom bytes,2m initialization bytes,4mk output
read/write bytes. SHA/loader copies, weights, source table, temporary arrays and
elimination integers are additional cold memory/work. No total peak measurement.

Source size is L*ceil(r/8)+2*m*R*r plus header/domain/hash, L=Q^(n/2),R=Q^(n/2).
The constructor performs Q^n original evaluations. Main-source MAC total78852096,
not including elimination/LUT/serialization/verification. Resident small weights are
not falsely charged as Q^n SSD scans. Original weights and generated representation
both count. At width16384, even Q=2 requires2^16384 evaluations in this constructor.
This rejects this enumeration schedule, not every possible representation or algorithm.

## Alternative comparisons and follow-on

B uses output(z)=XOR_{s subset z} coefficient[s]. Full Mobius inversion and explicit
queries match. All16/256/4096/65536 word coefficients are nonzero in generic Q2/4/8/16
fixtures; identity control uses9. Q8 average payload read alone is1037.970703125 B;
Q16 is5254.7266845703125 B. Exact Boolean expansion need not be sparse.

C stores exact contiguous equal-output runs in the fixed tuple order. Generic Q2/4/8
have a separate run for every input. Q16 retains61440 of65536 runs, whose start+word
payload is737280 bytes. This does not exclude another ordering or representation.

Post-main scope check (FOLLOWON.md), G=U=[1,1],D=[1]: native F(1,1)=3.515625, while
RN_BF16(F(1,0)+F(0,1))=1.4609375. Thus independent small input tables cannot simply
be added to cover a larger nonlinear block. This is not a universal composition bound.

## Decision, assumption, failure, architecture and hardware addendum

The bounded source has a complete constructor and query, but its source creation
and representation do not scale to the mission. Do not enlarge the truth table,
rank sample or address-order search as the next core round. A new constructor must
produce current native effects without enumerating all input combinations or falling
back to full transformed-weight scans. Do not assume low response rank, and do not
infer high rank implies full reads. Three target-qualifying new principles are not
established. O1-O6 remain OPEN at mission level; conditional exactness evidence is not
complete theory. Prior root ledgers are unchanged; this linked addendum records the
new scoped result. Hardware plans/reproducibility rules and all old evidence remain.

## Reproduce and persistence

Python3.13.5,numpy2.3.5,torch2.10.0+cpu,gcc14.2.0; local CPU only.

    cd experiments/native_response_code_20260907
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/run.py --out results/run
    python -m unittest discover -s tests -v
    python src/scope.py
    python src/verify.py

The output directory must be empty.20 tests passed.171 generated-file hashes matched
between two local runs. verify.py checks the canonical full-manifest SHA recorded in
results/validation.json. The source regenerates original synthetic weights, inputs,
activation table, compiled programs, response tables, per-query traces and comparison
coefficients. The full binary arrays and expanded Korean report are in the user ZIP;
remote source/summary/validation preserve a reproducible record, not a claim that all
binary bytes were embedded. Archival compression is not an inference improvement.

No public checkpoint, full Transformer KV/RNG, CUDA,405B,8GiB allocation,4BQ4 latency,
TTFT,full repository suite or Actions was run. Handoff is established only by actual
commit/ref/readback; a snapshot does not contain its own future commit hash.

Sources: PyTorch numerical accuracy https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html;
E.Manino et al. software-level floating-point verification https://arxiv.org/abs/2510.23389;
I.V.Oseledets,Tensor-Train Decomposition(2011),https://epubs.siam.org/doi/10.1137/090752286.
These support provenance and scope, not a claim that those papers solve the VORTEX mission.
