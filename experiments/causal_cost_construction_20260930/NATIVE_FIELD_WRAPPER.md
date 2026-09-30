# Conditional numeric-code/native wrapper

Scope: a finite supplemental construction, not an admitted source or a proposed
implementation campaign. No independent cheap coded source survives the parallel
investigation. This note supplies its native bridge only if such a source is
later constructed. Full O1-O6 and the fixed mission remain open.

## Exact finite program

Assume finite BF16 coefficients/current inputs, FP32 RNE FMA with gradual
underflow and seed +0 per reduction stream. No FP status flags are observed;
nonfinites, other seeds, FTZ/DAZ and unknown internal precision are not covered.
Take the fixed prime p=2^31-1.

1. Constructor: scan every original coefficient and record row finiteness.
   For a finite nonzero row, define v_i=min v2(nonzero w_ij), exact
   A_i=sum|w_ij| and the nonnegative integer S_i=A_i/2^v_i. Such a row is
   statically eligible only if S_i<=2^24-1.
   Decide eligibility with fully paid bounded multiword constructor scratch;
   do not cast an oversized S_i to uint32. For N<=16384, the exact A_i in units
   of 2^-133 and the integer S_i each need at most 275 bits, so two five-limb
   uint64 arrays are a sufficient explicit scratch reservation, plus scalar
   decoding/shift state. A_i, v_i, shift/division, comparisons and writes are
   charged. Other widths require an explicitly enlarged limb bound
   
   An illustrative packed row record is eight bytes: int16 valuation, one-byte
   flags, one-byte pad, uint32 norm field. For an eligible nonzero finite row,
   the valuation and S_i are stored exactly. An ineligible row stores the
   ineligible flag and unused safe sentinels (zero valuation and zero norm),
   never a wrapped/truncated norm; it always takes original fallback. All-zero
   rows use a distinct flag with unused zero valuation/norm fields, since their
   minimum nonzero valuation is undefined. Nonfinite rows likewise use a
   distinct fallback flag and unused zero fields. Dispatch checks flags before
   reading a valuation/norm as numeric data
2. For eligible rows, make another paid pass over their original coefficients
   and convert a_ij=w_ij/2^v_i modulo p. These are 31-bit residues stored in
   four-byte words in the direct representation. The original checkpoint is
   retained for fallback. Building any coded layout from these residues has its
   own additional explicit constructor/storage cost
3. Query: scan x to check finiteness and obtain v_x=min v2(nonzero x_j) and
   Xtilde=max|x_j|/2^v_x. If x is nonzero and Xtilde>2^24-1, no nonzero row can
   pass this guard; take the original path. Otherwise a second pass converts
   x_j/2^v_x into 31-bit residues. Both vector reads, conversion and four-byte
   residue writes are paid. A zero finite input has no v_x: after checking row
   flags, eligible and all-zero rows return +0, while ineligible/nonfinite rows
   still use original fallback. No unused valuation/norm enters arithmetic
4. Read all row records and dispatch on flags. Ineligible and nonfinite rows
   always use the original fallback; an all-zero row uses its special path
   only for finite inputs under this declared +0-seed ABI. Only a statically
   eligible row's exact valuation/norm fields enter the following test:
   -149<=e_i=v_i+v_x<=104 and S_i*Xtilde<=2^24-1. The product fits 48 bits.
   This is a fixed finite word comparison, not an arbitrary-precision oracle
5. Evaluate the supplied exact numeric GF(p) source on eligible rows. It may
   compute all eligible outputs, including rows whose dynamic guard fails; that
   complete cost is charged. Do not assume selective output support for free
6. For an accepted row, center the residue in [-(p-1)/2,(p-1)/2], yielding z_i.
   Encode z_i*2^e_i exactly as FP32, with +0 for z_i=0. Apply any specified
   original output cast. Rows failing the guard execute their original native
   reduction and output procedure; no unpaid repair or future result is used

For eligible rows, each integer coefficient has magnitude <=S_i<=2^24-1 and
accepted queries satisfy |sum a_ij*x_j/2^v_x|<=2^24-1. Since p/2 exceeds that
bound, centered modular recovery is unique. Every original exact FMA prefix is
an integer multiple of 2^e_i with magnitude at most (2^24-1)*2^e_i. It is an
exactly representable FP32 word. Induction therefore identifies the decoded
sum with every prescribed original FMA step. RNE exact cancellation yields +0;
a +0 seed followed only by signed-zero products also stays +0.

This is not a claim about arbitrary tensor-core reduction order or precision.
The same magnitude argument covers a supplied FP32 sum tree when all its exact
partial sums are subset sums, but zero signs and seeds must separately match
that actual tree. No BF16/FP32 reassociation is assumed outside the guard.

Ordinary field multiplication is implementable with a uint64 product (<2^62)
and Mersenne reduction: split at bit31, add the two pieces, split/fold once more,
then conditionally subtract p. Every shift/mask/add/compare and multiplication
is charged; a field operation is not treated as a native FMA at equal time.

## Complete conditional source cost

For M rows, N columns and P=MN, let E be the statically eligible coefficient
count and F(x) the original coefficients in rows falling back. Let C_code,
S_code and Q_code(x) be the ACTUAL finite constructor, storage and query
inventories of a separately supplied numeric GF(p) layout. They are unresolved
parameters here, not a cheaply supplied subroutine.

Constructor inventory includes:
- 2P original bytes read, fully paid bounded multiword norm/valuation and
  eligibility construction (including the two five-limb arrays above at
  N<=16384, scans/shifts/comparisons), and 8M flagged/sentinel record writes
- 2E original bytes reread, conversion and up to 4E direct residue writes
- all C_code work, original 2P bytes retained, S_code and construction workspace

Query inventory includes:
- two N-word input reads and N residue writes (or the explicitly counted cache
  equivalent), 8M metadata bytes, guard/address/dispatch work
- all Q_code(x) source probes, physical transfers, instructions and workspace
- F(x) original coefficient reads/products and complete original row reductions
- centered decoding, exact word encoding/cast, output writes and synchronization
- downstream original state/RNG work and full load/prefill/TTFT obligations

If a code accesses a fraction beta of its direct four-byte field source, its
coefficient-byte contribution relative to original two-byte BF16 is 2*beta*E/P.
Fallback adds F(x)/P; row metadata adds 4/N, before input/output/address/cache-line
traffic and all other work. For the candidate-specific goal of tenfold
coefficient-byte reduction, even E=P and F=0 require beta<0.05 after overhead.
That byte goal is not a universal architecture requirement. Field/native
arithmetic and all movement/overhead must fit the complete target-time schedule;
neither term must universally fall by 90%. Probe or byte fractions are not
latency ratios. See [the correction record](ARCHITECTURE_NEUTRALITY_CORRECTION.md).

GF(2) rank-one parity is NOT this source. It omits integer carries. A binary
coefficient-plane parity gives the dot product only modulo two; reconstructing
integer counts requires an independently paid procedure. Neither the wrapper
nor the grid guard provides that procedure from parity alone.

## Coverage remains absent

For N=16384 with every weight and activation equal to 255/128, S_i=16384*255,
Xtilde=255, and the guard product is 1065369600, well above 2^24-1. Prefix259
has exact scaled numerator16841475, an odd value above2^24 and genuinely rounds.
This is a scalar witness, not an asserted released-checkpoint causal state.
No invariant/coverage theorem excludes such failures on the fixed mission.

The parallel source worker found no surviving explicit numeric coded layout.
Thus this finite wrapper is retained only as a conditional native-lift artifact,
with NO core admission, model/backend experiment or dominant-cost claim.
