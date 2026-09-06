# Boundary-generated convolution and native-BF16 certification — 2026-09-07

Parent: PR132, `2c9ae1d3dc6c29d3da9b7da436df4eabc287e7f9`.

## Constructed object and limits

A streaming constructor tests the ORIGINAL BF16 matrix for exact Toeplitz structure,
then emits BCV1 signed/absolute packed integer polynomial sources. The query uses no
original matrix, computes exact S=Wx and A=|W||x| collectively, and certifies the
specified balanced FP32 reduction's final BF16 words using S +/- gamma_log2(n)*A.
Only equal finite NONZERO endpoint roundings are accepted. Any uncertain row causes
whole-query UNRESOLVED; no fallback/replay is counted as success.

Known displacement/Kronecker algebra is acknowledged, not claimed as a new field.
The three examined formulations are not three qualifying new core principles.
A small displacement generator for arbitrary original W was NOT constructed.

```
THEORY_STATUS=NOT_ESTABLISHED
CORE_ADMISSION=false
THREE_QUALIFYING_NEW_PRINCIPLES=false
HARDWARE_STATUS=NOT_TESTED
FULL_MISSION_O1_O6=OPEN
```

32 registered constructors:16 succeed,16 reject (all8 generic and all8 individually
perturbed controls).128 queries on DESIGNED structured controls:86 complete,42
unresolved.53916/54272 coordinate certificates match; every native FP32 value is
inside its proved enclosure.53937 native sums differ from the exact real sums.
15 tests pass.435 generated files plus the manifest reproduce byte-exactly.

A 1024 ordinary Toeplitz file is36874 bytes vs2097152 original bytes (1.7583%),
but that family completes ZERO of16 whole queries. Dominant1024 controls complete
15/16 with45062-byte files. These are planted structures, not public checkpoints.
File size is NOT physical traffic or latency. Huge integer products are NOT unit
operations; operand bit lengths, working payloads and certificate work are recorded.
Even minimal construction+8-query accounting is about14.67% in the ordinary1024 case,
before full bigint/memory/rounding work. No whole-work10x or target bound is proved.

Fixed-shift modular minors give exact LOWER bounds on displacement rank, not rank
estimates. On generic1024 controls, rank>=64 rules out source<=10% in the specific
64-bit two-factor format (8r/n>=0.5). This is not a bound on all encodings or on405B.
No public HF/Transformer KV/RNG/GPU/405B/4B latency/TTFT or Actions/full-repo suite ran.

## Full proof, raw evidence and reproduction

The checked capsule preserves14 text files: all sources/tests/preregistration,
complete Korean proof and budget report, raw per-query records, logs and SHA256s.
Binary matrices/inputs/BCV1 and per-coordinate records are recreated with the same
source and compared against the recorded435-file manifest; they are NOT all embedded
in the capsule. XZ/base64 is evidence archival, not inference compression.

```bash
python experiments/boundary_convolution_20260907/restore.py --out /tmp/vortex-boundary
cd /tmp/vortex-boundary
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python src/run.py --out regenerated
python -c "import json;from pathlib import Path;assert json.loads(Path('results/run/manifest.json').read_text())==json.loads(Path('regenerated/manifest.json').read_text())"
```

The full restored report is `docs/REPORT_KO.md`. The runtime envelope is finite nonzero
normal BF16 weights/input with explicit exponents, separate FP32 products, balanced
RNE sum and BF16 store; not CUDA/FMA semantics. Whole mission and preparation costs
are not relaxed. A session boundary and remote commit do not establish the theory.

## Decision / assumption / anti-repetition addendum

Keep only bounded auxiliary evidence. Do not expand Toeplitz controls, grow displacement
search indefinitely, treat near-structure as exact, add uncharged native row fallback,
or count coordinate success as a whole-state executor. Exact small displacement and
complete native-vector certification are UNESTABLISHED for original public checkpoints.
Earlier root decision/assumption/failure ledgers remain intact; this linked addendum
records the new conditional source and its failure gates without changing architecture.

Primary sources: Sindhwani et al., NeurIPS2015 structured transforms;
Harvey, arXiv0712.4046 Kronecker substitution;
NVIDIA Floating Point and IEEE754 dot-product ordering guide.
Remote handoff is established only by final ref/commit/PR read-back, not this text.
