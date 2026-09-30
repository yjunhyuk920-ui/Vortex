# Bounded-representation refinement, before validation 02

Validation 01 passed with unrestricted exact integer offsets. Its source is
preserved in `results/source_01`; its log and JSON are unchanged.

For a normal-binade interval, a live offset is the difference of two supported
accumulator lattice indices. Hence its magnitude is at most 2^(p-1). A parity
with an empty domain needs no offset. This refinement canonicalizes such offsets
to zero, so a production representation never retains irrelevant huge integers.

The same frozen validation population will be rerun without changing acceptance
criteria. For nonempty domains test the finite bound on every offset and endpoint.
The Python Fraction implementation still is an exact reference, not a 24-byte
runtime implementation or a measured speedup. The packed representation is a
derived payload bound only.
