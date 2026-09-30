# Revision log

## Revision 02: independent symbolic review and scope clarifications

2026-09-30 UTC. The first sealed report and complete initial document-check
manifest population are preserved under `history/revision_01/`. The archived
report SHA-256 matches its original receipt. Archived relative links retain
their original root-package context. No numerical/model/GPU experiment, threshold
or runtime result is introduced or changed.

Changes after that initial seal:

- State explicitly that RNG consumption-sensitive contracts keep required draw
  counts and ordered observable effects as roots; final RNG state alone is not
  claimed sufficient for arbitrary effect contracts
- Make supplied finite control unrolling/predication and safe total Boolean
  primitive-relation evaluation explicit. Invalid local tuples return false;
  defined reference errors retain their actual semantics; inactive instructions
  preserve effect-state and have a defined unused output
- Make each L_t a uniform bound over bucket assignments, so the displayed
  construction inventory is a sufficient bound
- Finish the width argument for the actual compiler: append the N symbolic inputs
  to the elimination order. Bags in that final part have at most N variables,
  so the N+1 bag occurs during accumulator elimination and creates the asserted
  message over at least N remaining words
- Restrict the fully dirty witness to changed dense input vectors and rounding.
  It does not itself instantiate a one-token transition in a released checkpoint
- Explicitly link the existing compiled-transducer rejection so finite quotient
  enumeration cannot be mistaken for a newly admitted source

The exact bounded witness-table compiler and positive-prefix arithmetic were
not found incorrect in independent manual review. Review and these clarifications
do not establish a compact native source or close any whole-mission obligation.
