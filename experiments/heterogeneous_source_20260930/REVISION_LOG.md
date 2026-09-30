# Revision log

## Revision 02: signed-zero and conservative-budget precision

2026-09-30 UTC. Independent proof review identified an overstatement in report
section 7.3. A suffix of +0 products maps -0 to +0, so an interval [-0,+0] can
coalesce even though it is not a native-word singleton. The initial report and
its original document-check result/checksum manifest are preserved under
`history/revision_01/` before this correction.

The corrected reduction distinguishes the cases:

- A positive integer prefix sum requires the exact native-word singleton
- A zero prefix sum can admit [-0,+0], which still determines the unique numeric
  binary matrix-vector answer zero

The source remains an exact numeric binary MatVec producer on that family;
this remains an obligation reduction, not a general lower bound. The endpoint
algorithm already compared complete native words and explicitly retained both
zero signs, so its algorithm/proof is unchanged. No numerical experiment,
model run, threshold or performance result was changed or introduced.

The same independent review identified two additional bounded clarifications:

- Section 6's read/FMA inventories are conservative upper bounds. Its displayed
  fallback allowances now apply to that conservative budget, rather than being
  necessary conditions on every actual guarded schedule. Guard failures can
  skip a suffix trial. The formulas and thresholds are unchanged
- The suffix length is explicitly constrained to 1<=b<=N, and section 7.1 states
  b>=1 before dividing by b*u; the empty suffix is identified separately as
  the identity
- The binary-reduction witness explicitly uses suffix coefficients +1 and input
  words +0, and calls the identity domain the words from +0 upward. This avoids
  relying on an ambiguous numeric-zero sign in the wording

The complete original manifest population is archived beside its manifest,
which can be checked from `history/revision_01/`. Relative links in the archived
report retain their original root-package context; the live report contains the
current local links. A new root manifest will cover all retained files after
review finishes.
