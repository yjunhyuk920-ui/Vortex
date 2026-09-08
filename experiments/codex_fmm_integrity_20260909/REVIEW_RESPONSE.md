# Primary response to post-repair independent review

Accepted the concrete classifier gap. Added a mutation test for paired direct/
rejected undercount, partial nonempty depth list, unknown termination and zero
raw oracle evaluation. All four subcases were actually red; post_review_red.log
preserves them. HARDENING_PREREGISTRATION precedes the implementation change.

Now expected direct count at depth d is states[d]*(1+d*number_of_static_widths);
expected oracle count is states[d]. The classifier takes the actual search
contract, checks exact populated/evaluated depths, and distinguishes terminal
zero from completed maximum depth. The oracle counter increments only after
its evaluator returns. Accounting uses raw evaluations, not retained survivors.

The primary reran all11 legacy functions and6 focused tests, then the full gate
and independent saved-artifact comparison. 84,712 direct/15,608 oracle evaluations
reconcile in all50 cells, zero integrity/control mismatches. Numerical costs and
selected plan bytes are unchanged. The final hardened_run is authoritative.

The review was of the first correction, not the later hardened source. No claim
of a new agent approval. Primary source review and actual execution support the
final integration. These guards detect the identified structural omissions;
they are not an adversarially tamper-proof execution proof. Hashes and preserved
source/results remain required. Scope and O1-O6 status are unchanged.
