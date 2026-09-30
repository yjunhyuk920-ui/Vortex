# EXP100A classification repair, registered before new tests

This is the previously specified control repair, not a new execution principle.
The committed workflow `repair-exp100a-control-classification.yml` contains an
unapplied repair, while the checked-in runner still treats any resource-empty
frontier as an integrity failure. The old raw run is preserved without alteration.

Hypothesis: resource/bounded-search exhaustion and missing search coverage can be
classified separately, without changing schemes, search bounds, costs or gates.
Controls must distinguish (a) complete population with empty frontiers, (b) a
missing search, (c) a same-sized but wrong identity population and (d) a key whose
SearchResult identity does not match it. Genuine identity/catalog controls remain.
Use exact expected key sets rather than only counts in the old proposed patch.

No original search rerun, target experiment, model generation or Actions is needed.
Tests run on CPU with the existing recorded result metadata. Derived workspace
diagnostics are not universal lower bounds and cannot prove every filtered search
was exhaustive. No promotion follows from removal of a misclassified error.

Acceptance: targeted old and new tests pass, code compiles, the original result and
scientific input bytes remain unchanged. Update only code/repair evidence and
current continuation pointers. The original INVALID receipt remains historical;
a new metadata audit must not masquerade as a fresh full experiment.

Budget: 60 CPU seconds for focused tests, no GPU, no large search. This repair
supports O6 auditability; O1–O5 and the missing native producer remain open.
