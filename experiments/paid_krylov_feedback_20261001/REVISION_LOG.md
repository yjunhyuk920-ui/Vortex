# Revision and preservation log

1. PLAN.md was registered before the first numerical run. The initial compiler,
   summary and independent-review receipt/script are preserved under
   history/initial_counter_inventory. Initial numerical source/trace files remain
   byte-identical; CHECKSUMS.sha256 covers them.
2. Independent review verified the algebra/native domain and flagged incomplete
   operation counters. Added postcheck decoder visits/XORs, observer work, matrix
   postcheck work/comparisons, and explicit per-step/readback bounds. Removed an
   unused scratch counter. No arithmetic or recurrence algorithm changed.
3. Fresh counter-audit replay leaves all 32 numerical files identical. Summary
   event counters were replaced by the complete revised summary, with the old
   summary retained. Partial event counts are never called physical costs.
4. COST_MODEL.md specifies a separate conservative fixed-array word-RAM bound,
   including branches/indexing and original-state decoding. It is derived, not
   Python timing or a target hardware result. Its independent supplement is
   retained with the final review records.
5. The review script was made portable by resolving verify.py beside itself and
   requiring --out; the original reviewer script remains unchanged in history.
   Additional n<=2/full-rank/boundary checks are explicitly post-registration
   independent audit checks, not silently added to the registered four-case run.
