# Handoff — native output envelopes

Parent66c01825635a357f86f9dfa5369088d7a239c92d/PR146.
Worktree C:/ChatOnSteroids/Vortex-output-envelope.
Branch research/native-output-envelope-20260908.

THEORY_STATUS=NOT_ESTABLISHED;HARDWARE_STATUS=NOT_TESTED;
CORE_ADMISSION=false;FULL_MISSION_O1_O6=OPEN;
HANDOFF_STATUS=IN_PROGRESS until actual commit/push/readback.

Read docs/REPORT_KO.md and OBLIGATIONS.md before test counts. Native-tree endpoint
constructor/query returns full BF16 vector with paid direct rows on uncertified
packets. Pinned SmolLM2-135M12matrices,96syntheticvectors,101376outputs,
432packets:0mismatch,0certified,0even-oracle-shared packets,fulloriginalreads.
No original HF forward/real hidden trace or target performance claim.

Sources/outputs are stored directly. Upstream weights and derived .npz min/max
arrays stay local/ignored and regenerate from pinned Range data/frozen SHA256.
portable_replay regenerates38numerical files. Initial source-before-payload
snapshot is retained; post-result script additions are not backdated.

Other session's correlation_source was untracked at startup and is now separately
committed as PR147/378fcd584330d17f9d144f64ccbc01721903ea6a. It was not changed or
included here. See FRONTIER_SYNC.md. Local-only mantissa_source was not imported.
Four old root entrypoints are preserved byte-for-byte in
docs/research/history/pre_output_envelope_20260908/.

User-requested session_finish was called and returned HELD/automatic goal pending.
Follow subsequent in-scope queued instructions and call again before final.
Do not invent an exact five-minute interval, release, commit or full-goal success.
Post-commit evidence belongs in a later receipt/PR, not a self-referential SHA.
