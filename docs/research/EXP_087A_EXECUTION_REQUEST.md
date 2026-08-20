# EXP-087A execution request

This commit intentionally triggers the already-frozen quadratic BF16 residual-generator workflow without changing its checkpoint, prompt population, program count, context width, polynomial degree, resource equations, success thresholds, or stop rule.

Reason for execution:

- EXP-088B later rejected retain/zero cross-layer page masks because only the full mask was exact.
- The frozen EXP-087A language is materially richer: it attempts to replace the complete composed SwiGLU weight effect with a checkpoint-static non-affine finite-word program.
- The scientific public-checkpoint Gate was not present in the branch history, so the mechanism cannot be treated as accepted or rejected until the pinned workflow runs and commits its raw evidence.

The result remains scoped to the complete layer-0 DEV-W MLP. A pass would authorize only a complete-MLP operation-replacement Gate; it would not establish a complete Transformer transition, TARGET-W execution, 8-GiB residency, or 4B-class latency.
