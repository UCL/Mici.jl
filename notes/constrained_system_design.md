# Constrained system and integrator design note

## Summary
Extend Mici with a constrained-system and constrained-integrator family while keeping the existing Euclidean `PhasePoint`, `AbstractSystem`, and MCMC plumbing intact.

The constrained layer should be Mici-owned and Julia-native, with a small public interface that is expressive enough to support manifold-style problems in the spirit of `LogDensityProblems.jl`, but focused on MCMC execution rather than general geometry.

## Design constraints
- The constrained features must not depend on GNI.
- GNI may be used as an optional backend when explicitly selected.
- When the GNI backend is active, Mici may delegate tractable subflow execution and any compatible projection operations to GNI/GE internals.
- The public constrained API still belongs to Mici, not to GNI.
- Keep the implementation tightly contained so the constrained layer could later be extracted into its own package, or potentially upstreamed into a `Manifold.jl` subpackage.

## Key changes
- Add a minimal constrained-problem interface for model-side declaration of:
  - constraint residuals
  - constraint Jacobians
  - cotangent-space projection
  - projection/retraction onto the manifold
  - any geometry helpers needed by the constrained integrators
- Compile that interface into a concrete constrained-system object used by samplers and integrators.
- Add constrained integrator types with their own `step!` methods, rather than forcing every path through one generic wrapper.
- Preserve the current default Euclidean hand-rolled integrator path.
- Add explicit GNI-backed constrained integrator variants only as optional backends.
- Keep backend delegation behind Mici integrator types so users do not pay GNI ceremony unless they opt in.

## Unit tests
Use pyMici as the behavioral reference for the constrained layer. Add tests covering:
- constraint-system construction and trait access
- projection and cotangent-space projection correctness
- reversibility checks for constrained steps
- cache refresh / invalidation behavior after constrained integration
- backend parity between the native constrained path and any GNI-backed constrained path
- regression coverage for buffer reuse and absence of unintended state mutation

## Packaging / containment
- Keep the constrained code in a small number of files/modules with narrow dependencies.
- Avoid coupling the new layer to the current Euclidean implementation beyond the shared `PhasePoint` and `AbstractSystem` contracts.
- Make the interface stable and compact so it can be lifted into a separate package later if needed.

## Assumptions
- The first implementation pass should prioritize clean API shape and isolation over broad coverage of every possible constrained method.
- GNI support should be additive, not structural: it can optimize execution, but it should not define the constrained API.
- The initial goal is parity with the constrained concepts present in pyMici, not a wholesale port of its entire hierarchy.
