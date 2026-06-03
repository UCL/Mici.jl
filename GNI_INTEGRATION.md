# GeometricIntegrators integration design

This branch integrates GeometricIntegrators.jl as an implementation detail of
Mici's existing HMC state machine. Mici's `PhasePoint` remains the source of
truth for sampler state, including accept/reject behavior and cached target
density information.

## Object ownership

`GNISplittingIntegrator` is constructed during the initializing
`AbstractMCMC.step` call, at the same point as the native leapfrog integrator.
It owns all long-lived GeometricIntegrators/GeometricEquations objects:

- a combined `(q, p)` buffer stored as one vector for the SODE state,
- the immutable `SODEProblem`,
- the selected splitting method,
- the reusable `SolutionStep`,
- the GeometricIntegrators `GeometricIntegrator`.

No GNI problem, integrator, solution step, or Mici-owned phase buffer is created
inside the trajectory integration loop.

## PhasePoint synchronization

Each integration step follows this data flow:

1. Copy `PhasePoint.q` and `PhasePoint.p` into the adapter buffer.
2. Copy the adapter buffer into the current GNI `SolutionStep`.
3. Run one `GeometricIntegrators.integrate!` step.
4. Copy the resulting GNI state back into the adapter buffer.
5. Copy the buffer back into the proposed `PhasePoint`.
6. Mark the `PhasePoint` cache invalid with `refresh!`.

This avoids mutating the accepted `PhasePoint` during proposal integration. The
Metropolis transition still copies the accepted point into
`proposed_phase_point`, integrates only the proposal, and copies it back to the
current point only on acceptance.

## Initial conditions

GeometricEquations `EquationProblem` objects are immutable and store the initial
conditions used to construct the problem. MCMC proposals need different initial
conditions after momentum refresh and accept/reject decisions, so the adapter
does not rebuild or mutate the problem. Instead, it resets the reusable
`SolutionStep` from the current `PhasePoint` before every GNI step.

## Cache validity

The GNI flows use scratch `PhasePoint`s to call Mici's existing Hamiltonian flow
functions. Scratch points are refreshed whenever data is copied from the SODE
buffer. The sampler's proposed `PhasePoint` is refreshed after GNI copies a new
position back, so later calls to `h`, `logdens`, or `grad` recompute target
values at the final proposal position.

Momentum-only updates do not require target cache invalidation, matching the
native Mici behavior.

## Current scope

The first adapter targets Euclidean/separable HMC through `StrangA`, which is
equivalent to the native leapfrog ordering used by `LeapfrogIntegrator`. The
transition code uses a small `step_size` interface so native and GNI integrators
can share the same Metropolis integration machinery without assuming a concrete
field layout.
