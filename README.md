# Mici

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://github-pages.ucl.ac.uk/Mici.jl/dev/)
[![Build Status](https://github.com/UCL/Mici.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UCL/Mici.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/UCL/Mici.jl/graph/badge.svg?token=Y487DP8WI5)](https://codecov.io/gh/UCL/Mici.jl)

Mici.jl is a Julia package for Markov chain Monte Carlo methods based on Hamiltonian dynamics.
It implements the `AbstractMCMC` interface and works with `LogDensityProblems`-style target models.
The package is inspired by the Python Mici project, but this branch keeps the Julia-side integrator
layer and system layer explicit so the hot path stays under Mici's control.

## Architecture

The current branch is organized around a few small layers:

- `PhasePoint` stores position, momentum, cached log density, cached gradient, and cache validity.
- `AbstractSystem` and `EuclideanSystem` cover the existing unconstrained Hamiltonian path.
- `LeapfrogIntegrator` is the default hand-rolled Euclidean integrator.
- `CompositionIntegrator` is a Mici-native coefficient-based two-flow composition family.
- `ConstrainedEuclideanMetricSystem`, `ConstrainedLeapfrogIntegrator`, and `ConstrainedCompositionIntegrator`
  implement the current constrained path without changing the existing phase-point or sampler plumbing.
- GNI is kept as an optional internal backend for Euclidean tractable-flow experiments, but it is not part
  of the public export surface on this branch.

The constrained layer is the first step toward a small "constraint problem" abstraction, in the same spirit
that `LogDensityProblems.jl` provides a minimal interface for target densities. The idea is to define a
compact problem interface and compile that into a concrete Mici system that the integrators can run against.
That shape should stay contained enough that it could later be lifted into its own package, or even shared
with a future `Manifold.jl`-style geometry layer.

## Constrained systems

The constrained API currently models an embedded manifold through:

- a constraint residual function
- a Jacobian for that constraint
- a Euclidean metric on the ambient space
- a projection solver onto the manifold
- a cotangent-space projection for the momentum

This branch does not yet include a constrained GNI backend. The constrained code path is native Mici code.
If GNI is used in the future, it should be delegated to explicitly as a backend implementation, not as a
dependency of the public constrained API.

## Installation

To install the latest development version of the package, open a Julia REPL, enter package mode with `]`,
then run:

```julia
add https://github.com/UCL/Mici.jl
```

## Torus example

The repository includes a runnable torus example that mirrors the spirit of the Python Mici README while
using the current Julia constrained path.

Run it from the repository root with:

```bash
julia --project=. examples/torus_constrained_hmc.jl
```

The script samples on a torus using `ConstrainedEuclideanMetricSystem` and `ConstrainedLeapfrogIntegrator`,
writes the sample cloud to `torus_samples.csv`, and prints a short summary including acceptance rate and
constraint violation.

The full script is in [examples/torus_constrained_hmc.jl](examples/torus_constrained_hmc.jl).

## Tests

Run the package tests from a local checkout with:

```bash
julia --project=. test/runtests.jl
```

## Documentation

Development documentation is published on GitHub Pages:

https://github-pages.ucl.ac.uk/Mici.jl/dev/

## License

The package is released under an MIT license. See [LICENSE](LICENSE).
