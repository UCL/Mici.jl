# Mici

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://github-pages.ucl.ac.uk/Mici.jl/dev/)
[![Build Status](https://github.com/UCL/Mici.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UCL/Mici.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/UCL/Mici.jl/graph/badge.svg?token=Y487DP8WI5)](https://codecov.io/gh/UCL/Mici.jl)

Mici.jl is a Julia package implementing _Markov chain Monte Carlo_ (MCMC) methods 
for generating approximate samples from target probability distributions of interest,
for example in Bayesian inference applications.
The package is particularly focused on MCMC methods which simulate Hamiltonian dynamics on a manifold.
Mici.jl implements the [AbstractMCMC](https://turinglang.org/AbstractMCMC.jl/stable/) interface
and can be used to generate samples from distributions specified using the [`LogDensityPoblems.jl` interface](https://www.tamaspapp.eu/LogDensityProblems.jl/stable/).
The design of the package is inspired by the corresponding [Python Mici package](https://github.com/matt-graham/mici).

## Installation

To install the latest development version of the package, open a [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/), enter the package manager by typing `]` then run

```
add https://github.com/UCL/Mici.jl
```

## Documentation

[Documentation for the current development version of the package is available on GitHub Pages.](https://github-pages.ucl.ac.uk/Mici.jl/dev/)

## Tests

Packages tests are included in the [`test`](test) directory and can be run from a local clone of the repository by launching a REPL in the root of the repository, typing `]` to enter the package manager then running `test`.

## License

The package is released under an [MIT license](LICENSE).