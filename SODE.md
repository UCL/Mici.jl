# SODE Method Checks in GeometricEquations

This note explains why `SODEProblem(...)` can fail with

- `AssertionError: check_methods(equ, _timespan, _ics, parameters)`

and what `applicable(...)` is checking.

## Where the assertion comes from

In `EquationProblem(...)`, GeometricEquations asserts:

1. `check_initial_conditions(...)`
2. `check_methods(...)`

If either returns `false`, you get an `AssertionError`.

For `SODE`, `check_methods` verifies that each provided substep function can be called with the required signature.

## Required SODE function signatures

For split vector fields `v = (v1, v2, ...)`, each `vi` must be callable as:

```julia
vi(vbuf, t, q, params)
```

where:

- `vbuf` is a preallocated output vector to fill in-place
- `t` is time
- `q` is current state vector
- `params` is parameter object (often `NullParameters()`)

Important: these are in-place functions. They should mutate `vbuf` and return `nothing`.

## Why `applicable(...)` matters

`check_methods` uses `applicable(...)` to test callability at runtime. Conceptually it asks:

- "Can this function be called with these argument types?"

For each `vi`, GeometricEquations checks something equivalent to:

```julia
applicable(vi, vectorfield(ics.q), timespan[1], ics.q, params)
```

If this is `false` for any substep function, assertion fails.

## Common failure patterns

1. Wrong arity

You wrote `vi(x)` instead of `vi(v, t, q, params)`.

2. Wrong style

You return a new vector (`vcat(...)`) instead of mutating `v` in-place.

3. Wrong argument order

Constructor args to `SODEProblem` are in wrong positions.

4. Hidden type mismatch

Your system derivative methods expect a different state type than `q` passed by `SODE`.

## Practical debugging checklist

1. Confirm signatures

```julia
methods(v1)
methods(v2)
```

2. Confirm callability directly

```julia
applicable(v1, similar(x0), t0, x0, GeometricEquations.NullParameters())
applicable(v2, similar(x0), t0, x0, GeometricEquations.NullParameters())
```

3. Probe constructor in isolation

```julia
prob = SODEProblem((v1, v2), (0.0, 1.0), 0.1, x0)
```

4. If you changed code and behavior looks stale, restart REPL or use Revise.

## Summary

`check_methods` is a strict interface gate.
If your split functions are callable as `vi(v, t, q, params)` and mutate `v` in-place, the assertion should pass.
