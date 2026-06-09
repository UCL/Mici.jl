# Allocation Trace Report

Generated from:

- `julia --track-allocation=user benchmark/integrator_allocation_trace.jl`
- warmup run followed by `Profile.clear_malloc_data()` and 5 measured native steps plus 5 measured GNI steps

Raw trace files:

- [benchmark/integrator_allocation_trace.jl.26796.mem](/mnt/c/Users/guile/Documents/code/Mici.jl/benchmark/integrator_allocation_trace.jl.26796.mem)
- [src/system.jl.26796.mem](/mnt/c/Users/guile/Documents/code/Mici.jl/src/system.jl.26796.mem)
- [src/state.jl.26796.mem](/mnt/c/Users/guile/Documents/code/Mici.jl/src/state.jl.26796.mem)
- [src/gni.jl.26796.mem](/mnt/c/Users/guile/Documents/code/Mici.jl/src/gni.jl.26796.mem)
- [src/integrator.jl.26796.mem](/mnt/c/Users/guile/Documents/code/Mici.jl/src/integrator.jl.26796.mem)
- [/home/jsimba/.julia/packages/GeometricIntegrators/3IYSU/src/integrators/splitting/splitting_integrator.jl.26796.mem](</home/jsimba/.julia/packages/GeometricIntegrators/3IYSU/src/integrators/splitting/splitting_integrator.jl.26796.mem>)
- [/home/jsimba/.julia/packages/GeometricEquations/22YER/src/odes/sode.jl.26796.mem](</home/jsimba/.julia/packages/GeometricEquations/22YER/src/odes/sode.jl.26796.mem>)

Sanity check from the same run:

- Native `@allocated` for one `step!`: `480` bytes
- GNI `@allocated` for one `step!`: `29904` bytes

## Native path

The native leapfrog trace is concentrated in the exact flow update:

| File | Line | Bytes in trace | Why it allocates |
| --- | --- | ---: | --- |
| [src/system.jl:25](/mnt/c/Users/guile/Documents/code/Mici.jl/src/system.jl:25) | `Φ₁!` | `800` | Broadcasted momentum update in the half-step flow |

The rest of the repo-side native trace is zero after warmup in this run, which means the measured cost is not coming from the benchmark harness or the phase-point setup.

## GNI path

The repo-side GNI wrapper itself is not where the big allocation cost lands after warmup. The measured step cost is dominated by the package internals:

| File | Line | Bytes in trace | Why it allocates |
| --- | --- | ---: | --- |
| [GeometricIntegrators splitting_integrator.jl:67](</home/jsimba/.julia/packages/GeometricIntegrators/3IYSU/src/integrators/splitting/splitting_integrator.jl:67>) | `cache(int).q .= sol.q` | `3840` | Copies the previous solution into the splitting cache for every split stage |
| [GeometricIntegrators splitting_integrator.jl:68](</home/jsimba/.julia/packages/GeometricIntegrators/3IYSU/src/integrators/splitting/splitting_integrator.jl:68>) | `cache(int).t = ...` | `240` | Computes the stage time |
| [GeometricIntegrators splitting_integrator.jl:71](</home/jsimba/.julia/packages/GeometricIntegrators/3IYSU/src/integrators/splitting/splitting_integrator.jl:71>) | `solutions(problem(int)).q[...]` | `5040` | Calls into the stage solution function for each split component |

The wrappers in [src/gni.jl:141-145](/mnt/c/Users/guile/Documents/code/Mici.jl/src/gni.jl:141) are still relevant for data movement, but in the measured trace they were not the main byte sinks. The heavy work is in the `GeometricIntegrators` splitting loop above.

## Interpretation

The current benchmark run says:

- the manual leapfrog path is spending most of its visible allocation budget in our own flow update
- the GNI path is spending its allocation budget inside the generic splitting integrator cache/callback loop
- the constructor/setup path is not the issue here; the per-step repeated copy-and-dispatch work is

If you want to inspect the raw lines, the `.mem` files above are the authoritative artifact.
