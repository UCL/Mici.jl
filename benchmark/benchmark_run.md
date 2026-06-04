jsimba@Simba:/mnt/c/Users/guile/Documents/code/Mici.jl$ julia --project=benchmark benchmark/integrator_step_benchmarks.jl
Info Given Mici was explicitly requested, output will be shown live
┌ Warning: attempting to remove probably stale pidfile
│   path = "/home/jsimba/.julia/compiled/v1.12/GeometricIntegrators/fhpp1_cpbtc.ji.pidfile"
└ @ FileWatching.Pidfile ~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/share/julia/stdlib/v1.12/FileWatching/src/pidfile.jl:247
┌ Warning: attempting to remove probably stale pidfile
│   path = "/home/jsimba/.julia/compiled/v1.12/RungeKutta/sbsrk_cpbtc.ji.pidfile"
└ @ FileWatching.Pidfile ~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/share/julia/stdlib/v1.12/FileWatching/src/pidfile.jl:247
┌ Warning: attempting to remove probably stale pidfile
│   path = "/home/jsimba/.julia/compiled/v1.12/CompactBasisFunctions/yEj0J_cpbtc.ji.pidfile"
└ @ FileWatching.Pidfile ~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/share/julia/stdlib/v1.12/FileWatching/src/pidfile.jl:247
┌ Warning: attempting to remove probably stale pidfile
│   path = "/home/jsimba/.julia/compiled/v1.12/ContinuumArrays/R9DIY_cpbtc.ji.pidfile"
└ @ FileWatching.Pidfile ~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/share/julia/stdlib/v1.12/FileWatching/src/pidfile.jl:247
┌ Warning: attempting to remove probably stale pidfile
│   path = "/home/jsimba/.julia/compiled/v1.12/BandedMatrices/OxlqV_cpbtc.ji.pidfile"
└ @ FileWatching.Pidfile ~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/share/julia/stdlib/v1.12/FileWatching/src/pidfile.jl:247
Precompiling Mici finished.
  1 dependency successfully precompiled in 261 seconds
  1 dependency had output during precompilation:
┌ Mici
│  [Output was shown above]
└
Running integrator benchmarks...

Single step benchmark
dimension: 2, step_size: 0.1, steps per sample: 1

case                            median          mean      allocs        memory    slowdown
------------------------  ------------  ------------  ----------  ------------  ----------
manual leapfrog             194.313 ns    331.106 ns          12     480 bytes        1.0x
GNI splitting                65.871 μs     68.589 μs         670     29.20 KiB      339.0x

Repeated step benchmark
dimension: 2, step_size: 0.1, steps per sample: 10

case                            median          mean      allocs        memory    slowdown
------------------------  ------------  ------------  ----------  ------------  ----------
manual leapfrog               1.705 μs      1.892 μs         120      4.71 KiB        1.0x
GNI splitting               635.337 μs    747.008 μs        6700    292.03 KiB     372.73x