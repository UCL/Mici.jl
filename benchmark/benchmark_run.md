julia --project=benchmark benchmark/integrator_step_benchmarks.jl

Running integrator benchmarks...

Single step benchmark
dimension: 2, step_size: 0.1, steps per sample: 1

case                            median          mean      allocs        memory    slowdown
------------------------  ------------  ------------  ----------  ------------  ----------
manual leapfrog             260.808 ns    316.439 ns          12     480 bytes        1.0x
GNI splitting                90.788 μs    105.936 μs         670     29.20 KiB      348.1x
GNI composition              93.938 μs    120.559 μs         557     36.84 KiB     360.18x

Repeated step benchmark
dimension: 2, step_size: 0.1, steps per sample: 10

case                            median          mean      allocs        memory    slowdown
------------------------  ------------  ------------  ----------  ------------  ----------
manual leapfrog               3.724 μs      4.112 μs         120      4.71 KiB        1.0x
GNI splitting                 1.173 ms      1.378 ms        6700    292.03 KiB     315.01x
GNI composition               1.204 ms      1.370 ms        5570    368.44 KiB     323.43x

Notes:
- Manual leapfrog mutates the PhasePoint directly with the three split flows.
- GNI splitting includes adapter buffer, SolutionStep, and scratch PhasePoint synchronization.
- GNI composition uses GeometricIntegrators' composition path while preserving the same phase-point updates.
- The agreement tables below check that both adapter paths preserve the same q/p sample statistics as leapfrog in this toy problem.
- These are local-machine timings for comparison, not pass/fail test thresholds.

Agreement vs manual leapfrog after one step
dimension: 2, step_size: 0.1

case                            max |Δq|        max |Δp|       valid
------------------------  --------------  --------------  ----------
GNI splitting                  0.000e+00       0.000e+00       false
GNI composition                0.000e+00       0.000e+00       false

Agreement vs manual leapfrog after repeated steps
dimension: 2, step_size: 0.1

case                            max |Δq|        max |Δp|       valid
------------------------  --------------  --------------  ----------
GNI splitting                  0.000e+00       0.000e+00       false
GNI composition                0.000e+00       0.000e+00       false
