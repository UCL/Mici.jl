# Run from the repository root:
#
#     julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark benchmark/integrator_step_benchmarks.jl
#
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
if !(REPO_ROOT in LOAD_PATH)
    push!(LOAD_PATH, REPO_ROOT)
end

using BenchmarkTools
using LogDensityProblems
using Printf
using Mici:
    EuclideanSystem,
    GNICompositionIntegrator,
    GNISplittingIntegrator,
    LeapfrogIntegrator,
    PhasePoint,
    step!
using PDMats

struct GaussianLogDensity{T,M}
    mean::Vector{T}
    covariance::M
end

function GaussianLogDensity(dimension::Integer)
    GaussianLogDensity(zeros(dimension), ScalMat(dimension, 1.0))
end

LogDensityProblems.dimension(p::GaussianLogDensity) = length(p.mean)

function LogDensityProblems.capabilities(::Type{<:GaussianLogDensity})
    LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity_and_gradient(p::GaussianLogDensity, q)
    residual = q .- p.mean
    logdensity = -0.5 * invquad(p.covariance, residual)
    gradient = -(p.covariance \ residual)
    return logdensity, gradient
end

function phase_point(q, p)
    PhasePoint(copy(q), copy(p), NaN, similar(q), false)
end

function benchmark_problem(; dimension=2, step_size=0.1)
    density = GaussianLogDensity(dimension)
    system = EuclideanSystem(ScalMat(dimension, 1.0), density)
    q = collect(range(0.2, 1.0; length=dimension))
    p = collect(range(1.1, 0.3; length=dimension))
    z0 = phase_point(q, p)
    native_integrator = LeapfrogIntegrator(; step_size)
    gni_integrator = GNISplittingIntegrator(; system, phase_point=copy(z0), step_size)
    composition_integrator = GNICompositionIntegrator(;
        system, phase_point=copy(z0), step_size
    )
    return (; system, z0, native_integrator, gni_integrator, composition_integrator, step_size, dimension)
end

function run_steps!(z, integrator, system, n_steps)
    for _ in 1:n_steps
        step!(z, integrator, system)
    end
    return z
end

function final_state(z0, integrator, system, n_steps)
    z = phase_point(z0.q, z0.p)
    run_steps!(z, integrator, system, n_steps)
    return z
end

function state_deltas(reference, candidate)
    return (
        max_q_delta=maximum(abs, candidate.q .- reference.q),
        max_p_delta=maximum(abs, candidate.p .- reference.p),
        valid_match=(candidate.valid == reference.valid),
    )
end

function trial_summary(name, trial, baseline_time)
    estimate = median(trial)
    mean_estimate = mean(trial)
    return (;
        name,
        median_time=estimate.time,
        mean_time=mean_estimate.time,
        allocs=estimate.allocs,
        memory=estimate.memory,
        slowdown=estimate.time / baseline_time,
    )
end

function print_summary(title, rows; dimension, step_size, n_steps)
    println()
    println(title)
    println("dimension: $(dimension), step_size: $(step_size), steps per sample: $(n_steps)")
    println()
    header = (
        rpad("case", 24),
        lpad("median", 12),
        lpad("mean", 12),
        lpad("allocs", 10),
        lpad("memory", 12),
        lpad("slowdown", 10),
    )
    println(join(header, "  "))
    println(join((
        repeat("-", 24),
        repeat("-", 12),
        repeat("-", 12),
        repeat("-", 10),
        repeat("-", 12),
        repeat("-", 10),
    ), "  "))
    for row in rows
        values = (
            rpad(row.name, 24),
            lpad(BenchmarkTools.prettytime(row.median_time), 12),
            lpad(BenchmarkTools.prettytime(row.mean_time), 12),
            lpad(string(row.allocs), 10),
            lpad(BenchmarkTools.prettymemory(row.memory), 12),
            lpad("$(round(row.slowdown; digits=2))x", 10),
        )
        println(join(values, "  "))
    end
end

function print_agreement(title, rows; dimension, step_size)
    println()
    println(title)
    println("dimension: $(dimension), step_size: $(step_size)")
    println()
    header = (
        rpad("case", 24),
        lpad("max |Δq|", 14),
        lpad("max |Δp|", 14),
        lpad("valid", 10),
    )
    println(join(header, "  "))
    println(join((
        repeat("-", 24),
        repeat("-", 14),
        repeat("-", 14),
        repeat("-", 10),
    ), "  "))
    for row in rows
        values = (
            rpad(row.name, 24),
            lpad(@sprintf("%.3e", row.max_q_delta), 14),
            lpad(@sprintf("%.3e", row.max_p_delta), 14),
            lpad(string(row.valid_match), 10),
        )
        println(join(values, "  "))
    end
end

function run_integrator_benchmarks(;
    dimension=2,
    step_size=0.1,
    trajectory_steps=10,
    seconds=0.25,
    samples=200,
)
    problem = benchmark_problem(; dimension, step_size)
    (; system, z0, native_integrator, gni_integrator, composition_integrator) = problem

    step!(copy(z0), native_integrator, system)
    step!(copy(z0), gni_integrator, system)
    step!(copy(z0), composition_integrator, system)
    run_steps!(copy(z0), native_integrator, system, trajectory_steps)
    run_steps!(copy(z0), gni_integrator, system, trajectory_steps)
    run_steps!(copy(z0), composition_integrator, system, trajectory_steps)

    println("Running integrator benchmarks...")
    flush(stdout)

    BenchmarkTools.DEFAULT_PARAMETERS.seconds = seconds
    BenchmarkTools.DEFAULT_PARAMETERS.samples = samples
    BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

    native_step = @benchmark step!(z, $native_integrator, $system) setup = (z = copy($z0))
    gni_step = @benchmark step!(z, $gni_integrator, $system) setup = (z = copy($z0))
    composition_step = @benchmark step!(z, $composition_integrator, $system) setup = (z = copy($z0))
    native_trajectory = @benchmark run_steps!(z, $native_integrator, $system, $trajectory_steps) setup = (z = copy($z0))
    gni_trajectory = @benchmark run_steps!(z, $gni_integrator, $system, $trajectory_steps) setup = (z = copy($z0))
    composition_trajectory = @benchmark run_steps!(z, $composition_integrator, $system, $trajectory_steps) setup = (z = copy($z0))

    step_baseline = median(native_step).time
    trajectory_baseline = median(native_trajectory).time

    print_summary(
        "Single step benchmark",
        [
            trial_summary("manual leapfrog", native_step, step_baseline),
            trial_summary("GNI splitting", gni_step, step_baseline),
            trial_summary("GNI composition", composition_step, step_baseline),
        ];
        dimension,
        step_size,
        n_steps=1,
    )

    print_summary(
        "Repeated step benchmark",
        [
            trial_summary("manual leapfrog", native_trajectory, trajectory_baseline),
            trial_summary("GNI splitting", gni_trajectory, trajectory_baseline),
            trial_summary("GNI composition", composition_trajectory, trajectory_baseline),
        ];
        dimension,
        step_size,
        n_steps=trajectory_steps,
    )

    println()
    println("Notes:")
    println("- Manual leapfrog mutates the PhasePoint directly with the three split flows.")
    println("- GNI splitting includes adapter buffer, SolutionStep, and scratch PhasePoint synchronization.")
    println("- GNI composition uses GeometricIntegrators' composition path while preserving the same phase-point updates.")
    println("- The agreement tables below check that both adapter paths preserve the same q/p sample statistics as leapfrog in this toy problem.")
    println("- These are local-machine timings for comparison, not pass/fail test thresholds.")

    return (;
        native_step,
        gni_step,
        composition_step,
        native_trajectory,
        gni_trajectory,
        composition_trajectory,
    )
end

function benchmark_agreement_report(;
    dimension=2,
    step_size=0.1,
    trajectory_steps=10,
)
    problem = benchmark_problem(; dimension, step_size)
    (; system, z0, native_integrator, gni_integrator, composition_integrator) = problem

    one_step_reference = final_state(z0, native_integrator, system, 1)
    trajectory_reference = final_state(z0, native_integrator, system, trajectory_steps)

    one_step_rows = [
        merge((name="GNI splitting",), state_deltas(one_step_reference, final_state(z0, gni_integrator, system, 1))),
        merge((name="GNI composition",), state_deltas(one_step_reference, final_state(z0, composition_integrator, system, 1))),
    ]

    trajectory_rows = [
        merge((name="GNI splitting",), state_deltas(trajectory_reference, final_state(z0, gni_integrator, system, trajectory_steps))),
        merge((name="GNI composition",), state_deltas(trajectory_reference, final_state(z0, composition_integrator, system, trajectory_steps))),
    ]

    print_agreement(
        "Agreement vs manual leapfrog after one step",
        one_step_rows;
        dimension,
        step_size,
    )
    print_agreement(
        "Agreement vs manual leapfrog after repeated steps",
        trajectory_rows;
        dimension,
        step_size,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_integrator_benchmarks()
    benchmark_agreement_report()
end
