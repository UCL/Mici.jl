# # A small story about splitting coefficients
#
# This example compares a few representative Hamiltonian integrators on the
# same loop-shaped target distribution used in the basic usage example. The
# point is not to be exhaustive. The point is to show how changing the
# splitting coefficients and the composition order changes the runtime/error
# tradeoff.

using AbstractMCMC
import Mici:
    AbstractIntegrator,
    CompositionAdapter,
    EuclideanSystem,
    HMC,
    LeapfrogIntegrator,
    PhasePoint,
    StaticMetropolisIntegrationTransition,
    h,
    step!,
    step_size
using CairoMakie
using GeometricIntegrators: McLachlan2, McLachlan4, StrangA, StrangB, SuzukiFractal, TripleJump
using LogDensityProblems
using Markdown
using PDMats
using Random
using LinearAlgebra: dot
using Printf
using Statistics

CairoMakie.activate!(type = "svg") #hide

# ## A loop target
#
# We reuse the same loop-shaped density from the basic example, but keep the
# implementation local so the story is self-contained.

@kwdef struct LoopProblem{T}
    σ::T = 0.2
    y::T = 1.0
end

LogDensityProblems.dimension(::LoopProblem) = 2
LogDensityProblems.capabilities(::Type{<:LoopProblem}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(ℓ::LoopProblem, θ)
    (; σ, y) = ℓ
    f = θ[2]^2 + 2 * θ[1]^2 * (θ[1]^2 - 0.5)
    -sum(θ .^ 2) / 2 - ((y - f) / σ)^2 / 2
end

function LogDensityProblems.logdensity_and_gradient(ℓ::LoopProblem, θ)
    (; σ, y) = ℓ
    f = θ[2]^2 + 2 * θ[1]^2 * (θ[1]^2 - 0.5)
    residual = y - f
    ℓπ = -sum(θ .^ 2) / 2 - (residual / σ)^2 / 2
    ∂f∂θ₁ = 8 * θ[1]^3 - 2 * θ[1]
    ∂f∂θ₂ = 2 * θ[2]
    ∇ℓπ = -θ .+ (residual / σ^2) .* [∂f∂θ₁, ∂f∂θ₂]
    return ℓπ, ∇ℓπ
end

loop_problem = LoopProblem()
loop_system = EuclideanSystem(ScalMat(2, 1.0), loop_problem)

# A fixed phase point is enough to illustrate the tradeoffs.
q0 = [0.45, -0.75]
p0 = [1.05, 0.35]
z0 = PhasePoint(copy(q0), copy(p0), NaN, similar(q0), false)

function run_steps!(z, integrator, system, n_steps)
    for _ in 1:n_steps
        step!(z, integrator, system)
    end
    return z
end

function final_state(z0, integrator, system, n_steps)
    z = copy(z0)
    run_steps!(z, integrator, system, n_steps)
    return z
end

function markdown_table(title, subtitle, headers, rows)
    io = IOBuffer()
    println(io, "### $(title)")
    println(io, subtitle)
    println(io)
    println(io, "| $(join(headers, " | ")) |")
    println(io, "| $(join(fill("---", length(headers)), " | ")) |")
    for row in rows
        println(io, "| $(join(row, " | ")) |")
    end
    return Markdown.parse(String(take!(io)))
end

function markdown_table_text(title, subtitle, headers, rows)
    io = IOBuffer()
    println(io, "## $(title)")
    println(io, "")
    println(io, subtitle)
    println(io, "")
    println(io, "| $(join(headers, " | ")) |")
    println(io, "| $(join(fill("---", length(headers)), " | ")) |")
    for row in rows
        println(io, "| $(join(row, " | ")) |")
    end
    return String(take!(io))
end

struct LeapfrogHMCAdapter <: AbstractIntegrator
    inner::LeapfrogIntegrator
end

function LeapfrogHMCAdapter(; step_size=0.1, kwargs...)
    LeapfrogHMCAdapter(LeapfrogIntegrator(; step_size = step_size))
end

step_size(integrator::LeapfrogHMCAdapter) = step_size(integrator.inner)

function step!(z::PhasePoint, integrator::LeapfrogHMCAdapter, system)
    step!(z, integrator.inner, system)
end

struct MethodAdapter{M} <: AbstractIntegrator
    inner::CompositionAdapter
end

function MethodAdapter{M}(; system, phase_point::PhasePoint, step_size=0.1) where {M}
    MethodAdapter{M}(CompositionAdapter(;
        system = system,
        phase_point = phase_point,
        step_size = step_size,
        method = M(),
    ))
end

step_size(integrator::MethodAdapter{M}) where {M} = step_size(integrator.inner)

function step!(z::PhasePoint, integrator::MethodAdapter, system)
    step!(z, integrator.inner, system)
end

# ## Representative methods
#
# `StrangA` is the baseline second-order split, `StrangB` is the alternate
# ordering, and the others are higher-order composition rules. The public API
# here is the method type itself; the adapter builds the corresponding
# coefficients internally.

const INTEGRATORS = [
    (
        label = "Leapfrog / StrangA",
        order = 2,
        make = (system, z0, ϵ) -> LeapfrogIntegrator(; step_size = ϵ),
        sampler = LeapfrogHMCAdapter,
    ),
    (
        label = "Composition / StrangB",
        order = 2,
        make = (system, z0, ϵ) -> MethodAdapter{StrangB}(;
            system = system, phase_point = copy(z0), step_size = ϵ
        ),
        sampler = MethodAdapter{StrangB},
    ),
    (
        label = "Composition / McLachlan2",
        order = 2,
        make = (system, z0, ϵ) -> MethodAdapter{McLachlan2}(;
            system = system, phase_point = copy(z0), step_size = ϵ
        ),
        sampler = MethodAdapter{McLachlan2},
    ),
    (
        label = "Composition / McLachlan4",
        order = 4,
        make = (system, z0, ϵ) -> MethodAdapter{McLachlan4}(;
            system = system, phase_point = copy(z0), step_size = ϵ
        ),
        sampler = MethodAdapter{McLachlan4},
    ),
    (
        label = "Composition / TripleJump",
        order = 4,
        make = (system, z0, ϵ) -> MethodAdapter{TripleJump}(;
            system = system, phase_point = copy(z0), step_size = ϵ
        ),
        sampler = MethodAdapter{TripleJump},
    ),
    (
        label = "Composition / SuzukiFractal",
        order = 4,
        make = (system, z0, ϵ) -> MethodAdapter{SuzukiFractal}(;
            system = system, phase_point = copy(z0), step_size = ϵ
        ),
        sampler = MethodAdapter{SuzukiFractal},
    ),
]

const STEP_SIZES = [0.1, 0.05, 0.025]
const INTEGRATION_TIME = 1.0
const REPEATS = 12
const ESS_REFERENCE_STEP = 0.05
const CHAIN_LENGTH = 1000
const BURNIN = 200

function trajectory_stats(system, z0, integrator, step_size; repeats = REPEATS)
    n_steps = round(Int, INTEGRATION_TIME / step_size)
    reference_energy = h(z0, system)

    step!(copy(z0), integrator, system)

    elapsed = @elapsed begin
        for _ in 1:repeats
            z = copy(z0)
            run_steps!(z, integrator, system, n_steps)
        end
    end

    z_final = final_state(z0, integrator, system, n_steps)
    final_energy = h(z_final, system)
    delta_h = final_energy - reference_energy

    return (;
        n_steps,
        mean_time_ms = 1000 * elapsed / repeats,
        abs_delta_h = abs(delta_h),
        accept_probability = exp(min(0.0, -delta_h)),
    )
end

function effective_sample_size(x)
    n = length(x)
    n < 3 && return float(n)
    centered = collect(x) .- mean(x)
    γ0 = dot(centered, centered) / n
    γ0 == 0 && return 0.0
    ρsum = 0.0
    for lag in 1:(n - 1)
        ρ = dot(view(centered, 1:(n - lag)), view(centered, (1 + lag):n)) / ((n - lag) * γ0)
        ρ <= 0 && break
        ρsum += ρ
    end
    return n / (1 + 2ρsum)
end

function chain_stats(spec, system, z0; seed=2024)
    sampler = HMC{EuclideanSystem, spec.sampler}(StaticMetropolisIntegrationTransition(1.0))
    rng = Xoshiro(seed)
    timed = @timed sample(
        rng,
        AbstractMCMC.LogDensityModel(loop_problem),
        sampler,
        CHAIN_LENGTH;
        initial_q = copy(z0.q),
        initial_ϵ = ESS_REFERENCE_STEP,
        initial_metric = ScalMat(2, 1.0),
        progress = false,
        trace_function = state -> (; q1 = state.phase_point.q[1]),
    )
    result = timed.value
    elapsed = timed.time
    q1 = result.traces.q1[(BURNIN + 1):end]
    ess = effective_sample_size(q1)
    return (;
        acceptance_rate = mean(result.statistics.accept_probability),
        ess,
        ess_per_sec = ess / elapsed,
        wall_time_sec = elapsed,
    )
end

function summarize_method(spec, system, z0)
    rows = NamedTuple[]
    for step_size in STEP_SIZES
        integrator = spec.make(system, z0, step_size)
        stats = trajectory_stats(system, z0, integrator, step_size)
        push!(
            rows,
            merge(
                (
                    label = spec.label,
                    order = spec.order,
                    step_size = step_size,
                ),
                stats,
            ),
        )
    end
    return rows
end

function summarize_ess(spec, system, z0)
    stats = chain_stats(spec, system, z0)
    return merge(
        (
            label = spec.label,
            order = spec.order,
            step_size = ESS_REFERENCE_STEP,
        ),
        stats,
    )
end

results = reduce(vcat, (summarize_method(spec, loop_system, z0) for spec in INTEGRATORS))
ess_results = [summarize_ess(spec, loop_system, z0) for spec in INTEGRATORS]

reference_step = 0.05
reference_rows = filter(row -> row.step_size == reference_step, results)

println("Reference step size: $(reference_step)")
println()
println(
    join(
        (
            rpad("method", 30),
            lpad("order", 6),
            lpad("steps", 7),
            lpad("time / traj", 14),
            lpad("|ΔH|", 14),
            lpad("accept", 10),
        ),
        "  ",
    ),
)
println(join((repeat("-", 30), repeat("-", 6), repeat("-", 7), repeat("-", 14), repeat("-", 14), repeat("-", 10)), "  "))
for row in reference_rows
    println(
        join(
            (
                rpad(row.label, 30),
                lpad(string(row.order), 6),
                lpad(string(row.n_steps), 7),
                lpad(@sprintf("%.3f ms", row.mean_time_ms), 14),
                lpad(@sprintf("%.3e", row.abs_delta_h), 14),
                lpad(@sprintf("%.3f", row.accept_probability), 10),
            ),
            "  ",
        ),
    )
end

display(
    markdown_table(
        "Trajectory summary",
        "Reference step size: $(reference_step)",
        ["method", "order", "steps", "time / traj", "|ΔH|", "accept"],
        [
            [
                row.label,
                string(row.order),
                string(row.n_steps),
                @sprintf("%.3f ms", row.mean_time_ms),
                @sprintf("%.3e", row.abs_delta_h),
                @sprintf("%.3f", row.accept_probability),
            ] for row in reference_rows
        ],
    ),
)

println()
println("ESS report")
println("reference step size: $(ESS_REFERENCE_STEP), chain length: $(CHAIN_LENGTH), burnin: $(BURNIN)")
println()
println(
    join(
        (
            rpad("method", 30),
            lpad("order", 6),
            lpad("ESS", 12),
            lpad("ESS/sec", 12),
            lpad("accept", 10),
        ),
        "  ",
    ),
)
println(join((repeat("-", 30), repeat("-", 6), repeat("-", 12), repeat("-", 12), repeat("-", 10)), "  "))
for row in ess_results
    println(
        join(
            (
                rpad(row.label, 30),
                lpad(string(row.order), 6),
                lpad(@sprintf("%.1f", row.ess), 12),
                lpad(@sprintf("%.1f", row.ess_per_sec), 12),
                lpad(@sprintf("%.3f", row.acceptance_rate), 10),
            ),
            "  ",
        ),
    )
end

display(
    markdown_table(
        "ESS report",
        "Reference step size: $(ESS_REFERENCE_STEP), chain length: $(CHAIN_LENGTH), burnin: $(BURNIN)",
        ["method", "order", "ESS", "ESS/sec", "accept"],
        [
            [
                row.label,
                string(row.order),
                @sprintf("%.1f", row.ess),
                @sprintf("%.1f", row.ess_per_sec),
                @sprintf("%.3f", row.acceptance_rate),
            ] for row in ess_results
        ],
    ),
)

results_fragment_text = join(
    (
        markdown_table_text(
            "Trajectory summary",
            "Reference step size: $(reference_step)",
            ["method", "order", "steps", "time / traj", "|ΔH|", "accept"],
            [
                [
                    row.label,
                    string(row.order),
                    string(row.n_steps),
                    @sprintf("%.3f ms", row.mean_time_ms),
                    @sprintf("%.3e", row.abs_delta_h),
                    @sprintf("%.3f", row.accept_probability),
                ] for row in reference_rows
            ],
        ),
        markdown_table_text(
            "ESS report",
            "Reference step size: $(ESS_REFERENCE_STEP), chain length: $(CHAIN_LENGTH), burnin: $(BURNIN)",
            ["method", "order", "ESS", "ESS/sec", "accept"],
            [
                [
                    row.label,
                    string(row.order),
                    @sprintf("%.1f", row.ess),
                    @sprintf("%.1f", row.ess_per_sec),
                    @sprintf("%.3f", row.acceptance_rate),
                ] for row in ess_results
            ],
        ),
    ),
    "\n\n",
)

results_fragment_candidates = [
    joinpath(pwd(), "integrator_comparison_story_results.md"),
    joinpath(pwd(), "..", "integrator_comparison_story_results.md"),
    joinpath(pwd(), "..", "..", "integrator_comparison_story_results.md"),
    joinpath(pwd(), "..", "..", "..", "integrator_comparison_story_results.md"),
    joinpath(@__DIR__, "integrator_comparison_story_results.md"),
    joinpath(@__DIR__, "..", "integrator_comparison_story_results.md"),
    joinpath(@__DIR__, "..", "..", "integrator_comparison_story_results.md"),
]

for results_fragment in results_fragment_candidates
    mkpath(dirname(results_fragment))
    open(results_fragment, "w") do io
        write(io, results_fragment_text)
    end
end

fig = Figure(size = (980, 560))
ax = Axis(
    fig[1, 1],
    xlabel = "mean trajectory time (ms)",
    ylabel = "|ΔH| after one trajectory",
    title = "Loop problem: runtime vs energy error",
    xscale = log10,
    yscale = log10,
)

colors = Dict(
    "Leapfrog / StrangA" => :black,
    "Composition / StrangB" => :steelblue,
    "Composition / McLachlan2" => :purple,
    "Composition / McLachlan4" => :darkorange,
    "Composition / TripleJump" => :firebrick,
    "Composition / SuzukiFractal" => :seagreen,
)

for spec in INTEGRATORS
    rows = sort(filter(row -> row.label == spec.label, results), by = row -> row.mean_time_ms)
    xs = [row.mean_time_ms for row in rows]
    ys = [max(row.abs_delta_h, 1e-16) for row in rows]
    color = colors[spec.label]
    lines!(ax, xs, ys, color = color, linewidth = 2)
    scatter!(
        ax,
        xs,
        ys,
        color = color,
        markersize = 13,
        label = "$(spec.label) (order $(spec.order))",
    )
    for row in rows
        text!(
            ax,
            @sprintf("ϵ=%.3f", row.step_size),
            position = (row.mean_time_ms, row.abs_delta_h),
            align = (:left, :bottom),
            offset = (5, 5),
            fontsize = 11,
            color = color,
        )
    end
end

Legend(fig[1, 2], ax; framevisible = false, tellheight = false, tellwidth = false)
fig

# ## Reading the plot
#
# The second-order methods are the cheapest per trajectory, but they lose
# energy fidelity sooner as the step size grows. The fourth-order composition
# methods spend more work per trajectory, but they buy down the Hamiltonian
# error much faster. That is the use case for GeometricIntegrators here:
# not beating a handwritten leapfrog on a simple split, but making higher-order
# methods easy to express, compare, and present.
