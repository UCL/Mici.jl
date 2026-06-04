const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
if !(REPO_ROOT in LOAD_PATH)
    push!(LOAD_PATH, REPO_ROOT)
end

using Mici: EuclideanSystem, GNISplittingIntegrator, LeapfrogIntegrator, PhasePoint, step!
using PDMats
using LogDensityProblems
using Profile

struct GaussianLogDensity{T,M}
    mean::Vector{T}
    covariance::M
end

function GaussianLogDensity(dimension::Integer)
    GaussianLogDensity(zeros(dimension), ScalMat(dimension, 1.0))
end

LogDensityProblems.dimension(p::GaussianLogDensity) = length(p.mean)
LogDensityProblems.capabilities(::Type{<:GaussianLogDensity}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity_and_gradient(p::GaussianLogDensity, q)
    residual = q .- p.mean
    logdensity = -0.5 * invquad(p.covariance, residual)
    gradient = -(p.covariance \ residual)
    return logdensity, gradient
end

function make_problem(; dimension=2, step_size=0.1)
    density = GaussianLogDensity(dimension)
    system = EuclideanSystem(ScalMat(dimension, 1.0), density)
    q = [0.3, -0.7]
    p = [1.2, 0.4]
    z_native = PhasePoint(copy(q), copy(p), NaN, similar(q), false)
    z_gni = PhasePoint(copy(q), copy(p), NaN, similar(q), false)
    native = LeapfrogIntegrator(; step_size)
    gni = GNISplittingIntegrator(; system, phase_point=copy(z_gni), step_size)
    return (; system, z_native, z_gni, native, gni)
end

function warmup!(state)
    step!(state.z_native, state.native, state.system)
    step!(state.z_gni, state.gni, state.system)
    return nothing
end

function run_native!(state, n)
    for _ in 1:n
        step!(state.z_native, state.native, state.system)
    end
    return nothing
end

function run_gni!(state, n)
    for _ in 1:n
        step!(state.z_gni, state.gni, state.system)
    end
    return nothing
end

state = make_problem()
warmup!(state)

Profile.clear_malloc_data()
run_native!(state, 5)

Profile.clear_malloc_data()
run_gni!(state, 5)
