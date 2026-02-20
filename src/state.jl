using LogDensityProblems: logdensity_and_gradient

"""
    PhasePoint

Struct representing a point in the phase space of a Hamiltonian system, consisting of:
    q       -- current position 
    p       -- current momentum
    logdens -- log density at the current position
    grad    -- gradient of the log density at the current position
    valid   -- indicator for whether the log density and gradient are up-to-date
"""
mutable struct PhasePoint{T}
    q::Vector{T}
    p::Vector{T}
    logdens::T
    grad::Vector{T}
    valid::Bool
end

function PhasePoint(::UndefInitializer, dimension::Integer, T::Type=Float64)
    PhasePoint(Vector{T}(undef, dimension), Vector{T}(undef, dimension), T(NaN), Vector{T}(undef, dimension), false)
end

function PhasePoint(::UndefInitializer, src::PhasePoint{T}) where {T}
    PhasePoint(undef, dimension(src), T)
end

function Base.copy!(dest::PhasePoint, src::PhasePoint)
    dest.q .= src.q
    dest.p .= src.p
    dest.logdens = src.logdens
    dest.grad .= src.grad
    dest.valid = src.valid
end

function Base.copy(src::PhasePoint)
    dest = PhasePoint(undef, src)
    copy!(dest, src)
    return dest
end

dimension(phase_point::PhasePoint) = length(phase_point.q)

function sample_initial_phase_point(
    rng::AbstractRNG, system::AbstractSystem, initial_q::Union{Vector,Nothing}
)
    !isnothing(initial_q) && @assert length(initial_q) == dimension(system)
    q = isnothing(initial_q) ? randn(rng, dimension(system)) : initial_q
    p = Vector{eltype(q)}(undef, dimension(system))
    logdens = NaN
    grad = Vector{eltype(q)}(undef, dimension(system))
    z = PhasePoint(q, p, logdens, grad, false)
    rand!(rng, z.p, z, system)
    return z
end

function logdens(z::PhasePoint, system::AbstractSystem)
    ensure_valid!(z, system)
    return z.logdens
end

function grad(z::PhasePoint, system::AbstractSystem)
    ensure_valid!(z, system)
    return z.grad
end

function ensure_valid!(z::PhasePoint, system::AbstractSystem)
    if !z.valid
        z.logdens, g = LogDensityProblems.logdensity_and_gradient(ℓ(system), z.q)
        z.grad .= g
        z.valid = true
    end
end

""" Indicate the log density and gradient needs to be updated at the current position. """
function refresh!(z::PhasePoint)
    z.valid = false
    return nothing
end

"""
    AbstractState{P,S,I}

Abstract supertype for states of MCMC samplers, parameterized by:
    P --  type of the phase point (e.g., `PhasePoint{T}`)
    S --  type of the system (e.g., `EuclideanSystem`)
    I --  type of the integrator (e.g., `LeapfrogIntegrator`)

Concrete subtypes of `AbstractState` should contain at least the following fields:
    - `phase_point::P` -- the current phase point of the sampler
    - `system::S` -- the Hamiltonian system being sampled
    - `integrator::I` -- the integrator used for simulating Hamiltonian dynamics
"""
abstract type AbstractState{P,S,I} end

"""
    MetropolisHMCState{P, S, I} <: AbstractState{P,S,I}

Concrete state type for a Metropolis-adjusted Hamiltonian Monte Carlo sampler.
"""
struct MetropolisHMCState{P,S,I} <: AbstractState{P,S,I}
    phase_point::P
    proposed_phase_point::P
    system::S
    integrator::I
end

function MetropolisHMCState(
    phase_point::PhasePoint, system::AbstractSystem, integrator::AbstractIntegrator
)
    MetropolisHMCState(phase_point, copy(phase_point), system, integrator)
end


