using LogDensityProblems: logdensity_and_gradient

abstract type AbstractPhasePoint end

struct PhasePoint{T} <: AbstractPhasePoint
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
        z.logdens, g = LogDensityProblems.logdensity_and_gradient(system.ℓ, z.q)
        z.grad .= g
        z.valid = true
    end
end

function refresh!(z::PhasePoint)
    z.valid = false
    return nothing
end

abstract type AbstractState{P,S,I} end
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


