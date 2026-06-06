struct ConstrainedLeapfrogIntegrator{T} <: AbstractIntegrator
    ϵ::T
end

function ConstrainedLeapfrogIntegrator(; step_size=0.1, kwargs...)
    ϵ = isnothing(step_size) ? 0.1 : step_size
    return ConstrainedLeapfrogIntegrator(ϵ)
end

step_size(integrator::ConstrainedLeapfrogIntegrator) = integrator.ϵ

function step!(
    z::PhasePoint,
    integrator::ConstrainedLeapfrogIntegrator,
    system::AbstractConstrainedTractableFlowSystem,
)
    Φ₁!(z, system, integrator.ϵ / 2)
    Φ₂!(z, system, integrator.ϵ)
    Φ₁!(z, system, integrator.ϵ / 2)
    return nothing
end

struct ConstrainedCompositionIntegrator{M,T,F,C} <: AbstractIntegrator
    ϵ::T
    f::F
    c::C
end

function ConstrainedCompositionIntegrator{M}(; step_size=0.1, kwargs...) where {M<:AbstractCompositionMethod}
    ϵ = isnothing(step_size) ? 0.1 : step_size
    f, c = coefficients(M(), typeof(float(ϵ)))
    return ConstrainedCompositionIntegrator{M, typeof(ϵ), typeof(f), typeof(c)}(ϵ, f, c)
end

function ConstrainedCompositionIntegrator(; method::M=StrangA(), step_size=0.1, kwargs...) where {M<:AbstractCompositionMethod}
    return ConstrainedCompositionIntegrator{M}(; step_size=step_size, kwargs...)
end

step_size(integrator::ConstrainedCompositionIntegrator) = integrator.ϵ
method(::ConstrainedCompositionIntegrator{M}) where {M} = M()

@inline function _apply_constrained_split_flow!(
    z::PhasePoint,
    system::AbstractConstrainedTractableFlowSystem,
    flow::Int,
    ϵ::Real,
)
    if flow == 1
        Φ₁!(z, system, ϵ)
    elseif flow == 2
        Φ₂!(z, system, ϵ)
    else
        throw(ArgumentError("Unsupported flow index $flow"))
    end
    return nothing
end

function step!(
    z::PhasePoint,
    integrator::ConstrainedCompositionIntegrator,
    system::AbstractConstrainedTractableFlowSystem,
)
    @inbounds for i in eachindex(integrator.f)
        _apply_constrained_split_flow!(z, system, integrator.f[i], integrator.c[i] * integrator.ϵ)
    end
    return nothing
end
