struct LeapfrogIntegrator{T} <: AbstractIntegrator
    ϵ::T
end

function step!(z::PhasePoint, integrator::LeapfrogIntegrator, system::AbstractTractableFlowSystem)
    Φ₁!(z, system, integrator.ϵ / 2)
    Φ₂!(z, system, integrator.ϵ)
    Φ₁!(z, system, integrator.ϵ / 2)
end

