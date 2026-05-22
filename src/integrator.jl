struct LeapfrogIntegrator{T} <: AbstractIntegrator
    ϵ::T
end

function step!(z::PhasePoint, integrator::LeapfrogIntegrator, system::AbstractTractableFlowSystem; direction=1)
    Φ₁!(z, system, direction*integrator.ϵ / 2)
    Φ₂!(z, system, direction*integrator.ϵ)
    Φ₁!(z, system, direction*integrator.ϵ / 2)
end