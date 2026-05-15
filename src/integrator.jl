struct LeapfrogIntegrator{T} <: AbstractIntegrator
    ϵ::T
end

function step!(z::PhasePoint, integrator::LeapfrogIntegrator, system::AbstractTractableFlowSystem; dir=1)
    Φ₁!(z, system, dir*integrator.ϵ / 2)
    Φ₂!(z, system, dir*integrator.ϵ)
    Φ₁!(z, system, dir*integrator.ϵ / 2)
end