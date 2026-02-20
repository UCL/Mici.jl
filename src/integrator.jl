# Integrator methods for solving discretized hamiltonian systems

struct LeapfrogIntegrator <: AbstractIntegrator
    ϵ::Float64
end

function step!(z::PhasePoint, integrator::LeapfrogIntegrator, system::AbstractTractableFlowSystem)
    Φ₁!(z, system, integrator.ϵ / 2)
    Φ₂!(z, system, integrator.ϵ)
    Φ₁!(z, system, integrator.ϵ / 2)
end

