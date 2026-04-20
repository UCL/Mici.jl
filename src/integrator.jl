# Integrator methods for solving discretized hamiltonian systems
abstract type AbstractIntegrator end

struct LeapfrogIntegrator<: AbstractIntegrator
    ϵ::Float64
end

struct LeapfrogAdapterIntegrator <: AbstractIntegrator
    ϵ::Float64
end

function step!(z::PhasePoint, integrator::LeapfrogIntegrator, system::AbstractTractableFlowSystem)
    Φ₁!(z, system, integrator.ϵ / 2)
    Φ₂!(z, system, integrator.ϵ)
    Φ₁!(z, system, integrator.ϵ / 2)
end

function step!(z::PhasePoint, integrator::LeapfrogAdapterIntegrator, system::AbstractTractableFlowSystem)
    x = vcat(z.q, z.p)

    # Integrate a single step
    adapter = Gni.LeapfrogAdapter(system, x, (0.0, integrator.ϵ), integrator.ϵ)
    Gni.GeometricIntegrators.integrate!(adapter.core.solution, adapter.core.integrator)
    x_next = adapter.core.solution.q
    d = length(x_next) ÷ 2
    # Do manual update of state
    z.q .= @view x_next[begin:d]
    z.p .= @view x_next[d+1:end]
end
