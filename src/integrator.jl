struct LeapfrogIntegrator{T} <: AbstractIntegrator
    ϵ::T
end


struct IntegratorAdapterCore{P,M,S,I}
    problem::P
    method::M
    solution::S
    integrator::I
end


function LeapfrogIntegrator(; step_size=0.1, kwargs...)
    LeapfrogIntegrator(step_size)
end
# struct LeapfrogGNIIntegrator{T} <: AbstractIntegrator
#     ϵ::T
#     adapter::Gni.LeapfrogAdapter
# end

function step!(z::PhasePoint, integrator::LeapfrogIntegrator, system::AbstractTractableFlowSystem)
    Φ₁!(z, system, integrator.ϵ / 2)
    Φ₂!(z, system, integrator.ϵ)
    Φ₁!(z, system, integrator.ϵ / 2)
end


function step!(z::PhasePoint, adapter::IntegratorAdapterCore, system::AbstractTractableFlowSystem)
    x = vcat(z.q, z.p)

    # Integrate a single step
    # adapter = Gni.LeapfrogAdapter(system, x, (0.0, integrator.ϵ), integrator.ϵ)

    GeometricIntegrators.integrate!(adapter.core.solution, adapter.integrator)
    x_next = adapter.core.solution.q
    d = length(x_next) ÷ 2
    # Do manual update of state
    z.q .= @view x_next[begin:d]
    z.p .= @view x_next[d+1:end]

    # Gni.Base.reset!(adapter.solution)
end