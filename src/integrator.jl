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
    LeapfrogIntegrator(isnothing(step_size) ? 0.1 : step_size)
end

step_size(integrator::LeapfrogIntegrator) = integrator.ϵ
# struct LeapfrogGNIIntegrator{T} <: AbstractIntegrator
#     ϵ::T
#     adapter::Gni.LeapfrogAdapter
# end

function step!(z::PhasePoint, integrator::LeapfrogIntegrator, system::AbstractTractableFlowSystem)
    Φ₁!(z, system, integrator.ϵ / 2)
    Φ₂!(z, system, integrator.ϵ)
    Φ₁!(z, system, integrator.ϵ / 2)
end
