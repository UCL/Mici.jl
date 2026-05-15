using GeometricIntegrators

struct LeapfrogIntegrator{T} <: AbstractIntegrator
    ϵ::T
end

struct LeapfrogAdapter{T} <: AbstractIntegrator
    ϵ::T
end

struct IntegratorAdapterCore{P,M,S,I}
    problem::P
    method::M
    solution::S
    integrator::I
end

# TODO This naming is pretty poor, need to get better understanding
# of the domain and its mapping to our MCMC space
struct SeparableODE{C<:IntegratorAdapterCore} <: AbstractIntegrator
    core::C
    ϵ::Real

    function SeparableODE(system::AbstractSystem, initial_state::AbstractArray, timespan::Tuple, step_size::Real, method::GeometricMethod)
        problem = construct_split_ode_problem(system, initial_state, timespan, step_size)
        integrator = GeometricIntegrator(problem, method)
        solution = SolutionStep(problem)
        core = IntegratorAdapterCore(problem, method, solution, integrator)
        return new{typeof(core)}(core, step_size)
    end

    SeparableODE(system, initial_state, timespan, step_size; method::GeometricMethod=StrangA()) =  SeparableODE(system, initial_state, timespan, step_size, method)
end

function LeapfrogAdapter(args...; system, phase_point, step_size)
    initial_state = vcat(phase_point.q, phase_point.p)
    SeparableODE(system, initial_state, (0,step_size), step_size, StrangA())
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


function step!(z::PhasePoint, adapter::SeparableODE, system::AbstractTractableFlowSystem)
    x = vcat(z.q, z.p)

    # Integrate a single step
    # adapter = Gni.LeapfrogAdapter(system, x, (0.0, integrator.ϵ), integrator.ϵ)

    GeometricIntegrators.integrate!(adapter.core.solution, adapter.core.integrator)
    x_next = adapter.core.solution.q
    d = length(x_next) ÷ 2
    # Do manual update of state
    z.q .= @view x_next[begin:d]
    z.p .= @view x_next[d+1:end]

    # Gni.Base.reset!(adapter.solution)
end