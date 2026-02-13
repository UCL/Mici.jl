# Integrator methods for solving discretized hamiltonian systems
#abstract type AbstractIntegrator end


#Original Implementation

struct LeapfrogIntegrator <: AbstractIntegrator
    ε::Float64
    T::Int
end

function step!(
    h::AbstractEuclideanSystem,
    state::AbstractChainState,
    ε::Float64,
)
    p(state) .-= (ε/2) .* ∂H₁∂q(h, state)
    q(state) .+= ε .* ∂H₂∂p(h, state)
    p(state) .-= (ε/2) .* ∂H₁∂q(h, state)
end

function integrate!(lfi::LeapfrogIntegrator, h::AbstractEuclideanSystem, state::AbstractChainState)
    for n = 1:lfi.T
        step!(h, state, lfi.ε)
    end
end



#GeometricIntegrators Implementation

struct GIIntegrator <: AbstractIntegrator
    method::GeometricMethod
    ε::Float64
    T::Int
end

function integrate!(
    gi::GIIntegrator,
    system::AbstractEuclideanSystem,
    state::AbstractChainState,
)
    # Initialise GI integrator + solution state
    problem = gi_problem(system, state, gi.T, gi.ε)
    solstep, integrator = initialise_step(problem, gi.method)

    # Perform T steps
    for i in 1:gi.T
        GeometricIntegratorsBase.integrate!(solstep, integrator)
    end

    q_new, p_new = current_qp(solstep)
    update_state!(state, copy(q_new), copy(p_new))
end


#Second attempt - making GIIntegrator mutable so problem and integrator only need to be defined once.

mutable struct GIIntegrator2 <: AbstractIntegrator
    method::GeometricMethod
    ε::Float64
    T::Int
    problem::EquationProblem
    solstep::SolutionStep
    integrator::GeometricIntegrator
end

function GIIntegrator2(method, ε, T, system, state)

    problem = gi_problem(system, state, T, ε)
    solstep, integrator = initialise_step(problem, method)

    return GIIntegrator2(method, ε, T, problem, solstep, integrator)
end


function integrate!(
    gi::GIIntegrator2,
    system::AbstractEuclideanSystem,
    state::AbstractChainState,
)
    # Initial conditions
    set_initial_condition!(gi.solstep, state)

    # Perform T steps
    for i in 1:gi.T
        GeometricIntegratorsBase.integrate!(gi.solstep, gi.integrator)
    end

    q_new, p_new = current_qp(gi.solstep)
    update_state!(state, copy(q_new), copy(p_new))
end
