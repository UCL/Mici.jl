"""
Functions and types for defining Hamiltonian systems constrained to a manifold.
"""

using LinearAlgebra

abstract type AbstractConstrainedSystem <: AbstractSystem end
abstract type AbstractConstrainedTractableFlowSystem <: AbstractTractableFlowSystem end

"""
    ConstrainedEuclideanMetricSystem{M,L,C,J,PS,PK}

Hamiltonian system constrained to an embedded manifold, with a Euclidean kinetic
energy defined by a fixed positive definite metric and constraints specified by a
residual function and Jacobian.
"""
struct ConstrainedEuclideanMetricSystem{M,L,C,J,PS,PK} <:
       AbstractConstrainedTractableFlowSystem
    metric::M
    ℓ::L
    constr_fn::C
    jacob_constr_fn::J
    projection_solver::PS
    projection_solver_kwargs::PK
end

function ConstrainedEuclideanMetricSystem(
    metric,
    ℓ;
    constr,
    jacob_constr,
    projection_solver=solve_projection_onto_manifold_newton,
    projection_solver_kwargs=(;
        constraint_tol=1e-9,
        position_tol=1e-8,
        divergence_tol=1e10,
        max_iters=50,
    ),
)
    return ConstrainedEuclideanMetricSystem(
        metric,
        ℓ,
        constr,
        jacob_constr,
        projection_solver,
        projection_solver_kwargs,
    )
end

dimension(system::ConstrainedEuclideanMetricSystem) =
    LogDensityProblems.dimension(ℓ(system))

metric(system::ConstrainedEuclideanMetricSystem) = system.metric

constr(z::PhasePoint, system::ConstrainedEuclideanMetricSystem) =
    system.constr_fn(z.q)
constr(q::AbstractVector, system::ConstrainedEuclideanMetricSystem) =
    system.constr_fn(q)

jacob_constr(z::PhasePoint, system::ConstrainedEuclideanMetricSystem) =
    system.jacob_constr_fn(z.q)
jacob_constr(q::AbstractVector, system::ConstrainedEuclideanMetricSystem) =
    system.jacob_constr_fn(q)

function jacob_constr_inner_product(
    jacob_constr_1,
    inner_product_matrix,
    jacob_constr_2=nothing,
)
    if isnothing(jacob_constr_2)
        return jacob_constr_1 * inner_product_matrix * jacob_constr_1'
    end
    return jacob_constr_1 * inner_product_matrix * jacob_constr_2'
end

function h₂(z::PhasePoint, system::ConstrainedEuclideanMetricSystem)
    return 0.5 * invquad(metric(system), z.p)
end

h2(z::PhasePoint, system::ConstrainedEuclideanMetricSystem) = h₂(z, system)

function ∂h₁∂q(z::PhasePoint, system::ConstrainedEuclideanMetricSystem)
    return -last(LogDensityProblems.logdensity_and_gradient(ℓ(system), z.q))
end

function ∂h₂∂p(z::PhasePoint, system::ConstrainedEuclideanMetricSystem)
    return metric(system) \ z.p
end

function ∂h₂∂q(z::PhasePoint, ::ConstrainedEuclideanMetricSystem)
    return zeros(eltype(z.q), dimension(z))
end

function _identity_matrix(::Type{T}, d::Integer) where {T}
    return Matrix{T}(LinearAlgebra.I, d, d)
end

function dh2_flow_dmom(z::PhasePoint, dt, system::ConstrainedEuclideanMetricSystem)
    d = dimension(z)
    metric_inv = metric(system) \ _identity_matrix(eltype(z.q), d)
    return dt .* metric_inv, _identity_matrix(eltype(z.q), d)
end

function project_onto_cotangent_space!(
    mom::AbstractVector,
    z::PhasePoint,
    system::ConstrainedEuclideanMetricSystem,
)
    jacob = jacob_constr(z, system)
    metric_inv = metric(system) \ _identity_matrix(eltype(mom), length(mom))
    gram = jacob_constr_inner_product(jacob, metric_inv)
    mom .-= jacob' * (gram \ (jacob * (metric_inv * mom)))
    return mom
end

function project_onto_cotangent_space(
    mom::AbstractVector,
    z::PhasePoint,
    system::ConstrainedEuclideanMetricSystem,
)
    return project_onto_cotangent_space!(copy(mom), z, system)
end

function rand!(rng::AbstractRNG, p::Vector, z::PhasePoint, system::ConstrainedEuclideanMetricSystem)
    randn!(rng, p)
    unwhiten!(system.metric, p)
    project_onto_cotangent_space!(p, z, system)
    return nothing
end

function solve_projection_onto_manifold_quasi_newton(
    state::PhasePoint,
    state_prev::PhasePoint,
    time_step,
    system::ConstrainedEuclideanMetricSystem;
    constraint_tol::Real=1e-9,
    position_tol::Real=1e-8,
    divergence_tol::Real=1e10,
    max_iters::Int=50,
    norm=LinearAlgebra.norm,
)
    mu = zeros(eltype(state.q), length(state.q))
    jacob_constr_prev = jacob_constr(state_prev, system)
    dpos_dmom, dmom_dmom = dh2_flow_dmom(state_prev, abs(time_step), system)
    inv_jacob_constr_inner_product =
        jacob_constr_inner_product(jacob_constr_prev, dpos_dmom) \
        _identity_matrix(eltype(state.q), size(jacob_constr_prev, 1))

    for i in 1:max_iters
        constr_val = constr(state, system)
        err = norm(constr_val)
        delta_mu = jacob_constr_prev' * (inv_jacob_constr_inner_product * constr_val)
        delta_pos = dpos_dmom * delta_mu
        if err > divergence_tol || isnan(err)
            throw(
                ArgumentError(
                    "Quasi-Newton solver diverged on iteration $i (|constr|=$(err)).",
                ),
            )
        end
        if err < constraint_tol && norm(delta_pos) < position_tol
            state.p .-= sign(time_step) .* (dmom_dmom * mu)
            return state
        end
        mu .+= delta_mu
        state.q .-= delta_pos
    end

    throw(ArgumentError("Quasi-Newton solver did not converge after $max_iters iterations."))
end

function solve_projection_onto_manifold_newton(
    state::PhasePoint,
    state_prev::PhasePoint,
    time_step,
    system::ConstrainedEuclideanMetricSystem;
    constraint_tol::Real=1e-9,
    position_tol::Real=1e-8,
    divergence_tol::Real=1e10,
    max_iters::Int=50,
    norm=LinearAlgebra.norm,
)
    mu = zeros(eltype(state.q), length(state.q))
    jacob_constr_prev = jacob_constr(state_prev, system)
    dpos_dmom, dmom_dmom = dh2_flow_dmom(state_prev, abs(time_step), system)

    for i in 1:max_iters
        jacob_constr_now = jacob_constr(state, system)
        constr_val = constr(state, system)
        err = norm(constr_val)
        delta_mu = jacob_constr_prev' * (
            jacob_constr_inner_product(jacob_constr_now, dpos_dmom, jacob_constr_prev) \
            constr_val
        )
        delta_pos = dpos_dmom * delta_mu
        if err > divergence_tol || isnan(err)
            throw(ArgumentError("Newton solver diverged on iteration $i (|constr|=$(err))."))
        end
        if err < constraint_tol && norm(delta_pos) < position_tol
            state.p .-= sign(time_step) .* (dmom_dmom * mu)
            return state
        end
        mu .+= delta_mu
        state.q .-= delta_pos
    end

    throw(ArgumentError("Newton solver did not converge after $max_iters iterations."))
end

function _project_onto_manifold!(
    state::PhasePoint,
    state_prev::PhasePoint,
    time_step,
    system::ConstrainedEuclideanMetricSystem,
)
    return system.projection_solver(
        state,
        state_prev,
        time_step,
        system;
        system.projection_solver_kwargs...,
    )
end

function Φ₁!(z::PhasePoint, system::AbstractConstrainedTractableFlowSystem, ϵ::Real)
    z.p .-= ϵ .* ∂h₁∂q(z, system)
    project_onto_cotangent_space!(z.p, z, system)
    return nothing
end

function Φ₂!(z::PhasePoint, system::AbstractConstrainedTractableFlowSystem, ϵ::Real)
    z_prev = copy(z)
    z.q .+= ϵ .* ∂h₂∂p(z, system)
    refresh!(z)
    _project_onto_manifold!(z, z_prev, ϵ, system)
    project_onto_cotangent_space!(z.p, z, system)
    refresh!(z)
    return nothing
end
