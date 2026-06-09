"""
    AbstractTransition

Abstract supertype for transitions in MCMC samplers. A transition represents a change in state 
of the Markov chain in either position or momentum.
"""
abstract type AbstractTransition end

"""
    AbstractIntegrationTransition <: AbstractTransition

Abstract supertype for integration transitions in MCMC samplers. Integration transitions
update the state according to the Hamiltonian dynamics of the system, typically using a numerical integrator.
"""
abstract type AbstractIntegrationTransition <: AbstractTransition end

"""
    AbstractMetropolisIntegrationTransition{T} <: AbstractIntegrationTransition

Abstract supertype for Metropolis-adjusted integration transitions in MCMC samplers, parameterized by:
    T -- type of the integration time (e.g., `Float64`)
"""
abstract type AbstractMetropolisIntegrationTransition{T} <: AbstractIntegrationTransition end

"""
    AbstractNUTSTransition{T} <: AbstractIntegrationTransition
"""
abstract type AbstractNUTSTransition <: AbstractIntegrationTransition end

"""
    AbstractMomentumTransition <: AbstractTransition

Abstract supertype for momentum transitions in MCMC samplers. 
Momentum transitions update the momentum component of the state, typically by resampling from a distribution or applying a transformation.
"""
abstract type AbstractMomentumTransition <: AbstractTransition end

"""
    IndependentMomentumTransition <: AbstractMomentumTransition

Struct representing an independent momentum transition, where the momentum is resampled independently from a distribution 
defined by the system (e.g., a Gaussian distribution with covariance given by the metric of the system).
"""
struct IndependentMomentumTransition <: AbstractTransition end

struct CorrelatedMomentumTransition{T} <: AbstractMomentumTransition
    correlation_coefficient::T
end

""" 
    StaticMetropolisIntegrationTransition{T} <: AbstractMetropolisIntegrationTransition{T}

Struct for a static Metropolis-adjusted integration transition, where the integration time is fixed.
"""
struct StaticMetropolisIntegrationTransition{T} <: AbstractMetropolisIntegrationTransition{T}
    integration_time::T
end

struct RandomMetropolisIntegrationTransition{T} <: AbstractMetropolisIntegrationTransition{T}
    integration_time_lower::T
    integration_time_upper::T
end

"""
    metropolis_integration_transition!(state::MetropolisHMCState, rng::AbstractRNG, integration_time::Real)

Perform a Metropolis-adjusted integration transition
"""
function metropolis_integration_transition!(
    state::MetropolisHMCState, rng::AbstractRNG, integration_time::Real
)
    n_step = Int(integration_time ÷ state.integrator.ϵ)
    copy!(state.proposed_phase_point, state.phase_point)
    for s in 1:n_step
        step!(state.proposed_phase_point, state.integrator, state.system)
    end
    state.proposed_phase_point.p .*= -1
    Δh = h(state.phase_point, state.system) - h(state.proposed_phase_point, state.system)
    accept_probability = isnan(Δh) ? 0.0 : exp(min(0.0, Δh))
    accepted = rand(rng) < accept_probability
    if accepted
        copy!(state.phase_point, state.proposed_phase_point)
    end
    return (; accept_probability, accepted, n_step)
end

function transition!(
    state::MetropolisHMCState,
    rng::AbstractRNG,
    transition::StaticMetropolisIntegrationTransition,
)
    metropolis_integration_transition!(state, rng, transition.integration_time)
end

function transition!(
    state::MetropolisHMCState,
    rng::AbstractRNG,
    transition::RandomMetropolisIntegrationTransition{T},
) where {T<:Real}
    integration_time =
        transition.integration_time_lower +
        rand(rng) * (transition.integration_time_upper - transition.integration_time_lower)
    metropolis_integration_transition!(state, rng, integration_time)
end

function transition!(
    state::AbstractState, rng::AbstractRNG, ::IndependentMomentumTransition
)
    rand!(rng, state.phase_point.p, state.phase_point, state.system)
    return nothing
end

function transition!(
    state::AbstractState, rng::AbstractRNG, transition::CorrelatedMomentumTransition
)
    tmp = copy(state.phase_point.p)
    rand!(rng, tmp, state.phase_point, state.system)
    state.phase_point.p .*= transition.correlation_coefficient
    state.phase_point.p .+= tmp * sqrt(1 - transition.correlation_coefficient^2)
    return nothing
end

function no_u_turn(system::AbstractSystem, left_phase_point::PhasePoint, right_phase_point::PhasePoint, sum_momentum::Vector)
    ∂h∂p(left_phase_point, system)' * sum_momentum < 0 || ∂h∂p(right_phase_point, system)' * sum_momentum < 0
end
struct SubTree{C, T, W}
    left::C
    right::C
    momentum::T
    weight::W
    depth::Int
end

struct NUTSTransition{T} <: AbstractNUTSTransition
    max_depth::Int
    max_Δh::T
end

struct NUTSTreeContext{I,S,T}
    integrator::I
    system::S
    initial_h::T
    max_Δh::T
end

mutable struct NUTSTreeStats
    n_steps::Int
    diverged::Bool
    reject_prob::Float64
    sum_accept_prob::Float64
end

NUTSTreeStats() = NUTSTreeStats(0, false, 1.0, 0.0)

function new_leaf(
    phase_point::PhasePoint,
    h,
)
    # ToDo numerical stabilisation, logsumexp
    return SubTree(phase_point, phase_point, phase_point.p, exp(-h), 0)
end

function merge_subtrees(
    left::SubTree,
    right::SubTree,
)
    left.depth == right.depth || error("Cannot merge subtrees of different depths.")

    return SubTree(
        left.left,
        right.right,
        left.momentum + right.momentum,
        left.weight + right.weight,
        left.depth + 1,
    )
end

function build_tree(rng::AbstractRNG, depth::Int, direction::Int, phase_point::PhasePoint, context::NUTSTreeContext, stats::NUTSTreeStats)
    if depth == 0

        new_phase_point = copy(phase_point)
        step!(new_phase_point, context.integrator, context.system; direction)
        h_value = h(new_phase_point, context.system)
        tree = new_leaf(new_phase_point, h_value)

        Δh = h_value - context.initial_h
        terminate = !isfinite(Δh) || Δh > context.max_Δh
        stats.diverged = terminate
        stats.n_steps += 1
        accept_prob = isfinite(Δh) ? exp(min(-Δh, 0.0)) : 0.0
        stats.sum_accept_prob += accept_prob

        return terminate, tree, new_phase_point
    end

    terminate, inner_tree, inner_proposal = build_tree(rng, depth - 1, direction, phase_point, context, stats)
    if terminate
        return true, nothing, nothing
    end

    phase_point = direction == 1 ? inner_tree.right : inner_tree.left

    terminate, outer_tree, outer_proposal = build_tree(rng, depth - 1, direction, phase_point, context, stats)
    if terminate
        return true, nothing, nothing
    end

    left_subtree, right_subtree = if direction == 1
        inner_tree, outer_tree
    else
        outer_tree, inner_tree
    end
    tree = merge_subtrees(left_subtree, right_subtree)

    accept_outer_prob = min(outer_tree.weight / tree.weight, 1.0)
    proposal = rand(rng) < accept_outer_prob ? outer_proposal : inner_proposal

    terminate = no_u_turn(context.system, tree.left, tree.right, tree.momentum)

    return terminate, tree, proposal
end

function transition!(state::AbstractState, rng::AbstractRNG, transition::NUTSTransition)

    initial_h = h(state.phase_point, state.system)
    tree = new_leaf(copy(state.phase_point), initial_h)
    next_phase_point = copy(state.phase_point)
    context = NUTSTreeContext(state.integrator, state.system, initial_h, transition.max_Δh)
    stats = NUTSTreeStats()
    final_depth = 0

    for depth in 0:(transition.max_depth - 1)
        final_depth = depth

        direction = rand(rng, Bool) ? 1 : -1
        if direction == 1
            copy!(next_phase_point, tree.right)
        else
            copy!(next_phase_point, tree.left)
        end

        terminate, new_tree, proposal = build_tree(rng, depth, direction, next_phase_point, context, stats)

        if terminate
            break
        end

        # bias proposals towards new subtrees to encourage exploration of the state space
        accept_prob = min(new_tree.weight / tree.weight, 1.0)
        if rand(rng) < accept_prob
            copy!(state.phase_point, proposal)
        end
        stats.reject_prob *= 1.0 - accept_prob

        left_subtree = direction == 1 ? tree : new_tree
        right_subtree = direction == 1 ? new_tree : tree
        tree = merge_subtrees(left_subtree, right_subtree)

        if no_u_turn(context.system, tree.left, tree.right, tree.momentum)
            break
        end
    end

    accept_probability = stats.n_steps == 0 ? 0.0 : stats.sum_accept_prob / stats.n_steps

    return (;
        n_steps = stats.n_steps,
        diverged = stats.diverged,
        accept_probability = accept_probability,
        reject_prob = stats.reject_prob,
        tree_depth = final_depth,
    )
end