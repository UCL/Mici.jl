abstract type AbstractTransition end

struct IndependentMomentumTransition <: AbstractTransition end

struct MetropolisTransition <: AbstractTransition end

struct CorrelatedMomentumTransition <: AbstractTransition
    resample_coefficient::Float64
    function CorrelatedMomentumTransition(resample_coefficient)
        @assert 0.0 ≤ resample_coefficient ≤ 1.0
        new(resample_coefficient)
    end
end


function transition!(::IndependentMomentumTransition, system::EuclideanSystem, state::ChainState, rng::AbstractRNG)
    state.p .= sample_p(system, rng)
    state.p_prop .= state.p
end

function transition!(
    ::MetropolisTransition,
    integrator::AbstractIntegrator,
    system::AbstractSystem,
    state::ChainState,
    rng::AbstractRNG,
    ℓπ,
)
    integrate!(integrator, system, state.proposal, ℓπ)
    ΔH = H(system, state.current) - H(system, state.proposal)
    accepted = log(rand(rng)) < ΔH

    update_state!(state, accepted, ℓπ)
end
