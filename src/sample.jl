"""
    AbstractMiciSampler{S, I} <: AbstractMCMC.AbstractSampler

Abstract supertype for Mici samplers, parameterized by the system type `S` and integrator type `I`.
"""
abstract type AbstractMiciSampler{S, I} <: AbstractMCMC.AbstractSampler end

"""
    HMC{S,I,TI,TM} <: AbstractMiciSampler{S,I}

Struct representing a Hamiltonian Monte Carlo sampler, parameterized by:
  - `S`  - type of the system (e.g., `EuclideanSystem`),
  - `I`  - type of the integrator (e.g., `LeapfrogIntegrator`),
  - `TI` - type of the integration transition (e.g., `StaticMetropolisIntegrationTransition`),
  - `TM` - type of the momentum transition (e.g., `IndependentMomentumTransition`).
"""
struct HMC{S,I,TI,TM} <: AbstractMiciSampler{S,I}
    integration_transition::TI
    momentum_transition::TM
end

function HMC{S,I}(integration_time::Real) where {S,I}
    HMC{S,I}(StaticMetropolisIntegrationTransition(integration_time))
end

function HMC{S,I}(integration_transition::TI, momentum_transition::TM=IndependentMomentumTransition()) where {S,I,TI,TM}
    HMC{S,I,TI,TM}(integration_transition, momentum_transition)
end

function HMC{S,I}(integration_time_lower::Real, integration_time_upper::Real) where {S,I}
    HMC{S,I}(
        RandomMetropolisIntegrationTransition(
            integration_time_lower, integration_time_upper
        ),
    )
end

const EuclideanHMC{I,TI,TM} = HMC{EuclideanSystem,I,TI,TM}

function EuclideanHMC(integration_time::Real)
    EuclideanHMC{LeapfrogIntegrator}(StaticMetropolisIntegrationTransition(integration_time))
end

function EuclideanHMC(integration_time_lower::Real, integration_time_upper::Real)
    EuclideanHMC{LeapfrogIntegrator}(integration_time_lower, integration_time_upper)
end

function state_type(
    ::HMC{S,I,TI,TM}
) where {S,I,TI<:AbstractMetropolisIntegrationTransition,TM}
    MetropolisHMCState
end


###OLD
# Generate samples from target distribution using Hamiltonian Monte Carlo
function hmc_step(
    h::AbstractSystem,
    integrator::AbstractIntegrator,
    q₁::AbstractVector,
    rng::AbstractRNG,
)
    p₁ = sample_p(h, rng)
    # println("sampled momentum: $(p₁)")
    state = ChainState(q₁, p₁)
    proposed_state = ChainState(copy(q₁), copy(p₁))

    integrate!(integrator, proposed_state)
    # println("after integration propsed_state: $(proposed_state)")
    # println("original_state: $(state)")

    accept_prob = exp(H(h, state) - H(h, proposed_state))
    random_draw = rand(rng)
    # println("accept_prob: $(accept_prob)")
    # println("random_draw: $(random_draw)")
    if random_draw < accept_prob
        return q(proposed_state), true
    else
        return q(state), false
    end
end


function sample_chain(
    h::AbstractSystem,
    integrator::AbstractIntegrator,
    q₁::AbstractVector,
    N::Int,
    rng::AbstractRNG,
)
    samples = zeros(eltype(q₁), N, length(q₁))
    accepts = BitVector(undef, N)
    q = q₁
    for n = 1:N
        # println("at iter $(n):")
        # println("state: $(q)")
        q, accepted = hmc_step(h, integrator, q, rng)
        # println("accepted: $(accepted)")
        samples[n, :] = q
        accepts[n] = accepted
    end
    return samples, accepts
end
