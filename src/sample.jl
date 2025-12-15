using AbstractMCMC


struct MiciSampler{I<:AbstractIntegrator, S<:AbstractChainState} <: AbstractMCMC.AbstractSampler
    integrator::I
    state::S
end

function MiciSampler(I::AbstractIntegrator, q::AbstractVector)
    return MiciSampler(I, ChainState(q))
end

function sample(
    h::AbstractSystem,
    integrator::AbstractIntegrator,
    chain_state::AbstractChainState,
    rng::AbstractRNG,
)
    q₁ = q(chain_state)
    p₁ = sample_p(h, rng)
    current_state = MarkovChainState(q₁, p₁)
    proposed_state = MarkovChainState(copy(q₁), copy(p₁))

    integrate!(integrator, proposed_state)

    accept_prob = exp(H(h, current_state) - H(h, proposed_state))
    accepted = rand(rng) < accept_prob
    new_state = accepted ? proposed_state : current_state

    update_state!(chain_state, q(new_state), p(new_state), accepted)

    return q(new_state), chain_state
end


function sample_chain(
    h::AbstractSystem,
    integrator::AbstractIntegrator,
    q₁::AbstractVector,
    N::Int,
    rng::AbstractRNG,
)
    samples = zeros(eltype(q₁), N, length(q₁))
    chain_state = ChainState(q₁)

    q = q₁
    for n = 1:N
        q, chain_state = sample(h, integrator, chain_state, rng)
        samples[n, :] = q
    end
    return samples, chain_state
end
