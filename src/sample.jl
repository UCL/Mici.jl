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


# function sample_chain(
#     h::AbstractSystem,
#     integrator::AbstractIntegrator,
#     q₁::AbstractVector,
#     N::Int,
#     rng::AbstractRNG,
# )
#     samples = zeros(eltype(q₁), N, length(q₁))
#     accepts = BitVector(undef, N)
#     q = q₁
#     for n = 1:N
#         q, accepted = hmc_step(h, integrator, q, rng)
#         samples[n, :] = q
#         accepts[n] = accepted
#     end
#     return samples, accepts
# end
