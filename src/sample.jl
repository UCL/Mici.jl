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
