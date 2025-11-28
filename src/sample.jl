# Generate samples from target distribution using Hamiltonian Monte Carlo
function hmc_step(
    h::AbstractSystem,
    integrator::AbstractIntegrator,
    q₁::AbstractVector,
    rng::AbstractRNG,
)
    p₁ = sample_p(h, rng)
    state = ChainState(q₁, p₁)
    proposed_state = ChainState(copy(q₁), copy(p₁))

    integrate!(integrator, proposed_state)

    accept_prob = exp(H(h, state) - H(h, proposed_state))

    if rand(rng) < accept_prob
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
        q, accepted = hmc_step(h, integrator, q, rng)
        samples[n, :] = q
        accepts[n] = accepted
    end
    return samples, accepts
end
