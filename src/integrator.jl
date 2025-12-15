# Integrator methods for solving discretized hamiltonian systems
abstract type AbstractIntegrator end

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
