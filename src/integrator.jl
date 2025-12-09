# Integrator methods for solving discretized hamiltonian systems
abstract type AbstractIntegrator end

struct LeapfrogIntegrator{H<:AbstractEuclideanSystem} <: AbstractIntegrator
    h::H
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

function integrate!(lfi::LeapfrogIntegrator, state::AbstractChainState)
    for n = 1:lfi.T
        step!(lfi.h, state, lfi.ε)
    end
end
