# Integrator methods for solving discretized hamiltonian systems
abstract type AbstractIntegrator end

struct LeapfrogIntegrator{H<:AbstractEuclideanSystem} <: AbstractIntegrator
    h::H
    ε::Float64
    T::Int
end

struct LeapfrogAdapterIntegrator{H<:AbstractSystem} <: AbstractIntegrator
    h::H
    ε::Float64
    T::Int
end

function LeapfrogAdapterIntegrator(
    h::AbstractSystem,
    ε::Real,
    T::Integer,
)
    LeapfrogAdapterIntegrator(h, Float64(ε), Int(T))
end

function step!(
    h::AbstractEuclideanSystem,
    state::ChainState,
    ε::Float64,
)
    p(state) .-= (ε/2) .* ∂H₁∂q(h, state)
    q(state) .+= ε .* ∂H₂∂p(h, state)
    p(state) .-= (ε/2) .* ∂H₁∂q(h, state)
end

function integrate!(lfi::LeapfrogIntegrator, state::ChainState)
    for n = 1:lfi.T
        step!(lfi.h, state, lfi.ε)
    end
end

function step!(lai::LeapfrogAdapterIntegrator, state::ChainState)
    println(state)
    x = vcat(q(state), p(state))

    # Integrate a single step
    adapter = Gni.LeapfrogAdapter(lai.h, x, (0.0, lai.ε), lai.ε)
    Gni.GeometricIntegrators.integrate!(adapter.core.solution, adapter.core.integrator)
    x_next = adapter.core.solution.q
    d = length(x_next) ÷ 2
    # Do manual update of state
    q(state) .= @view x_next[begin:d]
    p(state) .= @view x_next[d+1:end]
end

function integrate!(lai::LeapfrogAdapterIntegrator, state::ChainState)
    for n = 1:lai.T
        step!(lai, state)
    end
end
