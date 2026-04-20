# Integrator methods for solving discretized hamiltonian systems
abstract type AbstractIntegrator end

struct LeapfrogIntegrator{H<:AbstractTractableFlowSystem} <: AbstractIntegrator
    system::H
    ε::Float64
    T::Int
end

struct LeapfrogAdapterIntegrator{H<:AbstractTractableFlowSystem} <: AbstractIntegrator
    system::H
    ε::Float64
    T::Int
end

function LeapfrogAdapterIntegrator(
    system::AbstractTractableFlowSystem,
    ε::Real,
    T::Integer,
)
    LeapfrogAdapterIntegrator(system, Float64(ε), Int(T))
end

function step!(
    system::AbstractTractableFlowSystem,
    z::PhasePoint,
    ε::Float64,
)
    p(state) .-= (ε/2) .* ∂H₁∂q(z, system)
    q(state) .+= ε .* ∂H₂∂p(z, system)
    p(state) .-= (ε/2) .* ∂H₁∂q(z, system)
end

function integrate!(lfi::LeapfrogIntegrator, z::PhasePoint)
    for n = 1:lfi.T
        step!(lfi.system, z, lfi.ε)
    end
end

function step!(lai::LeapfrogAdapterIntegrator, z::PhasePoint)
    x = vcat(z.q, z.p)

    # Integrate a single step
    adapter = Gni.LeapfrogAdapter(lai.h, x, (0.0, lai.ε), lai.ε)
    Gni.GeometricIntegrators.integrate!(adapter.core.solution, adapter.core.integrator)
    x_next = adapter.core.solution.q
    d = length(x_next) ÷ 2
    # Do manual update of state
    z.q .= @view x_next[begin:d]
    z.p .= @view x_next[d+1:end]
end

function integrate!(lai::LeapfrogAdapterIntegrator, z::PhasePoint)
    for n = 1:lai.T
        step!(lai, z)
    end
end