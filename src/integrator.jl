# Integrator methods for solving discretized hamiltonian systems
abstract type AbstractIntegrator end

struct LeapfrogIntegrator <: AbstractIntegrator
    ε::Float64
    T::Int
end

function step!(
    h::AbstractEuclideanSystem,
    point::PhasePoint,
    ℓπ,
    ε::Float64,
)
    point.p .-= (ε/2) .* ∂H₁∂q(h, point)
    point.q .+= ε .* ∂H₂∂p(h, point)
    refresh_phasepoint!(point, ℓπ)
    point.p .-= (ε/2) .* ∂H₁∂q(h, point)
end

function integrate!(
    lfi::LeapfrogIntegrator,
    h::AbstractEuclideanSystem,
    point::PhasePoint,
    ℓπ,
)
    for n = 1:lfi.T
        step!(h, point, ℓπ, lfi.ε)
    end
end
