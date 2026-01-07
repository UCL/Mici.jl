# Integrator methods for solving discretized hamiltonian systems

abstract type AbstractIntegrator end

"""
    LeapfrogIntegrator <: AbstractIntegrator

Leapfrog integrator with manual step size `ε` and number of steps `T`.
"""
struct LeapfrogIntegrator <: AbstractIntegrator
    ε::Float64
    T::Int
end

"""
    step!(h::AbstractEuclideanSystem, point::PhasePoint, ε::Float64, ℓπ)

Perform a single leapfrog step of size `ε` on the given `point` in system `h`.
"""
function step!(
    h::AbstractEuclideanSystem,
    point::PhasePoint,
    ε::Float64,
    ℓπ,
)
    point.p .-= (ε/2) .* ∂H₁∂q(h, point)
    point.q .+= ε .* ∂H₂∂p(h, point)
    refresh_phasepoint!(point, ℓπ)
    point.p .-= (ε/2) .* ∂H₁∂q(h, point)
end

"""
    integrate!(lfi::LeapfrogIntegrator, h::AbstractEuclideanSystem, point::PhasePoint, ℓπ)

Perform leapfrog integration using the given `lfi` integrator on the `point` in system `h`.
"""
function integrate!(
    lfi::LeapfrogIntegrator,
    h::AbstractEuclideanSystem,
    point::PhasePoint,
    ℓπ,
)
    for n = 1:lfi.T
        step!(h, point, lfi.ε, ℓπ)
    end
end
