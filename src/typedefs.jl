"""
    AbstractSystem

Abstract supertype for Hamiltonian systems with energy
    h(q, p) = h₁(q) + h₂(q, p)
for position `q` and momentum `p` and where the energy is decomposed into two components, h₁ and h₂.

In a standard Euclidean System, h₁ and h₂ correspond to potential energy and
kinetic energy respectively. However, solving the Hamiltonian dynamics in more
complex systems may benefit from a more flexible distinction between (position) and
(momentum, position) energy components.
"""
abstract type AbstractSystem end

"""
    AbstractIntegrator
Abstract supertype for numerical integrators used to simulate Hamiltonian dynamics in MCMC samplers.
"""
abstract type AbstractIntegrator end



