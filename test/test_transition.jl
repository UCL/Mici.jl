include("dependencies_for_runtests.jl")

using Mici.Mici: MetropolisHMCState, PhasePoint, EuclideanSystem, LeapfrogIntegrator
using Mici.Mici: new_leaf, merge_subtrees

@testset "NUTS" begin
    ℓ = 𝒩()
    q = [1.0, 1.0]
    p = [1.0, 1.0]
    z = PhasePoint(q, p, 2)
    M = PDMat([1.0 0.5; 0.5 1.0])
    system = EuclideanSystem(M, ℓ)
    integrator = LeapfrogIntegrator(0.1)

    state = MetropolisHMCState(
        z,
        system,
        integrator
    )

    subtree_1 = new_leaf(state)

    subtree_2 = new_leaf(state)

    merge_subtrees(subtree_1, subtree_2)

end