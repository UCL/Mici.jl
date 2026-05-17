include("dependencies_for_runtests.jl")

using Mici.Mici: MetropolisHMCState, PhasePoint, EuclideanSystem, LeapfrogIntegrator, NUTSTransition
using Mici.Mici: new_leaf, merge_subtrees, build_tree, transition!, NUTSTreeContext, h

@testset "NUTS build_tree for depth $depth" for depth in (1, 2, 5, 10)
    ℓ = 𝒩()
    rng = StableRNG(1234)
    q = [1.0, 1.0]
    p = [1.0, 1.0]
    phase_point = PhasePoint(q, p, 2)
    M = PDMat([1.0 0.5; 0.5 1.0])
    system = EuclideanSystem(M, ℓ)
    integrator = LeapfrogIntegrator(0.1)
    context = NUTSTreeContext(integrator, system, h(phase_point, system), 10.0)

    direction = 1
    tree, _ = build_tree(rng, depth, direction, phase_point, context)

    Mici.step!(phase_point, integrator, system; direction)

    @test tree.depth == depth
    if direction == 1
        @test all(tree.left.q == phase_point.q)
        @test all(tree.left.p == phase_point.p)
    else
        @test all(tree.right.q == phase_point.q)
        @test all(tree.right.p == phase_point.p)
    end

    for _ in 1:(2^depth - 1)
        Mici.step!(phase_point, integrator, system; direction)
    end

    if direction == 1
        @test all(tree.right.q ≈ phase_point.q)
        @test all(tree.right.p ≈ phase_point.p)
    else
        @test all(tree.left.q ≈ phase_point.q)
        @test all(tree.left.p ≈ phase_point.p)
    end

end

@testset "NUTS transition" begin
    ℓ = 𝒩()
    q = [1.0, 1.0]
    p = [1.0, 1.0]
    rng = StableRNG(1234)
    phase_point = PhasePoint(q, p, 2)
    M = PDMat([1.0 0.5; 0.5 1.5])
    system = EuclideanSystem(M, ℓ)
    integrator = LeapfrogIntegrator(0.1)
    state = MetropolisHMCState(phase_point, system, integrator)

    transition = NUTSTransition(4, 10.0)
    transition!(state, rng, transition)

end