using Mici.Mici: MetropolisHMCState, PhasePoint, EuclideanSystem, LeapfrogIntegrator, NUTSTransition
using Mici.Mici: new_leaf, merge_subtrees, build_tree, transition!, h, NUTSTreeContext, NUTSTreeStats

@testset "NUTS build_tree for depth $depth" for depth in (1, 2, 5)
    ℓ = 𝒩()
    rng = StableRNG(1234)
    q = [1.0, 1.0]
    p = [0.1, 0.1]
    phase_point = PhasePoint(q, p, 2)
    M = PDMat([1.0 0.0; 0.0 1.0])
    system = EuclideanSystem(M, ℓ)
    integrator = LeapfrogIntegrator(0.001)
    context = NUTSTreeContext(integrator, system, h(phase_point, system), 1000.0)
    stats = NUTSTreeStats()

    direction = 1
    _, tree, _ = build_tree(rng, depth, direction, phase_point, context, stats)

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

@testset "NUTS transition! divergence check" begin
    ℓ = 𝒩()
    q = [1.0, 1.0]
    p = [1000.0, 1000.0]
    rng = StableRNG(1234)
    phase_point = PhasePoint(q, p, 2)
    M = PDMat([1.0 0.5; 0.5 1.5])
    system = EuclideanSystem(M, ℓ)
    integrator = LeapfrogIntegrator(0.1)
    state = MetropolisHMCState(phase_point, system, integrator)
    transition = NUTSTransition(4, 100.0)
    stats = transition!(state, rng, transition)

    @test stats.diverged == true

end

@testset "NUTS transition! statistics check" begin
    ℓ = 𝒩()
    q = [0.0, 0.0]
    p = [0.25, 1.5]
    rng = StableRNG(1234)
    phase_point = PhasePoint(q, p, 2)
    M = PDMat([1. 0.5; 0.5 10.])
    system = EuclideanSystem(M, ℓ)
    integrator = LeapfrogIntegrator(1.9)
    state = MetropolisHMCState(phase_point, system, integrator)
    transition = NUTSTransition(5, 1000.0)
    stats = transition!(state, rng, transition)

    display(stats)

end

