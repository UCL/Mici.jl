@testset "Integrator unit tests for dimension $dimension" for dimension in (1, 2, 5, 10)
    rng = StableRNG(SEED)
    metric = ScalMat(dimension, 1.)
    log_density = 𝒩(zeros(dimension), ScalMat(dimension, 1.))
    system = EuclideanSystem(metric, log_density)
    phase_point = Mici.sample_initial_phase_point(rng, system, nothing)
    initial_phase_point = copy(phase_point)
    ϵ = 0.5
    integrator = LeapfrogIntegrator(ϵ)
    Mici.step!(phase_point, integrator, system)
    # Single step should keep phase point components finite and change values
    @test all(isfinite.(phase_point.q))
    @test all(isfinite.(phase_point.p))
    @test all(phase_point.q != initial_phase_point.q)
    @test all(phase_point.p != initial_phase_point.p)
    # Composing step with momentum flips should return us to initial phase point 
    phase_point.p *= -1
    Mici.step!(phase_point, integrator, system)
    phase_point.p *= -1
    @test all(phase_point.q ≈ initial_phase_point.q)
    @test all(phase_point.p ≈ initial_phase_point.p)
    # Stepping with ϵ=0 should not change phase point value
    phase_point = copy(initial_phase_point)
    zero_step_integrator = LeapfrogIntegrator(0.)
    Mici.step!(phase_point, zero_step_integrator, system)
    @test all(phase_point.q ≈ initial_phase_point.q)
    @test all(phase_point.p ≈ initial_phase_point.p)
    # Integrating over multiple steps should approximately conserve Hamiltonian
    initial_h = Mici.h(initial_phase_point, system)
    phase_point = copy(initial_phase_point)
    for _ in 1:100
        Mici.step!(phase_point, integrator, system)
    end
    final_h = Mici.h(phase_point, system)
    @test abs(final_h - initial_h) < 0.1
end