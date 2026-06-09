#!/usr/bin/env julia

using DelimitedFiles
using LogDensityProblems
using PDMats
using Printf
using Random
using Mici

const R = 1.0
const r = 0.35
const alpha = 0.9

struct TorusLogDensity end

LogDensityProblems.dimension(::TorusLogDensity) = 3

function LogDensityProblems.logdensity(::TorusLogDensity, q)
    x, y, z = q
    rho = sqrt(x^2 + y^2)
    theta = atan(y, x)
    phi = atan(z, rho - R)

    area_correction = log1p(r * cos(phi) / R)
    modulation = log1p(alpha * sin(4 * theta) * cos(phi))
    return area_correction - modulation
end

function LogDensityProblems.logdensity_and_gradient(::TorusLogDensity, q)
    x, y, z = q
    rho = sqrt(x^2 + y^2)
    theta = atan(y, x)
    phi = atan(z, rho - R)

    area_denom = 1 + r * cos(phi) / R
    modulation_denom = 1 + alpha * sin(4 * theta) * cos(phi)

    dlogdens_dtheta = -4 * alpha * cos(4 * theta) * cos(phi) / modulation_denom
    dlogdens_dphi =
        -(r / R) * sin(phi) / area_denom +
        alpha * sin(4 * theta) * sin(phi) / modulation_denom

    rho_safe = max(rho, eps(eltype(q)))
    rho2 = max(rho^2, eps(eltype(q)))
    torus_denom = max((rho - R)^2 + z^2, eps(eltype(q)))

    dtheta_dx = -y / rho2
    dtheta_dy = x / rho2
    dphi_dx = -z * x / (rho_safe * torus_denom)
    dphi_dy = -z * y / (rho_safe * torus_denom)
    dphi_dz = (rho - R) / torus_denom

    grad = [
        dlogdens_dtheta * dtheta_dx + dlogdens_dphi * dphi_dx,
        dlogdens_dtheta * dtheta_dy + dlogdens_dphi * dphi_dy,
        dlogdens_dphi * dphi_dz,
    ]
    return LogDensityProblems.logdensity(TorusLogDensity(), q), grad
end

function torus_constraint(q)
    x, y, z = q
    rho = sqrt(x^2 + y^2)
    return [((rho - R)^2 + z^2 - r^2)]
end

function torus_constraint_jacobian(q)
    x, y, z = q
    rho = sqrt(x^2 + y^2)
    if rho == 0
        return reshape([0.0, 0.0, 2.0 * z], 1, 3)
    end
    return reshape(
        [
            2.0 * (rho - R) * x / rho,
            2.0 * (rho - R) * y / rho,
            2.0 * z,
        ],
        1,
        3,
    )
end

function sample_torus_chain(; n_samples=2000, n_warmup=500, n_steps=3, step_size=0.03, seed=1234)
    rng = MersenneTwister(seed)
    system = ConstrainedEuclideanMetricSystem(
        ScalMat(3, 1.0),
        TorusLogDensity();
        constr=torus_constraint,
        jacob_constr=torus_constraint_jacobian,
        projection_solver=solve_projection_onto_manifold_quasi_newton,
        projection_solver_kwargs=(;
            constraint_tol=1e-10,
            position_tol=1e-10,
            max_iters=100,
        ),
    )
    integrator = ConstrainedLeapfrogIntegrator(step_size=step_size)

    theta0 = 0.7
    phi0 = 1.2
    q0 = [
        (R + r * cos(phi0)) * cos(theta0),
        (R + r * cos(phi0)) * sin(theta0),
        r * sin(phi0),
    ]
    z = Mici.PhasePoint(copy(q0), zeros(3), NaN, zeros(3), false)
    Mici.project_onto_cotangent_space!(z.p, z, system)

    samples = Matrix{Float64}(undef, 3, n_samples)
    accepted = 0
    max_constraint_violation = 0.0

    for iter in 1:(n_warmup + n_samples)
        Mici.rand!(rng, z.p, z, system)
        proposed = copy(z)
        for _ in 1:n_steps
            Mici.step!(proposed, integrator, system)
        end
        proposed.p .*= -1

        delta_h = Mici.h(z, system) - Mici.h(proposed, system)
        accepted_step = !isnan(delta_h) && log(rand(rng)) < min(0.0, delta_h)
        if accepted_step
            copy!(z, proposed)
            if iter > n_warmup
                accepted += 1
            end
        end

        if iter > n_warmup
            samples[:, iter - n_warmup] .= z.q
            max_constraint_violation = max(max_constraint_violation, maximum(abs.(torus_constraint(z.q))))
        end
    end

    return samples, accepted / n_samples, max_constraint_violation
end

function main()
    samples, acceptance_rate, max_constraint_violation = sample_torus_chain()
    writedlm("torus_samples.csv", samples', ',')
    @printf("Wrote torus_samples.csv\n")
    @printf("Acceptance rate: %.2f%%\n", 100 * acceptance_rate)
    @printf("Max constraint violation: %.3e\n", max_constraint_violation)
    @printf("First sample: [%.4f, %.4f, %.4f]\n", samples[1, 1], samples[2, 1], samples[3, 1])
end

main()
