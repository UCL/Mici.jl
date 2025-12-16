function gi_problem(
    h::AbstractSystem,
    state::AbstractChainState,
    T::Int,
    ε::Float64
)

    q₀ = q(state)
    p₀ = p(state)

    # This is not ideal
    function v!(dq, t, q, p, params)
        state_tmp = MarkovChainState(q, p)
        dq .= ∂H₂∂p(h, state_tmp)
    end

    function f!(dp, t, q, p, params)
        state_tmp = MarkovChainState(q, p)
        dp .= -∂H∂q(h, state_tmp)
    end

    function hamiltonian(t, q, p, params)
        state_tmp = MarkovChainState(q, p)
        return H(h, state_tmp)
    end

    return HODEProblem(v!,f!,hamiltonian, T, ε, q₀, p₀)
end

function initialise_step(problem::GeometricProblem, method::GeometricMethod)
    integrator = GeometricIntegrator(problem, method)

    solstep = SolutionStep(
        problem;
        internal = internal_variables(integrator)
    )

    return solstep, integrator
end

current_qp(solstep::SolutionStep) =
    solution(solstep).q, solution(solstep).p