module Mici

using Random
using PDMats
using AbstractMCMC
using LogDensityProblems

include("typedefs.jl")

include("state.jl")

include("system.jl")
export EuclideanSystem

include("constrained_system.jl")
export
    AbstractConstrainedSystem,
    AbstractConstrainedTractableFlowSystem,
    ConstrainedEuclideanMetricSystem,
    solve_projection_onto_manifold_newton,
    solve_projection_onto_manifold_quasi_newton

include("integrator.jl")
export LeapfrogIntegrator

include("composition_integrator.jl")
export CompositionIntegrator, AbstractCompositionMethod, StrangA, StrangB, McLachlan2, McLachlan4, TripleJump, SuzukiFractal

include("constrained_integrator.jl")
export ConstrainedLeapfrogIntegrator, ConstrainedCompositionIntegrator

include("gni.jl")


include("transition.jl")
export IndependentMomentumTransition, CorrelatedMomentumTransition, RandomMetropolisIntegrationTransition, StaticMetropolisIntegrationTransition

include("sample.jl")
export HMC, EuclideanHMC

include("abstractmcmc.jl")

end
