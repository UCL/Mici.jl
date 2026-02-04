using LogDensityProblems
using Random
# Source code https://github.com/tpapp/LogDensityProblems.jl/blob/master/src/LogDensityProblems.jl
# LDP defines an evaluation interface only
# Can build upon LogDensityProblems, happy to discuss

# No geometric assumptions:
# does not impose geometry
# does not assume Euclidean metric (expects gradients to be vectors, for manifold HMC, gradients are vecotrs in ambient space, keep an eye on if this is restrictive)
# does not assume unconstrained space
# does not assume coordinate type
# does not do any linear algebra

methods(LogDensityProblems.logdensity)
# 0 methods for generic function "logdensity" from LogDensityProblems

# Define ToyModel logdensity
struct ToyModel end

LogDensityProblems.dimension(::ToyModel) = 2
LogDensityProblems.logdensity(::ToyModel, x) = -sum(abs2, x)

base = ToyModel()

println("Dimension: ", LogDensityProblems.dimension(base))
println("Log density at [1,2]: ",
        LogDensityProblems.logdensity(base, [1.0, 2.0]))

# # Define a custom state type
# struct CustomPoint
#     a::Float64
#     b::Float64
# end

# # Define a custom model to check for restrictions on state
# struct CustomModel end

# # Implement LDP interface
# LogDensityProblems.dimension(::CustomModel) = 2
# LogDensityProblems.logdensity(::CustomModel, p::CustomPoint) =
#     -(p.a^2 + p.b^2)

# custommodel = CustomModel()
# x = CustomPoint(1.0, 2.0)

# println(LogDensityProblems.logdensity(custommodel, x))

# does not constrain the state type
# 2 methods for generic function "logdensity" from LogDensityProblems:
#  [1] logdensity(::CustomModel, p::CustomPoint)
#      @ ~/Code/Mici.jl/test/manifold_design.jl:26
#  [2] logdensity(::ToyModel, x)
#      @ ~/Code/Mici.jl/test/manifold_design.jl:8

struct WrappedModel{LD}
    inner::LD
end

LogDensityProblems.dimension(m::WrappedModel) =
    LogDensityProblems.dimension(m.inner)

LogDensityProblems.logdensity(m::WrappedModel, x) =
    LogDensityProblems.logdensity(m.inner, x)

wrapped = WrappedModel(base)

println("Wrapped LogDensity: ", LogDensityProblems.logdensity(wrapped, [1.0, 2.0]))

double_wrapped = WrappedModel(wrapped)

println("Double wrapped logdensity: ", LogDensityProblems.logdensity(double_wrapped, [1.0, 2.0]))
# wrapping / composition allowed with no problems

# for AD, there exists https://github.com/tpapp/LogDensityProblemsAD.jl which supports ForwardDiff.jl, Enzyme.jl, Zygote.jl, Tracker.jl, ReverseDiff.jl and FiniteDifferences.jl.
# It states other AD frameworks are supported.

struct ConstrainedModel end

LogDensityProblems.dimension(::ConstrainedModel) = 1

# LogDensityProblems.logdensity(::ConstrainedModel, x) =
#     x[1] > 0 ? -x[1]^2 : -Inf

LogDensityProblems.logdensity(::ConstrainedModel, x) =
    abs(x[1]) < 1 ? -x[1]^2 : -Inf

model = ConstrainedModel()

println(LogDensityProblems.logdensity(model, [1.0]))
println(LogDensityProblems.logdensity(model, [-1.0]))

# allows constrained support, does not interfere with support handling, also allows hard constraints

struct SamplerModel end
LogDensityProblems.dimension(::SamplerModel) = 1
LogDensityProblems.logdensity(::SamplerModel, x) = -x[1]^2

model = SamplerModel()
rng = Random.default_rng()

# A naive random walk sampler
function simple_sampler(model, n)
    x = [0.0]
    samples = zeros(n)
    for i in 1:n
        proposal = x .+ 0.5 .* randn(rng, 1)
        logp_current = LogDensityProblems.logdensity(model, x)
        logp_proposal = LogDensityProblems.logdensity(model, proposal)

        if log(rand(rng)) < (logp_proposal - logp_current)
            x = proposal
        end

        samples[i] = x[1]
    end
    return samples
end

println(simple_sampler(model, 10))

# seems to be sampler agnostic
