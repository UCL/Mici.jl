using Mici
using Test

include("dependencies_for_runtests.jl")

@testset "Mici.jl" begin
    include("test_transition.jl")
    include("test_e2e.jl")
end
