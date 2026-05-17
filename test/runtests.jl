using Mici
using Test

@testset "Mici.jl" begin
    include("test_transition.jl")
    include("test_e2e.jl")
end
