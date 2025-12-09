using Mici
using Test

@testset "Mici.jl" begin
    include("test_end_to_end.jl")
    include("test_abstractmcmc.jl")
end
