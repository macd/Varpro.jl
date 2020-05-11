# Unit tests for Varpro
#

using DelimitedFiles
using Test
using Random
using Varpro

include("set1.jl")
include("gaussmix.jl")

@testset "Varpro" begin

    verbose = false
    @test runone(rexp, verbose=verbose)
    @test runone(cexp, verbose=verbose)
    @test runone(example, verbose=verbose, atol=1e-4)
    @test runone(double_exponential, verbose=verbose)
    @test runone(ctoo, verbose=verbose)
    @test runone(h1_ringdown, verbose=verbose, atol=1e-2)
    @test testgauss(;verbose=verbose)
    
end
