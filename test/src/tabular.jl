using CSV
using Arrow
using DataFrames


@testset "Serialize and load an MDP file" begin
    filein  = joinpath(dirname(pathof(MDPs)), "..", "data", "population.arr")
    
    model = load_mdp(Arrow.Table(filein))
    output = save_mdp(DataFrame, model)
    model2 = load_mdp(output)
    output2 = save_mdp(DataFrame, model2)
    @test all(map(all, eachcol(output .≈ output2)))
end
