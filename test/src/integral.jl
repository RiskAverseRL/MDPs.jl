using CSV: File
#using Arrow

@testset "Integral MDP" begin
    filepath = joinpath(dirname(pathof(MDPs)), "..",
                        "data", "riverswim.csv")
    
    model = load_mdp(File(filepath); idoutcome = 1)
    #model = load_mdp(Arrow.Table(filepath); idoutcome = 1)

    sol_t = value_iteration(model, FiniteH(0.9, 30))
    sol_vi = value_iteration(model, InfiniteH(0.9); iterations = 30)
    
    @test all(sol_t.value[1] .≈ sol_vi.value)

    sol_vi_close = value_iteration(model, InfiniteH(0.9); ϵ = 0.01)
    greedy_pol = greedy(model, InfiniteH(0.9), sol_vi_close.value)
    sol_pi = policy_iteration(model, 0.9)

    @test all(abs.(sol_pi.value .- sol_vi_close.value) .≤ 0.1)
    @test all(greedy_pol .== sol_pi.policy)
end
