using Main.MDPs

@testset "Solve Garnet" begin

    g = Domains.Garnet.GarnetMDP([[1,1],[2,0]],[[[1,0],[0,1]],[[0,1],[1,0]]],2,[2,2])
    """
    simulate(g, random_π(g), 1, 10000, 500)
    g1 = make_int_mdp(g; docompress=false)
    g2 = make_int_mdp(g; docompress=true)

    v1 = value_iteration(g, InfiniteH(0.99); ϵ=1e-7)
    v2 = value_iteration(g1, InfiniteH(0.99); ϵ=1e-7)
    v3 = value_iteration(g2, InfiniteH(0.99); ϵ=1e-7)
    v4 = policy_iteration(g2, 0.95)

    # Ensure value functions are close
    V = hcat(v1.value, v2.value[1:end-1], v3.value[1:end-1], v4.value[1:end-1])
    @test map(x -> x[2] - x[1], mapslices(extrema, V; dims=2)) |> maximum ≤ 1e-6

    # Ensure policies are identical
    p1 = greedy(model, InfiniteH(0.95), v1.value)
    p2 = greedy(model_g, InfiniteH(0.95), v2.value)
    p3 = greedy(model_gc, InfiniteH(0.95), v3.value)
    p4 = v4.policy

    P = hcat(p1, p2[1:end-1], p3[1:end-1], p4[1:end-1])
    @test all(mapslices(allequal, P; dims=2))
    """
end
