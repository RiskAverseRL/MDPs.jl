using MDPs.Domains

@testset "Solve Simple Twostate" begin
    reward = [1, 0]
    epsilon = 0.2
    model = Domains.Simple.TwoStates(reward, epsilon)
    simulate(model, random_π(model), 1, 10000, 500)
    model_g = make_int_mdp(model; docompress=false)
    model_gc = make_int_mdp(model; docompress=true)

    v1 = value_iteration(model, InfiniteH(0.95); ϵ=1e-10)
    v2 = value_iteration(model_g, InfiniteH(0.95); ϵ=1e-10)
    v3 = value_iteration(model_gc, InfiniteH(0.95); ϵ=1e-10)
    v4 = policy_iteration(model_gc, 0.95)

    # Ensure value functions are close
    V = hcat(v1.value, v2.value, v3.value, v4.value)
    @test map(x -> x[2] - x[1], mapslices(extrema, V; dims=2)) |> maximum ≤ 1e-6

    # Ensure policies are identical
    p1 = greedy(model, InfiniteH(0.95), v1.value)
    p2 = greedy(model_g, InfiniteH(0.95), v2.value)
    p3 = greedy(model_gc, InfiniteH(0.95), v3.value)
    p4 = v4.policy

    P = hcat(p1, p2, p3, p4)
    @test all(mapslices(allequal, P; dims=2))
end
