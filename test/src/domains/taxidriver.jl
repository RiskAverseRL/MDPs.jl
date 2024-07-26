using MDPs.Domains

@testset "Solve TaxiDriver.Taxi" begin
    pickup_rates = [.9,.3,.1]
    destination_prob = [0.0 .8 .2; .6 0.0 .4; .1 .9 0.0]
    transition_cost = [0.0 1.5 2.0; 3.0 0.0 1.0; 2.0 1.5 0.0]
    transition_profit = [0.0 8.5 10.0; 9.0 0.0 12.0; 10.0 7.0 0.0]
    num_locs = 3

    model = TaxiDriver.Taxi(pickup_rates, destination_prob,transition_cost, transition_profit,num_locs )
    simulate(model, random_π(model), 1, 10000, 500)
    model_g = make_int_mdp(model; docompress=false)
    model_gc = make_int_mdp(model; docompress=true)

    v1 = value_iteration(model, InfiniteH(0.95); ϵ=1e-10)
    v2 = value_iteration(model_g, InfiniteH(0.95); ϵ=1e-10)
    v3 = value_iteration(model_gc, InfiniteH(0.95); ϵ=1e-10)
    v4 = policy_iteration(model_gc, 0.95)

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
end
