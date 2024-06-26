using MDPs.Domains

@testset "Solve Gridworld" begin
    reward = [0.1, 0.1, 0.2, -10, -15, 100, 1, 0.5]
    max_side_length = 3
    wind = 0.2
    params = GridWorld.Parameters(reward, max_side_length, wind)

    # Initialize flags for tests
    stateok = true
    actionok = true
    transitionok = true

    for s in 1:GridWorld.state_count(params)
        state = GridWorld.state2state(params, s)  # Assuming state2state function exists
        stateok &= (GridWorld.state2state(params, state) == s)
        for a in 1:GridWorld.action_count(params, s)
            action = GridWorld.action2action(params, a)  # Assuming action2action function exists
            actionok &= (GridWorld.action2action(params, action) == a)
            transitionok &= (GridWorld.transition(params, state, action, 0).state == state + action)  # Adjust transition logic as per the actual implementation
        end
    end

    @test stateok
    @test actionok
    @test transitionok

    model = GridWorld.Model(params)
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
