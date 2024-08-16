using MDPs
using MDPs.Domains


@test "basic simulation" begin

    
    model = Domains.Gambler.Ruin(0.7, 10)
    initstate = 1
    horizon = 100
    episodes = 30

    π = ones(Int, state_count(model)) * 2
    π[1] = 1
    
    # simulate a single initial state
    inistate = 3
    H = simulate(model, π, inistate, horizon, episodes)
    @test all(H.states[1,:] .== 3)

    # simulate a distribution
    initial = ones(state_count(model))
    initial[end] = 0
    initial /= sum(initial)
    H = simulate(model, π, initial, horizon, episodes)
    @test length(unique(H.states[1,:])) ≥ 3
end
