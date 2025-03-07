using HiGHS
# TODO: need better tests for transient MDPs

@testset "Transience - all" begin
    opt = HiGHS.Optimizer
    model = Domains.Gambler.RuinTransient(0.5, 20, false) # no noop

    @test anytransient(model, opt)
    @test alltransient(model, opt) # should be transient
    val = lp_solve(model, TotalReward(), opt)
    # @test val.value[2] ≈ 0.5
    # @test val.policy[2] = 14
    model = Domains.Simple.TwoStates([-0.2, 0.0], 0.1, transient=true)
    @test anytransient(model, opt)
    @test alltransient(model, opt) # should be transient
    reward = [0.1, 0.1, 0.2, -10, -15, 100, 1, 0.5, 0.1]
    max_side_length = 3
    wind = 0.2
    # revolve and transient
    params = GridWorld.Parameters(reward, max_side_length, wind, revolve=false, transient=true)
    model = GridWorld.Model(params)
    @test anytransient(model, opt)
    @test alltransient(model, opt) # should be transient
end


@testset "Transience - some" begin
    opt = HiGHS.Optimizer
    model = Domains.Gambler.RuinTransient(0.5, 20, true) # noop meaning you can stay still, never terminating

    @test anytransient(model, opt)
    @test ~alltransient(model, opt) # should not be transient
    val = lp_solve(model, TotalReward(), opt)
    # @test val.value[2] ≈ 0.5
    # @test val.policy[2] = 20
end
