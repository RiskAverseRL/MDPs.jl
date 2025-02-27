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

# TODO add xihongs two state transient test
