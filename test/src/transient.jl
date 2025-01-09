using HiGHS
# TODO: need better tests for transient MDPs

@testset "Transience - all" begin
    opt = HiGHS.Optimizer
    model = Domains.Gambler.RuinTransient(0.5, 20, false) # no noop

    @test anytransient(model, opt)
    @test alltransient(model, opt)
    val = lp_solve(model, TotalReward(), opt)
    # @test val.value[2] ≈ 0.5
    # @test val.policy[2] = 14
end


@testset "Transience - some" begin
    opt = HiGHS.Optimizer
    model = Domains.Gambler.RuinTransient(0.5, 20, true)

    @test anytransient(model, opt)
    @test ~alltransient(model, opt)
    val = lp_solve(model, TotalReward(), opt)
    # @test val.value[2] ≈ 0.5 
    # @test val.policy[2] = 20
end
