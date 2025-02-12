using MDPs
using DataFrames

#include("domains/make_domains.jl")

function solve_domain(probname, prob)

    episodes::Int = 10000

    # evaluation helper variables
    rweights::Vector{Float64} = prob.γ .^ (0:prob.horizon-1)     # reward weights
    edist::Vector{Float64} = ones(episodes) / episodes # distribution over episodes

    results::DataFrame = DataFrame()

    # Risk neutral solution
    #println("Risk neutral infinite ...")
    #time = @elapsed v = value_iteration(model, γ; ϵ = 0.1)
    #π = greedy(model, γ, v.value)
    #report_disc!(results, "Neutral, inf", π, v, time)

    # Risk-neutral finite
    vp = value_iteration(prob.model, FiniteH(prob.γ, prob.horizon))
    v = vp.value
    π = vp.policy

    # confirm using simulation
    roundresult(x) = round(x; sigdigits=3)

    H = simulate(prob.model, π, prob.initstate, prob.horizon, episodes)
    returns = rweights' * H.rewards |> vec
    rmean = sum(returns) / length(returns)

    @test rmean ≈ v[1][prob.initstate] rtol = 0.05
    #println(rmean, "   <===>    ", vp.value[1][prob.initstate])
    #println(isapprox(rmean, vp.value[1][prob.initstate], rtol = 0.05))
end

@testset "Solve benchmark domains" begin
    # general parameters

    domains::Dict{String,Problem} = make_domains()

    for (dname, domain) ∈ domains
        solve_domain(dname, domain)
    end
end
