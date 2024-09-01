using MDPs.Domains
import HiGHS, JuMP

@testset "Solve Garnet" begin

    g = Garnet.GarnetMDP([[1,1],[2,0]],[[[1,0],[0,1]],[[0,1],[1,0]]],2,[2,2])
    simulate(g, random_π(g), 1, 10000, 500)
    g1 = make_int_mdp(g; docompress=false)
    g2 = make_int_mdp(g; docompress=true)

    v1 = value_iteration(g, InfiniteH(0.95); ϵ=1e-10)
    v2 = value_iteration(g1, InfiniteH(0.95); ϵ=1e-10)
    v3 = value_iteration(g2, InfiniteH(0.95); ϵ=1e-10)
    v4 = policy_iteration(g2, 0.95)
    v5 = lp_solve(g, .95, JuMP.Model(HiGHS.Optimizer))

    # Ensure value functions are close
    V = hcat(v1.value, v2.value, v3.value, v4.value, v5.value)
    @test map(x -> x[2] - x[1], mapslices(extrema, V; dims=2)) |> maximum ≤ 1e-6

    # Ensure policies are identical
    p1 = greedy(g, InfiniteH(0.95), v1.value)
    p2 = greedy(g1, InfiniteH(0.95), v2.value)
    p3 = greedy(g2, InfiniteH(0.95), v3.value)
    p4 = v4.policy
    p5 = v5.policy

    P = hcat(p1, p2, p3, p4)
    @test all(mapslices(allequal, P; dims=2))
end
