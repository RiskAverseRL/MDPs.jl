using MDPs.Domains
import HiGHS, JuMP

@testset "Solve Inventory" begin

    demand = Inventory.Demand([1,2,3,4,5,10,3,2],
                            [0.1,0.3,0.1,0.1,0.1,0.1,0.0,0.2])
    costs = Inventory.Costs(5.,20.,1.,0.5)
    limits = Inventory.Limits(50, 10, 30)
    params = Inventory.Parameters(demand, costs, 10, limits)

    # do not want to have a million "fake" tests
    stockok = true
    orderok = true
    transok = true
    for s in 1:Inventory.state_count(params)
        stock = Inventory.state2stock(params, s)
        stockok &= (Inventory.stock2state(params, stock) == s)
        for a in 1:Inventory.action_count(params, s)
            order = Inventory.action2order(params, a)
            orderok &= (Inventory.order2action(params, order) == a)
            transok &= (Inventory.transition(params, stock, order, 0).stock
                        == stock + order)
        end
    end
    @test stockok
    @test orderok
    @test transok

    model = Inventory.Model(params)
    simulate(model, random_π(model), 1, 10000, 500)
    model_g = make_int_mdp(model; docompress = false);
    model_gc = make_int_mdp(model; docompress = true);

    v1 = value_iteration(model, InfiniteH(0.95); ϵ = 1e-10)
    v2 = value_iteration(model_g, InfiniteH(0.95); ϵ = 1e-10)
    v3 = value_iteration(model_gc, InfiniteH(0.95); ϵ = 1e-10)
    v4 = policy_iteration(model_gc, 0.95)
    v5 = lp_solve(model, .95, JuMP.Model(HiGHS.Optimizer))

    # note that the IntMDP does not have terminal states,
    # so the last action will not be -1

    #make sure value functions are close
    V = hcat(v1.value, v2.value[1:(end-1)], v3.value[1:(end-1)], v4.value[1:(end-1)], v5.value)
    @test map(x->x[2] - x[1], mapslices(extrema, V; dims = 2)) |> maximum ≤ 1e-6

    # make sure policies are identical
    p1 = greedy(model, InfiniteH(0.95),  v1.value)
    p2 = greedy(model_g, InfiniteH(0.95),  v2.value)
    p3 = greedy(model_gc, InfiniteH(0.95), v3.value)
    p4 = v4.policy
    p5 = v5.policy

    P = hcat(p1, p2[1:(end-1)], p3[1:(end-1)], p4[1:(end-1)])
    @test all(mapslices(allequal, P; dims = 2))
end
