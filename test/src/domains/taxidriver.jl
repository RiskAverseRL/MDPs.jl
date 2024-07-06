using MDPs.Domains
using Test

@testset "Solve TaxiDriver" begin
    # Define the test parameters
    locs = [1, 2]
    pickup_loc = 1
    dropoff_loc = 2
    mv_cost = 1.0
    pickup_rew = 10.0
    dropoff_rew = 20.0

    #params = TaxiDriver.Parameters(locs, pickup_loc, dropoff_loc, mv_cost, pickup_rew, dropoff_rew)


    model = TaxiDriver.Taxi(locs, pickup_loc, dropoff_loc, mv_cost, pickup_rew, dropoff_rew)
    state = TaxiDriver.TaxiState(1, false)

    Stay = 1
    MoveTo = 2

    s_c = state_count(model)
    a_c = action_count(model, state)
    result_stay = transition(model, state, Stay)

    state_pickup = TaxiDriver.TaxiState(1, false)
    result_move_pickup = transition(model, state, MoveTo)

    # Value iteration and policy tests
    discount_factor = 0.95
    v = value_iteration(model, InfiniteH(discount_factor); ϵ=1e-10)
    p = greedy(model, InfiniteH(discount_factor), v.value)
end
