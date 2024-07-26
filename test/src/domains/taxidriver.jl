using MDPs.Domains

@testset "Solve TaxiDriver.Taxi" begin
    pickup_rates = [.9,.3,.1]
    destination_prob = [0.0 .8 .2; .6 0.0 .4; .1 .9 0.0]
    transition_cost = [0.0 1.5 2.0; 3.0 0.0 1.0; 2.0 1.5 0.0]
    transition_profit = [0.0 8.5 10.0; 9.0 0.0 12.0; 10.0 7.0 0.0]
    num_locs = 3

    model = TaxiDriver.Taxi(pickup_rates, destination_prob,transition_cost, transition_profit,num_locs )

    # Value iteration and policy tests
    discount_factor = 0.95
    v = value_iteration(model, InfiniteH(discount_factor); ϵ=1e-10)
    p = greedy(model, InfiniteH(discount_factor), v.value)
end
