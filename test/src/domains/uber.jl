using MDPs
using Test

@testset "Solve TaxiDriver" begin
    # Define the test parameters
    passenger_request = PassengerRequest([true, false], [0.5, 0.5])
    costs = Costs(1.0, 10.0)
    limits = Limits(5)

    params = Parameters(passenger_request, costs, limits)

    # Initialize flags for tests
    locationok = true
    passengerok = true
    transok = true

    for s in 1:TaxiDriver.state_count(params)
        location = TaxiDriver.state2location(params, s)
        passenger = TaxiDriver.state2passenger(params, s)
        locationok &= (TaxiDriver.location2state(params, location, passenger) == s)
        for a in 1:TaxiDriver.action_count(params, s)
            next_location = TaxiDriver.action2next_location(params, a)
            transok &= (TaxiDriver.transition(params, location, passenger, next_location, true).location == next_location)
            transok &= (TaxiDriver.transition(params, location, passenger, next_location, false).location == next_location)
        end
    end

    @test locationok
    @test passengerok
    @test transok

    model = TaxiDriver.Model(params)
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
