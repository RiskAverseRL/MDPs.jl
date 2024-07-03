module TaxiDriver

using MDPs.Domains, Test

export PassengerRequest, Costs, Limits, Parameters, Model, transition, location2state, state2location, state2passenger, state_count, next_location2action, action2next_location, action_count

"""
Struct to define the probabilities of passenger requests at each location.
"""
struct PassengerRequest
    locations :: Vector{Int}
    probabilities :: Vector{Float64}

    function PassengerRequest(locations, probabilities)
        length(locations) == length(probabilities) ||
            error("Locations and probabilities must have the same length.")
        all(probabilities .≥ 0.0) ||
            error("Passenger request probabilities must be non-negative.")
        sum(probabilities) ≈ 1.0 || error("Passenger request probabilities must sum to 1.")
        new(locations, probabilities)
    end
end

"""
Struct to define the costs and earnings in the taxi driver problem.
"""
struct Costs
    move_cost :: Float64
    ride_earnings :: Float64
end

"""
Struct to define the limits in the taxi driver problem.
"""
struct Limits
    max_locations :: Int
end

"""
Parameters that define a taxi driver problem.
"""
struct Parameters
    passenger_request :: PassengerRequest
    costs :: Costs
    limits :: Limits
end

"""
    transition(params, location, passenger, next_location, passenger_pickup)

Update the location and passenger status.

Starting with a `location`, the taxi moves to `next_location`. If `passenger`
is true, the passenger is dropped off. If not, the taxi may pick up a passenger
based on `passenger_pickup` probability.
"""
function transition(params::Parameters, location::Int, passenger::Bool, next_location::Int, passenger_pickup::Bool)
    location ≥ 1 || error("Invalid location.")
    location ≤ params.limits.max_locations || error("Location over limit.")
    next_location ≥ 1 || error("Invalid next location.")
    next_location ≤ params.limits.max_locations || error("Next location over limit.")

    # Compute the next state
    next_passenger = passenger ? false : passenger_pickup

    # Compute the reward
    reward = passenger ? params.costs.ride_earnings - params.costs.move_cost : -params.costs.move_cost

    (reward = reward, location = next_location, passenger transok
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
p3 = greedy(model_gc, InfiniteH p4 = v4.policy

P = hcat(p1, p2[1:end-1], p3[1:end-1], p4[1:end-1])
@test all(mapslices(allequal, P; dims=2))
end
end # Module: TaxiDriver
