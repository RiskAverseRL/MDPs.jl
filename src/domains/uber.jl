module TaxiDriver

import ...TabMDP, ...transition, ...state_count, ...action_count
import ...actions, ...states

"""
Struct to define the probabilities of passenger requests at each location.
"""
struct PassengerRequest
    locations :: Vector{Int}
    probabilities :: Vector{Float64}

    function PassengerRequest(locations, probabilities)
        length(locations) == length(probabilities) ||
            error("Locations and probabilities must have the same length")
        all(probabilities .≥ 0.0) ||
            error("Passenger request probabilities must be non-negative.")
        sum(probabilities) ≈ 1.0 || error("Passenger request probabilities must sum to 1")
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

    (reward = reward, location = next_location, passenger = next_passenger)
end

location2state(params::Parameters, location::Int, passenger::Bool)  =
    (location - 1) * 2 + (passenger ? 2 : 1)
state2location(params::Parameters, state::Int) = 
    ((state - 1) ÷ 2) + 1
state2passenger(params::Parameters, state::Int) = 
    ((state - 1) % 2) == 1

state_count(params::Parameters) = params.limits.max_locations * 2

next_location2action(params::Parameters, next_location::Int) = next_location
action2next_location(params::Parameters, action::Int) = action
action_count(params::Parameters, state::Int) = params.limits.max_locations

"""
A taxi driver MDP problem simulator

The states and actions are 1-based integers.
"""
struct Model <: TabMDP
    params :: Parameters
end

function transition(model::Model, state::Int, action::Int)
    location = state2location(model.params, state)
    passenger = state2passenger(model.params, state)
    next_location = action2next_location(model.params, action)

    function make_transition(v, p)
        t = transition(model.params, location, passenger, next_location, v)
        (location2state(model.params, t.location, t.passenger), p, t.reward)
    end

    passenger_requests = zip(model.params.passenger_request.locations, model.params.passenger_request.probabilities)
    (make_transition(passenger_pickup, p) for (passenger_pickup, p) ∈ passenger_requests)
end

state_count(model::Model) = state_count(model.params)
action_count(model::Model, state::Int) = action_count(model.params, state)

states(model::Model) = 1:state_count(model.params)
actions(model::Model, state::Int) = 1:action_count(model.params, state)

end # Module: TaxiDriver
