module TaxiDriver

import ...TabMDP, ...transition, ...state_count, ...action_count

"""
    Taxi(locs, pickup_loc, dropoff_loc, mv_cost, pickup_rew, dropoff_rew)

A Taxi MDP where the taxi can either stay in the current location or move to a new location.

- locs: Array of possible locations.
- pickup_loc: Location where the passenger is picked up.
- dropoff_loc: Location where the passenger is dropped off.
- mv_cost: Cost of moving to a new location.
- pickup_rew: Reward for picking up a passenger.
- dropoff_rew: Reward for dropping off a passenger.

Available actions are 1 (Stay) and 2 (MoveTo).
"""
struct Taxi <: TabMDP
    locs :: Vector{Int}
    pickup_loc :: Int
    dropoff_loc :: Int
    mv_cost :: Float64
    pickup_rew :: Float64
    dropoff_rew :: Float64

    function Taxi(locs::Vector{Int}, pickup_loc::Int, dropoff_loc::Int, mv_cost::Number, pickup_rew::Number, dropoff_rew::Number)
        new(locs, pickup_loc, dropoff_loc, mv_cost, pickup_rew, dropoff_rew)
    end
end

struct TaxiState
    location::Int
    has_passenger::Bool
end

const Stay = 1
const MoveTo = 2

function transition(model::Taxi, state::TaxiState, action::Int)
    if action == Stay
        return [(state.location, 1.0, 0.0)]
    elseif action == MoveTo
        new_location = (state.location == model.pickup_loc && !state.has_passenger) ? model.dropoff_loc : model.pickup_loc
        new_state = TaxiState(new_location, !state.has_passenger)
        
        if new_state.location == model.pickup_loc && !new_state.has_passenger
            return [(new_state.location, 1.0, model.pickup_rew - model.mv_cost)]
        elseif new_state.location == model.dropoff_loc && new_state.has_passenger
            return [(new_state.location, 1.0, model.dropoff_rew - model.mv_cost)]
        else
            return [(new_state.location, 1.0, -model.mv_cost)]
        end
    else
        error("Invalid action")
    end
end

function state_count(model::Taxi)
    return length(model.locs) * 2
end

function action_count(model::Taxi, state::TaxiState)
    return 2
end

end # TaxiDriver
