module TaxiDriver

import ...TabMDP, ...transition, ...state_count, ...action_count
using LinearAlgebra
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
    pickup_rates::Vector{Number}
    destination_prob::Matrix{Number}
    transition_cost::Matrix{Number}
    transition_profit::Matrix{Number}
    num_locs ::Int
    #pickup_loc :: Int
    #dropoff_loc :: Int
    #mv_cost :: Float64
    #pickup_rew :: Float64
    #dropoff_rew :: Float64

   # function Taxi(locs::Vector{Int}, pickup_loc::Int, dropoff_loc::Int, mv_cost::Number, pickup_rew::Number, dropoff_rew::Number)
    #    new(locs, pickup_loc, dropoff_loc, mv_cost, pickup_rew, dropoff_rew)
    #end
end
function transition(model::Taxi, state::Int, action::Int)
    if state ≤ model.num_locs 
        return [(action, 1-model.pickup_rates[action], -model.transition_cost[state, action]),(model.num_locs+action, model.pickup_rates[action], -model.transition_cost[state, action])]
    else
        out = []
        current_loc = state-model.num_locs
        for s in 1:current_loc-1
            push!(out, (s, model.destination_prob[current_loc,s], model.transition_profit[current_loc,s]))
        end
        for s in current_loc+1:model.num_locs
            push!(out, (s, model.destination_prob[current_loc,s], model.transition_profit[current_loc,s])) 
        end
        return out
    end
    
end

# function transition(model::Taxi, state::Int, action::Int)
#     n = state_count(model)//2
#     if action == Stay
#         return [(state%n, 1.0, 0.0)]
#     elseif action == MoveTo
#         new_location = (state.location == model.pickup_loc && !state.has_passenger) ? model.dropoff_loc : model.pickup_loc
#         new_state = TaxiState(new_location, !state.has_passenger)
        
#         if new_state.location == model.pickup_loc && !new_state.has_passenger
#             return [(new_state.location, 1.0, model.pickup_rew - model.mv_cost)]
#         elseif new_state.location == model.dropoff_loc && new_state.has_passenger
#             return [(new_state.location, 1.0, model.dropoff_rew - model.mv_cost)]
#         else
#             return [(new_state.location, 1.0, -model.mv_cost)]
#         end
#     else
#         error("Invalid action")
#     end
# end

state_count(model::Taxi) = model.num_locs * 2

action_count(model::Taxi, state::Int64) = (state ≤ model.num_locs) ? model.num_locs : 1

end # TaxiDriver
