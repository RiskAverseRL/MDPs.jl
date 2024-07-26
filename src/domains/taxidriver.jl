module TaxiDriver

import ...TabMDP, ...transition, ...state_count, ...action_count
using LinearAlgebra
"""
    Taxi(pickup_rates, destination_prob,transition_cost,transition_profit,num_locs )

A Taxi MDP where the taxidriver can either stay in the current location with a probability of picking a passenger or 
move to a new location with a better probability of pickup thereby incurring some cost for such a decision. However, the objective is to maximize profit.
Here, there are 2n locations, the first n states (1,...,n) are locations  without passenger, while the states n+1 to 2n are locatons 1 to n witha passenger.

"""

#--------------------------------------------------------
# Definition of MDP Parameters
#--------------------------------------------------------

"""
- pickup_rates: the vector of probability of picking up a passenger in different locations, ∈ [0,1]^n.
- destination_prob: the matrix of probabilities of a different destinations after pickup, 0 if the destination is the current state, > 0 otherwise but sums up to 1.
- transition_cost: the matrix of cost of movement between locations, ≥ 0.
- transition_profit: the matrix of reward earned after a successful job, ≥ 0.
- num_locs: the number of states without passenger.
"""


#--------------------------------------------------------
# Definition of MDP models and functions
#--------------------------------------------------------

struct Taxi <: TabMDP
    pickup_rates::Vector{Number}
    destination_prob::Matrix{Number}
    transition_cost::Matrix{Number}
    transition_profit::Matrix{Number}
    num_locs ::Int
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

state_count(model::Taxi) = model.num_locs * 2

action_count(model::Taxi, state::Int64) = (state ≤ model.num_locs) ? model.num_locs : 1

end # TaxiDriver
