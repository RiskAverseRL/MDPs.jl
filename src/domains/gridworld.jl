module GridWorld

import ...TabMDP, ...transition, ...state_count, ...action_count
import ...actions, ...states

# TODO: Add docs
# TODO: Add tests
"""
Models values of demand in `values` and probabilities in `probabilities`.
for Gersi
"""

@enum Action begin
    UP
    DOWN
    LEFT
    RIGHT
end

"""
Parameters that define a GridWorld problem

- `rewards_s`: A vector of rewards for each state
- `max_side_length`: An integer that represents the maximum side length of the grid
- `wind`: A float that represents the wind
"""
struct Parameters
    rewards_s::Vector{Float64}
    max_side_length::Int
    wind::Float64
end


# ----------------------------------------------------------------
# Definition of MDP models and functions
# ----------------------------------------------------------------

"""
An inventory MDP problem simulator

The states and actions are 1-based integers.
"""
struct Model <: TabMDP
    params::Parameters
end

function transition(model::Model, state::Int, action::Int)
    n = model.params.max_side_length
    n_states = state_count(model.params)
    ret = []
    # Wrap the state around the grid
    upstate = ((state - n) + n_states) % n_states # Julia for the love of God please implement a proper modulo function
    downstate = (state + n) % n_states
    leftstate = ((state - 1) + n_states) % n_states
    rightstate = (state + 1) % n_states
    if action == Action.UP
        push!(ret, (upstate, 1.0 - model.params.wind, model.params.rewards_s[upstate]))
        push!(ret, (downstate, model.params.wind / 3, model.params.rewards_s[downstate]))
        push!(ret, (leftstate, model.params.wind / 3, model.params.rewards_s[leftstate]))
        push!(ret, (rightstate, model.params.wind / 3, model.params.rewards_s[rightstate]))
    elseif action == Action.DOWN
        push!(ret, (upstate, model.params.wind / 3, model.params.rewards_s[upstate]))
        push!(ret, (downstate, 1.0 - model.params.wind, model.params.rewards_s[downstate]))
        push!(ret, (leftstate, model.params.wind / 3, model.params.rewards_s[leftstate]))
        push!(ret, (rightstate, model.params.wind / 3, model.params.rewards_s[rightstate]))
    elseif action == Action.LEFT
        push!(ret, (upstate, model.params.wind / 3, model.params.rewards_s[upstate]))
        push!(ret, (downstate, model.params.wind / 3, model.params.rewards_s[downstate]))
        push!(ret, (leftstate, 1.0 - model.params.wind, model.params.rewards_s[leftstate]))
        push!(ret, (rightstate, model.params.wind / 3, model.params.rewards_s[rightstate]))
    elseif action == Action.RIGHT
        push!(ret, (upstate, model.params.wind / 3, model.params.rewards_s[upstate]))
        push!(ret, (downstate, model.params.wind / 3, model.params.rewards_s[downstate]))
        push!(ret, (leftstate, model.params.wind / 3, model.params.rewards_s[leftstate]))
        push!(ret, (rightstate, 1.0 - model.params.wind, model.params.rewards_s[rightstate]))
    end
    return ret
end

state_count(model::Model) = model.params.max_side_length * model.params.max_side_length
action_count(model::Model, state::Int) = 4

states(model::Model) = 1:state_count(model.params)
actions(model::Model, state::Int) = 1:action_count(model.params, state)

end # Module: GridWorld
