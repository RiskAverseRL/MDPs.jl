module GridWorld

import ...TabMDP, ...transition, ...state_count, ...action_count
import ...actions, ...states

"""
Models values of demand in `values` and probabilities in `probabilities`.
"""
@enum Action begin
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
end

"""
    Parameters(reward_s, max_side_length, wind)


Parameters that define a GridWorld problem

- `rewards_s`: A vector of rewards for each state
- `max_side_length`: An integer that represents the maximum side length of the grid
- `wind`: A float that represents the wind
"""
struct Parameters
    rewards_s::Vector{Float64}
    max_side_length::Int
    wind::Float64

    function Parameters(rewards_s, max_side_length, wind)
        length(rewards_s) == max_side_length * max_side_length ||
            error("Rewards must have the same length as the number of states.")
        wind ≥ 0.0 || error("Wind must be non-negative.")
        wind ≤ 1.0 || error("Wind must be less than or equal to 1.")

        new(rewards_s, max_side_length, wind)
    end
end


# ----------------------------------------------------------------
# Definition of MDP models and functions
# ----------------------------------------------------------------

"""
A GridWorld MDP problem simulator

The states and actions are 1-based integers.
"""
struct Model <: TabMDP
    params::Parameters
end

function transition(model::Model, state::Int, action::Int)
    n = model.params.max_side_length
    n_states = state_count(model.params)
    compl_wind = (1.0 - model.params.wind)
    remaining_wind = model.params.wind / 3
    ret = []
    # Wrap the state around the grid 1-based indexing
    upstate = state - n <= 0 ? state + n_states - n : state - n
    downstate = (state + n) > n_states ? state - n_states + n : state + n
    leftstate = state % n == 1 ? state + (n - 1) : state - 1
    rightstate = state % n == 0 ? state - (n - 1) : state + 1
    if action == Int(UP)
        push!(ret, (upstate, compl_wind, model.params.rewards_s[upstate]))
        push!(ret, (downstate, remaining_wind, model.params.rewards_s[downstate]))
        push!(ret, (leftstate, remaining_wind, model.params.rewards_s[leftstate]))
        push!(ret, (rightstate, remaining_wind, model.params.rewards_s[rightstate]))
    elseif action == Int(DOWN)
        push!(ret, (downstate, compl_wind, model.params.rewards_s[downstate]))
        push!(ret, (upstate, remaining_wind, model.params.rewards_s[upstate]))
        push!(ret, (leftstate, remaining_wind, model.params.rewards_s[leftstate]))
        push!(ret, (rightstate, remaining_wind, model.params.rewards_s[rightstate]))
    elseif action == Int(LEFT)
        push!(ret, (leftstate, compl_wind, model.params.rewards_s[leftstate]))
        push!(ret, (upstate, remaining_wind, model.params.rewards_s[upstate]))
        push!(ret, (downstate, remaining_wind, model.params.rewards_s[downstate]))
        push!(ret, (rightstate, remaining_wind, model.params.rewards_s[rightstate]))
    elseif action == Int(RIGHT)
        push!(ret, (rightstate, compl_wind, model.params.rewards_s[rightstate]))
        push!(ret, (upstate, remaining_wind, model.params.rewards_s[upstate]))
        push!(ret, (downstate, remaining_wind, model.params.rewards_s[downstate]))
        push!(ret, (leftstate, remaining_wind, model.params.rewards_s[leftstate]))
    else
        throw(ArgumentError("Invalid action " * string(action) * " for GridWorld."))
    end
    return ret
end

state_count(params::Parameters) = params.max_side_length * params.max_side_length
action_count(params::Parameters, state::Int) = 4

state_count(model::Model) = model.params.max_side_length * model.params.max_side_length
action_count(model::Model, state::Int) = 4

states(model::Model) = 1:state_count(model.params)
actions(model::Model, state::Int) = 1:action_count(model.params, state)

end # Module: GridWorld
