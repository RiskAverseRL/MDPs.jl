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
- `wind`: A float that represents the wind ∈ [0, 1]
- 'revolve': Whether or not the agennt can wrap around the grid by moving off the edge and appearing on the other side *default True*
- 'transient': Whether or not there is an absorbing state *default False*
"""
struct Parameters
    rewards_s::Vector{Float64}
    max_side_length::Int
    wind::Float64
    revolve::Bool
    transient::Bool

    function Parameters(rewards_s::AbstractVector{<:Real}, max_side_length::Int, wind::Real; revolve::Bool=true, transient::Bool=false)
        length(rewards_s) == max_side_length * max_side_length ||
            error("Rewards must have the same length as the number of states.")
        wind ≥ 0.0 || error("Wind must be non-negative.")
        wind ≤ 1.0 || error("Wind must be less than or equal to 1.")
        if transient
            revolve && error("Cannot have a transient model that also revolves. Set kwarg revolve=false.")
            rewards_s = vcat(rewards_s, [0.0]) # Absorbing state reward is 0
        end
        new(rewards_s, max_side_length, float(wind), revolve, transient)
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
    n_states = n * n
    if state == (n_states + 1) # Absorbing state
        model.params.transient && return [(state, 1.0, 0.0)]
        error("Non-transient model found in absorbing state, file a github issue.")
    end
    compl_wind = (1.0 - model.params.wind)
    remaining_wind = model.params.wind / 3
    # Default you stay in the same state
    upstate = state - n <= 0 ? state : state - n
    downstate = (state + n) > n_states ? state : state + n
    leftstate = state % n == 1 ? state : state - 1
    rightstate = state % n == 0 ? state : state + 1
    if model.params.revolve
        # wrap the state around the grid 1-based indexing
        upstate = state - n <= 0 ? state + n_states - n : state - n
        downstate = (state + n) > n_states ? state - n_states + n : state + n
        leftstate = state % n == 1 ? state + (n - 1) : state - 1
        rightstate = state % n == 0 ? state - (n - 1) : state + 1
    elseif model.params.transient
        # transient model, i.e. going over the edge results in an absorbing state
        upstate = state - n <= 0 ? n_states + 1 : state - n
        downstate = (state + n) > n_states ? n_states + 1 : state + n
        leftstate = state % n == 1 ? n_states + 1 : state - 1
        rightstate = state % n == 0 ? n_states + 1 : state + 1
    end
    ret = []
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

state_count(params::Parameters) = params.transient ? (params.max_side_length * params.max_side_length) + 1 : params.max_side_length * params.max_side_length
action_count(params::Parameters, state::Int) = state == state_count(params) ? 1 : 4 # Absorbing state has a single action

state_count(model::Model) = state_count(model.params)
action_count(model::Model, state::Int) = action_count(model.params, state)

states(model::Model) = 1:state_count(model.params)
actions(model::Model, state::Int) = 1:action_count(model.params, state)

end # Module: GridWorld
