module GridWorld

import ...TabMDP, ...transition, ...state_count, ...action_count
import ...actions, ...states

# TODO: Add docs
# TODO: Add tests
"""
Models values of demand in `values` and probabilities in `probabilities`.
"""

@enum Action begin
    UP
    DOWN
    LEFT
    RIGHT
end

struct Rewards
    rewards_s::Vector{Float64}
end

struct Limits
    max_side_length::Int
end

"""
Parameters that define a GridWorld problem
"""
struct Parameters
    costs::Rewards
    limits::Limits
    actions::Action
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
    stock = state2stock(model.params, state)
    order = action2order(model.params, action)

    function make_transition(v, p)
        t = transition(model.params, stock, order, v)
        (stock2state(model.params, t.stock), p, t.reward)
    end

    demands = zip(model.params.demand.values, model.params.demand.probabilities)
    (make_transition(v, p) for (v, p) ∈ demands)
end

state_count(model::Model) = model.params.limits.max_side_length * model.params.limits.max_side_length
action_count(model::Model) = length(model.params.actions)

states(model::Model) = 1:state_count(model.params)
actions(model::Model) = 1:action_count(model.params)

end # Module: GridWorld
