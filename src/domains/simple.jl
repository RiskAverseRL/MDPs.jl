module Simple

import ...TabMDP, ...transition, ...state_count, ...action_count
# ----------------------------------------------------------------
# A model with a single state
# ----------------------------------------------------------------

struct OneState <: TabMDP
    reward::Float64
end

function transition(model::OneState, state::Int, action::Int)
    @assert action == 1
    @assert state == 1
    # returns: state, probability, reward
    ((1::Int, 1.0::Float64, model.reward::Float64),)
end

state_count(model::OneState) = 1
action_count(model::OneState, s::Int) = 1


struct OneStatePlusMinus <: TabMDP
    reward::Float64
end

function transition(model::OneStatePlusMinus, state::Int, action::Int)
    @assert action == 1
    @assert state == 1
    # returns: state, probability, reward
    ((1::Int, 0.8::Float64, model.reward::Float64), (1::Int, 0.2::Float64, -model.reward::Float64))
end

state_count(model::OneStatePlusMinus) = 1
action_count(model::OneStatePlusMinus, s::Int) = 1

# ----------------------------------------------------------------
# A model with two states
# States: 1, 2
# Actions: 1
# Episilon: Probability of transitioning to the other state
# ----------------------------------------------------------------
struct TwoStates <: TabMDP
    rewards::Vector{Float64}
    epsilon::Float64

    function TwoStates(rewards, epsilon=0.4)
        @assert length(rewards) == 2
        @assert 0 <= epsilon <= 1 "epsilon must be in [0, 1]"
        new(rewards, epsilon)
    end
end

function transition(model::TwoStates, state::Int, action::Int)
    @assert state ∈ (1, 2)
    @assert action == 1
    # returns: state, probability, reward
    if state == 1
        ((1::Int, 1.0 - model.epsilon::Float64, model.rewards[1]::Float64), (2::Int, model.epsilon::Float64, model.rewards[1]::Float64))
    else
        ((1::Int, model.epsilon::Float64, model.rewards[2]::Float64), (2::Int, 1.0 - model.epsilon::Float64, model.rewards[2]::Float64))
    end

end

state_count(model::TwoStates) = 2
action_count(model::TwoStates, s::Int) = 1

end   # Module: Simple
