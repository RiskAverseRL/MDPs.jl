module garnet

import ...TabMDP, ...transition, ...state_count, ...action_count
# ----------------------------------------------------------------
# A model with a single state
# ----------------------------------------------------------------

struct Garnet <: TabMDP
    reward::Vector{Vector{Float64}}
    transition::Vector{Vector{Vector{Float64}}}
    S::Int64
    A::Vector{Int64}
end


function transition(model::Garnet, state::Int, action::int)
    @assert state in 1:model.S
    @assert action in 1:model.A[state]

    next = []
    for i in 1:transition[state,action,:]

struct TwoStates <: TabMDP
    rewards::Vector{Float64}

    function TwoStates(rewards)
        @assert length(rewards) == 2
        new(rewards)
    end
end

function transition(model::TwoStates, state::Int, action::Int)
    @assert state ∈ (1,2)
    @assert action == 1
    # returns: state, probability, reward
    if state == 1
        ((1::Int, 0.6::Float64, model.rewards[1]::Float64),
         (2::Int, 0.4::Float64, model.rewards[1]::Float64))
    else
        ((1::Int, 0.4::Float64, model.rewards[2]::Float64),
         (2::Int, 0.6::Float64, model.rewards[2]::Float64))
    end
        
end

state_count(model::TwoStates) = 2
action_count(model::TwoStates, s::Int) = 1

end   # Module: Simple