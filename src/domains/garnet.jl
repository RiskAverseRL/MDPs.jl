module Garnet

import ...TabMDP, ...transition, ...state_count, ...action_count, StatsBase, Distributions
# ----------------------------------------------------------------
# A Garnet MDP
# ----------------------------------------------------------------

struct GarnetMDP <: TabMDP
    reward::Vector{Vector{Float64}}
    transition::Vector{Vector{Vector{Float64}}}
    S::Int64
    A::Vector{Int64}

    function GarnetMDP(numStates::Int, numActions::Vector{Int}, nBranch::Float64, minReward::Int, maxReward::Int)
        S = numStates
        A = numActions
        reward = Vector{Vector{Float64}}([])
        transition = Vector{Vector{Vector{Float64}}}([])
        dist = Exponential(1)
        sout = Int(round(nBranch*S))
        for i in 1:numStates
            r = Vector{Float64}([])
            p = Vector{Vector{Float64}}([])
            for j in 1:numActions[i]
                push!(r,rand(minReward:maxReward))
                inds = sample(1:S,sout,replace=false)
                z = rand(dist,sout)
                z /= sum(z)
                pp = zeros(S)
                for (k,l) in enumerate(inds) pp[l] = z[k] end
                push!(p,pp)
            end
            push!(reward,r)
            push!(transition, p)
        end
        new(reward,transition,S,A)
    end
end


function transition(model::GarnetMDP, state::Int, action::Int)
    @assert state in 1:model.S
    @assert action in 1:model.A[state]

    next = []
    for (s,p) in enumerate(model.transition[state][action])
        if p != 0
            push!(next, (s,p,model.reward[state][action]))
        end
    end
    return next
end

state_count(model::GarnetMDP) = model.S
action_count(model::GarnetMDP, s::Int) = model.A[s]

end   
# Module: Garnet