module Garnet

import ...TabMDP, ...transition, ...state_count, ...action_count
import ...actions, ...states

# TODO: are these reasonable or can we replace them?
import StatsBase, Distributions
# ----------------------------------------------------------------
# A Garnet MDP
# ----------------------------------------------------------------

struct GarnetMDP <: TabMDP
    reward::Vector{Vector{Float64}}
    transition::Vector{Vector{Vector{Float64}}}
    S::Int
    A::Vector{Int}

    # TODO: add a constructor that checks for consistency
end

function make_garnet(S::Integer, A::AbstractVector{Int}, nbranch::Number, min_reward::Integer, max_reward::Integer)

    0.0 ≤ nbranch ≤ 1.0 || error("nbranch must be in [0,1]")
    
    reward = Vector{Vector{Float64}}()
    transition = Vector{Vector{Vector{Float64}}}()
    dist = Distributions.Exponential(1)
    sout = Int(round(nbranch*S))
    
    for i in 1:S
        r = Vector{Float64}()
        p = Vector{Vector{Float64}}()
        for j in 1:A[i]
            push!(r, rand(min_reward:max_reward))
            inds = StatsBase.sample(1:S, sout, replace=false)
            z = rand(dist,sout)
            z /= sum(z)
            pp = zeros(S)
            for (k,l) in enumerate(inds) pp[l] = z[k] end
            push!(p,pp)
        end
        push!(reward,r)
        push!(transition, p)
    end
    
    GarnetMDP(reward,transition,S,A)
end

make_garnet(S::Integer, A::Integer, nbranch, min_reward, max_reward) = make_garnet(S, fill(Int(A),S), nbranch, min_reward, max_reward)

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
