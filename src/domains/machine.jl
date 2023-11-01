module Machine

import ...TabMDP, ...transition, ...state_count, ...action_count

mt(st, prob,rew) =
    (Int(st), Float64(prob), Float64(rew))::Tuple{Int, Float64, Float64}

"""
Standard machine replacement simulator. See Figure 3 in Delage 2009
for details.

States are:
1: repair 1
2: repair 2
3 - 10: utility state

Actions:
1: Do nothing
2: Repair
"""
struct Replacement <: TabMDP
# This in intentionally left empty
end

function transition(model::Replacement, state::Int, action::Int)
    # returns: state, probability, reward
    if action == 1 # do nothing
        if state == 1
            (mt(1,0.2,-2.), mt(3,0.8,0.))
        elseif state == 2
            (mt(2, 1.0,-10.),)
        elseif state ≤ 8
            (mt(state, 0.2, 0.), mt(state+1, 0.8, 0.))
        elseif state == 9
            (mt(state, 0.2, 0.), mt(state+1, 0.8, -20.))
        elseif state == 10
            (mt(state, 1, -20.),)
        else
            error("invalid state index")
        end
    elseif action == 2 # repair
        if state == 1
            (mt(1,1.0,-2.), ) 
        elseif state == 2
            (mt(1, 0.6, -2.), mt(2,0.4,-10.))
        elseif state ≤ 8
            (mt(state+1,0.3,0.), mt(1,0.6,0.), mt(2,0.1,0.))
        elseif state == 9
            (mt(state+1,0.3,-20.), mt(1,0.6,0.), mt(2,0.1,0.))
        elseif state == 10
            (mt(state,0.3,-20.), mt(1,0.6,-2.), mt(2,0.1,-10.))
        else
            error("invalid state index")
        end
    else
        error("invalid action")
    end
            
end

state_count(model::Replacement) = 10
action_count(model::Replacement, state::Int) = 2

end # Module: Machine replacement
