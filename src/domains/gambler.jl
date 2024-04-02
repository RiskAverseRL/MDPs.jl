module Gambler

import ...TabMDP, ...transition, ...state_count, ...action_count

mt(st, prob,rew) =
    (Int(st), Float64(prob), Float64(rew))::Tuple{Int, Float64, Float64}

"""
    Ruin(win, max_capital)

Gambler's ruin. Can decide how much to bet at any point in time. With some
probability `win`, the bet is doubled, and with `1-win` it is lost. The
reward is 1 if it achieves some terminal capital and 0 otherwise.

- Capital = state - 1
- Bet = action - 1 

Available actions are 1, ..., state - 1.

Special states: state=1 is broke and state=max_capital+1 is a terminal winning state.
"""
struct Ruin <: TabMDP
    win :: Float64
    max_capital :: Int

    function Ruin(win::Number, max_capital::Integer)
        zero(win) ≤ win ≤ one(win) || error("win probability must be in [0,1]")
        max_capital ≥ one(max_capital) || error("Max capital must be positive")
        new(win, max_capital)
    end
end

function transition(model::Ruin, state::Int, action::Int)
    1 ≤ state ≤ model.max_capital+1 || error("invalid state")
    1 ≤ action ≤ state || error("invalid action")

    if state == 1
        (mt(1, 1.0, 0.0),)
    elseif state == model.max_capital + 1 # the state is absorbing
        (mt(state, 1.0, 1.0),)
    else
        win_state = min(model.max_capital + 1, (state - 1) + (action - 1) + 1)
        lose_state = max(1, (state - 1) - (action - 1) + 1)
        (mt(win_state, model.win, 0.), mt(lose_state, 1.0 - model.win, 0.))
    end
end

state_count(model::Ruin) = model.max_capital + 1
action_count(model::Ruin, state::Int) = state

end # Gambler
