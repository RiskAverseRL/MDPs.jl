module Gambler

import ...TabMDP, ...transition, ...state_count, ...action_count

# create the transition representation for this domain
# (state_to, probability, reward)
mt(st, prob,rew) =
    (Int(st), Float64(prob), Float64(rew))::Tuple{Int, Float64, Float64}


# ------------------------------------------------------------------------------------------------------------------------
# Discounted ruin
# ------------------------------------------------------------------------------------------------------------------------


"""
    Ruin(win, max_capital)

Gambler's ruin; the discounted version. Can decide how much to bet at any point in time. With some
probability `win`, the bet is doubled, and with `1-win` it is lost. The
reward is `1` if it achieves some terminal capital and `0` otherwise. State `max_capital+1`
is an absorbing win state in which `1` is received forever.

- Capital = `state - 1`
- Bet     = `action - 1` 

Available actions are `1`, ..., `state`.

Special states: `state=1` is broke and `state=max_capital+1` is a terminal winning state.
"""
struct Ruin <: TabMDP
    win :: Float64
    max_capital :: Int

    function Ruin(win::Number, max_capital::Integer)
        zero(win) ≤ win ≤ one(win) || error("Win probability must be in [0,1]")
        max_capital ≥ one(max_capital) || error("Max capital must be positive")
        new(win, max_capital)
    end
end

state_count(model::Ruin) = model.max_capital + 1
action_count(model::Ruin, state::Int) = state < model.max_capital + 1 ? state : 1 # only one action in the terminal state

function transition(model::Ruin, state::Int, action::Int)
    1 ≤ state ≤ model.max_capital + 1 || error("invalid state")
    1 ≤ action ≤ action_count(model, state) || error("invalid action")

    if state == 1  # overall loss state
        (mt(1, 1.0, 0.0),)
    elseif state == model.max_capital + 1 # overall win state
        (mt(state, 1.0, 1.0),)
    else
        win_state = min(model.max_capital + 1, (state - 1) + (action - 1) + 1)
        lose_state = max(1, (state - 1) - (action - 1) + 1)
        (mt(win_state, model.win, 0.), mt(lose_state, 1.0 - model.win, 0.))
    end
end


# ------------------------------------------------------------------------------------------------------------------------
# Transient ruin
# ------------------------------------------------------------------------------------------------------------------------


"""
    RuinTransient(win, max_capital)

Gambler's ruin; the transient version. Can decide how much to bet at any point in time. With some
probability `win`, the bet is doubled, and with `1-win` it is lost. The reward is `1` if it achieves
some terminal capital and `0` otherwise. State `max_capital+1` is an absorbing win state
in which `1` is received forever.

- Capital = `state - 1`
- Bet     = `action - 1` 

Available actions are `1`, ..., `state`.

Special states: `state=1` is broke and `state=max_capital+1` is an absorbing state.

The reward is `-1` when the gambler goes broke and `+1` when it achieves the target capital.
"""
struct RuinTransient <: TabMDP
    win :: Float64
    max_capital :: Int

    function RuinTransient(win::Number, max_capital::Integer)
        zero(win) ≤ win ≤ one(win) || error("Win probability must be in [0,1]")
        max_capital ≥ one(max_capital) || error("Max capital must be positive")
        new(win, max_capital)
    end
end

state_count(model::RuinTransient) = model.max_capital + 1
action_count(model::RuinTransient, state::Int) = state < model.max_capital + 1 ? state : 1 # only one action in the terminal state

function transition(model::RuinTransient, state::Int, action::Int)
    absorbing :: Int = model.max_capital + 1
    
    1 ≤ state ≤ absorbing || error("invalid state")
    1 ≤ action ≤ action_count(model, state) || error("invalid action")

    if state == 1  # broke
        (mt(absorbing, 1.0, -1.0),)
    elseif state == absorbing   # absorbing terminal state; no reward
        (mt(state, 1.0, 1.0),)
    else
        win_state = min(model.max_capital + 1, (state - 1) + (action - 1) + 1)
        lose_state = max(1, (state - 1) - (action - 1) + 1)

        # reward 1.0 if an donly if we achieve the target capital
        win_reward = win_state == absorbing ? 1.0 : 0.0

        # the reward is 0 when we lose
        (mt(win_state, model.win, win_reward), mt(lose_state, 1.0 - model.win, 0.))
    end
end

end # Gambler
