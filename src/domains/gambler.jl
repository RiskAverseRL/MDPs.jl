module Gambler

import ...TabMDP, ...transition, ...state_count, ...action_count

# create the transition representation for this domain
# (state_to, probability, reward)
mt(st, prob,rew) =
    (Int(st), Float64(prob), Float64(rew))::Tuple{Int, Float64, Float64}


# -------------------------------------------------------------------------------------
# Discounted ruin
# -------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# Transient ruin
# --------------------------------------------------------------------------------------


"""
    RuinTransient(win, max_capital, noop[, win_reward = 1.0, lose_reward = 0.0])

Gambler's ruin; the transient version. Can decide how much to bet at any point in time. With some
probability `win`, the bet is doubled, and with `1-win` it is lost. The reward is `1` if it achieves
some terminal capital and `0` otherwise. State `max_capital+1` is an absorbing win state
in which `1` is received forever.

- Capital = `state - 1`

If `noop = true` then the available actions are `1, ..., capital+1` and bet = `action - 1`. This
allows a bet of 0 which is not a transient policy. 

If `noop = false` then the available actions are `1, ..., capital` and bet = `action `. The MDP is not
transient if `noop = true`, but has some transient policies. When `noop = false`, the MDP is
transient.

Special states: `state=1` is broke and `state=max_capital+1` is maximal capital. Both of the
states are absorbing/terminal.

By default, the reward is `0` when the gambler goes broke and `+1` when it achieves the
target capital. The difference from `Ruin` is that no reward received in the terminal state.
The rewards for overall win and loss can be adjusted by providing `win_reward` and
`lose_reward` optional parameters.
"""
struct RuinTransient <: TabMDP
    win :: Float64
    max_capital :: Int
    noop :: Bool
    win_reward :: Float64
    lose_reward :: Float64

    function RuinTransient(win::Number, max_capital::Integer, noop::Bool;
                           win_reward = 1.0, lose_reward = 0.0)
        zero(win) ≤ win ≤ one(win) || error("Win probability must be in [0,1]")
        max_capital ≥ one(max_capital) || error("Max capital must be positive")
        new(win, max_capital, noop, win_reward, lose_reward)
    end
end

state_count(model::RuinTransient) = model.max_capital + 1

function action_count(model::RuinTransient, state::Int)
    ns = state_count(model)
    @assert state ≥ 1 && state ≤ ns 
    if state == 1 || state == ns 
        1
    else
        capital = state - 1
        model.noop ? capital + 1 : capital
    end
end

function transition(model::RuinTransient, state::Int, action::Int)
    absorbing = state_count(model)  # the "last" state
    
    1 ≤ state ≤ absorbing || error("invalid state: $state")
    1 ≤ action ≤ action_count(model, state) || error("invalid action $action in state $state")

    if state == 1  # broke
        (mt(state, 1.0, 0.0),)
    elseif state == model.max_capital+1   # absorbing terminal state; no reward
        (mt(state, 1.0, 0.0),)
    else
        bet = model.noop ? action - 1 : action
        
        win_state = min(model.max_capital + 1, (state - 1) + bet + 1)
        lose_state = max(1, (state - 1) - bet + 1)

        zero_rew = 1e-8 * rand()
        
        # reward 1.0 if an donly if we achieve the target capital
        win_reward = win_state == absorbing ? model.win_reward : 0.0
        lose_reward = lose_state == 1 ? model.lose_reward : 0.0

        # transition to the absorbing last state
        if lose_state == 1
            lose_state = absorbing
        end

        # the reward is 0 when we lose
        (mt(win_state, model.win, win_reward), mt(lose_state, 1.0 - model.win, lose_reward))
    end
end

end # Gambler
