module Inventory

import ...TabMDP, ...transition, ...state_count, ...action_count
import ...actions, ...states

"""
Models values of demand in `values` and probabilities in `probabilities`.
"""
struct Demand
    values :: Vector{Int}
    probabilities :: Vector{Float64}

    function Demand(values, probabilities)
        length(values) == length(probabilities) ||
            error("Demand values and probabilities must have the same length")
        all(probabilities .≥ 0.0) ||
            error("Demand probabilities must be non-negative.")
        sum(probabilities) ≈ 1.0 || error("Demand probabilities must sum to 1")
        new(values, probabilities)
    end
end

struct Costs
    purchase :: Float64
    delivery :: Float64
    holding :: Float64
    backlog :: Float64
end

struct Limits
    max_inventory :: Int
    max_backlog :: Int
    max_order :: Int
end

"""
Parameters that define an inventory problem
"""
struct Parameters
    demand :: Demand
    costs :: Costs
    sale_price :: Float64
    limits :: Limits
end

"""
    transition(params, stock, order, demand)

Update the inventory value and compute the profit.

Starting with a `stock` number of items, then `order` of items arrive,
after `demand` of items are sold. Sale price is collected even if it is backlogged
(not beyond backlog level). Negative stock means backlog.

Stocking costs are asessed after all the orders are fulfilled. 

Causes an error when the `order` is too large, but no error when the demand cannot be
satisfied or backlogged.
"""
function transition(params::Parameters, stock::Int, order::Int, demand::Int)
    stock ≥ -params.limits.max_backlog || error("Stock below max backlog.")
    stock ≤ params.limits.max_inventory || error("Stock over limit.")
    order ≥ 0 || error("Negative order.")
    order ≤ params.limits.max_order || error("Order over order limit.")
    # Make sure to adjust the maximum possible order based on available
    # inventory storage because the order arrives before any of the demand
    order + stock ≤ params.limits.max_inventory  ||
        error("Order over inventory limit")
    demand ≥ 0 || error("Nagtive demand.")

    # Compute the next inventory level.inventory
    next_stock = max(order + stock - demand, -params.limits.max_backlog)
    @assert next_stock ≤ params.limits.max_inventory
    # Back-calculate how many items were sold
    sold_amount = stock - next_stock + order
    # Compute the obtained revenue
    revenue = sold_amount * params.sale_price
    @assert next_stock ≥ -params.limits.max_backlog
    # Compute the expense of purchase, holding cost, and backlog cost
    expense = order * params.costs.purchase +
                (order > 0 ? params.costs.delivery : 0.0) +
                params.costs.holding * max(next_stock, 0) +
                params.costs.backlog * -min(next_stock, 0)
    (reward = revenue - expense, stock = next_stock)
end

stock2state(params::Parameters, stock::Int)  =
    stock + params.limits.max_backlog + 1:: Int
state2stock(params::Parameters, state::Int) = 
    state - 1 - params.limits.max_backlog :: Int
state_count(params::Parameters) = 1 + params.limits.max_inventory +
    params.limits.max_backlog :: Int

order2action(params::Parameters, order::Int) = order + 1 :: Int
action2order(params::Parameters, action::Int) = action - 1 :: Int
action_count(params::Parameters, state::Int) =
    min(params.limits.max_inventory - state2stock(params, state),
        params.limits.max_order) + 1 :: Int


# ----------------------------------------------------------------
# Definition of MDP models and functions
# ----------------------------------------------------------------

"""
An inventory MDP problem simulator

The states and actions are 1-based integers.
"""
struct Model <: TabMDP
    params :: Parameters
end

function transition(model::Model, state::Int, action::Int)
    stock = state2stock(model.params, state)
    order = action2order(model.params, action)

    function make_transition(v, p) 
        t = transition(model.params, stock, order, v)
        (stock2state(model.params, t.stock), p, t.reward)
    end
        
    demands = zip(model.params.demand.values, model.params.demand.probabilities)
    (make_transition(v,p) for (v,p) ∈ demands)
end

state_count(model::Model) = state_count(model.params)
action_count(model::Model, state::Int) = action_count(model.params, state)

states(model::Model) = 1:state_count(model.params)
actions(model::Model, state::Int) = 1:action_count(model.params, state)

end # Module: Inventory
