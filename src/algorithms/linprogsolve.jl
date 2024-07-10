using JuMP

"""
Implments the linear programming method of solving an MDP "model" with an infinite horizon and discount factor γ.
The function utilizes the HiGHS optimizer which is free to use.
"""


function linear_program_solve(model::TabMDP, objective::InfiniteH, optimizer)
    lpm = Model(optimizer)
    set_silent(lpm)
    n = state_count(model)
    @variable(lpm, v[1:n])
    @objective(lpm,Min, sum(v[1:n]))
    for s in 1:n
        m = action_count(model,s)
        for a in 1:m
            snext = transition(model,s,a)
            @constraint(lpm, v[s] ≥ sum(sp[2]*(sp[3]+objective.γ*v[sp[1]]) for sp in snext))
        end
    end
    optimize!(lpm)
    return value.(v)
end
