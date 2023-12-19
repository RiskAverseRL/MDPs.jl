"""
    make_value(model, objective)

Creates an *undefined* policy and value function for the
`model` and `objective`.

See Also
--------
`value_iteration!`
"""
function make_value(model::TabMDP, objective::Markov)
    n::Integer = state_count(model)
    T::Integer = horizon(objective) 

    v = Vector{Vector{Float64}}(undef, horizon(objective)+1)
    π = Vector{Vector{Int}}(undef, horizon(objective))

    v[T+1] = Vector{Float64}(undef, n)

    for t ∈ T:-1:1
        # initialize vectors
        v[t] = Vector{Float64}(undef, n)
        π[t] = Vector{Int}(undef, n)
    end
    (policy = π, value = v)
end
