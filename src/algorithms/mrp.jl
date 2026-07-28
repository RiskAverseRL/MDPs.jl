
# ----------------------------------------------------------------
# Markov reward process and Markov chain
# ----------------------------------------------------------------
using Graphs

"""
    mrp!(P_π, r_π, model, π)

Save the transition matrix `P_π` and reward vector `r_π` for the
MDP `model` and policy `π`. Also supports terminal states.

Does not support duplicate entries in transition probabilities.
"""
function mrp!(P_π::AbstractMatrix{<:Real}, r_π::AbstractVector{<:Real},
    model::TabMDP, π::AbstractVector{<:Integer})
    S = state_count(model)
    fill!(P_π, 0.)
    fill!(r_π, 0.)
    for s ∈ 1:S
        for (sn, p, r) ∈ transition(model, s, π[s])
            # P_π[s,sn] ≈ 0. || error("duplicated transition entries (s1->s2, s1->s2) not allowed")
            P_π[s, sn] += p
            r_π[s] += p * r
        end
    end
end

"""
    mrp(model, π)

Compute the transition matrix `P_π` and reward vector `r_π` for the
MDP `model` and policy `π`. See mrp! for more details.
"""
function mrp(model::TabMDP, π::AbstractVector{<:Integer})
    S = state_count(model)
    P_π = Matrix{Float64}(undef, S, S)
    r_π = Vector{Float64}(undef, S)
    mrp!(P_π, r_π, model, π)
    (P_π, r_π)
end

"""
    mrp_sparse(model, π)

Compute a sparse transition matrix `P_π` and reward vector `r_π` for the
MDP `model` and policy `π`.

This function does not support duplicate entries in transition probabilities.
"""
function mrp_sparse(model::TabMDP, π::AbstractVector{Int}, ignore_duplicates::Bool=false)
    S = state_count(model)
    r_π = zeros(S)

    g = SimpleDiGraph(S)
    rows = Vector{Int}(undef, 0)
    columns = Vector{Int}(undef, 0)
    probabilities = Vector{Float64}(undef, 0)
    for s ∈ 1:S
        for (sn, p, r) ∈ transition(model, s, π[s])
            append!(rows, s)
            append!(columns, sn)
            append!(probabilities, p)
            add_edge!(g, s, sn)
            r_π[s] += p * r
        end
    end
    if ignore_duplicates
        P_π = sparse(rows, columns, probabilities, S, S)
    else
        P_π = sparse(rows, columns, probabilities, S, S, (i, j) ->
            error("Duplicate transition entries (s1->s2, s1->s2) are unsupported"))
    end
    (P_π, r_π, g)
end

"""
     decompose_chain(P)

Uses a Markov chain decomposition to turn the matrix `P` into a set
of transient states, and then recurrent groups. This decomposition is
useful when constructing the stationary matrix of a Markov chain.

The decomposition preserves the structure of matrix and does not
renumber any states.

# Return

A named tuple with the following components
  - `rec` is the list of recurrent components, each a list of states
  - `trans` is the list of transient states
"""
function decompose_chain(P::Matrix{<:Real})
    S = size(P)[1]
    S == size(P)[2] || error("Only square P is supported")

    g = SimpleDiGraph(S)
    for s₁ ∈ 1:S, s₂ ∈ 1:S
        P[s₁, s₂] > 0 && add_edge!(g, s₁, s₂)
    end

    rec = attracting_components(g)
    trans_b = trues(S)
    @inbounds foreach(i -> trans_b[i] = false, Iterators.flatten(rec))
    trans = findall(trans_b)
    (rec=rec, trans=trans)
end


"""
    stationary_matrix(P)

Compute the stationary matrix for a Markov chain with transition matrix `P`.

# Return
  - Stationary matrix `Pstar` such that: `P * Pstar = Pstar` and `Pstar * P = Pstar`
"""
function stationary_matrix(P::Matrix{<:Real})
    chain = decompose_chain(P)
    all(P .≥ zero(eltype(P))) || error("Transition probabilities must be non-negative.")
    all(sum.(eachrow(P)) .≈ 1) || error("Probabilities must sum to 1.")

    S = size(P)[1] # decompose_chain fails is the matrix is not square

    Pstar = zeros(S, S)
    # handle recurrent states
    for cmp ∈ chain.rec
        Pr = view(P, cmp, cmp)
        rhs = zeros(length(cmp) + 1)
        rhs[end] = 1
        stationary = vcat(I - Pr', ones(1, length(cmp))) \ rhs
        for r ∈ eachrow(view(Pstar, cmp, cmp))
            r .= stationary
        end
    end
    # handle transient states
    Pt = lu(I - view(P, chain.trans, chain.trans)) # speed up inverses
    for cmp ∈ chain.rec
        B = view(P, chain.trans, cmp)
        D = view(Pstar, cmp, cmp)
        view(Pstar, chain.trans, cmp) .= Pt \ B * D
    end
    Pstar
end

"""
    occupancy(model, π, μ, γ)

Compute the normalized occupancy measure, transition matrix, and state rewards for the MDP `model`
under policy `π`, given an initial distribution `μ` and discount factor `γ`.
"""
function occupancy(model::TabMDP, π::AbstractVector{Int}, μ::AbstractVector{Float64}, γ::Number)
    0 ≤ γ && γ ≤ 1 || error("γ must be between 0 and 1.")
    sum(μ) ≈ 1 && all(μ .≥ 0) || error("Initial distribution must be non-negative and sum to one.")

    S = state_count(model)
    dist = Vector{Float64}(undef, S)

    (P_π, r_π, g) = mrp_sparse(model, π, true)

    if γ < 1
        dist .= (1 - γ) * ((I - γ * P_π') \ μ)
    else
        comps = attracting_components(g)
        nr = fill(true, S)
        for comp ∈ comps
            nr[comp] .= false
        end
        tr = findall(nr)
        nav = (I - P_π[tr, tr])' \ μ[tr]
        for comp ∈ comps
            dist[comp] = (only(nav' * sum(P_π[tr, comp], dims=2)) + sum(μ[comp])) * (vcat(I - P_π[comp, comp]', ones(1, length(comp))) \ vcat(zeros(length(comp)), 1))
        end
        dist[tr] .= 0
    end
    return (distribution=dist, transition=P_π, reward=r_π)
end
