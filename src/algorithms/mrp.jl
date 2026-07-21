
# ----------------------------------------------------------------
# Markov reward process and Markov chain
# ----------------------------------------------------------------
using Graphs
using LinearAlgebra

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
    mrp(model, π)

Compute a sparse transition matrix `P_π` and reward vector `r_π` for the
MDP `model` and policy `π`.
"""
function mrp_sparse(model::TabMDP, π::AbstractVector{Int})
    S = state_count(model)
    r_π = zeros(S)
    P_π = spzeros(S, S)

    for s ∈ 1:S
        for (sn, p, r) ∈ transition(model, s, π[s])
            P_π[s, sn] += p
            r_π[s] += p * r
        end
    end
    (P_π, r_π)
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
function decompose_chain(P :: Matrix{<:Real})
    S = size(P)[1]
    S == size(P)[2] || error("Only square P is supported")
    
    g = SimpleDiGraph(S)
    for s₁ ∈ 1:S, s₂ ∈ 1:S
        P[s₁, s₂] > 0 && add_edge!(g, s₁, s₂)
    end

    rec = attracting_components(g)
    trans_b = trues(S)
    @inbounds foreach(i-> trans_b[i] = false, Iterators.flatten(rec))
    trans = findall(trans_b)
    (rec = rec, trans = trans)
end


"""
    stationary_matrix(P)

Compute the stationary matrix for a Markov chain with transition matrix `P`. 

# Return
  - Stationary matrix `Pstar` such that: `P * Pstar = Pstar` and `Pstar * P = Pstar`
"""
function stationary_matrix(P :: Matrix{<:Real})
    chain = decompose_chain(P)
    all(P .≥ zero(eltype(P))) || error("Transition probabilities must be non-negative.")
    all(sum.(eachrow(P)) .≈ 1)  || error("Probabilities must sum to 1.")
    
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
    stationary_dist_sparse(model, π, μ)

Compute the stationary matrix for an MDP `model`, a deterministic policy `π`, and initial distribution `μ`. The function used sparse linear algebra for scalability.
"""
function stationary_dist_sparse(model::TabMDP, π::AbstractVector{Int}, μ::AbstractVector{Float64})
    sum(μ) ≈ 1 && all(μ .≥ 0) || error("Initial distribution must be non-negative and sum to one.")

    S = state_count(model)
    r_π = zeros(S)
    P_π = spzeros(S, S)
    g = SimpleDiGraph(S)
    dist = Vector{Float64}(undef, S)

    for s ∈ 1:S
        for (sn, p, r) ∈ transition(model, s, π[s])
            P_π[s, sn] += p
            add_edge!(g, s, sn)
        end
    end

    comps = attracting_components(g)
    nr = fill(true, S)  # notrecurrent
    nr[Iterators.flatten(comp)] .= false
    
    tr = findall(nr)
    t2r = spzeros(length(tr), length(comps))
    for (i, comp) ∈ enumerate(comps)
        t2r[:, i] = sum(P_π[tr, comp], dims=2)
    end
    nav_probs = t2r' * ((I - P_π[tr, tr])' \ μ[tr])
    for (prob, comp) ∈ zip(nav_probs, comps)
        dist[comp] = (prob + sum(μ[comp])) * (vcat(I - P_π[comp, comp]', ones(1, length(comp))) \ vcat(zeros(length(comp)), 1))
    end
    dist[tr] .= 0
    return dist
end
