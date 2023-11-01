"""
Abstract objective for an MDP.
"""
abstract type Objective end

"""
Objective that is solved by a deterministic stationary policy
"""
abstract type Stationary <: Objective end

"""
Objective that is solved by a randomized stationary policy
"""
abstract type StationaryRand <: Objective end

"""
Objective that is solved using a deterministic Markov but non-stationary policy.
In other words, the solution is time-dependent.
"""
abstract type Markov <: Objective end

"""
Objective that is solved using a randomized Markov policy
"""
abstract type MarkovRand <: Objective end


"""
Objective that is solved using a deterministic policy
which is Markov on an augmented state space
"""
abstract type AugmentedMarkov <: Markov end


"""
Inifinite-horizon discounted objective. The discount factor `γ` can
be in [0,1]. The optimal policy is stationary.
"""
struct InfiniteH <: Stationary
    γ::Float64

    function InfiniteH(γ::Number)
        one(γ) ≥ γ ≥ zero(γ) || error("Discount γ must be in [0,1]")
        new(γ)
    end
end

"""
Finite-horizon discounted model. The discount factor `γ` can
be in [0,1]. The optimal policy is Markov but time dependent.
"""
struct FiniteH <: Markov 
    γ::Float64
    T::Int

    function FiniteH(γ::Number, T::Integer) 
        one(γ) ≥ γ ≥ zero(γ) || error("Discount γ must be in [0,1]")
        T ≥ one(T) || error("Horizon must be at least one")
        new(γ, T)
    end
end

"""
    horizon(objective)

Return the horizon length for `objective`.
"""
function horizon end

horizon(o::FiniteH) = o.T


