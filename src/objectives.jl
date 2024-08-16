"""
Abstract objective for an MDP.
"""
abstract type Objective end

"""
Objective solved by a randomized Markov non-stationary policy.
In other words, the solution is time-dependent.
"""
abstract type Markov <: Objective end

"""
Objective that is solved by a randomized stationary policy
"""
abstract type Stationary <: Markov end

"""
Objective solved by a deterministic Markov non-stationary policy.
In other words, the solution is time-dependent.
"""
abstract type MarkovDet <: Markov end

"""
Objective that is solved by a randomized stationary policy
"""
abstract type StationaryDet <: Stationary end

"""
Inifinite-horizon discounted objective. The discount factor `γ` can
be in [0,1]. The optimal policy is stationary.
"""
struct InfiniteH <: StationaryDet
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
struct FiniteH <: MarkovDet
    γ::Float64
    T::Int

    function FiniteH(γ::Number, T::Integer) 
        one(γ) ≥ γ ≥ zero(γ) || error("Discount γ must be in [0,1]")
        T ≥ one(T) || error("Horizon must be at least one")
        new(γ, T)
    end
end


"""
Total reward criterion. The objective is to maximize the sum
of the rewards. The model assumes that there is a terminal state,
which must

1) have a single action,
2) transition to itself,
3) have a reward of 0. 
"""
struct TotalReward <: StationaryDet
    terminal_state :: Int 
end

"""
    horizon(objective)

Return the horizon length for `objective`.
"""
function horizon end

horizon(o::FiniteH) = o.T

discount(o::FiniteH) = o.γ
discount(o::InfiniteH) = o.γ

