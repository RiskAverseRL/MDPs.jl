using Arrow
using MDPs.Domains
using CSV


struct Problem{M <: TabMDP}
    γ :: Float64
    horizon :: Int
    initstate :: Int
    model :: M
end

# creates a set of benchmark problems
function make_domains()
    problems = Dict{String, Problem}()
    # inventory
    begin
        # risk parameters
        γ = 0.8
        initstate = 1         # initial state
        horizon = 100
        # Define the inventory model
        demand = Inventory.Demand([0,2,3,4,5,30,3,2],
                                  [0.1,0.3,0.1,0.1,0.1,0.1,0.0,0.2])
        costs = Inventory.Costs(5.,2.,0.3,0.5)
        limits = Inventory.Limits(100, 0, 50)
        params = Inventory.Parameters(demand, costs, 16., limits)
        model = Inventory.Model(params)
        problems["inventory"] = Problem(γ, horizon, initstate, model)
    end
    #invetory_generic
    begin
        γ = 0.9
        filein  = joinpath(dirname(pathof(MDPs)), "..", "data", "inventory.arr")
        model = load_mdp(Arrow.Table(filein))
        initstate = 1         # initial state
        horizon = 100
        problems["inventory_generic"] = Problem(γ, horizon, initstate, model)
    end
    # machine
    begin
        γ = 0.8
        initstate = 1         # initial state
        horizon = 100
        model = Domains.Machine.Replacement()
        problems["machine"] = Problem(γ, horizon, initstate, model)
    end
    # ruin
    begin
        α = 0.9           # var, cvar, evar
        horizon = 200
        initstate = 8  # capital: state - 1
        model = Domains.Gambler.Ruin(0.7, 10)
        problems["ruin"] = Problem(γ, horizon, initstate, model)
   end
    # riverswim
    begin
        filein = joinpath(dirname(pathof(MDPs)), "..", "data", "riverswim.csv")
        model = load_mdp(CSV.File(filein); idoutcome = 1)
        γ = 0.98
        horizon = 100
        initstate = 1         # initial state
        problems["riverswim"] = Problem(γ, horizon, initstate, model)
    end
    # population
    begin
        filein  = joinpath(dirname(pathof(MDPs)), "..", "data", "population.arr")
        model = load_mdp(Arrow.Table(filein))
        α = 0.9           # var, cvar, evar
        β = 0.5           # erm
        γ = 0.7
        horizon = 50
        initstate = 1         # initial state
        problems["population"] = Problem(γ, horizon, initstate, model)
    end
    # onestatepm
    begin
        model = Domains.Simple.OneStatePlusMinus(100)
        initstate = 1         # initial state
        horizon = 100
        γ = 0.95
        problems["onestatepm"] = Problem(γ, horizon, initstate, model)
    end
    problems
end    

#make_domains()
