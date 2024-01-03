# Recipes

## Converting a file format of an MDP

Converting from a CSV to an Arrow file
```jldoctest
using MDPs
using DataFrames
using Arrow
using CSV

filein  = joinpath(dirname(pathof(MDPs)), "..", "data", "riverswim.csv")
fileout = tempname() 

model = load_mdp(CSV.File(filein); idoutcome = 1)

output = save_mdp(DataFrame, model)
```	   

Converting from an Arrow to a CSV file
```jldoctest
using MDPs
using DataFrames
using Arrow
using CSV

filein  = joinpath(dirname(pathof(MDPs)), "..", "data", "inventory.arr")
fileout = tempname()

model = load_mdp(Arrow.Table(filein))

output = save_mdp(DataFrame, model)
CSV.write(fileout, output)
```	   
