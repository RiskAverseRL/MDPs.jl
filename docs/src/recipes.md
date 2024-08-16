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
1

# output

1
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
1

# output

1
```

## Making a small MDP

```jldoctest
using MDPs

ε = 0.01
P1 = [1 0 0; 0   1 0; 0 0 1]
P2 = [0 1 0; 1-ε 0 ε; 0 0 1]
Ps = [P1, P2]
R = [10 -4 0; -1 -3 0; 0 0 0] # use the same reward for both actions
Rs = [R, R]

M = make_int_mdp(Ps, Rs)
state_count(M)

# output

3
```


## Saving an MDP to a file

```jldoctest
using MDPs
using DataFrames
using MDPs.Domains
using CSV

model = Gambler.Ruin(0.7, 10)
domainoutput = MDPs.save_mdp(DataFrame, model)
CSV.write("output_gambler.csv", domainoutput)

1

# output

1
```
