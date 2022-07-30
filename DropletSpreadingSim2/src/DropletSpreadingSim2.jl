module DropletSpreadingSim2

# __precompile__(false)

using Reexport

include("Model/Model.jl")
include("Experiments.jl")

@reexport using .Model
@reexport using .Experiments

end # module
