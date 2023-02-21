# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson
using WGLMakie
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# %%
p = Dict(
    :tmax => 2000,
    :hₛ_ratio => 0.5,
    :hₛ => 1e-1,
    :ndrops => 1,
    :hdrop_std => 0.1,
    :h₀ => 0.0001,
    :μ => 0.001,
    :σ => 0.075,
    :θₛ => 30,
    :dθₛ => 0,
    :save_timestep => 0.3,
    :θτ => 0.0,
    :mass => 220,
    :aspect_ratio => 2,
    :ρ => 1000.0,
    :τ => 8.0,
    :L => 100,
    :two_dim => false,
)

@unpack h₀, σ, ρ, μ, τ, θτ, L, hₛ, hₛ_ratio, θₛ, dθₛ, hₛ, aspect_ratio, tmax,
mass, ndrops, hdrop_std, two_dim = p
θₐ = deg2rad(θₛ + dθₛ)
θᵣ = deg2rad(θₛ - dθₛ)

# %%
experiment = DropletSpreadingExperiment(;
    h₀, σ, ρ, μ, τ, θτ, L, hₛ_ratio, hₛ, θₐ, θᵣ,
    aspect_ratio, mass, ndrops, hdrop_std, two_dim, smooth=3.0
)

# %%
prob = ODEProblem(experiment, (0.0, p[:tmax]), on=:cpu, compute_sparsity=true)
reproject_cb = build_reprojection_callback(experiment; thresh=1e-3)

# %%
# Vizualisation
fig = Figure()
field = unpack_fields(experiment.U₀, experiment)
funcat = 0:1:p[:tmax] |> collect
if p[:two_dim]
    h_node = Observable(field.h[:, :])
    ax, hm = heatmap(fig[1, 1][1, 1], experiment.grid.x, experiment.grid.y, h_node)
    viz_cb = FunctionCallingCallback(;funcat) do u, t, integrator
        u = collect(u) # put data on CPU
        field = unpack_fields(u, experiment)
        h_node[] = field.h[:, :]
    end
else
    h_node = Observable(field.h[:, 1])
    ax, hm = lines(fig[1, 1][1, 1], experiment.grid.x, h_node)
    viz_cb = FunctionCallingCallback(;funcat) do u, t, integrator
        u = collect(u) # put data on CPU
        field = unpack_fields(u, experiment)
        h_node[] = field.h[:, 1]
    end
end
display(fig)

# save_cb = build_save_callback(
#     "data/test.nc", prob, experiment;
#     saveat=get(p, :save_timestep, nothing), attrib=p
# )

cfl_limiter = build_cfl_limiter(experiment; safety_factor=0.9, max_step=false)

# %%
@info "launch sim" p
@time sol = solve(
    prob,
    IMEXEuler();
    callback=CallbackSet(reproject_cb, cfl_limiter, viz_cb),
    progress=true,
    progress_steps=1,
    save_everystep=false,
    saveat=get(p, :keep_timestep, []),
    dt=1e-6,
)
