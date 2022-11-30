# %%
using DropletSpreadingSim2
using DifferentialEquations, Sundials, Logging, DrWatson
using Makie, WGLMakie
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# %%
p = Dict(
    :N => 100,
    :h₀ => 100e-6,
    :σ => 0.075,
    :ρ => 1000.0,
    :μ => 1e-3,
    :τ => 16.0,
    :θτ => 0.0,
    :θₛ => 30,
    :dθₛ => 0,
    :L => 24.0,
    :aspect_ratio => 3,
    :tmax => 2000,
    # :keep_timestep => 0.3,
    :save_timestep => 0.3,
    :hₛ => 0.01,
    :ndrops => 1,
    :hdrop_std => 0.2,
    :mass => 220,
    :two_dim => false,
)

@unpack h₀, σ, ρ, μ, τ, θτ, L, N, θₛ, dθₛ, hₛ, aspect_ratio, tmax,
mass, ndrops, hdrop_std, two_dim = p
θₐ = deg2rad(θₛ + dθₛ)
θᵣ = deg2rad(θₛ - dθₛ)
experiment = DropletSpreadingExperiment(; h₀, σ, ρ, μ, τ, θτ, L, N, θₐ, θᵣ,
    hₛ, aspect_ratio, mass, ndrops, hdrop_std, two_dim, smooth=0.5)

# %%

# Use SparsityTracing to build the jacobian sparsity pattern, compute (slowly) a
# first jacobian that will be used as a pototype and compute the color matrix.
using SparsityTracing, SparseDiffTools

u_ad = SparsityTracing.create_advec(experiment.U₀);
du_ad = similar(u_ad);
@time experiment.eveq!(du_ad, u_ad, experiment.p, 0.0)
@time Jad = SparsityTracing.jacobian(du_ad, length(du_ad));
@time colors = matrix_colors(Jad)

# %%
prob = ODEProblem(ODEFunction(experiment.eveq!, jac_prototype=Jad), experiment.U₀, tmax, experiment.p)

# %%
# reprojection : may need better thresholding
reproject_cb = build_reprojection_callback(experiment; thresh=0.1)

# Vizualisation
fig = Figure()
field = unpack_fields(experiment.U₀, experiment)
if p[:two_dim]
    h_node = Observable(field.h[:, :])
    ax, hm = heatmap(fig[1, 1][1, 1], experiment.grid.x, experiment.grid.y, h_node)
    viz_cb = FunctionCallingCallback() do u, t, integrator
        field = unpack_fields(u, experiment)
        h_node[] = field.h[:, :]
    end
else
    h_node = Observable(field.h[:, 1])
    ax, hm = lines(fig[1, 1][1, 1], experiment.grid.x, h_node)
    viz_cb = FunctionCallingCallback() do u, t, integrator
        field = unpack_fields(u, experiment)
        h_node[] = field.h[:, 1]

    end
end
display(fig)

# %%
# Now, every solver should be available with autodiff + sparsity pattern.
# Better performance may be achieve with preconditionning
@info "launch sim" p
@time sol = solve(
    prob,
    Midpoint(),
    callback=CallbackSet(viz_cb, reproject_cb),
    progress=true,
    progress_steps=1,
    save_everystep=false,
    saveat=get(p, :keep_timestep, []),
    # dtmin=get(p, :dtmin, nothing),
)