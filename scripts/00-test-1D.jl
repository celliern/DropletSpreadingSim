# %%
using DropletSpreadingSim2

using SparsityTracing, SparseDiffTools
using DifferentialEquations, Sundials, Logging, DrWatson
using WGLMakie
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# %%
p = Dict(
    :tmax => 500,
    :hₛ_ratio => 1,
    # :N => 61,
    :hₛ => 0.2,
    :ndrops => 1,
    :hdrop_std => 0.2,
    :h₀ => 0.0001,
    :μ => 0.001,
    :σ => 0.075,
    :θₛ => 30,
    :dθₛ => 0,
    :save_timestep => 0.3,
    :θτ => 0.0,
    :mass => 220,
    :hₛ => 0.01,
    :aspect_ratio => 3,
    :ρ => 1000.0,
    :τ => 8.0,
    :L => 24,
    :two_dim => false,
)

@unpack h₀, σ, ρ, μ, τ, θτ, L, hₛ_ratio, hₛ, θₛ, dθₛ, hₛ, aspect_ratio, tmax,
mass, ndrops, hdrop_std, two_dim = p
θₐ = deg2rad(θₛ + dθₛ)
θᵣ = deg2rad(θₛ - dθₛ)

# %%
experiment = DropletSpreadingExperiment(; h₀, σ, ρ, μ, τ, θτ, L, hₛ_ratio, hₛ, θₐ, θᵣ,
    aspect_ratio, mass, ndrops, hdrop_std, two_dim, smooth=30.0)

# %%
# Use SparsityTracing to build the jacobian sparsity pattern, compute (slowly) a
# first jacobian that will be used as a pototype and compute the color matrix.

u_ad = SparsityTracing.create_advec(experiment.U₀);
du_ad = similar(u_ad);
@time experiment.eveq!(du_ad, u_ad, experiment.p, 0.0)
@time Jad = SparsityTracing.jacobian(du_ad, length(du_ad));
@time colors = matrix_colors(Jad)

# %%
prob = ODEProblem(
    ODEFunction(experiment.eveq!, jac_prototype=Jad, colorvec=colors),
    experiment.U₀, tmax, experiment.p
)

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

using ODEInterfaceDiffEq

@info "launch sim" p
@time sol = solve(
    prob,
    SSPRK432(),
    callback=CallbackSet(viz_cb, reproject_cb),
    progress=true,
    progress_steps=1,
    save_everystep=true,
    saveat=get(p, :keep_timestep, []),
    reltol=1e-6,
    dt=1e-10
    # dtmin=get(p, :dtmin, nothing),
)

# %%
