# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson
using TerminalLoggers: TerminalLogger
using GLMakie
global_logger(TerminalLogger(stderr))

# %%
function do_simulate(p; filename, viz=false)
    @unpack h₀, σ, ρ, μ, τ, θτ, L, hₛ, hₛ_ratio, θₛ, dθₛ, hₛ, aspect_ratio, tmax,
    mass, ndrops, hdrop_std, two_dim, reproject = p
    θₐ = deg2rad(θₛ + dθₛ)
    θᵣ = deg2rad(θₛ - dθₛ)

    # %%
    experiment = DropletSpreadingExperiment(; h₀, σ, ρ, μ, τ, θτ, L, hₛ_ratio, hₛ, θₐ, θᵣ,
        aspect_ratio, mass, ndrops, hdrop_std, two_dim)

    # %%
    prob = ODEProblem(experiment, (0.0, p[:tmax]), on=:gpu)
    # cfl_limiter = build_cfl_limiter(experiment; safety_factor=p[:cfl_safety_factor])
    callbacks = Any[]
    if ~isnothing(filename)
        save_cb = build_save_callback(
            filename, prob, experiment;
            saveat=get(p, :save_timestep, nothing), attrib=p
        )
        push!(callbacks, save_cb)
    end
    if reproject
        reproject_cb = build_reprojection_callback(experiment; thresh=1e-3)
        push!(callbacks, reproject_cb)
    end
    if viz
        fig, viz_cb = build_viz_cb(prob, experiment)
        push!(callbacks, viz_cb)
        display(fig)
    end


    @info "launch sim" p
    @time sol = solve(
        prob,
        SSPRK432();
        callback=CallbackSet(callbacks...),
        progress=true,
        progress_steps=1,
        save_everystep=false,
        saveat=get(p, :keep_timestep, []),
    )

    return sol, experiment
end

# %%
parameters = Dict(
    :hdrop_std => 0.2,
    :tmax => 400,
    :hₛ_ratio => 1.0,
    :hₛ => 5e-2,
    :ndrops => 1,
    :h₀ => 0.0001,
    :μ => 0.001,
    :σ => 0.075,
    :θₛ => 30,
    :dθₛ => [0, 5, 9],
    :save_timestep => 0.5,
    :θτ => 0.0,
    :mass => 220,
    :aspect_ratio => 3,
    :ρ => 1000.0,
    :τ => 8.0,
    :L => 24,
    :two_dim => true,
    :reproject => true,
)
parameters = dict_list(parameters)

# %%
for p ∈ parameters
    out_dir = "data/outputs/2D_simple"
    filename = savename(p, "nc", accesses=[:dθₛ])
    if ~isnothing(filename) && isfile(joinpath(out_dir, "$(basename(filename)).done"))
        @info "skipping" filename
        continue
    end
    # remove filename if it exists
    if ~isnothing(filename) && isfile(filename)
        rm(filename)
    end

    sol, experiment = do_simulate(p; filename=joinpath(out_dir, filename), viz=false)
    if ~isnothing(filename)
        touch(joinpath(out_dir, "$(basename(filename)).done"))
    end
end
