# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson, FLoops
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# %%
function do_simulate(p; filename)
    @unpack h₀, σ, ρ, μ, τ, θτ, L, hₛ, hₛ_ratio, θₛ, dθₛ, hₛ, aspect_ratio, tmax,
    mass, ndrops, hdrop_std, two_dim, reproject = p
    θₐ = deg2rad(θₛ + dθₛ)
    θᵣ = deg2rad(θₛ - dθₛ)

    # %%
    experiment = DropletSpreadingExperiment(; h₀, σ, ρ, μ, τ, θτ, L, hₛ_ratio, hₛ, θₐ, θᵣ,
        aspect_ratio, mass, ndrops, hdrop_std, two_dim)

    # %%
    prob = ODEProblem(experiment, (0.0, p[:tmax]))
    cfl_limiter = build_cfl_limiter(experiment; safety_factor=p[:cfl_safety_factor])
    callbacks = Any[cfl_limiter]
    if ~isnothing(filename)
        save_cb = build_save_callback(
            filename, prob, experiment;
            saveat=get(p, :save_timestep, nothing), attrib=p
        )
        push!(callbacks, save_cb)
    end
    if reproject
        reproject_cb = build_reprojection_callback(experiment; thresh=1e-6)
        push!(callbacks, reproject_cb)
    end

    @info "launch sim" p
    @time sol = solve(
        prob,
        SSPRK432(),
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
    :mass => 220,
    :two_dim => false,
    :tmax => 1000,
    :hₛ_ratio => 1.0,
    :hₛ => [2e-2, 2.5e-2, 3e-2, 5e-2],
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
    :aspect_ratio => 2,
    :ρ => 1000.0,
    :τ => 8.0,
    :L => 24,
    :two_dim => false,
    :cfl_safety_factor => 0.9,
    :reproject => [true, false],
)
parameters = dict_list(parameters)

# %%
for p ∈ parameters
    out_dir = "data/outputs/reprojection/"
    filename = savename(p, "nc", accesses=[:reproject, :hₛ])
    sol, experiment = do_simulate(p; filename=joinpath(out_dir, filename))
end
