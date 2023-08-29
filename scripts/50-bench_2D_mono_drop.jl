# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson
using TerminalLoggers: TerminalLogger
using BenchmarkTools
global_logger(TerminalLogger(stderr))

# %%
function do_simulate(p)
    @unpack h₀, σ, ρ, μ, τ, θτ, L, hₛ, hₛ_ratio, θₛ, dθₛ, hₛ, aspect_ratio, tmax,
    mass, ndrops, hdrop_std, two_dim, reproject = p
    θₐ = deg2rad(θₛ + dθₛ)
    θᵣ = deg2rad(θₛ - dθₛ)

    experiment = DropletSpreadingExperiment(; h₀, σ, ρ, μ, τ, θτ, L, hₛ_ratio, hₛ, θₐ, θᵣ,
        aspect_ratio, mass, ndrops, hdrop_std, two_dim)

    prob = ODEProblem(experiment, (0.0, p[:tmax]), on=:cpu)
    callbacks = Any[]
    if reproject
        reproject_cb = build_reprojection_callback(experiment; thresh=1e-3)
        push!(callbacks, reproject_cb)
    end

    @info "launch sim" p
    @time sol = solve(
        prob,
        SSPRK432();
        callback=CallbackSet(callbacks...),
        save_everystep=false,
        progress=true,
        progress_steps=1,
    )

    return sol, experiment
end

# %%
p = Dict(
    :hdrop_std => 0.2,
    :tmax => 20,
    :hₛ_ratio => 1.0,
    :hₛ => 5e-2,
    :ndrops => 1,
    :h₀ => 0.0001,
    :μ => 0.001,
    :σ => 0.075,
    :θₛ => 30,
    :dθₛ => 0,
    :θτ => 0.0,
    :mass => 220,
    :aspect_ratio => 3,
    :ρ => 1000.0,
    :τ => 8.0,
    :L => 24,
    :two_dim => true,
    :reproject => true,
)

# %%
@time (sol, experiment) = do_simulate(p);