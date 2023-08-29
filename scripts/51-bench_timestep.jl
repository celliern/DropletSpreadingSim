# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson
using TerminalLoggers: TerminalLogger
using BenchmarkTools
global_logger(TerminalLogger(stderr))

# %%
function do_simulate(p)
    @unpack h₀, σ, ρ, μ, τ, θτ, L, N, θₛ, dθₛ, hₛ_ratio, aspect_ratio, tmax,
    mass, ndrops, hdrop_std, two_dim, reproject = p
    θₐ = deg2rad(θₛ + dθₛ)
    θᵣ = deg2rad(θₛ - dθₛ)

    experiment = DropletSpreadingExperiment(; h₀, σ, ρ, μ, τ, θτ, L, hₛ_ratio, θₐ, θᵣ,
        N, aspect_ratio, mass, ndrops, hdrop_std, two_dim)

    prob = ODEProblem(experiment, (0.0, p[:tmax]), on=:cpu)
    callbacks = Any[]
    if reproject
        reproject_cb = build_reprojection_callback(experiment;)
        push!(callbacks, reproject_cb)
    end

    @time sol = solve(
        prob,
        SSPRK432();
        callback=CallbackSet(callbacks...),
        save_everystep=true,
        progress=true,
        progress_steps=1,
    )

    return experiment.grid.n₁, median(diff(sol.t)[end-100:end])
end

# %%
ps = Dict(
    :hdrop_std => 0.2,
    :tmax => 20,
    :hₛ_ratio => 1.0,
    :N => 120:120:960 |> collect,
    # :hₛ => 5e-2,
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
    :two_dim => false,
    :reproject => true,
)
ps = dict_list(ps)

# %%
ns = zeros(length(ps))
dts = zeros(length(ps))
for (i, p) in enumerate(ps)
    n, dt = do_simulate(p)
    ns[i] = n
    dts[i] = dt
    @info n dt
end

# %%
using CSV, DataFrames
CSV.write("./data/outputs/bench_timestep.csv", DataFrame(n=ns, dt=dts))

# %%
using Plots

# %%
plot(ns, dts, seriestype=:scatter, xlabel="n", ylabel="dt", legend=false, xscale=:log10, yscale=:log10)
plot!(ns, dts, seriestype=:line)