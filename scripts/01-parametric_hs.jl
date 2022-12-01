# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson
using TerminalLoggers: TerminalLogger
using SparsityTracing, SparseDiffTools
global_logger(TerminalLogger())

# %%
function do_simulate(p; filename)
    @unpack h₀, σ, ρ, μ, τ, θτ, L, hₛ, θₛ, dθₛ, hₛ_ratio, aspect_ratio, tmax, mass, ndrops, hdrop_std, two_dim = p
    θₐ = deg2rad(θₛ + dθₛ)
    θᵣ = deg2rad(θₛ - dθₛ)
    experiment = DropletSpreadingExperiment(
            ; h₀, σ, ρ, μ, τ, θτ, L, hₛ, θₐ, θᵣ,
            hₛ_ratio, aspect_ratio, mass, ndrops,
            hdrop_std, two_dim, smooth=0.5
        )
    u_ad = SparsityTracing.create_advec(experiment.U₀);
    du_ad = similar(u_ad);
    experiment.eveq!(du_ad, u_ad, experiment.p, 0.0)
    Jad = SparsityTracing.jacobian(du_ad, length(du_ad));
    colors = matrix_colors(Jad)

    prob = ODEProblem(
        ODEFunction(
            experiment.eveq!, jac_prototype=Jad, colorvec=colors
            ),
        experiment.U₀, tmax, experiment.p
        )

    # reprojection : may need better thresholding
    reproject_cb = build_reprojection_callback(experiment; thresh=0.1)

    callbacks = Any[reproject_cb]
    if ~isnothing(filename)
        save_cb = build_save_callback(
            filename, prob, experiment;
            saveat=get(p, :save_timestep, nothing), attrib=p
        )
        push!(callbacks, save_cb)
    end

    @info "launch sim" p
    @time sol = solve(
        prob,
        Midpoint(),
        callback=CallbackSet(callbacks...),
        progress=true,
        progress_steps=1,
        save_everystep=false,
        saveat=get(p, :keep_timestep, [])
    )
    return sol, experiment
end

# %%
parameters = Dict(
    :hₛ_ratio => 0.5, # hₛ_ratio= 2 hₛ/δ
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
    :tmax => 500,
    :save_timestep => 0.3,
    :hₛ => [
        0.05, 0.01
        ], # hₛ = hₛ_ratio / 2 * δ
    :ndrops => 1,
    :hdrop_std => 0.2,
    :mass => 220,
    :two_dim => false,
)
parameters = dict_list(parameters)

# %%
for p ∈ parameters
    out_dir = "data/outputs/hs_effect/"
    filename = savename(p, "nc", accesses=[:hₛ])
    sol, experiment = do_simulate(p; filename=joinpath(out_dir, filename))
end
