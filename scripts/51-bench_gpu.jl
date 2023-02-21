# %%
using DropletSpreadingSim2

using DifferentialEquations, Sundials, Logging, DrWatson
using WGLMakie
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# %%
p = Dict(
    :tmax => 50,
    :hₛ_ratio => 1.0,
    :hₛ => 5e-2,
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
)

@unpack h₀, σ, ρ, μ, τ, θτ, L, hₛ, hₛ_ratio, θₛ, dθₛ, hₛ, aspect_ratio, tmax,
mass, ndrops, hdrop_std, two_dim = p
θₐ = deg2rad(θₛ + dθₛ)
θᵣ = deg2rad(θₛ - dθₛ)

# %%
experiment = DropletSpreadingExperiment(; h₀, σ, ρ, μ, τ, θτ, L, hₛ_ratio, hₛ, θₐ, θᵣ,
    aspect_ratio, mass, ndrops, hdrop_std, two_dim, smooth=3.0)
prob = ODEProblem(experiment, (0.0, p[:tmax]), on=:gpu)
# %%
using BenchmarkTools, CUDA
@show Threads.nthreads()

U = cu(experiment.U₀)
dU = similar(U)
p = NamedTuple((key => Float32(value) for (key, value) in pairs(experiment.p)))

# %%
@info "Benchmarking"
@info "non hyperbolic"
@btime experiment.cap!($dU, $U, $p, 0);
@info "hyperbolic"
@btime experiment.hyp!($dU, $U, $p, 0);

@info "F update"
@btime prob.f($dU, $U, $p, 0);

# %%
@info "launch sim, explicit" p
@time sol = solve(
    prob,
    SSPRK432(),
    progress=true,
    progress_steps=1,
    save_everystep=false,
)

# %%
cfl_limiter = build_cfl_limiter(experiment; safety_factor=0.5)

@info "launch sim, IMEX (implicit)" p
@time sol = solve(
    prob,
    IMEXEuler(),
    callback=cfl_limiter,
    progress=true,
    progress_steps=1,
    save_everystep=false,
    dt=1e-3,
)

# %%
using Sundials

@info "launch sim, sundials (implicit)" p
@time sol = solve(
    prob,
    CVODE_BDF(linear_solver=:GMRES),
    progress=true,
    progress_steps=1,
    save_everystep=false,
    on=:gpu,
)

# # %%
# # Now, every solver should be available with autodiff + sparsity pattern.
# # Better performance may be achieve with preconditionning