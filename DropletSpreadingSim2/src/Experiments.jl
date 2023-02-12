module Experiments
export DropletSpreadingExperiment,
    unpack_fields,
    build_save_callback,
    build_reprojection_callback,
    init_model,
    build_cfl_limiter,
    build_manifold_project_callback

using UnPack, NCDatasets, DiffEqCallbacks, Printf, StaticArrays, DiffEqBase
using CUDA, FoldsCUDA
using Distributions, LinearAlgebra, DataStructures
using SparsityTracing, SparseDiffTools, SparseArrays
using FLoops
using ImageFiltering
using Base.Filesystem: dirname, mkpath
import DiffEqBase: ODEProblem
import Base: show

using ..Model
using ..Model: nᵤ

drop(R₀, θ, r) = @. max(sqrt(max(R₀^2 / sin(θ)^2 - r^2, 0)) - R₀ / tan(θ), 0)
drop(R₀, x₀, y₀, θ, x, y) = (r = @.(√((x - x₀)^2 + (y - y₀)'^2));
drop(R₀, θ, r))

function vdrop(Rθ, θ)
    R = Rθ / sin(θ)
    h = R * (1 - cos(θ))
    vdrop = π * h^2 * (R - h / 3)
    return vdrop
end

struct DropletSpreadingExperiment
    U₀::Vector{Float64}
    p::NamedTuple
    grid::NamedTuple
    hyp!::Function
    cap!::Function
    unpack::Function
    caches::AbstractDict{Type,NamedTuple}
end

show(io::IO, exp::DropletSpreadingExperiment) =
    print(io, "DropletSpreadingExperiment($(exp.p), $(exp.grid))")

function unpack_fields_flat(U, exp::DropletSpreadingExperiment; raw=false)
    @unpack n₁, n₂ = exp.grid
    h, hux, huy, hvx, hvy, hϕxx, hϕxy, hϕyy = eachslice(reshape(U, 8, n₁, n₂); dims=1)
    if raw
        return (
            h=h,
            ux=hux,
            uy=huy,
            vx=hvx,
            vy=hvy,
            ϕxx=hϕxx,
            ϕxy=hϕxy,
            ϕyy=hϕyy,
        )
    end
    ux, uy, vx, vy, ϕxx, ϕxy, ϕyy =
        [hux, huy, hvx, hvy, hϕxx, hϕxy, hϕyy] .|> ((var) -> var ./ h)
    return (h=h, ux=ux, uy=uy, vx=vx, vy=vy, ϕxx=ϕxx, ϕxy=ϕxy, ϕyy=ϕyy)
end

function unpack_fields_vect(U, exp::DropletSpreadingExperiment; raw=false)
    @unpack n₁, n₂ = exp.grid
    @unpack h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy = unpack_fields_flat(U, exp; raw)
    u = [@SVector([ux[i, j], uy[i, j]]) for i = 1:n₁, j = 1:n₂]
    v = [@SVector([vx[i, j], vy[i, j]]) for i = 1:n₁, j = 1:n₂]
    ϕ = [@SMatrix([ϕxx[i, j] ϕxy[i, j]; ϕxy[i, j] ϕyy[i, j]]) for i = 1:n₁, j = 1:n₂]
    return (h=h, u=u, v=v, ϕ=ϕ)
end

function unpack_fields(U, exp::DropletSpreadingExperiment; vect=false, raw=false)
    if vect
        return unpack_fields_vect(U, exp; raw)
    else
        return unpack_fields_flat(U, exp; raw)
    end
end

coerce_attrib(value) = value
coerce_attrib(value::Bool) = Int(value)

function build_save_callback(
    filename,
    prob,
    exp::DropletSpreadingExperiment;
    saveat::Number=0,
    attrib=Dict{Symbol,Any}
)
    @unpack x, y = exp.grid
    saveat =
        ifelse(saveat == 0, Vector{Float64}(), range(extrema(prob.tspan)..., step=saveat))
    mkpath(dirname(filename))
    attrib = Dict(zip(String.(keys(attrib)), coerce_attrib.(values(attrib))))
    Dataset(filename, "c", attrib=attrib) do ds
        defDim(ds, "t", Inf)
        defDim(ds, "x", size(x, 1))
        defDim(ds, "y", size(y, 1))
        t_ = defVar(ds, "t", Float64, ("t",))
        x_ = defVar(ds, "x", Float64, ("x",))
        y_ = defVar(ds, "y", Float64, ("y",))
        x_[:] = x |> collect
        y_[:] = y |> collect

        for var in ["h", "ux", "uy", "vx", "vy", "ϕxx", "ϕxy", "ϕyy"]
            defVar(ds, var, Float64, ("t", "x", "y"))
        end
    end

    save_cb =
        FunctionCallingCallback(; funcat=saveat, func_start=false) do u, t, integrator
            Dataset(filename, "a", attrib=attrib) do ds
                fields = unpack_fields_flat(u, exp)
                next_tindex = size(ds["t"], 1) + 1
                ds["t"][next_tindex] = t
                [ds[var][next_tindex, :, :] = fields[var] for var in keys(fields)]
            end
        end
    return save_cb
end

function build_reprojection_callback(
    exp::DropletSpreadingExperiment;
    thresh=nothing,
    kwargs...
)
    reproj_cb = FunctionCallingCallback(; kwargs...) do Uvec, _, integrator
        @unpack n₁, n₂, Δx, Δy = exp.grid
        @unpack h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy = exp.caches[typeof(Uvec)].cap
        vx_new = copy(vx)
        vy_new = copy(vy)
        @unpack κ = exp.p
        unpack_Uvec!(h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, Uvec, n₁, n₂)
        compute_v!(vx_new, vy_new, h, κ, Δx, Δy, n₁, n₂)
        if isnothing(thresh) || (
            (norm(vx - vx_new) / norm(vx) > thresh) ||
            (norm(vy - vy_new) / norm(vy) > thresh)
        )
            @info "err more than thresh, reprojection..."
            pack_Uvec!(integrator.u, h, ux, uy, vx_new, vy_new, ϕxx, ϕxy, ϕyy, n₁, n₂)
            return
        end
    end
    return reproj_cb
end

function build_manifold_project_callback(
    exp::DropletSpreadingExperiment;
    kwargs...
)
    function g(resid, u, p, t)
        @unpack n₁, n₂, Δx, Δy = exp.grid
        @unpack h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy = exp.caches[typeof(u)].cap
        resid_0 = zeros(size(h)...)
        vx_new = copy(vx)
        vy_new = copy(vy)
        @unpack κ = exp.p
        unpack_Uvec!(h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, u, n₁, n₂)
        compute_v!(vx_new, vy_new, h, κ, Δx, Δy, n₁, n₂)
        res_vx = vx .- vx_new
        res_vy = vy .- vy_new
        pack_Uvec!(
            resid, resid_0, resid_0, resid_0, res_vx, res_vy, resid_0, resid_0, resid_0, n₁, n₂
            )
        return
    end
    return ManifoldProjection(g)
end

function build_cfl_limiter(exp::DropletSpreadingExperiment; kwargs...)
    @unpack Δx, Δy, n₁, n₂ = exp.grid
    function dtFE(Uvec, p, t)
        Uvec_raw = Uvec |> collect
        @unpack U = exp.caches[typeof(Uvec_raw)].hyp
        matricize_Uvec!(U, Uvec_raw, n₁, n₂)
        dtmax = 0.0
        @floop ThreadedEx() for I in CartesianIndices((n₁, n₂))
            i, j = Tuple(I)
            visc_vel = U[i, j, 2] / U[i, j, 1] + √(3U[i, j, 1]) * √(max(U[i, j, 6], 0))
            dt = Δx / visc_vel
            @reduce() do (dtmax = 0; dt)
                if isless(dtmax, dt)
                    dtmax = dt
                end
            end
        end
        return dtmax
    end
    return StepsizeLimiter(dtFE; kwargs...)
end

function init_model(x, y, h, p)

    if p isa Dict{Symbol,Any}
        p = dict2ntuple(p)
    end
    n₁, n₂ = length.([x, y])
    Δx = n₁ > 1 ? step(x) : 1.0
    Δy = n₂ > 1 ? step(y) : 1.0
    caches = DefaultDict{Type,NamedTuple}(passkey=true) do T
        return build_cache(T, n₁, n₂)
    end
    gridinfo = (x=x, y=y, Δx=Δx, Δy=Δy, n₁=n₁, n₂=n₂)

    @unpack κ, τx, τy = p

    ux = @. h * τx / 2
    uy = @. h * τy / 2
    vx = zeros(n₁, n₂)
    vy = zeros(n₁, n₂)
    ϕxx = zeros(n₁, n₂)
    ϕxy = zeros(n₁, n₂)
    ϕyy = zeros(n₁, n₂)

    compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂)
    compute_ϕ!(h, ux, uy, ϕxx, ϕxy, ϕyy, τx, τy)

    U₀ = zeros(nᵤ * n₁ * n₂)
    pack_Uvec!(U₀, h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, n₁, n₂)

    return (
        U₀=U₀,
        (hyp!)=(dU, U, p, t) -> update_hyp!(dU, U, p, t; gridinfo, caches),
        (cap!)=(dU, U, p, t) -> update_cap!(dU, U, p, t; gridinfo, caches),
        unpack=(U; vect) -> unpack_fields(U; gridinfo, vect),
        grid=gridinfo,
        caches=caches,
    )
end

function DropletSpreadingExperiment(
    hi=[],
    ;
    h₀,
    σ,
    ρ,
    μ,
    τ,
    θτ,
    L,
    θₐ=0.0,
    θᵣ=0.0,
    N=nothing,
    hₛ_ratio=nothing,
    hₛ,
    ndrops=1,
    hdrop_std=0.2,
    aspect_ratio=1,
    two_dim=true,
    holdup=0.02,
    mass=nothing,
    smooth=false
)
    if L < 2h₀
        error("Domain length < 2h₀")
    end

    if (isnothing(N) & isnothing(hₛ)) | (~isnothing(N) & ~isnothing(hₛ))
        error("You should give either hₛ or N")
    end

    if isnothing(N)
        δ = 2 * hₛ / hₛ_ratio
        N = ceil(1 + L / δ)
    end

    if isnothing(hₛ)
        hₛ = hₛ_ratio / 2 * δ
    end

    δ = L / (N - 1)

    if isnothing(mass)
        mass = holdup * L^2 * aspect_ratio
    end

    u₀ = h₀ * τ / μ
    ν = μ / ρ

    Re = u₀ * h₀ / ν
    κ = σ / (ρ * h₀ * u₀^2)
    β = (3π)^2 / 4

    # echelle de vitesse sur  τ donc norme de (τx, τy) = 1
    τx = cos(θτ)
    τy = sin(θτ)
    # attention ! A cause de la peridocité, il ne faut pas le dernier point du domaine
    x = range(-L / 2, L * (aspect_ratio - 1 / 2) - δ, step=δ)

    if two_dim
        y = range(-L / 2, L / 2 - δ, step=δ)
    else
        y = [0.0]
    end

    n₁, n₂ = length.([x, y])
    Δx = Δy = δ

    p = (Re=Re, κ=κ, β=β, τx=τx, τy=τy, hₛ=hₛ, θₐ=θₐ, θᵣ=θᵣ)

    if isempty(hi)
        hi = hₛ
    end
    # random distribution of drops with a total mass equal to mass
    θₛ = 0.5 * (θₐ + θᵣ)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    d_R = Normal(1.0, hdrop_std)
    R = rand(d_R, ndrops)
    voldrop = vdrop.(R, θₛ)
    vol = sum(voldrop)
    R = R * (abs(mass) / vol)^(1 / 3)
    Rmoy = mean(R)
    # drops must be deposited within the numerical domain
    d_posx = Uniform(xmin + Rmoy, xmax - Rmoy)
    if two_dim
        d_posy = Uniform(ymin + Rmoy, ymax - Rmoy)
    else
        d_posy = 0
    end
    # withdraw mass if mass is negative (2*mass)
    if mass < 0
        hw = 2 * mass / (L^2 * aspect_ratio)
    else
        hw = 0
    end

    if ndrops == 1
        h = sum(drop.(R, 0, 0, θₛ, Ref(x), Ref(y))) .+ hi .+ hw
    else
        h =
            sum(drop.(R, rand(d_posx, ndrops), rand(d_posy, ndrops), θₛ, Ref(x), Ref(y))) .+
            hi .+ hw
    end
    if smooth > 0
        h = imfilter(h, Kernel.gaussian(smooth))
    end
    U₀, hyp!, cap!, unpack, grid, caches = init_model(x, y, h, p)
    return DropletSpreadingExperiment(U₀, p, grid, hyp!, cap!, unpack, caches)
end

function ODEProblem(f::DropletSpreadingExperiment, tspan, args...; on=:cpu, kwargs...)
    U₀ = f.U₀
    p = f.p
    u_ad = SparsityTracing.create_advec(U₀)
    du_ad = similar(u_ad)
    f.cap!(du_ad, u_ad, p, 0.0)
    Jad = SparsityTracing.jacobian(du_ad, length(du_ad))
    colors = matrix_colors(Jad)
    if on == :gpu
        U₀ = cu(f.U₀)
        p = NamedTuple((key => Float32(value) for (key, value) in pairs(f.p)))
        Jad = cu(Jad)
        colors = cu(colors)
    end
    cap_func = ODEFunction(f.cap!, jac_prototype=Jad, colorvec=colors, sparsity=Jad)
    hyp_func = ODEFunction(f.hyp!)
    return SplitODEProblem(cap_func, hyp_func, U₀, tspan, p, args...; kwargs...)
end

end