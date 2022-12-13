module Model
export update_cap!, update_hyp!, compute_v!, compute_ϕ!, build_cache, build_cache_cap, build_cache_hyp, MODE, nᵤ
using SparseArrays, StaticArrays, LinearAlgebra, UnPack, Reexport
using UnPack

const nᵤ = 8
const MODE = :full


include("./Helpers.jl")
include("./Grids.jl")
include("./Ops.jl")
include("./Hyperbolic.jl")
include("./NonConservative.jl")

@reexport using .Helpers
@reexport using .Grids
@reexport using .Ops
@reexport using .Hyperbolic
@reexport using .NonConservative

@static if MODE == :full
    function compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂, i, j)
        vx[i, j] = √κ * √(2.0 / (1.0 + √(1.0 + (@dx(h))^2 + (@dy(h))^2))) * (@dx(h)) / √max(h[i, j], 0.0)
        vy[i, j] = √κ * √(2.0 / (1.0 + √(1.0 + (@dx(h))^2 + (@dy(h))^2))) * (@dy(h)) / √max(h[i, j], 0.0)
        return
    end
else
    function compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂, i, j)
        vx[i, j] = @. √κ * (@dx(h)) / √max(h[i, j], 0.0)
        vy[i, j] = @. √κ * (@dy(h)) / √max(h[i, j], 0.0)
        return
    end
end

function compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂)
    for I in CartesianIndices((n₁, n₂))
        compute_v!(vx, vy, h, κ, Δx, Δy, n₁, n₂, Tuple(I)...)
    end
end

function compute_ϕ!(h, ux, uy, ϕxx, ϕxy, ϕyy, τx, τy, i, j)
    u = @SVector [ux[i, j], uy[i, j]]
    τ = @SVector [τx, τy]
    ϕ = (u ⊗ u) / 3h[i, j]^2 - 1 / 12h[i, j]^2 * ((u ⊗ u) - h[i, j]^2 * (τ ⊗ τ) / 4)

    ϕxx[i, j] = ϕ[1, 1]
    ϕxy[i, j] = ϕ[1, 2]
    ϕyy[i, j] = ϕ[2, 2]
    return
end

function compute_ϕ!(h, ux, uy, ϕxx, ϕxy, ϕyy, τx, τy)
    for I in CartesianIndices(h)
        compute_ϕ!(h, ux, uy, ϕxx, ϕxy, ϕyy, τx, τy, Tuple(I)...)
    end
end

function build_cache_hyp(T, n₁, n₂)
    @preallocate U, Fx, Fy = zeros(T, n₁, n₂, nᵤ)
    @preallocate dUhypx, dUhypy = zeros(T, n₁ * n₂ * nᵤ)

    @preallocate Ue₋, Ue₊, Uw₋, Uw₊, Us₋, Us₊, Un₋, Un₊ = zeros(T, n₁, n₂, nᵤ)
    @preallocate ce₋, ce₊, cw₋, cw₊, cs₋, cs₊, cn₋, cn₊ = zeros(T, n₁, n₂)
    @preallocate ae₋, ae₊, aw₋, aw₊, as₋, as₊, an₋, an₊ = zeros(T, n₁, n₂)
    @preallocate Fe₋, Fe₊, Fw₋, Fw₊, Fs₋, Fs₊, Fn₋, Fn₊ = zeros(T, n₁, n₂, nᵤ)
    @preallocate fe, fw, fs, fn = zeros(T, n₁, n₂, nᵤ)

    return @ntuple U Fx Fy dUhypx dUhypy Ue₋ Ue₊ Uw₋ Uw₊ Us₋ Us₊ Un₋ Un₊ ce₋ ce₊ cw₋ cw₊ cs₋ cs₊ cn₋ cn₊ ae₋ ae₊ aw₋ aw₊ as₋ as₊ an₋ an₊ Fe₋ Fe₊ Fw₋ Fw₊ Fs₋ Fs₊ Fn₋ Fn₊ fe fw fs fn
end

function build_cache_cap(T, n₁, n₂)
    @preallocate h, hux, huy, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy = zeros(T, n₁, n₂)
    @preallocate fxx, fxy, fyy, gv, fvx, fvy, gx, gy, Pid = zeros(T, n₁, n₂)

    return @ntuple h hux huy ux uy vx vy ϕxx ϕxy ϕyy fxx fxy fyy gv fvx fvy gx gy Pid
end

build_cache(T, n₁, n₂) = (cap=build_cache_cap(T, n₁, n₂), hyp=build_cache_hyp(T, n₁, n₂))

build_cache_hyp(n₁, n₂) = build_cache_hyp(Float64, n₁, n₂)
build_cache_cap(n₁, n₂) = build_cache_cap(Float64, n₁, n₂)
build_cache(n₁, n₂) = build_cache(Float64, n₁, n₂)

end