module NonConservative
export update_cap!
using StaticArrays, UnPack, FLoops
using FoldsCUDA, CUDA
using ..Ops
using ..Grids
using ..Model: MODE, nᵤ

function unpack_hu!(hux, huy, Uvec, n₁, n₂, i, j)
    hux[i, j] = Uvec[gridded_to_flat(2, i, j; nᵤ, n₁, n₂)]
    huy[i, j] = Uvec[gridded_to_flat(3, i, j; nᵤ, n₁, n₂)]
    return
end

function unpack_hu!(hux, huy, Uvec, n₁, n₂; executor=ThreadedEx())
    @floop executor for I in CartesianIndices((n₁, n₂))
        unpack_hu!(hux, huy, Uvec, n₁, n₂, Tuple(I)...)
    end
    return hux, huy
end

function compute_skew_cap_coeffs!(
    fxx, fxy, fyy, gx, gy, gv, fvx, fvy, Pid,
    h, vx, vy, κ, θₐ, θᵣ, hₛ,
    hux, huy, ux, uy, τx, τy,
    Δx, Δy, n₁, n₂, i, j
)
    @static if MODE == :full
        fxx[i, j] = √κ * √h[i, j] * 1 / √(1 + h[i, j] / 4κ * (vx[i, j]^2 + vy[i, j]^2)) * (1 - 1 / (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2)) * h[i, j] / 4κ * (vx[i, j]^2))
        fxy[i, j] = √κ * √h[i, j] * 1 / √(1 + h[i, j] * 1 / 4κ * (vx[i, j]^2 + vy[i, j]^2)) * (-1 / (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2)) * h[i, j] / 4κ * (vx[i, j] * vy[i, j]))
        fyy[i, j] = √κ * √h[i, j] * 1 / √(1 + h[i, j] * 1 / 4κ * (vx[i, j]^2 + vy[i, j]^2)) * (1 - 1 / (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2)) * h[i, j] / 4κ * (vy[i, j]^2))
        gx[i, j] = h[i, j] * vx[i, j] / 2 * (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2))^(-1)
        gy[i, j] = h[i, j] * vy[i, j] / 2 * (1 + h[i, j] / 2κ * (vx[i, j]^2 + vy[i, j]^2))^(-1)
    else
        fxx[i, j] = fyy[i, j] = √κ * √h[i, j]
        fxy[i, j] = 0.0
        gx[i, j] = h[i, j] * vx[i, j] / 2
        gy[i, j] = h[i, j] * vy[i, j] / 2
    end
    v = @SVector [vx[i, j], vy[i, j]]
    g = @SVector [gx[i, j], gy[i, j]]
    f = @SMatrix(
        [
            fxx[i, j] fxy[i, j]
            fxy[i, j] fyy[i, j]
        ]
    )
    u = @SVector [ux[i, j], uy[i, j]]
    τ = @SVector [τx, τy]

    gv[i, j] = g' * v
    fv = f * v

    dej = (hₛ / h[i, j])^4 - (hₛ / h[i, j])^3
    ε = 1.e-3
    θₛ = 0.5 * (θₐ + θᵣ) + 0.5 * (θᵣ - θₐ) * tanh((@div(hux, huy)) / ε)

    fvx[i, j] = fv[1]
    fvy[i, j] = fv[2]
    Pid[i, j] = (6 / hₛ) * κ * (1 - cos(θₛ)) * dej
    return
end

function compute_skew_cap_coeffs!(
    fxx, fxy, fyy, gx, gy, gv, fvx, fvy, Pid,
    h, vx, vy, κ, θₐ, θᵣ, hₛ,
    hux, huy, ux, uy, τx, τy,
    Δx, Δy, n₁, n₂; executor=ThreadedEx()
)
    @floop executor for I in CartesianIndices(h)
        i, j = Tuple(I)
        compute_skew_cap_coeffs!(
            fxx, fxy, fyy, gx, gy, gv, fvx, fvy, Pid,
            h, vx, vy, κ, θₐ, θᵣ, hₛ,
            hux, huy, ux, uy, τx, τy,
            Δx, Δy, n₁, n₂, i, j
        )
    end
end

function skew_cap_kernel!(
    dU, h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy,
    gx, gy, fxx, fxy, fyy, gv, fvx, fvy, Pid,
    Re, β, τx, τy,
    Δx, Δy, n₁, n₂, i, j
)

    g = @SVector [gx[i, j], gy[i, j]]
    f = @SMatrix [fxx[i, j] fxy[i, j]
        fxy[i, j] fyy[i, j]]
    ϕ = @SMatrix [ϕxx[i, j] ϕxy[i, j]
        ϕxy[i, j] ϕyy[i, j]]
    u = @SVector [ux[i, j], uy[i, j]]
    v = @SVector [vx[i, j], vy[i, j]]
    τ = @SVector [τx, τy]

    dhu = -(@∇(gv)) + (@divh∇(fvx, fvy)) + 3 / Re * (τ / 2 - u / h[i, j]) + h[i, j] * (@∇(Pid))
    dhv = @SVector [vx[i, j], vy[i, j]]
    dhv = -g * (@div(ux, uy)) - f * (@divh∇(ux, uy))


#          equation d'origine   

#     dhϕ = (
#         2h[i, j] * (@div(ux, uy)) * ϕ - (@∇(ux, uy)) * h[i, j] * ϕ - h[i, j] * ϕ * (@∇(ux, uy))'
#         -
#         β / Re / h[i, j] * (
#    ϕ - (u ⊗ u) / (3h[i, j]^2) + 1 / (12h[i, j]^2) * ((u ⊗ u) - h[i, j]^2 / 4 * (τ ⊗ τ))
#         )
#     )

#          pour dhPhi   avec phi=0

#     dhϕ =  - β / Re / h[i, j] * (
#             ϕ #ligne modifiee par Mouloud pour test quand phi=0.0
#         )

#          pour dh^3Phi   avec ϕ=h^2Phi

    dhϕ = (
        -h[i, j] * ϕ * (@∇(ux, uy))   - h[i, j] * ϕ * (@∇(ux, uy))'
        -
        β / Re / h[i, j] * (
   ϕ - (u ⊗ u) / 3 + 1 / 12 * ((u ⊗ u) - h[i, j]^2 / 4 * (τ ⊗ τ))
        )
    )

    dU[gridded_to_flat(1, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(2, i, j; nᵤ, n₁, n₂)] = dhu[1]
    dU[gridded_to_flat(3, i, j; nᵤ, n₁, n₂)] = dhu[2]
    dU[gridded_to_flat(4, i, j; nᵤ, n₁, n₂)] = dhv[1]
    dU[gridded_to_flat(5, i, j; nᵤ, n₁, n₂)] = dhv[2]
    dU[gridded_to_flat(6, i, j; nᵤ, n₁, n₂)] = dhϕ[1, 1]
    dU[gridded_to_flat(7, i, j; nᵤ, n₁, n₂)] = dhϕ[1, 2]
    dU[gridded_to_flat(8, i, j; nᵤ, n₁, n₂)] = dhϕ[2, 2]

    return
end

function skew_cap_kernel!(
    dU, h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy,
    gx, gy, fxx, fxy, fyy, gv, fvx, fvy, Pid,
    Re, β, τx, τy,
    Δx, Δy, n₁, n₂; executor=ThreadedEx(),
)
    @floop executor for I in CartesianIndices(h)
        i, j = Tuple(I)
        skew_cap_kernel!(
            dU, h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy,
            gx, gy, fxx, fxy, fyy, gv, fvx, fvy, Pid,
            Re, β, τx, τy,
            Δx, Δy, n₁, n₂, i, j
        )
    end
end

function update_cap!(dUvec, Uvec, p, t; gridinfo, caches, executor=:auto)
    typed_caches = caches[typeof(Uvec)]
    cache_cap = typed_caches.cap
    @unpack h, hux, huy, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy = cache_cap
    @unpack fxx, fxy, fyy, gv, fvx, fvy, gx, gy, Pid = cache_cap
    @unpack Δx, Δy, n₁, n₂ = gridinfo
    @unpack κ, Re, β, τx, τy, θₐ, θᵣ, hₛ = p

    if executor == :auto
        executor = Uvec isa CuArray ? CUDAEx() : ThreadedEx()
    end

    unpack_Uvec!(h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, Uvec, n₁, n₂; executor)
    unpack_hu!(hux, huy, Uvec, n₁, n₂; executor)

    compute_skew_cap_coeffs!(
        fxx, fxy, fyy, gx, gy, gv, fvx, fvy, Pid, h,
        vx, vy, κ, θₐ, θᵣ, hₛ, hux, huy, ux, uy, τx, τy,
        Δx, Δy, n₁, n₂;
        executor
    )

    skew_cap_kernel!(
        dUvec, h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy,
        gx, gy, fxx, fxy, fyy, gv, fvx, fvy, Pid,
        Re, β, τx, τy,
        Δx, Δy, n₁, n₂;
        executor
    )

    return dUvec
end
end
