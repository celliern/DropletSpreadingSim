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
    fx1, fx2, fx3, fy1, fy2, fy3, gx, gy, gv, fvx, fvy,
    h, vx, vy, κ, θₐ, θᵣ, hₛ,
    hux, huy, ux, uy, τx, τy,
    Δx, Δy, n₁, n₂, i, j
)
#=    @static if MODE == :full
        fx1[i, j] = 0.0
        fx2[i, j] = 0.0
        fx3[i, j] = 0.0
        fy1[i, j] = 0.0
        fy2[i, j] = 0.0
        fy3[i, j] = 0.0
        gx[i, j] = 0.0
        gy[i, j] = 0.0
        fxx[i, j] = fyy[i, j] = 0.0
        fxy[i, j] = 0.0
    elseif MODE == :simple
        fxx[i, j] = fyy[i, j] = √κ * √h[i, j]
        fxy[i, j] = 0.0
        gx[i, j] = h[i, j] * vx[i, j] / 2
        gy[i, j] = h[i, j] * vy[i, j] / 2
    else
#       # fxx[i, j] = fyy[i, j] = 0.0
#        fxy[i, j] = 0.0
        gx[i, j] = 0.0
        gy[i, j] = 0.0
    end
    v = @SVector [vx[i, j], vy[i, j]]
    g = @SVector [gx[i, j], gy[i, j]]
    f = @SMatrix(
        [
            fx1[i, j] fx2[i, j] fx3[i, j]
            fy1[i, j] fy2[i, j] fy3[i, j]
        ]
    )
    u = @SVector [ux[i, j], uy[i, j]]
    τ = 0.0

    gv[i, j] = g' * v
    fv = f * v

#    dej = (hₛ / h[i, j])^4 - (hₛ / h[i, j])^3
#    ε = 1.e-3
    θₛ = 0.0

    fvx[i, j] = fv[1]
    fvy[i, j] = fv[2]=#
    
    return
end

function compute_skew_cap_coeffs!(
    fx1, fx2, fx3, fy1, fy2, fy3, gx, gy, gv, fvx, fvy,
    h, vx, vy, κ, θₐ, θᵣ, hₛ,
    hux, huy, ux, uy, τx, τy,
    Δx, Δy, n₁, n₂; executor=ThreadedEx()
)
    @floop executor for I in CartesianIndices(h)
        i, j = Tuple(I)
        compute_skew_cap_coeffs!(
            fx1, fx2, fx3, fy1, fy2, fy3, gx, gy, gv, fvx, fvy,
            h, vx, vy, κ, θₐ, θᵣ, hₛ,
            hux, huy, ux, uy, τx, τy,
            Δx, Δy, n₁, n₂, i, j
        )
    end
end

function skew_cap_kernel!(
    dU, h, ux, uy, vx, vy, ϕx1, ϕx2, ϕx1, ϕy1, ϕy2, ϕy3,
    gx, gy, fx1, fx2, fx3, fy1, fy2, fy3, gv, fvx, fvy,
    Re, β, τx, τy,
    Δx, Δy, n₁, n₂, i, j
)

#=    g = @SVector [gx[i, j], gy[i, j]]
    f = @SMatrix [fx1[i, j] fx2[i, j] fx3[i, j]
        fy1[i, j] fy2[i, j] fy2[i, j]]
    ϕ = @SMatrix [ϕx1[i, j] ϕx2[i, j] ϕx3[i, j]
        ϕy1[i, j] ϕy2[i, j]  ϕy3[i, j]]
    u = @SVector [ux[i, j], uy[i, j]]
    v = @SVector [vx[i, j], vy[i, j]]
    τ = @SVector [τx, τy]

    dhu = @SVector [0, 0]
    dhv = @SVector [0, 0]
#    dhv = 0.0
    dhϕ =  [
        0 0 0
        0 0 0 
    ]=#

    dU[gridded_to_flat(1, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(2, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(3, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(4, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(5, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(6, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(7, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(8, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(9, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(10, i, j; nᵤ, n₁, n₂)] = 0.0
    dU[gridded_to_flat(11, i, j; nᵤ, n₁, n₂)] = 0.0

    return
end

function skew_cap_kernel!(
    dU, h, ux, uy, vx, vy, ϕx1, ϕx2, ϕx1, ϕy1, ϕy2, ϕy3,
    gx, gy, fx1, fx2, fx3, fy1, fy2, fy3, gv, fvx, fvy,
    Re, β, τx, τy,
    Δx, Δy, n₁, n₂; executor=ThreadedEx(),
)
    @floop executor for I in CartesianIndices(h)
        i, j = Tuple(I)
        skew_cap_kernel!(
            dU, h, ux, uy, vx, vy, ϕx1, ϕx2, ϕx1, ϕy1, ϕy2, ϕy3,
            gx, gy, fx1, fx2, fx3, fy1, fy2, fy3, gv, fvx, fvy,
            Re, β, τx, τy,
            Δx, Δy, n₁, n₂, i, j
        )
    end
end

function update_cap!(dUvec, Uvec, p, t; gridinfo, caches, executor=:auto)
    typed_caches = caches[typeof(Uvec)]
    cache_cap = typed_caches.cap
    @unpack h, hux, huy, ux, uy, vx, vy, ϕx1, ϕx2, ϕx1, ϕy1, ϕy2, ϕy3 = cache_cap
    @unpack fx1, fx2, fx3, fy1, fy2, fy3, gv, fvx, fvy, gx, gy = cache_cap
    @unpack Δx, Δy, n₁, n₂ = gridinfo
    @unpack κ, Re, β, τx, τy, θₐ, θᵣ, hₛ = p

    if executor == :auto
        executor = Uvec isa CuArray ? CUDAEx() : ThreadedEx()
    end

    unpack_Uvec!(h, ux, uy, vx, vy, ϕx1, ϕx2, ϕx1, ϕy1, ϕy2, ϕy3, Uvec, n₁, n₂; executor)
    unpack_hu!(hux, huy, Uvec, n₁, n₂; executor)

    compute_skew_cap_coeffs!(
        fx1, fx2, fx3, fy1, fy2, fy3, gx, gy, gv, fvx, fvy, h,
        vx, vy, κ, θₐ, θᵣ, hₛ, hux, huy, ux, uy, τx, τy,
        Δx, Δy, n₁, n₂;
        executor
    )

    skew_cap_kernel!(
        dUvec, h, ux, uy, vx, vy, ϕx1, ϕx2, ϕx1, ϕy1, ϕy2, ϕy3,
        gx, gy, fx1, fx2, fx3, fy1, fy2, fy3, gv, fvx, fvy, 
        Re, β, τx, τy,
        Δx, Δy, n₁, n₂;
        executor
    )

    return dUvec
end
end