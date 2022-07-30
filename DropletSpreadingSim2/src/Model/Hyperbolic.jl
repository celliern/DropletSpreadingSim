module Hyperbolic
export update_hyp!

using UnPack, FLoops
using ..Helpers
using ..Grids
using ..Model: MODE, nᵤ

@inline minmod(x, y) = 0.5 * (sign(x) + sign(y)) * min(abs(x), abs(y))

@bc U function update_bounds_x!(Uw₋, Uw₊, Ue₋, Ue₊, U, n₁, n₂, Δx, Δy, i, j, k)
    ∇Ui = minmod((U[i, j, k] - U[i-1, j, k]) / Δx, (U[i+1, j, k] - U[i, j, k]) / Δx)
    ∇Ue = minmod((U[i+1, j, k] - U[i, j, k]) / Δx, (U[i+2, j, k] - U[i+1, j, k]) / Δx)
    ∇Uw = minmod((U[i-1, j, k] - U[i-2, j, k]) / Δx, (U[i, j, k] - U[i-1, j, k]) / Δx)

    Ue₋[i, j, k] = U[i, j, k] + Δx / 2 * ∇Ui
    Ue₊[i, j, k] = U[i+1, j, k] - Δx / 2 * ∇Ue

    Uw₋[i, j, k] = U[i-1, j, k] + Δx / 2 * ∇Uw
    Uw₊[i, j, k] = U[i, j, k] - Δx / 2 * ∇Ui
    return
end

@bc U function update_bounds_y!(Us₋, Us₊, Un₋, Un₊, U, n₁, n₂, Δx, Δy, i, j, k)
    ∇Ui = minmod((U[i, j, k] - U[i, j-1, k]) / Δy, (U[i, j+1, k] - U[i, j, k]) / Δy)
    ∇Un = minmod((U[i, j+1, k] - U[i, j, k]) / Δy, (U[i, j+2, k] - U[i, j+1, k]) / Δy)
    ∇Us = minmod((U[i, j-1, k] - U[i, j-2, k]) / Δy, (U[i, j, k] - U[i, j-1, k]) / Δy)

    Un₋[i, j, k] = U[i, j, k] + Δy / 2 * ∇Ui
    Un₊[i, j, k] = U[i, j+1, k] - Δy / 2 * ∇Un

    Us₋[i, j, k] = U[i, j-1, k] + Δy / 2 * ∇Us
    Us₊[i, j, k] = U[i, j, k] - Δy / 2 * ∇Ui
    return
end

function compute_caF_x!(c, a, F, U, i, j)
    c[i, j] = U[i, j, 2] / U[i, j, 1]
    a[i, j] = √(3U[i, j, 1]) * √(max(U[i, j, 6], 0))
    for k in 1:nᵤ
        F[i, j, k] = c[i, j] * U[i, j, k]
    end
    F[i, j, 2] = U[i, j, 1]^2 * U[i, j, 6]
    F[i, j, 3] = U[i, j, 1]^2 * U[i, j, 7]
    return
end

function compute_caF_y!(c, a, F, U, i, j)
    c[i, j] = U[i, j, 3] / U[i, j, 1]
    a[i, j] = √(3U[i, j, 1]) * √(max(U[i, j, 8], 0))
    for k in 1:nᵤ
        F[i, j, k] = c[i, j] * U[i, j, k]
    end
    F[i, j, 2] = U[i, j, 1]^2 * U[i, j, 7]
    F[i, j, 3] = U[i, j, 1]^2 * U[i, j, 8]
    return
end

@inline ps(cₗ, cᵣ, aₗ, aᵣ) = max(abs(cₗ) + aₗ, abs(cᵣ) + aᵣ)

function compute_boundaries_flux!(f, U₊, U₋, c₊, c₋, a₊, a₋, F₊, F₋, i, j, k)
    f[i, j, k] = 0.5 * ((F₊[i, j, k] + F₋[i, j, k]) - ps(c₊[i, j], c₋[i, j], a₊[i, j], a₋[i, j]) * (U₊[i, j, k] - U₋[i, j, k]))
    return
end

function compute_flux_balance!(F, f1, f2, Δx, i, j, k)
    F[i, j, k] = (f2[i, j, k] - f1[i, j, k]) / Δx
    return
end

function update_hyp_x!(dUvec, U, p, t; gridinfo, cache_hyp) where {T}
    @unpack Fx = cache_hyp
    @unpack Ue₋, Ue₊, Uw₋, Uw₊ = cache_hyp
    @unpack ce₋, ce₊, cw₋, cw₊ = cache_hyp
    @unpack ae₋, ae₊, aw₋, aw₊ = cache_hyp
    @unpack Fe₋, Fe₊, Fw₋, Fw₊ = cache_hyp
    @unpack fe, fw = cache_hyp
    @unpack Δx, Δy, n₁, n₂ = gridinfo

    @floop for I in CartesianIndices((n₁, n₂, nᵤ))
        update_bounds_x!(Uw₋, Uw₊, Ue₋, Ue₊, U, n₁, n₂, Δx, Δy, Tuple(I)...)
    end

    @floop for I in CartesianIndices((n₁, n₂))
        i, j = Tuple(I)
        compute_caF_x!(cw₋, aw₋, Fw₋, Uw₋, i, j)
        compute_caF_x!(cw₊, aw₊, Fw₊, Uw₊, i, j)
        compute_caF_x!(ce₋, ae₋, Fe₋, Ue₋, i, j)
        compute_caF_x!(ce₊, ae₊, Fe₊, Ue₊, i, j)
        for k in 1:nᵤ
            compute_boundaries_flux!(fw, Uw₊, Uw₋, cw₊, cw₋, aw₊, aw₋, Fw₊, Fw₋, i, j, k)
            compute_boundaries_flux!(fe, Ue₊, Ue₋, ce₊, ce₋, ae₊, ae₋, Fe₊, Fe₋, i, j, k)
            compute_flux_balance!(Fx, fe, fw, Δx, i, j, k)
            vectorize_U!(dUvec, Fx, n₁, n₂, i, j, k)
        end
    end
    return dUvec
end

function update_hyp_y!(dUvec, U, p, t; gridinfo, cache_hyp)
    @unpack Fy = cache_hyp
    @unpack Un₋, Un₊, Us₋, Us₊ = cache_hyp
    @unpack cn₋, cn₊, cs₋, cs₊ = cache_hyp
    @unpack an₋, an₊, as₋, as₊ = cache_hyp
    @unpack Fn₋, Fn₊, Fs₋, Fs₊ = cache_hyp
    @unpack fn, fs = cache_hyp
    @unpack Δx, Δy, n₁, n₂ = gridinfo

    @floop for I in CartesianIndices((n₁, n₂, nᵤ))
        update_bounds_y!(Us₋, Us₊, Un₋, Un₊, U, n₁, n₂, Δx, Δy, Tuple(I)...)
    end

    @floop for I in CartesianIndices((n₁, n₂))
        i, j = Tuple(I)
        compute_caF_y!(cs₋, as₋, Fs₋, Us₋, i, j)
        compute_caF_y!(cs₊, as₊, Fs₊, Us₊, i, j)
        compute_caF_y!(cn₋, an₋, Fn₋, Un₋, i, j)
        compute_caF_y!(cn₊, an₊, Fn₊, Un₊, i, j)
        for k in 1:nᵤ
            compute_boundaries_flux!(fs, Us₊, Us₋, cs₊, cs₋, as₊, as₋, Fs₊, Fs₋, i, j, k)
            compute_boundaries_flux!(fn, Un₊, Un₋, cn₊, cn₋, an₊, an₋, Fn₊, Fn₋, i, j, k)
            compute_flux_balance!(Fy, fn, fs, Δy, i, j, k)
            vectorize_U!(dUvec, Fy, n₁, n₂, i, j, k)
        end
    end
    return dUvec
end

function update_hyp!(dUvec, Uvec, p, t; gridinfo, cache_hyp)
    @unpack dUhypx, dUhypy, U = cache_hyp
    @unpack Δx, Δy, n₁, n₂ = gridinfo

    matricize_Uvec!(U, Uvec, n₁, n₂)

    update_hyp_x!(dUhypx, U, p, t; gridinfo, cache_hyp)
    update_hyp_y!(dUhypy, U, p, t; gridinfo, cache_hyp)
    dUvec .= oftype(Uvec, dUhypx .+ dUhypy)
end

end