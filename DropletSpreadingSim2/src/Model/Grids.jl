module Grids

export gridded_to_flat, flat_to_gridded, vectorize_U!, matricize_Uvec!, pack_Uvec!, unpack_Uvec!

using FLoops
using ..Model: nᵤ


"""
gridded_to_flat(k, i, j; nᵤ, n₁, n₂)

convert indices from a cartesian grid to a flat grid
"""
gridded_to_flat(k, i, j; nᵤ, n₁, n₂) = ((k - 1) + (i - 1) * nᵤ + (j - 1) * nᵤ * n₁) + 1
gridded_to_flat(coords; nᵤ, n₁, n₂) = gridded_to_flat(coords...; nᵤ, n₁, n₂)

"""
flat_to_gridded(ix; nᵤ, n₁, n₂)

convert indices from a flat grid to a cartesian grid
"""
function flat_to_gridded(ix; nᵤ, n₁, n₂)
    k = (ix - 1) % nᵤ
    i = div(ix - 1, nᵤ) % n₁
    j = div(ix - 1, (nᵤ * n₁))
    return (k + 1, i + 1, j + 1)
end

function vectorize_U!(Uvec, U, n₁, n₂, i, j, k)
    Uvec[gridded_to_flat(k, i, j; nᵤ, n₁, n₂)] = U[i, j, k]
    return
end
function vectorize_U!(Uvec, U, n₁, n₂; executor=ThreadedEx())
    @floop executor for I in CartesianIndices((n₁, n₂, nᵤ))
        i, j, k = Tuple(I)
        vectorize_U!(Uvec, U, n₁, n₂, i, j, k)
    end
    return Uvec
end

function matricize_Uvec!(U, Uvec, n₁, n₂, i, j, k)
    U[i, j, k] = Uvec[gridded_to_flat(k, i, j; nᵤ, n₁, n₂)]
    return
end
function matricize_Uvec!(U, Uvec, n₁, n₂; executor=ThreadedEx())
    @floop executor for I in CartesianIndices((n₁, n₂, nᵤ))
        i, j, k = Tuple(I)
        matricize_Uvec!(U, Uvec, n₁, n₂, i, j, k)
    end
    return U
end

function pack_Uvec!(Uvec, h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, n₁, n₂, i, j)
    Uvec[gridded_to_flat(1, i, j; nᵤ, n₁, n₂)] = h[i, j]
    Uvec[gridded_to_flat(2, i, j; nᵤ, n₁, n₂)] = h[i, j] * ux[i, j]
    Uvec[gridded_to_flat(3, i, j; nᵤ, n₁, n₂)] = h[i, j] * uy[i, j]
    Uvec[gridded_to_flat(4, i, j; nᵤ, n₁, n₂)] = h[i, j] * vx[i, j]
    Uvec[gridded_to_flat(5, i, j; nᵤ, n₁, n₂)] = h[i, j] * vy[i, j]
    Uvec[gridded_to_flat(6, i, j; nᵤ, n₁, n₂)] = h[i, j] * ϕxx[i, j]
    Uvec[gridded_to_flat(7, i, j; nᵤ, n₁, n₂)] = h[i, j] * ϕxy[i, j]
    Uvec[gridded_to_flat(8, i, j; nᵤ, n₁, n₂)] = h[i, j] * ϕyy[i, j]
    return
end

function pack_Uvec!(Uvec, h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, n₁, n₂; executor=ThreadedEx())
    @floop executor for I in CartesianIndices((n₁, n₂))
        i, j = Tuple(I)
        pack_Uvec!(Uvec, h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, n₁, n₂, i, j)
    end
    return Uvec
end

function unpack_Uvec!(h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, Uvec, n₁, n₂, i, j)
    h[i, j] = Uvec[gridded_to_flat(1, i, j; nᵤ, n₁, n₂)]
    ux[i, j] = Uvec[gridded_to_flat(2, i, j; nᵤ, n₁, n₂)] / h[i, j]
    uy[i, j] = Uvec[gridded_to_flat(3, i, j; nᵤ, n₁, n₂)] / h[i, j]
    vx[i, j] = Uvec[gridded_to_flat(4, i, j; nᵤ, n₁, n₂)] / h[i, j]
    vy[i, j] = Uvec[gridded_to_flat(5, i, j; nᵤ, n₁, n₂)] / h[i, j]
    ϕxx[i, j] = Uvec[gridded_to_flat(6, i, j; nᵤ, n₁, n₂)] / h[i, j]
    ϕxy[i, j] = Uvec[gridded_to_flat(7, i, j; nᵤ, n₁, n₂)] / h[i, j]
    ϕyy[i, j] = Uvec[gridded_to_flat(8, i, j; nᵤ, n₁, n₂)] / h[i, j]
    return
end

function unpack_Uvec!(h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, Uvec, n₁, n₂; executor=ThreadedEx())
    @floop executor for I in CartesianIndices((n₁, n₂))
        i, j = Tuple(I)
        unpack_Uvec!(h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy, Uvec, n₁, n₂, i, j)
    end
    return h, ux, uy, vx, vy, ϕxx, ϕxy, ϕyy
end

end