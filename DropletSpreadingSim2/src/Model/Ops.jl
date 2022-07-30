module Ops
export @dx, @dy, @∇, @div, @divh∇, ⊗

using ..Helpers
using StaticArrays

@inline @bc m dx(m, Δx, Δy, n₁, n₂, i, j) = (m[i+1, j] - m[i-1, j]) / 2Δx
@inline @bc m dy(m, Δx, Δy, n₁, n₂, i, j) = (m[i, j+1] - m[i, j-1]) / 2Δy
macro dx(m)
    esc(:(Ops.dx($m, Δx, Δy, n₁, n₂, i, j)))
end
macro dy(m)
    esc(:(Ops.dy($m, Δx, Δy, n₁, n₂, i, j)))
end

@inline ∇(m, Δx, Δy, n₁, n₂, i, j) = @SVector [(@dx(m)), (@dy(m))]
@inline ∇(mx, my, Δx, Δy, n₁, n₂, i, j) = @SMatrix [(@dx(mx)) (@dx(my))
    (@dy(mx)) (@dy(my))]
macro ∇(m)
    esc(:(Ops.∇($m, Δx, Δy, n₁, n₂, i, j)))
end
macro ∇(mx, my)
    esc(:(Ops.∇($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end

@inline @bc (mx, mx) div(mx, my, Δx, Δy, n₁, n₂, i, j) = (@dx(mx)) + (@dy(my))
@inline @bc (mxx, mxy, myy) div(mxx, mxy, myy, Δx, Δy, n₁, n₂, i, j) = @SVector[(@dx(mxx)) + (@dy(mxy)), (@dx(mxy)) + (@dy(myy))]
macro div(mx, my)
    esc(:(Ops.div($mx, $my, Δx, Δy, n₁, n₂, i, j)))
end
macro div(mxx, mxy, myy)
    esc(:(Ops.div($mxx, $mxy, $myy, Δx, Δy, n₁, n₂, i, j)))
end

@inline @bc (h, mx, my) function divh∇(h, mx, my, Δx, Δy, n₁, n₂, i, j)
    ∂x⁰_h∂x_mx = (1 / 2 * (h[i+1, j] + h[i, j]) * (mx[i+1, j] - mx[i, j]) - 1 / 2 * (h[i-1, j] + h[i, j]) * (mx[i, j] - mx[i-1, j])) / Δx^2
    ∂y⁰⁰_h∂x⁰⁰_my = (h[i, j+1] * (my[i+1, j+1] - my[i-1, j+1]) - h[i, j-1] * (my[i+1, j-1] - my[i-1, j-1])) / (2Δx * 2Δy)
    ∂x⁰⁰_h∂y⁰⁰_mx = (h[i+1, j] * (mx[i+1, j+1] - mx[i+1, j-1]) - h[i-1, j] * (mx[i-1, j+1] - mx[i-1, j-1])) / (2Δx * 2Δy)
    ∂y⁰_h∂y_my = (1 / 2 * (h[i, j+1] + h[i, j]) * (my[i, j+1] - my[i, j]) - 1 / 2 * (h[i, j-1] + h[i, j]) * (my[i, j] - my[i, j-1])) / Δy^2
    return @SVector [∂x⁰_h∂x_mx + ∂y⁰⁰_h∂x⁰⁰_my, ∂x⁰⁰_h∂y⁰⁰_mx + ∂y⁰_h∂y_my]
end

macro divh∇(mx, my)
    esc(:(Ops.divh∇(h, $mx, $my, Δx, Δy, n₁, n₂, i, j)))
end

⊗(a::AbstractVector, b::AbstractVector) = a * b'
end