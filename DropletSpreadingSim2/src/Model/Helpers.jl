module Helpers

export gridded_to_flat, flat_to_gridded, dict2ntuple
export @preallocate, @bc, @ntuple

using MacroTools
using MacroTools: @capture, postwalk, splitdef, combinedef

"""
    @ntuple vars...
Create a `NamedTuple` out of the given variables that has as keys the variable
names and as values their values.
## Examples
```julia
julia> ω = 5; χ = "test"; ζ = 3.14;
julia> @ntuple ω χ ζ
(ω = 5, χ = "test", ζ = 3.14)
```
"""
macro ntuple(vars...)
   args = Any[]
   for i in 1:length(vars)
       push!(args, Expr(:(=), esc(vars[i]), :($(esc(vars[i])))))
   end
   expr = Expr(:tuple, args...)
   return expr
end


"""
    dict2ntuple(dict) -> ntuple
Convert a dictionary (with `Symbol` or `String` as key type) to
a `NamedTuple`.
"""
function dict2ntuple(dict::Dict{String,T}) where T
    NamedTuple{Tuple(Symbol.(keys(dict)))}(values(dict))
end
function dict2ntuple(dict::Dict{Symbol,T}) where T
    NamedTuple{Tuple(keys(dict))}(values(dict))
end

"""
    @preallocate caches... = template

build a preallocated array for each of the `caches` variable.

# Examples
```julia-repl
julia> @preallocate A, B = zeros(50, 50)
```
"""
macro preallocate(expr)
    expr.head != :(=) && error("Expression needs to be of form `a, b = c`")
    items, template = expr.args
    items = isa(items, Symbol) ? [items] : items.args
    kd = [:( $key = $template) for key in items]
    kd_namedtuple = :(NamedTuple{Tuple($items)}(Tuple([$template for _ in $items])))
    kdblock = Expr(:block, kd...)
    expr = quote
        $kdblock
    $kd_namedtuple
    end
    return esc(expr)
end


"""
    @bc var, expr

Deal with boundary on the expression. Transform each var[i + ix, j + jx] with
var[bc(i + ix, n₁), bc(j + jx, n₂)], bc being defined in the Model source.

For a periodic bc, you can use `bc(i, n) = mod1(i, n)` and for a 0 flux boundary,
`bc(i, n) = clamp(i, 1, n)`.

# Examples
```julia-repl
julia> @bc h (h[i - 1, j] + h[i + 1, j]) / Δx
```
will lead to (h[bc(i - 1), j] + h[bc(i + 1), j]) / Δx
"""
macro bc(var, expr)
    bound_ij(i, j) = :(mod1($i, n₁)), :(mod1($j, n₂))
    function eval_var(var, ex)
        ex = postwalk(ex) do x
            @capture(x, $var[i_, j_]) || return x
            i, j = bound_ij(i, j)
            return :($var[$i, $j])
            end
        ex = postwalk(ex) do x
            @capture(x, $var[i_, j_, k_]) || return x
            i, j = bound_ij(i, j)
        return :($var[$i, $j, $k])
            end
        ex = postwalk(ex) do x
            @capture(x, $var(i_, j_)) || return x
            i, j = bound_ij(i, j)
            return :($var($i, $j))
    end
        ex
    end

    if (typeof(var) != Symbol) && (var.head == :tuple)
        for var in var.args
            expr = eval_var(var, expr)
        end
    else
        expr = eval_var(var, expr)
    end
    return esc(expr)
end

end