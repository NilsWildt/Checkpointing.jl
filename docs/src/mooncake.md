# Mooncake.jl Backend

Checkpointing.jl now supports [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) as an alternative AD backend to Enzyme.jl.

## Why Mooncake.jl?

- **Pure Julia**: No LLVM dependency, easier installation
- **Automatic Tangent Derivation**: Works out of the box for most structs
- **Growing Ecosystem**: Active development and improving performance

## Quick Start

```julia
using Checkpointing
using Mooncake

# Define your model - NO custom tangent types needed for simple structs!
mutable struct MyModel
    x::Vector{Float64}
    y::Vector{Float64}
    scale::Float64
end

function my_loss(model, scheme, tsteps)
    @ad_checkpoint scheme for i = 1:tsteps
        # Your loop body here
        model.x .+= model.scale .* model.y
    end
    return sum(model.x)
end

# Use Mooncake's native API
model = MyModel([1.0, 2.0], [0.1, 0.2], 0.5)
scheme = Revolve(4)

cache = Mooncake.prepare_gradient_cache(my_loss, model, scheme, 100)
val, grads = Mooncake.value_and_gradient!!(cache, my_loss, model, scheme, 100)

println("Loss: $val")
println("Gradient: $(grads[2])")  # grads[2] contains gradient w.r.t. model
```

## What Works Automatically

Mooncake.jl automatically derives tangent types for:

✅ **Simple mutable structs** with arrays and scalars
```julia
mutable struct Simple
    data::Vector{Float64}
    param::Float64
end  # Works out of the box!
```

✅ **Nested structs**
```julia
mutable struct Inner
    x::Vector{Float64}
end

mutable struct Outer
    inner::Inner
    bias::Vector{Float64}
end  # Also works!
```

✅ **Multiple array fields**
```julia
mutable struct MultiArray
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
end  # Works!
```

✅ **Mixed field types** (differentiable + non-differentiable)
```julia
mutable struct Mixed
    data::Vector{Float64}  # Differentiable
    param::Float64          # Differentiable
    count::Int              # Non-differentiable (handled automatically)
    name::String            # Non-differentiable (handled automatically)
end  # Works!
```

## When You Need Custom Tangent Types

You only need custom tangent types for:

❌ **Recursive structs** (types that reference themselves)
```julia
mutable struct Recursive
    data::Float64
    next::Union{Recursive, Nothing}  # Self-reference!
end
# Requires custom tangent type - see Mooncake.jl docs
```

See `test/mooncake.jl` for examples of custom tangent types.

## Comparison: Enzyme vs Mooncake

| Feature | Enzyme.jl | Mooncake.jl |
|---------|-----------|-------------|
| **Installation** | Requires LLVM | Pure Julia |
| **Tangent Types** | Automatic | Automatic |
| **Recursive Types** | Automatic | Custom needed |
| **Performance** | Very fast | Improving |
| **Mutating Code** | Full support | Full support |

## Using Mooncake with Different Schemes

All checkpointing schemes work with Mooncake:

```julia
# Revolve (binomial checkpointing)
scheme = Revolve(4)
cache = Mooncake.prepare_gradient_cache(loss, model, scheme, tsteps)

# Periodic checkpointing
scheme = Periodic(10)
cache = Mooncake.prepare_gradient_cache(loss, model, scheme, tsteps)

# Online checkpointing (for while loops with unknown iterations)
scheme = Online_r2(4)
cache = Mooncake.prepare_gradient_cache(while_loss, model, scheme)
```

## Troubleshooting

### "Failed to derive tangent type"

This usually happens with recursive structs. Solutions:
1. **Restructure your code** to avoid recursive types (recommended)
2. **Define custom tangent types** (see Mooncake.jl documentation)

### "MethodError: no method matching tangent_type(...)"

Mooncake needs to know the tangent type for all fields. Make sure:
- All field types are concrete (not `Any`)
- Custom types have proper tangent definitions

### Performance Tips

1. **Reuse gradient caches**:
```julia
cache = Mooncake.prepare_gradient_cache(loss, model, scheme, tsteps)
for i = 1:100
    val, grads = Mooncake.value_and_gradient!!(cache, loss, model, scheme, tsteps)
    # Update model...
end
```

2. **Use appropriate checkpoint count**:
```julia
# Too few checkpoints → lots of recomputation
scheme = Revolve(2)  # May be slow for many steps

# Too many checkpoints → high memory usage  
scheme = Revolve(100)  # May use lots of memory

# Balanced
scheme = Revolve(sqrt(tsteps))  # Good rule of thumb
```

## Examples

See the `examples/` directory:
- `heat_mooncake.jl` - Heat equation with Mooncake backend

See the `test/` directory:
- `mooncake.jl` - Comprehensive test suite with many examples
