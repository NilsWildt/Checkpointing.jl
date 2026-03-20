"""
    MooncakeRules.jl

Mooncake.jl differentiation rules for Checkpointing.jl.

This file provides `rrule!!` definitions for the checkpointing functions,
enabling Mooncake.jl to differentiate through checkpointed loops.

## Usage with DifferentiationInterface

```julia
using Checkpointing
using DifferentiationInterface as DI
import DifferentiationInterface.Mooncake

# Define your function with checkpointing
function my_loss(model, scheme, tsteps)
    @ad_checkpoint scheme for i = 1:tsteps
        step!(model)
    end
    return compute_loss(model)
end

# Use DI.Constant to mark non-differentiable arguments
backend = DI.AutoMooncake()
prep = DI.prepare_gradient(
    my_loss,
    backend,
    model,                    # differentiable
    DI.Constant(scheme),      # NOT differentiable
    DI.Constant(tsteps)       # NOT differentiable
)

val, grad = DI.value_and_gradient(
    my_loss,
    prep,
    backend,
    model,
    DI.Constant(scheme),
    DI.Constant(tsteps)
)
```

## Direct Mooncake API

For advanced use, you can also use Mooncake's CoDual types directly:

```julia
using Checkpointing, Mooncake

body_cd = Mooncake.CoDual(my_body_func, Mooncake.NoTangent())
alg_cd = Mooncake.CoDual(my_scheme, Mooncake.NoFData())  # Scheme is non-differentiable
range_cd = Mooncake.CoDual(1:100, Mooncake.NoFData())

y_cd, pullback = Mooncake.rrule!!(
    Mooncake.CoDual(checkpoint_for, Mooncake.NoTangent()),
    body_cd, alg_cd, range_cd
)
```
"""

module MooncakeRules

using Mooncake

import Mooncake: rrule!!, CoDual, primal, tangent, NoTangent, NoFData, NoRData,
                 @is_primitive, MinimalCtx, fdata, rdata

# Import checkpoint functions and types from parent module
import ..checkpoint_for, ..checkpoint_while, ..instantiate
import ..mooncake_rev_checkpoint_for, ..mooncake_rev_checkpoint_while
import .._accumulate_grads!
import ..Scheme, ..Revolve

# ============================================================================
# Core Type Inference Overrides for Mooncake
# ============================================================================

# These overrides are critical for DifferentiationInterface.jl compatibility.
# When a user passes DI.Constant(scheme), DI unwraps it and passes it to Mooncake.
# By default, Mooncake would try to automatically differentiate it, leading to type
# mismatches since checkpointing schemes are fundamentally non-differentiable.
# These overrides force Mooncake to treat all Scheme types as having NoTangent.

Mooncake.tangent_type(::Type{<:Scheme}) = Mooncake.NoTangent
Mooncake.zero_tangent(::Scheme) = Mooncake.NoTangent()

# ============================================================================
# rrule!! for checkpoint_for (for-loops with UnitRange)
# ============================================================================

function check_closure_captures_mooncake(body)
    closure_type = typeof(body)
    field_names = fieldnames(closure_type)
    field_types = fieldtypes(closure_type)

    scalar_vars = String[]
    struct_vars = String[]

    for (name, ftype) in zip(field_names, field_types)
        field = getfield(body, name)
        if isa(field, Core.Box)
            error(
                "[Checkpointing.jl]: Variable `$name` is reassigned inside the loop. " *
                "Please make sure that `$name` is only modified in-place.",
            )
        elseif ftype <: Union{Float16,Float32,Float64}
            push!(scalar_vars, string(name))
        elseif ismutabletype(ftype)
            push!(struct_vars, string(name))
        end
    end

    if !isempty(scalar_vars) && !isempty(struct_vars)
        scalar_list = join(["`$v`" for v in scalar_vars], ", ")
        struct_list = join(["`$v`" for v in struct_vars], ", ")
        error(
            "[Checkpointing.jl]: The loop body captures floating-point variable(s) $scalar_list " *
            "alongside mutable struct(s) $struct_list.\n" *
            "This can cause AD issues.\n\n" *
            "Solution: Store these values as fields in your mutable struct instead of " *
            "capturing them as separate variables.",
        )
    end
end

# ============================================================================
# rrule!! for checkpoint_for (for-loops with UnitRange)
# ============================================================================

@is_primitive MinimalCtx Tuple{typeof(checkpoint_for),Function,Scheme,UnitRange{Int64}}

# Note: We accept any tangent type for the scheme (A_tangent) because Mooncake may
# create MutableTangent even when DI.Constant is used. The scheme is not actually
# differentiable, so we just extract its primal value.
function rrule!!(
    ::CoDual{typeof(checkpoint_for)},
    body_cd::CoDual{F},
    alg_cd::CoDual{A,A_tangent},
    range_cd::CoDual{UnitRange{Int64},NoFData},
) where {F<:Function,A<:Scheme,A_tangent}
    body_primal = primal(body_cd)
    body_tangent = tangent(body_cd)
    
    check_closure_captures_mooncake(body_primal)
    
    # Store a deep copy of the PRE-forward-pass state for the reverse pass.
    # The reverse pass needs to replay from the initial state.
    tape_body = deepcopy(body_primal)
    
    # Run forward pass - extract primal value from alg_cd regardless of tangent type
    alg_primal = primal(alg_cd)
    checkpoint_for(body_primal, alg_primal, primal(range_cd))
    
    # Output is nothing
    y_cd = CoDual(nothing, NoFData())
    
    function checkpoint_for_pullback(::NoRData)
        # body_tangent now contains the adjoint seed from downstream operations
        # (e.g., from sum(heat.Tnext) → d(sum)/d(Tnext) = [1,1,...,1]).
        # We pass it as the initial adjoint for the reverse checkpointing pass.
        
        # Instantiate scheme for reverse pass
        scheme = instantiate(typeof(tape_body), alg_primal, length(primal(range_cd)))
        
        # Run reverse pass with Mooncake-based differentiation.
        # body_tangent carries the adjoint seed and will be updated in-place
        # by build_rrule pullbacks at each reverse iteration.
        mooncake_rev_checkpoint_for(
            tape_body, 
            body_tangent, 
            scheme, 
            primal(range_cd)
        )
        
        # After the reverse pass, body_tangent contains the final adjoint
        # (gradient w.r.t. the initial state). No additional accumulation needed.
        
        # Return zero tangents for arguments
        return (NoRData(), NoRData(), NoRData(), NoRData())
    end
    
    return y_cd, checkpoint_for_pullback
end

# ============================================================================
# rrule!! for checkpoint_while (while-loops)
# ============================================================================

@is_primitive MinimalCtx Tuple{typeof(checkpoint_while),Function,Scheme}

# Note: We accept any tangent type for the scheme (A_tangent) because Mooncake may
# create MutableTangent even when DI.Constant is used. The scheme is not actually
# differentiable, so we just extract its primal value.
function rrule!!(
    ::CoDual{typeof(checkpoint_while)},
    body_cd::CoDual{F},
    alg_cd::CoDual{A,A_tangent},
) where {F<:Function,A<:Scheme,A_tangent}
    body_primal = primal(body_cd)
    body_tangent = tangent(body_cd)
    
    check_closure_captures_mooncake(body_primal)
    
    # Store a deep copy of the PRE-forward-pass state for the reverse pass
    tape_body = deepcopy(body_primal)
    
    # Run forward pass - extract primal value from alg_cd regardless of tangent type
    alg_primal = primal(alg_cd)
    checkpoint_while(body_primal, alg_primal)
    
    y_cd = CoDual(nothing, NoFData())
    
    function checkpoint_while_pullback(::NoRData)
        scheme = instantiate(typeof(tape_body), alg_primal)
        
        # body_tangent carries the adjoint seed from downstream and will be
        # updated in-place by the reverse pass
        mooncake_rev_checkpoint_while(
            tape_body, 
            body_tangent, 
            scheme
        )
        
        # Return zero tangents for arguments
        return (NoRData(), NoRData(), NoRData())
    end
    
    return y_cd, checkpoint_while_pullback
end

end # module MooncakeRules
