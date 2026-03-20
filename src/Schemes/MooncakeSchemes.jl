"""
    MooncakeSchemes.jl

Mooncake.jl-based implementations of checkpointing scheme reverse passes.

This file provides `mooncake_rev_checkpoint_for` and `mooncake_rev_checkpoint_while`
functions that use Mooncake.jl for automatic differentiation instead of Enzyme.jl.

## Key Design

1. **Uses Mooncake's `build_rrule`**: Each reverse iteration uses `build_rrule` to
   construct a forward/pullback pair. The adjoint (cotangent) is seeded into the
   tangent of the closure's captured mutable struct, then the pullback propagates
   it backwards through the in-place mutation.

2. **Automatic Tangent Derivation**: Mooncake automatically derives tangent types
   for most mutable structs - users don't need custom tangent types.

3. **Same Checkpointing Logic**: The store/restore/forward actions are identical
   to Enzyme version - only the differentiation mechanism changes.

## How Mooncake's adjoint propagation works for in-place mutations

For a closure `body` that captures a mutable struct `heat` and calls `body(step)`:
- `build_rrule(body, step)` returns a `DerivedRule`
- The rule is called with `CoDual(body, fdata)` where `fdata` wraps the adjoint
  (tangent) for the captured struct
- The forward pass runs the primal computation
- After the forward pass, the tangent on the MutableTangent IS the adjoint seed
  (set it to the cotangent from downstream)
- The pullback propagates the adjoint backwards, updating the tangent in-place

This is analogous to Enzyme's `Duplicated(body, dbody)` where `dbody` holds the
adjoint state.
"""

using Mooncake

import Mooncake: NoTangent

# ============================================================================
# Mooncake-based Reverse Pass for Revolve (for-loops)
# ============================================================================

"""
    mooncake_rev_checkpoint_for(body_input, dbody, alg, range)

Mooncake-based reverse pass for for-loops with Revolve checkpointing.

Uses Mooncake's native autodiff to differentiate each iteration, automatically
deriving tangent types for user structs.
"""
function mooncake_rev_checkpoint_for(
    body_input::Function,
    dbody,
    alg::Revolve{FT},
    range,
) where {FT}
    body = deepcopy(body_input)
    if alg.verbose > 0
        @info "[Checkpointing/Mooncake] Size per checkpoint: $(Base.format_bytes(Base.summarysize(dbody)))"
    end
    storemap = Dict{Int64,Int64}()
    check = 0
    model_check = alg.storage
    if !alg.gc
        GC.enable(false)
    end
    step = alg.steps
    
    while true
        next_action = next_action!(alg)
        if next_action.actionflag == store
            check = check + 1
            storemap[next_action.iteration-1] = check
            save!(model_check, deepcopy(body), check)
        elseif next_action.actionflag == forward
            for j = next_action.startiteration:(next_action.iteration-1)
                body(j)
            end
        elseif next_action.actionflag == firstuturn
            dump_prim(alg.chkp_dump, step, body)
            if alg.verbose > 0
                @info "[Checkpointing/Mooncake] First uturn"
                @info "[Checkpointing/Mooncake] Size of total storage: $(Base.format_bytes(Base.summarysize(alg.storage)))"
            end
            # Mooncake-based differentiation: build_rrule forward + pullback.
            # dbody carries the adjoint seed and is updated in-place.
            _mooncake_diff_iteration!(body, dbody, step)
            dump_adj(alg.chkp_dump, step, dbody)
            step -= 1
            if !alg.gc
                GC.gc()
            end
        elseif next_action.actionflag == uturn
            dump_prim(alg.chkp_dump, step, body)
            _mooncake_diff_iteration!(body, dbody, step)
            dump_adj(alg.chkp_dump, step, dbody)
            step -= 1
            if !alg.gc
                GC.gc()
            end
            if haskey(storemap, next_action.iteration - 1 - 1)
                delete!(storemap, next_action.iteration - 1 - 1)
                check = check - 1
            end
        elseif next_action.actionflag == restore
            body = deepcopy(load(body, model_check, storemap[next_action.iteration-1]))
        elseif next_action.actionflag == done
            if haskey(storemap, next_action.iteration - 1 - 1)
                delete!(storemap, next_action.iteration - 1 - 1)
                check = check - 1
            end
            break
        end
    end
    if !alg.gc
        GC.enable(true)
    end
    return nothing
end

# ============================================================================
# Mooncake-based Reverse Pass for Periodic (for-loops)
# ============================================================================

"""
    mooncake_rev_checkpoint_for(body_input, dbody, alg, range)

Mooncake-based reverse pass for for-loops with Periodic checkpointing.
"""
function mooncake_rev_checkpoint_for(
    body_input::Function,
    dbody,
    alg::Periodic{FT},
    range,
) where {FT}
    body = deepcopy(body_input)
    model_check_outer = alg.storage
    model_check_inner = ArrayStorage{FT}(alg.period)
    if !alg.gc
        GC.enable(false)
    end
    
    # Forward pass: store checkpoints
    for i = 1:alg.acp
        save!(model_check_outer, deepcopy(body), i)
        for j = ((i-1)*alg.period):((i)*alg.period-1)
            body(j)
        end
    end

    # Reverse pass with checkpointing
    for i = alg.acp:-1:1
        body = deepcopy(load(body, model_check_outer, i))
        for j = 1:alg.period
            save!(model_check_inner, deepcopy(body), j)
            body(j)
        end
        for j = alg.period:-1:1
            dump_prim(alg.chkp_dump, j, body)
            body = deepcopy(load(body, model_check_inner, j))
            _mooncake_diff_iteration!(body, dbody, j)
            dump_adj(alg.chkp_dump, j, dbody)
            if !alg.gc
                GC.gc()
            end
        end
    end
    if !alg.gc
        GC.enable(true)
    end
    return nothing
end

# ============================================================================
# Mooncake-based Reverse Pass for Online_r2 (while-loops)
# ============================================================================

"""
    mooncake_rev_checkpoint_while(body_input, dbody, alg)

Mooncake-based reverse pass for while-loops with Online_r2 checkpointing.
"""
function mooncake_rev_checkpoint_while(
    body_input::Function,
    dbody,
    alg::Online_r2{FT},
) where {FT}
    body = deepcopy(body_input)
    
    storemapinv = Dict{Int64,Int64}()
    freeindices = Int64[]
    onlinesteps = 0
    oldcapo = 1
    go = true
    model_check = alg.storage
    
    # Online phase: run forward and store checkpoints
    while go
        next_action = next_action!(alg)
        if next_action.actionflag == store
            check = next_action.cpnum + 1
            storemapinv[check] = next_action.iteration
            save!(model_check, deepcopy(body), check)
        elseif next_action.actionflag == forward
            for j = oldcapo:(next_action.iteration-1)
                go = body()
                onlinesteps = onlinesteps + 1
                if !go
                    break
                end
            end
            oldcapo = next_action.iteration
        else
            @error("Unexpected action in online phase: ", next_action.actionflag)
            go = false
        end
    end
    
    storemap = Dict{Int64,Int64}()
    for (key, value) in storemapinv
        storemap[value] = key
    end

    # Switch to offline revolve
    update_revolve(alg, onlinesteps + 1)
    
    while true
        next_action = next_action!(alg.revolve)
        if next_action.actionflag == store
            check = pop!(freeindices)
            storemap[next_action.iteration-1] = check
            save!(model_check, deepcopy(body), check)
        elseif next_action.actionflag == forward
            for j = next_action.startiteration:(next_action.iteration-1)
                body()
            end
        elseif next_action.actionflag == firstuturn
            model_final = deepcopy(body)
        elseif next_action.actionflag == uturn
            _mooncake_diff_iteration_while!(body, dbody)
            if haskey(storemap, next_action.iteration - 1 - 1)
                push!(freeindices, storemap[next_action.iteration-1-1])
                delete!(storemap, next_action.iteration - 1 - 1)
            end
        elseif next_action.actionflag == restore
            body = deepcopy(load(body, model_check, storemap[next_action.iteration-1]))
        elseif next_action.actionflag == done
            if haskey(storemap, next_action.iteration - 1 - 1)
                delete!(storemap, next_action.iteration - 1 - 1)
            end
            break
        end
    end
    
    return nothing
end

# ============================================================================
# Core Differentiation Functions (using Mooncake's native AD)
# ============================================================================

"""
    _mooncake_diff_iteration!(body, dbody, step)

Differentiate a single for-loop iteration using Mooncake's `build_rrule`.

This mirrors Enzyme's `autodiff(Reverse, Duplicated(body, dbody), Const, Const(step))`:
1. Build a forward/pullback rule for `body(step)`
2. Run the forward pass with `CoDual(body, fdata(dbody))` — the tangent carries
   the adjoint seed from the previous (downstream) iteration
3. After forward, the adjoint seed is in the MutableTangent fields of `dbody`
4. Call pullback to propagate adjoints backwards through the in-place mutation
5. After pullback, `dbody` contains the adjoint for the NEXT (upstream) iteration
"""
function _mooncake_diff_iteration!(body::Function, dbody, step::Int)
    # Build reverse-mode rule for body(step)
    rule = Mooncake.build_rrule(body, step)

    # Get FData view of the tangent. If dbody is already FData (from rrule!!),
    # use it directly. Otherwise convert from Tangent to FData.
    body_fdata = dbody isa Mooncake.FData ? dbody : Mooncake.fdata(dbody)

    # Save the current adjoint seed (cotangent from the downstream iteration)
    adjoint_seed = deepcopy(body_fdata)

    # Zero the tangent before the forward pass. Mooncake's pullback computes:
    #   tangent = saved_before_forward + J^T * current_tangent
    # By zeroing first, saved_before_forward = 0, and we control the adjoint
    # seed by setting current_tangent after the forward pass.
    _zero_tangent!(body_fdata)

    # Run forward pass (body_fdata is zero, forward saves zeros internally)
    body_cd = Mooncake.CoDual(body, body_fdata)
    step_cd = Mooncake.CoDual(step, Mooncake.NoFData())
    _, pb = rule(body_cd, step_cd)

    # Restore the adjoint seed AFTER forward, BEFORE pullback.
    # Pullback will compute: 0 + J^T * adjoint_seed = J^T * adjoint_seed
    _restore_tangent!(body_fdata, adjoint_seed)

    # Run pullback: propagates the adjoint through the reverse of body(step).
    # After pullback, body_fdata contains the adjoint for the next upstream iteration.
    pb(Mooncake.NoRData())

    return nothing
end

"""
    _mooncake_diff_iteration_while!(body, dbody)

Differentiate a single while-loop iteration using Mooncake's `build_rrule`.
Same approach as `_mooncake_diff_iteration!` but for `body()` (no step argument).
"""
function _mooncake_diff_iteration_while!(body::Function, dbody)
    rule = Mooncake.build_rrule(body)
    body_fdata = dbody isa Mooncake.FData ? dbody : Mooncake.fdata(dbody)

    adjoint_seed = deepcopy(body_fdata)
    _zero_tangent!(body_fdata)

    body_cd = Mooncake.CoDual(body, body_fdata)
    _, pb = rule(body_cd)

    _restore_tangent!(body_fdata, adjoint_seed)
    pb(Mooncake.NoRData())
    return nothing
end

# ============================================================================
# Tangent Save/Restore
# ============================================================================

"""
    _zero_tangent!(t)

Zero all differentiable fields in a tangent structure in-place.
"""
function _zero_tangent!(t::Mooncake.FData)
    _zero_tangent_fields!(t.data)
end

function _zero_tangent!(t::Mooncake.MutableTangent)
    _zero_tangent_fields!(t.fields)
end

function _zero_tangent_fields!(nt::NamedTuple)
    for name in keys(nt)
        _zero_tangent_field!(getfield(nt, name))
    end
end

function _zero_tangent_field!(arr::AbstractArray{<:Number})
    arr .= zero(eltype(arr))
end

function _zero_tangent_field!(t::Mooncake.MutableTangent)
    _zero_tangent_fields!(t.fields)
end

function _zero_tangent_field!(t::Mooncake.FData)
    _zero_tangent_fields!(t.data)
end

function _zero_tangent_field!(::Mooncake.NoTangent)
end

function _zero_tangent_field!(x)
    # Fallback: do nothing for immutable/non-differentiable fields
end

"""
    _restore_tangent!(dst::Mooncake.FData, src::Mooncake.FData)

Restore tangent values from `src` into `dst` in-place.
This is needed because Mooncake's forward pass zeros the tangent,
but we need to set the adjoint seed (from the downstream iteration)
before calling the pullback.
"""
function _restore_tangent!(dst::Mooncake.FData, src::Mooncake.FData)
    _restore_tangent_fields!(dst.data, src.data)
end

function _restore_tangent!(dst::Mooncake.MutableTangent, src::Mooncake.MutableTangent)
    _restore_tangent_fields!(dst.fields, src.fields)
end

function _restore_tangent_fields!(dst::NamedTuple, src::NamedTuple)
    for name in keys(dst)
        _restore_tangent_field!(getfield(dst, name), getfield(src, name))
    end
end

function _restore_tangent_field!(dst::AbstractArray, src::AbstractArray)
    dst .= src
end

function _restore_tangent_field!(dst::Mooncake.MutableTangent, src::Mooncake.MutableTangent)
    _restore_tangent_fields!(dst.fields, src.fields)
end

function _restore_tangent_field!(dst::Mooncake.FData, src::Mooncake.FData)
    _restore_tangent_fields!(dst.data, src.data)
end

# No-op for non-differentiable fields
function _restore_tangent_field!(::Mooncake.NoTangent, ::Mooncake.NoTangent)
end

# Fallback for scalar tangents
function _restore_tangent_field!(dst, src)
    # For immutable scalar tangents, no restoration needed
    # (they are not modified by the forward pass)
end

# ============================================================================
# Gradient Accumulation (kept for backward compatibility, but no longer used
# in the main code path — build_rrule handles accumulation in-place)
# ============================================================================

"""
    _accumulate_grads!(dbody, grads, step)

Accumulate gradients from `grads` into `dbody` field-by-field.

Note: With the `build_rrule`-based differentiation approach, adjoint propagation
is handled in-place by Mooncake's pullback mechanism. This function is kept for
backward compatibility but is no longer called in the main code path.
"""
function _accumulate_grads!(dbody, grads, step::Int)
    if grads === nothing || grads isa NoTangent
        return
    end
    
    dbody_type = typeof(dbody)
    grads_type = typeof(grads)
    
    try
        for name in fieldnames(dbody_type)
            if hasfield(grads_type, name)
                d_field = getfield(dbody, name)
                g_field = getfield(grads, name)
                
                if d_field !== nothing && g_field !== nothing
                    _accumulate_field!(d_field, g_field)
                end
            end
        end
    catch
        # Silently ignore if field access fails
    end
end

"""
    _accumulate_field!(dst, src)

Accumulate gradient from src into dst, handling different types appropriately.
"""
function _accumulate_field!(dst, src)
    if dst isa AbstractArray && src isa AbstractArray
        dst .+= src
    else
        for name in fieldnames(typeof(dst))
            if hasfield(typeof(src), name)
                _accumulate_field!(getfield(dst, name), getfield(src, name))
            end
        end
    end
end
