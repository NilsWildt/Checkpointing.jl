module CheckpointingMooncakeExt

# Reverse-mode rules for `checkpoint_for` and `checkpoint_while` under Mooncake.
#
# The schemes drive the schedule themselves and call `adjoint_step!` once per
# reversed iteration, so this extension only has to say how one iteration is
# differentiated. Everything else -- stores, restores, recomputation, storage
# backends -- is shared with the Enzyme rules.

using Checkpointing:
    Checkpointing,
    Scheme,
    checkpoint_for,
    checkpoint_while,
    instantiate,
    fwd_checkpoint_for,
    rev_checkpoint_for,
    fwd_checkpoint_while,
    rev_checkpoint_while,
    checkpoint_alloc,
    checkpoint_copy!,
    check_closure_captures
import Checkpointing: adjoint_step!
using EnzymeCore: Const
using Mooncake
using Mooncake: CoDual, NoRData, MinimalCtx, @is_primitive, primal, tangent

# A scheme is configuration and mutable schedule state, never something to
# differentiate with respect to.
Mooncake.tangent_type(::Type{<:Scheme}) = Mooncake.NoTangent

"""
    MooncakeAdjoint(rule)

The `config` the schemes pass to [`adjoint_step!`](@ref) during a Mooncake
reverse sweep. `rule` is the derived rule for one call of the loop body. Every
reversed iteration calls the body with arguments of the same types, so the rule
is built once per sweep and reused.
"""
struct MooncakeAdjoint{R}
    rule::R
end

# The schemes hand loop indices over wrapped in `Const`, as Enzyme expects.
_unwrap(x::Const) = x.val
_unwrap(x) = x

# Zero a forward tangent in place. `set_to_zero!!` has no method for `FData` or
# `NoFData`, but every field of an `FData` is either mutable or `NoFData`, so
# zeroing it field by field zeroes it.
_zero!(x::Mooncake.FData) = (foreach(_zero!, x.data); x)
_zero!(x::Mooncake.NoFData) = x
_zero!(x) = Mooncake.set_to_zero!!(x)

function adjoint_step!(config::MooncakeAdjoint, body, dbody, args...)
    # On entry `dbody` holds the adjoint of the state after this iteration. A
    # Mooncake pullback adds into whatever the tangent holds once the forward
    # pass is done, and the forward pass records the tangent it finds so it can
    # put it back, so the seed has to go in between the two. The schemes keep
    # passing the same `dbody`, so zeroing and incrementing must both update its
    # mutable leaves in place.
    seed = deepcopy(dbody)
    _zero!(dbody)
    _, pullback =
        config.rule(CoDual(body, dbody), map(Mooncake.zero_fcodual ∘ _unwrap, args)...)
    Mooncake.increment!!(dbody, seed)
    pullback(NoRData())
    return nothing
end

@is_primitive MinimalCtx Tuple{typeof(checkpoint_for),Any,Scheme,UnitRange{Int}}

function Mooncake.rrule!!(
    ::CoDual{typeof(checkpoint_for)},
    body_cd::CoDual,
    alg_cd::CoDual,
    range_cd::CoDual,
)
    body, dbody = primal(body_cd), tangent(body_cd)
    range = primal(range_cd)
    check_closure_captures(Const(body))
    scheme = instantiate(typeof(body), primal(alg_cd), length(range))
    # Mooncake requires a rule's pullback to leave its mutable arguments as it
    # found them, and the loop overwrites the state its body captures.
    initial = checkpoint_alloc(body)
    tape = fwd_checkpoint_for(body, scheme, range)
    function checkpoint_for_pullback(::NoRData)
        if tape !== nothing
            config = MooncakeAdjoint(Mooncake.build_rrule(body, first(range)))
            rev_checkpoint_for(config, tape, dbody, scheme, range)
        end
        checkpoint_copy!(body, initial)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return Mooncake.zero_fcodual(nothing), checkpoint_for_pullback
end

@is_primitive MinimalCtx Tuple{typeof(checkpoint_while),Any,Scheme}

function Mooncake.rrule!!(
    ::CoDual{typeof(checkpoint_while)},
    body_cd::CoDual,
    alg_cd::CoDual,
)
    body, dbody = primal(body_cd), tangent(body_cd)
    check_closure_captures(Const(body))
    scheme = instantiate(typeof(body), primal(alg_cd))
    initial = checkpoint_alloc(body)
    tape = fwd_checkpoint_while(body, scheme)
    function checkpoint_while_pullback(::NoRData)
        config = MooncakeAdjoint(Mooncake.build_rrule(body))
        rev_checkpoint_while(config, tape, dbody, scheme)
        checkpoint_copy!(body, initial)
        return NoRData(), NoRData(), NoRData()
    end
    return Mooncake.zero_fcodual(nothing), checkpoint_while_pullback
end

end
