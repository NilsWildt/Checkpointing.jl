# Rules

EnzymeRules rules are provided for checkpointing, and Mooncake `rrule!!`s when Mooncake is loaded. The `augmented_primal` rules for `checkpoint_for` and `checkpoint_while` instantiate the scheme and run its forward half, `fwd_checkpoint_for` or `fwd_checkpoint_while`, which computes the primal and stores the checkpoints. The `reverse` rules pass the resulting tape to the scheme's reverse half, `rev_checkpoint_for` or `rev_checkpoint_while`; see [Checkpointing Schemes](schemes.md). Adding support for another rule system requires making the same two calls from its forward and reverse rules.

## Mooncake

Loading Mooncake activates the `CheckpointingMooncakeExt` extension, which declares `checkpoint_for` and `checkpoint_while` Mooncake primitives. Their `rrule!!`s make the same two calls; the reverse half differentiates each iteration with a Mooncake rule derived for the loop body, which is built once per reverse sweep. All schemes and storage backends work unchanged.

```julia
using Checkpointing, Mooncake

function loss(model, scheme, steps)
    @ad_checkpoint scheme for i in 1:steps
        advance!(model)
    end
    return sum(model.x)
end

cache = Mooncake.prepare_gradient_cache(loss, model, Revolve(4), 100)
val, (_, dmodel, _, _) = Mooncake.value_and_gradient!!(cache, loss, model, Revolve(4), 100)
```

Mooncake requires a rule to leave mutable arguments as it found them, so the pullback restores the state the loop body captures to its value before the loop. That takes one snapshot of the loop state in addition to the scheme's checkpoints.
