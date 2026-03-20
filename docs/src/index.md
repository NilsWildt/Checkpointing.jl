# Checkpointing

[Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl) provides checkpointing schemes for adjoint computations using automatic differentiation (AD) of time-stepping loops. Currently, we support the macro `@ad_checkpoint`, which differentiates and checkpoints a struct used in a while or for the loop with a `UnitRange`.

Each loop iteration is differentiated using an AD backend. We currently support:
- **Enzyme.jl** (default): [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) - LLVM-based AD
- **Mooncake.jl** (new): [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) - Pure Julia AD

The schemes are agnostic to the AD tool being used and can be easily interfaced with any Julia AD tool. Currently, the package supports:

## Scheme
* Revolve/Binomial checkpointing [1]
* Periodic checkpointing
* Online r=2 checkpointing for a while loops with a priori unknown number of iterations [2]

## Rules
* [EnzymeRules.jl](https://enzyme.mit.edu/julia/stable/generated/custom_rule/) (default)
* Mooncake.jl rrule!! (alternative backend)

## Storage
* ArrayStorage: Stores all checkpoints values in an array of type `Array`
* HDF5Storage: Stores all checkpoints values in an HDF5 file

## Choosing an AD Backend

### Enzyme.jl (Default)
```julia
using Enzyme
autodiff(Enzyme.Reverse, my_function, Duplicated(model, dmodel), ...)
```

### Mooncake.jl
```julia
using Mooncake
cache = Mooncake.prepare_gradient_cache(my_function, model, scheme, tsteps)
val, grads = Mooncake.value_and_gradient!!(cache, my_function, model, scheme, tsteps)
```

**Key Feature**: Mooncake automatically derives tangent types for most structs - no custom code needed for simple cases! Only recursive structs require custom tangent definitions.

See the [Mooncake Backend Guide](mooncake.md) for detailed documentation.

## Limitations
* Currently, the package only supports `UnitRange` ranges in `for` loops. We will add range types on a per-need basis. Please, open an issue if you need support for a specific range type.
* We support both Enzyme.jl and Mooncake.jl as differentiation backends. Enzyme.jl is the default due to its maturity.
* We don't support any activity analysis. This implies that loop iterators have to be part of the checkpointed struct if they are used in the loop body. Currently, we store the entire struct at each checkpoint. This is not necessary, and we will add support for storing only the required fields in the future.

## Future
The following features are planned for development:

* Support checkpoints on GPUs
## Quick Start

```@contents
Pages = [
    "quickstart.md",
    "mooncake.md",
]
Depth=1
```
## API
```@contents
Pages = [
    "lib/checkpointing.md",
]
Depth = 1
```
## References
[1] Andreas Griewank and Andrea Walther. 2000. Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation. ACM Trans. Math. Softw. 26, 1 (March 2000), 19–45. DOI:https://doi.org/10.1145/347837.347846