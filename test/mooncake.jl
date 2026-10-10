# Gradients through every scheme under Mooncake, checked against Enzyme on the
# same loop without checkpointing. The loop bodies are the nonlinear ones from
# correctness.jl, so a reverse sweep that replays the wrong steps, the wrong
# indices or too few steps is caught.
using Checkpointing
using Enzyme
using Mooncake
using Test

mutable struct MTraced
    x::Vector{Float64}
    n::Int
end

mstep!(s::MTraced, i) = (s.x .= s.x .+ 0.01 .* i .* s.x .^ 2; nothing)

function mc_checkpointed_for(s::MTraced, scheme::Scheme, r::UnitRange{Int})
    @ad_checkpoint scheme for i in r
        mstep!(s, i)
    end
    return sum(s.x)
end

function mc_checkpointed_while(s::MTraced, scheme::Scheme, n::Int)
    s.n = 0
    @ad_checkpoint scheme while s.n < n
        mstep!(s, 1)
        s.n += 1
    end
    return sum(s.x)
end

function mc_plain_for(s::MTraced, r::UnitRange{Int})
    for i in r
        mstep!(s, i)
    end
    return sum(s.x)
end

function mc_plain_while_steps(s::MTraced, n::Int)
    for _ = 1:n
        mstep!(s, 1)
    end
    return sum(s.x)
end

# Work before the loop whose pullback reads the state the loop then overwrites:
# it only gets the right gradient if the loop's pullback restores that state.
function mc_prelude_for(s::MTraced, scheme::Scheme, r::UnitRange{Int})
    s.x .= s.x .* s.x
    return mc_checkpointed_for(s, scheme, r)
end

function mc_prelude_plain_for(s::MTraced, r::UnitRange{Int})
    s.x .= s.x .* s.x
    return mc_plain_for(s, r)
end

function mc_prelude_while(s::MTraced, scheme::Scheme, n::Int)
    s.x .= s.x .* s.x
    return mc_checkpointed_while(s, scheme, n)
end

function mc_prelude_plain_while(s::MTraced, n::Int)
    s.x .= s.x .* s.x
    return mc_plain_while_steps(s, n)
end

mc_start() = MTraced([0.5, 0.8], 0)

function enzyme_gradient(f, args...)
    x, dx = mc_start(), MTraced([0.0, 0.0], 0)
    _, primal =
        autodiff(Enzyme.ReverseWithPrimal, f, Active, Duplicated(x, dx), Const.(args)...)
    return primal, dx.x
end

function mooncake_gradient(f, args...)
    x = mc_start()
    cache = Mooncake.prepare_gradient_cache(f, x, args...)
    primal, grads = Mooncake.value_and_gradient!!(cache, f, x, args...)
    # Mooncake hands mutable inputs back as it found them.
    @test x.x == mc_start().x
    return primal, grads[2].fields.x
end

function final_state(f, args...)
    x = mc_start()
    f(x, args...)
    return x.x
end

@testset "Mooncake gradients match an uncheckpointed loop" begin
    @testset "$(nameof(typeof(scheme)))(3) over $r" for scheme in (Revolve(3), Periodic(3)),
        r in (1:10, 3:12, 1:9, 1:1, 1:0)

        want_primal, want_grad = enzyme_gradient(mc_plain_for, r)
        primal, grad = mooncake_gradient(mc_checkpointed_for, scheme, r)
        @test primal ≈ want_primal
        @test grad ≈ want_grad
        @test final_state(mc_checkpointed_for, scheme, r) == final_state(mc_plain_for, r)
    end
    @testset "Online_r2($c) over $n iterations" for c = 1:5,
        n in
        filter(<=(Checkpointing.online_r2_limit(c)), [1, 2, 3, 4, 5, 8, 10, 14, 20, 30])

        want_primal, want_grad = enzyme_gradient(mc_plain_while_steps, n)
        primal, grad = mooncake_gradient(mc_checkpointed_while, Online_r2(c), n)
        @test primal ≈ want_primal
        @test grad ≈ want_grad
    end
    @testset "work before the loop, $(nameof(typeof(scheme)))" for scheme in
                                                                   (Revolve(3), Periodic(3))
        want_primal, want_grad = enzyme_gradient(mc_prelude_plain_for, 1:10)
        primal, grad = mooncake_gradient(mc_prelude_for, scheme, 1:10)
        @test primal ≈ want_primal
        @test grad ≈ want_grad
    end
    @testset "work before the loop, Online_r2" begin
        want_primal, want_grad = enzyme_gradient(mc_prelude_plain_while, 10)
        primal, grad = mooncake_gradient(mc_prelude_while, Online_r2(3), 10)
        @test primal ≈ want_primal
        @test grad ≈ want_grad
    end
end
