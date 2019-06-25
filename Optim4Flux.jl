module Optim4Flux
using Optim, Flux, Zygote


# Grads =============================================
gradlength(grads) = sum(length(g[1]) for g in grads.grads)

function flatten!(v, grads::Zygote.Grads)
    @assert length(v) == gradlength(grads)
    s = 1
    for g in grads.grads
        l = length(g[2])
        v[s:s+l-1] .= vec(g[2])
        s += l
    end
    v
end

flatten(grads::Zygote.Grads) = flatten!(zeros(gradlength(grads)), grads::Zygote.Grads)

function unflatten!(v, grads::Zygote.Grads)
    s = 1
    for g in grads.grads
        l = length(g[2])
        g[2] .= reshape(v[s:s+l-1], size(g[2]))
        s += l
    end
    grads
end

# @test unflatten!(flatten(grads), grads) == grads
# Params =============================================

paramlength(params) = sum(length, params.params)

function flatten!(v, pars::Zygote.Params)
    @assert length(v) == gradlength(pars)
    s = 1
    for g in pars.pars
        l = length(g[2])
        v[s:s+l-1] .= vec(g[2])
        s += l
    end
    v
end

flatten(pars::Zygote.Params) = flatten!(zeros(gradlength(pars)), pars::Zygote.Params)

function unflatten!(v, pars::Zygote.Params)
    s = 1
    for p in pars.params
        l = length(p)
        p .= reshape(v[s:s+l-1], size(p))
        s += l
    end
    pars
end

# @test unflatten!(flatten(pars), pars) == pars

function Optim.OnceDifferentiable(loss, pars::Zygote.Params)
    grads = Zygote.gradient(loss, pars)
    gradvec = flatten(grads)
    gradfun = function (w)
        unflatten!(w, pars)
        grads = Zygote.gradient(loss, pars)
        flatten!(gradvec, grads)
    end
    lossfun = function (w)
        grads = unflatten!(v, grads)
        loss()
    end


end
