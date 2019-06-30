module Optim4Flux
using LinearAlgebra, Optim, Flux, Zygote, Plots # TODO: replace for recipesbase
import Base.copyto!

export gradlength, paramlength, optfuns

gradlength(grads::Zygote.Grads) = sum(length(g[1]) for g in grads.grads)
paramlength(params::Flux.Params) = sum(length, params.params)
Base.zeros(grads::Zygote.Grads) = zeros(gradlength(grads))
Base.zeros(pars::Flux.Params) = zeros(paramlength(pars))

# Grads =============================================

function copyto!(v::AbstractArray, grads::Zygote.Grads)
    @assert length(v) == gradlength(grads)
    s = 1
    for g in grads.grads
        l = length(g[2])
        v[s:s+l-1] .= vec(g[2])
        s += l
    end
    v
end

function copyto!(grads::Zygote.Grads, v::AbstractArray)
    s = 1
    for g in grads.grads
        l = length(g[2])
        g[2] .= reshape(v[s:s+l-1], size(g[2]))
        s += l
    end
    grads
end

# Params =============================================

function copyto!(v::AbstractArray, pars::Flux.Params)
    @assert length(v) == paramlength(pars)
    s = 1
    for g in pars.params
        l = length(g)
        v[s:s+l-1] .= vec(g)
        s += l
    end
    v
end

function copyto!(pars::Flux.Params, v::AbstractArray)
    s = 1
    for p in pars.params
        l = length(p)
        p .= reshape(v[s:s+l-1], size(p))
        s += l
    end
    pars
end


function optfuns(loss, pars::Flux.Params)
    grads = Zygote.gradient(loss, pars)
    gradvec = zeros(grads)
    gradfun = function (g,w)
        copyto!(pars, w)
        grads = Zygote.gradient(loss, pars)
        copyto!(g, grads)
    end
    lossfun = function (w)
        copyto!(pars, w)
        loss()
    end
    lossfun, gradfun
end

@recipe function lossplot(loss::Function, pars::Flux.Params, l=0.1)
    p       = zeros(pars)
    copyto!(p,pars)
    pcopy   = deepcopy(p)
    n0      = norm(p)
    dx,dy   = randn(length(p)),randn(length(p))
    dx     *= n0*l/norm(dx)
    dy     *= n0*l/norm(dy)
    pertvec = LinRange(-1,1,20)
    losses = map(Iterators.product(pertvec,pertvec)) do (lx,ly)
        pi = p + lx*dx + ly*dy
        copyto!(pars, pi)
        loss()
    end
    copyto!(pars, pcopy)
    seriestype := :contour
    pertvec,pertvec,losses
end

end

using Main.Optim4Flux, Optim, Zygote, Flux, Test
##
m = Chain(Dense(1,10,tanh),Dense(10,10,tanh) , Dense(10,1))
x = randn(1,100)
y = sin.(x)

loss() = mean(abs2, m(x) .- y)
@show loss()
Zygote.refresh()
pars = Flux.params(m)
pars0 = deepcopy(pars)
npars = Optim4Flux.paramlength(pars)
# @test begin
#     copyto!(pars, zeros(pars))
#     all(all(iszero, p) for p in pars)
# end
# @test begin
#     p = zeros(pars)
#     copyto!(pars, 1:npars)
#     copyto!(p, pars)
#     p == 1:npars
# end
# grads = Zygote.gradient(loss, pars)
# grads0 = deepcopy(grads)
# @test begin
#     copyto!(grads, zeros(grads))
#     all(all(iszero,grads[k]) for k in keys(grads.grads))
# end
# @test begin
#     p = zeros(grads)
#     copyto!(grads, 1:npars)
#     copyto!(p, grads)
#     p == 1:npars
# end


# @test copyto!(copyto(grads), grads) == grads

opt = ADAM(0.01)
cb = ()-> @show loss()
for i = 1:5000
    grads = Zygote.gradient(loss, pars)
    Flux.Optimise.update!(opt, pars, grads)
    @show loss()
end

plot(loss, pars,1) |> display

lossfun, gradfun = optfuns(loss, pars)
res = Optim.optimize(lossfun, gradfun, randn(paramlength(pars)))
plot(loss, pars,0.1) |> display

sp = sortperm(x[:])
plot([y[sp] m(x)[sp]])
