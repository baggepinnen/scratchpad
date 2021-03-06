# TODO: På vilka ställen interagerar olika partiklar med varandra? Enligt Berntorp är feedback gain en funktion av partikelns yhat och medelpartikelns yhat
# Gaussian ll kan hanteras analytiskt
# kan man träna modellen på detta sättet först och sedan träna en klassisk observerare utan att rucka på f?
# Vi kan hantera godtycklig dg, men är det vettigt att alltid anta Gaussisk df? z_post borde ändå kunna vara multimodal, tänk pf-testmodellen
#
cd(@__DIR__)
using Plots, Flux, Zygote, LinearAlgebra, Statistics, Random, Printf, OrdinaryDiffEq, IterTools, Distributions, ChangePrecision, DSP, SliceMap
using Flux: params
default(lab="", grid=false)
# include("SkipRNN.jl")
Random.seed!(0)

@userplot Eigvalplot
@recipe function eigvalplot(A::Eigvalplot)
    e = data(A.args[1]) |> eigvals# .|> log
    title --> "Eigenvalues"
    ar = range(0, stop=2pi, length=40)
    @series begin
        linestyle := :dash
        color := :black
        cos.(ar), sin.(ar)
    end
    @series begin
        seriestype := :scatter
        real.(e), imag.(e)
    end
end

const h = 0.1
const Ta = 5000

function hstack(xs, n)
    buf = Zygote.Buffer(xs, length(xs), n)
    for i = 1:n
        buf[:, i] = xs
    end
    return copy(buf)
end

@changeprecision Float32 function generate_data_test(T)
    f(x,t) = 0.5x + 25x/(1 + x^2) + 8cos(1.2*t)
    g(x) = 0.05x^2
    x = [randn()]
    for t = 1:T
        push!(x, f(x[end], t) + randn())
    end
    x', g.(x)' .+ randn.()
end

@changeprecision Float32 function pendcart(xd,x,p,t)
    g = 9.82; l = 1.0; d = 0.5
    u = 0
    xd[1] = x[2]
    xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1]) - d*x[2]
    xd
end

@changeprecision Float32 function generate_data_pendcart(T)
    u0 = @. Float32[pi, 6] * 2 *(rand(Float32)-0.5)
    tspan = (0f0,Float32(T))
    prob = ODEProblem(pendcart,u0,tspan)
    sol = solve(prob,Tsit5())
    z = reduce(hcat, sol(0:h:T).u)
    y = vcat((sin.(z[1:1,:])), cos.(z[1:1,:])) .+ 0.05 .* randn.()
    z,y
end

Zygote.@adjoint function Base.reduce(::typeof(hcat), V::AbstractVector{<:AbstractVector})
    reduce(hcat, V), dV -> (nothing, collect(eachcol(dV)))
end

function stats(e)
    μ = mean(e, dims=2)
    μ, sum(abs2, e .- μ, dims=2) ./ size(e,2) .+ 1f-3
end

@changeprecision Float32 function kl(e::AbstractMatrix, μ2, σ2, c = 1)
    μ1, σ1² = stats(e)
    lσ1 = log.(sqrt.(σ1²))
    lσ2 = log.(σ2)#log.(sqrt.(var(dy)))
    l = 0f0
    for i = eachindex(μ1)
        l += c*2lσ2[i] - 2lσ1[i] +
        c*(σ1²[i] + abs2(μ1[i] - μ2[i]))/(σ2[i]^2 + 1f-5)
    end
    0.5f0l
end

@changeprecision Float32 function klng(μ1, lσ1, c = 1)
    σ1² = exp(lσ1)^2
    # lσ1 = log.(σ1)
    0.5*(2lσ1 + c*(σ1² + abs2(μ1)))
end

@changeprecision Float32 function kl2(e, c = 1)
    μ1, σ1² = stats(e)
    0.5c*sum(abs2,μ1[i])
end
# Zygote.refresh()
# Zygote.gradient(e->kl(e,dy), randn(1,10))

##
trajs_full = [generate_data_pendcart(1.5) for i = 1:80]
trajs_meas = map(trajs_full) do (_,t)
    [copy(c) for c in eachcol(t)]
end
trajs_state = map(trajs_full) do (t,_)
    [copy(c) for c in eachcol(t)]
end

const dy2 = MvNormal(2, 0.05f0)
const nz = 3
const ny = 2
const nu = 0
nh  = 20
np = 50
const nα = 5
const αnet = Chain(Dense(nz,nα), softmax)
const A = [(0.001randn(nz,nz)) for _ = 1:nα]
const C = [(0.001randn(nz,nz)) for _ = 1:nα]
function f(z,w)
    α  = αnet(z)
    Ai = sum(α.*A)
    Ci = sum(α.*C)
    z  = Ai*z + Ci*w
end
const g  = Chain(Dense(nz,nh,tanh), Dense(nh,ny))
const z0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
const w0 = Chain(Dense(4ny,nh,tanh), Dense(nh,2nz))
# const fn  = Chain(Dense(2nz+nu,nh,tanh), Dense(nh,nz))
const kn  = Chain(Dense(3ny,nh,tanh), Dense(nh,2nz))
function k(z,y)
    yh = g(z)
    kn([yh;hstack(y, np);hstack(mean(yh,dims=2), np)])
end
pars = params((A...,C...,αnet,g,kn,z0,w0))


function train(loss, ps, dataset, opt; cb=i->nothing, schedule=I->copy(opt.eta))
    @progress for (i,d) in enumerate(dataset)
        I = length(loss1)+1
        opt.eta = schedule(I)
        # Flux.reset!(model)
        (l1,l2), back = Zygote.forward(()->loss(I, d), ps)
        grads = back((1f0,1f0))
        push!(loss1, l1)
        push!(loss2, l2)
        Flux.Optimise.update!(opt, ps, grads)
        cb(i)
    end
end

const ⊗ = Zygote.dropgrad

loss1 = Float64[]
loss2 = Float64[]

cb = function (i=0)
    i % 1500 == 0 || return
    lm = [loss1 loss2]
    lm = length(loss1) > Ta ? lm[Ta:end,:] : lm
    lm = filt(ones(40), [40], lm, fill(lm[1,1], 39))
    fig = plot(lm, layout=3, sp=1, yscale=minimum(lm) < 0 ? :identity : :log10)

    yh,_,_ = sim(trajs_meas[1])
    ##
    plot!(reduce(hcat,trajs_meas[1])', sp=2:3)
    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.2,:green), sp=2, markerstrokecolor=:auto)
    scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:green), sp=3, markerstrokecolor=:auto)

    yh,zh,zp = sim(trajs_meas[1], false)

    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.2,:black), sp=2, markerstrokecolor=:auto)
    scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:black), sp=3, markerstrokecolor=:auto)
    display(fig)
end

##

function sim(y, feedback=true)
    T   = length(y)
    z   = z0(samplenet(w0([y[1];y[2];y[3];y[4]]))[3])
    yh = []
    # yh2 = []
    zh  = []
    zp  = []
    for t in 1:length(y)-1
        ŷ   = g(z)
        push!(yh, ŷ)
        e   = y[t] .- ŷ
        μ, σ, w = samplenet(k(z,y[t+1]))
        push!(zh, mean(z, dims=2)[:])
        push!(zp, z)
        ŷ   = g(z)
        # push!(yh2, ŷ)
        z   = f(z, feedback.*w)
    end
    yh, zh, zp
end

function varloss(e,σ)
    μ = mean(e, dims=2)
    sum(0.5*(abs2(μ[i])/σ[i]^2 + log(2π)) + 0*log(σ[i])  for i in eachindex(μ))
end
function varloss(e)
    μ,σ² = stats(e)
    sum(0.5*(abs2(μ[i])/σ²[i] + log(2π)) + log(√(σ²[i]))  for i in eachindex(μ))
end

function partlik(e, σ)
    w = mapcols(e->-0.5*sum(abs2.(e)./σ^2), e)
    offset = maximum(w)
    log(sum(w->exp(w - offset), w)) + offset - log(np)
end

Base.size(b::Zygote.Buffer) = size(b.data)
const wbuf = Zygote.Buffer(randn(Float32,nz,np))
function samplenet(μσ)
    μ = μσ[1:end÷2,:]
    σ = exp.(μσ[(end÷2+1):end,:])
    # wbuf.freeze = false
    # for i in 1:np
    #     wbuf[:,i] = μ .+ σ .* randn.(Float32)
    # end
    w = μ .+ σ .* randn(Float32, nz, np)
    μ, σ, w
end


function loss(i,y)
    T = length(y)
    c = min(1, 0.01 + i/Ta)
    μ, σ, w = samplenet(w0([y[1];y[2];y[3];y[4]]))
    z = z0(w) # The first state should be a sample!
    l1 = 0f0
    l2 = sum(klng.(μ, σ, c))
    for t in 1:length(y)-1
        ŷ   = g(z)
        e   = y[t] .- ŷ
        l1 -= partlik(e, 0.05)
        # l1 += kl(e, zeros(ny), 0.05*ones(ny))
        # l1 += varloss(e)
        # l1 += varloss(e, [0.05, 0.05])
        # l1 += sum(x->norm(x)^2,e)/np
        μ, σ, w = samplenet(k(z,y[t+1]))
        # μ,σ2 = stats(zc)
        l2 += kl(w, zeros(nz), ones(nz), c)
        # l2 += sum(klng.(μ, σ, c))/np
        # l2  += klng(mean(μ, dims=2), mean(σ, dims=2), c)
        # ŷ   = g(zc)
        # e   = y[t] .- ŷ
        # l2 += varloss(e)
        # l2 += sum(x->norm(x)^2,e)/np
        z = f(z, w)
    end
    Float32(c*l1/T), Float32(l2/T)
end
# ll = loss(y)[2]

# loss(first(datas)...)
opt = ADAGrad(0.01f0)
sched = I -> (I ÷ 2000) % 2 == 0 ? 0.01 : 0.05
# opt = RMSProp(0.005f0)
# opt = Nesterov(0.001)
# opt = ADAM(0.01)
Zygote.refresh()
(l1,l2), back = Zygote.forward(()->loss(1,trajs_meas[1]), pars)
grads = back((1f0,1f0))
# grads = Zygote.gradient(()->loss(trajs[1]), pars)
train(loss, pars, IterTools.ncycle(trajs_meas, 1000), opt, cb=cb, schedule=sched)
##
i = 3
yh,zh,zp = sim(trajs_meas[i], false)
plot(reduce(hcat,trajs_meas[i])', layout=2)
scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.5,:black), sp=1, markerstrokecolor=:auto)
scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.5,:black), sp=2, markerstrokecolor=:auto) |> display

## Plot tubes

Z = map(1:length(trajs_state)) do i
    yh,zh,_ = sim(trajs_meas[i])
    zh = reduce(hcat, zh)'
    zmat = reduce(hcat,trajs_state[i])'[1:end-1,:]
    zmat[:,1] .= mod2pi.(zmat[:,1])
    zh,zmat
end

zh   = reduce(vcat, getindex.(Z,1))
zmat = reduce(vcat, getindex.(Z,2))
s    = svd(zh .- mean(zh, dims=1))
# zh   = s.U.*s.S'

fig = plot(layout=2)
scatter3d!(eachcol(zh)..., m=(2,), zcolor=zmat[:,1], sp=1, markerstrokealpha=0, layout=2)
scatter3d!(eachcol(zh)..., m=(2,), zcolor=zmat[:,2], sp=2, markerstrokealpha=0)
display(fig)

## Plot correlations
plots = map(Iterators.product(eachcol(zh), eachcol(zmat))) do (zh,z)
    scatter(z,zh, m=(0.5,2))
end
plot(plots...) |> display


## Plot particles


i = 1
yh,zh,zp = sim(trajs_meas[i])
zmat = reduce(hcat,trajs_state[i])'
plots = map(1:nz) do j
    global zp
    # zh = reduce(hcat, getindex.(zh,j,:))'
    zpj = reduce(hcat, getindex.(zp,j,:))'
    fig = plot(zmat)
    scatter!(zpj, m=(2,:black,0.5), markerstrokealpha=0)
end
plot(plots...)
