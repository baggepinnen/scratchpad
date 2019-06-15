# TODO: På vilka ställen interagerar olika partiklar med varandra?
# Gaussian ll kan hanteras analytiskt
# kan man träna modellen åp detta sättet först och sedan träna en klassisk observerare utan att rucka på f?
# Vi kan hantera godtycklig dg, men är det vettigt att alltid anta Gaussisk df? z_post borde ändå kunna vara multimodal, tänk pf-testmodellen
cd(@__DIR__)
using Plots, Flux, Zygote, LinearAlgebra, Statistics, Random, Printf, OrdinaryDiffEq, IterTools, Distributions, ChangePrecision, DSP
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
    l2π = log(2π)
    l = 0f0
    for i = eachindex(μ1)
        l += c*(l2π + 2lσ2[i]) - (l2π + 2lσ1[i]) +
        c*(σ1²[i] + abs2(μ1[i] - μ2[i]))/(σ2[i]^2 + 1f-5) - 1f0
    end
    0.5f0l
end

@changeprecision Float32 function klng(μ1, lσ1, c = 1)
    σ1² = exp.(lσ1).^2
    # lσ1 = log.(σ1)
    l2π = log(2π)
    l = 0f0
    for i = eachindex(μ1)
        l += c*l2π - (l2π + 2lσ1[i]) +
        c*(σ1²[i] + abs2(μ1[i])) - 1f0
    end
    0.5f0l
end

@changeprecision Float32 function kl2(e, c = 1)
    μ1, σ1² = stats(e)
    l2π = log(2π)
    l = 0f0
    for i = eachindex(μ1)
        l += c*l2π - l2π +  c*(1 + abs2(μ1[i])) - 1f0
    end
    0.5f0l
end
# Zygote.refresh()
# Zygote.gradient(e->kl(e,dy), randn(1,10))

##
trajs_full = [generate_data_pendcart(2) for i = 1:80]
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
nh  = 30
np = 1
const g  = Chain(Dense(nz,nh,tanh), Dense(nh,ny))
const z0 = Chain(Dense(4ny,nh,tanh), Dense(nh,nz))
# const w0 = Chain(Dense(4ny,nh,tanh), Dense(nh,2nz))
const fn  = Chain(Dense(2nz+nu,nh,tanh), Dense(nh,nz))
f(z,noise) = fn([z;noise]) + 0.5z #(z,noise,t) -> fn(z) .+  8cos(1.2*t) .+ noise
const kn  = Chain(Dense(nz+ny,nh,tanh), Dense(nh,nz))
function k(z,e)
    kn([z;e])
end

pars = params((fn,g,kn,z0))
parsim = params((fn,g,z0))

function train(loss, ps, pss, dataset, opt, opts; cb=i->nothing)
    @progress for (i,d) in enumerate(dataset)
        # Flux.reset!(model)
        (l1,l2), back = Zygote.forward(()->loss(length(loss1)+1, d), ps)
        grads = back((1f0,1f0))
        Flux.Optimise.update!(opt, ps, grads)
        l4, back = Zygote.forward(()->simloss(length(loss1)+1, d), pss)
        grads = back(1f0)
        Flux.Optimise.update!(opts, pss, grads)
        push!(loss1, l1)
        push!(loss2, l2)
        # push!(loss3, l3)
        push!(loss4, l4)
        cb(i)
    end
end

const ⊗ = Zygote.dropgrad

loss1 = Float64[]
loss2 = Float64[]
loss3 = Float64[]
loss4 = Float64[]

cb = function (i=0)
    i % 10000 == 0 || return
    lm = [loss1 loss2 loss4]
    # lm = length(loss1) > Ta ? lm[Ta:end,:] : lm
    lm = filt(ones(80), [80], lm, fill(lm[1,1], 79))
    fig = plot(lm, layout=3, sp=1, yscale=minimum(lm) < 0 ? :identity : :log10)

    yh,yh2,_,_ = sim(trajs_meas[1])
    ##
    plot!(reduce(hcat,trajs_meas[1])', sp=2:3)
    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.2,:red), sp=2, markerstrokecolor=:auto)
    scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:red), sp=3, markerstrokecolor=:auto)
    scatter!(reduce(hcat, getindex.(yh2, 1, :))', m=(2,0.2,:green), sp=2, markerstrokecolor=:auto)
    scatter!(reduce(hcat, getindex.(yh2, 2, :))', m=(2,0.2,:green), sp=3, markerstrokecolor=:auto)

    yh,yh2,zh,zp = sim(trajs_meas[1], false)

    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.2,:black), sp=2, markerstrokecolor=:auto)
    scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:black), sp=3, markerstrokecolor=:auto)
    display(fig)
end

##

function sim(y, feedback=true)
    T   = length(y)
    z   = z0([y[1];y[2];y[3];y[4]])
    yh = []
    yh2 = []
    zh  = []
    zp  = []
    for t in 1:length(y)-1
        ŷ   = g(z)
        push!(yh, ŷ)
        e   = y[t] .- ŷ
        zc = k(z,feedback.*e)
        z += zc
        push!(zh, mean(z, dims=2)[:])
        push!(zp, z)
        ŷ   = g(z)
        push!(yh2, ŷ)
        z   = f(z, randn(nz,np))
    end
    yh, yh2, zh, zp
end

function varloss(e,σ)
    μ = mean(e, dims=2)
    sum(0.5*(abs2(μ[i])/σ[i]^2 + log(2π)) + 0*log(σ[i])  for i in eachindex(μ))
end
function varloss(e)
    μ,σ² = stats(e)
    sum(0.5*(abs2(μ[i])/σ²[i] + log(2π)) + log(√(σ²[i]))  for i in eachindex(μ))
end

function samplenet(μσ,r,c)
    μ = μσ[1:end÷2]
    σ = exp.(μσ[(end÷2+1):end])
    w = μ .+ σ .* randn(Float32,r,c)
    μ, σ, w
end

function loss(i,y)
    T = length(y)
    c = min(1, 0.01 + i/Ta)
    z = z0([y[1];y[2];y[3];y[4]]) .+ 0.01randn(nz,np)
    l1 = l2 = l3 = 0f0
    for t in 1:length(y)-1
        ŷ   = g(z)
        e   = y[t] .- ŷ
        # l1 += varloss(e, [0.05, 0.05])
        l1 += sum(x->norm(x)^2,e)/np
        zc  = k(z,e)
        z   = z + zc
        l3 += sum(abs2, zc)/np
        ŷ   = g(z)
        e   = y[t] .- ŷ
        l2  += sum(x->norm(x)^2,e)/np
        z = f(⊗(z), randn(nz,np))
    end
    Float32(l1/T), Float32(l2/T)
end
function simloss(i,y)
    T = length(y)
    c = min(1, 0.01 + i/Ta)
    z = z0([y[1];y[2];y[3];y[4]]) .+ randn(nz,np)
    l1  = 0f0
    for t in 1:length(y)-1
        ŷ   = g(z)
        e   = y[t] .- ŷ
        # l1 += varloss(e, [0.05, 0.05])
        l1 += sum(x->norm(x)^2,e)/np
        z = f(z, randn(nz,np))
    end
    Float32(l1/T)
end
# ll = loss(y)[2]

# loss(first(datas)...)
opt = ADAGrad(0.05f0)
opts = ADAGrad(0.05f0)
# opt = RMSProp(0.005f0)
# opt = Nesterov(0.001)
# opt = ADAM(0.02)
Zygote.refresh()
# (l1,l2), back = Zygote.forward(()->loss(1,trajs_meas[1]), pars)
# grads = back((1f0,1f0,1f0))
# grads = Zygote.gradient(()->loss(trajs[1]), pars)
train(loss, pars, parsim, IterTools.ncycle(trajs_meas, 2000), opt, opts, cb=cb)

##
i = 3
yh,yh2,zh,zp = sim(trajs_meas[i], true)
plot(reduce(hcat,trajs_meas[i])', layout=2)
scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.5,:red), sp=1, markerstrokecolor=:auto)
scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.5,:red), sp=2, markerstrokecolor=:auto)
scatter!(reduce(hcat, getindex.(yh2, 1, :))', m=(2,0.5,:green), sp=1, markerstrokecolor=:auto)
scatter!(reduce(hcat, getindex.(yh2, 2, :))', m=(2,0.5,:green), sp=2, markerstrokecolor=:auto)
yh,yh2,zh,zp = sim(trajs_meas[i], false)
scatter!(reduce(hcat, getindex.(yh2, 1, :))', m=(2,0.5,:black), sp=1, markerstrokecolor=:auto)
scatter!(reduce(hcat, getindex.(yh2, 2, :))', m=(2,0.5,:black), sp=2, markerstrokecolor=:auto) |> display

## Plot tubes

Z = map(1:length(trajs_state)) do i
    yh,yh2,zh,_ = sim(trajs_meas[i])
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
yh,yh2,zh,zp = sim(trajs_meas[i])
zmat = reduce(hcat,trajs_state[i])'
plots = map(1:nz) do j
    global zp
    # zh = reduce(hcat, getindex.(zh,j,:))'
    zpj = reduce(hcat, getindex.(zp,j,:))'
    fig = plot(zmat)
    scatter!(zpj, m=(2,:black,0.5), markerstrokealpha=0)
end
plot(plots...)
