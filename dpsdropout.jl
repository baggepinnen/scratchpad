# TODO: På vilka ställen interagerar olika partiklar med varandra? Enligt Berntorp är feedback gain en funktion av partikelns yhat och medelpartikelns yhat
# Gaussian ll kan hanteras analytiskt
# kan man träna modellen på detta sättet först och sedan träna en klassisk observerare utan att rucka på f?
# Vi kan hantera godtycklig dg, men är det vettigt att alltid anta Gaussisk df? z_post borde ändå kunna vara multimodal, tänk pf-testmodellen
# Är det vettigt att öka nz?
cd(@__DIR__)
using Plots, Flux, Zygote, LinearAlgebra, Statistics, Random, Printf, OrdinaryDiffEq, IterTools, Distributions, ChangePrecision, DSP, SliceMap
using Flux: params
default(lab="", grid=false)
# include("SkipRNN.jl")
Random.seed!(0)


const h = 0.1
const Ta = 4000

function hstack(xs, n)
    buf = Zygote.Buffer(xs, length(xs), n)
    for i = 1:n
        buf[:, i] = xs
    end
    return copy(buf)
end

struct SawtoothGenerator{T <: Real} <: Function
    f::T #Frequency
    p::T #Phase
end
sawfun(t) = t - floor(t)
SawtoothGenerator(f)    = SawtoothGenerator(f, 0)
# SawtoothGenerator(f, p) = SawtoothGenerator{typeof(f)}(f, p)

function (ref::SawtoothGenerator)(t::Real)
    sawfun(ref.f*t+ref.p)
end

saw = SawtoothGenerator(1)
function controller(x, t)
    l = 1.0; d = 0.5
    t < 1 && (return 0.)
    u = 0#t > 1 ? 4(saw(t)-0.5) : 0.
    # u += - 2θ - 2x[2]
    u += l*(d+0.1)*x[2]/cos(x[1])
    clamp(u, -5, 5)
end


@changeprecision Float32 function pendcart(xd,x,p,t)
    g = 9.82; l = 1.0; d = 0.5
    u = controller(x, t)
    xd[1] = x[2]
    xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1]) - d*x[2]
    xd
end

function centerangle(x)
    c = round(Int, median(x)/(2π))
    mod2pi.(x .- (2π*c - π)) .- π
end

@changeprecision Float32 function generate_data_pendcart(T,
    u0 = @. Float32[pi, 6] * 2 *(rand(Float32)-0.5))
    tspan = (0f0,Float32(T))
    prob = ODEProblem(pendcart,u0,tspan)
    sol = solve(prob,Tsit5())
    z = reduce(hcat, sol(0:h:T).u)
    # y = vcat(abs.(sin.(z[1:1,:])), cos.(z[1:1,:])) .+ 0.05 .* randn.()
    y = cos.(z[1:1,:]) .+ 0.05 .* randn.()
    z[1,:] .= centerangle(z[1,:])
    z,y, controller.(eachcol(z), 0:h:T)'
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
    0.5c*(sum(abs2,μ1) + sum(σ1²)) + sum(log, σ1²)
end
# Zygote.refresh()
# Zygote.gradient(e->kl(e,dy), randn(1,10))

##
# trajs_full = [generate_data_pendcart(3) for i = 1:80]
trajs_full = [generate_data_pendcart(5, [pi-0.1, 0]), generate_data_pendcart(5, [pi+0.1, 0])]
trajs_meas = map(trajs_full) do (_,t,_)
    [copy(c) for c in eachcol(t)]
end
trajs_u = map(trajs_full) do (_,_,t)
    [copy(c) for c in eachcol(t)]
end
trajs_state = map(trajs_full) do (t,_,_)
    [copy(c) for c in eachcol(t)]
end

YU = collect(zip(trajs_meas, trajs_u))

const dy2 = MvNormal(2, 0.05f0)
const nz = 3
const ny = 1
const nu = 1
nh  = 30
np = 20 # 10
# const nα = 5
# const αnet = Chain(Dense(nz,nα), softmax)
# const A = [(0.001randn(nz,nz)) for _ = 1:nα]
# const C = [(0.001randn(nz,nz)) for _ = 1:nα]
# function f(z,w)
#     α  = αnet(z)
#     Ai = sum(α.*A)
#     Ci = sum(α.*C)
#     z  = Ai*z + Ci*w
# end
const fn  = Chain(Dense(2nz+nu,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
f(z,u,noise) = fn([z;u[]*ones(1,np);noise]) + 0.9z
const g  = Chain(Dense(nz,nh,tanh), Dense(nh,ny))
const z0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
const w0 = Chain(Dense(4ny,nh,tanh), Dense(nh,2nz))
# w0[2].b[nz+1:end] .= 1
const kn  = Chain(Dense(nz+ny,nh,tanh), Dense(nh,nz))
function k(z,e)
    kn([z;e])
    # kn([z;e;mean(e)*ones(1,np)])
end
pars = params((fn,g,kn,z0,w0))

Base.:(+)(a::Tuple{Float32,Float32}, b::Tuple{Float32,Float32}) = (a[1]+b[1], a[2]+b[2])
Base.:(/)(a::Tuple{Float32,Float32}, b::Real) = (a[1]/b, a[2]/b)
function train(loss, ps, dataset, epochs, opt; cb=i->nothing, schedule=I->copy(opt.eta), bs=2)
    @assert length(dataset) % bs == 0 "bs must divide length(dataset)"
    @progress for epoch = 1:epochs
        for i = 1:bs:length(dataset)-bs+1
            I = length(loss1)+1
            opt.eta = schedule(I)
            # Flux.reset!(model)
            (l1,l2), back = Zygote.forward(ps) do
                mean(loss(I, d) for d in dataset[i:i+bs-1])
            end
            grads = back((1f0,1f0))
            push!(loss1, l1)
            push!(loss2, l2)
            Flux.Optimise.update!(opt, ps, grads)
            cb(I*bs)
        end
    end
end

const ⊗ = Zygote.dropgrad

loss1 = Float64[]
loss2 = Float64[]

cb = function (i=0)
    i % 1000 == 0 || return
    lm = [loss1 loss2]
    # lm = length(loss1) > Ta ? lm[Ta:end,:] : lm
    # lm = filt(ones(80), [80], lm, fill(lm[1,1], 79))
    fig = plot(lm, layout=@layout([[a;b] c]), sp=1:2, yscale=minimum(lm) < 0 ? :identity : :log10)

    yh,_,_ = sim(YU[1])
    ##
    plot!(reduce(hcat,YU[1][1])', sp=3)
    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.2,:green), sp=3, markerstrokecolor=:auto)
    # scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:green), sp=3, markerstrokecolor=:auto)

    yh,zh,zp = sim(YU[1], false, true)

    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.2,:black), sp=3, markerstrokecolor=:auto)
    # scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:black), sp=3, markerstrokecolor=:auto)
    display(fig)
end

##

function sim(yu, feedback=true, noise=true)
    y,u = yu
    T   = length(y)
    z   = z0(samplenet(w0([y[1];y[2];y[3];y[4]]),noise)[3])
    yh = []
    # yh2 = []
    zh  = []
    zp  = []
    for t in 1:length(y)-1
        ŷ   = g(z)
        push!(yh, ŷ)
        e   = y[t] .- ŷ
        zc = k(z,feedback.*e)
        push!(zh, mean(z, dims=2)[:])
        push!(zp, z)
        # ŷ   = g(z)
        # push!(yh2, ŷ)
        z   = f(z,u[t], zc + noise*randn(nz,np))
    end
    yh, zh, zp
end

function partlik(e, σ)
    w = mapcols(e->-0.5*sum(abs2.(e)./σ^2), e)
    offset = maximum(w)
    log(sum(w->exp(w - offset), w)) + offset - log(np)
end

function samplenet(μσ, noise=true)
    μ = μσ[1:end÷2,:]
    σ = exp.(μσ[(end÷2+1):end,:])
    w = μ .+ σ .* randu(nz, np).*noise
    μ, σ, w
end

function drop(e)
    mapcols(e->rand((0,1))*e, e)
end

Zygote.@adjoint function drop(e)
    mask = rand((0,1), 1, np)
    e.*mask, x->(x.*mask,)
end

Zygote.@nograd function randu(nz,np)
    rand(Float32, nz, np) .- 0.5
    # randn(Float32, nz, np)
end

function loss(i,yu)
    y,u = yu
    T = length(y)
    c = min(1, 0.01 + i/Ta)
    z = z0(samplenet(w0([y[1];y[2];y[3];y[4]]))[3]) # TODO: this is not the same as what we had before, where the sample went into a network
    # z = z0(w) # The first state should be a sample!
    l1 = l2 = 0f0
    # l2 = sum(klng.(μ, σ, c))
    for t in 1:length(y)-1
        ŷ   = g(z)
        e   = y[t] .- ŷ
        l1 -= partlik(e, 0.05)
        zc  = k(z,e)
        # μ,σ2 = stats(zc)
        # ŷ   = g(z)
        # e   = y[t] .- ŷ
        # l2 += varloss(e)
        # l2 += kl2(zc, c)
        # l2 -= partlik(e, 0.05)
        l2 += sum(x->norm(x)^2,zc)/np #- c*0.1*log(sum(stats(z)[2])) # TODO: ADDED THIS AS EXPERIMENT
        z = f(z,u[t], zc + randu(nz,np))
    end
    Float32(c*l1/T), Float32(l2/T)
end
# ll = loss(y)[2]

# loss(first(datas)...)
opt = ADAGrad(0.01f0)
sched = I -> (I ÷ 500) % 2 == 0 ? 0.01 : 0.01
# opt = RMSProp(0.005f0)
# opt = Nesterov(0.001, 0.5)
# opt = ADAM(0.01)
Zygote.refresh()
# (l1,l2), back = Zygote.forward(()->loss(1,YU[1]), pars)
# grads = back((1f0,1f0))
# grads = Zygote.gradient(()->loss(trajs[1]), pars)
train(loss, pars, YU, 1000, opt, cb=cb, schedule=sched, bs=1)
##
Random.seed!(123)
plots = map(1:9) do i
    # i = 10
    z,y,u = generate_data_pendcart(5)
    yt = cos.(z[1,:])
    yh,zh,zp = sim((y,u), false, true)
    YH = reduce(hcat,mean.(yh, dims=2)[:])'
    plot(reduce(hcat,yt)', layout=1, l=(2,))
    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.1,:black), sp=1, markerstrokecolor=:auto)
    # scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.5,:black), sp=2, markerstrokecolor=:auto)
    plot!(YH, l=(2,), xaxis=false, ylims=(-1.1,1.1))
    plot!(reduce(hcat,u)', l=(2,0.2,:green))
end
plot(plots...) |> display

## Plot tubes

Z = map(1:length(trajs_state)) do i
    yh,zh,_ = sim(YU[i], true, false)
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

z,y,u = generate_data_pendcart(5, [pi+0.1, 0])
i = 1
# yh,zh,zp = sim((YU[i][1], YU[i][2]), false, true)
# zmat = reduce(hcat,trajs_state[i])'
yh,zh,zp = sim((y,u), false, true)
zmat = z'
YH = mean(reduce(vcat, yh), dims=2)
plots = map(1:nz) do j
    global zp
    # zh = reduce(hcat, getindex.(zh,j,:))'
    zpj = reduce(hcat, getindex.(zp,j,:))'
    fig = plot(zmat)
    plot!(YH, lab="")
    plot!(u')
    scatter!(zpj, m=(2,:black,0.5), markerstrokealpha=0, ylims=extrema(zmat))
end
plot(plots...)

##
entropy(a::Vector) = -sum(normpdf(x)*normlogpdf(x) for x in a) / length(a)
entropy(σ::Number) = 0.5log(2π*ℯ*σ^2)
quota(σ,N) = entropy(σ) / entropy(σ*randn(N))
