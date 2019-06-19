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

include("dps_utils.jl")
# Zygote.refresh()
# Zygote.gradient(e->kl(e,dy), randn(1,10))

##
trajs_full = [generate_data_pendcart(4) for i = 1:80]
# trajs_full = [generate_data_pendcart(5, [pi-0.1, 0]), generate_data_pendcart(5, [pi+0.1, 0])]
Y = map(trajs_full) do (_,t,_)
    [copy(c) for c in eachcol(t)]
end
U = map(trajs_full) do (_,_,t)
    [copy(c) for c in eachcol(t)]
end
Z = map(trajs_full) do (t,_,_)
    [copy(c) for c in eachcol(t)]
end

Base.@kwdef struct OptTrace
    loss1::Vector{Float64} = Float64[]
    loss2::Vector{Float64} = Float64[]
end
Base.push!(ot::OptTrace, l1, l2) = (push!(ot.loss1, l1); push!(ot.loss2, l2);)
Base.length(ot::OptTrace) = length(ot.loss1)

struct DeepFilter{Tf, Tg, Tw, Tz, Tk}
    fn::Tf
    g::Tg
    w0::Tw
    z0::Tz
    kn::Tk
end
Flux.params(df::DeepFilter) = params((df.fn,df.g,df.kn,df.z0,df.w0))







function k(kn,z,e)
    # kn([z;mean(y)*ones(1,np)])
    kn([z;e])
end
f(fn,z,u,zc) = fn([z;u[]*ones(1,np);zc]) + 0.9z

function DeepFilter(ny::Int,nu::Int,nz::Int,nh::Int)

    dy2 = MvNormal(2, 0.05f0)

    np = 1
    fn  = Chain(Dense(2nz+nu,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    g  = Chain(Dense(nz,nh,tanh), Dense(nh,ny))
    # attention = Chain(Dense(nz+ny, nz), LSTM(nz, np), softmax)
    # attend(z,e) = sum(attention([z;e])' .* z, dims=2)
    z0 = Chain(Dense(nz,nh,tanh), Dense(nh,nz))
    w0 = Chain(Dense(4ny,nh,tanh), Dense(nh,2nz))
    # w0[2].b[nz+1:end] .= 1
    kn  = Chain(Dense(nz+ny,nh,tanh), Dense(nh,nh,tanh), Dense(nh,nz))
    df = DeepFilter(fn,g,w0,z0,kn)
end

cb = function (i=0)
    i % 1000 == 0 || return
    lm = [ot.loss1 ot.loss2]
    # lm = length(loss1) > Ta ? lm[Ta:end,:] : lm
    # lm = filt(ones(80), [80], lm, fill(lm[1,1], 79))
    fig = plot(lm, layout=@layout([[a;b] c]), sp=1:2, yscale=minimum(lm) < 0 ? :identity : :log10)

    z,y,u = generate_data_pendcart(5, [pi+0.1, 0])
    yh,_,_ = sim(df,y,u)
    ##
    plot!(reduce(hcat,y)', sp=3)
    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.2,:green), sp=3, markerstrokecolor=:auto)
    # scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:green), sp=3, markerstrokecolor=:auto)
    yh,zh,zp = sim(df,y,u, false, true)
    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.2,:black), sp=3, markerstrokecolor=:auto)
    # scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.2,:black), sp=3, markerstrokecolor=:auto)
    display(fig)
end

##

function sim(df, y, u, feedback=true, noise=true)
    fn,g,kn,z0,w0 = df.fn,df.g,df.kn,df.z0,df.w0
    # y,u = yu
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
        zc  = k(kn, z,feedback.*e)
        # μ, σ, zc = samplenet(k(z,e))
        # μ, σ, zc = samplenet(k(z,y[t]))
        push!(zh, mean(z, dims=2)[:])
        push!(zp, z)
        # ŷ   = g(z)
        # push!(yh2, ŷ)
        z   = f(fn, z,u[t], zc)
    end
    yh, zh, zp
end


function loss(i,y,u,df,Ta)
    fn,g,kn,z0,w0 = df.fn,df.g,df.kn,df.z0,df.w0
    # y,u = yu
    T = length(y)
    c = min(1, 0.01 + i/Ta)
    z = z0(samplenet(w0([y[1];y[2];y[3];y[4]]))[3])
    l1 = l2 = 0f0
    # l2 = sum(klng.(μ, σ, c))
    for t in 1:length(y)-1
        ŷ   = g(z)
        e   = y[t] .- ŷ
        l1 -= partlik(e, 0.05)
        # μ, σ, zc  = samplenet(k(z,e))
        zc  = k(kn,z,e)
        # μ, σ, zc  = samplenet(k(z,y[t]))
        # μ,σ2 = stats(zc)
        # ŷ   = g(z)
        # e   = y[t] .- ŷ
        # l2 += sum(klng.(μ, σ, c))
        l2 += sum(x->norm(x)^2,zc)/np
        z = f(fn,z,u[t], zc)
    end
    Float32(c*l1/T), Float32(l2/T)
end
function train(df, y, u, epochs, opt;
    cb=i->nothing,
    schedule=I->copy(opt.eta),
    bs=1,
    Ta = 4000,
    ot = OptTrace())

    ps = params(df)
    dataset = collect(zip(y, u))
    @assert length(dataset) % bs == 0 "bs must divide length(y)"
    @progress for epoch = 1:epochs
        for i = 1:bs:length(dataset)-bs+1
            I = length(ot)+1
            opt.eta = schedule(I)
            # Flux.reset!(model)
            (l1,l2), back = Zygote.forward(ps) do
                mean(loss(I, d..., df,Ta) for d in dataset[i:i+bs-1])
            end
            grads = back((1f0,1f0))
            push!(ot, l1, l2)
            Flux.Optimise.update!(opt, ps, grads)
            cb(I*bs)
        end
    end
    ot
end

opt = ADAGrad(0.01f0)
ot = OptTrace()
sched = I -> (I ÷ 500) % 2 == 0 ? 0.01 : 0.01
df = DeepFilter(1,1,3,30)
# opt = RMSProp(0.005f0)
# opt = Nesterov(0.001, 0.5)
# opt = ADAM(0.01)
Zygote.refresh()
# (l1,l2), back = Zygote.forward(()->loss(1,YU[1]), pars)
# grads = back((1f0,1f0))
# grads = Zygote.gradient(()->loss(trajs[1]), pars)
train(df, Y, U, 1000, opt, cb=cb, schedule=sched, bs=1, ot = ot)
##
Random.seed!(123)
plots = map(1:9) do i
    # i = 10
    z,y,u = generate_data_pendcart(5)
    yt = cos.(z[1,:])
    yh,zh,zp = sim(df,y,u, false, true)
    YH = reduce(hcat,mean.(yh, dims=2)[:])'
    plot(reduce(hcat,yt)', layout=1, l=(2,))
    scatter!(reduce(hcat, getindex.(yh, 1, :))', m=(2,0.1,:black), sp=1, markerstrokecolor=:auto)
    # scatter!(reduce(hcat, getindex.(yh, 2, :))', m=(2,0.5,:black), sp=2, markerstrokecolor=:auto)
    plot!(YH, l=(2,), xaxis=false, ylims=(-1.1,1.1))
    plot!(reduce(hcat,u)', l=(2,0.2,:green))
end
plot(plots...) |> display

## Plot tubes

Zs = map(1:length(Z)) do i
    yh,zh,_ = sim(df,Y[i],U[i], true, false)
    zh = reduce(hcat, zh)'
    zmat = reduce(hcat,Z[i])'[1:end-1,:]
    zmat[:,1] .= mod2pi.(zmat[:,1])
    zh,zmat
end

zh   = reduce(vcat, getindex.(Zs,1))
zmat = reduce(vcat, getindex.(Zs,2))
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

z,y,u = generate_data_pendcart(5, [pi-0.1, 0])
i = 1
# yh,zh,zp = sim((YU[i][1], YU[i][2]), false, true)
# zmat = reduce(hcat,trajs_state[i])'
yh,zh,zp = sim(df,y,u, false, true)
zmat = z'
YH = mean(reduce(vcat, yh), dims=2)
plots = map(1:size(z,1)) do j
    global zp
    # zh = reduce(hcat, getindex.(zh,j,:))'
    zpj = reduce(hcat, getindex.(zp,j,:))'
    fig = plot(zmat)
    yt = cos.(zmat[:,1])
    plot!(YH, lab="")
    plot!(yt)
    plot!(u')
    scatter!(zpj, m=(2,:black,0.5), markerstrokealpha=0, ylims=extrema(zmat))
end
plot(plots...)

##
entropy(a::Vector) = -sum(normpdf(x)*normlogpdf(x) for x in a) / length(a)
entropy(σ::Number) = 0.5log(2π*ℯ*σ^2)
quota(σ,N) = entropy(σ) / entropy(σ*randn(N))



# QUESTION: how can one construct a reparametrization trick for multimodal distributions?
