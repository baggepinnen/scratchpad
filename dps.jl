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

@changeprecision Float32 function kl(e, dy, c = 1)
    μ1 = mean(e, dims=2)
    μ2  = mean(dy)
    # lσ1 = log(std(reduce(hcat,e), dims=2))
    lσ1 = log.(sqrt.(sum(abs2, e .- μ1, dims=2) ./ size(e,2)))
    lσ2 = log.([0.05, 0.05])#log.(sqrt.(var(dy)))
    l2π = log(2π)
    l = 0f0
    for i = eachindex(μ1)
        l += c*(l2π + 2lσ2[i]) - (l2π + 2lσ1[i]) +
        c*(exp(2lσ1[i]) + abs2(μ1[i] - μ2[i]))/(exp(2lσ2[i]) + 1f-5) - 1f0
    end
    0.5f0l
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
N  = 50
np = 50
const g  = Chain(Dense(nz,N,tanh), Dense(N,ny))
const z0 = Chain(Dense(6ny,N,tanh), Dense(N,nz))
const fn  = Chain(Dense(2nz+nu,N,tanh), Dense(N,nz))
f(z,noise,t) = fn([z;noise]) + z #.+ noise#(z,noise,t) -> fn(z) .+  8cos(1.2*t) .+ noise
const kn  = Chain(Dense(nz+ny,N,tanh), Dense(N,nz))
k(z,e) = kn([z;e]) #+ z
pars = params((fn,g,kn,z0))
# -1.25, -2.5
true_obsmodel(x) = 0.05 * x^2#

function train(loss, ps, dataset, opt; cb=i->nothing)
    @progress for (i,d) in enumerate(dataset)
        # Flux.reset!(model)
        (l1,l2), back = Zygote.forward(()->loss(d), ps)
        grads = back((1f0, 1f0))
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
    i % 100 == 0 || return
    lm = filt(ones(80), [80], [loss1 loss2])
    fig = plot(lm, layout=3, sp=1, yscale=minimum(lm) < 0 ? :identity : :log10)

    yh1,yh2,_,_ = sim(trajs_meas[1])
    ##
    plot!(reduce(hcat,trajs_meas[1])', sp=2:3)
    scatter!(reduce(hcat, getindex.(yh2, 1, :))', m=(2,0.05,:green), sp=2, markerstrokecolor=:auto)
    scatter!(reduce(hcat, getindex.(yh2, 2, :))', m=(2,0.05,:green), sp=3, markerstrokecolor=:auto)

    yh1,yh2,zh,zp = sim(trajs_meas[1], false)

    scatter!(reduce(hcat, getindex.(yh1, 1, :))', m=(2,0.05,:black), sp=2, markerstrokecolor=:auto)
    scatter!(reduce(hcat, getindex.(yh1, 2, :))', m=(2,0.05,:black), sp=3, markerstrokecolor=:auto) |> display
    display(fig)

end

##

function sim(y, feedback=true)
    T   = length(y)
    z   = z0(vcat(y[1:6]...)) .+ randn(nz,np)
    yh1 = []
    yh2 = []
    zh  = []
    zp  = []
    for t in eachindex(y)
        ŷ   = g(z)
        push!(yh1, ŷ)
        e   = y[t] .- ŷ
        feedback && (z = k(z,e))
        push!(zh, mean(z, dims=2)[:])
        push!(zp, z)
        ŷ   = g(z)
        push!(yh2, ŷ)
        z   = f(z, randn(nz,np), t)
    end
    yh1, yh2, zh, zp
end

function varloss(e)
    μ = mean(e, dims=2)
    σ2 = sum(abs2, e .- μ, dims=2) ./ size(e,2) .+ 1f-3
    sum(abs2(μ[i])/σ2[i] + log(σ2[i])  for i in eachindex(μ))
end

function loss(y)
    T = length(y)
    z = z0([y[1];y[2];y[3];y[4];y[5];y[6]]) .+ randn(Float32,nz,np)
    l1 = l2 = 0f0
    for t in eachindex(y)
        ŷ   = g(z)
        e   = y[t] .- ŷ
        # l1 += varloss(e)
        l1 += sum(x->norm(x)^2,e)/np
        # l1 += kl(e, dy2)
        zc   = k((z),(e))
        ŷ   = g(zc)
        e   = y[t] .- ŷ
        # l2 += varloss(e)
        l2 += sum(x->norm(x)^2,e)/np
        # l2 += kl(e, dy2)
        α  = 0.1#max(0.01, 1 - 0.01t)
        zf = zc*α + z*(1-α)
        z = f(zf, randn(Float32,nz,np), t)
    end
    Float32(l1/T), Float32(l2/T)
end
# ll = loss(y)[2]

# loss(first(datas)...)
# opt = ADAGrad(0.02f0)
opt = RMSProp(0.005f0)
# opt = Nesterov(0.001)
# opt = ADAM(0.02)
Zygote.refresh()
# (l1,l2), back = Zygote.forward(()->loss(trajs_meas[1]), pars)
# grads = back((1f0,1f0))
# grads = Zygote.gradient(()->loss(trajs[1]), pars)
train(loss, pars, IterTools.ncycle(trajs_meas, 1000), opt, cb=cb)
yh1,yh2,zh,zp = sim(trajs_meas[1], false)

##
plot(reduce(hcat,trajs_meas[1])', layout=2)
scatter!(reduce(hcat, getindex.(yh1, 1, :))', m=(2,0.05,:black), sp=1, markerstrokecolor=:auto)
scatter!(reduce(hcat, getindex.(yh1, 2, :))', m=(2,0.05,:black), sp=2, markerstrokecolor=:auto) |> display
##
plot(reduce(hcat,trajs_meas[1])', layout=2)
scatter!(reduce(hcat, getindex.(yh2, 1, :))', m=(2,0.05,:green), sp=1, markerstrokecolor=:auto)
scatter!(reduce(hcat, getindex.(yh2, 2, :))', m=(2,0.05,:green), sp=2, markerstrokecolor=:auto) |> display

## Plot tubes

Z = map(1:length(trajs_state)) do i
    yh1,yh2,zh = sim(trajs_meas[i])
    zh = reduce(hcat, zh)'
    zmat = reduce(hcat,trajs_state[i])'
    zh,zmat
end

zh   = reduce(vcat, getindex.(Z,1))
zmat = reduce(vcat, getindex.(Z,2))
s    = svd(zh .- mean(zh, dims=1))
# zh   = s.U.*s.S'

fig = plot(layout=2)
scatter3d!(eachcol(zh)..., m=(2,), zcolor=mod2pi.(zmat[:,1]), sp=1, markerstrokealpha=0, layout=2)
scatter3d!(eachcol(zh)..., m=(2,), zcolor=zmat[:,2], sp=2, markerstrokealpha=0)
display(fig)



## Plot particles


i = 1
yh1,yh2,zh,zp = sim(trajs_meas[i])
zmat = reduce(hcat,trajs_state[i])'
plots = map(1:nz) do j
    global zp
    # zh = reduce(hcat, getindex.(zh,j,:))'
    zpj = reduce(hcat, getindex.(zp,j,:))'
    fig = plot(zmat)
    scatter!(zpj, m=(2,:black,0.1), markerstrokealpha=0)
end
plot(plots...)
