cd(@__DIR__)
using Plots, Flux, Zygote, LinearAlgebra, Parameters, Statistics, Random, Printf, OrdinaryDiffEq, IterTools, Distributions
using Flux: params
default(lab="", grid=false)
# include("SkipRNN.jl")

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

function pendcart(xd,x,p,t)
    g = 9.82; l = 1.0; d = 0.5
    u = 0
    xd[1] = x[2]
    xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1]) - d*x[2]
    xd
end

function generate_data(T)
    u0 = @. [pi, 6] * 2 *(rand()-0.5)
    tspan = (0.0,T)
    prob = ODEProblem(pendcart,u0,tspan)
    sol = solve(prob,Tsit5())
    z = reduce(hcat, sol(0:h:T).u)
    y = vcat(sin.(z[1:1,:]), cos.(z[1:1,:])) .+ 0.05 .* randn.()
    z,y
end

Zygote.@adjoint function Base.reduce(::typeof(hcat), V::AbstractVector{<:AbstractVector})
    reduce(hcat, V), dV -> (nothing, collect(eachcol(dV)))
end

function std2(x::Matrix,μ)
    s = 0.
    for x in x
        s += abs2(x - μ)
    end
    sqrt(s/length(x))
end

function kl(e, dy, c = 1)
    μ1  = mean(e) # TODO: dims = 1
    μ2  = mean(dy)
    # lσ1 = log(std(reduce(hcat,e), dims=2))
    lσ1 = log(std2(e, μ1))  # TODO: dims = 1
    lσ2 = log(std(dy))
    l2π = log(2π)
    l = 0.
    for i = eachindex(μ1)
        l += c*(l2π + 2lσ2[i]) - (l2π + 2lσ1[i]) +
        c*(exp(2lσ1[i]) + abs2(μ1[i] - μ2[i]))/(exp(2lσ2[i]) + 1e-5) - 1.0
    end
    0.5l
end
# Zygote.refresh()
# Zygote.gradient(e->kl(e,dy), randn(1,10))

##
const dy = Normal(0, 0.05)
const nz = 3
const ny = 2
const nu = 0
N  = 30
np = 100
const f  = Chain(Dense(nz+nu,N,tanh), Dense(N,nz))
const g  = Chain(Dense(nz,N,tanh), Dense(N,ny))
const k  = Chain(Dense(nz+ny,N,tanh), Dense(N,nz))
const z0 = randn(nz,np)
pars = params((f,g,k,z0))

trajs_full = [generate_data(10) for i = 1:50]
trajs_meas = map(trajs_full) do (_,t)
    [copy(c) for c in eachcol(t)]
end
trajs_state = map(trajs_full) do (t,_)
    [copy(c) for c in eachcol(t)]
end

function train(loss, ps, dataset, opt; cb=i->nothing)
    @progress for (i,d) in enumerate(dataset)
        # Flux.reset!(model)
        (l1,l2), back = Zygote.forward(()->loss(d), ps)
        grads = back((1., 1.))
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
    i % 5 == 0 || return
    fig = plot([loss1 loss2], layout=3, sp=1)

    yh1,yh2 = sim(trajs_meas[1])
    ##
    plot!(reduce(hcat,trajs_meas[1])', layout=2, sp=2:3)
    scatter!(reduce(hcat, getindex.(yh2, 1, :))', m=(2,0.05,:green), sp=2, markerstrokecolor=:auto)
    scatter!(reduce(hcat, getindex.(yh2, 2, :))', m=(2,0.05,:green), sp=3, markerstrokecolor=:auto)
    display(fig)

end

##

function sim(y)
    T = length(y)
    z = z0
    yh1 = []
    yh2 = []
    zh = []
    for y in y
        z   = f(z)
        push!(zh, mean(z, dims=2)[:])
        z .+= randn(nz,np)#f([z; repeat(u[i], 1, np)])
        ŷ   = g(z)
        push!(yh1, ŷ)
        e   = y .- ŷ
        z   = k(⊗([z;e]))
        ŷ   = g(z)
        push!(yh2, ŷ)
    end
    yh1, yh2, zh
end

function loss(y)
    T = length(y)
    z = z0
    l1 = l2 = 0
    for y in y
        z   = f(z) + randn(nz,np)#f([z; repeat(u[i], 1, np)])
        ŷ   = g(z)
        e   = y .- ŷ
        # l1 += sum(norm, e)/np
        l1 += kl(e, dy)
        z   = k(([z;e]))
        ŷ   = g(z)
        e   = y .- ŷ
        # l2 += sum(norm, e)/np
        l2 += kl(e, dy)
    end
    l1/T, l2/T
end
# TODO: add penalty on KL div between ŷ and postulated measurement noise density

# loss(first(datas)...)
opt = ADAM(0.005)
Zygote.refresh()
# (l1,l2), back = Zygote.forward(()->loss(trajs_meas[1]), pars)
# grads = Zygote.gradient(()->loss(trajs[1]), pars)
train(loss, pars, IterTools.ncycle(trajs_meas, 1), opt, cb=cb)
yh1,yh2,zh = sim(trajs_meas[1])

##
plot(reduce(hcat,trajs_meas[1])', layout=2)
scatter!(reduce(hcat, getindex.(yh1, 1, :))', m=(2,0.05,:black), sp=1, markerstrokecolor=:auto)
scatter!(reduce(hcat, getindex.(yh1, 2, :))', m=(2,0.05,:black), sp=2, markerstrokecolor=:auto)
##
plot(reduce(hcat,trajs_meas[1])', layout=2)
scatter!(reduce(hcat, getindex.(yh2, 1, :))', m=(2,0.05,:green), sp=1, markerstrokecolor=:auto)
scatter!(reduce(hcat, getindex.(yh2, 2, :))', m=(2,0.05,:green), sp=2, markerstrokecolor=:auto)

## Plot tubes

Z = map(1:length(trajs_state)) do i
    yh1,yh2,zh = sim(trajs_meas[i])
    zh = reduce(hcat, zh)'
    s = svd(zh)
    zh = s.U.*s.S'
    zmat = reduce(hcat,trajs_state[i])'
    zh,zmat
end

zh = reduce(vcat, getindex.(Z,1))
zmat = reduce(vcat, getindex.(Z,2))

fig = plot(layout=2)
scatter3d!(eachcol(zh)..., m=(2,), zcolor=mod2pi.(zmat[:,1]), sp=1, markerstrokealpha=0, layout=2)
scatter3d!(eachcol(zh)..., m=(2,), zcolor=zmat[:,2], sp=2, markerstrokealpha=0)
