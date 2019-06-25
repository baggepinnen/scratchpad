using Flux, Zygote, Test

gradlength(grads) = sum(length(g[1]) for g in grads.grads)

function flatten!(gradvec, grads::Zygote.Grads)
    @assert length(gradvec) == gradlength(grads)
    s = 1
    for g in grads.grads
        l = length(g[2])
        gradvec[s:s+l-1] .= vec(g[2])
        s += l
    end
    gradvec
end

flatten(grads::Zygote.Grads) = flatten!(zeros(gradlength(grads)), grads::Zygote.Grads)

function unflatten!(gradvec, grads::Zygote.Grads)
    s = 1
    for g in grads.grads
        l = length(g[2])
        g[2] .= reshape(gradvec[s:s+l-1], size(g[2]))
        s += l
    end
    grads
end

@test unflatten!(flatten(grads), grads) == grads

##

x = LinRange(-3,3,100)'
y = sin.(x)


function testtrain(loss, opt, ps)
    grads = Zygote.gradient(loss, ps)
    gradvec = flatten(grads)
    losses = Float64[]
    for i = 1:75
        push!(losses, loss())
        grads = Zygote.gradient(loss, ps)
        flatten!(gradvec, grads)
        gradvec .= (gradvec*gradvec' + 1e-5I)\gradvec
        unflatten!(gradvec, grads)
        Flux.Optimise.update!(opt, ps, grads)
    end
    losses
end

Zygote.refresh()
function testtrainlm(loss, opt, ps)
    grads = Zygote.gradient(()->loss(x[i],y[i]), ps)
    gradvec = flatten(grads)
    losses = Float64[]
    for i = 1:20
        push!(losses, sum(loss(x[i],y[i]) for i in eachindex(x)))
        losses[end] < 1e-2 && return losses
        gradvecs = map(eachindex(x)) do i
            grads = Zygote.gradient(()->loss(x[i],y[i]), ps)
            flatten(grads)
        end
        A = reduce(hcat, gradvecs)'
        if losses[end] > 0*1e1
            gs = sum(A', dims=2)
        else
            gs = 2sum((A'A + 1e-3I)\A', dims=2)
        end
        unflatten!(gs, grads)
        Flux.Optimise.update!(opt, ps, grads)
    end
    losses
end

losses = map(1:10) do i
    model = Chain(Dense(1,20,tanh), Dense(20,20,tanh), Dense(20,1))
    loss(x,y) = sum(abs2, model([x])[]-y)
    opt = Nesterov(0.0005, 0.9)
    # opt = ADAM(0.01)
    ps = params(model)
    losses = testtrainlm(loss, opt,ps)
end
plot(losses, yscale=:log10, xscale=:identity)
