using Flux, Flux.Tracker
using Flux.Tracker: data

m,n,k = 20,10,3

W = rand(m,k)
H = softmax(5randn(k,n))
A = W*H

function rank_k_svd(A,k)
    s = svd(A)
    Wsvd, Hsvd = s.U[:,1:k], s.Vt[1:k,:]
    Asvd = Wsvd*Diagonal(s.S[1:k])*Hsvd
    Wsvd,Hsvd,Asvd
end

Wsvd,Hsvd,Asvd = rank_k_svd(A,k)
@assert norm(A-Asvd)/norm(A) < 1e-10


α = 0.5
Wh,Hh = param(randn(size(Wsvd))), param(randn(size((Hsvd))))
p = params(Wh,Hh)
# cost() = norm(A-Wh*Hh) + sum(abs,Hh)
dir(H,α) = (α-1)*sum(log, H)
function cost()
    H = softmax(Hh)
    c = norm(A-Wh*H)
    rC = -0.000det(H*H')
    r = -dir(H,α) # Negative likelihood
    c+r+rC
end
cost()
opt = ADAM()
for i = 1:2000
    gs = Tracker.gradient(cost, p)
    Flux.Optimise.update!(opt, p, gs)
    if i % 100 == 0
        @show cost()
        H = softmax(Hh)
        @show c = norm(A-Wh*H)/norm(A)
        heatmap(data(softmax(Hh)), show=true)
    end
end

norm(A-Asvd)/norm(A)
norm(A-Wh*softmax(Hh))/norm(A)


## Using Turing
using Turing

@model dmf(A, k) = begin
    m,n = size(A)
    α = 1/m
    h = Vector{Vector{Real}}(undef, n)
    w = Vector{Vector{Real}}(undef, m)
    s ~ InverseGamma(1,0.1)
    for j = 1:n
        h[j] ~ Dirichlet(k,α)
    end
    for i = 1:m
        w[i] ~ MvNormal(k,3)
    end

  for j = 1:n, i = 1:m
      A[i,j] ~ Normal(w[i]'h[j], sqrt(s))
  end
end

#  Run sampler, collect results
chn = sample(dmf(A, k), HMC(600, 0.1, 5))
describe(chn)

using StatsPlots
plot(chn)
meanplot(chn)
s = sample(chn, 1)
Hh = reshape(Array(s[:h].value), k,n)
Wh = reshape(Array(s[:w].value), k,m)'

Ah = Wh*Hh
norm(A-Ah)/norm(A)



# KSVD
Ws, Hs = ksvd(A, 20, max_iter = 500, max_iter_mp = 600, sparsity_allowance = 0.99)
norm(A-Ws*Hs)
heatmap(abs.(Hs))


using LowRankModels
losses = QuadLoss() # minimize squared distance to cluster centroids
ry = SimplexConstraint()
rx = ZeroReg() # no regularization on the cluster centroids
glrm = GLRM(A,losses,rx,ry,k)
pars = ProxGradParams(1.0; # initial stepsize
				        max_iter=500, # maximum number of outer iterations
                inner_iter_X=1, # how many prox grad steps to take on X before moving on to Y (and vice versa)
                inner_iter_Y=1, # how many prox grad steps to take on Y before moving on to X (and vice versa)
                inner_iter=1,
                abs_tol=1e-9, # stop if objective decrease upon one outer iteration is less than this * number of observations
                rel_tol=1e-8)
Wlr,Hlr,ch = fit!(glrm, pars)
Alr = Wlr'*Hlr
norm(A-Alr)

plot(A, c=:blue, layout=n, legend=false)
plot!(Alr, c=:red)
