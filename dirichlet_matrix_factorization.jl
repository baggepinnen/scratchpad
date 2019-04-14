using Flux, Flux.Tracker
using Flux.Tracker: data

m,n,k = 100,20,5

W = rand(m,k)
H = rand(k,n)
H[H .< 0.3] .= 0 # H is now 30% sparse
H[:,(sum(H, dims=1) .== 0)[:]] .= 1/k
H = H ./ sum(H, dims=1)
@assert !any(isnan, H)
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

x = LinRange(0.1,0.9,20)
plot3d(,, (x,y)->dir([x,y],0.1))
