module LCA
export lca

#Make all columns of the input matrix have a norm of 1
# D - input matrix (modified in place)
function unitary_columns!(D::AbstractMatrix)
    for i = 1:size(D,2)
        D[:,i] = D[:,i]/norm(D[:,i]);
    end
end

#Perform the positive/negative soft threshold function on each element
# a - result vector (modified in place)
# u - input vector
# T - threshold
function soft_threshold!(a::AbstractVector, u::AbstractVector, T::Real)
    for i = 1:length(u)
        if abs(u[i]) > T
            a[i,1] = u[i]-sign(u[i])*T
        else
            a[i,1] = 0
        end
    end
end

#Perform Locally Competitive Algorithm for sparse recovery
# s      - Input to be approximated
# D      - Dictionary
# T_soft - Soft threshold value (higher promotes more sparseness)
# iter   - maximum number of iterations
# τ      - 1/learning rate
function lca(s::AbstractVector, D::AbstractMatrix; T_soft::Real=0.1,
             iter::Int=10000, τ::Real=100.0)
    #Impose Unitary norm
    unitary_columns!(D)

    # Inhibition Matrix
    G = D'D
    for i=1:size(G,1)
        G[i,i] = 0
    end

    # Initialize LCA Parameters
    b = D's # initial projection / excitatory input

    u = zeros(size(b)) # initial state of nodes = 0
    a = zeros(size(b)) # initial sparse rep = 0

    for i=1:iter
        # soft thresholding function
        soft_threshold!(a, u, T_soft)

        # node dynamics
        Δu = (b  - u - G*a)/τ;
        u += Δu;

        if(norm(Δu) < 1e-5)
            break
        end
    end

    return a
end

end # module
