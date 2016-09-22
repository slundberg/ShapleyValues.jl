using ShapleyValues
using StatsBase
using Base.Test

include("utils.jl")
include("coalitions.jl")
include("samplespaces.jl")

# simple test of logistic and linear regression
srand(1)
K = 4
X = rand(K,100)
beta = randn(K)
f(x) = x'beta
p(x) = logistic(f(x))
x = 2*randn(K)

"A direct way to compute the expected value conditioned on the given inds."
function raw_exp(f, x, inds, X)
    Xtmp = copy(X)
    for i in inds
        Xtmp[i,:] = x[i]
    end
    mean(f(Xtmp))
end

fnull,φ,φVar = shapleyvalues(x, f, X) # linear regression
@test raw_exp(f, x, 1:K, X) - raw_exp(f, x, Int64[], X) ≈ sum(φ)
@test sum(abs(φVar)) < 1e-10
fnull,φ,φVar = shapleyvalues(x, f, X, nsamples=8)
fnull,φ,φVar = shapleyvalues(x, f, sparse(X), nsamples=8)

fnull,φ,φVar = shapleyvalues(x, p, X, logit) # logistic regression
@test logit(raw_exp(p, x, 1:K, X)) - logit(raw_exp(p, x, Int64[], X)) ≈ sum(φ)
@test sum(abs(φVar)) < 1e-10

# test with many features
srand(1)
K = 400
X = rand(K,10)
beta = randn(K)
f(x) = x'beta
p(x) = logistic(f(x))
x = 2*randn(K)
fnull,φ,φVar = shapleyvalues(x, f, X)
@test raw_exp(f, x, 1:K, X) - raw_exp(f, x, Int64[], X) ≈ sum(φ)

# test with many features
srand(1)
K = 400
X = rand(K,10)
beta = randn(K)
f(x) = x'beta
p(x) = logistic(f(x))
x = .02*randn(K)
fnull,φ,φVar = shapleyvalues(x, p, X, logit)
@test logit(raw_exp(p, x, 1:K, X)) - logit(raw_exp(p, x, Int64[], X)) ≈ sum(φ)
