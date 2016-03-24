using ShapleyValues
using Base.Test

# generate data and ordinary least squares model
N = 1000000
P = 10
X = randn(N,P)
X .-= mean(X,1)
offsets = 0*randn(1,P)
X .+= offsets
betas = randn(P,1)
betas[5:end] = 0
model = x -> x'*betas
x = rand(1,P)
y = model(X')
trueVals = vec(betas.*x' - betas.*offsets');
baseValue = (betas.*offsets')[1]

# ensure that the total is with 10%
v = shapley_values(x, model, X', nsamples=10000)[1]
@test abs(1 - sum(v)/sum(trueVals)) < 0.1

# ensure that the adjusted total is perfect
trueSum = model(x') - baseValue
β = inv(v*v' + eye(length(v))*1e-6)*v*(sum(v) - trueSum)
v .-= β.*v
@test abs(1 - sum(v)/sum(trueVals)) < 0.0001
