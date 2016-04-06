
srand(1)
N = 100000
P = 10
X = randn(P,N)
X .-= mean(X,2)
offsets = 0*randn(P,1)
X .+= offsets
X[1,:] = rand([1,0], N)
betas = randn(P,1)
betas[5:end] = 0
model = x -> betas'x
x = rand(P,1)
x[1] = rand([1,0])
y = model(X)
trueVals = vec(betas.*x - betas.*mean(X,2));
baseValue = (betas.*offsets')[1]

featureGroups = Array{Int64,1}[Int64[1],[2,3],[4],[5],[6],[7],Int64[8,9,10]]
sampleSets = kmeans_sample_sets(X, 2, featureGroups)

vals,vars = shapleyvalues(x, model, sampleSets, nsamples=10000)

@test abs(vals[1] - trueVals[1]) < 1e-3
@test abs(vals[2] - sum(trueVals[2:3])) < 1e-3
@test abs(vals[3] - trueVals[4]) < 1e-3
@test abs(vals[4] - trueVals[5]) < 1e-3
