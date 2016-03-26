
# generate data and ordinary least squares model
srand(1)
N = 100000
P = 100
X = randn(N,P)
X .-= mean(X,1)
offsets = 0*randn(1,P)
X .+= offsets
X .*= rand(1,P)
betas = randn(P,1)
betas[50:end] = 0
model = x -> x'*betas
x = rand(1,P)
Xt = X'
y = model(Xt)
trueVals = vec(betas.*x' - betas.*offsets');
baseValue = (betas.*offsets')[1]

# time the varying_feature_groups function
featureGroups = Array{Int64,1}[Int64[i] for i in 1:length(x)]
xt = x'
ShapleyValues.varying_feature_groups(xt, Xt, featureGroups)
@time ShapleyValues.varying_feature_groups(xt, Xt, featureGroups)


# check the performance of update_estimates!
nsamples = 100000
varyingInds = ShapleyValues.varying_feature_groups(xt, Xt, featureGroups)
varyingFeatureGroups = featureGroups[varyingInds]
M = length(varyingFeatureGroups)
nextSamples = round(Int, (ones(M)./M) * min(100M, nsamples))
accumulators = [ShapleyValues.MeanVarianceAccumulator() for i in 1:M]
totals1 = zeros(M)
totals2 = zeros(M)
ShapleyValues.update_estimates!(totals1, totals2, accumulators, xt, model, Xt, varyingFeatureGroups, nextSamples)
@time ShapleyValues.update_estimates!(totals1, totals2, accumulators, xt, model, Xt, varyingFeatureGroups, nextSamples)

Profile.clear()
@profile ShapleyValues.update_estimates!(totals1, totals2, accumulators, xt, model, Xt, varyingFeatureGroups, nextSamples)
Profile.print()
