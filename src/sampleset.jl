
using Clustering

export kmeans_sample_set,kmeans_sample_sets,SampleSet

"Used to represent a summary of the dataset, often good at approximating the impact of one group of features."
type SampleSet{T <: Real}
    samples::Array{T,2}
    weights::Array{Float64,1}
    inds::Array{Int64,1}
    K::Int64
end
SampleSet{T}(samples::Array{T,2}, weights::Array{Float64,1}, inds::Array{Int64,1}) = SampleSet(samples, weights, inds, length(weights))

function nearest_values(centers, X)
    P,N = size(X)
    K = size(centers)[2]
    nearest = copy(X[:,1:K])
    diff = abs(nearest .- centers)
    for i in 1:N, k in 1:K, j in 1:P
        d = abs(X[j,i] - centers[j,k])
        if d < diff[j,k]
            diff[j,k] = d
            nearest[j,k] = X[j,i]
        end
    end
    nearest
end

function kmeans_sample_set(X, k, groupInds)

    # decide how much to inflate the focus dimension
    scaling = ones(size(X)[1])
    v = var(X,2)
    scaling[groupInds] = 10./(v[groupInds]/sum(v)) # 10X all other variances

    out = kmeans(X .* scaling, k)
    SampleSet(nearest_values(out.centers ./ scaling, X), out.counts ./ sum(out.counts), groupInds)
end
function kmeans_sample_sets(X, k, featureGroups)
    SampleSet[kmeans_sample_set(X, k, g) for g in featureGroups]
end

function varying_sample_sets(x, sampleSets, weightThreshold)
    varyingWeights = zeros(length(sampleSets))
    for (i,s) in enumerate(sampleSets)
        varyingWeights[i] = sum(s.weights[vec(sum(x[s.inds] .== s.samples[s.inds,:],1) .!= length(s.inds))])
    end
    find(varyingWeights .> weightThreshold)
end

"Designed to determine the Shapley values (importance) of each feature for f(x)."
function shapleyvalues(x, f::Function, sampleSets::Array{SampleSet,1}, g::Function=identity; nsamples=1000, fnull=nothing)
    P = length(x)

    # find the feature groups we will test. If a feature rarely changes from its
    # current value then we know it doesn't have a large impact on the model
    varyingInds = varying_sample_sets(x, sampleSets, 0.01)
    varyingSampleSets = sampleSets[varyingInds]
    M = length(varyingSampleSets)

    # loop through the estimation process focusing samples on groups with high variance
    nextSamples = allocate_samples(ones(M), min(20M, nsamples))
    deltas = [MeanVarianceAccumulator() for i in 1:M]
    totalSamples = 0
    while true

        # update our estimates for a block of samples
        update_estimates!(deltas, x, f, varyingSampleSets, g, nextSamples)

        # keep track of our samples and optimize their allocation to minimize variance (Neyman allocation)
        totalSamples += sum(nextSamples)
        if totalSamples < nsamples
            vs = [var(a) for a in deltas]
            nextSamples = allocate_samples(vs, min(round(Int, nsamples/3), nsamples-totalSamples))
        else break end
    end

    # compute the Shapley values along with estimated variances of the estimates
    φ = zeros(length(sampleSets))
    φ[varyingInds] = [mean(a) for a in deltas]
    φVar = zeros(length(sampleSets))
    φVar[varyingInds] = [var(a)/count(a) for a in deltas]

    # If a base value was provided then we ensure that the total of all features equals f(x)
    if fnull != nothing
        trueSum = g(f(x)[1]) - g(fnull)
        tmp = inv(φ*φ' + I*(trueSum*1e-8))
        β = inv(φ*φ' + I*(trueSum*1e-8))*φ*(sum(φ) - trueSum)
        φ .-= β.*φ
    end

    # return the Shapley values along with estimated variances of the estimates
    φ,φVar
end

"The core method that updates the Shapley value estimates."
function update_estimates!(deltas, x, f::Function, sampleSets::Array{SampleSet,1}, g::Function, sampleCounts)
    M = length(sampleSets)
    @assert length(sampleCounts) == M "sampleCounts should be an array of counts for each SampleSet!"

    # build the synthentic samples
    inds = collect(1:M)
    synthLength = sum([2*sampleCounts[i]*sampleSets[i].K for i in 1:M])
    synthSamples = zeros(Float32, length(x), synthLength)
    pos = 1
    for i in 1:M
        for j in 1:sampleCounts[i]
            shuffle!(inds)

            # find where in the permutation we are
            ind = findfirst(inds, i)

            for k in 1:sampleSets[i].K

                # save two synthetic samples with and without the current group replaced
                synthSamples[:,pos] = x
                synthSamples[:,pos+1] = x
                r = sampleSets[i].samples[:,k]
                synthSamples[sampleSets[inds[ind]].inds,pos+1] = r[sampleSets[inds[ind]].inds]
                for k in ind+1:M
                    for l in sampleSets[inds[k]].inds
                        synthSamples[l,pos] = r[l]
                        synthSamples[l,pos+1] = r[l]
                    end
                end

                pos += 2

            end
        end
    end

    # run the provided function
    y = f(synthSamples)

    # sum the totals and keep an estimate of the variance differences
    pos = 1
    for i in 1:M
        w = sampleSets[i].weights
        for j in 1:sampleCounts[i]
            withiExp = 0.0
            withoutiExp = 0.0
            sumw = 0.0
            for k in 1:sampleSets[i].K
                withiExp += y[pos]*w[k]
                withoutiExp += y[pos+1]*w[k]
                sumw += w[k]

                pos += 2
            end
            withiExp /= sumw
            withoutiExp /= sumw

            observe!(deltas[i], g(withiExp) - g(withoutiExp), 1)
        end
    end
end
