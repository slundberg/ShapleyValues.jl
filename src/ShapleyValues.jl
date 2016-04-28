module ShapleyValues

export shapleyvalues

include("utils.jl")
include("coalitions.jl")
include("samplespaces.jl")

"The core method that updates the Shapley value estimates."
function update_estimates!(deltas, csets, x, f, X, sampleWeights,
                           g, featureGroups::Array{Array{Int64,1}},
                           sampleCounts::Array{Int64,1}, sampleSpace)
    P,N = size(X)
    M = length(featureGroups)
    @assert length(sampleCounts) == M "sampleCounts should be an array of counts for each feature group!"

    # build the synthentic samples
    inds = collect(1:M)
    s1 = Array(eltype(X), P)
    s2 = Array(eltype(X), P)
    weights = Array(eltype(X), sum(sampleCounts))
    reset!(sampleSpace)
    pos = 1
    wpos = 1
    for i in 1:M
        for j in 1:sampleCounts[i]
            cset,w = next(csets[i])
            weights[wpos] = w
            wpos += 1

            for k in 1:N

                # save two synthetic samples with and without the current group replaced
                copy!(s2, X[:,k])
                for l in cset
                    for m in featureGroups[l]
                        s2[m] = x[m]
                    end
                end
                copy!(s1, s2)
                for m in featureGroups[i]
                    s1[m] = x[m]
                end
                addsample!(sampleSpace, s1)
                addsample!(sampleSpace, s2)
                pos += 2
            end
        end
    end

    # run the provided function
    y::Array{eltype(X),1} = vec(f(data(sampleSpace)))

    # sum the totals and keep an estimate of the variance differences
    pos = 1
    wpos = 1
    withiExp = zero(eltype(X))
    for i in 1:M
        for j in 1:sampleCounts[i]
            withiExp = zero(eltype(X))
            withoutiExp = zero(eltype(X))
            sumw = zero(eltype(X))
            for k in 1:N
                w = sampleWeights[k]
                withiExp += w*y[pos]
                withoutiExp += w*y[pos+1]
                sumw += w
                pos += 2
            end
            withiExp /= sumw
            withoutiExp /= sumw
            observe!(deltas[i], g(withiExp) - g(withoutiExp), weights[wpos])
            wpos += 1
        end
    end
end

"Designed to determine the Shapley values (importance) of each feature for f(x)."
function shapleyvalues(x, f::Function, X, g::Function=identity; featureGroups=nothing, synthSampleSpace=nothing,
                       weights=nothing, nsamples=nothing)
    P,N = size(X)

    # give default values to omitted arguments
    weights != nothing || (weights = ones(N)/N)
    featureGroups != nothing || (featureGroups = Array{Int64,1}[Int64[i] for i in 1:length(x)])
    featureGroups = convert(Array{Array{Int64,1},1}, featureGroups)
    @assert length(weights) == N "Provided 'weights' must match the number of representative data points (size(X)[2])!"

    # find the feature groups we will test. If a feature does not change from its
    # current value then we know it doesn't impact the model
    varyingInds = varying_groups(x, X, featureGroups)
    varyingFeatureGroups = featureGroups[varyingInds]
    M = length(varyingFeatureGroups)

    # more default values
    nsamples != nothing || (nsamples = 2M+1000)
    synthSampleSpace != nothing || (synthSampleSpace = samplespace(X, nsamples))
    @assert nsamples >= 2M "'nsamples' must be at least 2 times the number of varying feature groups!"

    # loop through the estimation process focusing samples on groups with high variance
    deltas = [MeanVarianceAccumulator() for i in 1:M]
    csets = [statefuliterator(coalitions(setdiff(collect(1:M), [i]))) for i in 1:M]
    numSets = length(csets[1])
    nsamples = min(nsamples, numSets*M)
    totalSamples = zeros(Int64, M)
    nextSamples = allocate_samples(ones(M), min(2M, nsamples), numSets*ones(M))
    while true

        # update our estimates for a block of samples
        update_estimates!(deltas, csets, x, f, X, weights, g, varyingFeatureGroups, nextSamples, synthSampleSpace)

        # keep track of our samples and optimize their allocation to minimize variance (Neyman allocation)
        totalSamples .+= nextSamples
        samplesLeft = nsamples-sum(totalSamples)
        if samplesLeft > 0
            # (N-n)/(N-1) is the finite sample variance correction without weighting
            # so we approximate weighted as (1-sumw)/1 since weights sum to 1 over the whole population
            vs = [(1-a.sumw) * var(a)/length(a) for a in deltas]
            nextSamples = allocate_samples(vs, min(round(Int, nsamples/3), samplesLeft), numSets .- totalSamples)
        else break end
    end

    # compute the Shapley values along with estimated variances of the estimates
    φ = zeros(length(featureGroups))
    φ[varyingInds] = [mean(a) for a in deltas]
    φVar = zeros(length(featureGroups))
    φVar[varyingInds] = [(1-a.sumw) * var(a)/length(a) for a in deltas]

    # find f(x) and E_x[f(x)]
    reset!(synthSampleSpace)
    addsample!(synthSampleSpace, x)
    fx = f(data(synthSampleSpace))[1]
    reset!(synthSampleSpace)
    for i in 1:N addsample!(synthSampleSpace, X[:,i]) end
    fnull = sum(f(data(synthSampleSpace)).*weights)

    # We ensure that the total of all features equals g(f(x)) - g(E_x[f(x)])
    trueSum = g(fx) - g(fnull)
    tmp = inv(φ*φ' + I*(trueSum*1e-8))
    β = inv(φ*φ' + I*(trueSum*1e-8))*φ*(sum(φ) - trueSum)
    #println(maximum(β))
    φ .-= β.*φ
    @assert all(β .<= 1) "Rescaling failed! (indicates poor Shapley value estimates)"

    # return the Shapley values along with variances of the estimates
    φ,φVar
end

end # module
