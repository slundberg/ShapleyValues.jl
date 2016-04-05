function update_estimates!{T <: Real}(deltas, x::Array{T,1}, f::Function, typicalx::Array{T,1}, featureGroups, sampleCounts::Array{Int64,1})
    M = length(featureGroups)
    P = length(x)
    @assert length(sampleCounts) == M "sampleCounts should be an array of counts for each feature group!"

    # build the synthentic samples
    inds = collect(1:M)
    synthSamples = zeros(T, P, sum(sampleCounts)*2)
    pos = 1
    for i in 1:M
        for j in 1:sampleCounts[i]

            # find where in the permutation we are
            ind = findfirst(inds, i)

            # save two synthetic samples with and without the current group replaced
            synthSamples[:,pos] = x
            synthSamples[:,pos+1] = x
            for k in ind:M
                for l in featureGroups[inds[k]]
                    if k != ind
                        synthSamples[l,pos] = typicalx[l]
                    end
                    synthSamples[l,pos+1] = typicalx[l]
                end
            end

            pos += 2
        end
    end

    # run the provided function
    y = f(synthSamples)

    # sum the totals and keep an estimate of the variance differences
    pos = 1
    for i in 1:M
        for j in 1:sampleCounts[i]
            observe!(deltas[i], y[pos] - y[pos+1], 1)
            pos += 2
        end
    end
end

function shapleyvalues{T <: Real}(x::Array{T,1}, f::Function, typicalx::Array{T,1}; featureGroups=nothing, nsamples=1000, tol=1e-6)
    P = length(x)
    varTol = tol^2

    # give default values to omitted arguments
    featureGroups != nothing || (featureGroups = Array{Int64,1}[Int64[i] for i in 1:length(x)])
    featureGroups = convert(Array{Array{Int64,1},1}, featureGroups)

    # see which feature groups are different than the typical values
    matches = x .== typicalx
    varyingInds = find(Bool[!all(matches[g]) for g in featureGroups])
    varyingFeatureGroups = featureGroups[varyingInds]
    M = length(varyingFeatureGroups)

    # loop through the estimation process focusing samples on groups with high variance
    nextSamples = allocate_samples(ones(M), min(20M, nsamples))
    deltas = [MeanVarianceAccumulator() for i in 1:M]
    totalSamples = 0
    while true

        # update our estimates for a block of samples
        update_estimates!(deltas, x, f, typicalx, varyingFeatureGroups, nextSamples)

        # keep track of our samples and optimize their allocation to minimize variance (Neyman allocation)
        totalSamples += sum(nextSamples)
        if totalSamples < nsamples
            vs = [var(d) for d in deltas]
            minimum(vs) < varTol && break # stop early if we reach our convergence tolerance
            nextSamples = allocate_samples(vs, min(round(Int, nsamples/3), nsamples-totalSamples))
        else break end
    end

    # compute the Shapley values along with estimated variances of the estimates
    φ = zeros(length(featureGroups))
    φ[varyingInds] = [mean(d) for d in deltas]
    φVar = zeros(length(featureGroups))
    φVar[varyingInds] = [var(a)/count(a) for a in deltas]

    # We ensure that the total of all features equals f(x) - f(typicalx)
    trueSum = f(x)[1] - f(typicalx)[1]
    tmp = inv(φ*φ' + I*(trueSum*1e-8))
    β = inv(φ*φ' + I*(trueSum*1e-8))*φ*(sum(φ) - trueSum)
    φ .-= β.*φ

    # return the Shapley values along with estimated variances of the estimates
    φ,φVar
end
