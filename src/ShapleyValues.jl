module ShapleyValues

export shapley_values

"Designed to determine the Shapley values (importance) of each feature for model(x)."
function shapley_values(x, f, Xt, featureGroups=nothing; nsamples=10000, maxStdDevFraction=0.02)
    P = length(x)

    # find the feature groups we will test
    featureGroups != nothing || (featureGroups = Array{Int64,1}[Int64[i] for i in 1:length(x)])
    featureGroups = convert(Array{Array{Int64,1},1}, featureGroups)
    varyingInds = varying_feature_groups(x, Xt, featureGroups)
    varyingFeatureGroups = featureGroups[varyingInds]
    M = length(varyingFeatureGroups)

    # loop through the estimation process focusing samples on groups with high variance
    nextSamples = round(Int, (ones(M)./M) * min(10M, nsamples))
    accumulators = [MeanVarianceAccumulator() for i in 1:M]
    totals = zeros(M)
    counts = zeros(Int64, M)
    sampleChunk = round(Int, nsamples/3)
    totalSamples = 0
    while totalSamples < nsamples

        # update our estimates for a block of samples
        update_estimates!(totals, accumulators, x, Xt, f, varyingFeatureGroups, nextSamples)

        # keep track of our samples and optimize their allocation to minimize variance (Neyman allocation)
        totalSamples += sum(nextSamples)
        counts .+= nextSamples
        vs = [var(a) for a in accumulators]
        nextSamples = round(Int, sampleChunk*vs/sum(vs))
        sum(abs(totals ./ counts))*maxStdDevFraction <= sqrt(sum(vs./counts)) || break
        #println("SDF")
    end

    # return the Shapley values along with estimated variances of the estimates
    est = zeros(length(featureGroups))
    est[varyingInds] = totals ./ counts
    estVar = zeros(length(featureGroups))
    estVar[varyingInds] = [var(a) for a in accumulators]./counts
    est,estVar
end

"Identifies which feature groups often vary from the observed value in data set."
function varying_feature_groups(x, Xt, featureGroups::Array{Array{Int64,1},1}; nsamples=100, threshold=5)
    N = size(Xt)[2]
    M = length(featureGroups)
    found = zeros(Int64, M)
    for i in 1:nsamples
        r = full(Xt[:,rand(1:N)])
        for j in 1:M
            for ind in featureGroups[j]
                if x[ind] != r[ind]
                    found[j] += 1
                end
            end
        end
    end
    find(found .> threshold)
end

"The core method that updates the Shapley value estimates."
function update_estimates!(totals, accumulators, x, Xt, model, featureGroups, sampleCounts)
    M = length(featureGroups)
    P = length(x)
    N = size(Xt)[2]
    @assert length(sampleCounts) == M "sampleCounts should be an array of counts for each feature group!"


    # build the synthentic samples
    inds = collect(1:M)
    r = zeros(P)
    unchangedCounts = zeros(Int64, M)
    synthSamples = zeros(Float32, P,round(Int, sum(sampleCounts)*2))
    pos = 1
    for i in 1:M
        for j in 1:sampleCounts[i]
            shuffle!(inds)
            r[:] = full(Xt[:,rand(1:N)])
            ind = findfirst(inds, i)

            ginds = featureGroups[inds[ind]]
            if all(x[ginds] .== r[ginds])
                unchangedCounts[i] += 1
                continue
            end

            synthSamples[:,pos] = x
            synthSamples[:,pos+1] = x
            synthSamples[featureGroups[inds[ind]],pos+1] = r[featureGroups[inds[ind]]]
            for k in ind+1:M
                ginds = featureGroups[inds[k]]
                synthSamples[ginds,pos] = r[ginds]
                synthSamples[ginds,pos+1] = r[ginds]
            end

            pos += 2
        end
    end

    # run the model
    y = model(synthSamples[:,1:2*(sum(sampleCounts)-sum(unchangedCounts))])

    # sum the differences
    pos = 1
    for i in 1:M
        for j in 1:(sampleCounts[i]-unchangedCounts[i])
            diff = y[pos] - y[pos+1]
            totals[i] += diff
            observe!(accumulators[i], diff, 1)
            pos += 2
        end

        # unchanged samples have a difference of zero
        for j in 1:unchangedCounts[i]
            observe!(accumulators[i], 0.0, 1)
        end
    end
end

# http://www.nowozin.net/sebastian/blog/streaming-mean-and-variance-computation.html
type MeanVarianceAccumulator
    sumw::Float64
    wmean::Float64
    t::Float64
    n::Int

    function MeanVarianceAccumulator()
        new(0.0, 0.0, 0.0, 0)
    end
end
function observe!(mvar::MeanVarianceAccumulator, value, weight)
    @assert weight >= 0.0
    q = value - mvar.wmean
    temp_sumw = mvar.sumw + weight
    r = q*weight / temp_sumw

    mvar.wmean += r
    mvar.t += q*r*mvar.sumw
    mvar.sumw = temp_sumw
    mvar.n += 1

    nothing
end
count(mvar::MeanVarianceAccumulator) = mvar.n
Base.mean(mvar::MeanVarianceAccumulator) = mvar.wmean
var(mvar::MeanVarianceAccumulator) = (mvar.t*mvar.n)/(mvar.sumw*(mvar.n-1))
std(mvar::MeanVarianceAccumulator) = sqrt(var(mvar))

end # module
