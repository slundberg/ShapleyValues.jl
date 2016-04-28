import Base.length
import Base.start
import Base.next
import Base.done

type StatefulIterator{T}
    iter::T
    state
end
"Captures the state of the iterator so the user does not need to."
statefuliterator(iter) = StatefulIterator(iter, start(iter))
function Base.next(si::StatefulIterator)
    item,state = next(si.iter, si.state)
    si.state = state
    item
end
Base.done(si::StatefulIterator) = done(si.iter, si.state)
Base.length(si::StatefulIterator) = length(si.iter)


"Distributes the given number of samples proportionally."
function allocate_samples(proportions, nsamples, maxCounts)
    counts = round(Int, nsamples*proportions/sum(proportions))
    counts[counts .> maxCounts] = maxCounts[counts .> maxCounts]
    total = sum(counts)
    changed = true
    while changed
        changed = false
        for ind in shuffle(find(counts .!= maxCounts))
            total != nsamples || break

            if total < nsamples && counts[ind] < maxCounts[ind]
                counts[ind] += 1
                total += 1
                changed = true
            elseif (total > nsamples && counts[ind] > 0) || (counts[ind] > maxCounts[ind])
                counts[ind] -= 1
                total -= 1
                changed = true
            end
        end
    end
    counts
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
Base.length(mvar::MeanVarianceAccumulator) = mvar.n
Base.mean(mvar::MeanVarianceAccumulator) = mvar.wmean
Base.var(mvar::MeanVarianceAccumulator) = (mvar.t*mvar.n)/(mvar.sumw*(mvar.n-1))
Base.std(mvar::MeanVarianceAccumulator) = sqrt(var(mvar))


"Identify which feature groups vary."
function varying_groups(x, X, featureGroups)
    varying = zeros(length(featureGroups))
    for (i,inds) in enumerate(featureGroups)
        varying[i] = sum(vec(sum(x[inds] .== X[inds,:],1) .!= length(inds)))
    end
    find(varying)
end
