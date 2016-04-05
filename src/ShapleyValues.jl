module ShapleyValues

import Base.var
import Base.mean

export shapleyvalues

include("typical.jl")
include("expected.jl")
include("sampleset.jl")

"Distributes the given number of samples proportionally."
function allocate_samples(proportions, nsamples)
    counts = round(Int, nsamples*proportions/sum(proportions))
    total = sum(counts)
    for ind in randperm(length(counts))
        total != nsamples || break

        if total < nsamples
            counts[ind] += 1
            total += 1
        elseif counts[ind] > 0
            counts[ind] -= 1
            total -= 1
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
count(mvar::MeanVarianceAccumulator) = mvar.n
Base.mean(mvar::MeanVarianceAccumulator) = mvar.wmean
var(mvar::MeanVarianceAccumulator) = (mvar.t*mvar.n)/(mvar.sumw*(mvar.n-1))
std(mvar::MeanVarianceAccumulator) = sqrt(var(mvar))

end # module
