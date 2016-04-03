import Base.push!
using Distributions

abstract OnlineStat{O}

type ScalarIdentity{I} <: OnlineStat{I}
    listeners::Array{OnlineStat,1}
    value::I

    ScalarIdentity() = new(OnlineStat[],zero(I))
end
function push!{I}(o::ScalarIdentity{I}, v::I, weight::Float64=1.0)
    o.value = v
    for i in 1:length(o.listeners)
        push!(o.listeners[i], v, weight)
    end
end
rawvalue{I}(o::ScalarIdentity{I}) = o.value
value{I}(o::ScalarIdentity{I}) = o.value


type VectorIdentity{I} <: OnlineStat{Array{I,1}}
    listeners::Array{OnlineStat,1}
    value::Array{I,1}

    VectorIdentity(n::Int) = new(OnlineStat[],zeros(I, n))
end
function push!{I}(o::VectorIdentity{I}, v::Array{I,1}, weight::Float64=1.0)
    o.value = v
    for i in 1:length(o.listeners)
        push!(o.listeners[i], v, weight)
    end
end
rawvalue{I}(o::VectorIdentity{I}) = o.value
value{I}(o::VectorIdentity{I}) = o.value


type OnlineStatArray{O <: OnlineStat} <: OnlineStat{Array{O,1}}
    onlineStats::Array{OnlineStat,1}

    OnlineStatArray(arr::Array{O,1}) = new(arr)
end
rawvalue{O}(o::OnlineStatArray{O}) = o.onlineStats
value{O}(o::OnlineStatArray{O}) = typeof(value(o.onlineStats[1]))[value(x) for x in o.onlineStats]


type Mean{I <: Real} <: OnlineStat{Float64}
    listeners::Array{OnlineStat,1}
    sumValues::Float64
    sumWeights::Float64
    value::Float64

    Mean() = new(OnlineStat[],0.0,0.0,0.0)
end
function mean{I <: Real}(o::OnlineStat{I})
    m = Mean{I}()
    push!(o.listeners, m)
    m
end
function mean{I <: OnlineStat}(o::OnlineStat{Array{I,1}})
    vals = [mean(v) for v in rawvalue(o)]
    OnlineStatArray{typeof(vals[1])}(vals)
end
function push!{I}(o::Mean{I}, v::I, weight::Float64=1.0)
    o.sumValues += v
    o.sumWeights += weight
    o.value = o.sumValues/o.sumWeights
    for i in 1:length(o.listeners)
        push!(o.listeners[i], o.value, weight)
    end
end
rawvalue{I}(o::Mean{I}) = o.value
value{I}(o::Mean{I}) = o.value


type VectorMean{I <: Real} <: OnlineStat{Array{Float64,1}}
    listeners::Array{OnlineStat,1}
    sumValues::Array{Float64,1}
    sumWeights::Float64
    value::Array{Float64,1}

    VectorMean(n::Int) = new(OnlineStat[],zeros(n),0.0,zeros(n))
end
function mean{I <: Real}(o::OnlineStat{Array{I,1}})
    m = VectorMean{I}(length(rawvalue(o)))
    push!(o.listeners, m)
    m
end
function push!{I}(o::VectorMean{I}, v::Array{I,1}, weight::Float64=1.0)
    o.sumWeights += weight
    for i in 1:length(v)
        o.sumValues[i] += v[i]*weight
        o.value[i] = o.sumValues[i] / o.sumWeights
    end

    for i in 1:length(o.listeners)
        push!(o.listeners[i], v, weight)
    end
end
rawvalue{I}(o::VectorMean{I}) = o.value
value{I}(o::VectorMean{I}) = o.value


type ScalarPoissonBootstrap{I} <: OnlineStat{Array{ScalarIdentity{I},1}}
    replicates::Array{ScalarIdentity{I},1}

    ScalarPoissonBootstrap(replicates::Array{ScalarIdentity{I},1}) = new(replicates)
end
function bootstrap{I <: Real}(o::OnlineStat{I}, r::Int, method=:poisson)
    b = ScalarPoissonBootstrap{I}([ScalarIdentity{I}() for i in 1:r])
    push!(o.listeners, b)
    b
end
const unitPoissonDist = Poisson(1)
function push!{I}(b::ScalarPoissonBootstrap{I}, v::I, weight::Float64=1.0)
    for replicate in b.replicates
        for i in 1:rand(unitPoissonDist)
            push!(replicate, v, weight)
        end
    end
end
rawvalue{I}(o::ScalarPoissonBootstrap{I}) = o.replicates
value{I}(o::ScalarPoissonBootstrap{I}) = [value(r) for r in o.replicates]


type VectorPoissonBootstrap{I} <: OnlineStat{Array{VectorIdentity{I},1}}
    replicates::Array{VectorIdentity{I},1}

    VectorPoissonBootstrap(replicates::Array{VectorIdentity{I},1}) = new(replicates)
end
function bootstrap{I <: Real}(o::OnlineStat{Array{I,1}}, r::Int, method=:poisson)
    len = length(rawvalue(o))
    b = VectorPoissonBootstrap{I}([VectorIdentity{I}(len) for i in 1:r])
    push!(o.listeners, b)
    b
end
const unitPoissonDist = Poisson(1)
function push!{I}(b::VectorPoissonBootstrap{I}, v::Array{I,1}, weight::Float64=1.0)
    for replicate in b.replicates
        for i in 1:rand(unitPoissonDist)
            push!(replicate, v, weight)
        end
    end
end
rawvalue{I}(o::VectorPoissonBootstrap{I}) = o.replicates
value{I}(o::VectorPoissonBootstrap{I}) = Array{I,1}[value(r) for r in o.replicates]
