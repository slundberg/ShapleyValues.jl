using Iterators

export coalitions

type Coalitions{T}
    members::Array{T,1}
    k::Int64
    lastk::Int64
    kIter::Iterators.Binomial{T}
    invSet::Array{T,1}
end

Base.eltype(it::Coalitions) = Array{eltype(it.members),1}
Base.length(it::Coalitions) = 2^length(it.members)

"Enumerate all the coalitions with a weight for computing the Shapley value."
coalitions{T}(members::Array{T,1}) = Coalitions(members, 0, 0, subsets(members, 0), T[])

function Base.start(it::Coalitions)
    start(it.kIter)
end

function coalition_weight(q, s)
    if q > s-q-1
        factorial(s-q-1)/factorial(s,q)
    else
        factorial(q)/factorial(s,s-q-1)
    end
end

function Base.next{T}(it::Coalitions{T}, state::(Tuple{Array{Int64,1}, Bool}))
    if length(it.invSet) > 0
        nextSet = it.invSet
        it.invSet = T[]

        # check if this was a final inv set to emit
        if length(state[1]) == 0
            state = (Int64[],true)
        end
    else
        nextSet,state = next(it.kIter, state)
        it.lastk = it.k

        if 2*it.k < length(it.members)

            # we also create an inverse set if we are not
            it.invSet = setdiff(it.members, nextSet)

            # if we are done with current set of sizes move to the next one
            if state[2] && 2*it.k < length(it.members)-1
                it.k += 1
                it.kIter = subsets(it.members, it.k)
                state = start(it)
            end

            # prevent us from ending if we still have a inv set to emit
            if state[2]
                state = (Int64[],false)
            end
        end
    end

    weight = coalition_weight(it.lastk, length(it.members)+1)

    (nextSet,weight),state
end

Base.done(it::Coalitions, state::Tuple{Array{Int64,1}, Bool}) = state[2]
