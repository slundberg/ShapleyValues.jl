
# each sample space supports a different type of data matrix (dense, sparse, etc.)

type DenseSampleSpace{T}
    data::Array{T,2}
    N::Int64
end
function samplespace{T}(X::Array{T,2}, nsamples::Int)
    DenseSampleSpace(Array(T, size(X)[1], nsamples*size(X)[2]), 0)
end
addsample!(s::DenseSampleSpace, x) = (s.N += 1; s.data[:,s.N] = x)
reset!(s::DenseSampleSpace) = (s.N = 0)
data(s::DenseSampleSpace) = s.data[:,1:s.N]
