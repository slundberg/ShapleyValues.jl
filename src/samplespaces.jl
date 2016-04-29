
export samplespace

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


type SparseSampleSpace{Tv,Ti}
    colptr::Array{Ti,1}
    colptrLen::Ti
    rowval::Array{Ti,1}
    rowvalLen::Ti
    nzval::Array{Tv,1}
    nzvalLen::Ti
    nrows::Ti
end
function samplespace{Tv,Ti}(X::SparseMatrixCSC{Tv,Ti}, nsamples::Int, maxDensity=1.0)
    s = SparseSampleSpace(
        Array(Ti, nsamples*size(X)[2]+1), one(Ti),
        Array(Ti, ceil(Int, maxDensity*size(X)[1]*nsamples*size(X)[2]+1)), zero(Ti),
        Array(Tv, ceil(Int, maxDensity*size(X)[1]*nsamples*size(X)[2]+1)), zero(Ti), convert(Ti, size(X)[1])
    )
    s.colptr[1] = 1
    s
end
function reset!(s::SparseSampleSpace)
    s.colptrLen = 1
    s.colptr[1] = 1
    s.rowvalLen = 0
    s.nzvalLen = 0
end
function addsample!(s::SparseSampleSpace, x)
    s.colptrLen += 1
    s.colptr[s.colptrLen] = s.colptr[s.colptrLen-1]
    for i in 1:length(x)
        if x[i] != 0
            s.colptr[s.colptrLen] += 1
            s.rowvalLen += 1
            s.rowval[s.rowvalLen] = i
            s.nzvalLen += 1
            s.nzval[s.nzvalLen] = x[i]
        end
    end
end
function data(s::SparseSampleSpace)
    SparseMatrixCSC(
        s.nrows, s.colptrLen-1,
        s.colptr[1:s.colptrLen],
        s.rowval[1:s.rowvalLen],
        s.nzval[1:s.nzvalLen]
    )
end
