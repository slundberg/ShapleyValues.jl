
# statefuliterator
it = ShapleyValues.statefuliterator(1:3)
@test length(it) == 3
@test done(it) == false
@test next(it) == 1
@test next(it) == 2
@test done(it) == false
tmp = next(it)
@test tmp == 3
@test done(it) == true
@test typeof(tmp) == Int64

# allocate_samples
@assert ShapleyValues.allocate_samples([1,1,1,1], 4, ones(Int64, 4)) == ones(4)
@assert ShapleyValues.allocate_samples([1,1,1,1], 8, ones(Int64, 4)) == ones(4)
@assert ShapleyValues.allocate_samples([1,2,1,1], 10, 4*ones(Int64, 4)) == [2,4,2,2]
@assert ShapleyValues.allocate_samples([1,2,1,1], 11, [4,4,4,1]) == [3,4,3,1]
@assert sum(ShapleyValues.allocate_samples([1,2,1,1], 10, 3*ones(Int64, 4))) == 10
@assert maximum(ShapleyValues.allocate_samples([1,2,1,1], 10, 3*ones(Int64, 4))) == 3
@assert minimum(ShapleyValues.allocate_samples([1,2,1,1], 10, 3*ones(Int64, 4))) == 2
@assert sum(ShapleyValues.allocate_samples([1,2,1,1], 21, 10*ones(Int64, 4))) == 21

# MeanVarianceAccumulator
v = [1,2,3.0]
mv = ShapleyValues.MeanVarianceAccumulator()
@test isnan(std(mv))
ShapleyValues.observe!(mv, v[1], 1)
@test mean(mv) == v[1]
@test isnan(std(mv))
ShapleyValues.observe!(mv, v[2], 1)
@test mean(mv) == mean(v[1:2])
@test std(mv) ≈ std(v[1:2])
ShapleyValues.observe!(mv, v[3], 1)
@test mean(mv) == mean(v)
@test std(mv) ≈ std(v)
@test length(mv) == 3

w = [0.1,0.4,0.5]
v = [1,2,3.0]
mv = ShapleyValues.MeanVarianceAccumulator()
@test isnan(std(mv))
ShapleyValues.observe!(mv, v[1], w[1])
@test mean(mv) == v[1]
@test isnan(std(mv))
ShapleyValues.observe!(mv, v[2], w[2])
@test mean(mv) == dot(v[1:2], w[1:2])/sum(w[1:2])
@test sum(((v[1:2] - mean(mv)).^2).*w[1:2])/sum(w[1:2]) * (2/1) ≈ var(mv)
ShapleyValues.observe!(mv, v[3], w[3])
@test mean(mv) ==  dot(v, w)/sum(w)
@test sum(((v - mean(mv)).^2).*w)/sum(w) * (3/2) ≈ var(mv)

# varying_groups
@test ShapleyValues.varying_groups(zeros(3), [0 0 1; 0 1 1; 0 0 1]', Array{Int64,1}[[1,2],[3]]) == [1,2]
@test ShapleyValues.varying_groups(zeros(3), [0 0 1; 0 1 1; 0 0 1]', Array{Int64,1}[[1],[2,3]]) == [2]
@test ShapleyValues.varying_groups(zeros(3), zeros(3,10), Array{Int64,1}[[1],[2,3]]) == Int64[]
