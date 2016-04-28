
@test length(coalitions([1,2,3])) == 2^3
@test length(coalitions([1,2,3,4])) == 2^4
@test sum([w for (c,w) in coalitions([1,2,3])]) ≈ 1
@test sum([w for (c,w) in coalitions(collect(1:9))]) ≈ 1
@test sum([w for (c,w) in coalitions(collect(1:10))]) ≈ 1
it = ShapleyValues.statefuliterator(coalitions([1,2]))
@test next(it)[1] == Int64[]
@test next(it)[1] == [1,2]
@test next(it)[1] == [1]
@test next(it)[1] == [2]
@test done(it)
it = ShapleyValues.statefuliterator(coalitions([1,2,3]))
@test next(it)[1] == Int64[]
@test next(it)[1] == [1,2,3]
@test next(it)[1] == [1]
@test next(it)[1] == [2,3]
@test next(it)[1] == [2]
@test next(it)[1] == [1,3]
@test next(it)[1] == [3]
@test next(it)[1] == [1,2]
@test done(it)
