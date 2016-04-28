
# DenseSampleSpace
s = ShapleyValues.samplespace(zeros(4,2), 100)
@test size(ShapleyValues.data(s)) == (4,0)
ShapleyValues.addsample!(s, ones(4))
@test ShapleyValues.data(s) == ones(4,1)
ShapleyValues.reset!(s)
@test size(ShapleyValues.data(s)) == (4,0)
ShapleyValues.addsample!(s, ones(4))
ShapleyValues.addsample!(s, ones(4))
@test ShapleyValues.data(s) == ones(4,2)
