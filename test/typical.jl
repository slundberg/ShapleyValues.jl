
typicalx = zeros(10)
x = Float64[1,1,1,1,1,0,0,0,0,0]
betas = randn(10,1)
model = x->betas'*x
vals,vars = shapleyvalues(x, model, typicalx, nsamples=1000)
@test all(abs(vals .- betas)[1:5] .< 1e-8)
@test abs(sum(vals) - (model(x)[1] - model(typicalx)[1])) < 1e-8
