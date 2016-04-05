
# generate data and ordinary least squares model
srand(1)
N = 1000000
P = 10
X = randn(N,P)
X .-= mean(X,1)
offsets = 0*randn(1,P)
X .+= offsets
betas = randn(P,1)
betas[5:end] = 0
model = x -> x'*betas
x = rand(P,1)
Xt = X'
y = model(Xt)
trueVals = vec(betas.*x - betas.*offsets');
baseValue = (betas.*offsets')[1]

# ensure that the total is with 10%
v = shapleyvalues(x, model, Xt, nsamples=10000)[1]
@test sum(abs(v .- trueVals))/sum(abs(trueVals)) < 0.1

# ensure that the adjusted total is perfect
trueSum = model(x) - baseValue
v = shapleyvalues(x, model, Xt, nsamples=10000, fnull=baseValue)[1]
@test abs(1 - sum(v)/sum(trueVals)) < 0.0001


# generate logistic regression data (this is a case where we know this won't work exactly)
logistic(x) = 1./(1+exp(-x))
logit(x) = -log((1-x)./x)
srand(1)
Nl = 100000
Xl = randn(Nl,4)
Xl .-= mean(Xl,1)
offsetsl = 0*randn(1,4)
Xl .+= offsetsl
xl = rand(4);
betasl = randn(4,1)
modell = x -> logistic(x'*betasl + 3)
ypredl = modell(Xl')
yl = Int64[rand() <= y for y in ypredl]
untrueValsl = vec(betasl.*xl - betasl.*offsetsl')
baseRate = mean(ypredl)

# ensure that the total is with 10%
vl = shapleyvalues(xl, modell, Xl', logit, nsamples=10000)[1]
trueDiff = logit(modell(xl)[1]) - logit(baseRate)
@test abs(sum(vl)-trueDiff)/abs(trueDiff) < 0.1

# ensure that the adjusted total is perfect
vl = shapleyvalues(xl, modell, Xl', logit, nsamples=10000, fnull=baseRate)[1]
@test abs(sum(vl)-trueDiff)/abs(trueDiff) < 0.0001

# this captured a bug with sample allocation at one point
srand(1)
N = 100000
P = 10
X = randn(N,P)
X .-= mean(X,1)
offsets = 0*randn(1,P)
X .+= offsets
betas = randn(P,1)
betas[5:end] = 0
model = x -> betas'*x
x = rand(P,1)
y = model(X')
trueVals = vec(betas.*x - betas.*offsets');

shapleyvalues(x, x->model(x), X', nsamples=1000, fnull=mean(y))
shapleyvalues(x, x->model(x), X', nsamples=1000, fnull=mean(y))
shapleyvalues(x, x->model(x), X', nsamples=1000, fnull=mean(y))
shapleyvalues(x, x->model(x), X', nsamples=1000, fnull=mean(y))
shapleyvalues(x, x->model(x), X', nsamples=1000, fnull=mean(y))
shapleyvalues(x, x->model(x), X', nsamples=1000, fnull=mean(y))

# check for a basic typicalx comparison
X = zeros(10,1)
x = Float64[1 1 1 1 1 0 0 0 0 0]'
betas = randn(10,1)
model = x->betas'*x
vals,vars = shapleyvalues(x, model, X, nsamples=1000, fnull=model(X)[1])
@test all(abs(vals .- betas)[1:5] .< 1e-8)
@test abs(sum(vals) - (model(x)[1] - model(X)[1])) < 1e-8
