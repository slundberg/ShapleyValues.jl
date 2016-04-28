# ShapleyValues

[![Build Status](https://travis-ci.org/slundberg/ShapleyValues.jl.svg?branch=master)](https://travis-ci.org/slundberg/ShapleyValues.jl)

This package is designed to explain individual predictions made by regression or classification methods. It does this by computing the [Shapley value](https://en.wikipedia.org/wiki/Shapley_value) for each feature or group of features. The Shapley value comes from game theory and is an additive representation of the contribution of each feature to an arbitrarily complex function.

This package build on methods in the paper [A General Method for Visualizing and Explaining Black-Box Regression Models](http://link.springer.com/chapter/10.1007%2F978-3-642-20267-4_3) by Strumbelj and Kononenko. However, it extends the ideas presented to allow for any non-linear link function (such as those used in generalized linear models), and also takes a different approach to estimation that can lead to much more stable results.

## Installation

```julia
Pkg.clone("https://github.com/slundberg/ShapleyValues.jl.git")
```


## Usage

### Least squares regression

```julia
using ShapleyValues

K = 4
X = rand(K,100)
beta = randn(K)
f(x) = x'beta
x = randn(K)

values,variances = shapleyvalues(x, f, X)
```

`x` is a specific data sample, `f` is a function for which we want the Shapley values over all the features, and `X` is a data matrix used to compute expectations. Typically `f` is a learned model, `X` is a representative subsample of the training dataset, and we want an additive representation of the prediction `f(x) = E[y | x]`. This is given by `E[y | x] - E[y] = \sum_i φ_i(x)` where `φ_i(x)` is the Shapley value for the `i`th feature group. Note that `E[y]` is the prediction of a model with no features provided, `E[y | x]` is the prediction when all features are provided, so the Shapley values additively account for the change in the prediction from base-line to the current prediction. Note that typically `E[y] != f(0)` and sampling under the assumption that features are independent is used to estimate the effect of not observing certain subsets of features.

The returned `values` are the Shapley values, while `variances` represents the estimated uncertainty in those estimates.

### Logistic regression

```julia
p(x) = x->logistic(f(x))
values,variances = shapleyvalues(x, p, X, logit)
```

While for least squares regression we had `E[y | x] - E[y] = \sum_i φ_i(x)`, for logistic regression we instead assume that the log-odds are additive (rather than the raw probabilities), this gives `logit(E[y | x]) - logit(E[y]) = \sum_i φ_i(x)`. Arbitrary link functions can be given.

Note that while using a `logit` link will improve performance for linear logistic regression, it will not create a perfectly additive model unless `X` has only one representative data point. This is because the expectations take place in probability space and then go through the non-linear `logit` link function.

### Feature grouping

```julia
values,variances = shapleyvalues(x, f, X, logit, Array[[1,2],[3],[4,5]])
```

The importance of groups of features can be computed by passing an array of group indexes. This is useful to group tightly coupled features without assuming independence between them.

### Controlling level of effort

The maximum number of subsets that are sampled can be set. This should always be greater than `2K_varying`, where `K_varying` is the number of feature groups where `x` and `X` have some difference (feature groups with no difference have a zero Shapley value).

```julia
values,variances = shapleyvalues(x, f, X, logit, nsamples=1000)
```

### Weighted datasets

The dataset `X` is intended to be a small set of samples representative of all possible samples. To better facilitate this weights are supported so some samples represent different proportions of the data space. These weights often come from k-means, k-medians or some other clustering method used to generate representative data points.

```julia
values,variances = shapleyvalues(x, f, X, logit, weights=[0.1,0.4,0.2,0.3])
```

## Performance

As opposed to the work by Strumbelj and Kononenko this code estimates the Shapley values using interaction subsets instead of permutations because there are `K! - 2^K` fewer of them (`K` is the number of feature groups). Some subsets (such as the singletons and their inverses) have much more weight than typical subsets. To take advantage of this we enumerate subsets by weight, alternating in a bias minimizing manner (see the `coalitions()` iterator). If the are `K` feature groups this allows `2/K` of the `K!` permutations to be covered with only two samples, `4/K` with `2K` samples, `6/K` with `2*(K choose 2)` samples, etc.

When `K` is large this leads to estimates driven by small subsets of features either held out, or included. This essentially explores the neighborhood surrounding the prediction `f(x)` and the base-line `E_x[f(x)]`. In other words the additive approximation will work best when included or holding out small subsets of features, which is an attractive property when we cannot exactly compute the Shapley values.

The provided function `f` should take in batches of samples in the same format as the representative data matrix `X`. By running in batches much faster execution is typically possible.
