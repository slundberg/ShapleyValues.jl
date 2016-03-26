# ShapleyValues

[![Build Status](https://travis-ci.org/slundberg/ShapleyValues.jl.svg?branch=master)](https://travis-ci.org/slundberg/ShapleyValues.jl)

This package is designed to explain individual predictions made by regression or classification methods. It does this by computing the [Shapley value](https://en.wikipedia.org/wiki/Shapley_value) for each feature or group of features. The Shapley value comes from game theory and is an additive representation of the contribution of each feature to an arbitrarily complex function.

This package is inspired by the paper [A General Method for Visualizing and Explaining Black-Box Regression Models](http://link.springer.com/chapter/10.1007%2F978-3-642-20267-4_3) by Strumbelj and Kononenko. However, it extends the ideas presented to allow for non-linear link functions from generalized linear models, and includes several performance enhancements.

## Installation

```julia
Pkg.clone("https://github.com/slundberg/ShapleyValues.jl.git")
```


## Usage

### Least squares regression

```julia
using ShapleyValues

values,variances = shapley_values(x, f, Xt)
```

`x` is a specfic data instance, `f` is a function for which we want the Shapley values over all the features, `Xt` is a data matrix used to compute expectations. Typically `f` is a learned model, `Xt` is the training dataset, and we want an additive explaination of the prediction `f(x) = E[y | x]`. This is given by `E[y | x] - E[y] = \sum_i φ_i(x)` where `φ_i(x)` is the Shapley value for the `i`th feature group. Note that `E[y]` is the prediction of a model with no features provided, `E[y | x]` is the prediction when all features are provided, so the Shapley values additively account for the change in the prediction from base line to the current prediction. Note that typically `E[y] != f(0)` and sampling under the assuption that features are independeny is used to estimate the effect of not observing certain subsets of features.

### Logistic regression

```julia
using ShapleyValues
using GLM

values,variances = shapley_values(x, f, Xt, LogisticLink())
```

While for least square regression we had `E[y | x] - E[y] = \sum_i φ_i(x)`, for logistic regression we instead assume that the log-odds are additive (rather than the raw probabilities), this gives `logit(E[y | x]) - logit(E[y]) = \sum_i φ_i(x)`. For other supported link functions see the top of the main source code file.

### Feature grouping

```julia
using ShapleyValues
using GLM

values,variances = shapley_values(x, f, Xt, LogisticLink(), Array[[1,2],[3],[4,5]])
```

The importance of groups of features can be computed by passing an array group indexes.

## Optimizations

Unlike the paper, this code optimally distributes samples in batches, this allows the model function `f` to be run on many samples at once, which can make it much more efficient. It also identifies features that rarely change and avoids ever computing `f` on these features. Finally it avoids running `f` on samples with a difference guaranteed to be zero, again increasing performance on discrete data. For more options and details see the code.
