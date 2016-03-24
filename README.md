# ShapleyValues

[![Build Status](https://travis-ci.org/slundberg/ShapleyValues.jl.svg?branch=master)](https://travis-ci.org/slundberg/ShapleyValues.jl)

An optimized implementation of [A General Method for Visualizing and Explaining Black-Box Regression Models](http://link.springer.com/chapter/10.1007%2F978-3-642-20267-4_3).

## Usage

```julia
values,variances = shapley_values(x, f, Xt)
```

`x` is a specfic data instance, `f` is a function for which we want the Shapley values over all the features, `Xt` is a data matrix used to compute expectations. Typically `f` is a learned model, `Xt` is the training dataset, and we want an additive explaination of the prediction `f(x)`.

Unlike the paper, this code optimally distributes samples in batches, this allows the function to be run on many samples at once, which can make it much more efficient. It also identifies features that rarely change and avoids ever computing the model on these features. Finally it avoids running the model on samples with a difference guaranteed to be zero, again increasing performance on discrete data. For more options and details see the code.
