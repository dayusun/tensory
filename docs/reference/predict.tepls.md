# Predict from a TEPLS Fit

Predict from a TEPLS Fit

## Usage

``` r
# S3 method for class 'tepls'
predict(object, newX = NULL, ...)
```

## Arguments

- object:

  A `tepls` object from
  [`tepls()`](https://dayusun.github.io/tensory/reference/tepls.md).

- newX:

  New predictors, in the same form accepted by
  [`tepls()`](https://dayusun.github.io/tensory/reference/tepls.md) (a
  list of observations or an order-`(m + 1)` tensor with observations in
  the last mode). If omitted, the fitted values are returned.

- ...:

  Unused.

## Value

A numeric vector of predictions (scalar response) or an `nnew x r`
matrix (multivariate response).
