# Predict from a Partial Quantile Tensor Regression Fit

Predict from a Partial Quantile Tensor Regression Fit

## Usage

``` r
# S3 method for class 'pqtr'
predict(object, newX = NULL, newZ = NULL, ...)
```

## Arguments

- object:

  A fit from
  [`pqtr()`](https://www.sundayu.me/tensory/reference/pqtr.md) or
  [`pqtr_cv()`](https://www.sundayu.me/tensory/reference/pqtr_cv.md).

- newX:

  New subjects, in either form accepted by
  [`pqtr()`](https://www.sundayu.me/tensory/reference/pqtr.md). If
  omitted, the fitted values for the subjects used in fitting are
  returned.

- newZ:

  New ordinary covariates, required when the fit used `Z`.

- ...:

  Unused.

## Value

A numeric vector of predicted conditional `tau`-th quantiles, one per
subject.

## Examples

``` r
set.seed(3)
X <- lapply(1:80, function(i) matrix(rnorm(20), 5, 4))
y <- vapply(X, function(xi) xi[1, 1], numeric(1)) + rnorm(80)
fit <- pqtr(X, y, tau = 0.5, u = c(1, 1))
predict(fit, X[1:5])
#> [1] -1.2025514 -0.6799958 -1.4658828 -0.6715724  0.4085330
```
