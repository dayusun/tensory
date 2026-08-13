# Predict from a Tensor Regression Fit

Predict from a Tensor Regression Fit

## Usage

``` r
# S3 method for class 'spgtr'
predict(
  object,
  newX = NULL,
  newZ = NULL,
  type = c("response", "link", "class"),
  ...
)
```

## Arguments

- object:

  A fit from
  [`spgtr()`](https://dayusun.github.io/tensory/reference/spgtr.md) or
  [`spgtr_cv()`](https://dayusun.github.io/tensory/reference/spgtr_cv.md).

- newX:

  New subjects, in either form accepted by
  [`spgtr()`](https://dayusun.github.io/tensory/reference/spgtr.md). If
  omitted, predictions for the subjects used in fitting are returned.

- newZ:

  New ordinary covariates, required when the fit used `Z`.

- type:

  `"response"` (default) for the outcome scale – a probability for
  [`binomial()`](https://rdrr.io/r/stats/family.html), an expected count
  for [`poisson()`](https://rdrr.io/r/stats/family.html); `"link"` for
  the linear predictor; `"class"` for a 0/1 label thresholded at `0.5`
  (binomial fits only).

- ...:

  Unused.

## Value

A numeric vector with one prediction per subject.

## Examples

``` r
set.seed(3)
X <- lapply(1:80, function(i) matrix(rnorm(20), 5, 4))
y <- rbinom(80, 1, 0.5)
fit <- spgtr(X, y, u = c(1, 1))
predict(fit, X[1:5], type = "response")
#> [1] 0.6807172 0.3221624 0.6238999 0.6248340 0.3706757
```
