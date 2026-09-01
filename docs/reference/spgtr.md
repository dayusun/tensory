# Sparse Partial Generalized Tensor Regression (SPGTR)

Fits a regression in which each subject's predictor is a whole array – a
brain image, say – and the outcome is a single number, such as a yes/no
diagnosis, a count, or a continuous score. One such array holds far more
numbers than there are subjects, which rules out ordinary regression.
`spgtr()` fits such a model, and with `lambda > 0` selects the rows,
columns, or slices of the array that enter it.

## Usage

``` r
spgtr(
  X,
  y,
  u = NULL,
  Z = NULL,
  family = stats::binomial(),
  basis = c("envelope", "simpls"),
  lambda = 0,
  maxit = 500L,
  tol = 1e-08,
  ridge = 1e-08
)

# S3 method for class 'spgtr'
coef(object, ...)

# S3 method for class 'spgtr'
print(x, ...)
```

## Arguments

- X:

  The tensor predictor, in either of two forms: a list of `n` equally
  shaped `Tensor` objects, matrices, or arrays (one per subject), or a
  single array/`Tensor` of order `m + 1` whose **last** dimension
  indexes the subjects.

- y:

  The outcome, with one entry per subject. For
  [`binomial()`](https://rdrr.io/r/stats/family.html) this may be a 0/1
  numeric vector, a logical vector, or a two-level factor (the first
  level is treated as the reference).

- u:

  Number of directions kept per dimension of the array: an integer
  vector of length `m`, or a single number used for every dimension.
  Leave as `NULL` (default) to have each `u[k]` chosen automatically by
  an eigenvalue-ratio rule. Larger values fit more flexible models; `1`
  or `2` per mode is typical.

- Z:

  Optional `n x q` matrix or data frame of ordinary (non-array)
  covariates such as age or sex. These are never penalized or reduced.

- family:

  The outcome type, as a name, a family function, or a family object:
  [`binomial()`](https://rdrr.io/r/stats/family.html) (default) for
  yes/no, [`poisson()`](https://rdrr.io/r/stats/family.html) for counts,
  [`gaussian()`](https://rdrr.io/r/stats/family.html) for continuous
  outcomes.

- basis:

  `"envelope"` (default) refines the SIMPLS directions by envelope
  optimization and is required for sparsity; `"simpls"` uses the
  closed-form SIMPLS directions alone, which is faster and needs no
  iteration.

- lambda:

  Amount of sparsity, `>= 0`. `0` (default) keeps every row of every
  factor matrix; larger values remove more slices of the array. Use
  [`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md)
  if you do not want to pick this by hand. Ignored when
  `basis = "simpls"`.

- maxit, tol:

  Iteration cap and stationarity tolerance of the manifold solver.

- ridge:

  Relative floor applied to covariance eigenvalues for numerical
  stability (default `1e-8`).

- object:

  A fit from `spgtr()` or
  [`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md).

- ...:

  Unused.

- x:

  A fit from `spgtr()` or
  [`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md).

## Value

An object of class `spgtr`, a list whose most useful elements are:

- `coef`:

  coefficient array as a
  [TTensor](https://www.sundayu.me/tensory/reference/TTensor.md) with
  core `D` and factor matrices `W`; `coef(fit)` returns it and
  `as.tensor(coef(fit))` expands it to a dense
  [Tensor](https://www.sundayu.me/tensory/reference/Tensor.md).

- `alpha`, `gamma`:

  intercept and coefficients of `Z`.

- `selected`, `nonzero`:

  indices, and counts, of the retained rows in each dimension.

- `W`, `core`, `scores`:

  factor matrices, latent coefficients, and the `n x prod(u)` matrix of
  latent scores.

- `fitted`, `linear.predictors`, `deviance`, `null.deviance`:

  as in a [`stats::glm()`](https://rdrr.io/r/stats/glm.html) fit.

## What you get back

`coef(fit)` is a coefficient array of the same shape as one subject's
data. A large positive entry means a high value at that position pushes
the outcome up; a zero entry means the position went unused. With
`lambda > 0` whole rows, columns, or slices are zero, so `fit$selected`
and
[`summary.spgtr()`](https://www.sundayu.me/tensory/reference/summary.spgtr.md)
report which parts of the array entered the model.

## How to use it

1.  Put the data in a list: `X[[i]]` is subject `i`'s matrix or array,
    all the same shape; `y` is a vector with one entry per subject.

2.  `fit <- spgtr(X, y)` fits a yes/no outcome; add `family = poisson()`
    or `family = gaussian()` for counts or continuous outcomes.

3.  `summary(fit)` and `coef(fit)` describe the fit, and
    `predict(fit, newX)` predicts new subjects.

4.  [`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md)
    chooses the amount of sparsity by cross-validation, and with it
    which parts of the array enter the model.

## How it works

Each mode of the array is compressed to a few directions carrying the
association with the outcome, the generalized linear model is fitted on
the compressed predictor by
[`stats::glm.fit()`](https://rdrr.io/r/stats/glm.html), and its
coefficients are expanded back to the shape of one subject's array. See
the reference below for the estimator and its properties.

## Speed

The mode-wise covariances and the manifold solver are compiled kernels,
and the latent scores go through the compiled
[`ttm()`](https://www.sundayu.me/tensory/reference/ttm.md). Reference
implementations in R are used automatically if the package was installed
without compilation, and the two paths agree to numerical tolerance.

## References

Sun, D., Peng, L., Qiu, Z., Stevens, J., Manatunga, A. and Guo, Y.
Sparse partial generalized tensor regression with application to
neuroimaging data. Submitted.

Zhang, X. and Li, L. (2017). Tensor envelope partial least-squares
regression. Technometrics 59(4), 426-436.

Cook, R. D. and Zhang, X. (2016). Algorithms for envelope estimation.
Journal of Computational and Graphical Statistics 25(1), 284-300.

Xiao, N., Liu, X. and Yuan, Y. (2021). Exact penalty function for L21
norm minimization over the Stiefel manifold. SIAM Journal on
Optimization 31(4), 3097-3126.

## See also

[`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md) to
choose `lambda`,
[`predict.spgtr()`](https://www.sundayu.me/tensory/reference/predict.spgtr.md),
[`summary.spgtr()`](https://www.sundayu.me/tensory/reference/summary.spgtr.md),
and [`tepls()`](https://www.sundayu.me/tensory/reference/tepls.md) for
the continuous-response version.

## Examples

``` r
# 120 subjects, each measured on an 8 x 6 grid; only the top-left corner
# of the grid actually drives the yes/no outcome.
set.seed(1)
B <- outer(c(1.5, rep(0, 7)), c(1.5, rep(0, 5)))
X <- lapply(1:120, function(i) matrix(rnorm(48), 8, 6))
eta <- vapply(X, function(xi) sum(B * xi), numeric(1))
y <- rbinom(120, 1, 1 / (1 + exp(-eta)))

fit <- spgtr(X, y, u = c(1, 1))
summary(fit)
#> <spgtr: sparse partial generalized tensor regression>
#> Outcome:         binomial with logit link
#> Subjects:        120 
#> Array shape:     8 x 6 
#> Directions (u):  1 1 
#> Basis:           envelope (lambda = 0) 
#> Slices kept:     8/8  6/6 
#> Deviance:        89.1411 
#> 
#> Slices used, by dimension:
#>   dim 1 (8): all
#>   dim 2 (6): all
#> 
#> Deviance explained: 46.4% (in-sample)
#> Coefficient array norm: 2.638
#> Accuracy: 0.825    AUC: 0.914  (in-sample)

# Coefficient array, same shape as one subject's data.
round(as.tensor(coef(fit))$as_array(), 2)
#>       [,1]  [,2]  [,3]  [,4] [,5]  [,6]
#> [1,]  2.36 -0.54 -0.32 -0.37 0.02  0.08
#> [2,] -0.28  0.07  0.04  0.04 0.00 -0.01
#> [3,] -0.03  0.01  0.00  0.00 0.00  0.00
#> [4,]  0.19 -0.04 -0.03 -0.03 0.00  0.01
#> [5,] -0.04  0.01  0.01  0.01 0.00  0.00
#> [6,]  0.36 -0.08 -0.05 -0.06 0.00  0.01
#> [7,] -0.27  0.06  0.04  0.04 0.00 -0.01
#> [8,]  0.68 -0.16 -0.09 -0.11 0.01  0.02

# Predicted probabilities and labels.
head(predict(fit, type = "response"))
#> [1] 0.26167064 0.55517741 0.09216838 0.03832707 0.03819162 0.68289116
head(predict(fit, type = "class"))
#> [1] 0 1 0 0 0 1

# With covariates, and with a count outcome.
Z <- cbind(age = rnorm(120))
fit_z <- spgtr(X, y, u = c(1, 1), Z = Z)
counts <- rpois(120, exp(eta / 2))
fit_p <- spgtr(X, counts, u = c(1, 1), family = poisson())
```
