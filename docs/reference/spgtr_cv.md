# Choose the Sparsity of a Tensor Regression by Cross-Validation

Runs [`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md) over
a range of sparsity levels, scores each one by `nfolds`-fold
cross-validation, and returns the model refitted at the best value. Use
this when you want the method to decide by itself how much of the array
to keep.

## Usage

``` r
spgtr_cv(
  X,
  y,
  u = NULL,
  Z = NULL,
  family = stats::binomial(),
  lambda = NULL,
  nfolds = 5L,
  nlambda = 20L,
  lambda_ratio = 100,
  maxit = 500L,
  tol = 1e-08,
  ridge = 1e-08
)
```

## Arguments

- X, y, u, Z, family, maxit, tol, ridge:

  As in [`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md).

- lambda:

  Optional vector of sparsity levels to try. Leave `NULL` (default) to
  use an automatic log-spaced grid.

- nfolds:

  Number of cross-validation folds (default `5`).

- nlambda, lambda_ratio:

  Length of the automatic grid and the factor between its smallest and
  largest value (defaults `20` and `100`).

## Value

The `spgtr` fit at the selected sparsity, with three extra elements:
`lambda_min` (the chosen value), `lambda_seq` (the grid), and `cv` (a
data frame of `lambda` and mean out-of-fold `deviance`).

## Details

Folds are drawn at random once; within each fold the covariances, SIMPLS
bases, and adaptive weights are computed a single time and reused across
the whole `lambda` path, and each fit is warm-started from the previous
(less sparse) solution. Held-out fits are scored by the deviance of the
chosen `family`, which is the residual sum of squares for
[`gaussian()`](https://rdrr.io/r/stats/family.html) and twice the
negative log-likelihood (up to a constant) otherwise; smaller is better.

If `lambda` is not supplied, the grid runs from
`lambda_max / lambda_ratio` up to `lambda_max`, the value at which the
first proximal step would zero every row of every factor matrix. That
endpoint is a heuristic, so inspect `fit$cv` and widen `lambda_ratio` if
the selected value sits at either end of the grid.

## See also

[`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md),
[`summary.spgtr()`](https://www.sundayu.me/tensory/reference/summary.spgtr.md)

## Examples

``` r
set.seed(2)
B <- outer(c(2, rep(0, 5)), c(2, rep(0, 4)))
X <- lapply(1:100, function(i) matrix(rnorm(30), 6, 5))
eta <- vapply(X, function(xi) sum(B * xi), numeric(1))
y <- rbinom(100, 1, 1 / (1 + exp(-eta)))

fit <- spgtr_cv(X, y, u = c(1, 1), nfolds = 3, nlambda = 6)
fit$lambda_min
#> [1] 0.02645926
fit$cv
#>        lambda deviance
#> 1 0.001669467 31.54996
#> 2 0.004193511 30.52904
#> 3 0.010533623 27.03286
#> 4 0.026459264 20.61433
#> 5 0.066462666 22.11656
#> 6 0.166946670 25.20600
fit$selected # rows kept in each dimension
#> [[1]]
#> [1] 1 4
#> 
#> [[2]]
#> [1] 1 2 4 5
#> 
```
