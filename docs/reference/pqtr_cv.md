# Choose the Reduced Dimension of a Quantile Tensor Regression

Runs [`pqtr()`](https://www.sundayu.me/tensory/reference/pqtr.md) over a
grid of candidate reduced dimensions, scores each one by `nfolds`-fold
cross-validated check loss, and returns the model refitted on all
subjects at the best value. Use this when the eigenvalue-ratio rule
built into [`pqtr()`](https://www.sundayu.me/tensory/reference/pqtr.md)
is not obviously right, for instance when the coefficient array is not
close to rank one.

## Usage

``` r
pqtr_cv(
  X,
  y,
  tau = 0.5,
  Z = NULL,
  u_grid = NULL,
  nfolds = 5L,
  ridge = 1e-08,
  maxit = 200L,
  tol = 1e-08
)
```

## Arguments

- X, y, tau, Z, ridge, maxit, tol:

  As in [`pqtr()`](https://www.sundayu.me/tensory/reference/pqtr.md).

- u_grid:

  Optional list of candidate dimension vectors, each of length `m` (a
  plain integer vector is read as one candidate per element, used in
  every mode). Leave `NULL` (default) for the automatic equal-dimension
  grid.

- nfolds:

  Number of cross-validation folds (default `5`).

## Value

The `pqtr` fit at the selected dimension, with three extra elements:
`u_grid` (the candidates tried), `cv` (a data frame of `u` and mean
out-of-fold check `loss`), and `u_min` (the chosen dimension vector).

## Details

The default grid keeps the same number of directions in every mode,
`u = rep(d, m)` for `d` from 1 up to the largest value for which the
reduced quantile regression stays estimable, capped at 12 as in the
reference implementation. That is a deliberate simplification:
enumerating every combination of per-mode dimensions costs `d^m` fits.
Pass `u_grid` explicitly, for instance
`u_grid = apply(expand.grid(1:2, 1:3), 1, as.integer, simplify = FALSE)`,
when the modes should be allowed to differ.

## See also

[`pqtr()`](https://www.sundayu.me/tensory/reference/pqtr.md)

## Examples

``` r
set.seed(2)
B <- outer(c(2, rep(0, 5)), c(2, rep(0, 4)))
X <- lapply(1:120, function(i) matrix(rnorm(30), 6, 5))
y <- vapply(X, function(xi) sum(B * xi), numeric(1)) + rnorm(120)

fit <- pqtr_cv(X, y, tau = 0.5, nfolds = 3)
fit$u_min
#> [1] 5 5
fit$cv
#>       u     loss
#> 1 1 x 1 27.13937
#> 2 2 x 2 27.28447
#> 3 3 x 3 25.10003
#> 4 4 x 4 21.85729
#> 5 5 x 5 20.12311
```
