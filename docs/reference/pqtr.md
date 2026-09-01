# Partial Quantile Tensor Regression (PQTR)

Fits a quantile regression in which each subject contributes a whole
array of measurements – an image, a connectivity matrix, a space-by-time
grid – together with an optional handful of ordinary covariates. Instead
of the mean of the outcome, `pqtr()` models a chosen quantile of it, so
it answers questions such as "which parts of the image go with an
unusually *low* score?" rather than only "which parts go with the
average score?".

## Usage

``` r
pqtr(
  X,
  y,
  tau = 0.5,
  Z = NULL,
  u = NULL,
  ridge = 1e-08,
  maxit = 200L,
  tol = 1e-08
)

# S3 method for class 'pqtr'
coef(object, ...)

# S3 method for class 'pqtr'
print(x, ...)
```

## Arguments

- X:

  The tensor predictor, in either of two forms: a list of `n` equally
  shaped `Tensor` objects, matrices, or arrays (one per subject), or a
  single array/`Tensor` of order `m + 1` whose **last** dimension
  indexes the subjects.

- y:

  The outcome, a numeric vector with one value per subject.

- tau:

  The quantile to model, a single number strictly between 0 and 1. `0.5`
  (default) is the median.

- Z:

  Optional `n x q` matrix or data frame of ordinary (non-array)
  covariates such as age or sex. These are never reduced.

- u:

  Number of directions kept per dimension of the array: an integer
  vector of length `m`, or a single number used for every dimension.
  Leave as `NULL` (default) to have each `u[k]` chosen by the
  eigenvalue-ratio rule of the reference implementation.

- ridge:

  Relative floor applied to covariance eigenvalues for numerical
  stability (default `1e-8`).

- maxit, tol:

  Iteration cap and convergence tolerance of the quantile regression
  solver.

- object:

  A fit from `pqtr()` or
  [`pqtr_cv()`](https://www.sundayu.me/tensory/reference/pqtr_cv.md).

- ...:

  Unused.

- x:

  A fit from `pqtr()` or
  [`pqtr_cv()`](https://www.sundayu.me/tensory/reference/pqtr_cv.md).

## Value

An object of class `pqtr`, a list whose most useful elements are:

- `coef`:

  coefficient array as a
  [TTensor](https://www.sundayu.me/tensory/reference/TTensor.md) with
  core `D` and weight matrices `W`; `coef(fit)` returns it and
  `as.tensor(coef(fit))` expands it to a dense
  [Tensor](https://www.sundayu.me/tensory/reference/Tensor.md).

- `alpha`, `gamma`:

  intercept and coefficients of `Z`, on the centered scale.

- `u`, `W`, `core`, `scores`:

  reduced dimensions, weight matrices, latent coefficients, and the
  `n x prod(u)` matrix of latent scores.

- `fitted`, `residuals`, `loss`:

  fitted conditional quantiles, their residuals, and the attained check
  loss.

## Details

Ordinary quantile regression cannot be run on the flattened array,
because one array holds far more numbers than there are subjects.
`pqtr()` solves that with a partial-least-squares construction: each
dimension of the array is compressed to a few directions chosen for
their association with the quantile of interest, the quantile regression
is fit on those, and the answer is mapped back to the original array
shape.

## What you get back

A coefficient array of exactly the same shape as one subject's data,
available through `coef(fit)`. A large positive entry means "a high
value at this position pushes the `tau`-th quantile of the outcome up".
Unlike [`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md)
there is no sparsity penalty, so every entry is non-zero; read the array
by magnitude rather than by which entries are exactly zero.

## How to use it (short version)

1.  Put your data in a list: `X[[i]]` is subject `i`'s matrix or array,
    all the same shape, and `y` is a numeric vector with one value per
    subject.

2.  Run `fit <- pqtr(X, y, tau = 0.5)` for the median, or another `tau`
    in `(0, 1)` for a different part of the outcome distribution.

3.  Fit several quantiles and compare the coefficient arrays: if they
    differ, the array affects the spread or shape of the outcome, not
    just its centre.

4.  Use
    [`pqtr_cv()`](https://www.sundayu.me/tensory/reference/pqtr_cv.md)
    if you would rather have the number of directions chosen by
    cross-validation than by the built-in eigenvalue-ratio rule.

## How it works (technical)

Writing `Q(tau)` for the fitted `tau`-th quantile of the outcome given
the covariates `Z` alone, the outcome enters only through the working
residual `tau - 1{y < Q(tau)}`, the subgradient of the check loss at the
covariate-only fit. Let `C` be the cross-covariance tensor between the
centered predictor and that residual, and `Sigma_k` the mode-`k`
marginal covariance of the predictor. The mode-`k` signal matrix is
`U_k = C_(k) (kron_{j != k} Sigma_j^-1) C_(k)'`, and the weight matrix
`W_k` collects `u[k]` directions obtained by repeatedly taking the
leading eigenvector of `U_k` and deflating with the oblique projector
`I - Sigma_k W (W' Sigma_k W)^-1 W'`. That projector leaves `W_k` with
orthonormal columns, which is the deflation used in the reference MATLAB
implementation; it differs from the SIMPLS deflation in
[`tepls()`](https://www.sundayu.me/tensory/reference/tepls.md) and
[`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md), which
instead makes the latent scores uncorrelated. The two agree whenever
`U_k` has rank one and span different subspaces otherwise.

The predictor is then reduced to latent scores
`T_i = X_i x_1 W_1' ... x_m W_m'`, the linear quantile regression of `y`
on `(Z, vec(T))` is fit, and the coefficient array is reconstructed as
`B = D x_1 W_1 ... x_m W_m` from the latent coefficients `D`. Nothing in
this chain inverts a `prod(p) x prod(p)` matrix, which is what makes the
method scale to large arrays.

## Divergences from the reference implementation

Ported from the MATLAB code accompanying Sun et al. (2024)
(<https://github.com/dayusun/PQTR>), with four deliberate differences.
(1) The inner quantile regressions are solved by the MM algorithm of
Hunter & Lange (2000) rather than by MATLAB's `fminunc`, which is a
smooth solver applied to a non-smooth objective; the MM surrogate is
exactly matched to the check loss. (2) The predictor is centered before
the latent scores are formed, so `alpha` is the intercept at the
training-sample means; the coefficient array `B` is unchanged by this.
(3) Cross-validated selection of the reduced dimension lives in
[`pqtr_cv()`](https://www.sundayu.me/tensory/reference/pqtr_cv.md),
which takes an explicit grid of candidate dimensions instead of
enumerating every combination. (4) The eigenvalue-ratio rule searches at
most five candidate dimensions per mode, rather than `sqrt(n - q)` of
them: past the rank of the signal matrix the eigenvalues are noise, and
the largest ratio among them otherwise wins and returns a dimension far
too big to estimate. This matches the rule used by
[`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md) and
[`tepls()`](https://www.sundayu.me/tensory/reference/tepls.md).

## References

Sun, D., Qiu, Z., Peng, L., Guo, Y. and Manatunga, A. (2024). Partial
quantile tensor regression. Journal of the American Statistical
Association 120(551), 1724-1735. doi:10.1080/01621459.2024.2422129

Hunter, D. R. and Lange, K. (2000). Quantile regression via an MM
algorithm. Journal of Computational and Graphical Statistics 9(1),
60-77.

## See also

[`pqtr_cv()`](https://www.sundayu.me/tensory/reference/pqtr_cv.md) to
choose `u` by cross-validation,
[`predict.pqtr()`](https://www.sundayu.me/tensory/reference/predict.pqtr.md),
and [`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md) /
[`tepls()`](https://www.sundayu.me/tensory/reference/tepls.md) for the
mean-regression counterparts.

## Examples

``` r
# 150 subjects measured on an 8 x 6 grid; only the top-left corner drives
# the outcome, and it does so more strongly in the upper tail.
set.seed(1)
B <- outer(c(1.5, rep(0, 7)), c(1, rep(0, 5)))
X <- lapply(1:150, function(i) matrix(rnorm(48), 8, 6))
signal <- vapply(X, function(xi) sum(B * xi), numeric(1))
y <- signal + (1 + 0.6 * signal) * rnorm(150)

med <- pqtr(X, y, tau = 0.5)
upper <- pqtr(X, y, tau = 0.9)
med
#> <pqtr: partial quantile tensor regression>
#> Quantile (tau):  0.5 
#> Subjects:        150 
#> Array shape:     8 x 6 
#> Directions (u):  1 2 (2 latent scores) 
#> Covariates (Z):  0 
#> Check loss:      77.0592 

# Coefficient arrays, same shape as one subject's data.
round(as.tensor(coef(med))$as_array()[1:3, 1:3], 2)
#>       [,1]  [,2]  [,3]
#> [1,]  1.21 -0.27 -0.26
#> [2,]  0.02 -0.01 -0.01
#> [3,] -0.14  0.03  0.03
round(as.tensor(coef(upper))$as_array()[1:3, 1:3], 2)
#>      [,1]  [,2] [,3]
#> [1,] 1.64 -0.54 0.10
#> [2,] 0.15 -0.04 0.01
#> [3,] 0.47 -0.15 0.04

# Predicted conditional quantiles for new subjects.
head(predict(med, X[1:5]))
#> [1] -0.3207214  0.1966758 -1.7688387 -2.0864251 -0.3739961
```
