# Tensor Envelope Partial Least Squares Regression (TEPLS)

Fits the tensor-predictor partial least squares regression of Zhang & Li
(2017). A tensor predictor `X` (order `m`) is reduced to a
low-dimensional latent tensor via per-mode envelope factor matrices
estimated with a SIMPLS iteration (their Algorithm 4), and the response
is regressed on the latent tensor. The coefficient tensor is
reconstructed in the original predictor space (their Lemma 2), giving
the `B_PLS` estimator.

## Usage

``` r
tepls(X, Y, u, ridge = 1e-08)
```

## Arguments

- X:

  Tensor predictor: a list of observations, or an order-`(m + 1)` tensor
  with observations in the last mode.

- Y:

  Response: numeric vector (length `n`) or `n x r` matrix.

- u:

  Envelope dimension per mode: an integer vector of length `m`, or a
  scalar recycled across modes. Each `u[k]` must satisfy
  `1 <= u[k] <= p_k`.

- ridge:

  Small ridge added when inverting covariance / normal-equation matrices
  for numerical stability (default `1e-8`).

## Value

An object of class `tepls`: a list with elements `coef` (the coefficient
`Tensor`, order `m` for scalar response or order `m + 1` with a trailing
response mode otherwise), `W` (list of factor matrices), `intercept`,
`Xbar`, `dims`, `u`, `r`, and `fitted`.

## Details

The predictor may be supplied either as a list of `n` `Tensor`/array
observations that share the same dimensions, or as a single
order-`(m + 1)` `Tensor`/array whose **last** mode indexes the `n`
observations. The response `Y` is a length-`n` numeric vector (scalar
response) or an `n x r` matrix (multivariate response).

Each mode's factor matrix `W_k` has `u[k]` columns; `u` is the envelope
dimension per mode (recycled if a scalar). The mode-`k` marginal
covariance uses the moment estimator (their eq. 1), the mode-`k`
cross-covariance is standardized as in Algorithm 4, and the reduced
regression of `Y` on the latent tensor is fit by (regularized) least
squares (their Step 6).

The mode-`k` second-moment matrix maximized in the SIMPLS step,
`C_(k) (Sigma_Y^-1 kron ... kron Sigma_1^-1) C_(k)^T` (excluding
`Sigma_k`), is identical to the `U U^T` matrix in the reference
implementation `TEReg::TensPLS_fit`. Two deliberate divergences: (1)
that package estimates each mode's basis with an envelope (`EnvMU`)
optimizer, whereas this function uses the SIMPLS deflation of Algorithm
4 exactly; (2) the estimated factor directions and the
reduced-regression fit are both invariant to the per-mode scale
ambiguity of the separable covariance, avoiding the need to pin the
Kronecker scale.

## References

Zhang, X. and Li, L. (2017). Tensor envelope partial least-squares
regression. Technometrics 59(4), 426-436.

## See also

[`predict.tepls()`](https://dayusun.github.io/tensory/reference/predict.tepls.md)

## Examples

``` r
set.seed(1)
p <- c(8, 6)
B <- outer(c(1, rep(0, 7)), c(1, rep(0, 5))) # rank-1, envelope dim 1 per mode
X <- lapply(1:80, function(i) matrix(rnorm(prod(p)), p[1], p[2]))
y <- vapply(X, function(xi) sum(B * xi), numeric(1)) + rnorm(80, sd = 0.1)
fit <- tepls(X, y, u = c(1, 1))
```
