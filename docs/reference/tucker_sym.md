# Symmetric Tucker Decomposition

Computes a Tucker decomposition with a single orthonormal factor matrix
shared by every mode via higher-order orthogonal iteration on the
symmetric subspace, mirroring the MATLAB Tensor Toolbox `tucker_sym`.

## Usage

``` r
tucker_sym(
  X,
  r,
  tol = 1e-06,
  maxiters = 100L,
  init = "nvecs",
  symmetrize = FALSE
)
```

## Arguments

- X:

  A symmetric Tensor (all modes the same size).

- r:

  Target subspace rank.

- tol:

  Convergence tolerance on change in core norm (default `1e-6`).

- maxiters:

  Maximum number of iterations (default `100`).

- init:

  `"nvecs"` (default) or an initial `n x r` matrix with orthonormal
  columns.

- symmetrize:

  Logical; if `TRUE`, symmetrize `X` first.

## Value

A `TTensor` whose factor matrices are all identical.

## Examples

``` r
set.seed(1)
X <- symmetrize(tensor(array(rnorm(27), dim = c(3, 3, 3))))
T <- tucker_sym(X, r = 2)
```
