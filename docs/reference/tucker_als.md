# Tucker Alternating Least Squares (HOOI)

Computes a Tucker decomposition of a dense tensor with target
multilinear ranks via Higher-Order Orthogonal Iteration. Mirrors the
MATLAB Tensor Toolbox `tucker_als`.

## Usage

``` r
tucker_als(
  X,
  ranks,
  tol = 1e-04,
  maxiters = 50L,
  dimorder = NULL,
  init = "nvecs",
  printitn = 0L
)
```

## Arguments

- X:

  A Tensor or array-like object.

- ranks:

  Integer vector of per-mode target ranks, or a scalar replicated across
  modes.

- tol:

  Convergence tolerance on change in fit (default `1e-4`).

- maxiters:

  Maximum number of HOOI sweeps (default `50`).

- dimorder:

  Integer permutation giving the order in which factor matrices are
  updated.

- init:

  Either `"nvecs"` (leading left singular vectors, default), `"random"`
  (random orthonormal), or a list of initial factor matrices.

- printitn:

  Print fit every `printitn` iterations (`0` to suppress).

## Value

A `TTensor`.

## Examples

``` r
set.seed(1)
X <- tensor(array(runif(60), dim = c(3, 4, 5)))
T <- tucker_als(X, ranks = c(2, 3, 3), maxiters = 20)
```
