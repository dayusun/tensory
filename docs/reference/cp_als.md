# CP Alternating Least Squares Decomposition

Computes a rank-R canonical polyadic (CP) decomposition of a dense
tensor by alternating least squares, mirroring the behavior of the
MATLAB Tensor Toolbox `cp_als`.

## Usage

``` r
cp_als(
  X,
  R,
  tol = 1e-04,
  maxiters = 50L,
  dimorder = NULL,
  init = "random",
  printitn = 0L,
  fixsigns = TRUE
)
```

## Arguments

- X:

  A Tensor or array-like object.

- R:

  Target CP rank (positive integer).

- tol:

  Convergence tolerance on change in fit (default `1e-4`).

- maxiters:

  Maximum number of ALS sweeps (default `50`).

- dimorder:

  Integer permutation of `1:ndims(X)` giving the order in which factor
  matrices are updated. Defaults to `1:ndims(X)`.

- init:

  Either `"random"` (i.i.d. normal), `"nvecs"` (leading left singular
  vectors of the mode-n unfolding; falls back to random for modes where
  the mode size is smaller than `R`), or a list of initial factor
  matrices.

- printitn:

  Print fit every `printitn` iterations (`0` to suppress).

- fixsigns:

  Logical; if `TRUE`, resolve sign ambiguity of the returned components.

## Value

A `KTensor` giving the rank-R CP decomposition of `X`.

## Examples

``` r
set.seed(1)
X <- tensor(array(runif(24), dim = c(2, 3, 4)))
K <- cp_als(X, R = 2, maxiters = 20)
```
