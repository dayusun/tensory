# Nonnegative CP Decomposition via Multiplicative Updates

Computes a nonnegative CP decomposition with Lee-Seung style
multiplicative updates, mirroring the MATLAB Tensor Toolbox `cp_nmu`.
All entries of `X` must be nonnegative; factors stay nonnegative
throughout.

## Usage

``` r
cp_nmu(X, R, tol = 1e-04, maxiters = 100L, init = "random", printitn = 0L)
```

## Arguments

- X:

  A nonnegative Tensor or array-like object.

- R:

  Target CP rank.

- tol:

  Convergence tolerance on change in fit (default `1e-4`).

- maxiters:

  Maximum number of sweeps (default `100`).

- init:

  `"random"` (uniform) or a list of nonnegative factor matrices.

- printitn:

  Print fit every `printitn` iterations (`0` to suppress).

## Value

A `KTensor` with nonnegative factors.

## Examples

``` r
set.seed(1)
X <- tensor(array(runif(24), dim = c(2, 3, 4)))
K <- cp_nmu(X, R = 2, maxiters = 20)
```
