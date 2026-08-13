# Higher-Order Singular Value Decomposition

Computes the (truncated) higher-order SVD of a tensor, returning a
Tucker representation whose factor matrices span the leading left
singular spaces of the mode-n unfoldings. Mirrors the MATLAB Tensor
Toolbox `hosvd`.

## Usage

``` r
hosvd(
  X,
  ranks = NULL,
  tol = NULL,
  dimorder = NULL,
  sequential = TRUE,
  verbosity = 0L
)
```

## Arguments

- X:

  A Tensor or array-like object.

- ranks:

  Optional integer vector of per-mode truncation ranks. Scalar inputs
  are replicated across modes.

- tol:

  Relative truncation tolerance. Ignored when `ranks` is supplied.
  Per-mode truncation keeps enough singular vectors to ensure the
  dropped energy is below `tol^2 / ndims(X)` of `fnorm(X)^2`.

- dimorder:

  Order in which modes are processed (matters only for
  `sequential = TRUE`).

- sequential:

  Logical; if `TRUE` (default) the ST-HOSVD variant is used,
  progressively contracting the core, which is typically more accurate
  and cheaper than the classical HOSVD.

- verbosity:

  Non-negative integer controlling diagnostic messages.

## Value

A `TTensor`.

## Examples

``` r
X <- tensor(array(runif(60), dim = c(3, 4, 5)))
T <- hosvd(X, ranks = c(2, 3, 4))
```
