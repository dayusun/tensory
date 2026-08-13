# Poisson CP Decomposition (CP-APR) via Multiplicative Updates

Fits a CP model to nonnegative (count) data by maximizing the Poisson
log-likelihood with the multiplicative-update algorithm of Chi & Kolda,
mirroring the MATLAB Tensor Toolbox `cp_apr` (`'mu'` method).

## Usage

``` r
cp_apr(
  X,
  R,
  tol = 1e-04,
  maxiters = 200L,
  maxinner = 10L,
  epsDivZero = 1e-10,
  init = "random",
  printitn = 0L
)
```

## Arguments

- X:

  A nonnegative Tensor or array-like object (typically counts).

- R:

  Target CP rank.

- tol:

  KKT-violation stopping tolerance (default `1e-4`).

- maxiters:

  Maximum number of outer iterations (default `200`).

- maxinner:

  Maximum inner updates per mode per outer iteration (default `10`).

- epsDivZero:

  Safeguard added before divisions (default `1e-10`).

- init:

  `"random"` or a list of initial nonnegative factor matrices.

- printitn:

  Print progress every `printitn` outer iterations.

## Value

A `KTensor` with nonnegative factors and weights.

## References

Chi, E. C. and Kolda, T. G. (2012). On tensors, sparsity, and
nonnegative factorizations. SIAM J. Matrix Anal. Appl. 33(4).

## Examples

``` r
set.seed(1)
X <- tensor(array(rpois(24, 3), dim = c(2, 3, 4)))
K <- cp_apr(X, R = 2, maxiters = 20)
```
