# CP Decomposition via Direct Optimization

Fits a CP model by minimizing `||X - K||^2` over all factor matrices
simultaneously with L-BFGS-B, mirroring the MATLAB Tensor Toolbox
`cp_opt`.

## Usage

``` r
cp_opt(
  X,
  R,
  init = "random",
  lower = -Inf,
  maxiters = 500L,
  factr = 1e+07,
  printitn = 0L
)
```

## Arguments

- X:

  A Tensor or array-like object.

- R:

  Target CP rank.

- init:

  `"random"` (scaled normal) or a list of initial factor matrices.

- lower:

  Optional lower bound for all factor entries (e.g. `0` for a
  nonnegative model); default unbounded.

- maxiters:

  Maximum optimizer iterations (default `500`).

- factr:

  `optim` L-BFGS-B `factr` convergence parameter.

- printitn:

  If positive, print the optimizer trace.

## Value

A `KTensor`.

## Examples

``` r
set.seed(1)
X <- tensor(array(rnorm(24), dim = c(2, 3, 4)))
K <- cp_opt(X, R = 2)
```
