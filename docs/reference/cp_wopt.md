# Weighted CP Decomposition via Direct Optimization

Fits a CP model to data with a weight (indicator) tensor by minimizing
`||W * (X - K)||^2`, mirroring the MATLAB Tensor Toolbox `cp_wopt`. Use
a 0/1 weight tensor to fit in the presence of missing entries.

## Usage

``` r
cp_wopt(
  X,
  W,
  R,
  init = "random",
  maxiters = 500L,
  factr = 1e+07,
  printitn = 0L
)
```

## Arguments

- X:

  A Tensor or array-like object (missing entries may hold any value,
  typically 0).

- W:

  A weight tensor of the same dimensions as `X` (commonly 0/1).

- R:

  Target CP rank.

- init:

  `"random"` or a list of initial factor matrices.

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
W <- tensor(array(rbinom(24, 1, 0.8), dim = c(2, 3, 4)))
K <- cp_wopt(X, W, R = 2)
```
