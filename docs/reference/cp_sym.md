# Symmetric CP Decomposition via Direct Optimization

Fits a symmetric CP model `sum_r lambda_r * u_r^(o m)` to a symmetric
tensor by minimizing the Frobenius residual over the weights and the
shared factor matrix with L-BFGS-B, mirroring the MATLAB Tensor Toolbox
`cp_sym`.

## Usage

``` r
cp_sym(
  X,
  R,
  init = "random",
  maxiters = 500L,
  factr = 1e+07,
  symmetrize = FALSE,
  printitn = 0L
)
```

## Arguments

- X:

  A symmetric Tensor (all modes the same size).

- R:

  Number of symmetric rank-one components.

- init:

  `"random"` or an `n x R` matrix of initial factors.

- maxiters:

  Maximum optimizer iterations (default `500`).

- factr:

  `optim` L-BFGS-B `factr` convergence parameter.

- symmetrize:

  Logical; if `TRUE`, symmetrize `X` first instead of requiring exact
  symmetry.

- printitn:

  If positive, print the optimizer trace.

## Value

A `SymKTensor`.

## Details

With `t_r = <X, u_r^(o m)>` and `c_rs = u_r . u_s`, the objective is
`||X||^2 - 2 sum_r lambda_r t_r + sum_rs lambda_r lambda_s c_rs^m`,
whose gradients are evaluated exactly using
[`ttsv()`](https://www.sundayu.me/tensory/reference/ttsv.md).

## Examples

``` r
set.seed(1)
u <- matrix(rnorm(6), 3, 2)
X <- as.tensor(symktensor(c(1, 2), u, m = 3))
S <- cp_sym(X, R = 2)
```
