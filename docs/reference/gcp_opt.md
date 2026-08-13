# Generalized CP Decomposition

Fits a CP model by minimizing the sum of an arbitrary elementwise loss
`f(x, m)` between the data and the model with L-BFGS-B, mirroring the
dense/deterministic mode of the MATLAB Tensor Toolbox `gcp_opt`.

## Usage

``` r
gcp_opt(
  X,
  R,
  type = "gaussian",
  init = "random",
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

- type:

  Loss name: one of `"gaussian"` (alias `"normal"`), `"poisson"` (alias
  `"count"`), `"poisson-log"`, `"bernoulli-odds"`, `"bernoulli-logit"`,
  `"rayleigh"`, `"gamma"`, `"huber"`. Alternatively a list with elements
  `f(x, m)`, `g(x, m)` (the derivative in `m`), and `lower` (factor
  lower bound, `-Inf` if unconstrained) for a custom loss.

- init:

  `"random"`, `"nvecs"`, or a list of initial factor matrices.

- maxiters:

  Maximum optimizer iterations (default `500`).

- factr:

  `optim` L-BFGS-B `factr` convergence parameter.

- printitn:

  If positive, print the optimizer trace.

## Value

A list with elements `K` (the fitted `KTensor`) and `objective` (the
final loss value).

## References

Hong, D., Kolda, T. G., and Duersch, J. A. (2020). Generalized canonical
polyadic tensor decomposition. SIAM Review 62(1).

## Examples

``` r
set.seed(1)
X <- tensor(array(rpois(24, 3), dim = c(2, 3, 4)))
res <- gcp_opt(X, R = 2, type = "poisson", maxiters = 100)
```
