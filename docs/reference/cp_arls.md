# CP Decomposition via Randomized (Sampled) ALS

Alternating least squares in which each subproblem is solved from a
uniform sample of tensor fibers rather than the full unfolding, in the
spirit of the MATLAB Tensor Toolbox `cp_arls`. Unlike `cp_arls`, no
FFT-based mixing is applied before sampling (a documented divergence);
sampling is plain uniform with replacement.

## Usage

``` r
cp_arls(
  X,
  R,
  tol = 1e-04,
  maxiters = 50L,
  nsamples = NULL,
  ridge = 1e-10,
  init = "random",
  printitn = 0L
)
```

## Arguments

- X:

  A Tensor or array-like object.

- R:

  Target CP rank.

- tol:

  Convergence tolerance on change in (exact) fit.

- maxiters:

  Maximum number of sweeps (default `50`).

- nsamples:

  Number of sampled fibers per solve. Defaults to
  `max(ceiling(10 * R * log2(R + 1)), 4 * R)` capped at the full count.

- ridge:

  Tikhonov regularizer added to the sampled normal equations (default
  `1e-10`).

- init:

  `"random"` or a list of initial factor matrices.

- printitn:

  Print fit every `printitn` iterations.

## Value

A `KTensor`.

## Examples

``` r
set.seed(1)
X <- tensor(array(rnorm(60), dim = c(3, 4, 5)))
K <- cp_arls(X, R = 2, maxiters = 20)
```
