# Generalized Eigenproblem Adaptive Power Method (GEAP)

Computes a generalized tensor eigenpair `A x^(m-1) = lambda B x^(m-1)`,
`||x|| = 1`, for symmetric `A` and symmetric positive definite `B` using
the GEAP method of Kolda & Mayo (Algorithm 1), mirroring the MATLAB
Tensor Toolbox `eig_geap`. With `B = teneye(m, n)` this reduces to
[`eig_sshopm()`](https://dayusun.github.io/tensory/reference/eig_sshopm.md).

## Usage

``` r
eig_geap(
  A,
  B,
  maximize = TRUE,
  start = NULL,
  tol = 1e-10,
  maxiters = 500L,
  tau = 1e-06
)
```

## Arguments

- A:

  A symmetric Tensor (all modes the same size).

- B:

  A symmetric positive definite Tensor of the same size and order.

- maximize:

  Logical; `TRUE` (default) seeks local maxima (`beta = 1`), `FALSE`
  seeks local minima (`beta = -1`).

- start:

  Optional starting vector; random normal if omitted.

- tol:

  Convergence tolerance on the eigenvalue change (default `1e-10`).

- maxiters:

  Maximum iterations (default `500`).

- tau:

  Positive-definiteness threshold for the adaptive shift (default
  `1e-6`).

## Value

A list with elements `lambda`, `x`, `converged`, `iterations`, and
`lambda_trace`.

## References

Kolda, T. G. and Mayo, J. R. (2014). An adaptive shifted power method
for computing generalized tensor eigenpairs. SIAM J. Matrix Anal. Appl.
35(4), 1563-1581.

## Examples

``` r
set.seed(1)
A <- symmetrize(tensor(array(rnorm(81), dim = c(3, 3, 3, 3))))
res <- eig_geap(A, teneye(4, 3))
```
