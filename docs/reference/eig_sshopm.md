# Shifted Symmetric Higher-Order Power Method (SS-HOPM)

Computes a Z-eigenpair `A x^(m-1) = lambda x`, `||x|| = 1`, of a
symmetric tensor by the shifted symmetric higher-order power method with
the adaptive shift of Kolda & Mayo, mirroring the MATLAB Tensor Toolbox
`eig_sshopm`.

## Usage

``` r
eig_sshopm(
  A,
  shift = "adaptive",
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

- shift:

  `"adaptive"` (default) or a fixed numeric shift.

- maximize:

  Logical; `TRUE` (default) seeks local maxima of `A x^m` (`beta = 1`),
  `FALSE` seeks local minima (`beta = -1`).

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

Kolda, T. G. and Mayo, J. R. (2011). Shifted power method for computing
tensor eigenpairs. SIAM J. Matrix Anal. Appl. 32(4); Kolda, T. G. and
Mayo, J. R. (2014). An adaptive shifted power method for computing
generalized tensor eigenpairs. SIAM J. Matrix Anal. Appl. 35(4).

## Examples

``` r
set.seed(1)
A <- symmetrize(tensor(array(rnorm(81), dim = c(3, 3, 3, 3))))
res <- eig_sshopm(A)
```
