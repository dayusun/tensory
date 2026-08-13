# Identity Tensor

Creates the order-`m` identity tensor `E` of size `n`, satisfying
`ttsv(E, x, -1) == x` for any vector `x` with `norm(x) == 1`. Mirrors
the MATLAB Tensor Toolbox `teneye`; as there, `m` must be even.

## Usage

``` r
teneye(m, n)
```

## Arguments

- m:

  Tensor order (even positive integer).

- n:

  Size of each mode.

## Value

A symmetric `Tensor` of order `m` and size `n`.

## Examples

``` r
E <- teneye(4, 3)
x <- rnorm(3); x <- x / sqrt(sum(x^2))
max(abs(ttsv(E, x, -1) - x)) < 1e-12
#> [1] TRUE
```
