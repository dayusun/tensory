# Frobenius Norm

Calculates the Frobenius norm of a tensor.

## Usage

``` r
fnorm(x, ...)
```

## Arguments

- x:

  A tensor or matrix.

- ...:

  Additional arguments.

## Value

A scalar value representing the Frobenius norm.

## Examples

``` r
t <- tensor(array(1:24, dim = c(3, 4, 2)))
fnorm(t)
#> [1] 70
```
