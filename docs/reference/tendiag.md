# Diagonal Tensor

Creates a tensor with the vector `v` on its superdiagonal and zeros
elsewhere, mirroring the MATLAB Tensor Toolbox `tendiag`.

## Usage

``` r
tendiag(v, dims = NULL)
```

## Arguments

- v:

  Numeric vector of diagonal values.

- dims:

  Optional integer vector of dimensions. Defaults to a square
  matrix-shaped tensor `c(length(v), length(v))`. Each entry must be at
  least `length(v)`.

## Value

A `Tensor` with `v` on the superdiagonal.

## Examples

``` r
D <- tendiag(1:3, c(3, 3, 3))
```
