# Random Dense Tensor

Creates a tensor with entries drawn independently from the uniform
distribution on `[0, 1]`, mirroring the MATLAB Tensor Toolbox `tenrand`.

## Usage

``` r
tenrand(dims)
```

## Arguments

- dims:

  Integer vector of dimensions.

## Value

A `Tensor` with uniform random entries.

## Examples

``` r
X <- tenrand(c(2, 3, 4))
```
