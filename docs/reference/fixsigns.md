# Fix Sign Ambiguity of a Kruskal Tensor

Flips the signs of factor-matrix columns so that the largest-magnitude
entry of each column is positive, compensating in the weights so the
represented tensor is unchanged. Mirrors the MATLAB Tensor Toolbox
`fixsigns`.

## Usage

``` r
fixsigns(x, ...)
```

## Arguments

- x:

  A `KTensor`.

- ...:

  Additional arguments passed to methods.

## Value

A sign-fixed `KTensor`.
