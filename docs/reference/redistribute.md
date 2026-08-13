# Redistribute Kruskal Weights into a Mode

Absorbs the lambda weights into the factor matrix of the given mode and
resets the weights to one, mirroring the MATLAB Tensor Toolbox
`redistribute`.

## Usage

``` r
redistribute(x, mode, ...)
```

## Arguments

- x:

  A `KTensor`.

- mode:

  Mode that absorbs the weights.

- ...:

  Additional arguments passed to methods.

## Value

A `KTensor` with unit weights.
