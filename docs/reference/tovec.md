# Kruskal Tensor to Vector

Stacks a `KTensor` into a single numeric vector, mirroring the MATLAB
Tensor Toolbox `tovec`.

## Usage

``` r
tovec(x, ...)

# S3 method for class 'KTensor'
tovec(x, lambda = TRUE, ...)
```

## Arguments

- x:

  A `KTensor`.

- ...:

  Additional arguments passed to methods.

- lambda:

  Logical; if `TRUE` (default) the weight vector is included as the
  leading block. If `FALSE`, the weights are first absorbed into mode 1.

## Value

A numeric vector of length `R + sum(dims) * R` (with weights) or
`sum(dims) * R` (without).
