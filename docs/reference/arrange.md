# Arrange the Components of a Kruskal Tensor

Normalizes the columns of each factor matrix and sorts the components by
weight in descending order, mirroring the MATLAB Tensor Toolbox
`arrange`. When `perm` is given, the components are permuted without
normalization.

## Usage

``` r
arrange(x, ...)

# S3 method for class 'KTensor'
arrange(x, perm = NULL, ...)
```

## Arguments

- x:

  A `KTensor`.

- ...:

  Additional arguments passed to methods.

- perm:

  Optional explicit permutation of `1:ncomponents(x)`.

## Value

A rearranged `KTensor`.
