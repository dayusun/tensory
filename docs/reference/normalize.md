# Normalize a Kruskal Tensor

Normalizes the columns of every factor matrix, absorbing the magnitudes
into the weight vector `lambda`. Mirrors the MATLAB Tensor Toolbox
`normalize(K, N)` semantics.

## Usage

``` r
normalize(x, ...)

# S3 method for class 'KTensor'
normalize(x, mode = NULL, sort = FALSE, normtype = 2, ...)
```

## Arguments

- x:

  A `KTensor`.

- ...:

  Additional arguments passed to methods.

- mode:

  `NULL` (default) keeps all weight in `lambda`; a mode number absorbs
  the weights into that factor matrix; `0` distributes the weights
  evenly across all modes (each factor absorbs `lambda^(1/N)`).

- sort:

  Logical; if `TRUE`, sort the components by weight, descending. Only
  allowed when the weights remain in `lambda` (i.e. `mode` is `NULL`).

- normtype:

  Norm to use for the factor columns (passed as the `p` in the vector
  p-norm; default `2`).

## Value

A normalized `KTensor`.
