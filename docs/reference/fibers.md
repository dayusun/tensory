# Extract Tensor Fibers

Extracts mode-k fibers specified by a matrix of sample indices.

## Usage

``` r
fibers(x, mode, midx, ...)
```

## Arguments

- x:

  A Tensor object.

- mode:

  Mode of the fibers to extract.

- midx:

  Matrix with one row per requested fiber and `ndims(x) - 1` columns
  containing the fixed indices for the other modes.

- ...:

  Additional arguments passed to methods.

## Value

A matrix of size `dim(x)[mode] x nrow(midx)`.
