# Leading Mode-n Vectors

Computes the leading left singular vectors of the mode-n unfolding.

## Usage

``` r
nvecs(x, mode, r = 1, flipsign = TRUE, ...)
```

## Arguments

- x:

  A Tensor object.

- mode:

  Mode along which to compute vectors.

- r:

  Number of vectors to return.

- flipsign:

  Logical; if `TRUE`, flip signs so the largest-magnitude entry in each
  column is positive.

- ...:

  Additional arguments passed to methods.

## Value

A matrix whose columns are the leading mode-n vectors.
