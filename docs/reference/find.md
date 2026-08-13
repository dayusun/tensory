# Find Nonzero Entries

Returns subscripts of nonzero entries and optionally their values.

## Usage

``` r
find(x, ...)

# S3 method for class 'Tensor'
find(x, values = FALSE, ...)
```

## Arguments

- x:

  A tensor-like object.

- ...:

  Additional arguments passed to methods.

- values:

  Logical; if `TRUE`, include values in the result.

## Value

If `values = FALSE`, a matrix of subscripts. Otherwise a list with
`subs` and `vals`.
