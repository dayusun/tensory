# Extract Components of a Kruskal Tensor

Returns a new `KTensor` containing only the selected rank-one
components, mirroring the MATLAB Tensor Toolbox `extract`.

## Usage

``` r
extract(x, idx, ...)
```

## Arguments

- x:

  A `KTensor`.

- idx:

  Integer vector of component indices to keep.

- ...:

  Additional arguments passed to methods.

## Value

A `KTensor` with `length(idx)` components.
