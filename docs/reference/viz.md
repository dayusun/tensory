# Visualize a Kruskal Tensor

Plots each factor-matrix column as a line plot in an
`ndims x ncomponents` grid, a lightweight analogue of the MATLAB Tensor
Toolbox `viz`.

## Usage

``` r
viz(x, ...)

# S3 method for class 'KTensor'
viz(x, normalize = TRUE, ...)
```

## Arguments

- x:

  A `KTensor`.

- ...:

  Additional arguments passed to methods.

- normalize:

  Logical; if `TRUE` (default) plot the normalized factors.

## Value

Invisibly, `x`.
