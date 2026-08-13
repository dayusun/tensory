# Collapse Tensor

Collapses a tensor over specified dimensions using an accumulation
function (e.g., sum, mean, max).

## Usage

``` r
collapse(x, ...)

# S3 method for class 'Tensor'
collapse(x, dims, fun = sum, ...)
```

## Arguments

- x:

  A Tensor object.

- ...:

  Additional arguments passed to `fun`.

- dims:

  Integer vector specifying the dimensions to collapse along. Negative
  dimensions exclude those dimensions.

- fun:

  The function to apply (default is sum).

## Value

A collapsed Tensor.

## Examples

``` r
t <- tensor(array(1:24, dim = c(3, 4, 2)))
collapse(t, dims = 1) # Collapse the first dimension
#> <Tensor object>
#> A tensor of order 2 with dimensions: 4 x 2 
collapse(t, dims = c(1, 2), fun = mean) # Mean over first two dimensions
#> <Tensor object>
#> A tensor of order 1 with dimensions: 2 
```
