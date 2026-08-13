# Scale Tensor

Scales a tensor along specified dimensions. Equivalent to broadcasting a
scaling vector or tensor for multiplication.

## Usage

``` r
t_scale(x, ...)

# S3 method for class 'Tensor'
t_scale(x, s, dims, ...)
```

## Arguments

- x:

  A Tensor.

- ...:

  Additional arguments.

- s:

  A vector or Tensor containing scaling factors.

- dims:

  Dimensions to scale.

## Value

A scaled Tensor.

## Examples

``` r
t <- tensor(array(1:24, dim = c(3, 4, 2)))
s <- c(10, 100, 1000)
t_scale(t, s, dims = 1) # Scale along the first dimension
#> <Tensor object>
#> A tensor of order 3 with dimensions: 3 x 4 x 2 
```
