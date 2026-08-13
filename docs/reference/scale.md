# Tensor Scaling

Alias for
[`t_scale()`](https://www.sundayu.me/tensory/reference/t_scale.md) with
Tensor Toolbox naming.

## Usage

``` r
scale(x, ...)

# S3 method for class 'Tensor'
scale(x, s, dims, ...)
```

## Arguments

- x:

  A Tensor object.

- ...:

  For tensors, pass `s =` and `dims =`. For non-tensors, arguments are
  forwarded to [`base::scale()`](https://rdrr.io/r/base/scale.html).

- s:

  Scaling tensor, matrix, or vector.

- dims:

  Modes of `x` that `s` scales along.

## Value

A scaled Tensor.
