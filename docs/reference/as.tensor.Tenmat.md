# Convert Tenmat to Tensor

Convert Tenmat to Tensor

## Usage

``` r
# S3 method for class 'Tenmat'
as.tensor(x, ...)
```

## Arguments

- x:

  A Tenmat object

- ...:

  Additional arguments (ignored)

## Value

A Tensor object

## Examples

``` r
t <- tensor(array(1:24, dim = c(3, 4, 2)))
tm <- tenmat(t, rdims = 1)
t2 <- as.tensor(tm)
```
