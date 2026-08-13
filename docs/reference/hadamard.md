# Hadamard Product

Element-wise multiplication of matrices or list of matrices.

## Usage

``` r
hadamard(x, ...)

# S3 method for class 'matrix'
hadamard(x, y, ...)
```

## Arguments

- x:

  A matrix or list of matrices.

- ...:

  Additional arguments.

- y:

  A matrix (if x is a matrix).

## Value

A matrix representing the Hadamard product.

## Examples

``` r
A <- matrix(1:4, nrow = 2)
B <- matrix(5:8, nrow = 2)
hadamard(A, B)
#>      [,1] [,2]
#> [1,]    5   21
#> [2,]   12   32

# With a list of matrices
C <- matrix(9:12, nrow = 2)
hadamard(list(A, B, C))
#>      [,1] [,2]
#> [1,]   45  231
#> [2,]  120  384
```
