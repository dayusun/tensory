# Khatri-Rao Product

Computes the Khatri-Rao product (column-wise Kronecker product) of two
matrices, or a list of matrices.

## Usage

``` r
khatri_rao(x, ...)

# S3 method for class 'matrix'
khatri_rao(x, y, reverse = FALSE, ...)
```

## Arguments

- x:

  A matrix or list of matrices.

- ...:

  Additional arguments.

- y:

  A matrix (if x is a matrix).

- reverse:

  Logical. If TRUE, computes y \\\odot\\ x instead of x \\\odot\\ y.

## Value

A matrix representing the Khatri-Rao product.

## Examples

``` r
A <- matrix(1:4, nrow = 2)
B <- matrix(5:8, nrow = 2)
khatri_rao(A, B)
#>      [,1] [,2]
#> [1,]    5   21
#> [2,]    6   24
#> [3,]   10   28
#> [4,]   12   32

# With a list of matrices
C <- matrix(9:12, nrow = 2)
khatri_rao(list(A, B, C))
#>      [,1] [,2]
#> [1,]   45  231
#> [2,]   50  252
#> [3,]   54  264
#> [4,]   60  288
#> [5,]   90  308
#> [6,]  100  336
#> [7,]  108  352
#> [8,]  120  384
```
