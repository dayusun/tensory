# Kronecker Product

Extends base::kronecker to support list of matrices.

## Usage

``` r
kronecker(X, Y = NULL, FUN = "*", make.dimnames = FALSE, ...)
```

## Arguments

- X:

  A matrix, Tensor, or list of matrices.

- Y:

  A matrix or Tensor (optional if X is a list).

- FUN:

  The function to use (default "\*").

- make.dimnames:

  Logical.

- ...:

  Additional arguments.

## Value

The Kronecker product.

## Examples

``` r
A <- matrix(1:4, nrow = 2)
B <- matrix(5:8, nrow = 2)
kronecker(A, B)
#>      [,1] [,2] [,3] [,4]
#> [1,]    5    7   15   21
#> [2,]    6    8   18   24
#> [3,]   10   14   20   28
#> [4,]   12   16   24   32

# With a list of matrices
C <- matrix(9:12, nrow = 2)
kronecker(list(A, B, C))
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8]
#> [1,]   45   55   63   77  135  165  189  231
#> [2,]   50   60   70   84  150  180  210  252
#> [3,]   54   66   72   88  162  198  216  264
#> [4,]   60   72   80   96  180  216  240  288
#> [5,]   90  110  126  154  180  220  252  308
#> [6,]  100  120  140  168  200  240  280  336
#> [7,]  108  132  144  176  216  264  288  352
#> [8,]  120  144  160  192  240  288  320  384
```
