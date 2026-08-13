# Convert object to Tenmat

Convert object to Tenmat

## Usage

``` r
as.tenmat(x)
```

## Arguments

- x:

  Object to convert

## Value

A Tenmat object

## Examples

``` r
t <- tensor(array(1:24, dim = c(3, 4, 2)))
as.tenmat(t)
#> <Tenmat object>
#> A matrix corresponding to a tensor of size 3 x 4 x 2 
#> rindices = [ 1 2 3 ] (modes of tensor corresponding to rows)
#> cindices = [  ] (modes of tensor corresponding to columns)
#> data =
#>       [,1]
#>  [1,]    1
#>  [2,]    2
#>  [3,]    3
#>  [4,]    4
#>  [5,]    5
#>  [6,]    6
#>  [7,]    7
#>  [8,]    8
#>  [9,]    9
#> [10,]   10
#> [11,]   11
#> [12,]   12
#> [13,]   13
#> [14,]   14
#> [15,]   15
#> [16,]   16
#> [17,]   17
#> [18,]   18
#> [19,]   19
#> [20,]   20
#> [21,]   21
#> [22,]   22
#> [23,]   23
#> [24,]   24
```
