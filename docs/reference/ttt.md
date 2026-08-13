# Tensor Times Tensor (ttt) Operation

Computes the generalized product (outer, inner, or contracted) of two
tensors.

## Usage

``` r
ttt(tensorA, tensorB, dimsA = NULL, dimsB = dimsA)
```

## Arguments

- tensorA:

  A Tensor object

- tensorB:

  A Tensor object

- dimsA:

  Subscripts of the dimensions in `tensorA` to contract over.

- dimsB:

  Subscripts of the dimensions in `tensorB` to contract over. Defaults
  to `dimsA`.

## Value

A new Tensor object representing the product.

## Details

- If `dimsA` and `dimsB` are NULL, computes the outer product.

- If `dimsA` and `dimsB` are provided, computes the contracted product
  along the specified dimensions. The sizes of the corresponding
  dimensions must match.

## Examples

``` r
t1 <- tensor(array(1:24, dim = c(4, 3, 2)))
t2 <- tensor(array(1:12, dim = c(3, 2, 2)))

# Outer product
result_outer <- ttt(t1, t2)

# Contract over dimensions
result_contracted <- ttt(t1, t2, dimsA = c(2, 3), dimsB = c(1, 2))
```
