# Random Sparse Tensor

Creates a sparse tensor with approximately `nnz` uniformly random
nonzeros at uniformly random positions, mirroring the MATLAB Tensor
Toolbox `sptenrand`.

## Usage

``` r
sptenrand(dims, nnz)
```

## Arguments

- dims:

  Integer vector of dimensions.

- nnz:

  Number of nonzeros, or (if less than 1) the target density.

## Value

An `Sptensor`.

## Examples

``` r
S <- sptenrand(c(10, 10, 10), 20)
```
