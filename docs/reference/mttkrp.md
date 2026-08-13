# Matricized Tensor Times Khatri-Rao Product

Computes the mode-n unfolding multiplied by the Khatri-Rao product of
all factor matrices except the skipped mode.

## Usage

``` r
mttkrp(x, U, mode, ...)
```

## Arguments

- x:

  A Tensor object.

- U:

  A list of factor matrices or a `KTensor`.

- mode:

  Mode to skip.

- ...:

  Additional arguments passed to methods.

## Value

A matrix of size `dim(x)[mode] x R`.
