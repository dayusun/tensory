# Tensor Times Vector

Convenience wrapper around
[`ttm()`](https://dayusun.github.io/tensory/reference/ttm.md) for vector
contractions.

## Usage

``` r
ttv(tensor, vector, mode = NULL)
```

## Arguments

- tensor:

  A Tensor-like object.

- vector:

  A numeric vector or list of numeric vectors.

- mode:

  Integer mode or modes to contract.

## Value

A Tensor object.
