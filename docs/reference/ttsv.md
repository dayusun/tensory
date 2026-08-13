# Tensor Times Same Vector

Multiplies a tensor by the same vector in all contracted modes.

## Usage

``` r
ttsv(x, v, n = 0, ...)
```

## Arguments

- x:

  A Tensor object with equal mode sizes.

- v:

  A vector whose length matches each contracted mode.

- n:

  Non-positive integer controlling how many leading modes remain.

- ...:

  Additional arguments passed to methods.

## Value

A scalar, vector, matrix, or Tensor depending on the number of remaining
modes.
