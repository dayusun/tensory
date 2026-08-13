# Check Tensor Symmetry

Checks whether a tensor is symmetric with respect to one or more groups
of modes.

## Usage

``` r
issymmetric(x, grps = NULL, ...)

# S3 method for class 'Tensor'
issymmetric(x, grps = NULL, tol = 0, ...)
```

## Arguments

- x:

  A Tensor object.

- grps:

  A vector of modes or list of mode vectors.

- ...:

  Additional arguments passed to methods.

- tol:

  Nonnegative tolerance for approximate symmetry (`0`, the default,
  requires exact equality and uses the compiled kernel).

## Value

Logical scalar.
