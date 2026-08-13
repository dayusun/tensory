# R6 Class for Sparse Matricized Tensors (Sptenmat)

A sparse matricization of an `Sptensor`: the nonzeros of the tensor
mapped to (row, column) coordinates of the unfolding defined by
`rdims`/`cdims`, mirroring the MATLAB Tensor Toolbox `sptenmat`.

## Usage

``` r
sptenmat(x, rdims = NULL, cdims = NULL)
```

## Arguments

- x:

  An `Sptensor`.

- rdims:

  Modes mapped to matrix rows.

- cdims:

  Modes mapped to matrix columns.

## Value

An `Sptenmat` object.

## Public fields

- `subs`:

  Two-column integer matrix of (row, col) coordinates

- `vals`:

  Numeric vector of nonzero values

- `rdims`:

  Tensor modes mapped to matrix rows

- `cdims`:

  Tensor modes mapped to matrix columns

- `tsize`:

  Dimensions of the original tensor

## Methods

### Public methods

- [`Sptenmat$new()`](#method-Sptenmat-initialize)

- [`Sptenmat$dim()`](#method-Sptenmat-dim)

- [`Sptenmat$print()`](#method-Sptenmat-print)

- [`Sptenmat$as_matrix()`](#method-Sptenmat-as_matrix)

- [`Sptenmat$clone()`](#method-Sptenmat-clone)

------------------------------------------------------------------------

### `Sptenmat$new()`

Initialize a new Sptenmat from a sparse tensor

#### Usage

    Sptenmat$new(x = NULL, rdims = NULL, cdims = NULL)

#### Arguments

- `x`:

  An `Sptensor`.

- `rdims`:

  Modes mapped to rows.

- `cdims`:

  Modes mapped to columns (defaults to the remaining modes in ascending
  order).

#### Returns

A new Sptenmat object

------------------------------------------------------------------------

### `Sptenmat$dim()`

Matrix dimensions of the unfolding

#### Usage

    Sptenmat$dim()

#### Returns

Integer vector `c(nrow, ncol)`

------------------------------------------------------------------------

### `Sptenmat$print()`

Print the Sptenmat object

#### Usage

    Sptenmat$print(...)

#### Arguments

- `...`:

  Additional arguments

#### Returns

Invisible self

------------------------------------------------------------------------

### `Sptenmat$as_matrix()`

Convert to a dense matrix

#### Usage

    Sptenmat$as_matrix()

#### Returns

A dense matrix of the unfolding

------------------------------------------------------------------------

### `Sptenmat$clone()`

The objects of this class are cloneable with this method.

#### Usage

    Sptenmat$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
S <- sptenrand(c(4, 3, 2), 5)
A <- sptenmat(S, rdims = 1)
```
