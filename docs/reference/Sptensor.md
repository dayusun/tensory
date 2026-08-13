# R6 Class for Sparse Tensors (Sptensor)

Stores a tensor as a list of subscripts and nonzero values, mirroring
the MATLAB Tensor Toolbox `sptensor`. Duplicate subscripts are summed
and explicit zeros dropped on construction.

## Usage

``` r
sptensor(subs = NULL, vals = NULL, dims = NULL)
```

## Arguments

- subs:

  Integer matrix of subscripts (`nnz x ndims`), or a dense Tensor/array
  to convert.

- vals:

  Numeric vector of values, one per subscript row.

- dims:

  Integer vector of tensor dimensions.

## Value

An `Sptensor` object.

## Public fields

- `subs`:

  Integer matrix of subscripts, one row per nonzero

- `vals`:

  Numeric vector of nonzero values

- `dims`:

  Integer vector of dimensions

## Methods

### Public methods

- [`Sptensor$new()`](#method-Sptensor-initialize)

- [`Sptensor$dim()`](#method-Sptensor-dim)

- [`Sptensor$ndims()`](#method-Sptensor-ndims)

- [`Sptensor$print()`](#method-Sptensor-print)

- [`Sptensor$full()`](#method-Sptensor-full)

- [`Sptensor$clone()`](#method-Sptensor-clone)

------------------------------------------------------------------------

### `Sptensor$new()`

Initialize a new Sptensor

#### Usage

    Sptensor$new(subs = NULL, vals = NULL, dims = NULL)

#### Arguments

- `subs`:

  Integer matrix of subscripts (`nnz x ndims`).

- `vals`:

  Numeric vector of values, one per subscript row (a scalar is
  recycled).

- `dims`:

  Integer vector of tensor dimensions.

#### Returns

A new Sptensor object

------------------------------------------------------------------------

### `Sptensor$dim()`

Get dimensions of the tensor

#### Usage

    Sptensor$dim()

#### Returns

Integer vector of dimensions

------------------------------------------------------------------------

### `Sptensor$ndims()`

Get number of dimensions

#### Usage

    Sptensor$ndims()

#### Returns

Integer tensor order

------------------------------------------------------------------------

### `Sptensor$print()`

Print the Sptensor object

#### Usage

    Sptensor$print(...)

#### Arguments

- `...`:

  Additional arguments

#### Returns

Invisible self

------------------------------------------------------------------------

### `Sptensor$full()`

Convert to a dense Tensor

#### Usage

    Sptensor$full()

#### Returns

A dense `Tensor`

------------------------------------------------------------------------

### `Sptensor$clone()`

The objects of this class are cloneable with this method.

#### Usage

    Sptensor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
S <- sptensor(rbind(c(1, 1, 1), c(2, 3, 4)), c(5, 7), c(2, 3, 4))
```
