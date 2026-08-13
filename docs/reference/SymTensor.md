# R6 Class for Symmetric Tensors (SymTensor)

Stores a symmetric tensor compactly by keeping only one value per
distinct (sorted) index class, mirroring the MATLAB Tensor Toolbox
`symtensor`. An order-`m`, size-`n` symmetric tensor stores
`choose(n + m - 1, m)` values instead of `n^m`.

Compactly stores a symmetric tensor using one value per distinct index
class, mirroring the MATLAB Tensor Toolbox `symtensor`.

## Usage

``` r
symtensor(x, symmetrize = FALSE)
```

## Arguments

- x:

  A symmetric `Tensor`, or any tensor if `symmetrize = TRUE`.

- symmetrize:

  Logical; if `TRUE` the input is symmetrized first.

## Value

A `SymTensor` object.

## Public fields

- `vals`:

  Values, one per distinct sorted index class

- `m`:

  Tensor order

- `n`:

  Size of each mode

## Methods

### Public methods

- [`SymTensor$new()`](#method-SymTensor-initialize)

- [`SymTensor$dim()`](#method-SymTensor-dim)

- [`SymTensor$ndims()`](#method-SymTensor-ndims)

- [`SymTensor$print()`](#method-SymTensor-print)

- [`SymTensor$full()`](#method-SymTensor-full)

- [`SymTensor$clone()`](#method-SymTensor-clone)

------------------------------------------------------------------------

### `SymTensor$new()`

Initialize a new SymTensor from a dense symmetric tensor

#### Usage

    SymTensor$new(x = NULL)

#### Arguments

- `x`:

  A symmetric `Tensor` (all modes the same size).

#### Returns

A new SymTensor object

------------------------------------------------------------------------

### `SymTensor$dim()`

Get dimensions of the tensor

#### Usage

    SymTensor$dim()

#### Returns

Integer vector of dimensions

------------------------------------------------------------------------

### `SymTensor$ndims()`

Get number of dimensions

#### Usage

    SymTensor$ndims()

#### Returns

Integer tensor order

------------------------------------------------------------------------

### `SymTensor$print()`

Print the SymTensor object

#### Usage

    SymTensor$print(...)

#### Arguments

- `...`:

  Additional arguments

#### Returns

Invisible self

------------------------------------------------------------------------

### `SymTensor$full()`

Convert to a dense Tensor

#### Usage

    SymTensor$full()

#### Returns

A dense `Tensor`

------------------------------------------------------------------------

### `SymTensor$clone()`

The objects of this class are cloneable with this method.

#### Usage

    SymTensor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
X <- symmetrize(tensor(array(rnorm(27), dim = c(3, 3, 3))))
S <- symtensor(X)
```
