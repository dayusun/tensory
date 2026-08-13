# R6 Class for Symmetric Kruskal Tensors (SymKTensor)

A symmetric Kruskal tensor is a sum of symmetric rank-one terms
`lambda_r * u_r o u_r o ... o u_r` (`m` copies of the same vector),
mirroring the MATLAB Tensor Toolbox `symktensor`. It is represented by a
weight vector, a single factor matrix shared by all modes, and the
order.

## Usage

``` r
symktensor(lambda = NULL, u = NULL, m = NULL)
```

## Arguments

- lambda:

  Numeric vector of weights.

- u:

  Shared factor matrix with one column per component.

- m:

  Tensor order.

## Value

A `SymKTensor` object.

## Public fields

- `lambda`:

  Vector of weights

- `u`:

  Shared factor matrix (`n x R`)

- `m`:

  Tensor order

## Methods

### Public methods

- [`SymKTensor$new()`](#method-SymKTensor-initialize)

- [`SymKTensor$dim()`](#method-SymKTensor-dim)

- [`SymKTensor$ndims()`](#method-SymKTensor-ndims)

- [`SymKTensor$print()`](#method-SymKTensor-print)

- [`SymKTensor$as_ktensor()`](#method-SymKTensor-as_ktensor)

- [`SymKTensor$full()`](#method-SymKTensor-full)

- [`SymKTensor$clone()`](#method-SymKTensor-clone)

------------------------------------------------------------------------

### `SymKTensor$new()`

Initialize a new SymKTensor

#### Usage

    SymKTensor$new(lambda = NULL, u = NULL, m = NULL)

#### Arguments

- `lambda`:

  Numeric vector of weights.

- `u`:

  Shared factor matrix with one column per component.

- `m`:

  Tensor order (positive integer).

#### Returns

A new SymKTensor object

------------------------------------------------------------------------

### `SymKTensor$dim()`

Get dimensions of the tensor

#### Usage

    SymKTensor$dim()

#### Returns

Integer vector of dimensions (`m` copies of `nrow(u)`)

------------------------------------------------------------------------

### `SymKTensor$ndims()`

Get number of dimensions

#### Usage

    SymKTensor$ndims()

#### Returns

Integer tensor order

------------------------------------------------------------------------

### `SymKTensor$print()`

Print the SymKTensor object

#### Usage

    SymKTensor$print(...)

#### Arguments

- `...`:

  Additional arguments

#### Returns

Invisible self

------------------------------------------------------------------------

### `SymKTensor$as_ktensor()`

Convert to an ordinary Kruskal tensor

#### Usage

    SymKTensor$as_ktensor()

#### Returns

A `KTensor` with the shared factor repeated in every mode

------------------------------------------------------------------------

### `SymKTensor$full()`

Convert to a dense Tensor

#### Usage

    SymKTensor$full()

#### Returns

A dense `Tensor`

------------------------------------------------------------------------

### `SymKTensor$clone()`

The objects of this class are cloneable with this method.

#### Usage

    SymKTensor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
S <- symktensor(c(1, 2), matrix(rnorm(6), 3, 2), m = 3)
```
