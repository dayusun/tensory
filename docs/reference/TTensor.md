# R6 Class for Tucker Tensors (TTensor)

A class representing Tucker tensors (TTensor). A Tucker tensor is a core
tensor multiplied by a matrix along each mode.

## Usage

``` r
ttensor(core = NULL, U = NULL)
```

## Arguments

- core:

  A Tensor object representing the core

- U:

  A list of factor matrices

## Value

A TTensor object

## Public fields

- `core`:

  The core Tensor object

- `U`:

  List of factor matrices

## Methods

### Public methods

- [`TTensor$new()`](#method-TTensor-initialize)

- [`TTensor$dim()`](#method-TTensor-dim)

- [`TTensor$ndims()`](#method-TTensor-ndims)

- [`TTensor$print()`](#method-TTensor-print)

- [`TTensor$full()`](#method-TTensor-full)

- [`TTensor$clone()`](#method-TTensor-clone)

------------------------------------------------------------------------

### `TTensor$new()`

Initialize a new TTensor

#### Usage

    TTensor$new(core = NULL, U = NULL)

#### Arguments

- `core`:

  A Tensor object representing the core

- `U`:

  A list of factor matrices

#### Returns

A new TTensor object

------------------------------------------------------------------------

### `TTensor$dim()`

Get dimensions of the tensor

#### Usage

    TTensor$dim()

#### Returns

Integer vector of dimensions

------------------------------------------------------------------------

### `TTensor$ndims()`

Get number of dimensions

#### Usage

    TTensor$ndims()

#### Returns

Integer number of dimensions

------------------------------------------------------------------------

### `TTensor$print()`

Print the TTensor object

#### Usage

    TTensor$print(...)

#### Arguments

- `...`:

  Additional arguments

#### Returns

Invisible self

------------------------------------------------------------------------

### `TTensor$full()`

Convert Tucker tensor to dense Tensor

#### Usage

    TTensor$full()

#### Returns

A new Tensor object representing the full dense tensor

------------------------------------------------------------------------

### `TTensor$clone()`

The objects of this class are cloneable with this method.

#### Usage

    TTensor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
