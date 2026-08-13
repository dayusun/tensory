# R6 Class for Implicit Sums of Tensors (SumTensor)

Represents a tensor that is the sum of several parts (dense tensors,
Kruskal tensors, or Tucker tensors) without materializing the sum,
mirroring the MATLAB Tensor Toolbox `sumtensor`. Operations distribute
over the parts where a structured implementation exists.

## Usage

``` r
sumtensor(...)
```

## Arguments

- ...:

  Tensor-like parts (or a single list of parts) with identical
  dimensions.

## Value

A `SumTensor` object.

## Public fields

- `parts`:

  List of tensor-like parts

## Methods

### Public methods

- [`SumTensor$new()`](#method-SumTensor-initialize)

- [`SumTensor$dim()`](#method-SumTensor-dim)

- [`SumTensor$ndims()`](#method-SumTensor-ndims)

- [`SumTensor$print()`](#method-SumTensor-print)

- [`SumTensor$full()`](#method-SumTensor-full)

- [`SumTensor$clone()`](#method-SumTensor-clone)

------------------------------------------------------------------------

### `SumTensor$new()`

Initialize a new SumTensor

#### Usage

    SumTensor$new(parts = NULL)

#### Arguments

- `parts`:

  List of tensor-like objects with identical dimensions.

#### Returns

A new SumTensor object

------------------------------------------------------------------------

### `SumTensor$dim()`

Get dimensions of the tensor

#### Usage

    SumTensor$dim()

#### Returns

Integer vector of dimensions

------------------------------------------------------------------------

### `SumTensor$ndims()`

Get number of dimensions

#### Usage

    SumTensor$ndims()

#### Returns

Integer tensor order

------------------------------------------------------------------------

### `SumTensor$print()`

Print the SumTensor object

#### Usage

    SumTensor$print(...)

#### Arguments

- `...`:

  Additional arguments

#### Returns

Invisible self

------------------------------------------------------------------------

### `SumTensor$full()`

Convert to a dense Tensor by materializing the sum

#### Usage

    SumTensor$full()

#### Returns

A dense `Tensor`

------------------------------------------------------------------------

### `SumTensor$clone()`

The objects of this class are cloneable with this method.

#### Usage

    SumTensor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
X <- tensor(array(rnorm(24), dim = c(2, 3, 4)))
K <- ktensor(1, list(matrix(1, 2), matrix(1, 3), matrix(1, 4)))
S <- sumtensor(X, K)
```
