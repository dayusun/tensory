# R6 Class for Kruskal Tensors (KTensor)

A class representing Kruskal tensors (KTensor), which are the sum of
outer products of vectors. This matches the behavior of MATLAB's Tensor
Toolbox ktensor.

A Kruskal tensor is represented by a set of factor matrices and a vector
of weights.

## Usage

``` r
ktensor(lambda = NULL, U = NULL)
```

## Arguments

- lambda:

  A numeric vector of weights, or list of matrices U if lambda is
  omitted.

- U:

  A list of factor matrices. Let NULL if passing list as `lambda`.

## Value

A KTensor object

## Public fields

- `lambda`:

  Vector of weights

- `U`:

  List of factor matrices

## Methods

### Public methods

- [`KTensor$new()`](#method-KTensor-initialize)

- [`KTensor$dim()`](#method-KTensor-dim)

- [`KTensor$ndims()`](#method-KTensor-ndims)

- [`KTensor$print()`](#method-KTensor-print)

- [`KTensor$full()`](#method-KTensor-full)

- [`KTensor$clone()`](#method-KTensor-clone)

------------------------------------------------------------------------

### `KTensor$new()`

Initialize a new KTensor

#### Usage

    KTensor$new(lambda = NULL, U = NULL)

#### Arguments

- `lambda`:

  A numeric vector of weights. If NULL, defaults to a vector of 1s.

- `U`:

  A list of factor matrices

#### Returns

A new KTensor object

------------------------------------------------------------------------

### `KTensor$dim()`

Get dimensions of the tensor

#### Usage

    KTensor$dim()

#### Returns

Integer vector of dimensions

------------------------------------------------------------------------

### `KTensor$ndims()`

Get number of dimensions

#### Usage

    KTensor$ndims()

#### Returns

Integer number of dimensions

------------------------------------------------------------------------

### `KTensor$print()`

Print the KTensor object

#### Usage

    KTensor$print(...)

#### Arguments

- `...`:

  Additional arguments

#### Returns

Invisible self

------------------------------------------------------------------------

### `KTensor$full()`

Convert Kruskal tensor to dense Tensor

#### Usage

    KTensor$full()

#### Returns

A new Tensor object representing the full dense tensor

------------------------------------------------------------------------

### `KTensor$clone()`

The objects of this class are cloneable with this method.

#### Usage

    KTensor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
