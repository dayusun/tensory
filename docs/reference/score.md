# Score the Similarity of Two Kruskal Tensors

Computes the factor match score between two `KTensor` objects with the
same number of components, mirroring the MATLAB Tensor Toolbox `score`.
Each candidate component pairing is scored by the product of the
absolute cosine similarities of the matched factor columns, optionally
discounted by the relative difference of the component weights;
components are matched greedily.

## Usage

``` r
score(x, ...)

# S3 method for class 'KTensor'
score(x, y, lambda_penalty = TRUE, greedy = TRUE, ...)
```

## Arguments

- x:

  A `KTensor` (the estimate).

- ...:

  Additional arguments passed to methods.

- y:

  A `KTensor` with the same dimensions and number of components (the
  reference).

- lambda_penalty:

  Logical; if `TRUE` (default) each pair score is multiplied by
  `1 - |la - lb| / max(la, lb)` computed from the normalized weights.

- greedy:

  Logical; must be `TRUE` (greedy matching is the only implemented
  strategy, as it is the MATLAB default).

## Value

A list with elements `score` (mean of the matched component scores),
`perm` (the permutation of `y`'s components matched to `x`), and `x` (a
normalized copy of `x` arranged to match `y`).
