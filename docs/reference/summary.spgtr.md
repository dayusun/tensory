# Summarize a Tensor Regression Fit

Prints a plain-language report: how much of the array was kept in each
dimension and which slices those are, how well the model fits, and how
far the coefficient array is from zero. For yes/no outcomes it also
reports the in-sample accuracy and AUC.

## Usage

``` r
# S3 method for class 'spgtr'
summary(object, ...)
```

## Arguments

- object:

  A fit from
  [`spgtr()`](https://dayusun.github.io/tensory/reference/spgtr.md) or
  [`spgtr_cv()`](https://dayusun.github.io/tensory/reference/spgtr_cv.md).

- ...:

  Unused.

## Value

Invisibly, a list with the quantities printed: `pseudo_r2`, `deviance`,
`null_deviance`, `selected`, `bnorm`, and, for binomial fits, `accuracy`
and `auc`.

## Details

In-sample fit statistics are optimistic. For an honest estimate, hold
out subjects or read the cross-validated deviance in `fit$cv` after
[`spgtr_cv()`](https://dayusun.github.io/tensory/reference/spgtr_cv.md).

## Examples

``` r
set.seed(4)
B <- outer(c(2, rep(0, 5)), c(2, rep(0, 4)))
X <- lapply(1:100, function(i) matrix(rnorm(30), 6, 5))
eta <- vapply(X, function(xi) sum(B * xi), numeric(1))
y <- rbinom(100, 1, 1 / (1 + exp(-eta)))
summary(spgtr(X, y, u = c(1, 1)))
#> <spgtr: sparse penalized generalized tensor regression>
#> Outcome:         binomial with logit link
#> Subjects:        100 
#> Array shape:     6 x 5 
#> Directions (u):  1 1 
#> Basis:           envelope (lambda = 0) 
#> Slices kept:     6/6  5/5 
#> Deviance:        45.0162 
#> 
#> Slices used, by dimension:
#>   dim 1 (6): all
#>   dim 2 (5): all
#> 
#> Deviance explained: 66.9% (in-sample)
#> Coefficient array norm: 4.851
#> Accuracy: 0.890    AUC: 0.970  (in-sample)
```
