# tensory

`tensory` is an R package for tensor algebra and for regression with
tensor-valued predictors. It provides dense, sparse, symmetric and
factorized tensor classes, the CP and Tucker decomposition families, and
three regression models that relate an array-valued predictor — one
array per subject — to a scalar outcome. Naming and argument semantics
track the MATLAB Tensor Toolbox.

Website: <https://www.sundayu.me/tensory/>

## Installation

``` r

# install.packages("remotes")
remotes::install_github("dayusun/tensory")
```

## Tensor classes and operations

The dense class is `Tensor`, an R6 object wrapping an R array. `Tenmat`
holds a matricization, `KTensor` and `TTensor` hold CP/Kruskal and
Tucker factorizations, `Sptensor` and `Sptenmat` hold sparse
subscript/value storage, `SymTensor` and `SymKTensor` hold symmetric
storage, and `SumTensor` holds a sum of parts. Every class converts back
to a dense `Tensor` with
[`as.tensor()`](https://www.sundayu.me/tensory/reference/as.tensor.md).

- Contractions and products:
  [`ttm()`](https://www.sundayu.me/tensory/reference/ttm.md),
  [`ttv()`](https://www.sundayu.me/tensory/reference/ttv.md),
  [`ttt()`](https://www.sundayu.me/tensory/reference/ttt.md),
  [`ttsv()`](https://www.sundayu.me/tensory/reference/ttsv.md),
  [`mttkrp()`](https://www.sundayu.me/tensory/reference/mttkrp.md),
  [`mttkrps()`](https://www.sundayu.me/tensory/reference/mttkrps.md),
  [`contract()`](https://www.sundayu.me/tensory/reference/contract.md),
  [`innerprod()`](https://www.sundayu.me/tensory/reference/innerprod.md),
  [`khatri_rao()`](https://www.sundayu.me/tensory/reference/khatri_rao.md),
  [`kronecker()`](https://www.sundayu.me/tensory/reference/kronecker.md),
  [`hadamard()`](https://www.sundayu.me/tensory/reference/hadamard.md)
- Reshaping and extraction:
  [`permute()`](https://www.sundayu.me/tensory/reference/permute.md),
  [`reshape()`](https://www.sundayu.me/tensory/reference/reshape.md),
  [`unfold()`](https://www.sundayu.me/tensory/reference/unfold.md),
  [`squeeze()`](https://www.sundayu.me/tensory/reference/squeeze.md),
  [`fibers()`](https://www.sundayu.me/tensory/reference/fibers.md),
  [`mask()`](https://www.sundayu.me/tensory/reference/mask.md)
- Norms, summaries and transforms:
  [`fnorm()`](https://www.sundayu.me/tensory/reference/fnorm.md),
  [`nvecs()`](https://www.sundayu.me/tensory/reference/nvecs.md),
  [`collapse()`](https://www.sundayu.me/tensory/reference/collapse.md),
  [`scale()`](https://www.sundayu.me/tensory/reference/scale.md),
  [`symmetrize()`](https://www.sundayu.me/tensory/reference/symmetrize.md)

The operators `+`, `-`, `*` and `%*%` work on tensors.

``` r

library(tensory)

x <- tensor(array(1:24, dim = c(2, 3, 4)))

A <- matrix(runif(6), nrow = 2)
y <- ttm(x, A, mode = 2)
z <- ttv(x, c(1, 2), mode = 1)

x_perm <- permute(x, c(3, 1, 2))
```

## Decompositions

CP fitting covers alternating least squares
([`cp_als()`](https://www.sundayu.me/tensory/reference/cp_als.md)),
nonnegative multiplicative updates
([`cp_nmu()`](https://www.sundayu.me/tensory/reference/cp_nmu.md)),
Poisson CP
([`cp_apr()`](https://www.sundayu.me/tensory/reference/cp_apr.md)),
direct optimization with and without missing data
([`cp_opt()`](https://www.sundayu.me/tensory/reference/cp_opt.md),
[`cp_wopt()`](https://www.sundayu.me/tensory/reference/cp_wopt.md)),
randomized ALS
([`cp_arls()`](https://www.sundayu.me/tensory/reference/cp_arls.md)) and
the symmetric case
([`cp_sym()`](https://www.sundayu.me/tensory/reference/cp_sym.md)).
[`gcp_opt()`](https://www.sundayu.me/tensory/reference/gcp_opt.md) fits
generalized CP under a loss catalog: gaussian, poisson, poisson-log,
bernoulli-odds, bernoulli-logit, rayleigh, gamma, huber, or a custom
loss. Tucker fitting is
[`tucker_als()`](https://www.sundayu.me/tensory/reference/tucker_als.md),
[`tucker_sym()`](https://www.sundayu.me/tensory/reference/tucker_sym.md)
and [`hosvd()`](https://www.sundayu.me/tensory/reference/hosvd.md).
Tensor eigenpairs come from
[`eig_sshopm()`](https://www.sundayu.me/tensory/reference/eig_sshopm.md)
and
[`eig_geap()`](https://www.sundayu.me/tensory/reference/eig_geap.md).

## Regression with a tensor predictor

[`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md) fits a
sparse partial generalized tensor regression: a generalized linear model
— binomial, poisson or gaussian — in which the predictor for each
subject is an array rather than a vector. Non-array covariates enter
through `Z` and are left unpenalized. A row-wise sparsity penalty drops
whole slices of the predictor array, so the fit selects along each mode
instead of over individual cells.
[`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md)
selects the penalty by cross-validation.

``` r

fit <- spgtr(X, y, Z = Z)
```

[`tepls()`](https://www.sundayu.me/tensory/reference/tepls.md) fits
tensor envelope partial least squares for a continuous response,
following Zhang and Li (2017).

[`pqtr()`](https://www.sundayu.me/tensory/reference/pqtr.md) fits
partial quantile tensor regression, targeting a conditional quantile of
the response rather than its mean;
[`pqtr_cv()`](https://www.sundayu.me/tensory/reference/pqtr_cv.md)
selects the reduced dimensions by cross-validation.

## Implementation

Performance-critical kernels — `ttm`, `mttkrp`, the mode covariances and
the manifold solver — are compiled C++ built on xtensor and on whichever
BLAS/LAPACK R itself was built against. Compiling them requires a C++20
compiler. Every compiled kernel is optional: an equivalent pure-R path
runs when the package is installed without compilation, and the two
paths are tested against each other.

R dependencies: R6, Rcpp, stats, graphics, utils.

## Articles

- [Getting started with the tensor
  classes](https://www.sundayu.me/tensory/articles/tensory.html)
- [Tensor
  identities](https://www.sundayu.me/tensory/articles/identities.html)
- [Regression with
  `spgtr()`](https://www.sundayu.me/tensory/articles/spgtr.html)
- [`tepls()`](https://www.sundayu.me/tensory/articles/tepls.html)
- [`pqtr()`](https://www.sundayu.me/tensory/articles/pqtr.html)
- [Performance](https://www.sundayu.me/tensory/articles/performance.html)
  and [timings against
  rTensor](https://www.sundayu.me/tensory/articles/benchmark.html)

## License

MIT. The repository vendors third-party C++ headers under
`inst/include/` that are distributed under BSD-style licenses; see
[LICENSE](https://www.sundayu.me/tensory/LICENSE),
[LICENSE.md](https://www.sundayu.me/tensory/LICENSE.md) and
[THIRD_PARTY_NOTICES.md](https://www.sundayu.me/tensory/THIRD_PARTY_NOTICES.md).

Naming and argument semantics take inspiration from the MATLAB Tensor
Toolbox, which is distributed under a BSD 2-Clause license. This
repository does not vendor Tensor Toolbox source code; if Tensor Toolbox
code or documentation is copied in the future, its copyright and license
notices must be retained.

## Citation

Sun, D., Peng, L., Qiu, Z., Stevens, J., Manatunga, A. and Guo, Y.
Sparse partial generalized tensor regression with application to
neuroimaging data. Submitted.

Sun, D., Qiu, Z., Peng, L., Guo, Y. and Manatunga, A. (2024). Partial
quantile tensor regression. *Journal of the American Statistical
Association* **120**(551), 1724-1735.
<doi:10.1080/01621459.2024.2422129>

Zhang, X. and Li, L. (2017). Tensor envelope partial least-squares
regression. *Technometrics* **59**(4), 426-436.
