# tensory

`tensory` is an R package for tensor algebra and for regression with
tensor-valued predictors. It provides dense, sparse, symmetric and factorized
tensor classes, the CP and Tucker decomposition families, and three regression
models that relate an array-valued predictor — one array per subject — to a
scalar outcome. Naming and argument semantics track the MATLAB Tensor Toolbox.

Website: <https://www.sundayu.me/tensory/>

## Installation

```r
# install.packages("remotes")
remotes::install_github("dayusun/tensory")
```

## Tensor classes and operations

The dense class is `Tensor`, an R6 object wrapping an R array. `Tenmat` holds a
matricization, `KTensor` and `TTensor` hold CP/Kruskal and Tucker
factorizations, `Sptensor` and `Sptenmat` hold sparse subscript/value storage,
`SymTensor` and `SymKTensor` hold symmetric storage, and `SumTensor` holds a sum
of parts. Every class converts back to a dense `Tensor` with `as.tensor()`.

- Contractions and products: `ttm()`, `ttv()`, `ttt()`, `ttsv()`, `mttkrp()`,
  `mttkrps()`, `contract()`, `innerprod()`, `khatri_rao()`, `kronecker()`,
  `hadamard()`
- Reshaping and extraction: `permute()`, `reshape()`, `unfold()`, `squeeze()`,
  `fibers()`, `mask()`
- Norms, summaries and transforms: `fnorm()`, `nvecs()`, `collapse()`,
  `scale()`, `symmetrize()`

The operators `+`, `-`, `*` and `%*%` work on tensors.

```r
library(tensory)

x <- tensor(array(1:24, dim = c(2, 3, 4)))

A <- matrix(runif(6), nrow = 2)
y <- ttm(x, A, mode = 2)
z <- ttv(x, c(1, 2), mode = 1)

x_perm <- permute(x, c(3, 1, 2))
```

## Decompositions

CP fitting covers alternating least squares (`cp_als()`), nonnegative
multiplicative updates (`cp_nmu()`), Poisson CP (`cp_apr()`), direct
optimization with and without missing data (`cp_opt()`, `cp_wopt()`), randomized
ALS (`cp_arls()`) and the symmetric case (`cp_sym()`). `gcp_opt()` fits
generalized CP under a loss catalog: gaussian, poisson, poisson-log,
bernoulli-odds, bernoulli-logit, rayleigh, gamma, huber, or a custom loss.
Tucker fitting is `tucker_als()`, `tucker_sym()` and `hosvd()`. Tensor
eigenpairs come from `eig_sshopm()` and `eig_geap()`.

## Regression with a tensor predictor

`spgtr()` fits a sparse partial generalized tensor regression: a generalized
linear model — binomial, poisson or gaussian — in which the predictor for each
subject is an array rather than a vector. Non-array covariates enter through
`Z` and are left unpenalized. A row-wise sparsity penalty drops whole slices of
the predictor array, so the fit selects along each mode instead of over
individual cells. `spgtr_cv()` selects the penalty by cross-validation.

```r
fit <- spgtr(X, y, Z = Z)
```

`tepls()` fits tensor envelope partial least squares for a continuous response,
following Zhang and Li (2017).

`pqtr()` fits partial quantile tensor regression, targeting a conditional
quantile of the response rather than its mean; `pqtr_cv()` selects the reduced
dimensions by cross-validation.

## Implementation

Performance-critical kernels — `ttm`, `mttkrp`, the mode covariances and the
manifold solver — are compiled C++ built on xtensor and on whichever
BLAS/LAPACK R itself was built against. Compiling them requires a C++20
compiler. Every compiled kernel is optional: an equivalent pure-R path runs when
the package is installed without compilation, and the two paths are tested
against each other.

R dependencies: R6, Rcpp, stats, graphics, utils.

## Articles

- [Getting started with the tensor classes](https://www.sundayu.me/tensory/articles/tensory.html)
- [Tensor identities](https://www.sundayu.me/tensory/articles/identities.html)
- [Regression with `spgtr()`](https://www.sundayu.me/tensory/articles/spgtr.html)
- [`tepls()`](https://www.sundayu.me/tensory/articles/tepls.html)
- [`pqtr()`](https://www.sundayu.me/tensory/articles/pqtr.html)
- [Performance](https://www.sundayu.me/tensory/articles/performance.html) and
  [timings against rTensor](https://www.sundayu.me/tensory/articles/benchmark.html)

## License

MIT. The repository vendors third-party C++ headers under `inst/include/` that
are distributed under BSD-style licenses; see [LICENSE](LICENSE),
[LICENSE.md](LICENSE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Naming and argument semantics take inspiration from the MATLAB Tensor Toolbox,
which is distributed under a BSD 2-Clause license. This repository does not
vendor Tensor Toolbox source code; if Tensor Toolbox code or documentation is
copied in the future, its copyright and license notices must be retained.

## Citation

Sun, D., Peng, L., Qiu, Z., Stevens, J., Manatunga, A. and Guo, Y. Sparse
partial generalized tensor regression with application to neuroimaging data.
Submitted.

Sun, D., Qiu, Z., Peng, L., Guo, Y. and Manatunga, A. (2024). Partial quantile
tensor regression. *Journal of the American Statistical Association* **120**(551),
1724-1735. doi:10.1080/01621459.2024.2422129

Zhang, X. and Li, L. (2017). Tensor envelope partial least-squares regression.
*Technometrics* **59**(4), 426-436.
