# Changelog

## tensory 0.0.1

First development version. Everything below is new.

### Tensor classes

- `Tensor`, an R6 class for dense multidimensional arrays, usable either
  method-style (`x$clone_tensor()$add(y)`) or through S3 generics and
  operators (`x + y`, `ttm(x, A, mode = 2)`).
- Structured companions, each with an
  [`as.tensor()`](https://www.sundayu.me/tensory/reference/as.tensor.md)
  method that materializes a dense `Tensor`: `Tenmat` (matricization),
  `KTensor` (CP/Kruskal), `TTensor` (Tucker), `Sptensor` and `Sptenmat`
  (sparse), `SymTensor` and `SymKTensor` (compact symmetric storage),
  and `SumTensor` (lazy sum of parts).
- Constructors
  [`tensor()`](https://www.sundayu.me/tensory/reference/Tensor.md),
  [`tenrand()`](https://www.sundayu.me/tensory/reference/tenrand.md),
  [`ones()`](https://www.sundayu.me/tensory/reference/ones.md),
  [`zeros()`](https://www.sundayu.me/tensory/reference/zeros.md),
  [`tendiag()`](https://www.sundayu.me/tensory/reference/tendiag.md),
  [`teneye()`](https://www.sundayu.me/tensory/reference/teneye.md),
  [`tenfun()`](https://www.sundayu.me/tensory/reference/tenfun.md), and
  [`sptenrand()`](https://www.sundayu.me/tensory/reference/sptenrand.md).

### Operations

- Products and contractions:
  [`ttm()`](https://www.sundayu.me/tensory/reference/ttm.md),
  [`ttv()`](https://www.sundayu.me/tensory/reference/ttv.md),
  [`ttt()`](https://www.sundayu.me/tensory/reference/ttt.md),
  [`ttsv()`](https://www.sundayu.me/tensory/reference/ttsv.md),
  [`mtimes()`](https://www.sundayu.me/tensory/reference/mtimes.md),
  `%*%`,
  [`innerprod()`](https://www.sundayu.me/tensory/reference/innerprod.md),
  [`contract()`](https://www.sundayu.me/tensory/reference/contract.md),
  [`khatri_rao()`](https://www.sundayu.me/tensory/reference/khatri_rao.md),
  [`kronecker()`](https://www.sundayu.me/tensory/reference/kronecker.md),
  [`hadamard()`](https://www.sundayu.me/tensory/reference/hadamard.md),
  [`mttkrp()`](https://www.sundayu.me/tensory/reference/mttkrp.md), and
  [`mttkrps()`](https://www.sundayu.me/tensory/reference/mttkrps.md).
- Shape and structure:
  [`permute()`](https://www.sundayu.me/tensory/reference/permute.md),
  [`reshape()`](https://www.sundayu.me/tensory/reference/reshape.md),
  [`squeeze()`](https://www.sundayu.me/tensory/reference/squeeze.md),
  [`unfold()`](https://www.sundayu.me/tensory/reference/unfold.md),
  [`vec()`](https://www.sundayu.me/tensory/reference/vec.md),
  [`collapse()`](https://www.sundayu.me/tensory/reference/collapse.md),
  [`t_scale()`](https://www.sundayu.me/tensory/reference/t_scale.md),
  [`mask()`](https://www.sundayu.me/tensory/reference/mask.md),
  [`fibers()`](https://www.sundayu.me/tensory/reference/fibers.md),
  [`find()`](https://www.sundayu.me/tensory/reference/find.md),
  [`symmetrize()`](https://www.sundayu.me/tensory/reference/symmetrize.md),
  [`issymmetric()`](https://www.sundayu.me/tensory/reference/issymmetric.md),
  [`fnorm()`](https://www.sundayu.me/tensory/reference/fnorm.md), and
  [`nvecs()`](https://www.sundayu.me/tensory/reference/nvecs.md).
- Naming and argument semantics follow the MATLAB Tensor Toolbox where
  that reads naturally in R; divergences are documented per function
  (most notably, a full contraction returns a scalar `Tensor` with
  `dims = integer(0)`).
- Compiled `xtensor` + BLAS kernels back `ttm`, `mttkrp`, `mttkrps`,
  `fibers`, `contract`, `mask`, and `issymmetric`. Every kernel is
  optional: each method keeps a complete R implementation and delegates
  only when the compiled symbol is present.

### Decompositions

- CP: [`cp_als()`](https://www.sundayu.me/tensory/reference/cp_als.md),
  plus the variants
  [`cp_nmu()`](https://www.sundayu.me/tensory/reference/cp_nmu.md)
  (nonnegative multiplicative updates),
  [`cp_apr()`](https://www.sundayu.me/tensory/reference/cp_apr.md)
  (Poisson),
  [`cp_opt()`](https://www.sundayu.me/tensory/reference/cp_opt.md) and
  [`cp_wopt()`](https://www.sundayu.me/tensory/reference/cp_wopt.md)
  (L-BFGS-B on the exact gradient, the latter handling missing data),
  [`cp_arls()`](https://www.sundayu.me/tensory/reference/cp_arls.md)
  (randomized ALS), and
  [`cp_sym()`](https://www.sundayu.me/tensory/reference/cp_sym.md)
  (symmetric CP).
- Tucker:
  [`tucker_als()`](https://www.sundayu.me/tensory/reference/tucker_als.md),
  [`tucker_sym()`](https://www.sundayu.me/tensory/reference/tucker_sym.md),
  and [`hosvd()`](https://www.sundayu.me/tensory/reference/hosvd.md).
- Generalized CP with a loss catalog:
  [`gcp_opt()`](https://www.sundayu.me/tensory/reference/gcp_opt.md).
- Tensor eigenpairs:
  [`eig_sshopm()`](https://www.sundayu.me/tensory/reference/eig_sshopm.md)
  and
  [`eig_geap()`](https://www.sundayu.me/tensory/reference/eig_geap.md),
  following Kolda & Mayo (2014).
- `KTensor` post-processing:
  [`arrange()`](https://www.sundayu.me/tensory/reference/arrange.md),
  [`normalize()`](https://www.sundayu.me/tensory/reference/normalize.md),
  [`fixsigns()`](https://www.sundayu.me/tensory/reference/fixsigns.md),
  [`score()`](https://www.sundayu.me/tensory/reference/score.md),
  [`ncomponents()`](https://www.sundayu.me/tensory/reference/ncomponents.md),
  [`extract()`](https://www.sundayu.me/tensory/reference/extract.md),
  [`redistribute()`](https://www.sundayu.me/tensory/reference/redistribute.md),
  [`tovec()`](https://www.sundayu.me/tensory/reference/tovec.md), and
  [`viz()`](https://www.sundayu.me/tensory/reference/viz.md).
- [`cp_als()`](https://www.sundayu.me/tensory/reference/cp_als.md)
  accepts an `Sptensor` natively and never densifies it.

### Regression with tensor predictors

- [`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md) and
  [`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md)
  fit a generalized linear model whose predictor is a whole array per
  observation. Each mode is reduced to an envelope basis, the GLM is fit
  on the latent scores, and the coefficient array is returned in the
  original shape as a `TTensor`. An adaptively weighted row-wise L2,1
  penalty selects whole slices of the array;
  [`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md)
  chooses its strength by cross-validated deviance. Any GLM family is
  supported, with optional unpenalized nuisance covariates. See
  [`vignette("spgtr")`](https://www.sundayu.me/tensory/articles/spgtr.md).
- [`tepls()`](https://www.sundayu.me/tensory/reference/tepls.md) fits
  the tensor envelope partial least-squares regression of Zhang and
  Li (2017) for continuous responses.
- Both accept the predictor either as a list of equally shaped
  observations or as a single array whose last mode indexes
  observations.

### Data exchange

- [`export_data()`](https://www.sundayu.me/tensory/reference/export_data.md)
  and
  [`import_data()`](https://www.sundayu.me/tensory/reference/import_data.md)
  read and write the MATLAB Tensor Toolbox text format.

### Fixes

- [`scale()`](https://www.sundayu.me/tensory/reference/scale.md) on an
  ordinary matrix no longer recurses until the C stack overflows. The
  default method called
  [`base::scale()`](https://rdrr.io/r/base/scale.html), which is itself
  a generic and dispatched straight back to it.
