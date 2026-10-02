# Changelog

## tensory 0.0.1

First development version. Everything below is new.

### Build

- The compiled kernels no longer use xtensor: they read R’s array
  storage directly through Rcpp and call BLAS/LAPACK themselves. The
  package now builds from a plain checkout with a C++11-or-later
  compiler (it previously needed C++20 and headers fetched by
  `inst/tools/vendor`), and `ttm` no longer copies its result a second
  time on the way back to R. Results are unchanged. Integer and logical
  arrays passed to a kernel are still converted to double, now without
  the “Coerced object” warning xtensor-r emitted.
- `ttm` is faster: the compiled kernel multiplies each contiguous slice
  of the tensor in place instead of first copying it into a gathered
  buffer (and scattering the result back). On a 200 x 200 x 200 tensor
  with J = 20 it is 6-9x faster with OpenBLAS and 1.4-1.6x faster with
  reference BLAS; results are unchanged. It also now returns zeros,
  instead of dividing by zero, when the contracted mode has size 0.
- [`mttkrp()`](https://www.sundayu.me/tensory/reference/mttkrp.md) and
  [`mttkrps()`](https://www.sundayu.me/tensory/reference/mttkrps.md) are
  faster and use less memory. The compiled kernels are now two-step
  MTTKRPs: one BLAS call contracts the tensor, in place, with the
  Khatri-Rao product of the factors on one side of the mode, so neither
  the mode-n unfolding nor the full Khatri-Rao product is built.
  [`mttkrps()`](https://www.sundayu.me/tensory/reference/mttkrps.md)
  reads the tensor twice in total instead of once per mode. With
  OpenBLAS, `mttkrp` on a 150^3 tensor (R = 10) is 3.4-6.1x faster,
  `mttkrps` 4.5-24x faster depending on the tensor order, and a
  [`cp_als()`](https://www.sundayu.me/tensory/reference/cp_als.md) sweep
  on 100^3 2.8x faster; with reference BLAS the gains are 1.0-1.3x, 2-7x
  and 1.1x.
- [`cp_als()`](https://www.sundayu.me/tensory/reference/cp_als.md) on a
  dense tensor of order 3 or more uses a dimension tree: the modes are
  split into two groups and each group’s MTTKRPs are finished from one
  partial contraction with the other group’s (fixed) factors, so a sweep
  reads the tensor twice instead of once per mode. Results are the same
  up to rounding. Over 10 sweeps at R = 10 it is 1.9-2.3x faster with
  OpenBLAS and 1.6-2.8x with reference BLAS for orders 4-6 (1.07-1.6x
  for order 3). `options(tensory.cp_dimtree = FALSE)` turns it off; a
  `dimorder` other than `1:N` or `N:1` and sparse tensors use per-mode
  [`mttkrp()`](https://www.sundayu.me/tensory/reference/mttkrp.md) as
  before.
- The vendored xtensor-r headers and `inst/tools/vendor` are removed;
  the package vendors no third-party code.
- [`mttkrp()`](https://www.sundayu.me/tensory/reference/mttkrp.md) in
  the first and last mode of high-order tensors no longer builds a
  Khatri-Rao product with `prod(dims) / I_n` rows: the kernel contracts
  a block of modes whose size is near `sqrt(prod(dims))` first.
- Creating a `Tensor` from a plain double array of the right shape no
  longer copies it. Every compiled kernel result went through that copy,
  which for a large
  [`ttm()`](https://www.sundayu.me/tensory/reference/ttm.md) cost as
  much as the multiplication itself.
- `gcp_opt(type = "bernoulli-logit")` evaluates `log(1 + exp(m))`
  without overflowing for large `m`, so L-BFGS-B no longer stops with
  “needs finite values of ‘fn’” when a line search probes a large step.
- Less copying on the R side, found with the new `bench/alloc.R` audit:
  [`fnorm()`](https://www.sundayu.me/tensory/reference/fnorm.md),
  [`innerprod()`](https://www.sundayu.me/tensory/reference/innerprod.md)
  and the full contraction in
  [`ttt()`](https://www.sundayu.me/tensory/reference/ttt.md) no longer
  build a tensor-sized temporary (BLAS `ddot` in place);
  [`collapse()`](https://www.sundayu.me/tensory/reference/collapse.md)
  with `sum` (the default) contracts with all-ones vectors instead of
  calling `sum` per cell through
  [`apply()`](https://rdrr.io/r/base/apply.html);
  [`t_scale()`](https://www.sundayu.me/tensory/reference/t_scale.md)
  scales in one pass instead of expanding `s` to the tensor’s size;
  [`isequal()`](https://www.sundayu.me/tensory/reference/isequal.md) no
  longer copies both tensors;
  [`ttt()`](https://www.sundayu.me/tensory/reference/ttt.md) no longer
  copies its result; `Tensor$new(vector, dims)` copies at most once.
- [`nvecs()`](https://www.sundayu.me/tensory/reference/nvecs.md)
  computes the leading eigenvectors of the Gram matrix `X_(n) X_(n)'`
  (as MATLAB’s `nvecs` does), formed slice by slice without unfolding
  the tensor, instead of an SVD of the unfolding.
- [`as.tensor()`](https://www.sundayu.me/tensory/reference/as.tensor.md)
  of a `KTensor` is one matrix product with a Khatri-Rao product instead
  of a per-component [`outer()`](https://rdrr.io/r/base/outer.html) loop
  (about 4 tensor-sized allocations per rank before; now just the
  result).
- [`hosvd()`](https://www.sundayu.me/tensory/reference/hosvd.md) takes
  each mode’s leading singular vectors from the Gram matrix (as MATLAB’s
  `hosvd` does) instead of [`svd()`](https://rdrr.io/r/base/svd.html) of
  the unfolding, which copied the tensor and also computed the
  tensor-sized right factor.
- [`symmetrize()`](https://www.sundayu.me/tensory/reference/symmetrize.md)
  uses a compiled single-pass kernel; the R fallback no longer sorts
  index rows one at a time with
  [`apply()`](https://rdrr.io/r/base/apply.html). A 40^3 tensor took 1.7
  s.
- Comparison and logical operators (`==`, `<`, `&`, `!`, …) kept `$dims`
  but dropped the `dim` attribute of `$data`, so compiled kernels
  treated the result as a vector. They now keep it.
- `bench/kernels.R` and `bench/compare.R` time every compiled kernel
  (and the R reference paths) and check that two builds compute the same
  results.

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
- Compiled Rcpp + BLAS kernels back `ttm`, `mttkrp`, `mttkrps`,
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
  fit the sparse partial generalized tensor regression of Sun, Peng,
  Qiu, Stevens, Manatunga and Guo (submitted): a generalized linear
  model whose predictor is a whole array per observation. Each mode is
  reduced to an envelope basis, the GLM is fit on the latent scores, and
  the coefficient array is returned in the original shape as a
  `TTensor`. An adaptively weighted row-wise L2,1 penalty selects whole
  slices of the array;
  [`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md)
  chooses its strength by cross-validated deviance. Any GLM family is
  supported, with optional unpenalized nuisance covariates. See
  [`vignette("spgtr")`](https://www.sundayu.me/tensory/articles/spgtr.md).
- [`tepls()`](https://www.sundayu.me/tensory/reference/tepls.md) fits
  the tensor envelope partial least-squares regression of Zhang and
  Li (2017) for continuous responses, one or several at a time. The
  per-mode envelope dimensions are chosen automatically when `u` is not
  given. See
  [`vignette("tepls")`](https://www.sundayu.me/tensory/articles/tepls.md).
- [`pqtr()`](https://www.sundayu.me/tensory/reference/pqtr.md) and
  [`pqtr_cv()`](https://www.sundayu.me/tensory/reference/pqtr_cv.md) fit
  the partial quantile tensor regression of Sun, Qiu, Peng, Guo and
  Manatunga (2024): a chosen quantile of the outcome, rather than its
  mean, is regressed on the array through per-mode partial-least-squares
  directions. Optional unreduced covariates are supported, the reduced
  dimension is chosen by an eigenvalue-ratio rule or by cross-validated
  check loss, and the inner quantile regressions use the MM algorithm of
  Hunter and Lange (2000), so no linear-programming dependency is
  needed. See
  [`vignette("pqtr")`](https://www.sundayu.me/tensory/articles/pqtr.md).
- All three accept the predictor either as a list of equally shaped
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
