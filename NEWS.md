# tensory 0.0.1

First development version. Everything below is new.

## Build

* The compiled kernels no longer use xtensor: they read R's array storage
  directly through Rcpp and call BLAS/LAPACK themselves. The package now builds
  from a plain checkout with a C++11-or-later compiler (it previously needed
  C++20 and headers fetched by `inst/tools/vendor`), and `ttm` no longer copies
  its result a second time on the way back to R. Results are unchanged.
  Integer and logical arrays passed to a kernel are still converted to double,
  now without the "Coerced object" warning xtensor-r emitted.
* `ttm` is faster: the compiled kernel multiplies each contiguous slice of the
  tensor in place instead of first copying it into a gathered buffer (and
  scattering the result back). On a 200 x 200 x 200 tensor with J = 20 it is
  6-9x faster with OpenBLAS and 1.4-1.6x faster with reference BLAS; results
  are unchanged. It also now returns zeros, instead of dividing by zero, when
  the contracted mode has size 0.
* `mttkrp()` and `mttkrps()` are faster and use less memory. The compiled
  kernels are now two-step MTTKRPs: one BLAS call contracts the tensor, in
  place, with the Khatri-Rao product of the factors on one side of the mode,
  so neither the mode-n unfolding nor the full Khatri-Rao product is built.
  `mttkrps()` reads the tensor twice in total instead of once per mode. With
  OpenBLAS, `mttkrp` on a 150^3 tensor (R = 10) is 3.4-6.1x faster,
  `mttkrps` 4.5-24x faster depending on the tensor order, and a `cp_als()`
  sweep on 100^3 2.8x faster; with reference BLAS the gains are 1.0-1.3x,
  2-7x and 1.1x.
* `mttkrp()` in the first and last mode of high-order tensors no longer
  builds a Khatri-Rao product with `prod(dims) / I_n` rows: the kernel
  contracts a block of modes whose size is near `sqrt(prod(dims))` first.
* Creating a `Tensor` from a plain double array of the right shape no longer
  copies it. Every compiled kernel result went through that copy, which for
  a large `ttm()` cost as much as the multiplication itself.
* `gcp_opt(type = "bernoulli-logit")` evaluates `log(1 + exp(m))` without
  overflowing for large `m`, so L-BFGS-B no longer stops with "needs finite
  values of 'fn'" when a line search probes a large step.
* Less copying on the R side, found with the new `bench/alloc.R` audit:
  `fnorm()`, `innerprod()` and the full contraction in `ttt()` no longer
  build a tensor-sized temporary (BLAS `ddot` in place); `collapse()` with
  `sum` (the default) contracts with all-ones vectors instead of calling
  `sum` per cell through `apply()`; `t_scale()` scales in one pass instead
  of expanding `s` to the tensor's size; `isequal()` no longer copies both
  tensors; `ttt()` no longer copies its result; `Tensor$new(vector, dims)`
  copies at most once.
* `nvecs()` computes the leading eigenvectors of the Gram matrix
  `X_(n) X_(n)'` (as MATLAB's `nvecs` does), formed slice by slice without
  unfolding the tensor, instead of an SVD of the unfolding.
* `as.tensor()` of a `KTensor` is one matrix product with a Khatri-Rao
  product instead of a per-component `outer()` loop (about 4 tensor-sized
  allocations per rank before; now just the result).
* `hosvd()` takes each mode's leading singular vectors from the Gram matrix
  (as MATLAB's `hosvd` does) instead of `svd()` of the unfolding, which
  copied the tensor and also computed the tensor-sized right factor.
* `symmetrize()` uses a compiled single-pass kernel; the R fallback no longer
  sorts index rows one at a time with `apply()`. A 40^3 tensor took 1.7 s.
* Comparison and logical operators (`==`, `<`, `&`, `!`, ...) kept `$dims`
  but dropped the `dim` attribute of `$data`, so compiled kernels treated the
  result as a vector. They now keep it.
* `bench/kernels.R` and `bench/compare.R` time every compiled kernel (and the
  R reference paths) and check that two builds compute the same results.

## Tensor classes

* `Tensor`, an R6 class for dense multidimensional arrays, usable either
  method-style (`x$clone_tensor()$add(y)`) or through S3 generics and operators
  (`x + y`, `ttm(x, A, mode = 2)`).
* Structured companions, each with an `as.tensor()` method that materializes a
  dense `Tensor`: `Tenmat` (matricization), `KTensor` (CP/Kruskal), `TTensor`
  (Tucker), `Sptensor` and `Sptenmat` (sparse), `SymTensor` and `SymKTensor`
  (compact symmetric storage), and `SumTensor` (lazy sum of parts).
* Constructors `tensor()`, `tenrand()`, `ones()`, `zeros()`, `tendiag()`,
  `teneye()`, `tenfun()`, and `sptenrand()`.

## Operations

* Products and contractions: `ttm()`, `ttv()`, `ttt()`, `ttsv()`, `mtimes()`,
  `%*%`, `innerprod()`, `contract()`, `khatri_rao()`, `kronecker()`,
  `hadamard()`, `mttkrp()`, and `mttkrps()`.
* Shape and structure: `permute()`, `reshape()`, `squeeze()`, `unfold()`,
  `vec()`, `collapse()`, `t_scale()`, `mask()`, `fibers()`, `find()`,
  `symmetrize()`, `issymmetric()`, `fnorm()`, and `nvecs()`.
* Naming and argument semantics follow the MATLAB Tensor Toolbox where that
  reads naturally in R; divergences are documented per function (most notably,
  a full contraction returns a scalar `Tensor` with `dims = integer(0)`).
* Compiled Rcpp + BLAS kernels back `ttm`, `mttkrp`, `mttkrps`, `fibers`,
  `contract`, `mask`, and `issymmetric`. Every kernel is optional: each method
  keeps a complete R implementation and delegates only when the compiled symbol
  is present.

## Decompositions

* CP: `cp_als()`, plus the variants `cp_nmu()` (nonnegative multiplicative
  updates), `cp_apr()` (Poisson), `cp_opt()` and `cp_wopt()` (L-BFGS-B on the
  exact gradient, the latter handling missing data), `cp_arls()` (randomized
  ALS), and `cp_sym()` (symmetric CP).
* Tucker: `tucker_als()`, `tucker_sym()`, and `hosvd()`.
* Generalized CP with a loss catalog: `gcp_opt()`.
* Tensor eigenpairs: `eig_sshopm()` and `eig_geap()`, following Kolda & Mayo
  (2014).
* `KTensor` post-processing: `arrange()`, `normalize()`, `fixsigns()`,
  `score()`, `ncomponents()`, `extract()`, `redistribute()`, `tovec()`, and
  `viz()`.
* `cp_als()` accepts an `Sptensor` natively and never densifies it.

## Regression with tensor predictors

* `spgtr()` and `spgtr_cv()` fit the sparse partial generalized tensor
  regression of Sun, Peng, Qiu, Stevens, Manatunga and Guo (submitted): a
  generalized linear model whose predictor is a whole array per observation. Each mode is reduced to an envelope basis, the
  GLM is fit on the latent scores, and the coefficient array is returned in the
  original shape as a `TTensor`. An adaptively weighted row-wise L2,1 penalty
  selects whole slices of the array; `spgtr_cv()` chooses its strength by
  cross-validated deviance. Any GLM family is supported, with optional
  unpenalized nuisance covariates. See `vignette("spgtr")`.
* `tepls()` fits the tensor envelope partial least-squares regression of Zhang
  and Li (2017) for continuous responses, one or several at a time. The
  per-mode envelope dimensions are chosen automatically when `u` is not given.
  See `vignette("tepls")`.
* `pqtr()` and `pqtr_cv()` fit the partial quantile tensor regression of Sun,
  Qiu, Peng, Guo and Manatunga (2024): a chosen quantile of the outcome, rather than its
  mean, is regressed on the array through per-mode partial-least-squares
  directions. Optional unreduced covariates are supported, the reduced
  dimension is chosen by an eigenvalue-ratio rule or by cross-validated check
  loss, and the inner quantile regressions use the MM algorithm of Hunter and
  Lange (2000), so no linear-programming dependency is needed. See
  `vignette("pqtr")`.
* All three accept the predictor either as a list of equally shaped
  observations or as a single array whose last mode indexes observations.

## Data exchange

* `export_data()` and `import_data()` read and write the MATLAB Tensor Toolbox
  text format.

## Fixes

* `scale()` on an ordinary matrix no longer recurses until the C stack
  overflows. The default method called `base::scale()`, which is itself a
  generic and dispatched straight back to it.
