# tensory 0.0.1

First development version. Everything below is new.

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
* Compiled `xtensor` + BLAS kernels back `ttm`, `mttkrp`, `mttkrps`, `fibers`,
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

* `spgtr()` and `spgtr_cv()` fit a generalized linear model whose predictor is
  a whole array per observation. Each mode is reduced to an envelope basis, the
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
  Zhang and Zhang (2024): a chosen quantile of the outcome, rather than its
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
