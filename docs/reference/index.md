# Package index

## Package

- [`tensory-package`](https://www.sundayu.me/tensory/reference/tensory-package.md)
  [`tensory`](https://www.sundayu.me/tensory/reference/tensory-package.md)
  : tensory: Tensory - Modern Tensor Operations for R

## Tensor classes and constructors

The dense `Tensor` object, its matricized and decomposed relatives, and
the constructors that build them.

- [`tensor()`](https://www.sundayu.me/tensory/reference/Tensor.md) : R6
  Tensor Class
- [`tenrand()`](https://www.sundayu.me/tensory/reference/tenrand.md) :
  Random Dense Tensor
- [`ones()`](https://www.sundayu.me/tensory/reference/ones.md) : Create
  a tensor of ones
- [`zeros()`](https://www.sundayu.me/tensory/reference/zeros.md) :
  Create a tensor of zeros
- [`tendiag()`](https://www.sundayu.me/tensory/reference/tendiag.md) :
  Diagonal Tensor
- [`teneye()`](https://www.sundayu.me/tensory/reference/teneye.md) :
  Identity Tensor
- [`tenfun()`](https://www.sundayu.me/tensory/reference/tenfun.md) :
  Apply Elementwise Function to Tensor Arguments
- [`tenmat()`](https://www.sundayu.me/tensory/reference/Tenmat.md) : R6
  Tenmat Class
- [`as.tenmat()`](https://www.sundayu.me/tensory/reference/as.tenmat.md)
  : Convert object to Tenmat
- [`ktensor()`](https://www.sundayu.me/tensory/reference/KTensor.md) :
  R6 Class for Kruskal Tensors (KTensor)
- [`ttensor()`](https://www.sundayu.me/tensory/reference/TTensor.md) :
  R6 Class for Tucker Tensors (TTensor)
- [`sptensor()`](https://www.sundayu.me/tensory/reference/Sptensor.md) :
  R6 Class for Sparse Tensors (Sptensor)
- [`sptenmat()`](https://www.sundayu.me/tensory/reference/Sptenmat.md) :
  R6 Class for Sparse Matricized Tensors (Sptenmat)
- [`sptenrand()`](https://www.sundayu.me/tensory/reference/sptenrand.md)
  : Random Sparse Tensor
- [`symtensor()`](https://www.sundayu.me/tensory/reference/SymTensor.md)
  : R6 Class for Symmetric Tensors (SymTensor)
- [`symktensor()`](https://www.sundayu.me/tensory/reference/SymKTensor.md)
  : R6 Class for Symmetric Kruskal Tensors (SymKTensor)
- [`sumtensor()`](https://www.sundayu.me/tensory/reference/SumTensor.md)
  : R6 Class for Implicit Sums of Tensors (SumTensor)
- [`as.tensor()`](https://www.sundayu.me/tensory/reference/as.tensor.md)
  : Convert object to Tensor
- [`as.tensor(`*`<KTensor>`*`)`](https://www.sundayu.me/tensory/reference/as.tensor.KTensor.md)
  : S3 function to convert KTensor to full Tensor
- [`as.tensor(`*`<TTensor>`*`)`](https://www.sundayu.me/tensory/reference/as.tensor.TTensor.md)
  : S3 function to convert TTensor to full Tensor
- [`as.tensor(`*`<Tenmat>`*`)`](https://www.sundayu.me/tensory/reference/as.tensor.Tenmat.md)
  : Convert Tenmat to Tensor

## Shape, indexing, and predicates

Reshaping, unfolding, conversion, and structural queries.

- [`permute()`](https://www.sundayu.me/tensory/reference/permute.md) :
  Permute Tensor Dimensions
- [`reshape()`](https://www.sundayu.me/tensory/reference/reshape.md) :
  Reshape Tensor
- [`squeeze()`](https://www.sundayu.me/tensory/reference/squeeze.md) :
  Squeeze Tensor
- [`unfold()`](https://www.sundayu.me/tensory/reference/unfold.md) :
  Unfold Tensor
- [`vec()`](https://www.sundayu.me/tensory/reference/vec.md) : Vectorize
  Tensor
- [`find()`](https://www.sundayu.me/tensory/reference/find.md) : Find
  Nonzero Entries
- [`nnz()`](https://www.sundayu.me/tensory/reference/nnz.md) : Number of
  Nonzeros
- [`full()`](https://www.sundayu.me/tensory/reference/full.md) : Dense
  Array Representation
- [`isequal()`](https://www.sundayu.me/tensory/reference/isequal.md) :
  Equality Test for Tensors
- [`isscalar()`](https://www.sundayu.me/tensory/reference/isscalar.md) :
  Scalar Tensor Predicate
- [`issymmetric()`](https://www.sundayu.me/tensory/reference/issymmetric.md)
  : Check Tensor Symmetry
- [`symmetrize()`](https://www.sundayu.me/tensory/reference/symmetrize.md)
  : Symmetrize Tensor
- [`transpose()`](https://www.sundayu.me/tensory/reference/transpose.md)
  : Transpose Tensor
- [`double.Tensor()`](https://www.sundayu.me/tensory/reference/double.Tensor.md)
  : MATLAB-Style Double Conversion
- [`double.Tenmat()`](https://www.sundayu.me/tensory/reference/double.Tenmat.md)
  : Convert Tenmat to standard R double array (alias for matrix)
- [`as.double(`*`<Tenmat>`*`)`](https://www.sundayu.me/tensory/reference/as.double.Tenmat.md)
  : Convert Tenmat to standard R Matrix using generic type conversion
- [`as.matrix(`*`<Tenmat>`*`)`](https://www.sundayu.me/tensory/reference/as.matrix.Tenmat.md)
  : Convert Tenmat to standard R Matrix
- [`as.vector(`*`<Tenmat>`*`)`](https://www.sundayu.me/tensory/reference/as.vector.Tenmat.md)
  : Convert Tenmat to standard R vector
- [`head(`*`<Tensor>`*`)`](https://www.sundayu.me/tensory/reference/head.Tensor.md)
  : S3 head method for Tensor
- [`tail(`*`<Tensor>`*`)`](https://www.sundayu.me/tensory/reference/tail.Tensor.md)
  : S3 tail method for Tensor
- [`show.Tensor()`](https://www.sundayu.me/tensory/reference/show.Tensor.md)
  : S3 show method for Tensor
- [`print(`*`<Tensor>`*`)`](https://www.sundayu.me/tensory/reference/print.Tensor.md)
  : S3 print method for Tensor
- [`print(`*`<KTensor>`*`)`](https://www.sundayu.me/tensory/reference/print.KTensor.md)
  : S3 print method for KTensor
- [`print(`*`<TTensor>`*`)`](https://www.sundayu.me/tensory/reference/print.TTensor.md)
  : S3 print method for TTensor
- [`Math(`*`<Tensor>`*`)`](https://www.sundayu.me/tensory/reference/Math.Tensor.md)
  : S3 Math group generic for Tensor
- [`Summary(`*`<Tensor>`*`)`](https://www.sundayu.me/tensory/reference/Summary.Tensor.md)
  : S3 Summary group generic for Tensor

## Products, contractions, and reductions

The computational core: tensor-times-matrix/vector/tensor products,
matricized products, norms, and reductions.

- [`ttm()`](https://www.sundayu.me/tensory/reference/ttm.md) : Tensor
  Times Matrix/Vector (ttm) Operation
- [`ttv()`](https://www.sundayu.me/tensory/reference/ttv.md) : Tensor
  Times Vector
- [`ttt()`](https://www.sundayu.me/tensory/reference/ttt.md) : Tensor
  Times Tensor (ttt) Operation
- [`ttsv()`](https://www.sundayu.me/tensory/reference/ttsv.md) : Tensor
  Times Same Vector
- [`mtimes()`](https://www.sundayu.me/tensory/reference/mtimes.md) :
  Matrix Multiplication Alias
- [`` `%*%` ``](https://www.sundayu.me/tensory/reference/grapes-times-grapes.md)
  : S3 Matrix Multiplication Generic
- [`innerprod()`](https://www.sundayu.me/tensory/reference/innerprod.md)
  : Inner Product
- [`contract()`](https://www.sundayu.me/tensory/reference/contract.md) :
  Contract Tensor Dimensions
- [`khatri_rao()`](https://www.sundayu.me/tensory/reference/khatri_rao.md)
  : Khatri-Rao Product
- [`kronecker()`](https://www.sundayu.me/tensory/reference/kronecker.md)
  : Kronecker Product
- [`hadamard()`](https://www.sundayu.me/tensory/reference/hadamard.md) :
  Hadamard Product
- [`mttkrp()`](https://www.sundayu.me/tensory/reference/mttkrp.md) :
  Matricized Tensor Times Khatri-Rao Product
- [`mttkrps()`](https://www.sundayu.me/tensory/reference/mttkrps.md) :
  Sequence of MTTKRP Calculations
- [`fibers()`](https://www.sundayu.me/tensory/reference/fibers.md) :
  Extract Tensor Fibers
- [`mask()`](https://www.sundayu.me/tensory/reference/mask.md) : Mask
  Tensor Values
- [`collapse()`](https://www.sundayu.me/tensory/reference/collapse.md) :
  Collapse Tensor
- [`scale()`](https://www.sundayu.me/tensory/reference/scale.md) :
  Tensor Scaling
- [`t_scale()`](https://www.sundayu.me/tensory/reference/t_scale.md) :
  Scale Tensor
- [`fnorm()`](https://www.sundayu.me/tensory/reference/fnorm.md) :
  Frobenius Norm
- [`nvecs()`](https://www.sundayu.me/tensory/reference/nvecs.md) :
  Leading Mode-n Vectors

## Decompositions

CP, Tucker, generalized CP, and tensor eigenpairs.

- [`cp_als()`](https://www.sundayu.me/tensory/reference/cp_als.md) : CP
  Alternating Least Squares Decomposition
- [`cp_nmu()`](https://www.sundayu.me/tensory/reference/cp_nmu.md) :
  Nonnegative CP Decomposition via Multiplicative Updates
- [`cp_apr()`](https://www.sundayu.me/tensory/reference/cp_apr.md) :
  Poisson CP Decomposition (CP-APR) via Multiplicative Updates
- [`cp_opt()`](https://www.sundayu.me/tensory/reference/cp_opt.md) : CP
  Decomposition via Direct Optimization
- [`cp_wopt()`](https://www.sundayu.me/tensory/reference/cp_wopt.md) :
  Weighted CP Decomposition via Direct Optimization
- [`cp_arls()`](https://www.sundayu.me/tensory/reference/cp_arls.md) :
  CP Decomposition via Randomized (Sampled) ALS
- [`cp_sym()`](https://www.sundayu.me/tensory/reference/cp_sym.md) :
  Symmetric CP Decomposition via Direct Optimization
- [`gcp_opt()`](https://www.sundayu.me/tensory/reference/gcp_opt.md) :
  Generalized CP Decomposition
- [`tucker_als()`](https://www.sundayu.me/tensory/reference/tucker_als.md)
  : Tucker Alternating Least Squares (HOOI)
- [`tucker_sym()`](https://www.sundayu.me/tensory/reference/tucker_sym.md)
  : Symmetric Tucker Decomposition
- [`hosvd()`](https://www.sundayu.me/tensory/reference/hosvd.md) :
  Higher-Order Singular Value Decomposition
- [`eig_sshopm()`](https://www.sundayu.me/tensory/reference/eig_sshopm.md)
  : Shifted Symmetric Higher-Order Power Method (SS-HOPM)
- [`eig_geap()`](https://www.sundayu.me/tensory/reference/eig_geap.md) :
  Generalized Eigenproblem Adaptive Power Method (GEAP)

## Regression with tensor predictors

Supervised models where each subject’s predictor is a whole array:
sparse partial generalized tensor regression, tensor envelope PLS, and
partial quantile tensor regression.

- [`spgtr()`](https://www.sundayu.me/tensory/reference/spgtr.md)
  [`coef(`*`<spgtr>`*`)`](https://www.sundayu.me/tensory/reference/spgtr.md)
  [`print(`*`<spgtr>`*`)`](https://www.sundayu.me/tensory/reference/spgtr.md)
  : Sparse Partial Generalized Tensor Regression (SPGTR)
- [`spgtr_cv()`](https://www.sundayu.me/tensory/reference/spgtr_cv.md) :
  Choose the Sparsity of a Tensor Regression by Cross-Validation
- [`predict(`*`<spgtr>`*`)`](https://www.sundayu.me/tensory/reference/predict.spgtr.md)
  : Predict from a Tensor Regression Fit
- [`summary(`*`<spgtr>`*`)`](https://www.sundayu.me/tensory/reference/summary.spgtr.md)
  : Summarize a Tensor Regression Fit
- [`tepls()`](https://www.sundayu.me/tensory/reference/tepls.md) :
  Tensor Envelope Partial Least Squares Regression (TEPLS)
- [`predict(`*`<tepls>`*`)`](https://www.sundayu.me/tensory/reference/predict.tepls.md)
  : Predict from a TEPLS Fit
- [`pqtr()`](https://www.sundayu.me/tensory/reference/pqtr.md)
  [`coef(`*`<pqtr>`*`)`](https://www.sundayu.me/tensory/reference/pqtr.md)
  [`print(`*`<pqtr>`*`)`](https://www.sundayu.me/tensory/reference/pqtr.md)
  : Partial Quantile Tensor Regression (PQTR)
- [`pqtr_cv()`](https://www.sundayu.me/tensory/reference/pqtr_cv.md) :
  Choose the Reduced Dimension of a Quantile Tensor Regression
- [`predict(`*`<pqtr>`*`)`](https://www.sundayu.me/tensory/reference/predict.pqtr.md)
  : Predict from a Partial Quantile Tensor Regression Fit

## Working with CP factors

Post-processing, comparison, and visualization of `KTensor` fits.

- [`arrange()`](https://www.sundayu.me/tensory/reference/arrange.md) :
  Arrange the Components of a Kruskal Tensor
- [`normalize()`](https://www.sundayu.me/tensory/reference/normalize.md)
  : Normalize a Kruskal Tensor
- [`fixsigns()`](https://www.sundayu.me/tensory/reference/fixsigns.md) :
  Fix Sign Ambiguity of a Kruskal Tensor
- [`score()`](https://www.sundayu.me/tensory/reference/score.md) : Score
  the Similarity of Two Kruskal Tensors
- [`ncomponents()`](https://www.sundayu.me/tensory/reference/ncomponents.md)
  : Number of Components of a Kruskal Tensor
- [`extract()`](https://www.sundayu.me/tensory/reference/extract.md) :
  Extract Components of a Kruskal Tensor
- [`redistribute()`](https://www.sundayu.me/tensory/reference/redistribute.md)
  : Redistribute Kruskal Weights into a Mode
- [`tovec()`](https://www.sundayu.me/tensory/reference/tovec.md) :
  Kruskal Tensor to Vector
- [`viz()`](https://www.sundayu.me/tensory/reference/viz.md) : Visualize
  a Kruskal Tensor

## Import and export

- [`import_data()`](https://www.sundayu.me/tensory/reference/import_data.md)
  : Import Tensor Data from a Text File
- [`export_data()`](https://www.sundayu.me/tensory/reference/export_data.md)
  : Export Tensor Data to a Text File
