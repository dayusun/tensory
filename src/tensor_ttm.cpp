#include "xtensor-blas/xlinalg.hpp"
#include "xtensor-r/rarray.hpp"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include <Rcpp.h>
#include <cstring>
#include <vector>

// Use R's BLAS interface via Fortran calls (standard R package approach)
// Fortran name mangling
#ifndef F77_NAME
#define F77_NAME(x) x##_
#endif

#ifndef FCONE
#define FCONE
#endif

// Externally declare the Fortran BLAS subroutines
extern "C" {
void F77_NAME(dgemm)(const char *transa, const char *transb, const int *m,
                     const int *n, const int *k, const double *alpha,
                     const double *a, const int *lda, const double *b,
                     const int *ldb, const double *beta, double *c,
                     const int *ldc FCONE FCONE);
}

using namespace Rcpp;

/**
 * @brief Helper function to create zero-copy R-aware matrix view using
 * xt::rarray
 * @param matrix R NumericMatrix
 * @return xtensor rarray view that directly maps R memory with correct
 * column-major layout
 */
inline auto create_matrix_view(const NumericMatrix &matrix) {
  // Create zero-copy rarray directly from R SEXP - this preserves R's
  // column-major layout
  return xt::rarray<double>(SEXP(matrix));
}

/**
 * @brief Tensor-times-matrix operation with optional transpose
 * @param tensor_data Input tensor as xtensor rarray
 * @param matrix_data Input matrix as xtensor rarray
 * @param mode Mode of the tensor to contract with the matrix
 * @param transpose Whether to transpose the matrix before multiplication
 * @return Resulting tensor after multiplication
 */
// [[Rcpp::export]]
xt::rarray<double> ttm_cpp(const xt::rarray<double> &tensor_data,
                           const NumericMatrix &matrix, int mode,
                           bool transpose = false) {
  try {
    const std::size_t axis = static_cast<std::size_t>(mode - 1);
    auto tensor_shape = tensor_data.shape();
    std::size_t n_dim = tensor_data.dimension();

    // Matrix dimensions
    const int M_nrow = matrix.nrow();
    const int M_ncol = matrix.ncol();

    std::size_t contract_len, new_dim;
    if (transpose) {
      contract_len = static_cast<std::size_t>(M_nrow);
      new_dim = static_cast<std::size_t>(M_ncol);
    } else {
      contract_len = static_cast<std::size_t>(M_ncol);
      new_dim = static_cast<std::size_t>(M_nrow);
    }

    // Output shape
    std::vector<std::size_t> final_shape(tensor_shape.begin(),
                                         tensor_shape.end());
    final_shape[axis] = new_dim;

    // Calculate slice dimensions
    std::size_t M1 = 1;
    for (std::size_t i = 0; i < axis; ++i) {
      M1 *= tensor_shape[i];
    }

    std::size_t Ik = tensor_shape[axis];

    std::size_t M2 = 1;
    for (std::size_t i = axis + 1; i < n_dim; ++i) {
      M2 *= tensor_shape[i];
    }

    // Allocate result tensor
    xt::xarray<double, xt::layout_type::column_major> result_tensor(
        final_shape);

    // Compute parameters for DGEMM
    // We compute Y = X * A^T (if !transpose) or Y = X * A (if transpose)
    // X is M1 x Ik, A is either J x Ik (if !transpose) or Ik x J (if transpose)
    // Y is M1 x J.
    const char *transa = "N";
    const char *transb = transpose ? "N" : "T";

    const int m = static_cast<int>(M1);
    const int n = static_cast<int>(new_dim);
    const int k = static_cast<int>(Ik);

    const double alpha = 1.0;
    const double beta = 0.0;

    const int lda = static_cast<int>(M1);
    const int ldb = static_cast<int>(transpose ? Ik : new_dim);
    const int ldc = static_cast<int>(M1);

    const double *X_base = tensor_data.data();
    const double *mat_ptr = REAL(matrix);
    double *Y_base = result_tensor.data();

    if (M1 == 1) {
      // OPTIMIZATION for axis == 0: Y (J x M2) = Mat * X
      const char *transa_opt = transpose ? "T" : "N";
      const char *transb_opt = "N";

      int m_opt = static_cast<int>(new_dim);
      int n_opt = static_cast<int>(M2);
      int k_opt = static_cast<int>(Ik);

      int lda_opt = static_cast<int>(transpose ? Ik : new_dim);
      int ldb_opt = static_cast<int>(Ik);
      int ldc_opt = static_cast<int>(new_dim);

      F77_NAME(dgemm)(transa_opt, transb_opt, &m_opt, &n_opt, &k_opt, &alpha,
                      mat_ptr, &lda_opt, X_base, &ldb_opt, &beta, Y_base,
                      &ldc_opt FCONE FCONE);
    } else {
      std::size_t X_stride = M1 * Ik;
      std::size_t Y_stride = M1 * new_dim;

      // Perform M2 matrix multiplications
      for (std::size_t m2 = 0; m2 < M2; ++m2) {
        const double *X_ptr = X_base + m2 * X_stride;
        double *Y_ptr = Y_base + m2 * Y_stride;

        F77_NAME(dgemm)(transa, transb, &m, &n, &k, &alpha, X_ptr, &lda,
                        mat_ptr, &ldb, &beta, Y_ptr, &ldc FCONE FCONE);
      }
    }

    return xt::rarray<double>(result_tensor);

  } catch (const std::exception &e) {
    Rcpp::stop("Error in optimized ttm_cpp: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
xt::rarray<double> ttm_multiple_cpp(const xt::rarray<double> &tensor_data,
                                    const List &matrices,
                                    const IntegerVector &modes,
                                    bool transpose = false) {
  try {
    if (matrices.size() == 0) {
      Rcpp::stop("Empty list of matrices provided");
    }

    xt::rarray<double> result = tensor_data;
    for (size_t i = 0; i < matrices.size(); ++i) {
      NumericMatrix matrix = as<NumericMatrix>(matrices[i]);
      result = ttm_cpp(result, matrix, modes[i], transpose);
    }

    return result;

  } catch (const std::exception &e) {
    Rcpp::stop("Error in ttm_multiple_cpp: " + std::string(e.what()));
  }
}
