#include "tensor_array.h"
#include <Rcpp.h>
#include <algorithm>
#include <climits>
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

namespace {

void gemm(char transa, char transb, int m, int n, int k, const double *a,
          int lda, const double *b, int ldb, double *c, int ldc) {
  const double alpha = 1.0;
  const double beta = 0.0;
  F77_NAME(dgemm)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb,
                  &beta, c, &ldc FCONE FCONE);
}

int blas_int(std::size_t value) {
  if (value > static_cast<std::size_t>(INT_MAX)) {
    Rcpp::stop("Tensor extents exceed BLAS 32-bit integer limit");
  }
  return static_cast<int>(value);
}

// Below this many leading elements (M1) a per-slice dgemm is too small to
// amortize its call overhead, and gathering into one contiguous buffer wins:
// at M1 = 2 the slice loop is up to 2x slower on reference BLAS, M1 = 3 is a
// tie, and from M1 = 4 slicing wins on both reference BLAS and OpenBLAS.
constexpr std::size_t kGatherBelowM1 = 4;

} // namespace

/**
 * @brief Tensor-times-matrix operation with optional transpose
 *
 * Views the column-major tensor as M1 x Ik x M2 (M1 = product of the dims
 * before `mode`, M2 = product of those after) and computes the result
 * M1 x J x M2 without permuting the tensor:
 *
 *  - M1 == 1 (mode 1): the tensor is already an Ik x M2 matrix, so one dgemm
 *    Y = A X reads R's storage in place.
 *  - M1 >= kGatherBelowM1: every M2 slice is a contiguous M1 x Ik matrix, so
 *    one dgemm per slice, Y_s = X_s A', writes straight into the result. For
 *    the last mode M2 == 1 and this is a single dgemm. (doc/lesson.md's
 *    512-row cache tiling of that call is deliberately absent: it was 0-13%
 *    faster on reference BLAS but 1.2-2.3x slower on OpenBLAS, which already
 *    blocks for cache internally.)
 *  - 1 < M1 < kGatherBelowM1: gather into an Ik x (M1 M2) buffer, one dgemm,
 *    scatter back.
 *
 * @param tensor_data Input tensor as an R array (aliased, not copied)
 * @param matrix Input matrix, J x Ik (or Ik x J when `transpose`)
 * @param mode Mode of the tensor to contract with the matrix
 * @param transpose Whether to transpose the matrix before multiplication
 * @return Resulting tensor after multiplication
 */
// [[Rcpp::export]]
NumericVector ttm_cpp(const NumericVector &tensor_data,
                      const NumericMatrix &matrix, int mode,
                      bool transpose = false) {
  try {
    const std::size_t axis = static_cast<std::size_t>(mode - 1);
    const std::vector<std::size_t> dims = tensory::array_dims(tensor_data);

    if (axis >= dims.size()) {
      Rcpp::stop("mode is out of bounds for the tensor order");
    }

    const std::size_t Ik = dims[axis];
    const std::size_t J = transpose ? static_cast<std::size_t>(matrix.ncol())
                                    : static_cast<std::size_t>(matrix.nrow());
    const std::size_t contracted = transpose
                                       ? static_cast<std::size_t>(matrix.nrow())
                                       : static_cast<std::size_t>(matrix.ncol());
    if (contracted != Ik) {
      Rcpp::stop("matrix does not match the tensor dimension of the mode");
    }

    std::size_t M1 = 1;
    for (std::size_t i = 0; i < axis; ++i) M1 *= dims[i];
    std::size_t M2 = 1;
    for (std::size_t i = axis + 1; i < dims.size(); ++i) M2 *= dims[i];

    std::vector<std::size_t> out_dims = dims;
    out_dims[axis] = J;
    NumericVector result = tensory::alloc_array(out_dims);
    double *Y = REAL(result);
    const std::size_t out_size = M1 * J * M2;

    if (out_size == 0) {
      return result;
    }
    if (Ik == 0) {
      // Empty contraction: every entry is an empty sum.
      std::fill(Y, Y + out_size, 0.0);
      return result;
    }

    const double *X = REAL(tensor_data);
    const double *A = REAL(matrix);
    const int j = blas_int(J);
    const int ik = blas_int(Ik);
    // Leading dimension of A as stored (J x Ik, or Ik x J when transposed).
    const int lda = transpose ? ik : j;

    if (M1 == 1) {
      gemm(transpose ? 'T' : 'N', 'N', j, blas_int(M2), ik, A, lda, X, ik, Y,
           j);
      return result;
    }

    if (M1 >= kGatherBelowM1) {
      const int m1 = blas_int(M1);
      blas_int(M2);
      const char transb = transpose ? 'N' : 'T';
      const std::size_t x_slice = M1 * Ik;
      const std::size_t y_slice = M1 * J;
      for (std::size_t s = 0; s < M2; ++s) {
        gemm('N', transb, m1, j, ik, X + s * x_slice, m1, A, lda,
             Y + s * y_slice, m1);
      }
      return result;
    }

    // Tiny M1: gather columns (m1, s) of the mode unfolding into a contiguous
    // Ik x (M1 M2) buffer, multiply once, scatter the J x (M1 M2) result.
    const std::size_t rest = M1 * M2;
    const int n = blas_int(rest);
    std::vector<double> x_mat(Ik * rest);
    for (std::size_t s = 0; s < M2; ++s) {
      for (std::size_t k = 0; k < Ik; ++k) {
        const double *src = X + (s * Ik + k) * M1;
        for (std::size_t i = 0; i < M1; ++i) {
          x_mat[(s * M1 + i) * Ik + k] = src[i];
        }
      }
    }

    std::vector<double> y_mat(J * rest);
    gemm(transpose ? 'T' : 'N', 'N', j, n, ik, A, lda, x_mat.data(), ik,
         y_mat.data(), j);

    for (std::size_t s = 0; s < M2; ++s) {
      for (std::size_t jj = 0; jj < J; ++jj) {
        double *dst = Y + (s * J + jj) * M1;
        for (std::size_t i = 0; i < M1; ++i) {
          dst[i] = y_mat[(s * M1 + i) * J + jj];
        }
      }
    }

    return result;
  } catch (const std::exception &e) {
    Rcpp::stop("Error in optimized ttm_cpp: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
NumericVector ttm_multiple_cpp(const NumericVector &tensor_data,
                               const List &matrices,
                               const IntegerVector &modes,
                               bool transpose = false) {
  try {
    if (matrices.size() == 0) {
      Rcpp::stop("Empty list of matrices provided");
    }

    NumericMatrix first_matrix = as<NumericMatrix>(matrices[0]);
    NumericVector result = ttm_cpp(tensor_data, first_matrix, modes[0], transpose);

    for (R_xlen_t i = 1; i < matrices.size(); ++i) {
      NumericMatrix matrix = as<NumericMatrix>(matrices[i]);
      result = ttm_cpp(result, matrix, modes[i], transpose);
    }

    return result;

  } catch (const std::exception &e) {
    Rcpp::stop("Error in ttm_multiple_cpp: " + std::string(e.what()));
  }
}
