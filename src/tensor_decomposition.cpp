#include "xtensor-r/rarray.hpp"
#include "xtensor/containers/xarray.hpp"
#include <Rcpp.h>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <numeric>
#include <vector>

#ifndef F77_NAME
#define F77_NAME(x) x##_
#endif

#ifndef FCONE
#define FCONE
#endif

extern "C" {
void F77_NAME(dgemm)(const char *transa, const char *transb, const int *m,
                     const int *n, const int *k, const double *alpha,
                     const double *a, const int *lda, const double *b,
                     const int *ldb, const double *beta, double *c,
                     const int *ldc FCONE FCONE);
}

using namespace Rcpp;

namespace {

std::vector<std::size_t> shape_vec_dec(const xt::rarray<double> &tensor_data) {
  return std::vector<std::size_t>(tensor_data.shape().begin(),
                                  tensor_data.shape().end());
}

std::vector<std::size_t>
column_major_strides_dec(const std::vector<std::size_t> &dims) {
  std::vector<std::size_t> strides(dims.size(), 1);
  for (std::size_t i = 1; i < dims.size(); ++i) {
    strides[i] = strides[i - 1] * dims[i - 1];
  }
  return strides;
}

std::size_t product_dims(const std::vector<std::size_t> &dims) {
  return std::accumulate(dims.begin(), dims.end(),
                         static_cast<std::size_t>(1),
                         std::multiplies<std::size_t>());
}

void check_blas_int(std::size_t value) {
  if (value > static_cast<std::size_t>(INT_MAX)) {
    Rcpp::stop("Tensor extents exceed BLAS 32-bit integer limit");
  }
}

} // namespace

// Khatri-Rao product of two matrices with matching column counts.
// reverse=false (default): out has rows ordered so A-row is slow, B-row is fast
//   (matches existing R khatri_rao.matrix convention: out[i*K + k, r] with i slow).
// reverse=true: flips to B slow, A fast.
// [[Rcpp::export]]
NumericMatrix khatri_rao_pair_cpp(const NumericMatrix &A,
                                  const NumericMatrix &B,
                                  bool reverse = false) {
  const int I = A.nrow();
  const int K = B.nrow();
  const int R = A.ncol();
  if (B.ncol() != R) {
    stop("Matrices must have the same number of columns.");
  }
  const std::size_t IK = static_cast<std::size_t>(I) * K;
  check_blas_int(IK); // R matrix dims are int; a wrapped row count would corrupt the heap
  NumericMatrix out(static_cast<int>(IK), R);
  const double *a_ptr = REAL(A);
  const double *b_ptr = REAL(B);
  double *o_ptr = REAL(out);

  if (reverse) {
    // out row layout: (k slow, i fast) — i.e. kron(B[:,r], A[:,r])
    for (int r = 0; r < R; ++r) {
      const double *ac = a_ptr + static_cast<std::size_t>(r) * I;
      const double *bc = b_ptr + static_cast<std::size_t>(r) * K;
      double *oc = o_ptr + static_cast<std::size_t>(r) * IK;
      for (int k = 0; k < K; ++k) {
        const double bk = bc[k];
        for (int i = 0; i < I; ++i) {
          oc[static_cast<std::size_t>(k) * I + i] = bk * ac[i];
        }
      }
    }
  } else {
    // out row layout: (i slow, k fast) — i.e. kron(A[:,r], B[:,r])
    for (int r = 0; r < R; ++r) {
      const double *ac = a_ptr + static_cast<std::size_t>(r) * I;
      const double *bc = b_ptr + static_cast<std::size_t>(r) * K;
      double *oc = o_ptr + static_cast<std::size_t>(r) * IK;
      for (int i = 0; i < I; ++i) {
        const double ai = ac[i];
        for (int k = 0; k < K; ++k) {
          oc[static_cast<std::size_t>(i) * K + k] = ai * bc[k];
        }
      }
    }
  }
  return out;
}

// Build the mode-n "reduced" Khatri-Rao product used in MTTKRP:
//   Z = U[N] \odot U[N-1] \odot ... \odot U[n+1] \odot U[n-1] \odot ... \odot U[1]
// in column-major row order with mode-1 fastest (skipping mode n). Rows index
// over prod(I_k, k != n) matching the column index of the mode-n unfolding.
namespace {
NumericMatrix build_mttkrp_kr(const List &factors, std::size_t skip,
                              std::size_t N, int R,
                              const std::vector<std::size_t> &dims) {
  std::vector<std::size_t> other_modes;
  std::vector<std::size_t> other_dims;
  other_modes.reserve(N - 1);
  other_dims.reserve(N - 1);
  for (std::size_t k = 0; k < N; ++k) {
    if (k == skip) continue;
    other_modes.push_back(k);
    other_dims.push_back(dims[k]);
  }
  const std::size_t P = product_dims(other_dims);
  NumericMatrix Z(static_cast<R_xlen_t>(P), R);

  // Keep the (possibly coerced) factor matrices alive for the whole build:
  // as<NumericMatrix> allocates a fresh SEXP when a factor is not REALSXP,
  // and a loop-local NumericMatrix would leave mat_ptrs dangling.
  std::vector<NumericMatrix> mats;
  mats.reserve(N - 1);
  std::vector<const double *> mat_ptrs(N, nullptr);
  for (std::size_t k = 0; k < N; ++k) {
    if (k == skip) continue;
    mats.push_back(as<NumericMatrix>(factors[k]));
    mat_ptrs[k] = REAL(mats.back());
  }

  double *z_ptr = REAL(Z);
  const std::size_t M = other_modes.size();
  std::vector<std::size_t> coord(M, 0);

  for (std::size_t p = 0; p < P; ++p) {
    std::size_t tmp = p;
    for (std::size_t q = 0; q < M; ++q) {
      coord[q] = tmp % other_dims[q];
      tmp /= other_dims[q];
    }
    for (int r = 0; r < R; ++r) {
      double prod = 1.0;
      for (std::size_t q = 0; q < M; ++q) {
        const std::size_t mode = other_modes[q];
        const std::size_t dim_mode = dims[mode];
        prod *= mat_ptrs[mode][static_cast<std::size_t>(r) * dim_mode +
                               coord[q]];
      }
      z_ptr[static_cast<std::size_t>(r) * P + p] = prod;
    }
  }

  return Z;
}
} // namespace

// BLAS-backed MTTKRP: V = X_(n) * KR, where KR is the skip-n Khatri-Rao.
// Gathers mode-n unfolding into contiguous (I_n x P) buffer, builds KR,
// then calls dgemm once. Returns an I_n x R matrix.
// [[Rcpp::export]]
NumericMatrix mttkrp_blas_cpp(const xt::rarray<double> &tensor_data,
                              const List &factors, int mode) {
  const std::vector<std::size_t> dims = shape_vec_dec(tensor_data);
  const std::size_t N = dims.size();
  if (N < 2) {
    stop("mttkrp is invalid for tensors with fewer than 2 dimensions.");
  }
  const std::size_t skip = static_cast<std::size_t>(mode - 1);
  if (skip >= N) {
    stop("mode is out of bounds for the tensor order");
  }
  if (static_cast<std::size_t>(factors.size()) != N) {
    stop("factors must have the same length as the tensor order.");
  }

  int R = -1;
  for (std::size_t k = 0; k < N; ++k) {
    if (k == skip) continue;
    NumericMatrix Uk = as<NumericMatrix>(factors[k]);
    if (static_cast<std::size_t>(Uk.nrow()) != dims[k]) {
      stop("factor matrix row dimension does not match tensor dimension");
    }
    if (R < 0) {
      R = Uk.ncol();
    } else if (Uk.ncol() != R) {
      stop("all factor matrices must have the same number of columns");
    }
  }
  if (R < 0) {
    stop("no non-skipped factors present");
  }

  const std::size_t In = dims[skip];
  const std::size_t total = product_dims(dims);
  if (total == 0) {
    // Zero-extent tensor: the unfolding is empty, so V is all zeros
    // (and In may itself be 0 — guard the division below).
    return NumericMatrix(static_cast<int>(In), R);
  }
  const std::size_t P = total / In;

  // Validate BLAS int extents before any allocation narrows them.
  check_blas_int(In);
  check_blas_int(P);
  check_blas_int(static_cast<std::size_t>(R));

  const auto strides = column_major_strides_dec(dims);
  const double *data_ptr = tensor_data.data();

  // Gather mode-n unfolding into Xn (I_n x P), column-major: Xn[p*I_n + i_n].
  // For mode 1 this is a straight copy (data already contiguous); keep branch.
  std::vector<double> Xn(In * P);
  const std::size_t axis_stride = strides[skip];

  if (skip == 0) {
    std::copy(data_ptr, data_ptr + total, Xn.begin());
  } else {
    std::vector<std::size_t> other_modes;
    std::vector<std::size_t> other_dims;
    other_modes.reserve(N - 1);
    other_dims.reserve(N - 1);
    for (std::size_t k = 0; k < N; ++k) {
      if (k == skip) continue;
      other_modes.push_back(k);
      other_dims.push_back(dims[k]);
    }
    const std::size_t M = other_modes.size();
    std::vector<std::size_t> coord(M, 0);
    for (std::size_t p = 0; p < P; ++p) {
      std::size_t tmp = p;
      std::size_t base = 0;
      for (std::size_t q = 0; q < M; ++q) {
        coord[q] = tmp % other_dims[q];
        tmp /= other_dims[q];
        base += coord[q] * strides[other_modes[q]];
      }
      const std::size_t col_off = p * In;
      for (std::size_t i = 0; i < In; ++i) {
        Xn[col_off + i] = data_ptr[base + i * axis_stride];
      }
    }
  }

  NumericMatrix Z = build_mttkrp_kr(factors, skip, N, R, dims);

  NumericMatrix V(static_cast<R_xlen_t>(In), R);

  const char *transa = "N";
  const char *transb = "N";
  const int m = static_cast<int>(In);
  const int n = R;
  const int k = static_cast<int>(P);
  const double alpha = 1.0;
  const double beta = 0.0;
  const int lda = static_cast<int>(In);
  const int ldb = static_cast<int>(P);
  const int ldc = static_cast<int>(In);

  F77_NAME(dgemm)(transa, transb, &m, &n, &k, &alpha, Xn.data(), &lda,
                  REAL(Z), &ldb, &beta, REAL(V), &ldc FCONE FCONE);

  return V;
}
