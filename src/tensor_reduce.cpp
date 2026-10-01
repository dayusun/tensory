// Allocation-free reductions and elementwise kernels for dense tensors.
//
//   dense_dot_cpp     sum(x * y) via BLAS ddot (fnorm, innerprod, ttt)
//   gram_cpp          X_(n) X_(n)' without unfolding (nvecs)
//   t_scale_cpp       scale along modes without a tensor-sized STATS array
//   symmetrize_cpp    average over index permutations within one mode group
//
// Each has a pure-R reference path in R/ that the tests pin it against.
#include "tensor_array.h"
#include <Rcpp.h>
#include <algorithm>
#include <climits>
#include <vector>

#ifndef FCONE
#define FCONE
#endif

extern "C" {
double F77_NAME(ddot)(const int *n, const double *x, const int *incx,
                      const double *y, const int *incy);
void F77_NAME(dsyrk)(const char *uplo, const char *trans, const int *n,
                     const int *k, const double *alpha, const double *a,
                     const int *lda, const double *beta, double *c,
                     const int *ldc FCONE FCONE);
}

using namespace Rcpp;

namespace {

int blas_int(std::size_t value) {
  if (value > static_cast<std::size_t>(INT_MAX)) {
    Rcpp::stop("Tensor extents exceed BLAS 32-bit integer limit");
  }
  return static_cast<int>(value);
}

// C (n x n, upper triangle) += A' A or A A' -- see dsyrk.
void syrk(char trans, std::size_t n, std::size_t k, const double *a,
          std::size_t lda, double beta, double *c) {
  const double one = 1.0;
  const int n_ = blas_int(n);
  const int k_ = blas_int(k);
  const int lda_ = blas_int(lda);
  F77_NAME(dsyrk)("U", &trans, &n_, &k_, &one, a, &lda_, &beta, c,
                  &n_ FCONE FCONE);
}

} // namespace

// sum(x * y) over all elements, read in place. Long vectors are summed in
// INT_MAX-sized chunks. A NaN result (any NA/NaN input) is returned as is;
// the R caller recomputes with sum() so NA vs NaN semantics match base R.
// [[Rcpp::export]]
double dense_dot_cpp(const NumericVector &x, const NumericVector &y) {
  const R_xlen_t n = Rf_xlength(x);
  if (Rf_xlength(y) != n) {
    stop("x and y must have the same number of elements");
  }
  const double *px = REAL(x);
  const double *py = REAL(y);
  const int one = 1;
  double total = 0.0;
  for (R_xlen_t start = 0; start < n; start += INT_MAX) {
    const int len = static_cast<int>(std::min<R_xlen_t>(INT_MAX, n - start));
    total += F77_NAME(ddot)(&len, px + start, &one, py + start, &one);
  }
  return total;
}

// Gram matrix of the mode-n unfolding, G = X_(n) X_(n)' (I_n x I_n), without
// forming the unfolding. The tensor is viewed as M1 x In x M2; every M2 slice
// X_s (M1 x In, contiguous) contributes X_s' X_s. Mode 1 is a single dsyrk.
// When M1 is tiny, per-slice calls are too small to be efficient, so slices
// are gathered (transposed) into an In x (M1 * chunk) buffer of bounded size.
// [[Rcpp::export]]
NumericMatrix gram_cpp(const NumericVector &tensor_data, int mode) {
  const std::vector<std::size_t> dims = tensory::array_dims(tensor_data);
  const std::size_t axis = static_cast<std::size_t>(mode - 1);
  if (axis >= dims.size()) {
    stop("mode is out of bounds for the tensor order");
  }
  const std::size_t In = dims[axis];
  std::size_t M1 = 1;
  for (std::size_t i = 0; i < axis; ++i) M1 *= dims[i];
  std::size_t M2 = 1;
  for (std::size_t i = axis + 1; i < dims.size(); ++i) M2 *= dims[i];

  NumericMatrix G(blas_int(In), blas_int(In));
  if (In == 0 || M1 * M2 == 0) {
    return G;
  }
  const double *X = REAL(tensor_data);
  double *g = REAL(G);

  if (M1 == 1) {
    syrk('N', In, M2, X, In, 0.0, g);
  } else if (M1 >= 4) {
    for (std::size_t s = 0; s < M2; ++s) {
      syrk('T', In, M1, X + s * M1 * In, M1, s == 0 ? 0.0 : 1.0, g);
    }
  } else {
    // Gather up to ~1M doubles of columns at a time: column (s, i1) of the
    // unfolding is X[i1 + M1 * (k + In * s)] for k = 0..In-1.
    const std::size_t per_slice = M1 * In;
    const std::size_t chunk = std::max<std::size_t>(1, (1u << 20) / per_slice);
    std::vector<double> buf(In * M1 * std::min(chunk, M2));
    bool first = true;
    for (std::size_t s0 = 0; s0 < M2; s0 += chunk) {
      const std::size_t ns = std::min(chunk, M2 - s0);
      for (std::size_t s = 0; s < ns; ++s) {
        const double *xs = X + (s0 + s) * per_slice;
        for (std::size_t i1 = 0; i1 < M1; ++i1) {
          double *col = buf.data() + (s * M1 + i1) * In;
          for (std::size_t k = 0; k < In; ++k) col[k] = xs[i1 + M1 * k];
        }
      }
      syrk('N', In, M1 * ns, buf.data(), In, first ? 0.0 : 1.0, g);
      first = false;
    }
  }
  // dsyrk filled the upper triangle; mirror it.
  for (std::size_t j = 0; j < In; ++j) {
    for (std::size_t i = j + 1; i < In; ++i) g[i + j * In] = g[j + i * In];
  }
  return G;
}

// out = x scaled by s along `modes` (1-based, in the order s is laid out:
// s is column-major over dims[modes], first listed mode fastest). One pass,
// no tensor-sized intermediate.
// [[Rcpp::export]]
NumericVector t_scale_cpp(const NumericVector &tensor_data,
                          const NumericVector &s, const IntegerVector &modes) {
  const std::vector<std::size_t> dims = tensory::array_dims(tensor_data);
  const std::size_t N = dims.size();
  // stride of each tensor mode inside s (0 for unscaled modes)
  std::vector<std::size_t> s_stride(N, 0);
  std::size_t s_len = 1;
  for (R_xlen_t g = 0; g < modes.size(); ++g) {
    const int m = modes[g] - 1;
    if (m < 0 || static_cast<std::size_t>(m) >= N) {
      stop("modes must be valid tensor modes");
    }
    s_stride[m] = s_len;
    s_len *= dims[m];
  }
  if (static_cast<std::size_t>(Rf_xlength(s)) != s_len) {
    stop("s must have one element per entry of the scaled modes");
  }
  NumericVector out = tensory::alloc_array(dims);
  const std::size_t total = tensory::product_of_dims(dims);
  const double *x = REAL(tensor_data);
  const double *sp = REAL(s);
  double *o = REAL(out);
  if (total == 0) return out;

  std::vector<std::size_t> coord(N, 0);
  std::size_t sidx = 0;
  for (std::size_t lin = 0; lin < total; ++lin) {
    o[lin] = x[lin] * sp[sidx];
    for (std::size_t k = 0; k < N; ++k) {
      if (++coord[k] < dims[k]) {
        sidx += s_stride[k];
        break;
      }
      sidx -= s_stride[k] * (dims[k] - 1);
      coord[k] = 0;
    }
  }
  return out;
}

// Symmetrize over one group of equally sized modes (1-based): every entry is
// replaced by the mean of its equivalence class (all entries whose indices in
// the group are a permutation of its own). Classes are keyed by the entry with
// the group indices sorted, so all members read the same slot and the result
// is exactly symmetric. Sums accumulate in linear-index order.
// [[Rcpp::export]]
NumericVector symmetrize_cpp(const NumericVector &tensor_data,
                             const IntegerVector &grp) {
  const std::vector<std::size_t> dims = tensory::array_dims(tensor_data);
  const std::vector<std::size_t> strides = tensory::column_major_strides(dims);
  const std::size_t N = dims.size();
  const std::size_t G = static_cast<std::size_t>(grp.size());
  std::vector<std::size_t> gm(G);
  for (std::size_t g = 0; g < G; ++g) {
    const int m = grp[g] - 1;
    if (m < 0 || static_cast<std::size_t>(m) >= N) {
      stop("group modes must be valid tensor modes");
    }
    gm[g] = static_cast<std::size_t>(m);
    if (dims[gm[g]] != dims[gm[0]]) {
      stop("Dimension mismatch for symmetrization.");
    }
  }
  const std::size_t total = tensory::product_of_dims(dims);
  NumericVector out = tensory::alloc_array(dims);
  if (total == 0) return out;
  const double *x = REAL(tensor_data);
  double *o = REAL(out);

  std::vector<double> sum(total, 0.0);
  std::vector<double> cnt(total, 0.0);
  std::vector<std::size_t> coord(N, 0);
  std::vector<std::size_t> key(G);

  auto canonical = [&](std::size_t lin) {
    for (std::size_t g = 0; g < G; ++g) key[g] = coord[gm[g]];
    std::sort(key.begin(), key.end());
    std::size_t c = lin;
    for (std::size_t g = 0; g < G; ++g) {
      c = c + key[g] * strides[gm[g]] - coord[gm[g]] * strides[gm[g]];
    }
    return c;
  };
  auto advance = [&]() {
    for (std::size_t k = 0; k < N; ++k) {
      if (++coord[k] < dims[k]) return;
      coord[k] = 0;
    }
  };

  for (std::size_t lin = 0; lin < total; ++lin) {
    const std::size_t c = canonical(lin);
    sum[c] += x[lin];
    cnt[c] += 1.0;
    advance();
  }
  std::fill(coord.begin(), coord.end(), 0);
  for (std::size_t lin = 0; lin < total; ++lin) {
    const std::size_t c = canonical(lin);
    o[lin] = sum[c] / cnt[c];
    advance();
  }
  return out;
}
