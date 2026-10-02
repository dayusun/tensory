#include "tensor_array.h"
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
void F77_NAME(dgemv)(const char *trans, const int *m, const int *n,
                     const double *alpha, const double *a, const int *lda,
                     const double *x, const int *incx, const double *beta,
                     double *y, const int *incy FCONE);
}

using namespace Rcpp;

namespace {

std::vector<std::size_t> shape_vec_dec(const NumericVector &tensor_data) {
  return tensory::array_dims(tensor_data);
}

using tensory::product_of_dims;

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

// ---------------------------------------------------------------------------
// MTTKRP: V = X_(n) (U_N (.) ... (.) U_{n+1} (.) U_{n-1} (.) ... (.) U_1).
//
// Both kernels below are "two-step" MTTKRPs (Phan, Tichavsky & Cichocki
// 2013): the tensor is contracted with a Khatri-Rao product of the factors on
// ONE side of a split in a single dgemm that reads R's storage in place, and
// the (much smaller) result is then contracted with the other side. Neither
// the mode-n unfolding nor the full prod(I_k, k != n) x R Khatri-Rao product
// is ever formed.
// ---------------------------------------------------------------------------
namespace {

void gemm(char transa, char transb, int m, int n, int k, const double *a,
          int lda, const double *b, int ldb, double *c, int ldc) {
  const double one = 1.0;
  const double zero = 0.0;
  F77_NAME(dgemm)(&transa, &transb, &m, &n, &k, &one, a, &lda, b, &ldb, &zero,
                  c, &ldc FCONE FCONE);
}

void gemv(char trans, int m, int n, const double *a, int lda, const double *x,
          double *y) {
  const double one = 1.0;
  const double zero = 0.0;
  const int inc = 1;
  F77_NAME(dgemv)(&trans, &m, &n, &one, a, &lda, x, &inc, &zero, y,
                  &inc FCONE);
}

int blas_int(std::size_t value) {
  check_blas_int(value);
  return static_cast<int>(value);
}

// Factor matrices after validation. `mats` keeps coerced copies alive
// (as<NumericMatrix> allocates when a factor is not REALSXP), so the raw
// pointers stay valid for the whole kernel.
struct Factors {
  std::vector<NumericMatrix> mats;
  std::vector<const double *> ptr;
  std::size_t rank = 0;
};

// Validates every factor except `skip` (pass dims.size() to validate all).
Factors read_factors(const List &factors, const std::vector<std::size_t> &dims,
                     std::size_t skip) {
  const std::size_t N = dims.size();
  if (N < 2) {
    stop("mttkrp is invalid for tensors with fewer than 2 dimensions.");
  }
  if (static_cast<std::size_t>(factors.size()) != N) {
    stop("factors must have the same length as the tensor order.");
  }
  Factors f;
  f.mats.resize(N);
  f.ptr.assign(N, nullptr);
  int R = -1;
  for (std::size_t k = 0; k < N; ++k) {
    if (k == skip) continue;
    f.mats[k] = as<NumericMatrix>(factors[k]);
    if (static_cast<std::size_t>(f.mats[k].nrow()) != dims[k]) {
      stop("factor matrix row dimension does not match tensor dimension");
    }
    if (R < 0) {
      R = f.mats[k].ncol();
    } else if (f.mats[k].ncol() != R) {
      stop("all factor matrices must have the same number of columns");
    }
    f.ptr[k] = REAL(f.mats[k]);
  }
  if (R < 0) {
    stop("no non-skipped factors present");
  }
  f.rank = static_cast<std::size_t>(R);
  return f;
}

// Khatri-Rao product of the factors of modes [first, last), rows ordered with
// mode `first` fastest -- i.e. row index = the column-major linear index over
// those modes. Returns a rows x R column-major buffer (rows = 1 when empty).
std::vector<double> kr_range(const Factors &f,
                             const std::vector<std::size_t> &dims,
                             std::size_t first, std::size_t last) {
  const std::size_t R = f.rank;
  std::vector<double> K(R, 1.0);
  std::size_t rows = 1;
  for (std::size_t k = first; k < last; ++k) {
    const std::size_t Ik = dims[k];
    std::vector<double> next(rows * Ik * R);
    for (std::size_t r = 0; r < R; ++r) {
      const double *u = f.ptr[k] + r * Ik;
      const double *prev = K.data() + r * rows;
      double *out = next.data() + r * rows * Ik;
      for (std::size_t i = 0; i < Ik; ++i) {
        const double ui = u[i];
        double *dst = out + i * rows;
        for (std::size_t p = 0; p < rows; ++p) dst[p] = prev[p] * ui;
      }
    }
    K.swap(next);
    rows *= Ik;
  }
  return K;
}

std::size_t prod_range(const std::vector<std::size_t> &dims, std::size_t first,
                       std::size_t last) {
  std::size_t p = 1;
  for (std::size_t k = first; k < last; ++k) p *= dims[k];
  return p;
}

// Single-mode MTTKRP on raw storage; writes the In x R result into V.
//
// Pick a block of "outer" modes at one end of the tensor -- a suffix [c, N)
// with c > n, or a prefix [0, c) with c <= n -- and contract the tensor with
// the Khatri-Rao product of that block's factors in one dgemm that reads R's
// storage in place. What is left is a partial tensor T of shape
// a x In x b (per column r), finished by two dgemv calls per column against
// the Khatri-Rao products of the remaining modes on each side of n.
//
// The block is chosen to minimise Mo + total / Mo (the outer Khatri-Rao
// product plus the partial tensor, both times R), so Mo lands near
// sqrt(total). Ties go to the smaller Mo: the dgemm is then rows x R x Mo
// with the shorter inner dimension, which OpenBLAS runs ~1.6x faster
// (200^3, mode 1: 3.8 -> 2.3 ms) and reference BLAS at the same speed. Taking the whole larger side instead -- the obvious choice --
// builds a Khatri-Rao product of total / In rows for mode 1 and mode N,
// which dominates the run time for high-order tensors.
void mttkrp_two_step(const double *X, const Factors &f,
                     const std::vector<std::size_t> &dims, std::size_t n,
                     double *V) {
  const std::size_t N = dims.size();
  const std::size_t R = f.rank;
  const std::size_t In = dims[n];
  const std::size_t total = prod_range(dims, 0, N);

  // Candidate outer blocks. suffix: [c, N) for c in (n, N); prefix: [0, c)
  // for c in (0, n]. N >= 2 guarantees at least one candidate.
  bool suffix = true;
  std::size_t cut = 0;
  double best = -1.0;
  double best_mo = 0.0;
  auto consider = [&](double Mo, bool is_suffix, std::size_t c) {
    const double cost = Mo + static_cast<double>(total) / Mo;
    if (best < 0 || cost < best || (cost == best && Mo < best_mo)) {
      best = cost;
      best_mo = Mo;
      suffix = is_suffix;
      cut = c;
    }
  };
  for (std::size_t c = n + 1; c < N; ++c) {
    consider(static_cast<double>(prod_range(dims, c, N)), true, c);
  }
  for (std::size_t c = 1; c <= n; ++c) {
    consider(static_cast<double>(prod_range(dims, 0, c)), false, c);
  }

  // T_r is a x In x b: `a` = modes between the outer block and n on the
  // left, `b` = the same on the right.
  const std::size_t a_first = suffix ? 0 : cut;
  const std::size_t b_last = suffix ? cut : N;
  const std::size_t a = prod_range(dims, a_first, n);
  const std::size_t b = prod_range(dims, n + 1, b_last);
  const std::size_t rows = a * In * b;
  const std::size_t Mo = total / rows;

  const int r_ = blas_int(R);
  const int rows_ = blas_int(rows);
  const int mo = blas_int(Mo);
  std::vector<double> T(rows * R);
  {
    std::vector<double> Ko =
        suffix ? kr_range(f, dims, cut, N) : kr_range(f, dims, 0, cut);
    if (suffix) {
      // X viewed as rows x Mo (outer modes slowest).
      gemm('N', 'N', rows_, r_, mo, X, rows_, Ko.data(), mo, T.data(), rows_);
    } else {
      // X viewed as Mo x rows (outer modes fastest).
      gemm('T', 'N', rows_, r_, mo, X, mo, Ko.data(), mo, T.data(), rows_);
    }
  }

  // Remaining factors on each side of n. When a side has no modes (or only
  // singleton modes) its Khatri-Rao product is a 1 x R row -- all ones, or
  // the singleton factor rows, which still scale the result.
  const std::vector<double> KA = kr_range(f, dims, a_first, n);
  const std::vector<double> KB = kr_range(f, dims, n + 1, b_last);
  const int a_ = blas_int(a);
  const int b_ = blas_int(b);
  const int in_ = blas_int(In);
  const int inb = blas_int(In * b);
  std::vector<double> u(In * b);
  for (std::size_t r = 0; r < R; ++r) {
    // u = T_r(a x In b)' KA_r, then V_r = u(In x b) KB_r.
    gemv('T', a_, inb, T.data() + r * rows, a_, KA.data() + r * a, u.data());
    gemv('N', in_, b_, u.data(), in_, KB.data() + r * b, V + r * In);
  }
}

// For a group of consecutive modes [first, last) and a partial tensor P
// (M_group x R, column-major, group modes in column-major order with `first`
// fastest) that already carries every other mode's contribution, finish the
// MTTKRP of each mode in the group: V_n(i, r) = sum over the group's other
// indices of P(idx, r) * prod_{k != n} U_k(idx_k, r).
// MTTKRP of mode n (first <= n < last) from a partial tensor P over the
// group [first, last): V(i, r) = sum over the group's other indices of
// P(idx, r) * prod_{k in group, k != n} U_k(idx_k, r). P is M_group x R,
// column-major, group modes in column-major order with `first` fastest, so
// column r is an a x In x b array (a = modes of the group before n, b =
// after); two dgemv calls against the Khatri-Rao products of those modes
// finish it, as in mttkrp_two_step.
NumericMatrix finish_mode(const double *P, const Factors &f,
                          const std::vector<std::size_t> &dims,
                          std::size_t first, std::size_t last, std::size_t n) {
  const std::size_t R = f.rank;
  const std::size_t In = dims[n];
  const std::size_t a = prod_range(dims, first, n);
  const std::size_t b = prod_range(dims, n + 1, last);
  const std::size_t Mg = a * In * b;
  NumericMatrix V(static_cast<int>(In), static_cast<int>(R));
  if (Mg == 0) return V;
  double *v = REAL(V);
  const std::vector<double> KA = kr_range(f, dims, first, n);
  const std::vector<double> KB = kr_range(f, dims, n + 1, last);
  const int a_ = blas_int(a);
  const int b_ = blas_int(b);
  const int in_ = blas_int(In);
  const int inb = blas_int(In * b);
  std::vector<double> u(In * b);
  for (std::size_t r = 0; r < R; ++r) {
    gemv('T', a_, inb, P + r * Mg, a_, KA.data() + r * a, u.data());
    gemv('N', in_, b_, u.data(), in_, KB.data() + r * b, v + r * In);
  }
  return V;
}

void finish_group(const std::vector<double> &P, const Factors &f,
                  const std::vector<std::size_t> &dims, std::size_t first,
                  std::size_t last, std::vector<NumericMatrix> &out) {
  for (std::size_t n = first; n < last; ++n) {
    out[n] = finish_mode(P.data(), f, dims, first, last, n);
  }
}

// Partial tensor for one side of the split at s: left = true contracts the
// modes [s, N) into the M_L x R partial over [0, s); left = false contracts
// [0, s) into the M_R x R partial over [s, N). One dgemm on R's storage,
// written straight into `out` (M_side * R doubles). A one-mode group's
// partial is that mode's MTTKRP, so it goes through mttkrp_two_step, which
// can split the contraction to avoid the slow long-inner-dimension dgemm
// (for 3-way tensors one group always has a single mode).
void side_partial(const double *X, const Factors &f,
                  const std::vector<std::size_t> &dims, std::size_t s,
                  bool left, double *out) {
  const std::size_t N = dims.size();
  if (left ? s == 1 : s == N - 1) {
    mttkrp_two_step(X, f, dims, left ? 0 : N - 1, out);
    return;
  }
  const std::size_t ML = prod_range(dims, 0, s);
  const std::size_t MR = prod_range(dims, s, N);
  const int ml = blas_int(ML);
  const int mr = blas_int(MR);
  const int r_ = blas_int(f.rank);
  if (left) {
    const std::vector<double> K = kr_range(f, dims, s, N);
    gemm('N', 'N', ml, r_, mr, X, ml, K.data(), mr, out, ml);
  } else {
    const std::vector<double> K = kr_range(f, dims, 0, s);
    gemm('T', 'N', mr, r_, ml, X, ml, K.data(), ml, out, mr);
  }
}

} // namespace

// Single-mode MTTKRP: returns the I_n x R matrix (see mttkrp_two_step).
// [[Rcpp::export]]
NumericMatrix mttkrp_blas_cpp(const NumericVector &tensor_data,
                              const List &factors, int mode) {
  const std::vector<std::size_t> dims = shape_vec_dec(tensor_data);
  if (dims.size() < 2) {
    stop("mttkrp is invalid for tensors with fewer than 2 dimensions.");
  }
  const std::size_t skip = static_cast<std::size_t>(mode - 1);
  if (skip >= dims.size()) {
    stop("mode is out of bounds for the tensor order");
  }
  const Factors f = read_factors(factors, dims, skip);
  const std::size_t In = dims[skip];
  check_blas_int(In);
  check_blas_int(f.rank);

  NumericMatrix V(static_cast<int>(In), static_cast<int>(f.rank));
  if (product_of_dims(dims) == 0 || f.rank == 0) {
    // Empty unfolding (In may itself be 0): V is all zeros.
    return V;
  }
  mttkrp_two_step(REAL(tensor_data), f, dims, skip, REAL(V));
  return V;
}

// MTTKRP for every mode with fixed factors, reading the tensor twice instead
// of N times. The modes are split into two consecutive groups L = [0, s) and
// Rg = [s, N) with prod(dims in L) as close as possible to prod(dims in Rg):
//   P_L = X(M_L x M_R) * KR(Rg)   (M_L x R, one dgemm)
//   P_R = X(M_L x M_R)' * KR(L)   (M_R x R, one dgemm)
// and each mode's result is finished from its group's small partial tensor.
// [[Rcpp::export]]
List mttkrps_cpp(const NumericVector &tensor_data, const List &factors) {
  const std::vector<std::size_t> dims = shape_vec_dec(tensor_data);
  const std::size_t N = dims.size();
  const Factors f = read_factors(factors, dims, N);
  const std::size_t R = f.rank;
  check_blas_int(R);

  std::vector<NumericMatrix> out(N);
  if (product_of_dims(dims) == 0 || R == 0) {
    for (std::size_t k = 0; k < N; ++k) {
      check_blas_int(dims[k]);
      out[k] = NumericMatrix(static_cast<int>(dims[k]), static_cast<int>(R));
    }
    return wrap(out);
  }

  // Balance the split: minimize max(M_L, M_R).
  std::size_t s = 1;
  std::size_t best = static_cast<std::size_t>(-1);
  for (std::size_t c = 1; c < N; ++c) {
    const std::size_t worst =
        std::max(prod_range(dims, 0, c), prod_range(dims, c, N));
    if (worst < best) {
      best = worst;
      s = c;
    }
  }
  const double *X = REAL(tensor_data);
  {
    std::vector<double> P(prod_range(dims, 0, s) * R);
    side_partial(X, f, dims, s, true, P.data());
    finish_group(P, f, dims, 0, s, out);
  }
  {
    std::vector<double> P(prod_range(dims, s, N) * R);
    side_partial(X, f, dims, s, false, P.data());
    finish_group(P, f, dims, s, N, out);
  }
  return wrap(out);
}

// Dimension-tree building blocks for CP-ALS (cp_als). With the modes split
// at s, every mode in [0, s) can be finished from one partial that contracts
// [s, N) -- valid while the factors of [s, N) stay fixed -- and vice versa.
// `s` is the number of modes in the left group (1 <= s < N).
//
// mttkrp_partial_cpp: the partial for the left (left = TRUE, M_L x R) or
// right (M_R x R) group, contracting the other group with its current
// factors.
// [[Rcpp::export]]
NumericMatrix mttkrp_partial_cpp(const NumericVector &tensor_data,
                                 const List &factors, int s, bool left) {
  const std::vector<std::size_t> dims = shape_vec_dec(tensor_data);
  const std::size_t N = dims.size();
  if (s < 1 || static_cast<std::size_t>(s) >= N) {
    stop("s must split the modes into two non-empty groups");
  }
  const Factors f = read_factors(factors, dims, N);
  check_blas_int(f.rank);
  const std::size_t split = static_cast<std::size_t>(s);
  const std::size_t rows =
      left ? prod_range(dims, 0, split) : prod_range(dims, split, N);
  NumericMatrix P(blas_int(rows), static_cast<int>(f.rank));
  if (product_of_dims(dims) == 0 || f.rank == 0) {
    return P;
  }
  side_partial(REAL(tensor_data), f, dims, split, left, REAL(P));
  return P;
}

// mttkrp_finish_cpp: mttkrp(X, factors, mode) from the partial of the group
// that contains `mode` (1-based), using the current factors of that group.
// [[Rcpp::export]]
NumericMatrix mttkrp_finish_cpp(const NumericMatrix &partial,
                                const List &factors, const IntegerVector &dims,
                                int s, int mode) {
  const std::vector<std::size_t> d(dims.begin(), dims.end());
  const std::size_t N = d.size();
  if (s < 1 || static_cast<std::size_t>(s) >= N) {
    stop("s must split the modes into two non-empty groups");
  }
  const std::size_t n = static_cast<std::size_t>(mode - 1);
  if (n >= N) {
    stop("mode is out of bounds for the tensor order");
  }
  const Factors f = read_factors(factors, d, N);
  const std::size_t split = static_cast<std::size_t>(s);
  const bool left = n < split;
  const std::size_t first = left ? 0 : split;
  const std::size_t last = left ? split : N;
  if (static_cast<std::size_t>(partial.nrow()) != prod_range(d, first, last) ||
      static_cast<std::size_t>(partial.ncol()) != f.rank) {
    stop("partial does not match the group of this mode");
  }
  return finish_mode(REAL(partial), f, d, first, last, n);
}
