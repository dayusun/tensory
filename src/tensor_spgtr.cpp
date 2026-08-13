// Kernels for spgtr(): the two hot spots of the sparse penalized generalized
// tensor regression fit. Everything else in the algorithm rides on ttm() and
// the R-level GLM, so it is not duplicated here.
//
//   spgtr_mode_covs_cpp  mode-wise marginal covariances Sigma_k, one dsyrk
//                        per observation instead of m array permutations.
//   spgtr_slpg_cpp       sequential linearized proximal gradient on the
//                        Stiefel manifold: the envelope objective, its
//                        gradient, the row-wise L2,1 prox and the polar
//                        retraction, all without returning to R per iteration.
//
// Both have pure-R fallbacks in R/spgtr.R; keep the two paths identical.

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstddef>
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
void F77_NAME(dsyrk)(const char *uplo, const char *trans, const int *n,
                     const int *k, const double *alpha, const double *a,
                     const int *lda, const double *beta, double *c,
                     const int *ldc FCONE FCONE);
void F77_NAME(dsyev)(const char *jobz, const char *uplo, const int *n,
                     double *a, const int *lda, double *w, double *work,
                     const int *lwork, int *info FCONE FCONE);
}

using namespace Rcpp;

namespace {

void check_blas_int_sp(std::size_t value) {
  if (value > static_cast<std::size_t>(INT_MAX)) {
    Rcpp::stop("Extents exceed BLAS 32-bit integer limit");
  }
}

// C <- alpha * op(A) op(B) + beta * C, all column major.
void gemm(char ta, char tb, int m, int n, int k, double alpha, const double *A,
          int lda, const double *B, int ldb, double beta, double *C, int ldc) {
  F77_NAME(dgemm)(&ta, &tb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C,
                  &ldc FCONE FCONE);
}

// Eigendecomposition of a symmetric matrix. `a` is overwritten with the
// eigenvectors, `w` receives the eigenvalues in ascending order.
void sym_eig(std::vector<double> &a, int n, std::vector<double> &w) {
  int lwork = -1;
  int info = 0;
  double wkopt = 0.0;
  F77_NAME(dsyev)("V", "U", &n, a.data(), &n, w.data(), &wkopt, &lwork,
                  &info FCONE FCONE);
  lwork = static_cast<int>(wkopt);
  if (lwork < 1) lwork = 1;
  std::vector<double> work(static_cast<std::size_t>(lwork));
  F77_NAME(dsyev)("V", "U", &n, a.data(), &n, w.data(), work.data(), &lwork,
                  &info FCONE FCONE);
  if (info != 0) Rcpp::stop("Symmetric eigendecomposition failed");
}

void symmetrize(std::vector<double> &a, int n) {
  for (int j = 0; j < n; ++j) {
    for (int i = j + 1; i < n; ++i) {
      const double v = 0.5 * (a[i + n * j] + a[j + n * i]);
      a[i + n * j] = v;
      a[j + n * i] = v;
    }
  }
}

// Symmetric matrix power with eigenvalues floored at ridge * max eigenvalue.
std::vector<double> sym_pow(const double *M, int n, double pw, double ridge) {
  std::vector<double> a(M, M + static_cast<std::size_t>(n) * n);
  std::vector<double> w(static_cast<std::size_t>(n));
  symmetrize(a, n);
  sym_eig(a, n, w);
  const double floor_at = ridge * std::max(w[n - 1], 1.0);
  std::vector<double> scaled(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    const double f = std::pow(std::max(w[j], floor_at), pw);
    for (int i = 0; i < n; ++i) scaled[i + n * j] = a[i + n * j] * f;
  }
  std::vector<double> out(static_cast<std::size_t>(n) * n);
  gemm('N', 'T', n, n, n, 1.0, scaled.data(), n, a.data(), n, 0.0, out.data(),
       n);
  return out;
}

// log|A| over the positive eigenvalues only, matching the R fallback, and the
// pseudo-inverse of the same matrix (both need the eigendecomposition).
double logdet_and_pinv(const double *A, int n, std::vector<double> &pinv) {
  std::vector<double> a(A, A + static_cast<std::size_t>(n) * n);
  std::vector<double> w(static_cast<std::size_t>(n));
  symmetrize(a, n);
  sym_eig(a, n, w);
  const double tol = std::max(std::abs(w[n - 1]), 1.0) * 1e-12;
  double ld = 0.0;
  std::vector<double> scaled(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    if (w[j] > 0.0) ld += std::log(w[j]);
    const double f = (w[j] > tol) ? 1.0 / w[j] : 0.0;
    for (int i = 0; i < n; ++i) scaled[i + n * j] = a[i + n * j] * f;
  }
  pinv.assign(static_cast<std::size_t>(n) * n, 0.0);
  gemm('N', 'T', n, n, n, 1.0, scaled.data(), n, a.data(), n, 0.0, pinv.data(),
       n);
  return ld;
}

double frob(const std::vector<double> &x) {
  double s = 0.0;
  for (double v : x) s += v * v;
  return std::sqrt(s);
}

} // namespace

//' @keywords internal
//' @noRd
// [[Rcpp::export]]
List spgtr_mode_covs_cpp(const NumericMatrix &Xt, const IntegerVector &dims) {
  const std::size_t P = static_cast<std::size_t>(Xt.nrow());
  const std::size_t n = static_cast<std::size_t>(Xt.ncol());
  const std::size_t m = static_cast<std::size_t>(dims.size());

  std::size_t total = 1;
  for (std::size_t k = 0; k < m; ++k) {
    total *= static_cast<std::size_t>(dims[k]);
  }
  if (total != P) Rcpp::stop("prod(dims) must equal nrow(Xt)");
  check_blas_int_sp(P);
  check_blas_int_sp(n);

  List out(m);
  std::vector<double> buf;
  for (std::size_t k = 0; k < m; ++k) {
    const std::size_t pk = static_cast<std::size_t>(dims[k]);
    std::size_t L = 1;
    for (std::size_t j = 0; j < k; ++j) L *= static_cast<std::size_t>(dims[j]);
    const std::size_t R = P / (L * pk);

    const int npk = static_cast<int>(pk);
    NumericMatrix S(npk, npk);
    double *Sp = S.begin();
    const double *Xall = Xt.begin();
    const double one = 1.0;
    // Accumulate X_{i(k)} X_{i(k)}' over observations. The mode-k unfolding of
    // one observation is a contiguous p_k x R matrix when L == 1 and the
    // transpose of a contiguous L x p_k matrix when R == 1; otherwise the
    // fibers are gathered once into a p_k x (L*R) buffer.
    if (L != 1 && R != 1) buf.assign(pk * L * R, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double *x = Xall + i * P;
      if (L == 1) {
        const int kk = static_cast<int>(R);
        F77_NAME(dsyrk)
        ("U", "N", &npk, &kk, &one, x, &npk, &one, Sp, &npk FCONE FCONE);
      } else if (R == 1) {
        const int kk = static_cast<int>(L);
        const int ld = static_cast<int>(L);
        F77_NAME(dsyrk)
        ("U", "T", &npk, &kk, &one, x, &ld, &one, Sp, &npk FCONE FCONE);
      } else {
        for (std::size_t r = 0; r < R; ++r) {
          for (std::size_t a = 0; a < pk; ++a) {
            const double *src = x + L * (a + pk * r);
            double *dst = &buf[pk * L * r + a];
            for (std::size_t l = 0; l < L; ++l) dst[pk * l] = src[l];
          }
        }
        const int kk = static_cast<int>(L * R);
        F77_NAME(dsyrk)
        ("U", "N", &npk, &kk, &one, buf.data(), &npk, &one, Sp,
         &npk FCONE FCONE);
      }
    }

    const double scale = static_cast<double>(pk) /
                         (static_cast<double>(n) * static_cast<double>(P));
    for (std::size_t j = 0; j < pk; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        const double v = Sp[i + pk * j] * scale;
        Sp[i + pk * j] = v;
        Sp[j + pk * i] = v;
      }
    }
    out[k] = S;
  }
  return out;
}

//' @keywords internal
//' @noRd
// [[Rcpp::export]]
NumericMatrix spgtr_slpg_cpp(const NumericMatrix &G0, const NumericMatrix &M,
                             const NumericMatrix &MUinv,
                             const NumericVector &gam, int maxit, double tol,
                             double ridge = 1e-10) {
  const int p = G0.nrow();
  const int d = G0.ncol();
  if (M.nrow() != p || MUinv.nrow() != p) {
    Rcpp::stop("M and MUinv must be p x p with p = nrow(G0)");
  }
  if (gam.size() != p) Rcpp::stop("gam must have one entry per row of G0");
  check_blas_int_sp(static_cast<std::size_t>(p));

  const std::size_t pd = static_cast<std::size_t>(p) * d;
  const std::size_t dd = static_cast<std::size_t>(d) * d;
  std::vector<double> G(G0.begin(), G0.end());
  std::vector<double> Gprev(pd), grad(pd), Gr(pd), Grprev(pd), Dv(pd), Yv(pd);
  std::vector<double> MG(pd), MUG(pd), A(dd), B(dd), Ai(dd), Bi(dd), lam(dd);
  std::vector<double> tmp(pd), GG(dd);

  bool penalized = false;
  for (int i = 0; i < p; ++i) {
    if (gam[i] > 0.0) penalized = true;
  }

  // f(G) = log|G'MG| + log|G'(M+U)^-1 G| and its Euclidean gradient.
  auto objective = [&](const std::vector<double> &Gc) {
    gemm('N', 'N', p, d, p, 1.0, M.begin(), p, Gc.data(), p, 0.0, MG.data(), p);
    gemm('N', 'N', p, d, p, 1.0, MUinv.begin(), p, Gc.data(), p, 0.0,
         MUG.data(), p);
    gemm('T', 'N', d, d, p, 1.0, Gc.data(), p, MG.data(), p, 0.0, A.data(), d);
    gemm('T', 'N', d, d, p, 1.0, Gc.data(), p, MUG.data(), p, 0.0, B.data(), d);
    const double f = logdet_and_pinv(A.data(), d, Ai) +
                     logdet_and_pinv(B.data(), d, Bi);
    gemm('N', 'N', p, d, d, 2.0, MG.data(), p, Ai.data(), d, 0.0, grad.data(),
         p);
    gemm('N', 'N', p, d, d, 2.0, MUG.data(), p, Bi.data(), d, 1.0, grad.data(),
         p);
    return f;
  };

  // Tangent-space projection, including the L2,1 multiplier term.
  auto project_grad = [&](const std::vector<double> &Gc) {
    gemm('T', 'N', d, d, p, 1.0, Gc.data(), p, grad.data(), p, 0.0, lam.data(),
         d);
    symmetrize(lam, d);
    if (penalized) {
      for (int i = 0; i < p; ++i) {
        double rn = 0.0;
        for (int j = 0; j < d; ++j) rn += Gc[i + p * j] * Gc[i + p * j];
        const double s = std::sqrt(gam[i]) / (1e-14 + std::sqrt(std::sqrt(rn)));
        for (int j = 0; j < d; ++j) tmp[i + p * j] = Gc[i + p * j] * s;
      }
      gemm('T', 'N', d, d, p, 1.0, tmp.data(), p, tmp.data(), p, 1.0,
           lam.data(), d);
    }
    Gr = grad;
    gemm('N', 'N', p, d, d, -1.0, Gc.data(), p, lam.data(), d, 1.0, Gr.data(),
         p);
  };

  objective(G);
  const double beta = 0.01 * frob(grad);
  project_grad(G);
  double step = 1.0 / (frob(G) / 2.0 * frob(Gr) + p * beta);
  if (!std::isfinite(step) || step <= 0.0) step = 1e-3;
  int stalled = 0;

  for (int it = 0; it < maxit; ++it) {
    Gprev = G;
    Grprev = Gr;

    // Proximal step: row-wise group soft threshold.
    for (int i = 0; i < p; ++i) {
      double rn = 0.0;
      for (int j = 0; j < d; ++j) {
        const double v = G[i + p * j] - step * Gr[i + p * j];
        tmp[i + p * j] = v;
        rn += v * v;
      }
      rn = std::sqrt(rn);
      const double thr = gam[i] * step;
      const double shrink =
          (thr > 0.0) ? 1.0 - thr / std::max(rn, thr) : 1.0;
      for (int j = 0; j < d; ++j) G[i + p * j] = shrink * tmp[i + p * j];
    }

    // Polar retraction G (G'G)^-1/2: a right multiplication, so rows zeroed by
    // the prox stay zero. Rank-deficient iterates cannot be retracted, so back
    // off the step instead of refilling the rows.
    gemm('T', 'N', d, d, p, 1.0, G.data(), p, G.data(), p, 0.0, GG.data(), d);
    std::vector<double> ev(GG);
    std::vector<double> evals(static_cast<std::size_t>(d));
    symmetrize(ev, d);
    sym_eig(ev, d, evals);
    if (!(evals[0] > 1e-10)) {
      G = Gprev;
      step *= 0.5;
      if (++stalled > 5) break;
      continue;
    }
    stalled = 0;
    std::vector<double> half = sym_pow(GG.data(), d, -0.5, ridge);
    gemm('N', 'N', p, d, d, 1.0, G.data(), p, half.data(), d, 0.0, tmp.data(),
         p);
    G.swap(tmp);

    objective(G);
    project_grad(G);

    for (std::size_t i = 0; i < pd; ++i) {
      Dv[i] = G[i] - Gprev[i];
      Yv[i] = Gr[i] - Grprev[i];
    }
    const double dnorm = frob(Dv);
    if (dnorm / step < tol) break;

    double sdy = 0.0;
    for (std::size_t i = 0; i < pd; ++i) sdy += Yv[i] * Dv[i];
    const double ynorm = frob(Yv);
    double next = (it % 2 == 1) ? dnorm * dnorm / sdy : sdy / (ynorm * ynorm);
    next = std::abs(next);
    step = (std::isfinite(next)) ? std::min(std::max(next, 1e-12), 1000.0)
                                 : 1e-3;
  }

  NumericMatrix out(p, d);
  std::copy(G.begin(), G.end(), out.begin());
  return out;
}
