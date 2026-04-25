suppressPackageStartupMessages({
  devtools::load_all(".", quiet = TRUE)
  library(bench)
})

set.seed(1)

# ---- MTTKRP: BLAS-backed vs cell-wise C++ vs R fallback ----------------------
bench_mttkrp <- function(dims, R) {
  X <- tensor(array(rnorm(prod(dims)), dim = dims))
  U <- lapply(dims, function(d) matrix(rnorm(d * R), d, R))

  ref_R <- function() {
    N <- length(dims); n <- 1L
    others <- setdiff(seq_len(N), n)
    V <- matrix(0, dims[n], R)
    for (r in seq_len(R)) {
      vs <- lapply(others, function(k) U[[k]][, r])
      V[, r] <- as.vector(ttv(X, vs, mode = others)$as_array())
    }
    V
  }

  bench::mark(
    blas      = mttkrp_blas_cpp(X$data, U, 1L),
    cellwise  = mttkrp_cpp(X$data, U, 1L),
    r_ttv     = ref_R(),
    iterations = 5,
    check = function(a, b) max(abs(a - b)) < 1e-8,
    filter_gc = FALSE
  )[, c("expression", "min", "median", "itr/sec", "mem_alloc")]
}

cat("\n=== MTTKRP: 30x30x30, R=10 ===\n")
print(bench_mttkrp(c(30, 30, 30), 10))

cat("\n=== MTTKRP: 50x50x50, R=20 ===\n")
print(bench_mttkrp(c(50, 50, 50), 20))

cat("\n=== MTTKRP: 20x20x20x20, R=10 ===\n")
print(bench_mttkrp(c(20, 20, 20, 20), 10))

# ---- Khatri-Rao: C++ pair vs R rep path --------------------------------------
bench_kr <- function(I, K, R) {
  A <- matrix(rnorm(I * R), I, R)
  B <- matrix(rnorm(K * R), K, R)
  r_rep <- function() {
    ret <- matrix(0, I * K, R)
    ret[] <- B[rep(1:K, times = I), ] * A[rep(1:I, each = K), ]
    ret
  }
  bench::mark(
    cpp = khatri_rao_pair_cpp(A, B, reverse = FALSE),
    r   = r_rep(),
    iterations = 10,
    check = function(a, b) max(abs(a - b)) < 1e-10,
    filter_gc = FALSE
  )[, c("expression", "min", "median", "itr/sec", "mem_alloc")]
}

cat("\n=== Khatri-Rao pair: 200x200, R=20 ===\n")
print(bench_kr(200, 200, 20))

cat("\n=== Khatri-Rao pair: 1000x500, R=10 ===\n")
print(bench_kr(1000, 500, 10))

# ---- CP / Tucker wall-clock --------------------------------------------------
cat("\n=== cp_als wall-clock ===\n")
X <- tensor(array(rnorm(40 * 40 * 40), dim = c(40, 40, 40)))
print(bench::mark(
  cp_als = cp_als(X, R = 5L, maxiters = 30L, tol = 1e-8, init = "random"),
  iterations = 3, filter_gc = FALSE
)[, c("expression", "min", "median", "itr/sec", "mem_alloc")])

cat("\n=== tucker_als wall-clock ===\n")
print(bench::mark(
  tucker_als = tucker_als(X, ranks = c(5, 5, 5), maxiters = 30L,
                          tol = 1e-8, init = "nvecs"),
  iterations = 3, filter_gc = FALSE
)[, c("expression", "min", "median", "itr/sec", "mem_alloc")])

cat("\n=== hosvd wall-clock ===\n")
print(bench::mark(
  hosvd_seq = hosvd(X, ranks = c(5, 5, 5), sequential = TRUE),
  hosvd_par = hosvd(X, ranks = c(5, 5, 5), sequential = FALSE),
  iterations = 5, filter_gc = FALSE, check = FALSE
)[, c("expression", "min", "median", "itr/sec", "mem_alloc")])
