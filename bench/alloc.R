# R-side allocation audit for the dense Tensor API.
#
# Usage:
#   Rscript bench/alloc.R [out.csv]
#
# For each operation, measures the bytes R allocates (bench::mark's
# mem_alloc, which counts every R vector allocation including those made by
# compiled code through Rcpp) and divides by the size of the result's data.
# An operation that only allocates its result scores ~1; each extra full copy
# of a tensor-sized object adds ~1 (relative to the input when the result is
# smaller). `excess_in` = (allocated - result) / input size is the number of
# avoidable tensor-sized copies, so anything clearly above 0 is overhead.
# Needs R built with memory profiling (the default for CRAN/Debian builds).

suppressPackageStartupMessages({
  library(bench)
  library(tensory)
})

args <- commandArgs(trailingOnly = TRUE)
out_file <- if (length(args) >= 1) args[[1]] else NULL

set.seed(1)
d <- c(60L, 50L, 40L)
a <- array(rnorm(prod(d)), dim = d)
X <- Tensor$new(a)
Y <- Tensor$new(array(rnorm(prod(d)), dim = d))
v <- as.vector(a)
A2 <- matrix(rnorm(10 * d[2]), 10, d[2])
U <- lapply(d, function(n) matrix(rnorm(n * 5), n, 5))
in_bytes <- 8 * prod(d)
C <- Tensor$new(array(rnorm(50 * 50 * 40), c(50, 50, 40)))
S <- Tensor$new(array(rnorm(40^3), c(40, 40, 40)))
P <- Tensor$new(array(1, c(20, 20)))
W <- Tensor$new(array(as.double(runif(prod(d)) < 0.1), dim = d))
K <- ktensor(rep(1, 5), U)
TT <- ttensor(Tensor$new(array(rnorm(5^3), c(5, 5, 5))),
              lapply(d, function(n) matrix(rnorm(n * 5), n, 5)))

data_bytes <- function(res) {
  if (inherits(res, "Tensor")) res <- res$data
  if (inherits(res, "Tenmat")) res <- res$data
  if (is.list(res)) return(sum(vapply(res, data_bytes, numeric(1))))
  if (is.numeric(res) || is.logical(res)) return(8 * length(res))
  NA_real_
}

ops <- list(
  "Tensor$new(array)"          = function() Tensor$new(a),
  "Tensor$new(vector, dims)"   = function() Tensor$new(v, d),
  "tensor(array)"              = function() tensor(a),
  "as.tensor(array)"           = function() as.tensor(a),
  "clone_tensor"               = function() X$clone_tensor(),
  "X + Y"                      = function() X + Y,
  "X * Y"                      = function() X * Y,
  "X * 2"                      = function() X * 2,
  "X + 1"                      = function() X + 1,
  "-X"                         = function() -X,
  "X^2"                        = function() X^2,
  "abs(X) (Math)"              = function() abs(X),
  "sum(X) (Summary)"           = function() sum(X),
  "X == Y"                     = function() X == Y,
  "fnorm"                      = function() fnorm(X),
  "innerprod"                  = function() innerprod(X, Y),
  "ttm mode 1"                 = function() ttm(X, matrix(rnorm(10 * d[1]), 10), mode = 1),
  "ttm mode 2"                 = function() ttm(X, A2, mode = 2),
  "ttm list (3 modes)"         = function() ttm(X, lapply(d, function(n) matrix(1, 5, n)), mode = 1:3),
  "ttv mode 2"                 = function() ttv(X, rnorm(d[2]), mode = 2),
  "ttv all modes"              = function() ttv(X, lapply(d, rnorm), mode = 1:3),
  "mttkrp mode 2"              = function() mttkrp(X, U, mode = 2),
  "mttkrps"                    = function() mttkrps(X, U),
  "permute"                    = function() permute(X, c(3, 1, 2)),
  "reshape"                    = function() reshape(X, c(d[1] * d[2], d[3])),
  "squeeze"                    = function() squeeze(X),
  "vec"                        = function() vec(X),
  "unfold mode 2"              = function() unfold(X, rdims = 2),
  "tenmat mode 2"              = function() tenmat(X, rdims = 2),
  "collapse mode 3"            = function() collapse(X, 3),
  "t_scale"                    = function() t_scale(X, rnorm(d[2]), 2),
  "contract 1,2"               = function() contract(C, 1, 2),
  "full"                       = function() full(X),
  "as.double"                  = function() as.double(X),
  "as_array"                   = function() X$as_array(),
  "isequal"                    = function() isequal(X, Y),
  "nnz"                        = function() nnz(X),
  "symmetrize (40^3)"          = function() symmetrize(S),
  "nvecs mode 2"               = function() nvecs(X, 2, 3),
  "ttt outer (small)"          = function() ttt(P, P),
  "ttt contract mode 3"        = function() ttt(X, Y, 3, 3),
  "find"                       = function() find(W),
  "mask"                       = function() mask(X, W),
  "X$sum(dims = 2)"            = function() X$sum(2),
  "as.tensor(KTensor)"         = function() as.tensor(K),
  "as.tensor(TTensor)"         = function() as.tensor(TT),
  "cp_als R=5, 5 iters"        = function() cp_als(X, R = 5L, init = U, maxiters = 5L, tol = 0, fixsigns = FALSE),
  "tucker_als (5,5,5), 5 iters" = function() tucker_als(X, c(5, 5, 5), maxiters = 5L, tol = 0),
  "hosvd ranks (5,5,5)"        = function() hosvd(X, ranks = c(5, 5, 5))
)

rows <- lapply(names(ops), function(nm) {
  fn <- ops[[nm]]
  res <- fn()
  bm <- bench::mark(fn(), iterations = 3, check = FALSE, filter_gc = FALSE)
  alloc <- as.numeric(bm$mem_alloc)
  out <- data_bytes(res)
  data.frame(op = nm, alloc_mb = round(alloc / 2^20, 2),
             result_mb = round(out / 2^20, 2),
             excess_in = round((alloc - out) / in_bytes, 2),
             median_ms = round(as.numeric(bm$median) * 1e3, 2),
             stringsAsFactors = FALSE)
})
tab <- do.call(rbind, rows)
cat(sprintf("input tensor: %s, %.2f MB\n", paste(d, collapse = "x"), in_bytes / 2^20))
print(tab, row.names = FALSE, right = FALSE)
if (!is.null(out_file)) utils::write.csv(tab, out_file, row.names = FALSE)
