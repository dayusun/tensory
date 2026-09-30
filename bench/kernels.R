# Kernel benchmark harness for tensory's compiled kernels.
#
# Usage:
#   Rscript bench/kernels.R <out.csv> [label]
#
# Times every compiled kernel (through its public R entry point where one
# exists, and the *_cpp symbol directly otherwise) plus the pure-R reference
# path for the kernels that keep one as a named internal. Loads whichever
# tensory is first on .libPaths(), so point R_LIBS at the build under test:
#
#   R_LIBS=/path/to/lib_old Rscript bench/kernels.R old.csv old
#   R_LIBS=/path/to/lib_new Rscript bench/kernels.R new.csv new
#   Rscript bench/compare.R old.csv new.csv
#
# Each row also records a checksum of the kernel's result so compare.R can
# confirm two builds computed the same thing, not just how fast.
#
# Build with R's normal optimisation flags before timing:
# `R CMD INSTALL` does that; `pkgbuild::compile_dll()` defaults to -O0.

suppressPackageStartupMessages({
  library(bench)
  library(tensory)
})

args <- commandArgs(trailingOnly = TRUE)
out_file <- if (length(args) >= 1) args[[1]] else "bench-kernels.csv"
label <- if (length(args) >= 2) args[[2]] else "run"

ns <- asNamespace("tensory")
cpp <- function(name) get(name, envir = ns)

set.seed(20260930)
rarr <- function(...) array(rnorm(prod(c(...))), dim = c(...))
rmat <- function(r, c) matrix(rnorm(r * c), r, c)

checksum <- function(x) {
  if (inherits(x, "Tensor")) x <- x$data
  if (is.list(x)) x <- unlist(lapply(x, checksum))
  signif(sum(abs(as.numeric(x))), 10)
}

cases <- list()
add <- function(kernel, shape, backend, fn) {
  cases[[length(cases) + 1L]] <<- list(
    kernel = kernel, shape = shape, backend = backend, fn = fn
  )
}

# ---- ttm -------------------------------------------------------------------
for (spec in list(
  list(dims = c(10, 10, 10), J = 5, modes = 1:3),
  list(dims = c(200, 200, 200), J = 20, modes = 1:3),
  list(dims = c(40, 40, 40, 40), J = 10, modes = c(1, 2, 4))
)) {
  X <- Tensor$new(rarr(spec$dims))
  for (m in spec$modes) {
    local({
      A <- rmat(spec$J, spec$dims[m])
      X <- X
      m <- m
      shape <- sprintf("%s mode %d J=%d", paste(spec$dims, collapse = "x"), m, spec$J)
      add("ttm", shape, "compiled", function() ttm(X, A, mode = m))
      add("ttm", shape, "R", function() cpp(".ttm_matrix_base")(X, A, m, FALSE))
    })
  }
}

local({
  X <- Tensor$new(rarr(200, 200, 200))
  At <- rmat(200, 20)
  add("ttm", "200x200x200 mode 2 J=20 transpose", "compiled",
      function() ttm(X, At, mode = 2, transpose = TRUE))
})

local({
  X <- Tensor$new(rarr(100, 100, 100))
  mats <- list(rmat(10, 100), rmat(10, 100), rmat(10, 100))
  add("ttm_multiple", "100x100x100 all modes J=10", "compiled",
      function() ttm(X, mats, mode = 1:3))
})

# ---- mttkrp ----------------------------------------------------------------
local({
  dims <- c(150, 150, 150)
  X <- Tensor$new(rarr(dims))
  U <- lapply(dims, function(d) rmat(d, 10))
  for (m in 1:3) {
    local({
      m <- m
      shape <- sprintf("150x150x150 R=10 mode %d", m)
      add("mttkrp", shape, "compiled", function() mttkrp(X, U, mode = m))
      add("mttkrp_cpp", shape, "compiled",
          function() cpp("mttkrp_cpp")(X$data, U, as.integer(m)))
    })
  }
  add("mttkrps", "150x150x150 R=10", "compiled", function() mttkrps(X, U))
  add("mttkrps_cpp", "150x150x150 R=10", "compiled",
      function() cpp("mttkrps_cpp")(X$data, U))
})

local({
  dims <- c(30, 30, 30, 30)
  X <- Tensor$new(rarr(dims))
  U <- lapply(dims, function(d) rmat(d, 16))
  add("mttkrps", "30x30x30x30 R=16", "compiled", function() mttkrps(X, U))
})

# ---- misc dense kernels ----------------------------------------------------
local({
  X <- Tensor$new(rarr(200, 200, 200))
  midx <- cbind(sample.int(200, 2000, TRUE), sample.int(200, 2000, TRUE))
  add("fibers", "200x200x200 mode 2 n=2000", "compiled",
      function() cpp("fibers_cpp")(X$data, 2L, midx))
})

local({
  X <- Tensor$new(rarr(100, 100, 50))
  add("contract", "100x100x50 modes 1,2", "compiled",
      function() contract(X, 1, 2))
})

local({
  X <- Tensor$new(rarr(200, 200, 200))
  W <- array(as.numeric(runif(200^3) < 0.1), dim = c(200, 200, 200))
  add("mask", "200x200x200 10% same-shape", "compiled",
      function() cpp("mask_cpp")(X$data, W))
  Ws <- array(as.numeric(runif(100^3) < 0.1), dim = c(100, 100, 100))
  add("mask", "200^3 tensor, 100^3 mask", "compiled",
      function() cpp("mask_cpp")(X$data, Ws))
})

local({
  S <- symmetrize(Tensor$new(rarr(60, 60, 60)))
  add("issymmetric", "60x60x60 symmetric", "compiled",
      function() issymmetric(S))
})

local({
  A <- rmat(2000, 20)
  B <- rmat(500, 20)
  add("khatri_rao_pair", "2000x20 (.) 500x20", "compiled",
      function() cpp("khatri_rao_pair_cpp")(A, B, FALSE))
})

# ---- spgtr -----------------------------------------------------------------
local({
  dims <- c(20L, 20L, 20L)
  n <- 200L
  Xt <- matrix(rnorm(prod(dims) * n), prod(dims), n)
  add("spgtr_mode_covs", "20x20x20 n=200", "compiled",
      function() cpp(".spgtr_mode_covs")(Xt, dims, n))
  add("spgtr_mode_covs", "20x20x20 n=200", "R",
      function() cpp(".tepls_mode_covs")(t(Xt), dims, n))
})

# ---- run -------------------------------------------------------------------
rows <- lapply(cases, function(cs) {
  res <- cs$fn()
  bm <- bench::mark(cs$fn(), check = FALSE, min_iterations = 5,
                    min_time = 0.5, filter_gc = FALSE)
  row <- data.frame(
    label = label,
    kernel = cs$kernel,
    shape = cs$shape,
    backend = cs$backend,
    median_ms = as.numeric(bm$median) * 1e3,
    min_ms = as.numeric(bm$min) * 1e3,
    mem_mb = as.numeric(bm$mem_alloc) / 2^20,
    n_itr = bm$n_itr,
    checksum = checksum(res),
    stringsAsFactors = FALSE
  )
  message(sprintf("%-16s %-36s %-8s %9.3f ms", row$kernel, row$shape,
                  row$backend, row$median_ms))
  row
})

out <- do.call(rbind, rows)
utils::write.csv(out, out_file, row.names = FALSE)
message("wrote ", nrow(out), " rows to ", out_file)
