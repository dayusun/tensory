# Shape / mode sweep for the ttm and mttkrp kernels on large tensors.
#
# Usage:
#   Rscript bench/sweep.R <out.csv> [label]
#
# Times ttm (J = 10) and mttkrp (R = 10) in every mode, plus mttkrps, for
# tensors of order 3-6 at roughly 64M elements (512 MB) in several shapes, and
# a size ramp of cubic 3-way tensors. Like bench/kernels.R it loads whichever
# tensory is first on .libPaths() and records a result checksum per case, so
# two builds can be compared with bench/compare.R.
#
# Each case runs after a gc() for 5-15 iterations (at most ~1 s of repeats
# after the first) and records the median and the minimum. For these large,
# memory-bound calls the median is sensitive to garbage collection and page
# faults on fresh allocations; the minimum is the more stable number to
# compare builds on. The whole sweep takes several minutes on reference BLAS.

suppressPackageStartupMessages({
  library(bench)
  library(tensory)
})

args <- commandArgs(trailingOnly = TRUE)
out_file <- if (length(args) >= 1) args[[1]] else "bench-sweep.csv"
label <- if (length(args) >= 2) args[[2]] else "run"

J <- 10L
R <- 10L

shapes <- list(
  # size ramp, cubic 3-way
  c(100, 100, 100), c(200, 200, 200), c(300, 300, 300),
  # ~64M elements, different orders and aspect ratios
  c(400, 400, 400),
  c(100, 400, 1600), c(1600, 400, 100), c(4000, 4000, 4),
  c(90, 90, 90, 90), c(20, 80, 200, 200),
  c(36, 36, 36, 36, 36),
  c(20, 20, 20, 20, 20, 20)
)

checksum <- function(x) {
  if (inherits(x, "Tensor")) x <- x$data
  if (is.list(x)) x <- unlist(lapply(x, checksum))
  signif(sum(abs(as.numeric(x))), 10)
}

time_ms <- function(fn) {
  invisible(gc())
  bm <- bench::mark(fn(), check = FALSE, min_iterations = 5,
                    max_iterations = 15, min_time = 1, filter_gc = FALSE)
  c(median = as.numeric(bm$median), min = as.numeric(bm$min)) * 1e3
}

rows <- list()
record <- function(op, d, mode, fn) {
  res <- fn()
  t <- time_ms(fn)
  M1 <- if (is.na(mode)) NA else prod(d[seq_len(mode - 1)])
  M2 <- if (is.na(mode)) NA else prod(d[-seq_len(mode)])
  row <- data.frame(
    label = label, kernel = op, backend = "compiled",
    shape = paste0(paste(d, collapse = "x"),
                   if (is.na(mode)) "" else paste0(" mode ", mode)),
    order = length(d), mode = mode, M1 = M1, M2 = M2,
    elements = prod(d), median_ms = t[["median"]], min_ms = t[["min"]],
    mem_mb = NA_real_,
    checksum = checksum(res),
    stringsAsFactors = FALSE
  )
  message(sprintf("%-8s %-28s %10.2f ms (min %.2f)", op, row$shape,
                  row$median_ms, row$min_ms))
  rows[[length(rows) + 1L]] <<- row
}

set.seed(20260930)
for (d in shapes) {
  X <- Tensor$new(array(rnorm(prod(d)), dim = d))
  U <- lapply(d, function(n) matrix(rnorm(n * R), n, R))
  for (m in seq_along(d)) {
    A <- matrix(rnorm(J * d[m]), J, d[m])
    record("ttm", d, m, function() ttm(X, A, mode = m))
    record("mttkrp", d, m, function() mttkrp(X, U, mode = m))
  }
  record("mttkrps", d, NA, function() mttkrps(X, U))
  rm(X)
  invisible(gc())
}

out <- do.call(rbind, rows)
utils::write.csv(out, out_file, row.names = FALSE)
message("wrote ", nrow(out), " rows to ", out_file)
