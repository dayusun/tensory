# Compare two bench/kernels.R result files.
#
# Usage:
#   Rscript bench/compare.R <old.csv> <new.csv>
#
# Prints old/new median times per case, the speedup (old / new, so > 1 means
# the new build is faster) and whether the result checksums agree. Exits with
# status 1 if any checksum differs beyond a relative 1e-8, so it can gate a
# refactor that is meant to be behavior-preserving.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: Rscript bench/compare.R <old.csv> <new.csv>")
}

old <- utils::read.csv(args[[1]], stringsAsFactors = FALSE)
new <- utils::read.csv(args[[2]], stringsAsFactors = FALSE)
key <- c("kernel", "shape", "backend")

m <- merge(old, new, by = key, suffixes = c(".old", ".new"), sort = FALSE)
m$speedup <- m$median_ms.old / m$median_ms.new
m$mem_ratio <- m$mem_mb.old / pmax(m$mem_mb.new, 1e-9)
rel_diff <- abs(m$checksum.old - m$checksum.new) /
  pmax(abs(m$checksum.old), 1e-300)
m$same <- rel_diff <= 1e-8

tab <- data.frame(
  kernel = m$kernel,
  shape = m$shape,
  backend = m$backend,
  old_ms = round(m$median_ms.old, 3),
  new_ms = round(m$median_ms.new, 3),
  speedup = round(m$speedup, 2),
  old_mb = round(m$mem_mb.old, 2),
  new_mb = round(m$mem_mb.new, 2),
  same_result = m$same
)
print(tab, row.names = FALSE, right = FALSE)

missing <- setdiff(
  do.call(paste, old[key]), do.call(paste, new[key])
)
if (length(missing)) {
  message("cases only in ", args[[1]], ": ", paste(missing, collapse = "; "))
}

if (!all(m$same)) {
  message("checksum mismatch in ", sum(!m$same), " case(s)")
  quit(status = 1)
}
