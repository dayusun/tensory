#' @include tensor_class.R ktensor_class.R
NULL

#' Random Dense Tensor
#'
#' Creates a tensor with entries drawn independently from the uniform
#' distribution on `[0, 1]`, mirroring the MATLAB Tensor Toolbox `tenrand`.
#'
#' @param dims Integer vector of dimensions.
#' @return A `Tensor` with uniform random entries.
#' @examples
#' X <- tenrand(c(2, 3, 4))
#' @export
tenrand <- function(dims) {
  dims <- as.integer(dims)
  if (length(dims) == 0 || anyNA(dims) || any(dims < 0)) {
    stop("dims must be a vector of non-negative integers.")
  }
  Tensor$new(array(stats::runif(prod(dims)), dim = dims), dims = dims, fast = TRUE)
}

#' Identity Tensor
#'
#' Creates the order-`m` identity tensor `E` of size `n`, satisfying
#' `ttsv(E, x, -1) == x` for any vector `x` with `norm(x) == 1`. Mirrors the
#' MATLAB Tensor Toolbox `teneye`; as there, `m` must be even.
#'
#' @param m Tensor order (even positive integer).
#' @param n Size of each mode.
#' @return A symmetric `Tensor` of order `m` and size `n`.
#' @examples
#' E <- teneye(4, 3)
#' x <- rnorm(3); x <- x / sqrt(sum(x^2))
#' max(abs(ttsv(E, x, -1) - x)) < 1e-12
#' @export
teneye <- function(m, n) {
  m <- as.integer(m)
  n <- as.integer(n)
  if (length(m) != 1 || m < 2 || m %% 2L != 0L) {
    stop("m must be a single even integer >= 2.")
  }
  if (length(n) != 1 || n < 1) {
    stop("n must be a single positive integer.")
  }

  # Build T(i1,...,im) = prod_k delta(i_{2k-1}, i_{2k}), then symmetrize.
  # The symmetrization average over all mode permutations yields the identity
  # tensor: contracting m-1 modes with a unit vector returns that vector.
  dims <- rep(n, m)
  data <- array(0, dim = dims)
  half <- m %/% 2L
  grid <- as.matrix(expand.grid(rep(list(seq_len(n)), half)))
  full_idx <- grid[, rep(seq_len(half), each = 2L), drop = FALSE]
  data[full_idx] <- 1
  symmetrize(Tensor$new(data, dims = dims, fast = TRUE))
}

#' Diagonal Tensor
#'
#' Creates a tensor with the vector `v` on its superdiagonal and zeros
#' elsewhere, mirroring the MATLAB Tensor Toolbox `tendiag`.
#'
#' @param v Numeric vector of diagonal values.
#' @param dims Optional integer vector of dimensions. Defaults to a square
#'   matrix-shaped tensor `c(length(v), length(v))`. Each entry must be at
#'   least `length(v)`.
#' @return A `Tensor` with `v` on the superdiagonal.
#' @examples
#' D <- tendiag(1:3, c(3, 3, 3))
#' @export
tendiag <- function(v, dims = NULL) {
  v <- as.double(v)
  k <- length(v)
  if (k == 0) {
    stop("v must be a non-empty numeric vector.")
  }
  if (is.null(dims)) {
    dims <- c(k, k)
  }
  dims <- as.integer(dims)
  if (length(dims) < 2 || any(dims < k)) {
    stop("dims must have length >= 2 with every entry >= length(v).")
  }
  data <- array(0, dim = dims)
  idx <- matrix(rep(seq_len(k), times = length(dims)), nrow = k)
  data[idx] <- v
  Tensor$new(data, dims = dims, fast = TRUE)
}

#' Export Tensor Data to a Text File
#'
#' Writes a `Tensor`, matrix, `KTensor`, or `Sptensor` to a plain-text file in
#' the MATLAB Tensor Toolbox `exportdata` format, so files are interchangeable
#' with MATLAB's `importdata`/`exportdata` pair.
#'
#' @details
#' Format by type (values in column-major / first-index-fastest order):
#' \itemize{
#'   \item `tensor`: `tensor`, order, sizes, one value per line.
#'   \item `matrix`: `matrix`, `2`, sizes, one row of values per line.
#'   \item `ktensor`: `ktensor`, order, sizes, rank, lambda line, then each
#'     factor as a `matrix` block.
#'   \item `sptensor`: `sptensor`, order, sizes, number of nonzeros, then one
#'     `i1 ... iN value` line per nonzero.
#' }
#'
#' @param x Object to export.
#' @param fname Path of the file to create.
#' @param fmt `sprintf` format used for numeric values (default `"%.16e"`).
#' @return Invisibly, `fname`.
#' @seealso [import_data()]
#' @export
export_data <- function(x, fname, fmt = "%.16e") {
  con <- file(fname, open = "wt")
  on.exit(close(con))

  emit <- function(...) writeLines(paste0(...), con)
  num_line <- function(v) paste(sprintf(fmt, v), collapse = " ")

  write_matrix <- function(M) {
    emit("matrix")
    emit("2")
    emit(paste(dim(M), collapse = " "))
    for (i in seq_len(nrow(M))) {
      emit(num_line(M[i, ]))
    }
  }

  if (inherits(x, "Sptensor")) {
    emit("sptensor")
    emit(length(x$dims))
    emit(paste(x$dims, collapse = " "))
    emit(nrow(x$subs))
    for (i in seq_len(nrow(x$subs))) {
      emit(paste(paste(x$subs[i, ], collapse = " "), sprintf(fmt, x$vals[i])))
    }
  } else if (inherits(x, "KTensor")) {
    emit("ktensor")
    emit(x$ndims())
    emit(paste(x$dim(), collapse = " "))
    emit(length(x$lambda))
    emit(num_line(x$lambda))
    for (U in x$U) {
      write_matrix(U)
    }
  } else if (inherits(x, "Tensor")) {
    x <- .tensor_as_dense(x)
    emit("tensor")
    emit(x$ndims())
    emit(paste(x$dim(), collapse = " "))
    vals <- as.vector(x$data)
    writeLines(sprintf(fmt, vals), con)
  } else if (is.matrix(x)) {
    write_matrix(x)
  } else {
    stop("export_data supports Tensor, Sptensor, KTensor, and matrix objects.")
  }

  invisible(fname)
}

#' Import Tensor Data from a Text File
#'
#' Reads a file written by [export_data()] (or MATLAB Tensor Toolbox
#' `exportdata`) and reconstructs the corresponding object.
#'
#' @param fname Path of the file to read.
#' @return A `Tensor`, matrix, `KTensor`, or `Sptensor` depending on the file
#'   header.
#' @seealso [export_data()]
#' @export
import_data <- function(fname) {
  lines <- readLines(fname)
  lines <- lines[nzchar(trimws(lines))]
  pos <- 1L

  take <- function() {
    line <- lines[pos]
    pos <<- pos + 1L
    line
  }
  scan_nums <- function(line) as.numeric(strsplit(trimws(line), "\\s+")[[1]])

  read_block <- function() {
    type <- trimws(take())
    if (type == "matrix") {
      order <- as.integer(take())
      if (order != 2L) stop("matrix blocks must have order 2.")
      sz <- as.integer(scan_nums(take()))
      rows <- lapply(seq_len(sz[1]), function(i) scan_nums(take()))
      M <- do.call(rbind, rows)
      if (!all(dim(M) == sz)) stop("matrix block has inconsistent dimensions.")
      M
    } else if (type == "tensor") {
      order <- as.integer(take())
      sz <- as.integer(scan_nums(take()))
      if (length(sz) != order) stop("tensor block has inconsistent order.")
      n <- prod(sz)
      vals <- numeric(0)
      while (length(vals) < n) {
        vals <- c(vals, scan_nums(take()))
      }
      if (length(vals) != n) stop("tensor block has the wrong number of values.")
      Tensor$new(array(vals, dim = sz), dims = sz, fast = TRUE)
    } else if (type == "sptensor") {
      order <- as.integer(take())
      sz <- as.integer(scan_nums(take()))
      if (length(sz) != order) stop("sptensor block has inconsistent order.")
      nnz <- as.integer(take())
      subs <- matrix(0L, nnz, order)
      vals <- numeric(nnz)
      for (i in seq_len(nnz)) {
        entry <- scan_nums(take())
        subs[i, ] <- as.integer(entry[seq_len(order)])
        vals[i] <- entry[order + 1L]
      }
      sptensor(subs, vals, sz)
    } else if (type == "ktensor") {
      order <- as.integer(take())
      sz <- as.integer(scan_nums(take()))
      if (length(sz) != order) stop("ktensor block has inconsistent order.")
      r <- as.integer(take())
      lambda <- scan_nums(take())
      if (length(lambda) != r) stop("ktensor lambda has the wrong length.")
      U <- lapply(seq_len(order), function(i) read_block())
      for (i in seq_len(order)) {
        if (!identical(dim(U[[i]]), c(sz[i], r))) {
          stop("ktensor factor block has inconsistent dimensions.")
        }
      }
      ktensor(lambda, U)
    } else {
      stop(sprintf("Unknown data type '%s' in file.", type))
    }
  }

  read_block()
}
