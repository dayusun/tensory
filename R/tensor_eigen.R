#' @include tensor_class.R tensor_dense_methods.R tensor_constructors.R
NULL

# Symmetrized outer product a (o) b = a b' + b a' (Kolda & Mayo notation).
.sym_outer <- function(a, b) {
  tcrossprod(a, b) + tcrossprod(b, a)
}

.check_eigen_input <- function(A, fname) {
  A <- .tensor_as_dense(A)
  dims <- A$dim()
  if (length(dims) < 2 || !all(dims == dims[1])) {
    stop(sprintf("%s requires a tensor with all modes of the same size.", fname))
  }
  if (!issymmetric(A, tol = 1e-8)) {
    stop(sprintf("%s requires a symmetric tensor.", fname))
  }
  A
}

#' Shifted Symmetric Higher-Order Power Method (SS-HOPM)
#'
#' Computes a Z-eigenpair `A x^(m-1) = lambda x`, `||x|| = 1`, of a symmetric
#' tensor by the shifted symmetric higher-order power method with the adaptive
#' shift of Kolda & Mayo, mirroring the MATLAB Tensor Toolbox `eig_sshopm`.
#'
#' @param A A symmetric Tensor (all modes the same size).
#' @param shift `"adaptive"` (default) or a fixed numeric shift.
#' @param maximize Logical; `TRUE` (default) seeks local maxima of `A x^m`
#'   (`beta = 1`), `FALSE` seeks local minima (`beta = -1`).
#' @param start Optional starting vector; random normal if omitted.
#' @param tol Convergence tolerance on the eigenvalue change (default
#'   `1e-10`).
#' @param maxiters Maximum iterations (default `500`).
#' @param tau Positive-definiteness threshold for the adaptive shift
#'   (default `1e-6`).
#' @return A list with elements `lambda`, `x`, `converged`, `iterations`, and
#'   `lambda_trace`.
#' @references Kolda, T. G. and Mayo, J. R. (2011). Shifted power method for
#'   computing tensor eigenpairs. SIAM J. Matrix Anal. Appl. 32(4);
#'   Kolda, T. G. and Mayo, J. R. (2014). An adaptive shifted power method for
#'   computing generalized tensor eigenpairs. SIAM J. Matrix Anal. Appl. 35(4).
#' @examples
#' set.seed(1)
#' A <- symmetrize(tensor(array(rnorm(81), dim = c(3, 3, 3, 3))))
#' res <- eig_sshopm(A)
#' @export
eig_sshopm <- function(A,
                       shift = "adaptive",
                       maximize = TRUE,
                       start = NULL,
                       tol = 1e-10,
                       maxiters = 500L,
                       tau = 1e-6) {
  A <- .check_eigen_input(A, "eig_sshopm")
  n <- A$dim()[1]
  m <- A$ndims()
  beta <- if (isTRUE(maximize)) 1 else -1
  adaptive <- is.character(shift)

  x <- if (is.null(start)) stats::rnorm(n) else as.double(start)
  x <- x / sqrt(sum(x^2))

  lambda <- ttsv(A, x, 0)
  trace <- numeric(0)
  converged <- FALSE
  iter <- 0L

  for (iter in seq_len(maxiters)) {
    Axm1 <- as.numeric(ttsv(A, x, -1))

    if (adaptive) {
      H <- m * (m - 1) * ttsv(A, x, -2)
      alpha <- beta * max(0, (tau - min(eigen(beta * H, symmetric = TRUE,
                                              only.values = TRUE)$values)) / m)
    } else {
      alpha <- as.double(shift)
    }

    xnew <- beta * (Axm1 + alpha * x)
    nrm <- sqrt(sum(xnew^2))
    if (nrm == 0) {
      break
    }
    x <- xnew / nrm

    lambda_new <- ttsv(A, x, 0)
    trace <- c(trace, lambda_new)
    if (abs(lambda_new - lambda) < tol) {
      lambda <- lambda_new
      converged <- TRUE
      break
    }
    lambda <- lambda_new
  }

  list(lambda = lambda, x = x, converged = converged,
       iterations = iter, lambda_trace = trace)
}

#' Generalized Eigenproblem Adaptive Power Method (GEAP)
#'
#' Computes a generalized tensor eigenpair `A x^(m-1) = lambda B x^(m-1)`,
#' `||x|| = 1`, for symmetric `A` and symmetric positive definite `B` using
#' the GEAP method of Kolda & Mayo (Algorithm 1), mirroring the MATLAB Tensor
#' Toolbox `eig_geap`. With `B = teneye(m, n)` this reduces to [eig_sshopm()].
#'
#' @param A A symmetric Tensor (all modes the same size).
#' @param B A symmetric positive definite Tensor of the same size and order.
#' @param maximize Logical; `TRUE` (default) seeks local maxima (`beta = 1`),
#'   `FALSE` seeks local minima (`beta = -1`).
#' @param start Optional starting vector; random normal if omitted.
#' @param tol Convergence tolerance on the eigenvalue change (default
#'   `1e-10`).
#' @param maxiters Maximum iterations (default `500`).
#' @param tau Positive-definiteness threshold for the adaptive shift
#'   (default `1e-6`).
#' @return A list with elements `lambda`, `x`, `converged`, `iterations`, and
#'   `lambda_trace`.
#' @references Kolda, T. G. and Mayo, J. R. (2014). An adaptive shifted power
#'   method for computing generalized tensor eigenpairs. SIAM J. Matrix Anal.
#'   Appl. 35(4), 1563-1581.
#' @examples
#' set.seed(1)
#' A <- symmetrize(tensor(array(rnorm(81), dim = c(3, 3, 3, 3))))
#' res <- eig_geap(A, teneye(4, 3))
#' @export
eig_geap <- function(A, B,
                     maximize = TRUE,
                     start = NULL,
                     tol = 1e-10,
                     maxiters = 500L,
                     tau = 1e-6) {
  A <- .check_eigen_input(A, "eig_geap")
  B <- .check_eigen_input(B, "eig_geap")
  if (!identical(A$dim(), B$dim())) {
    stop("A and B must have the same dimensions.")
  }
  n <- A$dim()[1]
  m <- A$ndims()
  beta <- if (isTRUE(maximize)) 1 else -1

  x <- if (is.null(start)) stats::rnorm(n) else as.double(start)
  x <- x / sqrt(sum(x^2))

  lambda <- NA_real_
  trace <- numeric(0)
  converged <- FALSE
  iter <- 0L

  for (iter in seq_len(maxiters)) {
    Axm2 <- ttsv(A, x, -2)
    Bxm2 <- ttsv(B, x, -2)
    Axm1 <- as.numeric(ttsv(A, x, -1))
    Bxm1 <- as.numeric(ttsv(B, x, -1))
    Axm <- ttsv(A, x, 0)
    Bxm <- ttsv(B, x, 0)

    if (Bxm <= 0) {
      stop("B must be positive definite (B x^m > 0 for the current iterate).")
    }

    lambda_new <- Axm / Bxm

    # Hessian of f(x) = (A x^m / B x^m) ||x||^m on the unit sphere,
    # eq. (3.3) of Kolda & Mayo (2014).
    H <- (m^2 * Axm / Bxm^3) * .sym_outer(Bxm1, Bxm1) +
      (m / Bxm) * ((m - 1) * Axm2 +
                     Axm * (diag(n) + (m - 2) * tcrossprod(x)) +
                     m * .sym_outer(Axm1, x)) -
      (m / Bxm^2) * ((m - 1) * Axm * Bxm2 +
                       m * .sym_outer(Axm1, Bxm1) +
                       m * Axm * .sym_outer(x, Bxm1))

    alpha <- beta * max(0, (tau - min(eigen(beta * H, symmetric = TRUE,
                                            only.values = TRUE)$values)) / m)

    xnew <- beta * (Axm1 - lambda_new * Bxm1 + (alpha + lambda_new) * Bxm * x)
    nrm <- sqrt(sum(xnew^2))
    if (nrm == 0) {
      break
    }
    x <- xnew / nrm

    trace <- c(trace, lambda_new)
    if (!is.na(lambda) && abs(lambda_new - lambda) < tol) {
      lambda <- lambda_new
      converged <- TRUE
      break
    }
    lambda <- lambda_new
  }

  # Final eigenvalue at the last iterate.
  lambda <- ttsv(A, x, 0) / ttsv(B, x, 0)

  list(lambda = lambda, x = x, converged = converged,
       iterations = iter, lambda_trace = trace)
}
