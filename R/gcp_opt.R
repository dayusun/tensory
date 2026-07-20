#' @include tensor_class.R ktensor_class.R cp_variants.R
NULL

# Loss catalog: elementwise objective f(x, m), derivative df/dm, and the
# factor lower bound the loss requires. `eps` guards logs and divisions.
.gcp_losses <- function(type, eps = 1e-10) {
  switch(type,
    "gaussian" = ,
    "normal" = list(
      f = function(x, m) (m - x)^2,
      g = function(x, m) 2 * (m - x),
      lower = -Inf
    ),
    "poisson" = ,
    "count" = list(
      f = function(x, m) m - x * log(m + eps),
      g = function(x, m) 1 - x / (m + eps),
      lower = 0
    ),
    "poisson-log" = list(
      f = function(x, m) exp(m) - x * m,
      g = function(x, m) exp(m) - x,
      lower = -Inf
    ),
    "bernoulli-odds" = list(
      f = function(x, m) log(m + 1) - x * log(m + eps),
      g = function(x, m) 1 / (m + 1) - x / (m + eps),
      lower = 0
    ),
    "bernoulli-logit" = list(
      f = function(x, m) log1p(exp(m)) - x * m,
      g = function(x, m) stats::plogis(m) - x,
      lower = -Inf
    ),
    "rayleigh" = list(
      f = function(x, m) 2 * log(m + eps) + (pi / 4) * (x / (m + eps))^2,
      g = function(x, m) 2 / (m + eps) - (pi / 2) * x^2 / (m + eps)^3,
      lower = 0
    ),
    "gamma" = list(
      f = function(x, m) x / (m + eps) + log(m + eps),
      g = function(x, m) -x / (m + eps)^2 + 1 / (m + eps),
      lower = 0
    ),
    "huber" = list(
      f = function(x, m) {
        d <- x - m
        ifelse(abs(d) <= 0.25, d^2, 0.5 * abs(d) - 0.0625)
      },
      g = function(x, m) {
        d <- x - m
        ifelse(abs(d) <= 0.25, -2 * d, -0.5 * sign(d))
      },
      lower = -Inf
    ),
    stop(sprintf("Unknown gcp_opt loss type '%s'.", type))
  )
}

#' Generalized CP Decomposition
#'
#' Fits a CP model by minimizing the sum of an arbitrary elementwise loss
#' `f(x, m)` between the data and the model with L-BFGS-B, mirroring the
#' dense/deterministic mode of the MATLAB Tensor Toolbox `gcp_opt`.
#'
#' @param X A Tensor or array-like object.
#' @param R Target CP rank.
#' @param type Loss name: one of `"gaussian"` (alias `"normal"`),
#'   `"poisson"` (alias `"count"`), `"poisson-log"`, `"bernoulli-odds"`,
#'   `"bernoulli-logit"`, `"rayleigh"`, `"gamma"`, `"huber"`. Alternatively a
#'   list with elements `f(x, m)`, `g(x, m)` (the derivative in `m`), and
#'   `lower` (factor lower bound, `-Inf` if unconstrained) for a custom loss.
#' @param init `"random"`, `"nvecs"`, or a list of initial factor matrices.
#' @param maxiters Maximum optimizer iterations (default `500`).
#' @param factr `optim` L-BFGS-B `factr` convergence parameter.
#' @param printitn If positive, print the optimizer trace.
#' @return A list with elements `K` (the fitted `KTensor`) and `objective`
#'   (the final loss value).
#' @references Hong, D., Kolda, T. G., and Duersch, J. A. (2020). Generalized
#'   canonical polyadic tensor decomposition. SIAM Review 62(1).
#' @examples
#' set.seed(1)
#' X <- tensor(array(rpois(24, 3), dim = c(2, 3, 4)))
#' res <- gcp_opt(X, R = 2, type = "poisson", maxiters = 100)
#' @export
gcp_opt <- function(X, R,
                    type = "gaussian",
                    init = "random",
                    maxiters = 500L,
                    factr = 1e7,
                    printitn = 0L) {
  chk <- .cp_check_input(X, R, "gcp_opt")
  X <- .tensor_as_dense(chk$X); R <- chk$R; dims <- chk$dims; N <- chk$N

  loss <- if (is.list(type)) {
    if (!all(c("f", "g", "lower") %in% names(type))) {
      stop("A custom loss must be a list with elements f, g, and lower.")
    }
    type
  } else {
    .gcp_losses(match.arg(type, c("gaussian", "normal", "poisson", "count",
                                  "poisson-log", "bernoulli-odds",
                                  "bernoulli-logit", "rayleigh", "gamma",
                                  "huber")))
  }

  if (is.list(init)) {
    U0 <- lapply(init, as.matrix)
    if (length(U0) != N) stop("init list must have length ndims(X).")
  } else {
    U0 <- .init_cp_factors(X, R, dims, N, init)
    if (is.finite(loss$lower)) {
      U0 <- lapply(U0, function(M) pmax(abs(M), loss$lower + 0.1))
    }
  }

  xvals <- X$data

  fg <- function(v) {
    U <- .vec_to_factors(v, dims, R)
    M <- as.tensor(ktensor(rep(1, R), U))$data
    f <- sum(loss$f(xvals, M))
    D <- Tensor$new(array(loss$g(xvals, M), dim = dims), dims = dims,
                    fast = TRUE)
    grad <- vector("list", N)
    for (n in seq_len(N)) {
      grad[[n]] <- mttkrp(D, U, mode = n)
    }
    list(value = f, gradient = .factors_to_vec(grad))
  }

  res <- stats::optim(
    par = .factors_to_vec(U0),
    fn = function(v) fg(v)$value,
    gr = function(v) fg(v)$gradient,
    method = "L-BFGS-B",
    lower = loss$lower,
    control = list(maxit = as.integer(maxiters), factr = factr,
                   trace = as.integer(printitn > 0))
  )

  U <- .vec_to_factors(res$par, dims, R)
  K <- fixsigns(arrange(ktensor(rep(1, R), U)))
  list(K = K, objective = res$value)
}
