#' @keywords internal
"_PACKAGE"

#' Tensory: Modern Tensor Operations for R
#'
#' The \code{tensory} package provides a modern implementation of tensor operations
#' for R, built on top of the high-performance \code{xtensor} C++ library. It offers
#' a comprehensive tensor class system with MATLAB Tensor Toolbox compatibility,
#' enabling efficient numerical computing with multi-dimensional arrays.
#'
#' @section Key Features:
#' \itemize{
#'   \item R6-based Tensor class for object-oriented tensor operations
#'   \item High-performance C++ backend via xtensor integration
#'   \item MATLAB Tensor Toolbox compatibility for familiar syntax
#'   \item Support for multi-dimensional arrays with broadcasting
#'   \item Element-wise operations and mathematical functions
#'   \item Memory-efficient operations with in-place modifications
#' }
#'
#' @section Main Classes:
#' \itemize{
#'   \item \code{\link{Tensor}}: The core R6 class for tensor operations
#' }
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{tensor}}: Create tensor objects from R data structures
#'   \item \code{\link{zeros}}: Create tensors filled with zeros
#'   \item \code{\link{ones}}: Create tensors filled with ones
#'   \item \code{\link{ttm}}: Tensor times matrix operation
#' }
#'
#' @section Performance:
#' The package leverages the \code{xtensor} C++ library for high-performance
#' numerical computing. Operations are implemented in C++ for speed while
#' maintaining R-friendly interfaces.
#'
#' @section Getting Started:
#' \preformatted{
#' # Create a tensor from a matrix
#' t <- tensor(matrix(1:6, nrow=2, ncol=3))
#' print(t)
#' 
#' # Create a 3D tensor
#' t3d <- tensor(1:24, c(2, 3, 4))
#' 
#' # Mathematical operations
#' result <- t3d$clone_tensor()$add(t3d)$multiply(2)
#' }
#'
#' @author Your Name
#' @name tensory-package
#' @aliases tensory
#' @useDynLib tensory, .registration = TRUE
#' @import R6
#' @importFrom Rcpp sourceCpp
#' @description
#' `tensory` is an experimental R package to implement tensor operations and functions based on `xtensor`.
NULL
