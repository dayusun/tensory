#' @description
#' `tensory` is an expermental R package to implement tensor operations and functions based on `xtensor`.
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @rawNamespace if (getRversion() < "4.3.0") importFrom("S7", "@")
#' @useDynLib tensory, .registration = TRUE
# enable usage of <S7_object>@name in package code
## usethis namespace: end
NULL

.onLoad <- function(...) {
  S7::methods_register()
}

.onAttach <- function(libname, pkgname) {
  if (getRversion() < "4.3.0")
    require(S7)
}
