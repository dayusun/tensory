// Shape helpers for R arrays handed to the compiled kernels.
//
// Kernels take tensor storage as Rcpp::NumericVector, which aliases R's own
// REALSXP buffer (no copy; integer/logical arrays are coerced to double once,
// as R's as.double() would). The shape is the "dim" attribute, or the length
// for a plain vector. Storage is column-major with mode 1 fastest.
#ifndef TENSORY_TENSOR_ARRAY_H
#define TENSORY_TENSOR_ARRAY_H

#include <Rcpp.h>
#include <climits>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

namespace tensory {

inline std::vector<std::size_t> array_dims(const Rcpp::NumericVector &x) {
  SEXP dim = Rf_getAttrib(x, R_DimSymbol);
  if (Rf_isNull(dim)) {
    return std::vector<std::size_t>(1, static_cast<std::size_t>(Rf_xlength(x)));
  }
  Rcpp::IntegerVector d(dim);
  return std::vector<std::size_t>(d.begin(), d.end());
}

inline std::vector<std::size_t>
column_major_strides(const std::vector<std::size_t> &dims) {
  std::vector<std::size_t> strides(dims.size(), 1);
  for (std::size_t i = 1; i < dims.size(); ++i) {
    strides[i] = strides[i - 1] * dims[i - 1];
  }
  return strides;
}

inline std::size_t product_of_dims(const std::vector<std::size_t> &dims) {
  return std::accumulate(dims.begin(), dims.end(), static_cast<std::size_t>(1),
                         std::multiplies<std::size_t>());
}

// Uninitialised double array with the given "dim" attribute. Every element
// must be written by the caller.
inline Rcpp::NumericVector alloc_array(const std::vector<std::size_t> &dims) {
  Rcpp::IntegerVector dim(dims.size());
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] > static_cast<std::size_t>(INT_MAX)) {
      Rcpp::stop("Tensor extent exceeds R's dimension limit");
    }
    dim[i] = static_cast<int>(dims[i]);
  }
  Rcpp::NumericVector out(Rcpp::no_init(
      static_cast<R_xlen_t>(product_of_dims(dims))));
  out.attr("dim") = dim;
  return out;
}

} // namespace tensory

#endif
