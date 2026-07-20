#include "xtensor-r/rarray.hpp"
#include "xtensor/containers/xarray.hpp"
#include <Rcpp.h>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace Rcpp;

namespace {

std::vector<std::size_t> shape_vec(const xt::rarray<double>& tensor_data) {
  return std::vector<std::size_t>(tensor_data.shape().begin(), tensor_data.shape().end());
}

std::vector<std::size_t> column_major_strides(const std::vector<std::size_t>& dims) {
  std::vector<std::size_t> strides(dims.size(), 1);
  for (std::size_t i = 1; i < dims.size(); ++i) {
    strides[i] = strides[i - 1] * dims[i - 1];
  }
  return strides;
}

std::size_t product_of_dims(const std::vector<std::size_t>& dims) {
  return std::accumulate(dims.begin(), dims.end(), static_cast<std::size_t>(1),
                         std::multiplies<std::size_t>());
}

std::vector<std::size_t> decode_linear_index(std::size_t idx,
                                             const std::vector<std::size_t>& dims,
                                             const std::vector<std::size_t>& strides) {
  std::vector<std::size_t> coord(dims.size(), 0);
  for (std::size_t i = dims.size(); i-- > 0;) {
    coord[i] = (idx / strides[i]) % dims[i];
  }
  return coord;
}

std::size_t encode_linear_index(const std::vector<std::size_t>& coord,
                                const std::vector<std::size_t>& strides) {
  std::size_t idx = 0;
  for (std::size_t i = 0; i < coord.size(); ++i) {
    idx += coord[i] * strides[i];
  }
  return idx;
}

List contract_impl(const xt::rarray<double>& tensor_data, int mode1, int mode2) {
  std::vector<std::size_t> dims = shape_vec(tensor_data);
  const std::size_t i = static_cast<std::size_t>(mode1 - 1);
  const std::size_t j = static_cast<std::size_t>(mode2 - 1);
  auto strides = column_major_strides(dims);

  std::vector<std::size_t> remdims;
  std::vector<std::size_t> new_dims;
  for (std::size_t k = 0; k < dims.size(); ++k) {
    if (k == i || k == j) continue;
    remdims.push_back(k);
    new_dims.push_back(dims[k]);
  }

  const std::size_t n = dims[i];
  const std::size_t out_size = new_dims.empty() ? 1 : product_of_dims(new_dims);
  NumericVector out(out_size, 0.0);
  const double* data_ptr = tensor_data.data();

  for (std::size_t out_idx = 0; out_idx < out_size; ++out_idx) {
    std::size_t tmp = out_idx;
    std::vector<std::size_t> coord(dims.size(), 0);

    for (std::size_t p = 0; p < remdims.size(); ++p) {
      const std::size_t dim_val = new_dims[p];
      coord[remdims[p]] = tmp % dim_val;
      tmp /= dim_val;
    }

    double accum = 0.0;
    for (std::size_t diag = 0; diag < n; ++diag) {
      coord[i] = diag;
      coord[j] = diag;
      accum += data_ptr[encode_linear_index(coord, strides)];
    }
    out[out_idx] = accum;
  }

  IntegerVector out_dims(new_dims.begin(), new_dims.end());
  return List::create(_["data"] = out, _["dims"] = out_dims);
}

} // namespace

// [[Rcpp::export]]
NumericMatrix mttkrp_cpp(const xt::rarray<double>& tensor_data, const List& factors, int mode) {
  const std::vector<std::size_t> dims = shape_vec(tensor_data);
  const std::size_t n_dim = dims.size();
  const std::size_t skip = static_cast<std::size_t>(mode - 1);
  const double* data_ptr = tensor_data.data();
  const auto strides = column_major_strides(dims);
  const std::size_t total = product_of_dims(dims);

  if (n_dim < 2) {
    stop("mttkrp is invalid for tensors with fewer than 2 dimensions.");
  }
  if (skip >= n_dim) {
    stop("mode is out of bounds for the tensor order");
  }
  if (factors.size() != static_cast<R_xlen_t>(n_dim)) {
    stop("factors must have the same length as the tensor order.");
  }

  std::vector<NumericMatrix> mats(n_dim);
  int rank = -1;
  for (std::size_t k = 0; k < n_dim; ++k) {
    if (k == skip) continue;
    mats[k] = as<NumericMatrix>(factors[k]);
    if (mats[k].nrow() != static_cast<int>(dims[k])) {
      stop("factor matrix row dimension does not match tensor dimension");
    }
    if (rank < 0) {
      rank = mats[k].ncol();
    } else if (mats[k].ncol() != rank) {
      stop("all factor matrices must have the same number of columns");
    }
  }

  NumericMatrix out(dims[skip], rank);

  for (std::size_t lin = 0; lin < total; ++lin) {
    const double x = data_ptr[lin];
    if (x == 0.0) continue;

    std::vector<std::size_t> coord = decode_linear_index(lin, dims, strides);
    const std::size_t row = coord[skip];

    for (int r = 0; r < rank; ++r) {
      double prod = x;
      for (std::size_t k = 0; k < n_dim; ++k) {
        if (k == skip) continue;
        prod *= mats[k](coord[k], r);
      }
      out(row, r) += prod;
    }
  }

  return out;
}

// [[Rcpp::export]]
List mttkrps_cpp(const xt::rarray<double>& tensor_data, const List& factors) {
  std::vector<std::size_t> dims = shape_vec(tensor_data);
  const std::size_t n_dim = dims.size();
  List out(n_dim);
  for (std::size_t mode = 0; mode < n_dim; ++mode) {
    out[mode] = mttkrp_cpp(tensor_data, factors, static_cast<int>(mode + 1));
  }
  return out;
}

// [[Rcpp::export]]
NumericMatrix fibers_cpp(const xt::rarray<double>& tensor_data, int mode, const IntegerMatrix& midx) {
  const std::vector<std::size_t> dims = shape_vec(tensor_data);
  const std::size_t n_dim = dims.size();
  const std::size_t axis = static_cast<std::size_t>(mode - 1);
  const auto strides = column_major_strides(dims);
  const double* data_ptr = tensor_data.data();

  if (axis >= n_dim) {
    stop("mode must be a valid tensor mode");
  }
  if (midx.ncol() != static_cast<int>(n_dim - 1)) {
    stop("midx must have ndims(x) - 1 columns");
  }

  const std::size_t nk = dims[axis];
  const int samples = midx.nrow();
  NumericMatrix out(nk, samples);

  std::vector<std::size_t> other_modes;
  for (std::size_t k = 0; k < n_dim; ++k) {
    if (k != axis) other_modes.push_back(k);
  }

  for (int s = 0; s < samples; ++s) {
    std::size_t base = 0;
    for (std::size_t c = 0; c < other_modes.size(); ++c) {
      const int sub = midx(s, static_cast<int>(c));
      if (sub < 1 || sub > static_cast<int>(dims[other_modes[c]])) {
        stop("fiber subscripts are out of bounds");
      }
      base += static_cast<std::size_t>(sub - 1) * strides[other_modes[c]];
    }

    for (std::size_t k = 0; k < nk; ++k) {
      out(k, s) = data_ptr[base + k * strides[axis]];
    }
  }

  return out;
}

// [[Rcpp::export]]
List contract_cpp(const xt::rarray<double>& tensor_data, int mode1, int mode2) {
  return contract_impl(tensor_data, mode1, mode2);
}

// [[Rcpp::export]]
NumericVector mask_cpp(const xt::rarray<double>& tensor_data, const xt::rarray<double>& mask_data) {
  const std::vector<std::size_t> dims = shape_vec(tensor_data);
  const std::vector<std::size_t> mask_dims = shape_vec(mask_data);
  const std::size_t total = product_of_dims(mask_dims);
  const double* data_ptr = tensor_data.data();
  const double* mask_ptr = mask_data.data();
  std::vector<double> out;
  out.reserve(total);

  if (mask_dims == dims) {
    // Same shape: mask and tensor share a linear index.
    for (std::size_t i = 0; i < total; ++i) {
      if (mask_ptr[i] != 0.0) {
        out.push_back(data_ptr[i]);
      }
    }
    return wrap(out);
  }

  // Smaller mask: a mask subscript must be re-encoded with the tensor's
  // strides, matching the R fallback's find() + sub2ind path.
  if (mask_dims.size() != dims.size()) {
    stop("mask must have the same number of modes as the data tensor");
  }
  for (std::size_t k = 0; k < dims.size(); ++k) {
    if (mask_dims[k] > dims[k]) {
      stop("Mask cannot be bigger than the data tensor.");
    }
  }

  const auto mask_strides = column_major_strides(mask_dims);
  const auto strides = column_major_strides(dims);
  for (std::size_t i = 0; i < total; ++i) {
    if (mask_ptr[i] != 0.0) {
      const std::vector<std::size_t> coord = decode_linear_index(i, mask_dims, mask_strides);
      out.push_back(data_ptr[encode_linear_index(coord, strides)]);
    }
  }

  return wrap(out);
}

// [[Rcpp::export]]
bool issymmetric_cpp(const xt::rarray<double>& tensor_data, const List& grps) {
  const std::vector<std::size_t> dims = shape_vec(tensor_data);
  const auto strides = column_major_strides(dims);
  const std::size_t total = product_of_dims(dims);
  const double* data_ptr = tensor_data.data();

  for (R_xlen_t g = 0; g < grps.size(); ++g) {
    IntegerVector grp = grps[g];
    if (grp.size() <= 1) continue;

    for (int i = 1; i < grp.size(); ++i) {
      if (dims[grp[i] - 1] != dims[grp[0] - 1]) {
        return false;
      }
    }

    for (std::size_t lin = 0; lin < total; ++lin) {
      std::vector<std::size_t> coord = decode_linear_index(lin, dims, strides);
      std::vector<std::size_t> grp_coords;
      grp_coords.reserve(grp.size());
      for (int idx : grp) {
        grp_coords.push_back(coord[static_cast<std::size_t>(idx - 1)]);
      }
      std::sort(grp_coords.begin(), grp_coords.end());
      for (R_xlen_t p = 0; p < grp.size(); ++p) {
        coord[static_cast<std::size_t>(grp[p] - 1)] = grp_coords[static_cast<std::size_t>(p)];
      }
      const std::size_t ref = encode_linear_index(coord, strides);
      if (data_ptr[lin] != data_ptr[ref]) {
        return false;
      }
    }
  }

  return true;
}
