#include "xtensor-r/rarray.hpp"

using namespace Rcpp;

// tensor addition with xtensor
// [[Rcpp::export]]
NumericVector tensor_add(NumericVector x, NumericVector y) {
    xt::rarray<double> x_ = as<xt::rarray<double>>(x);
    xt::rarray<double> y_ = as<xt::rarray<double>>(y);
    xt::rarray<double> z = x_ + y_;
    
    return wrap(z);
}