---
trigger: always_on
---

# Rcpp Development Standards

- Always include `<Rcpp.h>` at the top of .cpp files.
- Use `using namespace Rcpp;` to keep code clean.
- Ensure all loops in C++ use `size_t` for indexing to match R's vector sizes.
- If using Armadillo, include `<RcppArmadillo.h>` and add `// [[Rcpp::depends(RcppArmadillo)]]`.
- Clean temporary files after each task
