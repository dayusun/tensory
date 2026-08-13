# Identities and Relationships of Tensors

This vignette replicates the identities and relationships shown in the
[MATLAB Tensor Toolbox
examples](https://www.tensortoolbox.org/identities_doc.html),
demonstrating how the mathematical identities hold natively in
`tensory`.

``` r

library(tensory)
#> 
#> Attaching package: 'tensory'
#> The following object is masked from 'package:stats':
#> 
#>     reshape
#> The following object is masked from 'package:utils':
#> 
#>     find
#> The following object is masked from 'package:methods':
#> 
#>     kronecker
#> The following objects are masked from 'package:base':
#> 
#>     %*%, kronecker, scale
set.seed(123) # Setting random seed for reproducibility
```

## N-mode product properties

Create some data.

``` r

Y <- tensor(runif(4 * 3 * 2), c(4, 3, 2))
A <- matrix(runif(3 * 4), nrow=3, ncol=4)
B <- matrix(runif(3 * 3), nrow=3, ncol=3)
```

**Prop 3.4(a):** The order of the multiplication in different modes is
irrelevant.

``` r

X1 <- ttm(ttm(Y, A, mode=1), B, mode=2) # Y x_1 A x_2 B
X2 <- ttm(ttm(Y, B, mode=2), A, mode=1) # Y x_2 B x_1 A
fnorm(X1 - X2) # difference is zero
#> [1] 1.035547e-15
```

## N-mode product and matricization

Generate some data to work with.

``` r

Y <- tensor(runif(5 * 4 * 3), c(5, 4, 3))
A <- matrix(runif(4 * 5), nrow=4, ncol=5)
B <- matrix(runif(3 * 4), nrow=3, ncol=4)
C <- matrix(runif(2 * 3), nrow=2, ncol=3)
U <- list(A, B, C)
```

**Prop. 3.7a:** N-mode multiplication can be expressed in terms of
matricized tensors.

``` r

for (n in 1:Y$ndims()) {
  X <- ttm(Y, U[[n]], mode = n) # X = Y x_n U{n}
  Xn <- U[[n]] %*% as.matrix(tenmat(Y, rdims = n)) # Xn = U{n} * Yn
  print(norm(as.matrix(tenmat(X, rdims = n)) - Xn, type = "F")) # should be zero
}
#> [1] 0
#> [1] 0
#> [1] 0
```

**Prop. 3.7b:** We can do matricizations in various ways and still be
equivalent.

``` r

X <- ttm(Y, U, mode = 1:3) # X = Y x_1 A x_2 B x_3 C

# Kronecker product version
Xm1 <- kronecker(list(B, A)) %*% as.matrix(tenmat(Y, rdims = c(1, 2))) %*% t(C)
Xm2 <- as.matrix(tenmat(X, rdims = c(1, 2)))
norm(Xm1 - Xm2, type = "F") # should be zero
#> [1] 2.860845e-15

Xm1 <- B %*% as.matrix(tenmat(Y, rdims = 2, cdims = c(3, 1))) %*% t(kronecker(list(A, C)))
Xm2 <- as.matrix(tenmat(X, rdims = 2, cdims = c(3, 1)))
norm(Xm1 - Xm2, type = "F") # should be zero
#> [1] 2.7104e-15

# Vectorized
Xm1 <- as.numeric(Y$as_array()) %*% t(kronecker(list(C, B, A)))
Xm2 <- as.numeric(X$as_array())
# Depending on native vector mapping, we compare element-wise
norm(matrix(Xm1, ncol=1) - matrix(Xm2, ncol=1), type = "F") # should be zero
#> [1] 4.772668e-15
```

## Norm of difference between two tensors

**Prop. 3.9:** For tensors X and Y, we have:

``` r

X <- tensor(runif(5 * 4 * 3), c(5, 4, 3))
Y <- tensor(runif(5 * 4 * 3), c(5, 4, 3))

# The following 2 results should be equal
fnorm(X - Y)
#> [1] 2.743582
sqrt(fnorm(X)^2 - 2 * as.numeric(ttt(X, Y, dimsA=1:3, dimsB=1:3)$as_array()) + fnorm(Y)^2)
#> [1] 2.743582
```

This relationship makes it more convenient to compare the norm of the
difference between two different tensor objects, like dense and ktensor.

``` r

Y_k <- ktensor(c(1, 1, 1), list(matrix(runif(5 * 3), 5, 3), matrix(runif(4 * 3), 4, 3), matrix(runif(3 * 3), 3, 3)))
fnorm(X - as.tensor(Y_k))
#> [1] 3.538086
sqrt(fnorm(X)^2 - 2 * as.numeric(ttt(X, as.tensor(Y_k), dimsA=1:3, dimsB=1:3)$as_array()) + fnorm(as.tensor(Y_k))^2)
#> [1] 3.538086
```

## Tucker tensor properties

The properties of the Tucker operator follow directly from the
properties of n-mode multiplication.

``` r

Y <- tensor(1:24, c(4, 3, 2))
A1 <- matrix(1:20, nrow=5, ncol=4)
A2 <- matrix(1:12, nrow=4, ncol=3)
A3 <- matrix(1:6, nrow=3, ncol=2)
A <- list(A1, A2, A3)

B1 <- matrix(1:20, nrow=4, ncol=5)
B2 <- matrix(1:12, nrow=3, ncol=4)
B3 <- matrix(1:6, nrow=2, ncol=3)
B <- list(B1, B2, B3)
```

**Proposition 4.2a**

``` r

X <- ttm(ttensor(Y, A), B, mode = 1:3)
AB <- mapply(function(b, a) b %*% a, B, A, SIMPLIFY = FALSE)
Y_AB <- ttensor(Y, AB)
fnorm(as.tensor(X) - as.tensor(Y_AB)) # should be zero
#> [1] 0
```

**Proposition 4.2b**

``` r

X <- ttensor(Y, A)
Apinv <- lapply(A, MASS::ginv)
Y2 <- ttm(as.tensor(X), Apinv, mode = 1:3)
fnorm(Y - as.tensor(Y2)) # should be zero
#> [1] 3.872012e-13
```

**Proposition 4.2c**

``` r

Q1 <- qr.Q(qr(matrix(runif(5 * 4), nrow=5, ncol=4)))
Q2 <- qr.Q(qr(matrix(runif(4 * 3), nrow=4, ncol=3)))
Q3 <- qr.Q(qr(matrix(runif(3 * 2), nrow=3, ncol=2)))
Q <- list(Q1, Q2, Q3)

X <- ttensor(Y, Q)
Qt <- lapply(Q, t)
Y2 <- ttm(as.tensor(X), Qt, mode = 1:3)
fnorm(Y - as.tensor(Y2)) # should be zero
#> [1] 2.789997e-14
```

## Tucker operator and matricized tensors

**Proposition 4.3a**

``` r

X <- ttensor(Y, A)
for (n in 1:Y$ndims()) {
  rdims <- n
  cdims <- setdiff(1:Y$ndims(), rdims)
  Xn <- A[[n]] %*% as.matrix(tenmat(Y, rdims = rdims, cdims = cdims)) %*% t(kronecker(list(A[[cdims[2]]], A[[cdims[1]]])))
  print(norm(as.matrix(tenmat(as.tensor(X), rdims = rdims, cdims = cdims)) - Xn, type = "F")) # should be zero
}
#> [1] 0
#> [1] 0
#> [1] 0
```

## Orthogonalization of Tucker factors

**Proposition 4.4**

``` r

Y <- tensor(1:24, c(4, 3, 2))
A1 <- matrix(runif(5 * 4), nrow=5, ncol=4)
A2 <- matrix(runif(4 * 3), nrow=4, ncol=3)
A3 <- matrix(runif(3 * 2), nrow=3, ncol=2)
A <- list(A1, A2, A3)
X <- ttensor(Y, A)

R1 <- qr.R(qr(A1))
R2 <- qr.R(qr(A2))
R3 <- qr.R(qr(A3))
R <- list(R1, R2, R3)

Z <- ttm(Y, R, mode = 1:3)
abs(fnorm(as.tensor(X)) - fnorm(as.tensor(Z))) # should be near zero
#> [1] 5.684342e-14
```

## Kruskal operator properties

**Proposition 5.2**

``` r

A1 <- matrix(1:10, nrow=5, ncol=2)
A2 <- matrix(1:8, nrow=4, ncol=2)
A3 <- matrix(1:6, nrow=3, ncol=2)
K <- ktensor(c(1, 1), list(A1, A2, A3))

B1 <- matrix(1:20, nrow=4, ncol=5)
B2 <- matrix(1:12, nrow=3, ncol=4)
B3 <- matrix(1:6, nrow=2, ncol=3)
B <- list(B1, B2, B3)

X <- ttm(K, B, mode = 1:3)
Y <- ktensor(c(1, 1), list(B1 %*% A1, B2 %*% A2, B3 %*% A3))
fnorm(as.tensor(X) - as.tensor(Y)) # should be zero
#> [1] 0
```

**Proposition 5.3a (second part)**

``` r

A <- list(A1, A2, A3)
X <- ktensor(c(1, 1), A)

Z <- t(as.numeric(as.tensor(X)$as_array()))
Xn <- t(matrix(1, nrow = length(X$lambda), ncol = 1)) %*% t(khatri_rao(A, reverse = TRUE))
norm(Z - Xn, type = "F") # should be zero
#> [1] 0
```

**Proposition 5.3b**

``` r

for (n in 1:X$ndims()) {
  rdims <- n
  cdims <- setdiff(1:X$ndims(), rdims)
  Xn <- A[[rdims]] %*% t(khatri_rao(A[cdims], reverse = TRUE))
  Z <- as.matrix(tenmat(as.tensor(X), rdims = rdims, cdims = cdims))
  print(norm(Z - Xn, type = "F")) # should be zero
}
#> [1] 0
#> [1] 0
#> [1] 0
```

**Proposition 5.3a (first part)**

``` r

for (n in 1:X$ndims()) {
  cdims <- n
  rdims <- setdiff(1:X$ndims(), cdims)
  Xn <- khatri_rao(A[rdims], reverse = TRUE) %*% t(A[[cdims]])
  Z <- as.matrix(tenmat(as.tensor(X), rdims = rdims, cdims = cdims))
  print(norm(Z - Xn, type = "F")) # should be zero
}
#> [1] 0
#> [1] 0
#> [1] 0
```

## Norm of Kruskal operator

The norm of a ktensor has a special form because it can be reduced to
summing the entries of the Hadamard product of N matrices of size R x R.

**Proposition 5.4**

``` r

M <- matrix(1, nrow = ncol(A[[1]]), ncol = ncol(A[[1]]))
for (i in 1:length(A)) {
  M <- M * (t(A[[i]]) %*% A[[i]])
}
abs(fnorm(as.tensor(X)) - sqrt(sum(M))) # should be near zero
#> [1] 0
```

## Inner product of Kruskal operator with a tensor

The inner product of a ktensor with a tensor yields:

**Proposition 5.5**

``` r

X_dense <- tensor(1:60, c(5, 4, 3))
A1 <- matrix(1:10, nrow=5, ncol=2)
A2 <- matrix(2:9, nrow=4, ncol=2)
A3 <- matrix(3:8, nrow=3, ncol=2)
A <- list(A1, A2, A3)
K <- ktensor(c(1, 1), A)

v <- khatri_rao(A, reverse = TRUE) %*% matrix(1, nrow=ncol(A[[1]]), ncol=1)
v1 <- as.numeric(t(as.numeric(X_dense$as_array())) %*% v)

# In tensory, inner product is fully contracted ttt
v2 <- as.numeric(ttt(X_dense, as.tensor(K), dimsA=1:3, dimsB=1:3)$as_array())

abs(v1 - v2) # should be zero
#> [1] 0
```
