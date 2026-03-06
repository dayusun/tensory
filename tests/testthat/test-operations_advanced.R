library(tensory)

test_that("khatri_rao works correctly", {
    A <- matrix(1:4, nrow = 2, ncol = 2) # [1, 3; 2, 4]
    B <- matrix(5:8, nrow = 2, ncol = 2) # [5, 7; 6, 8]

    # A Khatri-Rao B (reverse=FALSE) => kron(A, B)
    # kron([1;2], [5;6]) = [5, 6, 10, 12]^T
    expected <- matrix(c(5, 6, 10, 12, 21, 24, 28, 32), nrow = 4, ncol = 2)
    expect_equal(khatri_rao(A, B), expected)

    # reverse=TRUE => kron(B, A)
    # kron([5;6], [1;2]) = [5, 10, 6, 12]^T
    expected_rev <- matrix(c(5, 10, 6, 12, 21, 28, 24, 32), nrow = 4, ncol = 2)
    expect_equal(khatri_rao(A, B, reverse = TRUE), expected_rev)

    # List dispatch
    expect_equal(khatri_rao(list(A, B)), expected)
})

test_that("kronecker handles list dispatch", {
    A <- matrix(1:2, nrow = 1, ncol = 2)
    B <- matrix(3:4, nrow = 1, ncol = 2)
    actual <- kronecker(list(A, B))
    expected <- base::kronecker(A, B)
    expect_equal(actual, expected)
})

test_that("hadamard works correctly", {
    A <- matrix(1:4, nrow = 2)
    B <- matrix(2:5, nrow = 2)
    expect_equal(hadamard(A, B), A * B)
    expect_equal(hadamard(list(A, B)), A * B)
})

test_that("fnorm works correctly", {
    A <- tensor(matrix(1:4, nrow = 2))
    val <- fnorm(A)
    expected <- sqrt(1 + 4 + 9 + 16)
    expect_equal(val, expected)

    expect_equal(A$fnorm(), expected)
})

test_that("collapse works correctly", {
    t3d <- tensor(array(1:24, dim = c(4, 3, 2)))

    # Collapse dim 1
    res1 <- collapse(t3d, 1)
    expect_equal(res1$dim(), c(3, 2))
    expect_equal(sum(res1$as_array()), sum(1:24))

    # Collapse dim -1 (equivalent to collapsing 2 and 3)
    res2 <- collapse(t3d, -1)
    expect_equal(res2$dim(), 4L)
    expect_equal(sum(res2$as_array()), sum(1:24))

    # Collapse using max
    res_max <- collapse(t3d, 1, fun = max)
    expect_equal(res_max$dim(), c(3, 2))
})

test_that("t_scale works correctly", {
    # 3x4x5 tensor of ones
    t <- ones(c(3, 4, 5))
    s <- 1:5 * 10

    # Scale mode 3
    res <- t_scale(t, s, 3)
    expect_equal(res$dim(), c(3, 4, 5))

    # test specific values
    arr <- res$as_array()
    expect_equal(arr[1, 1, 1], 10)
    expect_equal(arr[2, 2, 2], 20)
    expect_equal(arr[3, 3, 3], 30)
})

test_that("head and tail methods work", {
    t <- tensor(1:10, 10)
    expect_equal(length(head(t, 2)), 2)
    expect_equal(length(tail(t, 2)), 2)
})
