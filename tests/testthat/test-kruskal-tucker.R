test_that("KTensor initialization and conversion work", {
    lambda <- c(1, 2)
    U <- list(matrix(1:6, 3, 2), matrix(1:8, 4, 2), matrix(1:4, 2, 2))
    k <- ktensor(lambda, U)

    expect_s3_class(k, "KTensor")
    expect_s3_class(k, "Tensor")
    expect_equal(k$dim(), c(3, 4, 2))
    expect_equal(k$ndims(), 3)

    k_dense <- as.tensor(k)
    expect_s3_class(k_dense, "Tensor")
    expect_equal(k_dense$dim(), c(3, 4, 2))
})

test_that("TTensor initialization and conversion work", {
    core <- tensor(array(1:24, dim = c(2, 3, 4)))
    U <- list(matrix(1:10, 5, 2), matrix(1:18, 6, 3), matrix(1:28, 7, 4))
    t_tens <- ttensor(core, U)

    expect_s3_class(t_tens, "TTensor")
    expect_s3_class(t_tens, "Tensor")
    expect_equal(t_tens$dim(), c(5, 6, 7))
    expect_equal(t_tens$ndims(), 3)

    t_dense <- as.tensor(t_tens)
    expect_s3_class(t_dense, "Tensor")
    expect_equal(t_dense$dim(), c(5, 6, 7))
})

test_that("ttm works natively for KTensor", {
    lambda <- c(1, 2)
    U <- list(matrix(runif(6), 3, 2), matrix(runif(8), 4, 2), matrix(runif(4), 2, 2))
    k <- ktensor(lambda, U)

    # Multiply by matrix mode 1
    mat <- matrix(runif(6), 2, 3)
    k_mat <- ttm(k, mat, mode = 1)
    expect_s3_class(k_mat, "KTensor")
    expect_equal(k_mat$dim(), c(2, 4, 2))

    # Multiply by vector mode 2
    vec <- runif(4)
    k_vec <- ttm(k, vec, mode = 2)
    expect_s3_class(k_vec, "KTensor")
    expect_equal(k_vec$dim(), c(3, 2))

    # Check against dense evaluation
    k_dense <- as.tensor(k)
    k_dense_vec <- ttm(k_dense, vec, mode = 2)
    diff <- as.tensor(k_vec)$as_array() - k_dense_vec$as_array()
    expect_true(max(abs(diff)) < 1e-10)

    # Multiply by list of matrices
    mats <- list(matrix(runif(6), 2, 3), matrix(runif(12), 3, 4))
    k_list <- ttm(k, mats, mode = c(1, 2))
    expect_s3_class(k_list, "KTensor")
    expect_equal(k_list$dim(), c(2, 3, 2))
})

test_that("ttm works natively for TTensor", {
    core <- tensor(array(runif(24), dim = c(2, 3, 4)))
    U <- list(matrix(runif(10), 5, 2), matrix(runif(18), 6, 3), matrix(runif(28), 7, 4))
    t_tens <- ttensor(core, U)

    # Multiply by matrix mode 1
    mat <- matrix(runif(15), 3, 5)
    t_mat <- ttm(t_tens, mat, mode = 1)
    expect_s3_class(t_mat, "TTensor")
    expect_equal(t_mat$dim(), c(3, 6, 7))

    # Multiply by vector mode 2
    vec <- runif(6)
    t_vec <- ttm(t_tens, vec, mode = 2)
    expect_s3_class(t_vec, "TTensor")
    expect_equal(t_vec$dim(), c(5, 7))

    # Check against dense evaluation
    t_dense <- as.tensor(t_tens)
    t_dense_vec <- ttm(t_dense, vec, mode = 2)
    diff <- as.tensor(t_vec)$as_array() - t_dense_vec$as_array()
    expect_true(max(abs(diff)) < 1e-10)

    # Multiply by list of matrices
    mats <- list(matrix(runif(15), 3, 5), matrix(runif(12), 2, 6))
    t_list <- ttm(t_tens, mats, mode = c(1, 2))
    expect_s3_class(t_list, "TTensor")
    expect_equal(t_list$dim(), c(3, 2, 7))
})

test_that("ttm vector contraction preserves unrelated singleton Tucker modes", {
    core <- tensor(array(1:6, dim = c(1, 2, 3)))
    U <- list(
        matrix(1, nrow = 4, ncol = 1),
        matrix(1:10, nrow = 5, ncol = 2),
        matrix(1:18, nrow = 6, ncol = 3)
    )

    t_tens <- ttensor(core, U)
    result <- ttm(t_tens, 1:5, mode = 2)

    expect_s3_class(result, "TTensor")
    expect_equal(result$core$dim(), c(1, 3))
    expect_equal(result$dim(), c(4, 6))
})

test_that("ttm can fully contract a KTensor to a scalar", {
    lambda <- c(1, 2)
    U <- list(matrix(1:4, 2, 2), matrix(1:6, 3, 2))

    k_tens <- ktensor(lambda, U)
    result <- ttm(k_tens, list(1:2, 1:3), mode = c(1, 2))

    expect_s3_class(result, "Tensor")
    expect_equal(result$dim(), integer(0))
    expect_equal(as.numeric(result$as_array()), sum(as.tensor(k_tens)$as_array() * outer(1:2, 1:3)))
})

test_that("Multiple dispatch arithmetic operations work between KTensors and TTensors", {
    lambda <- c(1, 2)
    U <- list(matrix(runif(6), 3, 2), matrix(runif(8), 4, 2), matrix(runif(4), 2, 2))
    k <- ktensor(lambda, U)

    core <- tensor(array(runif(24), dim = c(3, 4, 2)))
    U2 <- list(diag(3), diag(4), diag(2)) # Keep dims same as k
    t_tens <- ttensor(core, U2)

    # Addition
    res_add <- k + t_tens
    expect_s3_class(res_add, "Tensor")
    expect_equal(res_add$dim(), c(3, 4, 2))

    # Subtraction
    res_sub <- k - t_tens
    expect_s3_class(res_sub, "Tensor")
    expect_equal(res_sub$dim(), c(3, 4, 2))

    # Test with scalar
    res_scalar <- k * 2
    expect_s3_class(res_scalar, "Tensor")
    expect_equal(res_scalar$dim(), c(3, 4, 2))
})
