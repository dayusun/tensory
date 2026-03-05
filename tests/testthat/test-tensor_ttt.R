test_that("ttt computes outer product correctly", {
    # 2D outer 2D = 4D
    A <- tensor(matrix(1:4, nrow = 2, ncol = 2))
    B <- tensor(matrix(1:6, nrow = 2, ncol = 3))

    C <- ttt(A, B)

    expect_s3_class(C, "Tensor")
    expect_equal(C$dim(), c(2, 2, 2, 3))

    # Check an element: C[2,1, 1,2] = A[2,1] * B[1,2]
    # A:
    # 1 3
    # 2 4
    # B:
    # 1 3 5
    # 2 4 6
    # A[2,1] = 2, B[1,2] = 3. 2 * 3 = 6
    expect_equal(C$as_array()[2, 1, 1, 2], 6)
})

test_that("ttt computes inner product (full contraction) correctly", {
    A <- tensor(array(1:24, dim = c(4, 3, 2)))

    # Inner product of A with itself
    # Result should be a scalar (0D tensor representing sum of squares)
    C <- ttt(A, A, dimsA = 1:3)

    expect_s3_class(C, "Tensor")
    expect_equal(C$dim(), integer(0)) # Scalar tensor has empty dim
    expect_equal(as.numeric(C$as_array()), sum((1:24)^2))
})

test_that("ttt computes partial contraction correctly", {
    A <- tensor(array(1:24, dim = c(4, 3, 2)))
    B <- tensor(array(1:12, dim = c(3, 2, 2)))

    # Contract A mode 2 (size 3) and 3 (size 2) with B mode 1 (size 3) and 2 (size 2)
    # Remaining dims: A mode 1 (size 4), B mode 3 (size 2)
    # Result should be 4x2 matrix
    C <- ttt(A, B, dimsA = c(2, 3), dimsB = c(1, 2))

    expect_s3_class(C, "Tensor")
    expect_equal(C$dim(), c(4, 2))

    # Let's compute manually C[1, 1] = sum_{j,k} A[1,j,k] * B[j,k,1]
    A_arr <- A$as_array()
    B_arr <- B$as_array()

    val_11 <- 0
    for (j in 1:3) {
        for (k in 1:2) {
            val_11 <- val_11 + A_arr[1, j, k] * B_arr[j, k, 1]
        }
    }

    expect_equal(C$as_array()[1, 1], val_11)

    val_42 <- 0
    for (j in 1:3) {
        for (k in 1:2) {
            val_42 <- val_42 + A_arr[4, j, k] * B_arr[j, k, 2]
        }
    }

    expect_equal(C$as_array()[4, 2], val_42)
})

test_that("ttt dimension mismatch throws error", {
    A <- tensor(array(1:24, dim = c(4, 3, 2)))
    B <- tensor(array(1:12, dim = c(3, 2, 2)))

    # Try to contract A mode 1 (size 4) with B mode 1 (size 3)
    expect_error(ttt(A, B, dimsA = 1, dimsB = 1), "Contracted dimension sizes do not match")

    # Mismatched length of dims vector
    expect_error(ttt(A, B, dimsA = c(1, 2), dimsB = 1), "dimsA and dimsB must have the same length")
})
