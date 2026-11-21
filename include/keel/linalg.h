#pragma once

#include "keel/complex.h"
#include "keel/core.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel {

// ============================================================================
// 1. Matrix Layout & Operation Descriptors (Column-Major BLAS/LAPACK Convention)
// ============================================================================

enum class MatrixSide : u8 {
    Left = 0,
    Right = 1,
};

enum class MatrixUplo : u8 {
    Upper = 0,
    Lower = 1,
};

enum class MatrixTranspose : u8 {
    NoTrans = 0,
    Trans = 1,
    ConjTrans = 2,
};

enum class MatrixDiag : u8 {
    NonUnit = 0,
    Unit = 1,
};

// ============================================================================
// 2. BLAS Level 1 (Vector Operations)
// ============================================================================

/// y = alpha * x + y (real double).
void axpy(usize n, f64 alpha, const f64* x, f64* y) noexcept;

/// y = alpha * x + y (complex double).
void axpy(usize n, c64 alpha, const c64* x, c64* y) noexcept;

/// Real dot product: sum_{i=0}^{n-1} x[i] * y[i].
[[nodiscard]] f64 dot(usize n, const f64* x, const f64* y) noexcept;

/// Conjugate complex dot product: sum_{i=0}^{n-1} conj(x[i]) * y[i].
[[nodiscard]] c64 dotc(usize n, const c64* x, const c64* y) noexcept;

/// Unconjugated complex dot product: sum_{i=0}^{n-1} x[i] * y[i].
[[nodiscard]] c64 dotu(usize n, const c64* x, const c64* y) noexcept;

/// Overflow-safe Euclidean 2-norm ||x||_2 (real double).
[[nodiscard]] f64 nrm2(usize n, const f64* x) noexcept;

/// Overflow-safe Euclidean 2-norm ||x||_2 (complex double).
[[nodiscard]] f64 nrm2(usize n, const c64* x) noexcept;

/// x = alpha * x (real double).
void scal(usize n, f64 alpha, f64* x) noexcept;

/// x = alpha * x (complex double).
void scal(usize n, c64 alpha, c64* x) noexcept;

// ============================================================================
// 3. BLAS Level 2 (Matrix-Vector Operations, Column-Major)
// ============================================================================

/// y = alpha * A * x + beta * y (A is m x n column-major with leading dim lda).
void gemv(usize m, usize n, f64 alpha, const f64* A, usize lda, const f64* x, usize incx, f64 beta, f64* y,
          usize incy) noexcept;

/// y = alpha * op(A) * x + beta * y (A is m x n column-major with leading dim lda).
void gemv(MatrixTranspose trans, usize m, usize n, f64 alpha, const f64* A, usize lda, const f64* x, usize incx,
          f64 beta, f64* y, usize incy) noexcept;

/// Rank-1 update: A = alpha * x * y^T + A (A is m x n column-major).
void ger(usize m, usize n, f64 alpha, const f64* x, usize incx, const f64* y, usize incy, f64* A, usize lda) noexcept;

// ============================================================================
// 4. BLAS Level 3 (Matrix-Matrix Operations, Column-Major)
// ============================================================================

/// General matrix-matrix multiply: C = alpha * op(A) * op(B) + beta * C (f64).
/// op(A) is (m x k), op(B) is (k x n), C is (m x n).
void gemm(MatrixTranspose transa, MatrixTranspose transb, usize m, usize n, usize k, f64 alpha, const f64* A, usize lda,
          const f64* B, usize ldb, f64 beta, f64* C, usize ldc) noexcept;

/// General matrix-matrix multiply: C = alpha * op(A) * op(B) + beta * C (c64).
void gemm(MatrixTranspose transa, MatrixTranspose transb, usize m, usize n, usize k, c64 alpha, const c64* A, usize lda,
          const c64* B, usize ldb, c64 beta, c64* C, usize ldc) noexcept;

/// Triangular matrix solve: op(A) * X = alpha * B (Left) or X * op(A) = alpha * B (Right) (f64).
void trsm(MatrixSide side, MatrixUplo uplo, MatrixTranspose trans, MatrixDiag diag, usize m, usize n, f64 alpha,
          const f64* A, usize lda, f64* B, usize ldb) noexcept;

/// Triangular matrix solve (c64).
void trsm(MatrixSide side, MatrixUplo uplo, MatrixTranspose trans, MatrixDiag diag, usize m, usize n, c64 alpha,
          const c64* A, usize lda, c64* B, usize ldb) noexcept;

/// Symmetric rank-k update: C = alpha * A * A^T + beta * C (NoTrans) or alpha * A^T * A + beta * C (f64).
void syrk(MatrixUplo uplo, MatrixTranspose trans, usize n, usize k, f64 alpha, const f64* A, usize lda, f64 beta,
          f64* C, usize ldc) noexcept;

/// Hermitian rank-k update: C = alpha * A * A^H + beta * C (NoTrans) or alpha * A^H * A + beta * C (c64).
void syrk(MatrixUplo uplo, MatrixTranspose trans, usize n, usize k, c64 alpha, const c64* A, usize lda, c64 beta,
          c64* C, usize ldc) noexcept;

/// Unblocked LU panel factorization with partial pivoting (f64).
/// Stores 0-based pivot indices (shifted by `offset_row`) in `piv[0..min(m,n)-1]`.
/// Returns false if a zero pivot is encountered.
[[nodiscard]] bool lu_panel(f64* A, usize lda, i32* piv, usize m, usize n, usize offset_row = 0) noexcept;

/// Unblocked LU panel factorization with partial pivoting (c64).
[[nodiscard]] bool lu_panel(c64* A, usize lda, i32* piv, usize m, usize n, usize offset_row = 0) noexcept;

/// Diagnostic: number of worker threads used in the most recent `gemm` call.
[[nodiscard]] usize linalg_last_gemm_threads_used() noexcept;

// ============================================================================
// 5. Linear Solvers & Core Matrix Decompositions (Column-Major)
// ============================================================================

/// Recursive blocked LU factorization with partial pivoting (f64).
/// Returns false if a zero pivot is encountered (singular matrix).
[[nodiscard]] bool lu_factor(f64* A, usize lda, i32* piv, usize m, usize n) noexcept;

/// Recursive blocked LU factorization with partial pivoting (c64).
[[nodiscard]] bool lu_factor(c64* A, usize lda, i32* piv, usize m, usize n) noexcept;

/// Solve A * X = B for X (f64).
///   A : m x n column-major (m >= n; square uses blocked LU, tall uses Householder QR)
///   B : m x nrhs column-major
///   X : n x nrhs column-major (caller-allocated)
/// Returns false if m < n, singular (square), or rank-deficient (tall).
[[nodiscard]] bool la_solve(const f64* A, usize m, usize n, const f64* B, usize nrhs, f64* X,
                            Allocator* alloc = nullptr) noexcept;

/// Solve A * X = B for X (c64).
[[nodiscard]] bool la_solve(const c64* A, usize m, usize n, const c64* B, usize nrhs, c64* X,
                            Allocator* alloc = nullptr) noexcept;

/// Compute matrix inverse A^{-1} of n x n column-major matrix A into A_inv (f64).
/// Returns false if A is singular.
[[nodiscard]] bool matrix_inv(const f64* A, usize n, f64* A_inv, Allocator* alloc = nullptr) noexcept;

/// Compute matrix inverse A^{-1} of n x n column-major matrix A into A_inv (c64).
[[nodiscard]] bool matrix_inv(const c64* A, usize n, c64* A_inv, Allocator* alloc = nullptr) noexcept;

/// Blocked upper-triangular Cholesky factorization A = R^T * R (f64).
/// `A` and `R` are n x n column-major (may alias).
/// Returns 0 on success (SPD), or the 1-based leading minor index where non-positive-definiteness occurred.
[[nodiscard]] usize cholesky_upper(const f64* A, f64* R, usize n) noexcept;

/// Blocked upper-triangular Cholesky factorization A = R^H * R (c64).
[[nodiscard]] usize cholesky_upper(const c64* A, c64* R, usize n) noexcept;

/// Full Householder QR factorization A = Q * R (f64).
///   A : m x n column-major
///   Q : m x m column-major (orthogonal)
///   R : m x n column-major (upper triangular)
void qr_householder(const f64* A, usize m, usize n, f64* Q, f64* R, Allocator* alloc = nullptr) noexcept;

/// Full Householder QR factorization A = Q * R (c64).
void qr_householder(const c64* A, usize m, usize n, c64* Q, c64* R, Allocator* alloc = nullptr) noexcept;

/// Column-pivoted Householder QR factorization A * P = Q * R (f64).
///   perm : n elements receiving 0-based column permutation indices.
void qr_pivoted(const f64* A, usize m, usize n, f64* Q, f64* R, usize* perm, Allocator* alloc = nullptr) noexcept;

/// Column-pivoted Householder QR factorization A * P = Q * R (c64).
void qr_pivoted(const c64* A, usize m, usize n, c64* Q, c64* R, usize* perm, Allocator* alloc = nullptr) noexcept;

/// One-sided Jacobi thin SVD for m >= n (f64): A = U * diag(S) * V^T.
/// On input `A` is m x n column-major; on output `A` is overwritten with U (m x n),
/// `V` (n x n column-major) holds right singular vectors, and `S` (n) holds non-negative
/// singular values in descending order.
void svd_jacobi_inplace(f64* A, usize m, usize n, f64* V, f64* S, Allocator* alloc = nullptr) noexcept;

/// One-sided Jacobi thin SVD for m >= n (c64): A = U * diag(S) * V^H.
void svd_jacobi_inplace(c64* A, usize m, usize n, c64* V, f64* S, Allocator* alloc = nullptr) noexcept;

} // namespace keel
