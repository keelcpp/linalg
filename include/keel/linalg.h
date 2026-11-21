#pragma once

#include "keel/complex.h"
#include "keel/core.h"
#include "keel/function_ref.h"
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

/// General LU factorization with partial pivoting for any m x n matrix (including wide or singular) (f64).
/// Stores 0-based pivot row indices in `piv[0..min(m,n)-1]`.
/// Never aborts on zero pivots; returns true iff all min(m,n) pivots were non-zero.
bool lu_factor_general(f64* A, usize m, usize n, i32* piv) noexcept;

/// General LU factorization with partial pivoting for any m x n matrix (c64).
bool lu_factor_general(c64* A, usize m, usize n, i32* piv) noexcept;

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

/// Full SVD for arbitrary m x n matrix (f64): A = U * diag(S) * V^T.
///   A : m x n column-major (unmodified)
///   U : m x m column-major (orthogonal)
///   S : min(m, n) descending non-negative singular values
///   V : n x n column-major (orthogonal)
void svd_full(const f64* A, usize m, usize n, f64* U, f64* S, f64* V, Allocator* alloc = nullptr) noexcept;

/// Full SVD for arbitrary m x n matrix (c64): A = U * diag(S) * V^H.
void svd_full(const c64* A, usize m, usize n, c64* U, f64* S, c64* V, Allocator* alloc = nullptr) noexcept;

// ============================================================================
// 6. Factorization Updates, LDL^T/LDL^H, and Matrix Balancing
// ============================================================================

/// Rank-1 QR update (Daniel-Gragg-Kaufman-Stewart): given A = Q * R
/// (Q is m x m, R is m x n), updates (Q, R) in place so Q_new * R_new = Q * R + u * v^T.
void qr_update(f64* Q, f64* R, usize m, usize n, const f64* u, const f64* v, Allocator* alloc = nullptr) noexcept;

/// Insert column `x` (length m) at 0-based index `col` (0 <= col <= n) into A = Q * R.
///   Q_in (m x m), R_in (m x n) -> Q_out (m x m), R_out (m x (n+1)).
void qr_insert_col(const f64* Q_in, const f64* R_in, usize m, usize n, usize col, const f64* x, f64* Q_out,
                   f64* R_out, Allocator* alloc = nullptr) noexcept;

/// Delete 0-based column `col` (0 <= col < n) from A = Q * R.
///   Q_in (m x m), R_in (m x n) -> Q_out (m x m), R_out (m x (n-1)).
void qr_delete_col(const f64* Q_in, const f64* R_in, usize m, usize n, usize col, f64* Q_out, f64* R_out) noexcept;

/// Insert row `x` (length n) at 0-based index `row` (0 <= row <= m) into A = Q * R.
///   Q_in (m x m), R_in (m x n) -> Q_out ((m+1) x (m+1)), R_out ((m+1) x n).
void qr_insert_row(const f64* Q_in, const f64* R_in, usize m, usize n, usize row, const f64* x, f64* Q_out,
                   f64* R_out) noexcept;

/// Delete 0-based row `row` (0 <= row < m) from A = Q * R.
///   Q_in (m x m), R_in (m x n) -> Q_out ((m-1) x (m-1)), R_out ((m-1) x n).
void qr_delete_row(const f64* Q_in, const f64* R_in, usize m, usize n, usize row, f64* Q_out, f64* R_out,
                   Allocator* alloc = nullptr) noexcept;

/// Rank-1 Cholesky update (downdate == false: R^T R + x x^T) or downdate (downdate == true: R^T R - x x^T).
/// `R` is n x n upper triangular (updated in place), `x` has length n.
/// Returns true on success, false if downdate would break positive-definiteness.
[[nodiscard]] bool cholesky_update(f64* R, usize n, const f64* x, bool downdate, Allocator* alloc = nullptr) noexcept;

/// Unpivoted Crout LDL^T factorization of symmetric n x n matrix A (f64): A = L * diag(D) * L^T.
/// Symmetrizes A internally into scratch; `L` is n x n unit lower triangular, `D` has length n.
/// Returns 0 on success, or the 1-based pivot index where a near-zero pivot occurred.
[[nodiscard]] usize ldl_factor(const f64* A, usize n, f64* L, f64* D, Allocator* alloc = nullptr) noexcept;

/// Unpivoted Crout LDL^H factorization of Hermitian n x n matrix A (c64): A = L * diag(D) * L^H.
/// Hermitianizes A internally into scratch; `L` is n x n unit lower triangular, `D` has length n (real).
/// Returns 0 on success, or the 1-based pivot index where a near-zero pivot occurred.
[[nodiscard]] usize ldl_factor(const c64* A, usize n, c64* L, f64* D, Allocator* alloc = nullptr) noexcept;

/// Parlett-Reinsch diagonal-similarity balancing (EISPACK balanc) for n x n matrix `B` in place (f64).
///   permute : if true, isolates rows/columns first and writes 0-based swap history to `perm` (length n, optional)
///   scale   : length-n output diagonal scaling factors
///   low_out, high_out : optional 0-based active submatrix bounds [low, high]
void balance_matrix(f64* B, usize n, bool permute, f64* scale, usize* perm = nullptr, usize* low_out = nullptr,
                    usize* high_out = nullptr) noexcept;

/// Parlett-Reinsch diagonal-similarity balancing for n x n matrix `B` in place (c64).
void balance_matrix(c64* B, usize n, bool permute, f64* scale, usize* perm = nullptr, usize* low_out = nullptr,
                    usize* high_out = nullptr) noexcept;

// ============================================================================
// 7. Eigenvalue Problems, Hessenberg, Schur, and Generalized Schur (QZ)
// ============================================================================

/// Symmetric eigendecomposition A = V * diag(D) * V^T (f64, column-major n x n).
/// Uses classical Jacobi for n <= 16 and Householder tridiagonalization + implicit QL for n > 16.
/// Eigenvalues in `D[0..n-1]` are sorted in ascending order; columns of `V` are orthonormal.
void eig_sym(const f64* A, usize n, f64* V, f64* D, Allocator* alloc = nullptr) noexcept;

/// Symmetric generalized eigendecomposition A * V = B * V * diag(D) via Cholesky reduction (f64).
/// Succeeds when B is SPD (standard reduction, B-orthonormal V) or when A is SPD and B is
/// singular/indefinite (reciprocal reduction, max-entry normalized V, Inf for zero eigenvalues of B).
/// Returns true if handled via symmetric Cholesky reduction, false if neither A nor B is SPD
/// (caller should fall back to general `qz_factor`).
[[nodiscard]] bool eig_sym_generalized(const f64* A, const f64* B, usize n, f64* V, f64* D,
                                       Allocator* alloc = nullptr) noexcept;

/// Compute the `k` smallest-magnitude generalized eigenpairs of symmetric `(A, B)` with SPD `A`
/// via skyline Cholesky + reorthogonalized Lanczos on H = R_A^{-T} B R_A^{-1}.
///   V : n x k column-major eigenvectors
///   D : length k eigenvalues sorted in ascending algebraic order
/// Returns true on success, false if `A` is not SPD or the Krylov subspace has dimension < k.
[[nodiscard]] bool eigs_sym_generalized_smallest_abs(const f64* A, const f64* B, usize n, usize k, f64* V, f64* D,
                                                     Allocator* alloc = nullptr) noexcept;

/// In-place upper-Hessenberg reduction H_out = P^T * H_in * P via Householder reflectors (f64).
/// `H` is overwritten with the upper Hessenberg matrix; `P` (n x n) receives the orthogonal transform.
void hess_reduce(f64* H, usize n, f64* P, Allocator* alloc = nullptr) noexcept;

/// In-place upper-Hessenberg reduction H_out = P^H * H_in * P via Householder reflectors (c64).
void hess_reduce(c64* H, usize n, c64* P, Allocator* alloc = nullptr) noexcept;

/// Real quasi-upper-triangular Schur decomposition A = U * T * U^T (f64).
/// Uses Hessenberg reduction + Francis double-shift implicit QR + LAPACK dlanv2 2x2 standardization.
void schur_real(const f64* A, usize n, f64* U, f64* T, Allocator* alloc = nullptr) noexcept;

/// Complex upper-triangular Schur decomposition A = U * T * U^H (c64).
/// Uses Hessenberg reduction + Wilkinson-shifted complex QR.
/// Returns false if QR iteration fails to converge.
[[nodiscard]] bool schur_complex(const c64* A, usize n, c64* U, c64* T, Allocator* alloc = nullptr) noexcept;

/// Compute unit-norm right eigenvectors `V` (n x n) and eigenvalues `evals` (length n) from a
/// complex upper-triangular Schur decomposition A = U * T * U^H via triangular back-substitution.
void schur_eigenvectors(const c64* U, const c64* T, usize n, c64* V, c64* evals, Allocator* alloc = nullptr) noexcept;

/// Reorder a complex Schur decomposition A = U * T * U^H in place so eigenvalues with `select[i] != 0`
/// are moved to the top-left diagonal block.
void ordschur_complex(c64* U, c64* T, usize n, const u8* select, Allocator* alloc = nullptr) noexcept;

/// Convert a real quasi-upper-triangular Schur form (UR, TR) to complex upper-triangular form (UC, TC).
void rsf2csf(const f64* UR, const f64* TR, usize n, c64* UC, c64* TC) noexcept;

/// Convert a complex diagonalized pair (V, D) with adjacent conjugate pairs into real block-diagonal
/// form (VR, DR). Returns false if a complex eigenvalue is dangling or lacks its conjugate partner.
[[nodiscard]] bool cdf2rdf(const c64* V, const c64* D, usize n, f64* VR, f64* DR) noexcept;

/// Complex generalized Schur (QZ) decomposition of square pencil (A, B):
///   Q * A * Z = AA,  Q * B * Z = BB   (AA, BB upper triangular; Q, Z unitary).
/// Returns false if QZ iteration fails to converge.
[[nodiscard]] bool qz_factor(const c64* A, const c64* B, usize n, c64* AA, c64* BB, c64* Q, c64* Z) noexcept;

/// Reorder a complex generalized Schur pencil (AA, BB, Q, Z) in place so entries with `select[i] != 0`
/// appear in the top-left block.
void ordqz_complex(c64* AA, c64* BB, c64* Q, c64* Z, usize n, const u8* select, Allocator* alloc = nullptr) noexcept;

/// Compute unit-norm right generalized eigenvectors `V` (n x n) and eigenvalues `evals` (length n)
/// from an upper-triangular complex QZ decomposition (AA, BB, Z).
void qz_eigenvectors(const c64* AA, const c64* BB, const c64* Z, usize n, c64* V, c64* evals,
                     Allocator* alloc = nullptr) noexcept;

// ============================================================================
// 8. Matrix Functions, Sylvester Equation, and Specialized Solvers
// ============================================================================

/// Matrix exponential exp(A) for n x n matrix via scaling-and-squaring Pade(6) (f64).
/// Returns false if the Pade denominator matrix is singular.
[[nodiscard]] bool expm(const f64* A, usize n, f64* X, Allocator* alloc = nullptr) noexcept;

/// Matrix exponential exp(A) for n x n matrix via scaling-and-squaring Pade(6) (c64).
[[nodiscard]] bool expm(const c64* A, usize n, c64* X, Allocator* alloc = nullptr) noexcept;

/// Principal matrix square root sqrt(A) for n x n matrix via complex Schur + Bjorck-Hammarling (c64).
/// Returns false if the underlying Schur decomposition fails to converge.
[[nodiscard]] bool sqrtm(const c64* A, usize n, c64* X, Allocator* alloc = nullptr) noexcept;

/// Principal matrix logarithm log(A) for n x n matrix via complex Schur + inverse scaling-and-squaring (c64).
/// Returns false if the underlying Schur decomposition fails to converge.
[[nodiscard]] bool logm(const c64* A, usize n, c64* X, Allocator* alloc = nullptr) noexcept;

/// General matrix function f(A) for n x n matrix via complex Schur + Parlett recurrence (c64).
/// Returns false if the underlying Schur decomposition fails to converge.
[[nodiscard]] bool funm(const c64* A, usize n, FunctionRef<c64(c64)> f, c64* X, Allocator* alloc = nullptr) noexcept;

/// Solve the Sylvester equation A * X + X * B = C via Bartels-Stewart (complex Schur) (c64).
///   A : m x m, B : n x n, C : m x n, X : m x n (column-major).
/// Returns false if Schur decomposition fails to converge.
[[nodiscard]] bool sylvester(const c64* A, usize m, const c64* B, usize n, const c64* C, c64* X,
                             Allocator* alloc = nullptr) noexcept;

/// Action of the matrix exponential exp(t * A) * v via Arnoldi Krylov projection (f64).
///   A : n x n, v : length n, out : length n.
/// Returns false if the projected Pade solve fails.
[[nodiscard]] bool expmv(f64 t, const f64* A, usize n, const f64* v, f64* out, Allocator* alloc = nullptr) noexcept;

/// Summary returned by `lsqnonneg`.
struct NnlsStats {
    f64 resnorm;
    usize iterations;
    bool converged;
};

/// Lawson-Hanson active-set non-negative least squares: min_{x >= 0} ||C * x - d||_2^2.
///   C        : m x n column-major
///   d        : length m
///   x        : length n (output)
///   residual : optional length m output receiving d - C * x
NnlsStats lsqnonneg(const f64* C, usize m, usize n, const f64* d, f64* x, f64* residual = nullptr,
                    Allocator* alloc = nullptr) noexcept;

/// Reduced row echelon form (Gauss-Jordan elimination with partial pivoting) in place on m x n matrix `A` (f64).
///   tol        : pivot threshold; if tol < 0.0, computed automatically from ||A||_inf * max(m, n) * eps
///   pivot_cols : optional caller buffer of length >= min(m, n) receiving 0-based pivot column indices
/// Returns the numerical rank (number of pivot columns found).
usize rref(f64* A, usize m, usize n, f64 tol = -1.0, usize* pivot_cols = nullptr) noexcept;

/// Reduced row echelon form in place on m x n matrix `A` (c64).
usize rref(c64* A, usize m, usize n, f64 tol = -1.0, usize* pivot_cols = nullptr) noexcept;

/// Generalized singular values of (A, B) where A is m x n, B is p x n (m + p >= n) (f64).
/// Writes min(m, n) values sorted ascending into `sigmas` (Inf where s_i == 0).
void gsvd_values(const f64* A, usize m, const f64* B, usize p, usize n, f64* sigmas,
                 Allocator* alloc = nullptr) noexcept;

/// Generalized singular values of (A, B) (c64).
void gsvd_values(const c64* A, usize m, const c64* B, usize p, usize n, f64* sigmas,
                 Allocator* alloc = nullptr) noexcept;

/// Generalized SVD of (A, B) where A is m x n, B is p x n (m + p >= n) (f64):
///   A = U * C * X^T,  B = V * S * X^T
///   U : m x m, V : p x n, X : n x n, C : m x n (diagonal), S : p x n (diagonal).
void gsvd_full(const f64* A, usize m, const f64* B, usize p, usize n, f64* U, f64* V, f64* X, f64* C, f64* S,
               Allocator* alloc = nullptr) noexcept;

/// Generalized SVD of (A, B) (c64): A = U * C * X^H, B = V * S * X^H.
void gsvd_full(const c64* A, usize m, const c64* B, usize p, usize n, c64* U, c64* V, c64* X, f64* C, f64* S,
               Allocator* alloc = nullptr) noexcept;

} // namespace keel
