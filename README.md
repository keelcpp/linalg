# keel::linalg

Dense linear algebra (BLAS Level 1/2/3, linear solvers, matrix decompositions, eigensolvers, Schur/QZ, and matrix functions) for the [keel](https://github.com/keelcpp/keel) C++20 ecosystem — column-major layout, optional Google Highway SIMD acceleration + multi-threaded BLIS GEMM via `keel::ThreadPool`, zero STL headers in public API, `-fno-exceptions -fno-rtti` compatible.

## Features

- **BLAS Level 1**: `axpy` (`f64`/`c64`), `dot`, `dotc`, `dotu`, `nrm2`, `scal`.
- **BLAS Level 2**: `gemv` (standard and transposed), `ger`.
- **BLAS Level 3** (BLIS-blocked Highway SIMD + portable fallback): `gemm`, `trsm`, `syrk`, `lu_panel` for both real `f64` and complex `c64`.
- **Linear Solvers & Core Decompositions**:
  - `lu_factor` (blocked divide-and-conquer) & `lu_factor_general` (arbitrary $m \times n$, singular-safe).
  - `la_solve` — general linear solve $A X = B$ (blocked LU for square, Householder QR for tall $m \ge n$).
  - `matrix_inv` — matrix inversion via blocked LU.
  - `cholesky_upper` — blocked upper-triangular Cholesky ($A = R^H R$).
  - `qr_householder` & `qr_pivoted` — Householder QR with optional column pivoting.
  - `svd_jacobi_inplace` (thin $m \ge n$) & `svd_full` (full arbitrary $m \times n$).
- **Factorization Updates, $LDL^T$/$LDL^H$, and Balancing**:
  - `qr_update`, `qr_insert_col`, `qr_delete_col`, `qr_insert_row`, `qr_delete_row` (Daniel–Gragg–Kaufman–Stewart).
  - `cholesky_update` — rank-1 Cholesky update & downdate (LINPACK `dchud`/`dchdd`).
  - `ldl_factor` — Crout $LDL^T$ / $LDL^H$ factorization (`f64` and `c64`).
  - `balance_matrix` — Parlett–Reinsch diagonal-similarity balancing (`f64` and `c64`).
- **Eigenvalue Problems, Hessenberg, Schur, and QZ**:
  - `eig_sym` (Jacobi for $n \le 16$, Householder tridiagonalization + implicit QL for $n > 16$).
  - `eig_sym_generalized` & `eigs_sym_generalized_smallest_abs` (skyline Cholesky + reorthogonalized Lanczos).
  - `hess_reduce`, `schur_real` (Francis double-shift QR + 2×2 standardization), `schur_complex`, `schur_eigenvectors`, `ordschur_complex`, `rsf2csf`, `cdf2rdf`.
  - `qz_factor` (Moler–Stewart complex QZ), `ordqz_complex`, `qz_eigenvectors`.
- **Matrix Functions & Specialized Solvers**:
  - `expm` (Higham 2005 optimal Padé-3/5/7/9/13 scaling & squaring), `sqrtm` (Schur–Björck–Hammarling), `logm` (Schur + inverse scaling & squaring + 7-point Gauss–Legendre `[7/7]` Padé), `funm` (Davies–Higham blocked Schur–Parlett with Cauchy-contour confluent blocks), `sylvester` (Bartels–Stewart with singularity detection), `expmv` (step-split reorthogonalized Arnoldi Krylov).
  - `lsqnonneg` (Lawson–Hanson active-set NNLS), `rref` (Gauss–Jordan), `gsvd_values` & `gsvd_full`.

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python tests/verify_reference.py build/Release/test_linalg.exe
```

`tests/verify_reference.py` validates both Highway SIMD and scalar portable BLAS/LAPACK builds (GEMM, LU, solve, inv, Cholesky, QR, SVD, symmetric & generalized eigen, Schur, QZ, `expm`, `logm`, `sqrtm`, `funm` incl. Jordan blocks, `sylvester`, `expmv`, `lsqnonneg`, `rref`, `gsvd`) against NumPy / SciPy (`scipy.linalg`, `scipy.optimize.nnls`) and MATLAB R2025b.
