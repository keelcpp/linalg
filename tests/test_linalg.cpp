#include "keel/array.h"
#include "keel/complex.h"
#include "keel/linalg.h"
#include "keel/math.h"

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

keel::f64 det_val(keel::usize i, keel::usize j, keel::usize seed) {
    keel::u64 z = static_cast<keel::u64>(i) * 0x9e3779b97f4a7c15ULL +
                  static_cast<keel::u64>(j) * 0xbf58476d1ce4e5b9ULL +
                  static_cast<keel::u64>(seed) * 0x94d049bb133111ebULL + 0x123456789abcdef0ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= (z >> 31);
    return (static_cast<keel::f64>(z >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
}

keel::c64 det_cval(keel::usize i, keel::usize j, keel::usize seed) {
    return keel::c64(det_val(i, j, seed), det_val(i, j, seed + 17));
}

template <typename T>
keel::f64 max_diff(const T* a, const T* b, keel::usize n) {
    keel::f64 mx = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 d = keel::abs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

template <typename T>
T op_elem(const T* M, keel::usize ld, keel::usize r, keel::usize c, keel::MatrixTranspose tr) {
    if (tr == keel::MatrixTranspose::NoTrans) return M[r + c * ld];
    const T v = M[c + r * ld];
    if constexpr (keel::IsSame<T, keel::c64>) {
        return (tr == keel::MatrixTranspose::ConjTrans) ? keel::conj(v) : v;
    } else {
        return v;
    }
}

template <typename T>
void naive_gemm(keel::MatrixTranspose ta, keel::MatrixTranspose tb, keel::usize m, keel::usize n, keel::usize k,
                T alpha, const T* A, keel::usize lda, const T* B, keel::usize ldb, T beta, T* C, keel::usize ldc) {
    for (keel::usize j = 0; j < n; ++j) {
        for (keel::usize i = 0; i < m; ++i) {
            T acc = T(0);
            for (keel::usize p = 0; p < k; ++p) {
                acc += op_elem(A, lda, i, p, ta) * op_elem(B, ldb, p, j, tb);
            }
            C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
        }
    }
}

void test_blas1() {
    keel::f64 x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    keel::f64 y[5] = {10.0, 20.0, 30.0, 40.0, 50.0};
    keel::axpy(5, 2.5, x, y);
    for (keel::usize i = 0; i < 5; ++i) {
        CHECK(keel::abs(y[i] - (10.0 * static_cast<keel::f64>(i + 1) + 2.5 * x[i])) < 1e-12);
    }

    CHECK(keel::abs(keel::dot(5, x, x) - 55.0) < 1e-12);
    CHECK(keel::abs(keel::nrm2(5, x) - keel::sqrt(55.0)) < 1e-12);

    // Overflow-safe nrm2 test.
    keel::f64 huge[2] = {3e200, 4e200};
    CHECK(keel::abs(keel::nrm2(2, huge) - 5e200) < 1e188);

    keel::scal(5, 3.0, x);
    CHECK(keel::abs(x[4] - 15.0) < 1e-12);

    keel::c64 cx[3] = {keel::c64(1.0, 2.0), keel::c64(3.0, -1.0), keel::c64(0.0, 4.0)};
    keel::c64 cy[3] = {keel::c64(2.0, -1.0), keel::c64(1.0, 1.0), keel::c64(-2.0, 3.0)};
    const keel::c64 dc = keel::dotc(3, cx, cy);
    // conj(1+2i)*(2-i) = (1-2i)(2-i) = 0 - 5i
    // conj(3-i)*(1+i)  = (3+i)(1+i)  = 2 + 4i
    // conj(4i)*(-2+3i) = (-4i)(-2+3i) = 12 + 8i
    CHECK(keel::abs(dc - keel::c64(14.0, 7.0)) < 1e-12);

    const keel::c64 du = keel::dotu(3, cx, cy);
    CHECK(keel::abs(du - keel::c64(-4.0, -3.0)) < 1e-12);
    CHECK(keel::abs(keel::nrm2(3, cx) - keel::sqrt(31.0)) < 1e-12);

    keel::axpy(3, keel::c64(2.0, 1.0), cx, cy);
    CHECK(keel::abs(cy[0] - (keel::c64(2.0, -1.0) + keel::c64(2.0, 1.0) * cx[0])) < 1e-12);
}

void test_blas2() {
    constexpr keel::usize m = 13;
    constexpr keel::usize n = 9;
    keel::Array<keel::f64> A(m * n);
    keel::Array<keel::f64> x(n);
    keel::Array<keel::f64> y(m, 1.5);
    keel::Array<keel::f64> y_ref(m, 1.5);
    for (keel::usize j = 0; j < n; ++j) {
        x[j] = det_val(j, 0, 1);
        for (keel::usize i = 0; i < m; ++i) A[i + j * m] = det_val(i, j, 2);
    }

    keel::gemv(m, n, 2.0, A.data(), m, x.data(), 1, 0.5, y.data(), 1);
    for (keel::usize i = 0; i < m; ++i) {
        keel::f64 s = 0.0;
        for (keel::usize j = 0; j < n; ++j) s += A[i + j * m] * x[j];
        y_ref[i] = 2.0 * s + 0.5 * y_ref[i];
    }
    CHECK(max_diff(y.data(), y_ref.data(), m) < 1e-12);

    // Transposed gemv.
    keel::Array<keel::f64> xt(m);
    keel::Array<keel::f64> yt(n, 0.0);
    keel::Array<keel::f64> yt_ref(n, 0.0);
    for (keel::usize i = 0; i < m; ++i) xt[i] = det_val(i, 3, 4);
    keel::gemv(keel::MatrixTranspose::Trans, m, n, -1.5, A.data(), m, xt.data(), 1, 0.0, yt.data(), 1);
    for (keel::usize j = 0; j < n; ++j) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < m; ++i) s += A[i + j * m] * xt[i];
        yt_ref[j] = -1.5 * s;
    }
    CHECK(max_diff(yt.data(), yt_ref.data(), n) < 1e-12);

    // Rank-1 ger.
    keel::Array<keel::f64> A_ref(m * n);
    for (keel::usize i = 0; i < m * n; ++i) A_ref[i] = A[i];
    keel::ger(m, n, 0.75, xt.data(), 1, x.data(), 1, A.data(), m);
    for (keel::usize j = 0; j < n; ++j) {
        for (keel::usize i = 0; i < m; ++i) {
            A_ref[i + j * m] += 0.75 * xt[i] * x[j];
        }
    }
    CHECK(max_diff(A.data(), A_ref.data(), m * n) < 1e-12);
}

void test_gemm() {
    // Small (stack fast-path) & large (BLIS + multi-thread + odd tail panels).
    const keel::usize dims[2][3] = {{17, 19, 13}, {137, 129, 97}};
    const keel::MatrixTranspose trs[2] = {keel::MatrixTranspose::NoTrans, keel::MatrixTranspose::Trans};

    for (keel::usize d = 0; d < 2; ++d) {
        const keel::usize m = dims[d][0];
        const keel::usize n = dims[d][1];
        const keel::usize k = dims[d][2];
        for (keel::usize a_t = 0; a_t < 2; ++a_t) {
            for (keel::usize b_t = 0; b_t < 2; ++b_t) {
                const auto ta = trs[a_t];
                const auto tb = trs[b_t];
                const keel::usize lda = (ta == keel::MatrixTranspose::NoTrans) ? m : k;
                const keel::usize colsA = (ta == keel::MatrixTranspose::NoTrans) ? k : m;
                const keel::usize ldb = (tb == keel::MatrixTranspose::NoTrans) ? k : n;
                const keel::usize colsB = (tb == keel::MatrixTranspose::NoTrans) ? n : k;

                keel::Array<keel::f64> A(lda * colsA);
                keel::Array<keel::f64> B(ldb * colsB);
                keel::Array<keel::f64> C(m * n);
                keel::Array<keel::f64> C_ref(m * n);
                for (keel::usize i = 0; i < A.size(); ++i) A[i] = det_val(i, 1, 10 + d);
                for (keel::usize i = 0; i < B.size(); ++i) B[i] = det_val(i, 2, 20 + d);
                for (keel::usize i = 0; i < C.size(); ++i) C[i] = C_ref[i] = det_val(i, 3, 30 + d);

                keel::gemm(ta, tb, m, n, k, 1.3, A.data(), lda, B.data(), ldb, -0.7, C.data(), m);
                naive_gemm(ta, tb, m, n, k, 1.3, A.data(), lda, B.data(), ldb, -0.7, C_ref.data(), m);
                CHECK(max_diff(C.data(), C_ref.data(), m * n) < 1e-10);
            }
        }
    }

    // Complex GEMM including ConjTrans.
    {
        const keel::usize m = 33, n = 29, k = 25;
        keel::Array<keel::c64> A(k * m); // ConjTrans => A stored as k x m
        keel::Array<keel::c64> B(k * n);
        keel::Array<keel::c64> C(m * n);
        keel::Array<keel::c64> C_ref(m * n);
        for (keel::usize i = 0; i < A.size(); ++i) A[i] = det_cval(i, 1, 41);
        for (keel::usize i = 0; i < B.size(); ++i) B[i] = det_cval(i, 2, 42);
        for (keel::usize i = 0; i < C.size(); ++i) C[i] = C_ref[i] = det_cval(i, 3, 43);

        const keel::c64 alpha(1.2, -0.4);
        const keel::c64 beta(0.3, 0.5);
        keel::gemm(keel::MatrixTranspose::ConjTrans, keel::MatrixTranspose::NoTrans, m, n, k, alpha, A.data(), k,
                   B.data(), k, beta, C.data(), m);
        naive_gemm(keel::MatrixTranspose::ConjTrans, keel::MatrixTranspose::NoTrans, m, n, k, alpha, A.data(), k,
                   B.data(), k, beta, C_ref.data(), m);
        CHECK(max_diff(C.data(), C_ref.data(), m * n) < 1e-10);
    }

    CHECK(keel::linalg_last_gemm_threads_used() >= 1);
}

void test_trsm_and_syrk() {
    // TRSM Left Upper NoTrans: U * X = B => verify U * X == B.
    constexpr keel::usize m = 12;
    constexpr keel::usize n = 7;
    keel::Array<keel::f64> U(m * m, 0.0);
    keel::Array<keel::f64> B(m * n);
    keel::Array<keel::f64> X(m * n);
    for (keel::usize j = 0; j < m; ++j) {
        for (keel::usize i = 0; i <= j; ++i) {
            U[i + j * m] = (i == j) ? (3.0 + static_cast<keel::f64>(i) * 0.2) : det_val(i, j, 55);
        }
    }
    for (keel::usize i = 0; i < m * n; ++i) B[i] = X[i] = det_val(i, 4, 56);

    keel::trsm(keel::MatrixSide::Left, keel::MatrixUplo::Upper, keel::MatrixTranspose::NoTrans,
               keel::MatrixDiag::NonUnit, m, n, 1.0, U.data(), m, X.data(), m);

    keel::Array<keel::f64> UX(m * n, 0.0);
    keel::gemm(keel::MatrixTranspose::NoTrans, keel::MatrixTranspose::NoTrans, m, n, m, 1.0, U.data(), m, X.data(), m,
               0.0, UX.data(), m);
    CHECK(max_diff(UX.data(), B.data(), m * n) < 1e-11);

    // SYRK Upper NoTrans: C = A * A^T.
    constexpr keel::usize sn = 15;
    constexpr keel::usize sk = 9;
    keel::Array<keel::f64> A(sn * sk);
    for (keel::usize i = 0; i < A.size(); ++i) A[i] = det_val(i, 5, 60);
    keel::Array<keel::f64> C(sn * sn, 0.0);
    keel::Array<keel::f64> C_ref(sn * sn, 0.0);
    keel::syrk(keel::MatrixUplo::Upper, keel::MatrixTranspose::NoTrans, sn, sk, 1.0, A.data(), sn, 0.0, C.data(), sn);
    naive_gemm(keel::MatrixTranspose::NoTrans, keel::MatrixTranspose::Trans, sn, sn, sk, 1.0, A.data(), sn, A.data(),
               sn, 0.0, C_ref.data(), sn);
    for (keel::usize j = 0; j < sn; ++j) {
        for (keel::usize i = 0; i <= j; ++i) {
            CHECK(keel::abs(C[i + j * sn] - C_ref[i + j * sn]) < 1e-11);
        }
    }
}

void test_solve_lu_inv() {
    // Blocked square real solve (n = 140 > base_n to exercise recursive LU + GEMM/TRSM).
    {
        constexpr keel::usize n = 140;
        constexpr keel::usize nrhs = 4;
        keel::Array<keel::f64> A(n * n);
        keel::Array<keel::f64> X_true(n * nrhs);
        keel::Array<keel::f64> B(n * nrhs, 0.0);
        keel::Array<keel::f64> X(n * nrhs, 0.0);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) {
                A[i + j * n] = det_val(i, j, 70) + ((i == j) ? 15.0 : 0.0);
            }
        }
        for (keel::usize i = 0; i < n * nrhs; ++i) X_true[i] = det_val(i, 7, 71);
        keel::gemm(keel::MatrixTranspose::NoTrans, keel::MatrixTranspose::NoTrans, n, nrhs, n, 1.0, A.data(), n,
                   X_true.data(), n, 0.0, B.data(), n);

        CHECK(keel::la_solve(A.data(), n, n, B.data(), nrhs, X.data()));
        CHECK(max_diff(X.data(), X_true.data(), n * nrhs) < 1e-11);
    }

    // Complex square solve + matrix_inv.
    {
        constexpr keel::usize n = 18;
        keel::Array<keel::c64> A(n * n);
        keel::Array<keel::c64> A_inv(n * n, keel::c64(0.0, 0.0));
        keel::Array<keel::c64> prod(n * n, keel::c64(0.0, 0.0));
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) {
                A[i + j * n] = det_cval(i, j, 80) + ((i == j) ? keel::c64(8.0, 1.0) : keel::c64(0.0, 0.0));
            }
        }
        CHECK(keel::matrix_inv(A.data(), n, A_inv.data()));
        keel::gemm(keel::MatrixTranspose::NoTrans, keel::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   A.data(), n, A_inv.data(), n, keel::c64(0.0, 0.0), prod.data(), n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) {
                const keel::c64 expected = (i == j) ? keel::c64(1.0, 0.0) : keel::c64(0.0, 0.0);
                CHECK(keel::abs(prod[i + j * n] - expected) < 1e-11);
            }
        }
    }

    // Overdetermined tall least-squares solve (m > n).
    {
        constexpr keel::usize m = 24;
        constexpr keel::usize n = 6;
        keel::Array<keel::f64> A(m * n);
        keel::Array<keel::f64> x_true(n);
        keel::Array<keel::f64> b(m, 0.0);
        keel::Array<keel::f64> x(n, 0.0);
        for (keel::usize j = 0; j < n; ++j) {
            x_true[j] = static_cast<keel::f64>(j + 1);
            for (keel::usize i = 0; i < m; ++i) A[i + j * m] = det_val(i, j, 90);
        }
        keel::gemv(m, n, 1.0, A.data(), m, x_true.data(), 1, 0.0, b.data(), 1);
        CHECK(keel::la_solve(A.data(), m, n, b.data(), 1, x.data()));
        CHECK(max_diff(x.data(), x_true.data(), n) < 1e-11);
    }

    // Singular matrix returns false.
    {
        keel::f64 S[4] = {1.0, 2.0, 2.0, 4.0};
        keel::f64 invS[4] = {};
        CHECK(!keel::matrix_inv(S, 2, invS));
    }
}

void test_cholesky() {
    // Blocked real SPD (n = 90 > kBlock=64).
    {
        constexpr keel::usize n = 90;
        keel::Array<keel::f64> M(n * n);
        keel::Array<keel::f64> A(n * n, 0.0);
        keel::Array<keel::f64> R(n * n, 0.0);
        keel::Array<keel::f64> RtR(n * n, 0.0);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) {
                M[i + j * n] = det_val(i, j, 100) + ((i == j) ? 5.0 : 0.0);
            }
        }
        keel::gemm(keel::MatrixTranspose::Trans, keel::MatrixTranspose::NoTrans, n, n, n, 1.0, M.data(), n, M.data(), n,
                   0.0, A.data(), n);
        CHECK(keel::cholesky_upper(A.data(), R.data(), n) == 0);
        keel::gemm(keel::MatrixTranspose::Trans, keel::MatrixTranspose::NoTrans, n, n, n, 1.0, R.data(), n, R.data(), n,
                   0.0, RtR.data(), n);
        CHECK(max_diff(RtR.data(), A.data(), n * n) < 1e-10);
    }

    // Complex Hermitian positive-definite Cholesky.
    {
        constexpr keel::usize n = 10;
        keel::Array<keel::c64> M(n * n);
        keel::Array<keel::c64> A(n * n, keel::c64(0.0, 0.0));
        keel::Array<keel::c64> R(n * n, keel::c64(0.0, 0.0));
        keel::Array<keel::c64> RhR(n * n, keel::c64(0.0, 0.0));
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) {
                M[i + j * n] = det_cval(i, j, 110) + ((i == j) ? keel::c64(4.0, 0.0) : keel::c64(0.0, 0.0));
            }
        }
        keel::gemm(keel::MatrixTranspose::ConjTrans, keel::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   M.data(), n, M.data(), n, keel::c64(0.0, 0.0), A.data(), n);
        CHECK(keel::cholesky_upper(A.data(), R.data(), n) == 0);
        keel::gemm(keel::MatrixTranspose::ConjTrans, keel::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   R.data(), n, R.data(), n, keel::c64(0.0, 0.0), RhR.data(), n);
        CHECK(max_diff(RhR.data(), A.data(), n * n) < 1e-10);
    }

    // Non-SPD matrix returns positive failure index.
    {
        keel::f64 bad[4] = {1.0, 2.0, 2.0, 1.0};
        keel::f64 R[4] = {};
        CHECK(keel::cholesky_upper(bad, R, 2) == 2);
    }
}

void test_qr() {
    // Real rectangular QR & pivoted QR.
    constexpr keel::usize m = 15;
    constexpr keel::usize n = 8;
    keel::Array<keel::f64> A(m * n);
    for (keel::usize i = 0; i < m * n; ++i) A[i] = det_val(i, 8, 120);

    keel::Array<keel::f64> Q(m * m);
    keel::Array<keel::f64> R(m * n);
    keel::qr_householder(A.data(), m, n, Q.data(), R.data());

    // Q^T * Q == I
    keel::Array<keel::f64> QtQ(m * m, 0.0);
    keel::gemm(keel::MatrixTranspose::Trans, keel::MatrixTranspose::NoTrans, m, m, m, 1.0, Q.data(), m, Q.data(), m,
               0.0, QtQ.data(), m);
    for (keel::usize j = 0; j < m; ++j) {
        for (keel::usize i = 0; i < m; ++i) {
            CHECK(keel::abs(QtQ[i + j * m] - ((i == j) ? 1.0 : 0.0)) < 1e-12);
        }
    }

    // Q * R == A
    keel::Array<keel::f64> QR(m * n, 0.0);
    keel::gemm(keel::MatrixTranspose::NoTrans, keel::MatrixTranspose::NoTrans, m, n, m, 1.0, Q.data(), m, R.data(), m,
               0.0, QR.data(), m);
    CHECK(max_diff(QR.data(), A.data(), m * n) < 1e-12);

    // Pivoted complex QR: A * P == Q * R.
    keel::Array<keel::c64> Ac(m * n);
    for (keel::usize i = 0; i < m * n; ++i) Ac[i] = det_cval(i, 9, 130);
    keel::Array<keel::c64> Qc(m * m);
    keel::Array<keel::c64> Rc(m * n);
    keel::Array<keel::usize> perm(n);
    keel::qr_pivoted(Ac.data(), m, n, Qc.data(), Rc.data(), perm.data());

    keel::Array<keel::c64> QRc(m * n, keel::c64(0.0, 0.0));
    keel::gemm(keel::MatrixTranspose::NoTrans, keel::MatrixTranspose::NoTrans, m, n, m, keel::c64(1.0, 0.0), Qc.data(),
               m, Rc.data(), m, keel::c64(0.0, 0.0), QRc.data(), m);
    for (keel::usize j = 0; j < n; ++j) {
        for (keel::usize i = 0; i < m; ++i) {
            CHECK(keel::abs(QRc[i + j * m] - Ac[i + perm[j] * m]) < 1e-12);
        }
    }
}

void test_svd() {
    // Real rectangular SVD (m > n).
    {
        constexpr keel::usize m = 12;
        constexpr keel::usize n = 7;
        keel::Array<keel::f64> A_orig(m * n);
        keel::Array<keel::f64> U(m * n);
        keel::Array<keel::f64> V(n * n);
        keel::Array<keel::f64> S(n);
        for (keel::usize i = 0; i < m * n; ++i) A_orig[i] = U[i] = det_val(i, 10, 140);

        keel::svd_jacobi_inplace(U.data(), m, n, V.data(), S.data());
        for (keel::usize i = 0; i + 1 < n; ++i) CHECK(S[i] >= S[i + 1]);

        // Reconstruct U * diag(S) * V^T == A_orig.
        keel::Array<keel::f64> US(m * n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) US[i + j * m] = U[i + j * m] * S[j];
        }
        keel::Array<keel::f64> rec(m * n, 0.0);
        keel::gemm(keel::MatrixTranspose::NoTrans, keel::MatrixTranspose::Trans, m, n, n, 1.0, US.data(), m, V.data(),
                   n, 0.0, rec.data(), m);
        CHECK(max_diff(rec.data(), A_orig.data(), m * n) < 1e-11);
    }

    // Complex SVD (m >= n).
    {
        constexpr keel::usize m = 9;
        constexpr keel::usize n = 5;
        keel::Array<keel::c64> A_orig(m * n);
        keel::Array<keel::c64> U(m * n);
        keel::Array<keel::c64> V(n * n);
        keel::Array<keel::f64> S(n);
        for (keel::usize i = 0; i < m * n; ++i) A_orig[i] = U[i] = det_cval(i, 11, 150);

        keel::svd_jacobi_inplace(U.data(), m, n, V.data(), S.data());
        for (keel::usize i = 0; i + 1 < n; ++i) CHECK(S[i] >= S[i + 1]);

        keel::Array<keel::c64> US(m * n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) US[i + j * m] = U[i + j * m] * S[j];
        }
        keel::Array<keel::c64> rec(m * n, keel::c64(0.0, 0.0));
        keel::gemm(keel::MatrixTranspose::NoTrans, keel::MatrixTranspose::ConjTrans, m, n, n, keel::c64(1.0, 0.0),
                   US.data(), m, V.data(), n, keel::c64(0.0, 0.0), rec.data(), m);
        CHECK(max_diff(rec.data(), A_orig.data(), m * n) < 1e-11);
    }
}

} // namespace

int main() {
    test_blas1();
    test_blas2();
    test_gemm();
    test_trsm_and_syrk();
    test_solve_lu_inv();
    test_cholesky();
    test_qr();
    test_svd();

    if (g_failures == 0) {
        std::printf("All keel::linalg tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d test failure(s).\n", g_failures);
    return 1;
}
