#include "keel/array.h"
#include "keel/complex.h"
#include "keel/format.h"
#include "keel/linalg.h"
#include "keel/math.h"

namespace {

keel::i32 g_failures = 0;

void check(bool cond, const char* expr, keel::i32 line) {
    if (!cond) {
        keel::eprintln("FAIL [line {}]: {}", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

keel::f64 det_val(keel::usize i, keel::usize j, keel::usize seed) {
    keel::u64 z = keel::as<keel::u64>(i) * 0x9e3779b97f4a7c15ULL +
                  keel::as<keel::u64>(j) * 0xbf58476d1ce4e5b9ULL +
                  keel::as<keel::u64>(seed) * 0x94d049bb133111ebULL + 0x123456789abcdef0ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= (z >> 31);
    return (keel::as<keel::f64>(z >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
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
T op_elem(const T* M, keel::usize ld, keel::usize r, keel::usize c, keel::linalg::MatrixTranspose tr) {
    if (tr == keel::linalg::MatrixTranspose::NoTrans) return M[r + c * ld];
    const T v = M[c + r * ld];
    if constexpr (keel::IsSame<T, keel::c64>) {
        return (tr == keel::linalg::MatrixTranspose::ConjTrans) ? keel::conj(v) : v;
    } else {
        return v;
    }
}

template <typename T>
void naive_gemm(keel::linalg::MatrixTranspose ta, keel::linalg::MatrixTranspose tb, keel::usize m, keel::usize n, keel::usize k,
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
    keel::linalg::axpy(5, 2.5, x, y);
    for (keel::usize i = 0; i < 5; ++i) {
        CHECK(keel::abs(y[i] - (10.0 * keel::as<keel::f64>(i + 1) + 2.5 * x[i])) < 1e-12);
    }

    CHECK(keel::abs(keel::linalg::dot(5, x, x) - 55.0) < 1e-12);
    CHECK(keel::abs(keel::linalg::nrm2(5, x) - keel::sqrt(55.0)) < 1e-12);

    // Overflow-safe nrm2 test.
    keel::f64 huge[2] = {3e200, 4e200};
    CHECK(keel::abs(keel::linalg::nrm2(2, huge) - 5e200) < 1e188);

    keel::linalg::scal(5, 3.0, x);
    CHECK(keel::abs(x[4] - 15.0) < 1e-12);

    keel::c64 cx[3] = {keel::c64(1.0, 2.0), keel::c64(3.0, -1.0), keel::c64(0.0, 4.0)};
    keel::c64 cy[3] = {keel::c64(2.0, -1.0), keel::c64(1.0, 1.0), keel::c64(-2.0, 3.0)};
    const keel::c64 dc = keel::linalg::dotc(3, cx, cy);
    // conj(1+2i)*(2-i) = (1-2i)(2-i) = 0 - 5i
    // conj(3-i)*(1+i)  = (3+i)(1+i)  = 2 + 4i
    // conj(4i)*(-2+3i) = (-4i)(-2+3i) = 12 + 8i
    CHECK(keel::abs(dc - keel::c64(14.0, 7.0)) < 1e-12);

    const keel::c64 du = keel::linalg::dotu(3, cx, cy);
    CHECK(keel::abs(du - keel::c64(-4.0, -3.0)) < 1e-12);
    CHECK(keel::abs(keel::linalg::nrm2(3, cx) - keel::sqrt(31.0)) < 1e-12);

    keel::linalg::axpy(3, keel::c64(2.0, 1.0), cx, cy);
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

    keel::linalg::gemv(m, n, 2.0, A.data(), m, x.data(), 1, 0.5, y.data(), 1);
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
    keel::linalg::gemv(keel::linalg::MatrixTranspose::Trans, m, n, -1.5, A.data(), m, xt.data(), 1, 0.0, yt.data(), 1);
    for (keel::usize j = 0; j < n; ++j) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < m; ++i) s += A[i + j * m] * xt[i];
        yt_ref[j] = -1.5 * s;
    }
    CHECK(max_diff(yt.data(), yt_ref.data(), n) < 1e-12);

    // Rank-1 ger.
    keel::Array<keel::f64> A_ref(m * n);
    for (keel::usize i = 0; i < m * n; ++i) A_ref[i] = A[i];
    keel::linalg::ger(m, n, 0.75, xt.data(), 1, x.data(), 1, A.data(), m);
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
    const keel::linalg::MatrixTranspose trs[2] = {keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans};

    for (keel::usize d = 0; d < 2; ++d) {
        const keel::usize m = dims[d][0];
        const keel::usize n = dims[d][1];
        const keel::usize k = dims[d][2];
        for (keel::usize a_t = 0; a_t < 2; ++a_t) {
            for (keel::usize b_t = 0; b_t < 2; ++b_t) {
                const auto ta = trs[a_t];
                const auto tb = trs[b_t];
                const keel::usize lda = (ta == keel::linalg::MatrixTranspose::NoTrans) ? m : k;
                const keel::usize colsA = (ta == keel::linalg::MatrixTranspose::NoTrans) ? k : m;
                const keel::usize ldb = (tb == keel::linalg::MatrixTranspose::NoTrans) ? k : n;
                const keel::usize colsB = (tb == keel::linalg::MatrixTranspose::NoTrans) ? n : k;

                keel::Array<keel::f64> A(lda * colsA);
                keel::Array<keel::f64> B(ldb * colsB);
                keel::Array<keel::f64> C(m * n);
                keel::Array<keel::f64> C_ref(m * n);
                for (keel::usize i = 0; i < A.size(); ++i) A[i] = det_val(i, 1, 10 + d);
                for (keel::usize i = 0; i < B.size(); ++i) B[i] = det_val(i, 2, 20 + d);
                for (keel::usize i = 0; i < C.size(); ++i) C[i] = C_ref[i] = det_val(i, 3, 30 + d);

                keel::linalg::gemm(ta, tb, m, n, k, 1.3, A.data(), lda, B.data(), ldb, -0.7, C.data(), m);
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
        keel::linalg::gemm(keel::linalg::MatrixTranspose::ConjTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, k, alpha, A.data(), k,
                   B.data(), k, beta, C.data(), m);
        naive_gemm(keel::linalg::MatrixTranspose::ConjTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, k, alpha, A.data(), k,
                   B.data(), k, beta, C_ref.data(), m);
        CHECK(max_diff(C.data(), C_ref.data(), m * n) < 1e-10);
    }

    CHECK(keel::linalg::linalg_last_gemm_threads_used() >= 1);
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
            U[i + j * m] = (i == j) ? (3.0 + keel::as<keel::f64>(i) * 0.2) : det_val(i, j, 55);
        }
    }
    for (keel::usize i = 0; i < m * n; ++i) B[i] = X[i] = det_val(i, 4, 56);

    keel::linalg::trsm(keel::linalg::MatrixSide::Left, keel::linalg::MatrixUplo::Upper, keel::linalg::MatrixTranspose::NoTrans,
               keel::linalg::MatrixDiag::NonUnit, m, n, 1.0, U.data(), m, X.data(), m);

    keel::Array<keel::f64> UX(m * n, 0.0);
    keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, 1.0, U.data(), m, X.data(), m,
               0.0, UX.data(), m);
    CHECK(max_diff(UX.data(), B.data(), m * n) < 1e-11);

    // SYRK Upper NoTrans: C = A * A^T.
    constexpr keel::usize sn = 15;
    constexpr keel::usize sk = 9;
    keel::Array<keel::f64> A(sn * sk);
    for (keel::usize i = 0; i < A.size(); ++i) A[i] = det_val(i, 5, 60);
    keel::Array<keel::f64> C(sn * sn, 0.0);
    keel::Array<keel::f64> C_ref(sn * sn, 0.0);
    keel::linalg::syrk(keel::linalg::MatrixUplo::Upper, keel::linalg::MatrixTranspose::NoTrans, sn, sk, 1.0, A.data(), sn, 0.0, C.data(), sn);
    naive_gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans, sn, sn, sk, 1.0, A.data(), sn, A.data(),
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
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, nrhs, n, 1.0, A.data(), n,
                   X_true.data(), n, 0.0, B.data(), n);

        CHECK(keel::linalg::la_solve(A.data(), n, n, B.data(), nrhs, X.data()));
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
        CHECK(keel::linalg::matrix_inv(A.data(), n, A_inv.data()));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
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
            x_true[j] = keel::as<keel::f64>(j + 1);
            for (keel::usize i = 0; i < m; ++i) A[i + j * m] = det_val(i, j, 90);
        }
        keel::linalg::gemv(m, n, 1.0, A.data(), m, x_true.data(), 1, 0.0, b.data(), 1);
        CHECK(keel::linalg::la_solve(A.data(), m, n, b.data(), 1, x.data()));
        CHECK(max_diff(x.data(), x_true.data(), n) < 1e-11);
    }

    // Singular matrix returns false.
    {
        keel::f64 S[4] = {1.0, 2.0, 2.0, 4.0};
        keel::f64 invS[4] = {};
        CHECK(!keel::linalg::matrix_inv(S, 2, invS));
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
        keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, M.data(), n, M.data(), n,
                   0.0, A.data(), n);
        CHECK(keel::linalg::cholesky_upper(A.data(), R.data(), n) == 0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, R.data(), n, R.data(), n,
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
        keel::linalg::gemm(keel::linalg::MatrixTranspose::ConjTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   M.data(), n, M.data(), n, keel::c64(0.0, 0.0), A.data(), n);
        CHECK(keel::linalg::cholesky_upper(A.data(), R.data(), n) == 0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::ConjTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   R.data(), n, R.data(), n, keel::c64(0.0, 0.0), RhR.data(), n);
        CHECK(max_diff(RhR.data(), A.data(), n * n) < 1e-10);
    }

    // Non-SPD matrix returns positive failure index.
    {
        keel::f64 bad[4] = {1.0, 2.0, 2.0, 1.0};
        keel::f64 R[4] = {};
        CHECK(keel::linalg::cholesky_upper(bad, R, 2) == 2);
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
    keel::linalg::qr_householder(A.data(), m, n, Q.data(), R.data());

    // Q^T * Q == I
    keel::Array<keel::f64> QtQ(m * m, 0.0);
    keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans, m, m, m, 1.0, Q.data(), m, Q.data(), m,
               0.0, QtQ.data(), m);
    for (keel::usize j = 0; j < m; ++j) {
        for (keel::usize i = 0; i < m; ++i) {
            CHECK(keel::abs(QtQ[i + j * m] - ((i == j) ? 1.0 : 0.0)) < 1e-12);
        }
    }

    // Q * R == A
    keel::Array<keel::f64> QR(m * n, 0.0);
    keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, 1.0, Q.data(), m, R.data(), m,
               0.0, QR.data(), m);
    CHECK(max_diff(QR.data(), A.data(), m * n) < 1e-12);

    // Pivoted complex QR: A * P == Q * R.
    keel::Array<keel::c64> Ac(m * n);
    for (keel::usize i = 0; i < m * n; ++i) Ac[i] = det_cval(i, 9, 130);
    keel::Array<keel::c64> Qc(m * m);
    keel::Array<keel::c64> Rc(m * n);
    keel::Array<keel::usize> perm(n);
    keel::linalg::qr_pivoted(Ac.data(), m, n, Qc.data(), Rc.data(), perm.data());

    keel::Array<keel::c64> QRc(m * n, keel::c64(0.0, 0.0));
    keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, keel::c64(1.0, 0.0), Qc.data(),
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

        keel::linalg::svd_jacobi_inplace(U.data(), m, n, V.data(), S.data());
        for (keel::usize i = 0; i + 1 < n; ++i) CHECK(S[i] >= S[i + 1]);

        // Reconstruct U * diag(S) * V^T == A_orig.
        keel::Array<keel::f64> US(m * n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) US[i + j * m] = U[i + j * m] * S[j];
        }
        keel::Array<keel::f64> rec(m * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans, m, n, n, 1.0, US.data(), m, V.data(),
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

        keel::linalg::svd_jacobi_inplace(U.data(), m, n, V.data(), S.data());
        for (keel::usize i = 0; i + 1 < n; ++i) CHECK(S[i] >= S[i + 1]);

        keel::Array<keel::c64> US(m * n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) US[i + j * m] = U[i + j * m] * S[j];
        }
        keel::Array<keel::c64> rec(m * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::ConjTrans, m, n, n, keel::c64(1.0, 0.0),
                   US.data(), m, V.data(), n, keel::c64(0.0, 0.0), rec.data(), m);
        CHECK(max_diff(rec.data(), A_orig.data(), m * n) < 1e-11);
    }

    // Wide complex svd_full (m < n) + tall real svd_full
    {
        const keel::usize m = 4;
        const keel::usize n = 7;
        keel::Array<keel::c64> A(m * n);
        keel::Array<keel::c64> U(m * m);
        keel::Array<keel::f64> S(m);
        keel::Array<keel::c64> V(n * n);
        for (keel::usize i = 0; i < m * n; ++i) A[i] = det_cval(i, 17, 210);

        keel::linalg::svd_full(A.data(), m, n, U.data(), S.data(), V.data());
        for (keel::usize i = 0; i + 1 < m; ++i) CHECK(S[i] >= S[i + 1]);

        keel::Array<keel::c64> Smat(m * n, keel::c64(0.0, 0.0));
        for (keel::usize i = 0; i < m; ++i) Smat[i + i * m] = keel::c64(S[i], 0.0);
        keel::Array<keel::c64> US(m * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, keel::c64(1.0, 0.0),
                   U.data(), m, Smat.data(), m, keel::c64(0.0, 0.0), US.data(), m);
        keel::Array<keel::c64> rec(m * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::ConjTrans, m, n, n, keel::c64(1.0, 0.0),
                   US.data(), m, V.data(), n, keel::c64(0.0, 0.0), rec.data(), m);
        CHECK(max_diff(rec.data(), A.data(), m * n) < 1e-11);

        keel::Array<keel::c64> VtV(n * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::ConjTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   V.data(), n, V.data(), n, keel::c64(0.0, 0.0), VtV.data(), n);
        keel::Array<keel::c64> eye(n * n, keel::c64(0.0, 0.0));
        for (keel::usize i = 0; i < n; ++i) eye[i + i * n] = keel::c64(1.0, 0.0);
        CHECK(max_diff(VtV.data(), eye.data(), n * n) < 1e-11);
    }

    // Rank-1 square svd_full (verifies fast zero-column handling and full U/V orthogonality)
    {
        const keel::usize n = 64;
        keel::Array<keel::f64> A(n * n, 0.0);
        for (keel::usize j = 16; j < 48; ++j) {
            for (keel::usize i = 16; i < 48; ++i) {
                A[i + j * n] = 1.0;
            }
        }
        keel::Array<keel::f64> U(n * n, 0.0);
        keel::Array<keel::f64> S(n, 0.0);
        keel::Array<keel::f64> V(n * n, 0.0);
        keel::linalg::svd_full(A.data(), n, n, U.data(), S.data(), V.data());
        CHECK(keel::abs(S[0] - 32.0) < 1e-11);
        for (keel::usize k = 1; k < n; ++k) CHECK(S[k] < 1e-11);

        keel::Array<keel::f64> Smat(n * n, 0.0);
        for (keel::usize i = 0; i < n; ++i) Smat[i + i * n] = S[i];
        keel::Array<keel::f64> US(n * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans,
                           n, n, n, 1.0, U.data(), n, Smat.data(), n, 0.0, US.data(), n);
        keel::Array<keel::f64> rec(n * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans,
                           n, n, n, 1.0, US.data(), n, V.data(), n, 0.0, rec.data(), n);
        CHECK(max_diff(rec.data(), A.data(), n * n) < 1e-11);

        keel::Array<keel::f64> UtU(n * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans,
                           n, n, n, 1.0, U.data(), n, U.data(), n, 0.0, UtU.data(), n);
        keel::Array<keel::f64> eye(n * n, 0.0);
        for (keel::usize i = 0; i < n; ++i) eye[i + i * n] = 1.0;
        CHECK(max_diff(UtU.data(), eye.data(), n * n) < 1e-11);
    }
}

void test_factor_updates_ldl_balance() {
    // 1. Wide + singular lu_factor_general
    {
        const keel::usize m = 3;
        const keel::usize n = 5;
        keel::Array<keel::f64> A(m * n);
        for (keel::usize i = 0; i < m * n; ++i) A[i] = det_val(i, 7, 300);
        keel::Array<keel::f64> LU(m * n);
        for (keel::usize i = 0; i < m * n; ++i) LU[i] = A[i];
        keel::Array<keel::i32> piv(m);
        CHECK(keel::linalg::lu_factor_general(LU.data(), m, n, piv.data()));

        keel::Array<keel::f64> L(m * m, 0.0);
        keel::Array<keel::f64> U(m * n, 0.0);
        for (keel::usize j = 0; j < m; ++j) {
            L[j + j * m] = 1.0;
            for (keel::usize i = j + 1; i < m; ++i) L[i + j * m] = LU[i + j * m];
        }
        for (keel::usize j = 0; j < n; ++j) {
            const keel::usize lim = (j + 1 < m) ? (j + 1) : m;
            for (keel::usize i = 0; i < lim; ++i) U[i + j * m] = LU[i + j * m];
        }
        keel::Array<keel::f64> prod(m * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, 1.0, L.data(), m, U.data(),
                   m, 0.0, prod.data(), m);
        keel::Array<keel::f64> PA(m * n);
        for (keel::usize i = 0; i < m * n; ++i) PA[i] = A[i];
        for (keel::usize j = 0; j < m; ++j) {
            const keel::usize p = keel::as<keel::usize>(piv[j]);
            if (p != j) {
                for (keel::usize c = 0; c < n; ++c) keel::swap(PA[j + c * m], PA[p + c * m]);
            }
        }
        CHECK(max_diff(prod.data(), PA.data(), m * n) < 1e-12);
    }

    // 2. qr_update, qr_insert_col, qr_delete_col, qr_insert_row, qr_delete_row
    {
        const keel::usize m = 6;
        const keel::usize n = 4;
        keel::Array<keel::f64> A(m * n);
        for (keel::usize i = 0; i < m * n; ++i) A[i] = det_val(i, 13, 320);
        keel::Array<keel::f64> Q(m * m);
        keel::Array<keel::f64> R(m * n);
        keel::linalg::qr_householder(A.data(), m, n, Q.data(), R.data());

        // Rank-1 update A + u*v^T
        keel::Array<keel::f64> u(m);
        keel::Array<keel::f64> v(n);
        for (keel::usize i = 0; i < m; ++i) u[i] = det_val(i, 3, 340);
        for (keel::usize j = 0; j < n; ++j) v[j] = det_val(j, 5, 350);
        keel::Array<keel::f64> A_upd(m * n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) A_upd[i + j * m] = A[i + j * m] + u[i] * v[j];
        }
        keel::Array<keel::f64> Qu(m * m);
        keel::Array<keel::f64> Ru(m * n);
        for (keel::usize i = 0; i < m * m; ++i) Qu[i] = Q[i];
        for (keel::usize i = 0; i < m * n; ++i) Ru[i] = R[i];
        keel::linalg::qr_update(Qu.data(), Ru.data(), m, n, u.data(), v.data());
        keel::Array<keel::f64> rec(m * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, 1.0, Qu.data(), m,
                   Ru.data(), m, 0.0, rec.data(), m);
        CHECK(max_diff(rec.data(), A_upd.data(), m * n) < 1e-11);

        // Insert col then delete col -> recovers A
        keel::Array<keel::f64> Qi(m * m);
        keel::Array<keel::f64> Ri(m * (n + 1));
        keel::linalg::qr_insert_col(Q.data(), R.data(), m, n, 2, u.data(), Qi.data(), Ri.data());
        keel::Array<keel::f64> Qd(m * m);
        keel::Array<keel::f64> Rd(m * n);
        keel::linalg::qr_delete_col(Qi.data(), Ri.data(), m, n + 1, 2, Qd.data(), Rd.data());
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, 1.0, Qd.data(), m,
                   Rd.data(), m, 0.0, rec.data(), m);
        CHECK(max_diff(rec.data(), A.data(), m * n) < 1e-11);

        // Insert row then delete row -> recovers A
        keel::Array<keel::f64> Qir((m + 1) * (m + 1));
        keel::Array<keel::f64> Rir((m + 1) * n);
        keel::linalg::qr_insert_row(Q.data(), R.data(), m, n, 3, v.data(), Qir.data(), Rir.data());
        keel::Array<keel::f64> Qdr(m * m);
        keel::Array<keel::f64> Rdr(m * n);
        keel::linalg::qr_delete_row(Qir.data(), Rir.data(), m + 1, n, 3, Qdr.data(), Rdr.data());
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, 1.0, Qdr.data(), m,
                   Rdr.data(), m, 0.0, rec.data(), m);
        CHECK(max_diff(rec.data(), A.data(), m * n) < 1e-11);
    }

    // 3. cholesky_update (update + downdate + failure)
    {
        const keel::usize n = 6;
        keel::Array<keel::f64> M(n * n);
        for (keel::usize i = 0; i < n * n; ++i) M[i] = det_val(i, 5, 360);
        keel::Array<keel::f64> A(n * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, M.data(), n, M.data(), n,
                   0.0, A.data(), n);
        for (keel::usize i = 0; i < n; ++i) A[i + i * n] += 4.0;
        keel::Array<keel::f64> R(n * n);
        CHECK(keel::linalg::cholesky_upper(A.data(), R.data(), n) == 0);

        keel::Array<keel::f64> x(n);
        for (keel::usize i = 0; i < n; ++i) x[i] = 0.5 * det_val(i, 7, 370);
        CHECK(keel::linalg::cholesky_update(R.data(), n, x.data(), false));

        keel::Array<keel::f64> RtR(n * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, R.data(), n, R.data(), n,
                   0.0, RtR.data(), n);
        keel::Array<keel::f64> expected(n * n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) expected[i + j * n] = A[i + j * n] + x[i] * x[j];
        }
        CHECK(max_diff(RtR.data(), expected.data(), n * n) < 1e-11);

        // Downdate back to A
        CHECK(keel::linalg::cholesky_update(R.data(), n, x.data(), true));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, R.data(), n, R.data(), n,
                   0.0, RtR.data(), n);
        CHECK(max_diff(RtR.data(), A.data(), n * n) < 1e-11);

        // Excessively large downdate must fail cleanly
        keel::Array<keel::f64> big(n, 100.0);
        CHECK(!keel::linalg::cholesky_update(R.data(), n, big.data(), true));
    }

    // 4. ldl_factor (real symmetric indefinite & complex Hermitian)
    {
        const keel::usize n = 5;
        keel::Array<keel::f64> A(n * n, 0.0);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i <= j; ++i) {
                const keel::f64 v = det_val(i + j * n, 11, 380);
                A[i + j * n] = v;
                A[j + i * n] = v;
            }
            A[j + j * n] = (j % 2 == 0) ? 5.0 : -4.0;
        }
        keel::Array<keel::f64> L(n * n);
        keel::Array<keel::f64> D(n);
        CHECK(keel::linalg::ldl_factor(A.data(), n, L.data(), D.data()) == 0);
        keel::Array<keel::f64> LD(n * n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) LD[i + j * n] = L[i + j * n] * D[j];
        }
        keel::Array<keel::f64> rec(n * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans, n, n, n, 1.0, LD.data(), n, L.data(),
                   n, 0.0, rec.data(), n);
        CHECK(max_diff(rec.data(), A.data(), n * n) < 1e-11);

        keel::Array<keel::c64> H(n * n, keel::c64(0.0, 0.0));
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < j; ++i) {
                const keel::c64 v = det_cval(i + j * n, 13, 390);
                H[i + j * n] = v;
                H[j + i * n] = keel::conj(v);
            }
            H[j + j * n] = keel::c64((j % 2 == 0) ? 6.0 : -5.0, 0.0);
        }
        keel::Array<keel::c64> Lc(n * n);
        CHECK(keel::linalg::ldl_factor(H.data(), n, Lc.data(), D.data()) == 0);
        keel::Array<keel::c64> LDc(n * n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) LDc[i + j * n] = Lc[i + j * n] * D[j];
        }
        keel::Array<keel::c64> recc(n * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::ConjTrans, n, n, n, keel::c64(1.0, 0.0),
                   LDc.data(), n, Lc.data(), n, keel::c64(0.0, 0.0), recc.data(), n);
        CHECK(max_diff(recc.data(), H.data(), n * n) < 1e-11);
    }

    // 5. balance_matrix (diagonal scaling similarity B = D^{-1} A D)
    {
        const keel::usize n = 3;
        keel::f64 A[9] = {1.0, 1e-4, 0.0, 1e4, 2.0, 1e-3, 0.0, 1e3, 3.0};
        keel::f64 B[9];
        for (keel::usize i = 0; i < 9; ++i) B[i] = A[i];
        keel::f64 d[3];
        keel::linalg::balance_matrix(B, n, false, d);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 reconstructed = d[i] * B[i + j * n] / d[j];
                CHECK(keel::abs(reconstructed - A[i + j * n]) <= 1e-12 * (1.0 + keel::abs(A[i + j * n])));
            }
        }
    }
}

void test_eig_schur_qz() {
    // 1. eig_sym (both Jacobi n=6 and tridiag QL n=24)
    for (keel::usize n : {keel::usize{6}, keel::usize{24}}) {
        keel::Array<keel::f64> A(n * n, 0.0);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i <= j; ++i) {
                const keel::f64 v = det_val(i + j * n, 17, 410);
                A[i + j * n] = v;
                A[j + i * n] = v;
            }
        }
        keel::Array<keel::f64> V(n * n);
        keel::Array<keel::f64> D(n);
        keel::linalg::eig_sym(A.data(), n, V.data(), D.data());
        for (keel::usize i = 0; i + 1 < n; ++i) CHECK(D[i] <= D[i + 1] + 1e-12);

        keel::Array<keel::f64> VD(n * n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) VD[i + j * n] = V[i + j * n] * D[j];
        }
        keel::Array<keel::f64> rec(n * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans, n, n, n, 1.0, VD.data(), n, V.data(),
                   n, 0.0, rec.data(), n);
        CHECK(max_diff(rec.data(), A.data(), n * n) < 1e-10);
    }

    // 2. eig_sym_generalized & eigs_sym_generalized_smallest_abs
    {
        const keel::usize n = 10;
        keel::Array<keel::f64> M1(n * n), M2(n * n), A(n * n, 0.0), B(n * n, 0.0);
        for (keel::usize i = 0; i < n * n; ++i) {
            M1[i] = det_val(i, 5, 420);
            M2[i] = det_val(i, 11, 430);
        }
        keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, M1.data(), n, M1.data(),
                   n, 0.0, A.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, M2.data(), n, M2.data(),
                   n, 0.0, B.data(), n);
        for (keel::usize i = 0; i < n; ++i) {
            A[i + i * n] += 5.0;
            B[i + i * n] += 4.0;
        }
        keel::Array<keel::f64> V(n * n), D(n);
        CHECK(keel::linalg::eig_sym_generalized(A.data(), B.data(), n, V.data(), D.data()));
        // Check A*V = B*V*D
        keel::Array<keel::f64> AV(n * n, 0.0), BV(n * n, 0.0), BVD(n * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, A.data(), n, V.data(),
                   n, 0.0, AV.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, B.data(), n, V.data(),
                   n, 0.0, BV.data(), n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) BVD[i + j * n] = BV[i + j * n] * D[j];
        }
        CHECK(max_diff(AV.data(), BVD.data(), n * n) < 1e-10);

        const keel::usize k = 3;
        keel::Array<keel::f64> Vk(n * k), Dk(k);
        CHECK(keel::linalg::eigs_sym_generalized_smallest_abs(A.data(), B.data(), n, k, Vk.data(), Dk.data()));
        for (keel::usize i = 0; i < k; ++i) CHECK(keel::abs(Dk[i] - D[i]) < 1e-9);
    }

    // 3. schur_real, rsf2csf, schur_complex, schur_eigenvectors, ordschur_complex, cdf2rdf
    {
        const keel::usize n = 6;
        keel::Array<keel::f64> A(n * n);
        for (keel::usize i = 0; i < n * n; ++i) A[i] = det_val(i, 19, 440);
        keel::Array<keel::f64> U(n * n), T(n * n);
        keel::linalg::schur_real(A.data(), n, U.data(), T.data());

        keel::Array<keel::f64> UT(n * n, 0.0), rec(n * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, 1.0, U.data(), n, T.data(),
                   n, 0.0, UT.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans, n, n, n, 1.0, UT.data(), n, U.data(),
                   n, 0.0, rec.data(), n);
        CHECK(max_diff(rec.data(), A.data(), n * n) < 1e-11);

        // rsf2csf: UC * TC * UC^H == A
        keel::Array<keel::c64> UC(n * n), TC(n * n);
        keel::linalg::rsf2csf(U.data(), T.data(), n, UC.data(), TC.data());
        keel::Array<keel::c64> Ac(n * n);
        for (keel::usize i = 0; i < n * n; ++i) Ac[i] = keel::c64(A[i], 0.0);
        keel::Array<keel::c64> UCTC(n * n, keel::c64(0.0, 0.0)), recc(n * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   UC.data(), n, TC.data(), n, keel::c64(0.0, 0.0), UCTC.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::ConjTrans, n, n, n, keel::c64(1.0, 0.0),
                   UCTC.data(), n, UC.data(), n, keel::c64(0.0, 0.0), recc.data(), n);
        CHECK(max_diff(recc.data(), Ac.data(), n * n) < 1e-11);

        // schur_complex + schur_eigenvectors + ordschur_complex
        keel::Array<keel::c64> Us(n * n), Ts(n * n);
        CHECK(keel::linalg::schur_complex(Ac.data(), n, Us.data(), Ts.data()));
        keel::Array<keel::c64> Vc(n * n), evals(n);
        keel::linalg::schur_eigenvectors(Us.data(), Ts.data(), n, Vc.data(), evals.data());
        keel::Array<keel::c64> AV(n * n, keel::c64(0.0, 0.0)), VD(n * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   Ac.data(), n, Vc.data(), n, keel::c64(0.0, 0.0), AV.data(), n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) VD[i + j * n] = Vc[i + j * n] * evals[j];
        }
        CHECK(max_diff(AV.data(), VD.data(), n * n) < 1e-10);

        keel::Array<keel::u8> sel(n, keel::u8{0});
        keel::usize n_sel = 0;
        for (keel::usize i = 0; i < n; ++i) {
            if (Ts[i + i * n].real() < 0.0) {
                sel[i] = keel::u8{1};
                ++n_sel;
            }
        }
        keel::linalg::ordschur_complex(Us.data(), Ts.data(), n, sel.data());
        for (keel::usize i = 0; i < n_sel; ++i) CHECK(Ts[i + i * n].real() < 0.0);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = j + 1; i < n; ++i) CHECK(keel::abs(Ts[i + j * n]) < 1e-11);
        }
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   Us.data(), n, Ts.data(), n, keel::c64(0.0, 0.0), UCTC.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::ConjTrans, n, n, n, keel::c64(1.0, 0.0),
                   UCTC.data(), n, Us.data(), n, keel::c64(0.0, 0.0), recc.data(), n);
        CHECK(max_diff(recc.data(), Ac.data(), n * n) < 1e-10);
    }

    // 4. qz_factor, qz_eigenvectors, ordqz_complex
    {
        const keel::usize n = 5;
        keel::Array<keel::c64> A(n * n), B(n * n);
        for (keel::usize i = 0; i < n * n; ++i) {
            A[i] = det_cval(i, 7, 460);
            B[i] = det_cval(i, 13, 470);
        }
        for (keel::usize i = 0; i < n; ++i) B[i + i * n] += keel::c64(3.0, 0.0);

        keel::Array<keel::c64> AA(n * n), BB(n * n), Q(n * n), Z(n * n);
        CHECK(keel::linalg::qz_factor(A.data(), B.data(), n, AA.data(), BB.data(), Q.data(), Z.data()));

        // Verify Q * A * Z == AA and Q * B * Z == BB
        keel::Array<keel::c64> QA(n * n, keel::c64(0.0, 0.0)), QAZ(n * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   Q.data(), n, A.data(), n, keel::c64(0.0, 0.0), QA.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   QA.data(), n, Z.data(), n, keel::c64(0.0, 0.0), QAZ.data(), n);
        CHECK(max_diff(QAZ.data(), AA.data(), n * n) < 1e-10);

        keel::Array<keel::c64> V(n * n), evals(n);
        keel::linalg::qz_eigenvectors(AA.data(), BB.data(), Z.data(), n, V.data(), evals.data());
        keel::Array<keel::c64> AV(n * n, keel::c64(0.0, 0.0)), BV(n * n, keel::c64(0.0, 0.0)), BVD(n * n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   A.data(), n, V.data(), n, keel::c64(0.0, 0.0), AV.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   B.data(), n, V.data(), n, keel::c64(0.0, 0.0), BV.data(), n);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) BVD[i + j * n] = BV[i + j * n] * evals[j];
        }
        CHECK(max_diff(AV.data(), BVD.data(), n * n) < 1e-10);

        const keel::c64 target_eig = AA[(n - 1) + (n - 1) * n] / BB[(n - 1) + (n - 1) * n];
        keel::Array<keel::u8> sel(n, keel::u8{0});
        sel[n - 1] = keel::u8{1};
        keel::linalg::ordqz_complex(AA.data(), BB.data(), Q.data(), Z.data(), n, sel.data());
        CHECK(keel::abs(AA[0] / BB[0] - target_eig) < 1e-10);
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = j + 1; i < n; ++i) {
                CHECK(keel::abs(AA[i + j * n]) < 1e-10);
                CHECK(keel::abs(BB[i + j * n]) < 1e-10);
            }
        }
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   Q.data(), n, A.data(), n, keel::c64(0.0, 0.0), QA.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   QA.data(), n, Z.data(), n, keel::c64(0.0, 0.0), QAZ.data(), n);
        CHECK(max_diff(QAZ.data(), AA.data(), n * n) < 1e-10);
    }
}

void test_matrix_functions_and_solvers() {
    // 1. expm, logm, sqrtm, funm
    {
        const keel::usize n = 4;
        keel::Array<keel::f64> A(n * n);
        for (keel::usize i = 0; i < n * n; ++i) A[i] = 0.25 * det_val(i, 7, 500);
        for (keel::usize i = 0; i < n; ++i) A[i + i * n] += 2.0;

        keel::Array<keel::f64> E(n * n);
        CHECK(keel::linalg::expm(A.data(), n, E.data()));

        keel::Array<keel::c64> Ac(n * n), Ec(n * n), Lc(n * n), Sc(n * n);
        for (keel::usize i = 0; i < n * n; ++i) {
            Ac[i] = keel::c64(A[i], 0.0);
            Ec[i] = keel::c64(E[i], 0.0);
        }
        CHECK(keel::linalg::logm(Ec.data(), n, Lc.data()));
        CHECK(max_diff(Lc.data(), Ac.data(), n * n) < 1e-10);

        CHECK(keel::linalg::sqrtm(Ac.data(), n, Sc.data()));
        keel::Array<keel::c64> S2(n * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   Sc.data(), n, Sc.data(), n, keel::c64(0.0, 0.0), S2.data(), n);
        CHECK(max_diff(S2.data(), Ac.data(), n * n) < 1e-10);

        // funm: sin(A)^2 + cos(A)^2 == I
        keel::Array<keel::c64> SinA(n * n), CosA(n * n), sum(n * n, keel::c64(0.0, 0.0));
        CHECK(keel::linalg::funm(Ac.data(), n, [](keel::c64 z) { return keel::sin(z); }, SinA.data()));
        CHECK(keel::linalg::funm(Ac.data(), n, [](keel::c64 z) { return keel::cos(z); }, CosA.data()));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   SinA.data(), n, SinA.data(), n, keel::c64(0.0, 0.0), sum.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, n, n, n, keel::c64(1.0, 0.0),
                   CosA.data(), n, CosA.data(), n, keel::c64(1.0, 0.0), sum.data(), n);
        keel::Array<keel::c64> eye(n * n, keel::c64(0.0, 0.0));
        for (keel::usize i = 0; i < n; ++i) eye[i + i * n] = keel::c64(1.0, 0.0);
        CHECK(max_diff(sum.data(), eye.data(), n * n) < 1e-9);

        // expmv vs expm(t * A) * v
        keel::Array<keel::f64> v(n), y_mv(n), y_ref(n, 0.0), tA(n * n), EtA(n * n);
        const keel::f64 t = 0.7;
        for (keel::usize i = 0; i < n; ++i) v[i] = det_val(i, 3, 510);
        for (keel::usize i = 0; i < n * n; ++i) tA[i] = t * A[i];
        CHECK(keel::linalg::expm(tA.data(), n, EtA.data()));
        keel::linalg::gemv(keel::linalg::MatrixTranspose::NoTrans, n, n, 1.0, EtA.data(), n, v.data(), 1, 0.0, y_ref.data(), 1);
        CHECK(keel::linalg::expmv(t, A.data(), n, v.data(), y_mv.data()));
        CHECK(max_diff(y_mv.data(), y_ref.data(), n) < 1e-10);
    }

    // 2. sylvester: A * X + X * B == C
    {
        const keel::usize m = 4;
        const keel::usize n = 3;
        keel::Array<keel::c64> A(m * m), B(n * n), C(m * n), X(m * n);
        for (keel::usize i = 0; i < m * m; ++i) A[i] = det_cval(i, 5, 520);
        for (keel::usize i = 0; i < m; ++i) A[i + i * m] += keel::c64(3.0, 0.0);
        for (keel::usize i = 0; i < n * n; ++i) B[i] = det_cval(i, 7, 530);
        for (keel::usize i = 0; i < n; ++i) B[i + i * n] += keel::c64(4.0, 0.0);
        for (keel::usize i = 0; i < m * n; ++i) C[i] = det_cval(i, 11, 540);

        CHECK(keel::linalg::sylvester(A.data(), m, B.data(), n, C.data(), X.data()));
        keel::Array<keel::c64> lhs(m * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, keel::c64(1.0, 0.0),
                   A.data(), m, X.data(), m, keel::c64(0.0, 0.0), lhs.data(), m);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, n, keel::c64(1.0, 0.0),
                   X.data(), m, B.data(), n, keel::c64(1.0, 0.0), lhs.data(), m);
        CHECK(max_diff(lhs.data(), C.data(), m * n) < 1e-10);
    }

    // 3. lsqnonneg
    {
        const keel::usize m = 4;
        const keel::usize n = 3;
        keel::f64 C[12] = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0};
        keel::f64 d[4] = {2.0, -1.0, 3.0, 1.0};
        keel::f64 x[3] = {0.0, 0.0, 0.0};
        keel::f64 res[4] = {0.0, 0.0, 0.0, 0.0};
        const keel::linalg::NnlsStats st = keel::linalg::lsqnonneg(C, m, n, d, x, res);
        CHECK(st.converged);
        for (keel::usize j = 0; j < n; ++j) CHECK(x[j] >= -1e-14);
        // KKT optimality: C^T * (d - C*x) <= tol, and == 0 for x[j] > 0
        keel::f64 grad[3] = {0.0, 0.0, 0.0};
        keel::linalg::gemv(keel::linalg::MatrixTranspose::Trans, m, n, 1.0, C, m, res, 1, 0.0, grad, 1);
        for (keel::usize j = 0; j < n; ++j) {
            CHECK(grad[j] <= 1e-10);
            if (x[j] > 1e-10) CHECK(keel::abs(grad[j]) < 1e-10);
        }
    }

    // 4. rref
    {
        // Column-major 3x4 rank-2 matrix: rows [1, 2, 3, 4], [2, 4, 6, 8], [1, 0, 1, 2]
        keel::f64 M[12] = {1.0, 2.0, 1.0, 2.0, 4.0, 0.0, 3.0, 6.0, 1.0, 4.0, 8.0, 2.0};
        keel::usize pivs[3] = {0, 0, 0};
        const keel::usize rank = keel::linalg::rref(M, 3, 4, -1.0, pivs);
        CHECK(rank == 2);
        CHECK(pivs[0] == 0);
        CHECK(pivs[1] == 1);
        CHECK(keel::abs(M[0 + 0 * 3] - 1.0) < 1e-12);
        CHECK(keel::abs(M[1 + 1 * 3] - 1.0) < 1e-12);
        CHECK(keel::abs(M[2 + 3 * 3]) < 1e-12);
    }

    // 5. gsvd_full & gsvd_values
    {
        const keel::usize m = 4;
        const keel::usize p = 3;
        const keel::usize n = 3;
        keel::Array<keel::f64> A(m * n), B(p * n);
        for (keel::usize i = 0; i < m * n; ++i) A[i] = det_val(i, 5, 550);
        for (keel::usize i = 0; i < p * n; ++i) B[i] = det_val(i, 9, 560);

        keel::Array<keel::f64> U(m * m), V(p * n), X(n * n), C(m * n), S(p * n);
        keel::linalg::gsvd_full(A.data(), m, B.data(), p, n, U.data(), V.data(), X.data(), C.data(), S.data());

        // Check A = U * C * X^T and B = V * S * X^T
        keel::Array<keel::f64> UC(m * n, 0.0), recA(m * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, m, n, m, 1.0, U.data(), m, C.data(),
                   m, 0.0, UC.data(), m);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans, m, n, n, 1.0, UC.data(), m, X.data(),
                   n, 0.0, recA.data(), m);
        CHECK(max_diff(recA.data(), A.data(), m * n) < 1e-10);

        keel::Array<keel::f64> VS(p * n, 0.0), recB(p * n, 0.0);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans, p, n, n, 1.0, V.data(), p, S.data(),
                   p, 0.0, VS.data(), p);
        // Note: S is p x n diagonal (p == n == 3 here).
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans, p, n, n, 1.0, VS.data(), p, X.data(),
                   n, 0.0, recB.data(), p);
        CHECK(max_diff(recB.data(), B.data(), p * n) < 1e-10);

        keel::Array<keel::f64> sigmas(n);
        keel::linalg::gsvd_values(A.data(), m, B.data(), p, n, sigmas.data());
        for (keel::usize i = 0; i + 1 < n; ++i) CHECK(sigmas[i] <= sigmas[i + 1]);
    }
}

void test_complex_gram_hermitian() {
    for (keel::usize n : {keel::usize{3}, keel::usize{16}}) {
        keel::Array<keel::c64> A(n * n);
        for (keel::usize i = 0; i < n * n; ++i) {
            A[i] = keel::c64(det_val(i, 7, 100), det_val(i, 11, 200));
        }
        keel::Array<keel::c64> C1(n * n, keel::c64(0.0, 0.0));
        keel::Array<keel::c64> C2(n * n, keel::c64(0.0, 0.0));
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans,
                           keel::linalg::MatrixTranspose::ConjTrans,
                           n, n, n, keel::c64(1.0, 0.0), A.data(), n, A.data(), n,
                           keel::c64(0.0, 0.0), C1.data(), n);
        keel::linalg::gemm(keel::linalg::MatrixTranspose::ConjTrans,
                           keel::linalg::MatrixTranspose::NoTrans,
                           n, n, n, keel::c64(1.0, 0.0), A.data(), n, A.data(), n,
                           keel::c64(0.0, 0.0), C2.data(), n);
        for (keel::usize i = 0; i < n; ++i) {
            CHECK(C1[i + i * n].imag() == 0.0);
            CHECK(C2[i + i * n].imag() == 0.0);
            for (keel::usize j = i + 1; j < n; ++j) {
                CHECK(C1[i + j * n].real() == C1[j + i * n].real());
                CHECK(C1[i + j * n].imag() == -C1[j + i * n].imag());
                CHECK(C2[i + j * n].real() == C2[j + i * n].real());
                CHECK(C2[i + j * n].imag() == -C2[j + i * n].imag());
            }
        }
    }
}

} // namespace

int main() {
    test_blas1();
    test_blas2();
    test_gemm();
    test_complex_gram_hermitian();
    test_trsm_and_syrk();
    test_solve_lu_inv();
    test_cholesky();
    test_qr();
    test_svd();
    test_factor_updates_ldl_balance();
    test_eig_schur_qz();
    test_matrix_functions_and_solvers();

    if (g_failures == 0) {
        keel::println("All keel::linalg::linalg tests passed.");
        return 0;
    }
    keel::eprintln("{} test failure(s).", g_failures);
    return 1;
}
