#include "keel/arena.h"
#include "keel/array.h"
#include "keel/complex.h"
#include "keel/core.h"
#include "keel/linalg.h"
#include "keel/math.h"
#include "linalg_internal.h"

namespace keel::linalg {

namespace {

template <typename T>
[[nodiscard]] f64 mat_1norm(const T* A, usize n) noexcept {
    f64 mx = 0.0;
    for (usize j = 0; j < n; ++j) {
        f64 s = 0.0;
        for (usize i = 0; i < n; ++i) s += keel::abs(A[i + j * n]);
        if (s > mx) mx = s;
    }
    return mx;
}

template <typename T>
inline void square_gemm(const T* A, const T* B, T* C, usize n) noexcept {
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans, n, n, n, T(1), A, n, B, n, T(0), C, n);
}

template <typename T>
bool expm_impl(const T* A, usize n, T* X, Allocator* alloc) noexcept {
    if (n == 0) return true;

    // Pade(6) coefficients (Higham table 10.4).
    static constexpr f64 c[7] = {
        1.0, 1.0 / 2.0, 5.0 / 44.0, 1.0 / 66.0, 1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0,
    };

    ArenaTemp scratch = get_scratch(alloc);
    Array<T> A_s(n * n, T(0), scratch.allocator());
    keel::mem_copy(A_s.data(), A, n * n * sizeof(T));

    const f64 a_norm = mat_1norm(A_s.data(), n);
    i32 s = 0;
    if (a_norm > 0.5) {
        s = keel::ceil_i32(keel::log2(a_norm / 0.5));
        if (s < 0) s = 0;
        const f64 scale = keel::ldexp(1.0, -s);
        for (usize i = 0; i < n * n; ++i) A_s[i] *= scale;
    }

    Array<T> A2(n * n, T(0), scratch.allocator());
    Array<T> A3(n * n, T(0), scratch.allocator());
    Array<T> A4(n * n, T(0), scratch.allocator());
    Array<T> A5(n * n, T(0), scratch.allocator());
    Array<T> A6(n * n, T(0), scratch.allocator());
    square_gemm(A_s.data(), A_s.data(), A2.data(), n);
    square_gemm(A2.data(), A2.data(), A4.data(), n);
    square_gemm(A2.data(), A4.data(), A6.data(), n);
    square_gemm(A_s.data(), A2.data(), A3.data(), n);
    square_gemm(A_s.data(), A4.data(), A5.data(), n);

    Array<T> P(n * n, T(0), scratch.allocator());
    Array<T> Q(n * n, T(0), scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        P[i + i * n] = T(c[0]);
        Q[i + i * n] = T(c[0]);
    }
    for (usize i = 0; i < n * n; ++i) {
        const T odd = c[1] * A_s[i] + c[3] * A3[i] + c[5] * A5[i];
        const T even = c[2] * A2[i] + c[4] * A4[i] + c[6] * A6[i];
        P[i] += even + odd;
        Q[i] += even - odd;
    }

    if (!la_solve(Q.data(), n, n, P.data(), n, X, scratch.allocator())) return false;

    if (s > 0) {
        Array<T> tmp(n * n, T(0), scratch.allocator());
        for (i32 k = 0; k < s; ++k) {
            square_gemm(X, X, tmp.data(), n);
            keel::mem_copy(X, tmp.data(), n * n * sizeof(T));
        }
    }
    return true;
}

void schur_reconstruct(const c64* U, const c64* F, usize n, c64* X, Allocator* scratch_alloc) noexcept {
    // X = U * F * U^H
    Array<c64> F_Uh(n * n, c64(0.0, 0.0), scratch_alloc);
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::ConjTrans, n, n, n, c64(1.0, 0.0), F, n, U, n, c64(0.0, 0.0),
         F_Uh.data(), n);
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans, n, n, n, c64(1.0, 0.0), U, n, F_Uh.data(), n,
         c64(0.0, 0.0), X, n);
}

void upper_tri_sqrt(const c64* T, usize n, c64* R) noexcept {
    for (usize i = 0; i < n * n; ++i) R[i] = c64(0.0, 0.0);
    for (usize i = 0; i < n; ++i) R[i + i * n] = keel::sqrt(T[i + i * n]);

    for (usize j = 1; j < n; ++j) {
        for (isize i = as<isize>(j) - 1; i >= 0; --i) {
            const usize ui = as<usize>(i);
            c64 sum = T[ui + j * n];
            for (usize k = ui + 1; k < j; ++k) sum -= R[ui + k * n] * R[k + j * n];
            c64 denom = R[ui + ui * n] + R[j + j * n];
            if (keel::abs(denom) < 1e-14) denom = c64(1e-14, 0.0);
            R[ui + j * n] = sum / denom;
        }
    }
}

[[nodiscard]] f64 mat_diff_i_1norm(const c64* M, usize n) noexcept {
    f64 max_col = 0.0;
    for (usize j = 0; j < n; ++j) {
        f64 col_sum = 0.0;
        for (usize i = 0; i < n; ++i) {
            const c64 diff = (i == j) ? (M[i + j * n] - c64(1.0, 0.0)) : M[i + j * n];
            col_sum += keel::abs(diff);
        }
        if (col_sum > max_col) max_col = col_sum;
    }
    return max_col;
}

} // namespace

bool expm(const f64* A, usize n, f64* X, Allocator* alloc) noexcept {
    return expm_impl(A, n, X, alloc);
}

bool expm(const c64* A, usize n, c64* X, Allocator* alloc) noexcept {
    return expm_impl(A, n, X, alloc);
}

bool sqrtm(const c64* A, usize n, c64* X, Allocator* alloc) noexcept {
    if (n == 0) return true;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> U(n * n, c64(0.0, 0.0), scratch.allocator());
    Array<c64> T(n * n, c64(0.0, 0.0), scratch.allocator());
    if (!schur_complex(A, n, U.data(), T.data(), scratch.allocator())) return false;

    Array<c64> R(n * n, c64(0.0, 0.0), scratch.allocator());
    upper_tri_sqrt(T.data(), n, R.data());
    schur_reconstruct(U.data(), R.data(), n, X, scratch.allocator());
    return true;
}

bool logm(const c64* A, usize n, c64* X, Allocator* alloc) noexcept {
    if (n == 0) return true;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> U(n * n, c64(0.0, 0.0), scratch.allocator());
    Array<c64> T(n * n, c64(0.0, 0.0), scratch.allocator());
    if (!schur_complex(A, n, U.data(), T.data(), scratch.allocator())) return false;

    i32 s_count = 0;
    Array<c64> T_next(n * n, c64(0.0, 0.0), scratch.allocator());
    while (s_count < 32 && mat_diff_i_1norm(T.data(), n) > 0.5) {
        upper_tri_sqrt(T.data(), n, T_next.data());
        keel::mem_copy(T.data(), T_next.data(), n * n * sizeof(c64));
        ++s_count;
    }

    Array<c64> E(n * n, c64(0.0, 0.0), scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i <= j; ++i) {
            E[i + j * n] = (i == j) ? (T[i + j * n] - c64(1.0, 0.0)) : T[i + j * n];
        }
    }

    Array<c64> log_t(n * n, c64(0.0, 0.0), scratch.allocator());
    Array<c64> Ek(n * n, c64(0.0, 0.0), scratch.allocator());
    Array<c64> Ek_next(n * n, c64(0.0, 0.0), scratch.allocator());
    for (usize i = 0; i < n; ++i) Ek[i + i * n] = c64(1.0, 0.0);

    for (usize k = 1; k <= 24; ++k) {
        for (usize i = 0; i < n * n; ++i) Ek_next[i] = c64(0.0, 0.0);
        for (usize j = 0; j < n; ++j) {
            for (usize i = 0; i <= j; ++i) {
                c64 sum(0.0, 0.0);
                for (usize l = i; l <= j; ++l) sum += Ek[i + l * n] * E[l + j * n];
                Ek_next[i + j * n] = sum;
            }
        }
        keel::mem_copy(Ek.data(), Ek_next.data(), n * n * sizeof(c64));

        const f64 coef = ((k % 2 == 1) ? 1.0 : -1.0) / as<f64>(k);
        for (usize i = 0; i < n * n; ++i) log_t[i] += coef * Ek[i];
    }

    const f64 scale = keel::ldexp(1.0, s_count);
    for (usize i = 0; i < n * n; ++i) log_t[i] *= scale;

    schur_reconstruct(U.data(), log_t.data(), n, X, scratch.allocator());
    return true;
}

bool funm(const c64* A, usize n, FunctionRef<c64(c64)> f, c64* X, Allocator* alloc) noexcept {
    if (n == 0) return true;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> U(n * n, c64(0.0, 0.0), scratch.allocator());
    Array<c64> T(n * n, c64(0.0, 0.0), scratch.allocator());
    if (!schur_complex(A, n, U.data(), T.data(), scratch.allocator())) return false;

    Array<c64> F(n * n, c64(0.0, 0.0), scratch.allocator());
    for (usize i = 0; i < n; ++i) F[i + i * n] = f(T[i + i * n]);

    for (usize j = 1; j < n; ++j) {
        for (isize i = as<isize>(j) - 1; i >= 0; --i) {
            const usize ui = as<usize>(i);
            const c64 denom = T[ui + ui * n] - T[j + j * n];
            if (keel::abs(denom) > 1e-12) {
                c64 sum = F[ui + ui * n] * T[ui + j * n] - T[ui + j * n] * F[j + j * n];
                for (usize k = ui + 1; k < j; ++k) {
                    sum += F[ui + k * n] * T[k + j * n] - T[ui + k * n] * F[k + j * n];
                }
                F[ui + j * n] = sum / denom;
            } else {
                const c64 lambda = 0.5 * (T[ui + ui * n] + T[j + j * n]);
                const c64 h(1e-6, 0.0);
                const c64 df = (f(lambda + h) - f(lambda - h)) / (2.0 * h);
                c64 sum = df * T[ui + j * n];
                for (usize k = ui + 1; k < j; ++k) sum += F[ui + k * n] * T[k + j * n];
                F[ui + j * n] = sum;
            }
        }
    }

    schur_reconstruct(U.data(), F.data(), n, X, scratch.allocator());
    return true;
}

bool sylvester(const c64* A, usize m, const c64* B, usize n, const c64* C, c64* X, Allocator* alloc) noexcept {
    if (m == 0 || n == 0) return true;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> Ua(m * m, c64(0.0, 0.0), scratch.allocator());
    Array<c64> Ta(m * m, c64(0.0, 0.0), scratch.allocator());
    Array<c64> Ub(n * n, c64(0.0, 0.0), scratch.allocator());
    Array<c64> Tb(n * n, c64(0.0, 0.0), scratch.allocator());

    if (!schur_complex(A, m, Ua.data(), Ta.data(), scratch.allocator())) return false;
    if (!schur_complex(B, n, Ub.data(), Tb.data(), scratch.allocator())) return false;

    // C_tilde = Ua^H * C * Ub
    Array<c64> UaH_C(m * n, c64(0.0, 0.0), scratch.allocator());
    Array<c64> C_tilde(m * n, c64(0.0, 0.0), scratch.allocator());
    gemm(MatrixTranspose::ConjTrans, MatrixTranspose::NoTrans, m, n, m, c64(1.0, 0.0), Ua.data(), m, C, m,
         c64(0.0, 0.0), UaH_C.data(), m);
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans, m, n, n, c64(1.0, 0.0), UaH_C.data(), m, Ub.data(), n,
         c64(0.0, 0.0), C_tilde.data(), m);

    Array<c64> Y(m * n, c64(0.0, 0.0), scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        for (isize i = as<isize>(m) - 1; i >= 0; --i) {
            const usize ui = as<usize>(i);
            c64 s = C_tilde[ui + j * m];
            for (usize k = ui + 1; k < m; ++k) s -= Ta[ui + k * m] * Y[k + j * m];
            for (usize k = 0; k < j; ++k) s -= Y[ui + k * m] * Tb[k + j * n];
            const c64 denom = Ta[ui + ui * m] + Tb[j + j * n];
            Y[ui + j * m] = s / denom;
        }
    }

    // X = Ua * Y * Ub^H
    Array<c64> Ua_Y(m * n, c64(0.0, 0.0), scratch.allocator());
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans, m, n, m, c64(1.0, 0.0), Ua.data(), m, Y.data(), m,
         c64(0.0, 0.0), Ua_Y.data(), m);
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::ConjTrans, m, n, n, c64(1.0, 0.0), Ua_Y.data(), m, Ub.data(), n,
         c64(0.0, 0.0), X, m);
    return true;
}

bool expmv(f64 t, const f64* A, usize n, const f64* v, f64* out, Allocator* alloc) noexcept {
    if (n == 0) return true;

    const f64 beta = nrm2(n, v);
    if (beta == 0.0) {
        for (usize i = 0; i < n; ++i) out[i] = 0.0;
        return true;
    }

    const usize m = linalg_detail::linalg_min(usize{30}, n);
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> V(n * (m + 1), 0.0, scratch.allocator());
    Array<f64> H(m * m, 0.0, scratch.allocator());
    Array<f64> work(n, 0.0, scratch.allocator());

    for (usize i = 0; i < n; ++i) V[i] = v[i] / beta;

    usize m_eff = m;
    for (usize j = 0; j < m; ++j) {
        gemv(MatrixTranspose::NoTrans, n, n, 1.0, A, n, V.data() + j * n, 1, 0.0, work.data(), 1);

        for (usize i = 0; i <= j; ++i) {
            const f64 hij = dot(n, V.data() + i * n, work.data());
            H[i + j * m] = hij;
            axpy(n, -hij, V.data() + i * n, work.data());
        }

        const f64 hjp1 = nrm2(n, work.data());
        if (hjp1 < 1e-14) {
            m_eff = j + 1;
            break;
        }
        if (j + 1 < m) H[(j + 1) + j * m] = hjp1;
        const f64 inv_h = 1.0 / hjp1;
        for (usize k = 0; k < n; ++k) V[k + (j + 1) * n] = work[k] * inv_h;
    }

    Array<f64> Hsq(m_eff * m_eff, 0.0, scratch.allocator());
    for (usize j = 0; j < m_eff; ++j) {
        for (usize i = 0; i < m_eff; ++i) {
            Hsq[i + j * m_eff] = t * H[i + j * m];
        }
    }

    Array<f64> F(m_eff * m_eff, 0.0, scratch.allocator());
    if (!expm(Hsq.data(), m_eff, F.data(), scratch.allocator())) return false;

    gemv(MatrixTranspose::NoTrans, n, m_eff, beta, V.data(), n, F.data(), 1, 0.0, out, 1);
    return true;
}

} // namespace keel::linalg
