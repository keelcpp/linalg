#include "keel/arena.h"
#include "keel/array.h"
#include "keel/complex.h"
#include "keel/core.h"
#include "keel/linalg.h"
#include "keel/math.h"
#include "linalg_internal.h"

namespace keel {

namespace {

template <typename T>
inline constexpr bool kIsComplex = false;

template <typename U>
inline constexpr bool kIsComplex<Complex<U>> = true;

template <typename T>
[[nodiscard]] inline f64 real_part(const T& x) noexcept {
    if constexpr (kIsComplex<T>) {
        return x.real();
    } else {
        return x;
    }
}

template <typename T>
[[nodiscard]] inline f64 abs_val(const T& x) noexcept {
    if constexpr (kIsComplex<T>) {
        return keel::abs(x);
    } else {
        return keel::abs(x);
    }
}

template <typename T>
[[nodiscard]] inline f64 abs_sq(const T& x) noexcept {
    if constexpr (kIsComplex<T>) {
        return keel::norm(x);
    } else {
        return x * x;
    }
}

template <typename T>
[[nodiscard]] inline T conj_if_complex(const T& x) noexcept {
    if constexpr (kIsComplex<T>) {
        return keel::conj(x);
    } else {
        return x;
    }
}

// ============================================================================
// 1. Blocked Upper-Triangular Cholesky (A = R^H * R)
// ============================================================================

template <typename T>
usize cholesky_upper_tiled(T* r, usize n, usize ldr) noexcept {
    constexpr usize kBlock = 64;
    for (usize j = 0; j < n; j += kBlock) {
        const usize jb = linalg_detail::linalg_min(kBlock, n - j);

        // 1. Factor diagonal block sequentially.
        for (usize k = j; k < j + jb; ++k) {
            if constexpr (kIsComplex<T>) {
                if (keel::abs(r[k + k * ldr].imag()) > 1e-11) return k + 1;
            }
            const f64 s_real = real_part(r[k + k * ldr]);
            if (s_real <= 0.0 || !keel::is_finite(s_real)) return k + 1;
            const f64 diag = keel::sqrt(s_real);
            r[k + k * ldr] = T(diag);
            const T inv_diag = T(1.0 / diag);
            for (usize col = k + 1; col < j + jb; ++col) {
                r[k + col * ldr] *= inv_diag;
                for (usize row = k + 1; row <= col; ++row) {
                    r[row + col * ldr] -= conj_if_complex(r[k + row * ldr]) * r[k + col * ldr];
                }
            }
        }

        // 2. Update right panel (TRSM) and trailing matrix (SYRK).
        if (j + jb < n) {
            const usize rem = n - j - jb;
            trsm(MatrixSide::Left, MatrixUplo::Upper, MatrixTranspose::ConjTrans, MatrixDiag::NonUnit, jb, rem, T(1),
                 r + j + j * ldr, ldr, r + j + (j + jb) * ldr, ldr);

            syrk(MatrixUplo::Upper, MatrixTranspose::ConjTrans, rem, jb, T(-1), r + j + (j + jb) * ldr, ldr, T(1),
                 r + (j + jb) + (j + jb) * ldr, ldr);
        }
    }
    return 0;
}

template <typename T>
usize cholesky_upper_impl(const T* A, T* R, usize n) noexcept {
    if (n == 0) return 0;
    for (usize col = 0; col < n; ++col) {
        for (usize row = 0; row <= col; ++row) {
            R[row + col * n] = A[row + col * n];
        }
        for (usize row = col + 1; row < n; ++row) {
            R[row + col * n] = T(0);
        }
    }
    return cholesky_upper_tiled(R, n, n);
}

// ============================================================================
// 2. Householder QR (Full & Column-Pivoted)
// ============================================================================

template <typename T>
void qr_householder_impl(const T* A_in, usize m, usize n, T* Qout, T* Rout, Allocator* alloc) noexcept {
    if (m == 0) return;

    ArenaTemp scratch = get_scratch(alloc);
    const usize k_lim = linalg_detail::linalg_min(m, n);
    Array<T> R_work(m * n, T(0), scratch.allocator());
    Array<T> V(m * k_lim, T(0), scratch.allocator());
    Array<T> tau(k_lim, T(0), scratch.allocator());

    for (usize i = 0; i < m * n; ++i) {
        R_work[i] = A_in[i];
    }

    for (usize k = 0; k < k_lim; ++k) {
        // LAPACK dlarfg rule: if subdiagonal tail is identically zero, leave diagonal untouched.
        f64 tail_sq = 0.0;
        for (usize i = k + 1; i < m; ++i) {
            tail_sq += abs_sq(R_work[i + k * m]);
        }
        if (tail_sq == 0.0) {
            tau[k] = T(0);
            continue;
        }
        const T xk = R_work[k + k * m];
        const f64 norm_sq = tail_sq + abs_sq(xk);
        if (norm_sq == 0.0) {
            tau[k] = T(0);
            continue;
        }
        const f64 norm = keel::sqrt(norm_sq);

        T alpha;
        if constexpr (kIsComplex<T>) {
            const f64 abs_xk = keel::abs(xk);
            const T phase = (abs_xk > 0.0) ? (xk / abs_xk) : T(1.0, 0.0);
            alpha = -phase * norm;
        } else {
            alpha = (xk >= 0.0) ? -norm : norm;
        }

        V[k + k * m] = xk - alpha;
        for (usize i = k + 1; i < m; ++i) {
            V[i + k * m] = R_work[i + k * m];
        }

        f64 v_norm_sq = 0.0;
        for (usize i = k; i < m; ++i) {
            v_norm_sq += abs_sq(V[i + k * m]);
        }
        if (v_norm_sq == 0.0) {
            R_work[k + k * m] = alpha;
            tau[k] = T(0);
            continue;
        }

        tau[k] = T(2.0 / v_norm_sq);
        for (usize j = k + 1; j < n; ++j) {
            T dp = T(0);
            for (usize i = k; i < m; ++i) {
                dp += conj_if_complex(V[i + k * m]) * R_work[i + j * m];
            }
            const T s = tau[k] * dp;
            for (usize i = k; i < m; ++i) {
                R_work[i + j * m] -= s * V[i + k * m];
            }
        }
        R_work[k + k * m] = alpha;
        for (usize i = k + 1; i < m; ++i) {
            R_work[i + k * m] = T(0);
        }
    }

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < m; ++i) {
            Rout[i + j * m] = (i <= j) ? R_work[i + j * m] : T(0);
        }
    }

    if (Qout != nullptr) {
        for (usize i = 0; i < m * m; ++i) Qout[i] = T(0);
        for (usize i = 0; i < m; ++i) Qout[i + i * m] = T(1);

        for (usize kk = k_lim; kk-- > 0;) {
            if (abs_sq(tau[kk]) == 0.0) continue;
            for (usize j = 0; j < m; ++j) {
                T dp = T(0);
                for (usize i = kk; i < m; ++i) {
                    dp += conj_if_complex(V[i + kk * m]) * Qout[i + j * m];
                }
                const T s = tau[kk] * dp;
                for (usize i = kk; i < m; ++i) {
                    Qout[i + j * m] -= s * V[i + kk * m];
                }
            }
        }
    }
}

template <typename T>
void qr_pivoted_impl(const T* A_in, usize m, usize n, T* Qout, T* Rout, usize* perm_out, Allocator* alloc) noexcept {
    for (usize j = 0; j < n; ++j) perm_out[j] = j;
    if (m == 0) return;

    ArenaTemp scratch = get_scratch(alloc);
    const usize k_lim = linalg_detail::linalg_min(m, n);
    Array<T> R_work(m * n, T(0), scratch.allocator());
    Array<T> V(m * k_lim, T(0), scratch.allocator());
    Array<T> tau(k_lim, T(0), scratch.allocator());

    for (usize i = 0; i < m * n; ++i) {
        R_work[i] = A_in[i];
    }

    for (usize k = 0; k < k_lim; ++k) {
        usize pj = k;
        f64 pmax = -1.0;
        for (usize j = k; j < n; ++j) {
            f64 s = 0.0;
            for (usize i = k; i < m; ++i) {
                s += abs_sq(R_work[i + j * m]);
            }
            if (s > pmax) {
                pmax = s;
                pj = j;
            }
        }
        if (pj != k) {
            for (usize i = 0; i < m; ++i) {
                keel::swap(R_work[i + k * m], R_work[i + pj * m]);
            }
            keel::swap(perm_out[k], perm_out[pj]);
        }

        f64 tail_sq = 0.0;
        for (usize i = k + 1; i < m; ++i) {
            tail_sq += abs_sq(R_work[i + k * m]);
        }
        if (tail_sq == 0.0) {
            tau[k] = T(0);
            continue;
        }
        const T xk = R_work[k + k * m];
        const f64 norm_sq = tail_sq + abs_sq(xk);
        if (norm_sq == 0.0) {
            tau[k] = T(0);
            continue;
        }
        const f64 norm = keel::sqrt(norm_sq);

        T alpha;
        if constexpr (kIsComplex<T>) {
            const f64 abs_xk = keel::abs(xk);
            const T phase = (abs_xk > 0.0) ? (xk / abs_xk) : T(1.0, 0.0);
            alpha = -phase * norm;
        } else {
            alpha = (xk >= 0.0) ? -norm : norm;
        }

        V[k + k * m] = xk - alpha;
        for (usize i = k + 1; i < m; ++i) {
            V[i + k * m] = R_work[i + k * m];
        }

        f64 v_norm_sq = 0.0;
        for (usize i = k; i < m; ++i) {
            v_norm_sq += abs_sq(V[i + k * m]);
        }
        if (v_norm_sq == 0.0) {
            R_work[k + k * m] = alpha;
            tau[k] = T(0);
            continue;
        }

        tau[k] = T(2.0 / v_norm_sq);
        for (usize j = k + 1; j < n; ++j) {
            T dp = T(0);
            for (usize i = k; i < m; ++i) {
                dp += conj_if_complex(V[i + k * m]) * R_work[i + j * m];
            }
            const T s = tau[k] * dp;
            for (usize i = k; i < m; ++i) {
                R_work[i + j * m] -= s * V[i + k * m];
            }
        }
        R_work[k + k * m] = alpha;
        for (usize i = k + 1; i < m; ++i) {
            R_work[i + k * m] = T(0);
        }
    }

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < m; ++i) {
            Rout[i + j * m] = (i <= j) ? R_work[i + j * m] : T(0);
        }
    }

    if (Qout != nullptr) {
        for (usize i = 0; i < m * m; ++i) Qout[i] = T(0);
        for (usize i = 0; i < m; ++i) Qout[i + i * m] = T(1);

        for (usize kk = k_lim; kk-- > 0;) {
            if (abs_sq(tau[kk]) == 0.0) continue;
            for (usize j = 0; j < m; ++j) {
                T dp = T(0);
                for (usize i = kk; i < m; ++i) {
                    dp += conj_if_complex(V[i + kk * m]) * Qout[i + j * m];
                }
                const T s = tau[kk] * dp;
                for (usize i = kk; i < m; ++i) {
                    Qout[i + j * m] -= s * V[i + kk * m];
                }
            }
        }
    }
}

// ============================================================================
// 3. One-Sided Jacobi Thin SVD (m >= n, A = U * diag(S) * V^H)
// ============================================================================

template <typename T>
void svd_jacobi_impl(T* A, usize m, usize n, T* V, f64* S, Allocator* alloc) noexcept {
    if (m == 0 || n == 0) return;

    for (usize i = 0; i < n * n; ++i) V[i] = T(0);
    for (usize i = 0; i < n; ++i) V[i + i * n] = T(1);

    auto col_dot = [&](usize p, usize q) noexcept -> T {
        T s = T(0);
        for (usize i = 0; i < m; ++i) {
            s += conj_if_complex(A[i + p * m]) * A[i + q * m];
        }
        return s;
    };

    auto rotate_cols = [](T* M, usize ld, usize nrows, usize p, usize q, f64 c, T s) noexcept {
        const T s_conj = conj_if_complex(s);
        for (usize i = 0; i < nrows; ++i) {
            const T mip = M[i + p * ld];
            const T miq = M[i + q * ld];
            M[i + p * ld] = c * mip - s_conj * miq;
            M[i + q * ld] = s * mip + c * miq;
        }
    };

    constexpr usize kMaxSweeps = 60;
    constexpr f64 kTol = 1e-15;

    for (usize sweep = 0; sweep < kMaxSweeps; ++sweep) {
        f64 off = 0.0;
        for (usize p = 0; p + 1 < n; ++p) {
            for (usize q = p + 1; q < n; ++q) {
                const f64 alpha = real_part(col_dot(p, p));
                const f64 beta = real_part(col_dot(q, q));
                const T gamma_val = col_dot(p, q);
                const f64 abs_gamma = abs_val(gamma_val);

                const f64 scale = keel::sqrt(alpha * beta);
                if (abs_gamma <= kTol * scale) continue;
                off += abs_gamma * abs_gamma;

                f64 c = 0.0;
                f64 s_real = 0.0;
                if (alpha == beta) {
                    c = 0.7071067811865476;
                    s_real = c;
                } else {
                    const f64 tau = (beta - alpha) / (2.0 * abs_gamma);
                    const f64 t =
                        (tau >= 0.0) ? (1.0 / (tau + keel::sqrt(1.0 + tau * tau)))
                                     : (1.0 / (tau - keel::sqrt(1.0 + tau * tau)));
                    c = 1.0 / keel::sqrt(1.0 + t * t);
                    s_real = t * c;
                }

                T s;
                if constexpr (kIsComplex<T>) {
                    const T phase = (abs_gamma > 0.0) ? (gamma_val / abs_gamma) : T(1.0, 0.0);
                    s = s_real * phase;
                } else {
                    s = s_real * ((gamma_val >= 0.0) ? 1.0 : -1.0);
                }

                rotate_cols(A, m, m, p, q, c, s);
                rotate_cols(V, n, n, p, q, c, s);
            }
        }
        if (off < kTol * kTol) break;
    }

    // Compute singular values (column norms) and sort columns in descending order.
    ArenaTemp scratch = get_scratch(alloc);
    Array<usize> order(n, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        S[j] = keel::sqrt(real_part(col_dot(j, j)));
        order[j] = j;
    }

    // Simple selection sort (n is column count, O(n^2) << O(m*n^2) Jacobi cost).
    for (usize i = 0; i + 1 < n; ++i) {
        usize best = i;
        for (usize j = i + 1; j < n; ++j) {
            if (S[order[j]] > S[order[best]]) {
                best = j;
            }
        }
        if (best != i) {
            keel::swap(order[i], order[best]);
        }
    }

    Array<T> U_sorted(m * n, T(0), scratch.allocator());
    Array<T> V_sorted(n * n, T(0), scratch.allocator());
    Array<f64> S_sorted(n, 0.0, scratch.allocator());

    for (usize j = 0; j < n; ++j) {
        const usize src = order[j];
        S_sorted[j] = S[src];
        for (usize i = 0; i < n; ++i) {
            V_sorted[i + j * n] = V[i + src * n];
        }
        if (S_sorted[j] > 0.0) {
            const f64 inv_s = 1.0 / S_sorted[j];
            for (usize i = 0; i < m; ++i) {
                U_sorted[i + j * m] = A[i + src * m] * inv_s;
            }
        }
    }

    // Complete orthonormal basis for any zero-norm U columns via modified Gram-Schmidt.
    const f64 s_max = (n > 0) ? S_sorted[0] : 0.0;
    const f64 zero_tol = 1e-14 * ((s_max > 0.0) ? s_max : 1.0);
    for (usize j = 0; j < n; ++j) {
        if (S_sorted[j] > zero_tol) continue;
        // Pick canonical basis vector e_r with largest orthogonal component against columns 0..j-1.
        for (usize r = 0; r < m; ++r) {
            for (usize i = 0; i < m; ++i) U_sorted[i + j * m] = (i == r) ? T(1) : T(0);
            for (usize prev = 0; prev < j; ++prev) {
                T dp = T(0);
                for (usize i = 0; i < m; ++i) {
                    dp += conj_if_complex(U_sorted[i + prev * m]) * U_sorted[i + j * m];
                }
                for (usize i = 0; i < m; ++i) {
                    U_sorted[i + j * m] -= dp * U_sorted[i + prev * m];
                }
            }
            f64 nrm_sq = 0.0;
            for (usize i = 0; i < m; ++i) nrm_sq += abs_sq(U_sorted[i + j * m]);
            if (nrm_sq > 1e-12) {
                const f64 inv_nrm = 1.0 / keel::sqrt(nrm_sq);
                for (usize i = 0; i < m; ++i) U_sorted[i + j * m] *= inv_nrm;
                break;
            }
        }
    }

    for (usize i = 0; i < m * n; ++i) A[i] = U_sorted[i];
    for (usize i = 0; i < n * n; ++i) V[i] = V_sorted[i];
    for (usize i = 0; i < n; ++i) S[i] = S_sorted[i];
}

template <typename T>
void svd_full_impl(const T* A, usize m_in, usize n_in, T* U, f64* S, T* V, Allocator* alloc) noexcept {
    if (m_in == 0 || n_in == 0) {
        for (usize i = 0; i < m_in * m_in; ++i) U[i] = T(0);
        for (usize i = 0; i < m_in; ++i) U[i + i * m_in] = T(1);
        for (usize i = 0; i < n_in * n_in; ++i) V[i] = T(0);
        for (usize i = 0; i < n_in; ++i) V[i + i * n_in] = T(1);
        return;
    }

    const bool transposed = (m_in < n_in);
    const usize m = transposed ? n_in : m_in;
    const usize n = transposed ? m_in : n_in;

    ArenaTemp scratch = get_scratch(alloc);
    Array<T> A_work(m * n, T(0), scratch.allocator());
    T* U_tall = transposed ? V : U;
    T* V_short = transposed ? U : V;

    if (!transposed) {
        for (usize i = 0; i < m * n; ++i) A_work[i] = A[i];
    } else {
        for (usize j = 0; j < n_in; ++j) {
            for (usize i = 0; i < m_in; ++i) {
                A_work[j + i * n_in] = conj_if_complex(A[i + j * m_in]);
            }
        }
    }

    svd_jacobi_impl(A_work.data(), m, n, V_short, S, scratch.allocator());

    for (usize i = 0; i < m * m; ++i) U_tall[i] = T(0);
    usize r_nz = 0;
    for (usize k = 0; k < n; ++k) {
        if (S[k] > 0.0) {
            for (usize i = 0; i < m; ++i) {
                U_tall[i + k * m] = A_work[i + k * m];
            }
            r_nz = k + 1;
        }
    }

    if (r_nz == 0) {
        for (usize i = 0; i < m; ++i) {
            U_tall[i + i * m] = T(1);
        }
    } else if (r_nz < m) {
        Array<T> Q_full(m * m, T(0), scratch.allocator());
        Array<T> R_dummy(m * r_nz, T(0), scratch.allocator());
        qr_householder_impl(U_tall, m, r_nz, Q_full.data(), R_dummy.data(), scratch.allocator());
        for (usize j = r_nz; j < m; ++j) {
            for (usize i = 0; i < m; ++i) {
                U_tall[i + j * m] = Q_full[i + j * m];
            }
        }
    }
}

} // namespace

usize cholesky_upper(const f64* A, f64* R, usize n) noexcept {
    return cholesky_upper_impl(A, R, n);
}

usize cholesky_upper(const c64* A, c64* R, usize n) noexcept {
    return cholesky_upper_impl(A, R, n);
}

void qr_householder(const f64* A, usize m, usize n, f64* Q, f64* R, Allocator* alloc) noexcept {
    qr_householder_impl(A, m, n, Q, R, alloc);
}

void qr_householder(const c64* A, usize m, usize n, c64* Q, c64* R, Allocator* alloc) noexcept {
    qr_householder_impl(A, m, n, Q, R, alloc);
}

void qr_pivoted(const f64* A, usize m, usize n, f64* Q, f64* R, usize* perm, Allocator* alloc) noexcept {
    qr_pivoted_impl(A, m, n, Q, R, perm, alloc);
}

void qr_pivoted(const c64* A, usize m, usize n, c64* Q, c64* R, usize* perm, Allocator* alloc) noexcept {
    qr_pivoted_impl(A, m, n, Q, R, perm, alloc);
}

void svd_jacobi_inplace(f64* A, usize m, usize n, f64* V, f64* S, Allocator* alloc) noexcept {
    svd_jacobi_impl(A, m, n, V, S, alloc);
}

void svd_jacobi_inplace(c64* A, usize m, usize n, c64* V, f64* S, Allocator* alloc) noexcept {
    svd_jacobi_impl(A, m, n, V, S, alloc);
}

void svd_full(const f64* A, usize m, usize n, f64* U, f64* S, f64* V, Allocator* alloc) noexcept {
    svd_full_impl(A, m, n, U, S, V, alloc);
}

void svd_full(const c64* A, usize m, usize n, c64* U, f64* S, c64* V, Allocator* alloc) noexcept {
    svd_full_impl(A, m, n, U, S, V, alloc);
}

} // namespace keel
