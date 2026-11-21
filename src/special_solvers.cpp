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
[[nodiscard]] f64 matrix_inf_norm(const T* A, usize m, usize n) noexcept {
    if (m == 0 || n == 0) return 0.0;
    f64 maxv = 0.0;
    for (usize i = 0; i < m; ++i) {
        f64 s = 0.0;
        for (usize j = 0; j < n; ++j) s += keel::abs(A[i + j * m]);
        if (s > maxv) maxv = s;
    }
    return maxv;
}

[[nodiscard]] inline f64 ulp_positive(f64 x) noexcept {
    if (x <= 0.0 || !keel::is_finite(x)) return Limits<f64>::kEpsilon;
    const u64 bits = __builtin_bit_cast(u64, x);
    return __builtin_bit_cast(f64, bits + 1ULL) - x;
}

template <typename T>
usize rref_impl(T* B, usize m, usize n, f64 tol_in, usize* pivot_cols) noexcept {
    if (m == 0 || n == 0) return 0;

    f64 tol = tol_in;
    if (tol < 0.0) {
        const f64 anorm = matrix_inf_norm(B, m, n);
        const f64 eps_anorm = (anorm == 0.0) ? Limits<f64>::kEpsilon : ulp_positive(anorm);
        tol = as<f64>(linalg_detail::linalg_max(m, n)) * eps_anorm;
    }

    usize row = 0;
    for (usize col = 0; col < n && row < m; ++col) {
        usize piv_row = row;
        f64 piv_val = keel::abs(B[row + col * m]);
        for (usize k = row + 1; k < m; ++k) {
            const f64 v = keel::abs(B[k + col * m]);
            if (v > piv_val) {
                piv_val = v;
                piv_row = k;
            }
        }

        if (piv_val <= tol) {
            for (usize k = row; k < m; ++k) B[k + col * m] = T(0);
            continue;
        }

        if (piv_row != row) {
            for (usize jj = 0; jj < n; ++jj) {
                keel::swap(B[row + jj * m], B[piv_row + jj * m]);
            }
        }

        const T piv = B[row + col * m];
        for (usize jj = 0; jj < n; ++jj) B[row + jj * m] /= piv;
        B[row + col * m] = T(1);

        for (usize k = 0; k < m; ++k) {
            if (k == row) continue;
            const T f = B[k + col * m];
            if (f == T(0)) continue;
            for (usize jj = 0; jj < n; ++jj) B[k + jj * m] -= f * B[row + jj * m];
            B[k + col * m] = T(0);
        }

        if (pivot_cols != nullptr) pivot_cols[row] = col;
        ++row;
    }
    return row;
}

template <typename T>
void gsvd_values_impl(const T* A, usize m, const T* B, usize p, usize n, f64* sigmas, Allocator* alloc) noexcept {
    if (m == 0 || n == 0 || p == 0) return;

    ArenaTemp scratch = get_scratch(alloc);
    const usize mp = m + p;
    Array<T> K(mp * n, T(0), scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        keel::mem_copy(K.data() + j * mp, A + j * m, m * sizeof(T));
        keel::mem_copy(K.data() + m + j * mp, B + j * p, p * sizeof(T));
    }

    Array<T> Qk(mp * mp, T(0), scratch.allocator());
    Array<T> Rk(mp * n, T(0), scratch.allocator());
    qr_householder(K.data(), mp, n, Qk.data(), Rk.data(), scratch.allocator());

    Array<T> Q1(m * n, T(0), scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        keel::mem_copy(Q1.data() + j * m, Qk.data() + j * mp, m * sizeof(T));
    }

    const usize num_vals = linalg_detail::linalg_min(m, n);
    Array<T> U_tmp(m * m, T(0), scratch.allocator());
    Array<T> V_tmp(n * n, T(0), scratch.allocator());
    Array<f64> c_vals(num_vals, 0.0, scratch.allocator());
    svd_full(Q1.data(), m, n, U_tmp.data(), c_vals.data(), V_tmp.data(), scratch.allocator());

    for (usize i = 0; i < num_vals; ++i) {
        f64 c = c_vals[i];
        if (c > 1.0) c = 1.0;
        if (c < 0.0) c = 0.0;
        const f64 s = keel::sqrt(linalg_detail::linalg_max(0.0, 1.0 - c * c));
        sigmas[i] = (s < 1e-14) ? Limits<f64>::infinity() : (c / s);
    }

    // Sort ascending.
    for (usize i = 0; i + 1 < num_vals; ++i) {
        usize best = i;
        for (usize j = i + 1; j < num_vals; ++j) {
            if (sigmas[j] < sigmas[best]) best = j;
        }
        if (best != i) keel::swap(sigmas[i], sigmas[best]);
    }
}

template <typename T>
void gsvd_full_impl(const T* A, usize m, const T* B, usize p, usize n, T* U, T* V, T* X, f64* C, f64* S,
                    Allocator* alloc) noexcept {
    if (m == 0 || n == 0 || p == 0) return;

    ArenaTemp scratch = get_scratch(alloc);
    const usize mp = m + p;
    Array<T> K(mp * n, T(0), scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        keel::mem_copy(K.data() + j * mp, A + j * m, m * sizeof(T));
        keel::mem_copy(K.data() + m + j * mp, B + j * p, p * sizeof(T));
    }

    Array<T> Qk(mp * mp, T(0), scratch.allocator());
    Array<T> Rk_full(mp * n, T(0), scratch.allocator());
    qr_householder(K.data(), mp, n, Qk.data(), Rk_full.data(), scratch.allocator());

    Array<T> Rk(n * n, T(0), scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        keel::mem_copy(Rk.data() + j * n, Rk_full.data() + j * mp, n * sizeof(T));
    }

    Array<T> Q1(m * n, T(0), scratch.allocator());
    Array<T> Q2(p * n, T(0), scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        keel::mem_copy(Q1.data() + j * m, Qk.data() + j * mp, m * sizeof(T));
        keel::mem_copy(Q2.data() + j * p, Qk.data() + m + j * mp, p * sizeof(T));
    }

    const usize min_mn = linalg_detail::linalg_min(m, n);
    Array<f64> c_diag(min_mn, 0.0, scratch.allocator());
    Array<T> W(n * n, T(0), scratch.allocator());
    svd_full(Q1.data(), m, n, U, c_diag.data(), W.data(), scratch.allocator());

    for (usize i = 0; i < m * n; ++i) C[i] = 0.0;
    for (usize j = 0; j < min_mn; ++j) C[j + j * m] = c_diag[j];

    // Q2_W = Q2 * W
    Array<T> Q2_W(p * n, T(0), scratch.allocator());
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans, p, n, n, T(1), Q2.data(), p, W.data(), n, T(0),
         Q2_W.data(), p);

    for (usize i = 0; i < p * n; ++i) S[i] = 0.0;
    const usize min_pn = linalg_detail::linalg_min(p, n);
    for (usize j = 0; j < n; ++j) {
        const f64 cj = (j < min_mn) ? keel::abs(c_diag[j]) : 0.0;
        const f64 sj = keel::sqrt(linalg_detail::linalg_max(0.0, 1.0 - cj * cj));
        if (j < min_pn) S[j + j * p] = sj;
        for (usize i = 0; i < p; ++i) {
            V[i + j * p] = (sj > 1e-15) ? (Q2_W[i + j * p] / sj) : T(0);
        }
    }

    // X = Rk^H * W
    gemm(MatrixTranspose::ConjTrans, MatrixTranspose::NoTrans, n, n, n, T(1), Rk.data(), n, W.data(), n, T(0), X, n);
}

} // namespace

NnlsStats lsqnonneg(const f64* C, usize m, usize n, const f64* d, f64* x, f64* residual_out,
                    Allocator* alloc) noexcept {
    NnlsStats stats{0.0, 0, true};

    for (usize j = 0; j < n; ++j) x[j] = 0.0;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> res_buf(m, 0.0, scratch.allocator());
    f64* rd = (residual_out != nullptr) ? residual_out : res_buf.data();
    for (usize i = 0; i < m; ++i) rd[i] = d[i];

    f64 resnorm = 0.0;
    for (usize i = 0; i < m; ++i) resnorm += rd[i] * rd[i];
    stats.resnorm = resnorm;
    if (n == 0) return stats;

    f64 cnorm = 0.0;
    for (usize k = 0; k < m * n; ++k) cnorm = linalg_detail::linalg_max(cnorm, keel::abs(C[k]));
    const f64 tol = 10.0 * Limits<f64>::kEpsilon * linalg_detail::linalg_max(cnorm, 1.0) *
                    as<f64>(linalg_detail::linalg_max(m, n));

    Array<u8> in_z(n, u8{1}, scratch.allocator());
    Array<f64> w(n, 0.0, scratch.allocator());
    Array<f64> s(n, 0.0, scratch.allocator());
    Array<usize> p_idx(scratch.allocator());
    Array<f64> cp_buf(scratch.allocator());
    Array<f64> ctc(scratch.allocator());
    Array<f64> ctd(scratch.allocator());
    Array<f64> sp(scratch.allocator());

    const usize max_outer = 3 * n;
    usize outer_iter = 0;

    auto recompute_residual = [&]() noexcept {
        for (usize i = 0; i < m; ++i) rd[i] = d[i];
        gemv(MatrixTranspose::NoTrans, m, n, -1.0, C, m, x, 1, 1.0, rd, 1);
        f64 rn = 0.0;
        for (usize i = 0; i < m; ++i) rn += rd[i] * rd[i];
        stats.resnorm = rn;
    };

    while (outer_iter < max_outer) {
        gemv(MatrixTranspose::Trans, m, n, 1.0, C, m, rd, 1, 0.0, w.data(), 1);

        usize t = n;
        f64 max_w = tol;
        for (usize j = 0; j < n; ++j) {
            if (in_z[j] != 0 && w[j] > max_w) {
                max_w = w[j];
                t = j;
            }
        }
        if (t == n) break;
        ++outer_iter;
        in_z[t] = 0;

        for (usize inner = 0; inner < max_outer; ++inner) {
            p_idx.clear();
            for (usize j = 0; j < n; ++j) {
                if (in_z[j] == 0) p_idx.push_back(j);
            }
            const usize p = p_idx.size();
            cp_buf.resize(m * p);
            for (usize k = 0; k < p; ++k) {
                keel::mem_copy(cp_buf.data() + k * m, C + p_idx[k] * m, m * sizeof(f64));
            }

            ctc.resize(p * p);
            ctd.resize(p);
            sp.resize(p);
            gemm(MatrixTranspose::Trans, MatrixTranspose::NoTrans, p, p, m, 1.0, cp_buf.data(), m, cp_buf.data(), m,
                 0.0, ctc.data(), p);
            gemv(MatrixTranspose::Trans, m, p, 1.0, cp_buf.data(), m, d, 1, 0.0, ctd.data(), 1);

            if (!la_solve(ctc.data(), p, p, ctd.data(), 1, sp.data(), scratch.allocator())) {
                break;
            }

            for (usize j = 0; j < n; ++j) s[j] = 0.0;
            for (usize k = 0; k < p; ++k) s[p_idx[k]] = sp[k];

            bool all_pos = true;
            for (usize k = 0; k < p; ++k) {
                if (sp[k] <= 0.0) {
                    all_pos = false;
                    break;
                }
            }
            if (all_pos) {
                keel::mem_copy(x, s.data(), n * sizeof(f64));
                break;
            }

            f64 alpha = Limits<f64>::infinity();
            for (usize k = 0; k < p; ++k) {
                const usize j = p_idx[k];
                if (sp[k] <= 0.0) {
                    const f64 denom = x[j] - sp[k];
                    if (denom > 0.0) {
                        const f64 a = x[j] / denom;
                        if (a < alpha) alpha = a;
                    }
                }
            }
            if (!keel::is_finite(alpha)) break;
            for (usize j = 0; j < n; ++j) x[j] += alpha * (s[j] - x[j]);
            for (usize k = 0; k < p; ++k) {
                const usize j = p_idx[k];
                if (keel::abs(x[j]) < tol) {
                    x[j] = 0.0;
                    in_z[j] = 1;
                }
            }
        }
        recompute_residual();
    }

    stats.iterations = outer_iter;
    stats.converged = true;
    return stats;
}

usize rref(f64* A, usize m, usize n, f64 tol, usize* pivot_cols) noexcept {
    return rref_impl(A, m, n, tol, pivot_cols);
}

usize rref(c64* A, usize m, usize n, f64 tol, usize* pivot_cols) noexcept {
    return rref_impl(A, m, n, tol, pivot_cols);
}

void gsvd_values(const f64* A, usize m, const f64* B, usize p, usize n, f64* sigmas, Allocator* alloc) noexcept {
    gsvd_values_impl(A, m, B, p, n, sigmas, alloc);
}

void gsvd_values(const c64* A, usize m, const c64* B, usize p, usize n, f64* sigmas, Allocator* alloc) noexcept {
    gsvd_values_impl(A, m, B, p, n, sigmas, alloc);
}

void gsvd_full(const f64* A, usize m, const f64* B, usize p, usize n, f64* U, f64* V, f64* X, f64* C, f64* S,
               Allocator* alloc) noexcept {
    gsvd_full_impl(A, m, B, p, n, U, V, X, C, S, alloc);
}

void gsvd_full(const c64* A, usize m, const c64* B, usize p, usize n, c64* U, c64* V, c64* X, f64* C, f64* S,
               Allocator* alloc) noexcept {
    gsvd_full_impl(A, m, B, p, n, U, V, X, C, S, alloc);
}

} // namespace keel::linalg
