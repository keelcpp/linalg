#include "keel/arena.h"
#include "keel/array.h"
#include "keel/complex.h"
#include "keel/core.h"
#include "keel/linalg.h"
#include "keel/math.h"
#include "linalg_internal.h"

namespace keel::linalg {

namespace {

// Compute Givens rotation (c, s) that zeros y in [x; y] -> [r; 0].
inline void givens(f64 x, f64 y, f64& c, f64& s, f64& r) noexcept {
    if (y == 0.0) {
        c = (x >= 0.0) ? 1.0 : -1.0;
        s = 0.0;
        r = keel::abs(x);
        return;
    }
    if (x == 0.0) {
        c = 0.0;
        s = (y >= 0.0) ? 1.0 : -1.0;
        r = keel::abs(y);
        return;
    }
    r = keel::hypot(x, y);
    c = x / r;
    s = y / r;
}

inline void apply_givens_rows(f64* M, usize ld, usize ncols, usize i, usize j, f64 c, f64 s) noexcept {
    for (usize k = 0; k < ncols; ++k) {
        const f64 a = M[i + k * ld];
        const f64 b = M[j + k * ld];
        M[i + k * ld] = c * a + s * b;
        M[j + k * ld] = -s * a + c * b;
    }
}

inline void apply_givens_cols(f64* M, usize ld, usize nrows, usize i, usize j, f64 c, f64 s) noexcept {
    for (usize k = 0; k < nrows; ++k) {
        const f64 a = M[k + i * ld];
        const f64 b = M[k + j * ld];
        M[k + i * ld] = c * a + s * b;
        M[k + j * ld] = -s * a + c * b;
    }
}

template <typename T>
void balance_permute(T* B, usize n, usize& low, usize& high, usize* perm) noexcept {
    if (perm != nullptr) {
        for (usize i = 0; i < n; ++i) perm[i] = i;
    }
    if (n <= 1) {
        low = 0;
        high = (n == 0) ? 0 : n - 1;
        return;
    }

    low = 0;
    high = n - 1;

    bool found = true;
    while (found) {
        found = false;

        // Search for isolated rows in [low, high].
        for (isize i = as<isize>(high); i >= as<isize>(low); --i) {
            const usize ui = as<usize>(i);
            bool isolated = true;
            for (usize j = low; j <= high; ++j) {
                if (j == ui) continue;
                if (B[ui + j * n] != T(0)) {
                    isolated = false;
                    break;
                }
            }
            if (isolated) {
                if (perm != nullptr) perm[high] = ui;
                if (ui != high) {
                    for (usize k = 0; k < n; ++k) keel::swap(B[ui + k * n], B[high + k * n]);
                    for (usize k = 0; k < n; ++k) keel::swap(B[k + ui * n], B[k + high * n]);
                }
                if (high == 0) break;
                --high;
                found = true;
                break;
            }
        }
        if (found) continue;

        // Search for isolated columns in [low, high].
        for (usize j = low; j <= high; ++j) {
            bool isolated = true;
            for (usize i = low; i <= high; ++i) {
                if (i == j) continue;
                if (B[i + j * n] != T(0)) {
                    isolated = false;
                    break;
                }
            }
            if (isolated) {
                if (perm != nullptr) perm[low] = j;
                if (j != low) {
                    for (usize k = 0; k < n; ++k) keel::swap(B[j + k * n], B[low + k * n]);
                    for (usize k = 0; k < n; ++k) keel::swap(B[k + j * n], B[k + low * n]);
                }
                ++low;
                found = true;
                break;
            }
        }
    }
}

template <typename T>
void balance_scale(T* B, usize n, f64* d, usize low, usize high) noexcept {
    constexpr f64 kRadix = 2.0;
    constexpr f64 kSqrdx = kRadix * kRadix;
    constexpr f64 kThreshold = 0.95;

    for (usize i = 0; i < n; ++i) d[i] = 1.0;
    if (n <= 1 || low >= high) return;

    bool done = false;
    while (!done) {
        done = true;
        for (usize i = low; i <= high; ++i) {
            f64 r = 0.0;
            f64 c = 0.0;
            for (usize j = low; j <= high; ++j) {
                if (j == i) continue;
                r += keel::abs(B[i + j * n]);
                c += keel::abs(B[j + i * n]);
            }
            if (r == 0.0 || c == 0.0) continue;

            f64 g = r / kRadix;
            f64 f = 1.0;
            const f64 s = c + r;
            while (c < g) {
                f *= kRadix;
                c *= kSqrdx;
            }
            g = r * kRadix;
            while (c >= g) {
                f /= kRadix;
                c /= kSqrdx;
            }
            if ((c + r) / f < kThreshold * s) {
                done = false;
                d[i] *= f;
                for (usize j = low; j <= high; ++j) B[i + j * n] /= f;
                for (usize j = low; j <= high; ++j) B[j + i * n] *= f;
            }
        }
    }
}

template <typename T>
void balance_matrix_impl(T* B, usize n, bool permute, f64* scale, usize* perm, usize* low_out,
                         usize* high_out) noexcept {
    usize low = 0;
    usize high = (n > 0) ? (n - 1) : 0;
    if (n == 0) {
        if (low_out != nullptr) *low_out = 0;
        if (high_out != nullptr) *high_out = 0;
        return;
    }

    if (permute) {
        balance_permute(B, n, low, high, perm);
    } else if (perm != nullptr) {
        for (usize i = 0; i < n; ++i) perm[i] = i;
    }
    balance_scale(B, n, scale, low, high);

    if (low_out != nullptr) *low_out = low;
    if (high_out != nullptr) *high_out = high;
}

} // namespace

// ============================================================================
// QR Updates (Daniel-Gragg-Kaufman-Stewart)
// ============================================================================

void qr_update(f64* Q, f64* R, usize m, usize n, const f64* u, const f64* v, Allocator* alloc) noexcept {
    if (m == 0) return;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> w(m, 0.0, scratch.allocator());

    // w = Q^T * u
    for (usize i = 0; i < m; ++i) {
        f64 s = 0.0;
        for (usize k = 0; k < m; ++k) s += Q[k + i * m] * u[k];
        w[i] = s;
    }

    // Reduce w to [r; 0; ...; 0] via chained Givens rotations from bottom up.
    for (usize i = m - 1; i > 0; --i) {
        f64 c = 0.0;
        f64 s = 0.0;
        f64 r = 0.0;
        givens(w[i - 1], w[i], c, s, r);
        w[i - 1] = r;
        w[i] = 0.0;
        apply_givens_rows(R, m, n, i - 1, i, c, s);
        apply_givens_cols(Q, m, m, i - 1, i, c, s);
    }

    const f64 w0 = w[0];
    for (usize j = 0; j < n; ++j) {
        R[0 + j * m] += w0 * v[j];
    }

    // Restore upper triangular form by chasing subdiagonal bulge top-down.
    const usize kmax = linalg_detail::linalg_min(m - 1, n);
    for (usize i = 0; i < kmax; ++i) {
        f64 c = 0.0;
        f64 s = 0.0;
        f64 r = 0.0;
        givens(R[i + i * m], R[i + 1 + i * m], c, s, r);
        R[i + i * m] = r;
        R[i + 1 + i * m] = 0.0;
        if (i + 1 < n) {
            apply_givens_rows(R + (i + 1) * m, m, n - i - 1, i, i + 1, c, s);
        }
        apply_givens_cols(Q, m, m, i, i + 1, c, s);
    }
}

void qr_insert_col(const f64* Q_in, const f64* R_in, usize m, usize n, usize col, const f64* x, f64* Q_out, f64* R_out,
                   Allocator* alloc) noexcept {
    if (m == 0) return;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> y(m, 0.0, scratch.allocator());

    for (usize i = 0; i < m; ++i) {
        f64 s = 0.0;
        for (usize r = 0; r < m; ++r) s += Q_in[r + i * m] * x[r];
        y[i] = s;
    }

    for (usize j = 0; j < col; ++j) {
        keel::mem_copy(R_out + j * m, R_in + j * m, m * sizeof(f64));
    }
    keel::mem_copy(R_out + col * m, y.data(), m * sizeof(f64));
    for (usize j = col; j < n; ++j) {
        keel::mem_copy(R_out + (j + 1) * m, R_in + j * m, m * sizeof(f64));
    }
    keel::mem_copy(Q_out, Q_in, m * m * sizeof(f64));

    for (usize i = m - 1; i > col; --i) {
        f64 c = 0.0;
        f64 s = 0.0;
        f64 r = 0.0;
        givens(R_out[(i - 1) + col * m], R_out[i + col * m], c, s, r);
        R_out[(i - 1) + col * m] = r;
        R_out[i + col * m] = 0.0;
        if (col + 1 <= n) {
            apply_givens_rows(R_out + (col + 1) * m, m, n - col, i - 1, i, c, s);
        }
        apply_givens_cols(Q_out, m, m, i - 1, i, c, s);
    }
}

void qr_delete_col(const f64* Q_in, const f64* R_in, usize m, usize n, usize col, f64* Q_out, f64* R_out) noexcept {
    if (m == 0 || n == 0) return;

    for (usize j = 0; j < col; ++j) {
        keel::mem_copy(R_out + j * m, R_in + j * m, m * sizeof(f64));
    }
    for (usize j = col + 1; j < n; ++j) {
        keel::mem_copy(R_out + (j - 1) * m, R_in + j * m, m * sizeof(f64));
    }
    keel::mem_copy(Q_out, Q_in, m * m * sizeof(f64));

    const usize lim = linalg_detail::linalg_min(m - 1, n - 1);
    for (usize i = col; i < lim; ++i) {
        f64 c = 0.0;
        f64 s = 0.0;
        f64 r = 0.0;
        givens(R_out[i + i * m], R_out[i + 1 + i * m], c, s, r);
        R_out[i + i * m] = r;
        R_out[i + 1 + i * m] = 0.0;
        if (i + 1 < n - 1) {
            apply_givens_rows(R_out + (i + 1) * m, m, (n - 1) - (i + 1), i, i + 1, c, s);
        }
        apply_givens_cols(Q_out, m, m, i, i + 1, c, s);
    }
}

void qr_insert_row(const f64* Q_in, const f64* R_in, usize m, usize n, usize row, const f64* x, f64* Q_out,
                   f64* R_out) noexcept {
    const usize m1 = m + 1;
    for (usize i = 0; i < m1 * m1; ++i) Q_out[i] = 0.0;
    Q_out[row + 0 * m1] = 1.0;

    for (usize c = 0; c < m; ++c) {
        for (usize r = 0; r < row; ++r) {
            Q_out[r + (c + 1) * m1] = Q_in[r + c * m];
        }
        for (usize r = row; r < m; ++r) {
            Q_out[(r + 1) + (c + 1) * m1] = Q_in[r + c * m];
        }
    }

    for (usize j = 0; j < n; ++j) {
        R_out[0 + j * m1] = x[j];
        for (usize r = 0; r < m; ++r) {
            R_out[(r + 1) + j * m1] = R_in[r + j * m];
        }
    }

    const usize lim = linalg_detail::linalg_min(m, n);
    for (usize i = 0; i < lim; ++i) {
        f64 c = 0.0;
        f64 s = 0.0;
        f64 r = 0.0;
        givens(R_out[i + i * m1], R_out[(i + 1) + i * m1], c, s, r);
        R_out[i + i * m1] = r;
        R_out[(i + 1) + i * m1] = 0.0;
        if (i + 1 < n) {
            apply_givens_rows(R_out + (i + 1) * m1, m1, n - (i + 1), i, i + 1, c, s);
        }
        apply_givens_cols(Q_out, m1, m1, i, i + 1, c, s);
    }
}

void qr_delete_row(const f64* Q_in, const f64* R_in, usize m, usize n, usize row, f64* Q_out, f64* R_out,
                   Allocator* alloc) noexcept {
    if (m == 0) return;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> Qd(m * m, 0.0, scratch.allocator());
    Array<f64> Rd(m * n, 0.0, scratch.allocator());
    keel::mem_copy(Qd.data(), Q_in, m * m * sizeof(f64));
    if (n > 0) {
        keel::mem_copy(Rd.data(), R_in, m * n * sizeof(f64));
    }

    Array<f64> q(m, 0.0, scratch.allocator());
    for (usize j = 0; j < m; ++j) q[j] = Qd[row + j * m];

    for (usize i = m - 1; i > 0; --i) {
        f64 c = 0.0;
        f64 s = 0.0;
        f64 r = 0.0;
        givens(q[i - 1], q[i], c, s, r);
        q[i - 1] = r;
        q[i] = 0.0;
        apply_givens_rows(Rd.data(), m, n, i - 1, i, c, s);
        apply_givens_cols(Qd.data(), m, m, i - 1, i, c, s);
    }

    const usize m1 = m - 1;
    for (usize c = 0; c < m1; ++c) {
        for (usize r = 0; r < row; ++r) {
            Q_out[r + c * m1] = Qd[r + (c + 1) * m];
        }
        for (usize r = row + 1; r < m; ++r) {
            Q_out[(r - 1) + c * m1] = Qd[r + (c + 1) * m];
        }
    }
    for (usize j = 0; j < n; ++j) {
        for (usize r = 1; r < m; ++r) {
            R_out[(r - 1) + j * m1] = Rd[r + j * m];
        }
    }
}

// ============================================================================
// Rank-1 Cholesky Update / Downdate (LINPACK dchud / dchdd)
// ============================================================================

bool cholesky_update(f64* R, usize n, const f64* x, bool downdate, Allocator* alloc) noexcept {
    if (n == 0) return true;

    ArenaTemp scratch = get_scratch(alloc);

    if (!downdate) {
        Array<f64> xb(n, 0.0, scratch.allocator());
        keel::mem_copy(xb.data(), x, n * sizeof(f64));

        for (usize k = 0; k < n; ++k) {
            const f64 rkk = R[k + k * n];
            const f64 xk = xb[k];
            const f64 r = keel::hypot(rkk, xk);
            if (r == 0.0) continue;
            const f64 c = rkk / r;
            const f64 s = xk / r;
            R[k + k * n] = r;
            for (usize j = k + 1; j < n; ++j) {
                const f64 rkj = R[k + j * n];
                R[k + j * n] = c * rkj + s * xb[j];
                xb[j] = c * xb[j] - s * rkj;
            }
        }
        return true;
    }

    // Zero strict lower triangle of R.
    for (usize j = 0; j < n; ++j) {
        for (usize i = j + 1; i < n; ++i) {
            R[i + j * n] = 0.0;
        }
    }

    Array<f64> p(n, 0.0, scratch.allocator());
    f64 norm_sq = 0.0;
    for (usize k = 0; k < n; ++k) {
        const f64 rkk = R[k + k * n];
        if (!(rkk > 0.0) || !keel::is_finite(rkk)) return false;
        f64 sum = x[k];
        for (usize i = 0; i < k; ++i) sum -= R[i + k * n] * p[i];
        if (!keel::is_finite(sum)) return false;
        const f64 pk = sum / rkk;
        p[k] = pk;
        norm_sq += pk * pk;
    }

    const f64 one_minus = 1.0 - norm_sq;
    if (!(one_minus > 0.0)) return false;

    f64 alpha = keel::sqrt(one_minus);
    Array<f64> z(n, 0.0, scratch.allocator());
    for (usize idx = n; idx > 0; --idx) {
        const usize k = idx - 1;
        const f64 scale = keel::hypot(alpha, p[k]);
        const f64 c = alpha / scale;
        const f64 s = p[k] / scale;
        alpha = scale;
        for (usize j = k; j < n; ++j) {
            const f64 rkj = R[k + j * n];
            R[k + j * n] = c * rkj - s * z[j];
            z[j] = s * rkj + c * z[j];
        }
        if (!(R[k + k * n] > 0.0)) return false;
    }
    return true;
}

// ============================================================================
// Crout LDL^T / LDL^H Factorization
// ============================================================================

usize ldl_factor(const f64* A, usize n, f64* L, f64* D, Allocator* alloc) noexcept {
    if (n == 0) return 0;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> A_sym(n * n, 0.0, scratch.allocator());
    keel::mem_copy(A_sym.data(), A, n * n * sizeof(f64));

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < j; ++i) {
            const f64 avg = 0.5 * (A_sym[i + j * n] + A_sym[j + i * n]);
            A_sym[i + j * n] = avg;
            A_sym[j + i * n] = avg;
        }
    }

    f64 anorm = 0.0;
    for (usize k = 0; k < n * n; ++k) anorm += A_sym[k] * A_sym[k];
    anorm = keel::sqrt(anorm);
    const f64 tol = linalg_detail::linalg_max(1e-300, anorm * Limits<f64>::kEpsilon * as<f64>(n));

    for (usize k = 0; k < n * n; ++k) L[k] = 0.0;

    for (usize k = 0; k < n; ++k) {
        f64 s = A_sym[k + k * n];
        for (usize i = 0; i < k; ++i) {
            s -= L[k + i * n] * L[k + i * n] * D[i];
        }
        D[k] = s;
        if (keel::abs(D[k]) < tol) return k + 1;

        L[k + k * n] = 1.0;
        const f64 inv_dk = 1.0 / D[k];
        for (usize j = k + 1; j < n; ++j) {
            f64 t = A_sym[j + k * n];
            for (usize i = 0; i < k; ++i) {
                t -= L[j + i * n] * L[k + i * n] * D[i];
            }
            L[j + k * n] = t * inv_dk;
        }
    }
    return 0;
}

usize ldl_factor(const c64* A, usize n, c64* L, f64* D, Allocator* alloc) noexcept {
    if (n == 0) return 0;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> A_herm(n * n, c64(0.0, 0.0), scratch.allocator());
    keel::mem_copy(A_herm.data(), A, n * n * sizeof(c64));

    for (usize j = 0; j < n; ++j) {
        A_herm[j + j * n] = c64(A_herm[j + j * n].real(), 0.0);
        for (usize i = 0; i < j; ++i) {
            const c64 avg = 0.5 * (A_herm[i + j * n] + keel::conj(A_herm[j + i * n]));
            A_herm[i + j * n] = avg;
            A_herm[j + i * n] = keel::conj(avg);
        }
    }

    f64 anorm = 0.0;
    for (usize k = 0; k < n * n; ++k) anorm += keel::norm(A_herm[k]);
    anorm = keel::sqrt(anorm);
    const f64 tol = linalg_detail::linalg_max(1e-300, anorm * Limits<f64>::kEpsilon * as<f64>(n));

    for (usize k = 0; k < n * n; ++k) L[k] = c64(0.0, 0.0);

    for (usize k = 0; k < n; ++k) {
        f64 s = A_herm[k + k * n].real();
        for (usize i = 0; i < k; ++i) {
            s -= keel::norm(L[k + i * n]) * D[i];
        }
        D[k] = s;
        if (keel::abs(D[k]) < tol) return k + 1;

        L[k + k * n] = c64(1.0, 0.0);
        const f64 inv_dk = 1.0 / D[k];
        for (usize j = k + 1; j < n; ++j) {
            c64 t = A_herm[j + k * n];
            for (usize i = 0; i < k; ++i) {
                t -= L[j + i * n] * keel::conj(L[k + i * n]) * D[i];
            }
            L[j + k * n] = t * inv_dk;
        }
    }
    return 0;
}

// ============================================================================
// Parlett-Reinsch Matrix Balancing
// ============================================================================

void balance_matrix(f64* B, usize n, bool permute, f64* scale, usize* perm, usize* low_out, usize* high_out) noexcept {
    balance_matrix_impl(B, n, permute, scale, perm, low_out, high_out);
}

void balance_matrix(c64* B, usize n, bool permute, f64* scale, usize* perm, usize* low_out, usize* high_out) noexcept {
    balance_matrix_impl(B, n, permute, scale, perm, low_out, high_out);
}

} // namespace keel::linalg
