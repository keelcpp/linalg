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
inline constexpr bool kIsComplex = false;

template <typename U>
inline constexpr bool kIsComplex<Complex<U>> = true;

template <typename T>
[[nodiscard]] inline f64 abs_sq(const T& x) noexcept {
    if constexpr (kIsComplex<T>) {
        return keel::norm(x);
    } else {
        return x * x;
    }
}

// ============================================================================
// 1. Symmetric Eigensolver Helpers (Jacobi for n <= 16, tred2 + tql2 for n > 16)
// ============================================================================

void jacobi_sym_inplace(f64* A, usize n, f64* V, usize max_sweeps, f64 tol) noexcept {
    for (usize i = 0; i < n * n; ++i) V[i] = 0.0;
    for (usize i = 0; i < n; ++i) V[i + i * n] = 1.0;
    if (n <= 1) return;

    f64 diag_sq = 0.0;
    for (usize i = 0; i < n; ++i) diag_sq += A[i + i * n] * A[i + i * n];

    auto off_sum = [&]() noexcept -> f64 {
        f64 s = 0.0;
        for (usize p = 0; p + 1 < n; ++p) {
            for (usize q = p + 1; q < n; ++q) {
                s += A[p + q * n] * A[p + q * n];
            }
        }
        return s;
    };

    const f64 eps = Limits<f64>::kEpsilon;
    const f64 scale_sq = linalg_detail::linalg_max(1.0, diag_sq + 2.0 * off_sum());
    const f64 conv_tol_sq = (tol * tol) * scale_sq;

    for (usize sweep = 0; sweep < max_sweeps; ++sweep) {
        if (off_sum() < conv_tol_sq) break;
        for (usize p = 0; p + 1 < n; ++p) {
            for (usize q = p + 1; q < n; ++q) {
                const f64 apq = A[p + q * n];
                const f64 app = A[p + p * n];
                const f64 aqq = A[q + q * n];
                if (keel::abs(apq) <= eps * (keel::abs(app) + keel::abs(aqq))) {
                    A[p + q * n] = 0.0;
                    continue;
                }
                f64 c = 0.0;
                f64 s = 0.0;
                if (app == aqq) {
                    c = 0.7071067811865476;
                    s = (apq >= 0.0 ? 1.0 : -1.0) * c;
                } else {
                    const f64 tau = (aqq - app) / (2.0 * apq);
                    const f64 t =
                        (tau >= 0.0) ? (1.0 / (tau + keel::sqrt(1.0 + tau * tau)))
                                     : (1.0 / (tau - keel::sqrt(1.0 + tau * tau)));
                    c = 1.0 / keel::sqrt(1.0 + t * t);
                    s = t * c;
                }

                A[p + p * n] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                A[q + q * n] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
                A[p + q * n] = 0.0;
                A[q + p * n] = 0.0;
                for (usize r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    auto get = [&](usize a, usize b) noexcept -> f64& {
                        return (a < b) ? A[a + b * n] : A[b + a * n];
                    };
                    const f64 arp = get(r, p);
                    const f64 arq = get(r, q);
                    get(r, p) = c * arp - s * arq;
                    get(r, q) = s * arp + c * arq;
                }
                for (usize r = 0; r < n; ++r) {
                    const f64 vrp = V[r + p * n];
                    const f64 vrq = V[r + q * n];
                    V[r + p * n] = c * vrp - s * vrq;
                    V[r + q * n] = s * vrp + c * vrq;
                }
            }
        }
    }

    for (usize p = 0; p + 1 < n; ++p) {
        for (usize q = p + 1; q < n; ++q) {
            A[q + p * n] = A[p + q * n];
        }
    }
}

void sym_tridiag_ql(f64* V, usize n, f64* d, Allocator* scratch_alloc) noexcept {
    Array<f64> e(n, 0.0, scratch_alloc);

    for (usize i = n - 1; i >= 1; --i) {
        const usize l = i - 1;
        f64 h = 0.0;
        f64 scale = 0.0;
        if (l > 0) {
            for (usize k = 0; k <= l; ++k) scale += keel::abs(V[i + k * n]);
            if (scale == 0.0) {
                e[i] = V[i + l * n];
            } else {
                for (usize k = 0; k <= l; ++k) {
                    V[i + k * n] /= scale;
                    h += V[i + k * n] * V[i + k * n];
                }
                f64 f = V[i + l * n];
                f64 g = (f >= 0.0) ? -keel::sqrt(h) : keel::sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                V[i + l * n] = f - g;
                f = 0.0;
                for (usize j = 0; j <= l; ++j) {
                    V[j + i * n] = V[i + j * n] / h;
                    g = 0.0;
                    for (usize k = 0; k <= j; ++k) g += V[j + k * n] * V[i + k * n];
                    for (usize k = j + 1; k <= l; ++k) g += V[k + j * n] * V[i + k * n];
                    e[j] = g / h;
                    f += e[j] * V[i + j * n];
                }
                const f64 hh = f / (h + h);
                for (usize j = 0; j <= l; ++j) {
                    f = V[i + j * n];
                    g = e[j] - hh * f;
                    e[j] = g;
                    for (usize k = 0; k <= j; ++k) {
                        V[j + k * n] -= (f * e[k] + g * V[i + k * n]);
                    }
                }
            }
        } else {
            e[i] = V[i + l * n];
        }
        d[i] = h;
    }

    d[0] = 0.0;
    e[0] = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (d[i] != 0.0) {
            for (usize j = 0; j < i; ++j) {
                f64 g = 0.0;
                for (usize k = 0; k < i; ++k) g += V[i + k * n] * V[k + j * n];
                for (usize k = 0; k < i; ++k) V[k + j * n] -= g * V[k + i * n];
            }
        }
        d[i] = V[i + i * n];
        V[i + i * n] = 1.0;
        for (usize j = 0; j < i; ++j) {
            V[j + i * n] = 0.0;
            V[i + j * n] = 0.0;
        }
    }

    for (usize i = 1; i < n; ++i) e[i - 1] = e[i];
    e[n - 1] = 0.0;

    const f64 eps = Limits<f64>::kEpsilon;
    for (usize l = 0; l < n; ++l) {
        usize iter = 0;
        usize m = l;
        for (;;) {
            for (m = l; m + 1 < n; ++m) {
                const f64 dd = keel::abs(d[m]) + keel::abs(d[m + 1]);
                if (keel::abs(e[m]) <= eps * dd) {
                    e[m] = 0.0;
                    break;
                }
            }
            if (m == l) break;
            if (++iter > 300) break;
            f64 g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            f64 r = keel::hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + (g >= 0.0 ? r : -r));
            f64 s = 1.0;
            f64 c = 1.0;
            f64 p = 0.0;
            bool underflow = false;
            for (usize ii = m; ii > l; --ii) {
                const usize i = ii - 1;
                const f64 f = s * e[i];
                const f64 b = c * e[i];
                r = keel::hypot(f, g);
                e[i + 1] = r;
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                f64* vi1 = V + (i + 1) * n;
                f64* vi = V + i * n;
                for (usize k = 0; k < n; ++k) {
                    const f64 t = vi1[k];
                    vi1[k] = s * vi[k] + c * t;
                    vi[k] = c * vi[k] - s * t;
                }
            }
            if (underflow) continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

// ============================================================================
// 2. Skyline Cholesky & Symmetric Generalized Helpers
// ============================================================================

bool try_cholesky_upper_skyline(const f64* M, usize n, f64* R, Allocator* scratch_alloc) noexcept {
    f64 diag_max = 0.0;
    for (usize i = 0; i < n; ++i) diag_max = linalg_detail::linalg_max(diag_max, keel::abs(M[i + i * n]));
    if (!(diag_max > 0.0)) return false;
    const f64 tol = 1e-12 * diag_max;

    Array<usize> first_nz(n, usize{0}, scratch_alloc);
    for (usize j = 0; j < n; ++j) {
        const f64* mj = M + j * n;
        usize fnz = j;
        for (usize i = 0; i < j; ++i) {
            if (mj[i] != 0.0) {
                fnz = i;
                break;
            }
        }
        first_nz[j] = fnz;
    }

    for (usize j = 0; j < n; ++j) {
        f64* rj = R + j * n;
        const usize fnz_j = first_nz[j];
        for (usize i = 0; i < fnz_j; ++i) rj[i] = 0.0;
        for (usize i = fnz_j; i < j; ++i) {
            const f64* ri = R + i * n;
            f64 s = M[i + j * n];
            const usize k_start = linalg_detail::linalg_max(first_nz[i], fnz_j);
            for (usize k = k_start; k < i; ++k) s -= ri[k] * rj[k];
            rj[i] = s / ri[i];
        }
        f64 sum = 0.0;
        for (usize k = fnz_j; k < j; ++k) sum += rj[k] * rj[k];
        const f64 d = M[j + j * n] - sum;
        if (!(d > tol)) return false;
        rj[j] = keel::sqrt(d);
        for (usize i = j + 1; i < n; ++i) rj[i] = 0.0;
    }
    return true;
}

void form_symmetric_reduced(const f64* M, const f64* R, usize n, f64* C, Allocator* scratch_alloc) noexcept {
    Array<f64> Y(n * n, 0.0, scratch_alloc);
    for (usize col = 0; col < n; ++col) {
        f64* ycol = Y.data() + col * n;
        const f64* mcol = M + col * n;
        for (usize i = 0; i < n; ++i) {
            const f64* ri = R + i * n;
            f64 s = mcol[i];
            for (usize k = 0; k < i; ++k) s -= ri[k] * ycol[k];
            ycol[i] = s / ri[i];
        }
    }
    Array<f64> Yt(n * n, 0.0, scratch_alloc);
    for (usize col = 0; col < n; ++col) {
        const f64* ycol = Y.data() + col * n;
        for (usize row = 0; row < n; ++row) Yt[col + row * n] = ycol[row];
    }
    for (usize r = 0; r < n; ++r) {
        f64* cr = C + r * n;
        const f64* ytr = Yt.data() + r * n;
        for (usize i = 0; i <= r; ++i) {
            const f64* ri = R + i * n;
            f64 s = ytr[i];
            for (usize k = 0; k < i; ++k) s -= ri[k] * cr[k];
            cr[i] = s / ri[i];
        }
    }
    for (usize r = 0; r < n; ++r) {
        for (usize i = 0; i < r; ++i) C[r + i * n] = C[i + r * n];
    }
}

void backsolve_r_mat(const f64* R, const f64* Q, usize n, f64* V) noexcept {
    keel::mem_copy(V, Q, n * n * sizeof(f64));
    for (usize col = 0; col < n; ++col) {
        f64* vcol = V + col * n;
        for (usize ii = n; ii > 0; --ii) {
            const usize i = ii - 1;
            const f64* ri = R + i * n;
            const f64 vii = vcol[i] / ri[i];
            vcol[i] = vii;
            if (vii != 0.0) {
                for (usize k = 0; k < i; ++k) vcol[k] -= ri[k] * vii;
            }
        }
    }
}

// ============================================================================
// 3. Hessenberg Reduction & Real/Complex Schur QR
// ============================================================================

template <typename T>
void hess_reduce_impl(T* A, usize n, T* P, Allocator* alloc) noexcept {
    for (usize i = 0; i < n * n; ++i) P[i] = T(0);
    for (usize i = 0; i < n; ++i) P[i + i * n] = T(1);
    if (n < 3) return;

    ArenaTemp scratch = get_scratch(alloc);
    Array<T> v_storage(n, T(0), scratch.allocator());
    T* v = v_storage.data();

    for (usize k = 0; k + 2 < n; ++k) {
        f64 norm_sq = 0.0;
        for (usize i = k + 1; i < n; ++i) norm_sq += abs_sq(A[i + k * n]);
        if (norm_sq == 0.0) continue;
        const T xk = A[k + 1 + k * n];
        const f64 norm = keel::sqrt(norm_sq);

        T alpha;
        if constexpr (kIsComplex<T>) {
            const f64 abs_xk = keel::abs(xk);
            const T phase = (abs_xk > 0.0) ? (xk / abs_xk) : T(1.0, 0.0);
            alpha = -phase * norm;
        } else {
            alpha = (xk >= 0.0) ? -norm : norm;
        }

        v[k + 1] = xk - alpha;
        for (usize i = k + 2; i < n; ++i) v[i] = A[i + k * n];
        f64 v_norm_sq = 0.0;
        for (usize i = k + 1; i < n; ++i) v_norm_sq += abs_sq(v[i]);
        if (v_norm_sq == 0.0) continue;
        const T tau = T(2.0 / v_norm_sq);

        for (usize j = k; j < n; ++j) {
            T dp = T(0);
            for (usize i = k + 1; i < n; ++i) {
                if constexpr (kIsComplex<T>) {
                    dp += keel::conj(v[i]) * A[i + j * n];
                } else {
                    dp += v[i] * A[i + j * n];
                }
            }
            const T s = tau * dp;
            for (usize i = k + 1; i < n; ++i) A[i + j * n] -= s * v[i];
        }
        for (usize i = 0; i < n; ++i) {
            T dp = T(0);
            for (usize j = k + 1; j < n; ++j) dp += A[i + j * n] * v[j];
            const T s = tau * dp;
            for (usize j = k + 1; j < n; ++j) {
                if constexpr (kIsComplex<T>) {
                    A[i + j * n] -= s * keel::conj(v[j]);
                } else {
                    A[i + j * n] -= s * v[j];
                }
            }
        }
        for (usize i = 0; i < n; ++i) {
            T dp = T(0);
            for (usize j = k + 1; j < n; ++j) dp += P[i + j * n] * v[j];
            const T s = tau * dp;
            for (usize j = k + 1; j < n; ++j) {
                if constexpr (kIsComplex<T>) {
                    P[i + j * n] -= s * keel::conj(v[j]);
                } else {
                    P[i + j * n] -= s * v[j];
                }
            }
        }
        A[k + 1 + k * n] = alpha;
        for (usize i = k + 2; i < n; ++i) A[i + k * n] = T(0);
    }
}

void francis_schur(f64* H, f64* Z, usize n) noexcept {
    if (n < 3) return;
    auto h = [&](isize i, isize j) noexcept -> f64& {
        return H[static_cast<usize>(i) + static_cast<usize>(j) * n];
    };
    auto z = [&](isize i, isize j) noexcept -> f64& {
        return Z[static_cast<usize>(i) + static_cast<usize>(j) * n];
    };

    const f64 eps = Limits<f64>::kEpsilon;
    const isize N = static_cast<isize>(n);

    f64 hnorm = 0.0;
    for (usize j = 0; j < n; ++j) {
        f64 col_sum = 0.0;
        const usize i_end = linalg_detail::linalg_min(n, j + 2);
        for (usize i = 0; i < i_end; ++i) col_sum += keel::abs(H[i + j * n]);
        if (col_sum > hnorm) hnorm = col_sum;
    }
    if (hnorm == 0.0) return;
    const f64 smlnum = Limits<f64>::kMinPositive * (static_cast<f64>(n) / eps);

    isize p = N - 1;
    i32 iter = 0;
    while (p >= 0) {
        isize l = p;
        while (l > 0) {
            f64 s = keel::abs(h(l - 1, l - 1)) + keel::abs(h(l, l));
            if (s == 0.0) s = hnorm;
            if (keel::abs(h(l, l - 1)) <= linalg_detail::linalg_max(eps * s, smlnum)) {
                h(l, l - 1) = 0.0;
                break;
            }
            --l;
        }
        if (l == p) {
            p -= 1;
            iter = 0;
            continue;
        }
        if (l == p - 1) {
            p -= 2;
            iter = 0;
            continue;
        }
        if (++iter > 600) {
            p -= 2;
            iter = 0;
            continue;
        }

        f64 s = h(p - 1, p - 1) + h(p, p);
        f64 t = h(p - 1, p - 1) * h(p, p) - h(p - 1, p) * h(p, p - 1);
        if (iter % 10 == 0) {
            const f64 e = keel::abs(h(p, p - 1)) + (p >= 2 ? keel::abs(h(p - 1, p - 2)) : 0.0);
            const f64 d1 = h(p, p) + 0.75 * e;
            s = 2.0 * d1;
            t = d1 * d1 - 0.4375 * e * e;
        }

        f64 x = h(l, l) * h(l, l) + h(l, l + 1) * h(l + 1, l) - s * h(l, l) + t;
        f64 y = h(l + 1, l) * (h(l, l) + h(l + 1, l + 1) - s);
        f64 w = (l + 2 <= p) ? h(l + 1, l) * h(l + 2, l + 1) : 0.0;

        for (isize k = l; k <= p - 1; ++k) {
            const isize r = (k <= p - 2) ? 3 : 2;
            const f64 scale = keel::abs(x) + keel::abs(y) + (r == 3 ? keel::abs(w) : 0.0);
            if (scale == 0.0) continue;
            const f64 xs = x / scale;
            const f64 ys = y / scale;
            const f64 ws = (r == 3) ? (w / scale) : 0.0;
            f64 alpha = keel::sqrt(xs * xs + ys * ys + ws * ws);
            if (xs < 0.0) alpha = -alpha;
            const f64 v0 = xs + alpha;
            const f64 v1 = ys;
            const f64 v2 = ws;
            const f64 vnorm2 = v0 * v0 + v1 * v1 + v2 * v2;
            if (vnorm2 == 0.0) continue;
            const f64 tau = 2.0 / vnorm2;

            const isize col0 = (k > l) ? (k - 1) : l;
            for (isize j = col0; j < N; ++j) {
                f64 d = v0 * h(k, j) + v1 * h(k + 1, j) + (r == 3 ? v2 * h(k + 2, j) : 0.0);
                d *= tau;
                h(k, j) -= d * v0;
                h(k + 1, j) -= d * v1;
                if (r == 3) h(k + 2, j) -= d * v2;
            }
            const isize row_end = linalg_detail::linalg_min(k + r, p);
            for (isize i = 0; i <= row_end; ++i) {
                f64 d = v0 * h(i, k) + v1 * h(i, k + 1) + (r == 3 ? v2 * h(i, k + 2) : 0.0);
                d *= tau;
                h(i, k) -= d * v0;
                h(i, k + 1) -= d * v1;
                if (r == 3) h(i, k + 2) -= d * v2;
            }
            for (isize i = 0; i < N; ++i) {
                f64 d = v0 * z(i, k) + v1 * z(i, k + 1) + (r == 3 ? v2 * z(i, k + 2) : 0.0);
                d *= tau;
                z(i, k) -= d * v0;
                z(i, k + 1) -= d * v1;
                if (r == 3) z(i, k + 2) -= d * v2;
            }
            x = h(k + 1, k);
            y = h(k + 2, k);
            w = (k + 3 <= p) ? h(k + 3, k) : 0.0;
        }
    }

    for (usize j = 0; j + 2 < n; ++j) {
        for (usize i = j + 2; i < n; ++i) H[i + j * n] = 0.0;
    }
}

void standardize_schur_2x2(f64* H, f64* Z, usize n) noexcept {
    if (n < 2) return;
    auto h = [&](isize i, isize j) noexcept -> f64& {
        return H[static_cast<usize>(i) + static_cast<usize>(j) * n];
    };
    auto z = [&](isize i, isize j) noexcept -> f64& {
        return Z[static_cast<usize>(i) + static_cast<usize>(j) * n];
    };
    const isize N = static_cast<isize>(n);

    for (isize k = 0; k + 1 < N; ++k) {
        if (h(k + 1, k) == 0.0) continue;
        if (k + 2 < N && h(k + 2, k + 1) != 0.0) continue;

        f64 a = h(k, k);
        f64 b = h(k, k + 1);
        f64 c = h(k + 1, k);
        f64 d = h(k + 1, k + 1);
        if (c == 0.0) continue;

        const f64 p = 0.5 * (a - d);
        const f64 bcmax = linalg_detail::linalg_max(keel::abs(b), keel::abs(c));
        const f64 bcmis =
            linalg_detail::linalg_min(keel::abs(b), keel::abs(c)) * (b >= 0.0 ? 1.0 : -1.0) * (c >= 0.0 ? 1.0 : -1.0);
        const f64 scale = linalg_detail::linalg_max(keel::abs(p), bcmax);
        const f64 zz = (p / scale) * p + (bcmax / scale) * bcmis;

        if (zz >= 0.0) {
            const f64 zr = p + (p >= 0.0 ? 1.0 : -1.0) * keel::sqrt(scale) * keel::sqrt(zz);
            a = d + zr;
            d -= (bcmax / zr) * bcmis;
            const f64 tau = keel::hypot(c, zr);
            const f64 cs = zr / tau;
            const f64 sn = c / tau;
            b = b - c;
            h(k, k) = a;
            h(k, k + 1) = b;
            h(k + 1, k) = 0.0;
            h(k + 1, k + 1) = d;
            for (isize j = k + 2; j < N; ++j) {
                const f64 t1 = h(k, j);
                const f64 t2 = h(k + 1, j);
                h(k, j) = cs * t1 + sn * t2;
                h(k + 1, j) = -sn * t1 + cs * t2;
            }
            for (isize i = 0; i < k; ++i) {
                const f64 t1 = h(i, k);
                const f64 t2 = h(i, k + 1);
                h(i, k) = cs * t1 + sn * t2;
                h(i, k + 1) = -sn * t1 + cs * t2;
            }
            for (isize i = 0; i < N; ++i) {
                const f64 t1 = z(i, k);
                const f64 t2 = z(i, k + 1);
                z(i, k) = cs * t1 + sn * t2;
                z(i, k + 1) = -sn * t1 + cs * t2;
            }
        }
        ++k;
    }
}

bool complex_schur_qr(c64* H, c64* Z, usize n) noexcept {
    if (n < 2) return true;
    auto h = [&](isize i, isize j) noexcept -> c64& {
        return H[static_cast<usize>(i) + static_cast<usize>(j) * n];
    };
    auto z = [&](isize i, isize j) noexcept -> c64& {
        return Z[static_cast<usize>(i) + static_cast<usize>(j) * n];
    };

    const f64 eps = Limits<f64>::kEpsilon;
    const isize N = static_cast<isize>(n);

    isize p = N - 1;
    isize iter = 0;

    while (p > 0) {
        isize l = p;
        while (l > 0) {
            f64 s = keel::abs(h(l - 1, l - 1)) + keel::abs(h(l, l));
            if (s == 0.0) s = 1.0;
            if (keel::abs(h(l, l - 1)) <= eps * s) {
                h(l, l - 1) = c64(0.0, 0.0);
                break;
            }
            --l;
        }
        if (l == p) {
            p -= 1;
            iter = 0;
            continue;
        }
        if (++iter > 60 * (N + 1)) return false;

        const c64 a = h(p - 1, p - 1);
        const c64 b = h(p - 1, p);
        const c64 c_blk = h(p, p - 1);
        const c64 d = h(p, p);

        const c64 tr = a + d;
        const c64 det = a * d - b * c_blk;
        const c64 disc = keel::sqrt(tr * tr - 4.0 * det);

        const c64 mu1 = 0.5 * (tr + disc);
        const c64 mu2 = 0.5 * (tr - disc);
        c64 shift = (keel::abs(mu1 - d) < keel::abs(mu2 - d)) ? mu1 : mu2;

        if (iter % 30 == 0) {
            const isize prev = (p >= 2) ? (p - 2) : 0;
            shift += c64(keel::abs(c_blk), keel::abs(h(p - 1, prev)));
        }

        for (isize k = l; k < p; ++k) {
            const c64 f = (k == l) ? (h(k, k) - shift) : h(k, k - 1);
            const c64 g = (k == l) ? h(k + 1, k) : h(k + 1, k - 1);

            const f64 norm = keel::sqrt(keel::norm(f) + keel::norm(g));
            if (norm == 0.0) continue;

            const f64 c = keel::abs(f) / norm;
            const c64 phase = (keel::abs(f) > 0.0) ? (f / keel::abs(f)) : c64(1.0, 0.0);
            const c64 s = phase * keel::conj(g) / norm;
            const c64 s_conj = keel::conj(s);

            const isize col0 = (k > 0) ? (k - 1) : 0;
            for (isize j = col0; j < N; ++j) {
                const c64 hk = h(k, j);
                const c64 hk1 = h(k + 1, j);
                h(k, j) = c * hk + s * hk1;
                h(k + 1, j) = -s_conj * hk + c * hk1;
            }
            const isize row1 = linalg_detail::linalg_min(k + 2, N - 1);
            for (isize i = 0; i <= row1; ++i) {
                const c64 hk = h(i, k);
                const c64 hk1 = h(i, k + 1);
                h(i, k) = c * hk + s_conj * hk1;
                h(i, k + 1) = -s * hk + c * hk1;
            }
            for (isize i = 0; i < N; ++i) {
                const c64 zk = z(i, k);
                const c64 zk1 = z(i, k + 1);
                z(i, k) = c * zk + s_conj * zk1;
                z(i, k + 1) = -s * zk + c * zk1;
            }
        }
    }

    for (usize j = 0; j + 1 < n; ++j) {
        for (usize i = j + 1; i < n; ++i) H[i + j * n] = c64(0.0, 0.0);
    }
    return true;
}

void swap_adjacent_schur(c64* T, c64* U, usize n, usize j) noexcept {
    const c64 d1 = T[j + j * n];
    const c64 d2 = T[j + 1 + (j + 1) * n];
    const c64 t12 = T[j + (j + 1) * n];

    // Right eigenvector of [d1, t12; 0, d2] for eigenvalue d2 is [t12; d2 - d1].
    const c64 a = t12;
    const c64 b = d2 - d1;
    const f64 norm_a = keel::abs(a);
    const f64 norm_b = keel::abs(b);
    const f64 r = keel::hypot(norm_a, norm_b);
    if (r == 0.0) return;

    const f64 c = norm_a / r;
    const c64 phase = (norm_a > 0.0) ? (a / norm_a) : c64(1.0, 0.0);
    const c64 s = phase * keel::conj(b) / r;

    const c64 g00(c, 0.0);
    const c64 g01 = s;
    const c64 g10 = -keel::conj(s);
    const c64 g11(c, 0.0);

    for (usize k = j; k < n; ++k) {
        const c64 r0 = T[j + k * n];
        const c64 r1 = T[j + 1 + k * n];
        T[j + k * n] = g00 * r0 + g01 * r1;
        T[j + 1 + k * n] = g10 * r0 + g11 * r1;
    }

    const c64 gh00 = keel::conj(g00);
    const c64 gh01 = keel::conj(g10);
    const c64 gh10 = keel::conj(g01);
    const c64 gh11 = keel::conj(g11);

    for (usize i = 0; i <= j + 1; ++i) {
        const c64 c0 = T[i + j * n];
        const c64 c1 = T[i + (j + 1) * n];
        T[i + j * n] = c0 * gh00 + c1 * gh10;
        T[i + (j + 1) * n] = c0 * gh01 + c1 * gh11;
    }
    T[(j + 1) + j * n] = c64(0.0, 0.0);

    for (usize i = 0; i < n; ++i) {
        const c64 c0 = U[i + j * n];
        const c64 c1 = U[i + (j + 1) * n];
        U[i + j * n] = c0 * gh00 + c1 * gh10;
        U[i + (j + 1) * n] = c0 * gh01 + c1 * gh11;
    }
}

inline bool is_approx_conj(c64 a, c64 b, f64 tol) noexcept {
    return keel::abs(a.real() - b.real()) <= tol * (1.0 + keel::abs(a.real())) &&
           keel::abs(a.imag() + b.imag()) <= tol * (1.0 + keel::abs(a.imag()));
}

} // namespace

// ============================================================================
// Public API Implementations
// ============================================================================

void eig_sym(const f64* A, usize n, f64* V, f64* D, Allocator* alloc) noexcept {
    if (n == 0) return;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> V_work(n * n, 0.0, scratch.allocator());
    Array<f64> evals(n, 0.0, scratch.allocator());

    if (n <= 16) {
        Array<f64> A_work(n * n, 0.0, scratch.allocator());
        keel::mem_copy(A_work.data(), A, n * n * sizeof(f64));
        jacobi_sym_inplace(A_work.data(), n, V_work.data(), 64, 1e-13);
        for (usize i = 0; i < n; ++i) evals[i] = A_work[i + i * n];
    } else {
        keel::mem_copy(V_work.data(), A, n * n * sizeof(f64));
        sym_tridiag_ql(V_work.data(), n, evals.data(), scratch.allocator());
    }

    // LAPACK dsteqr ascending selection sort (preserves exact order on ties).
    Array<usize> order(n, usize{0}, scratch.allocator());
    for (usize i = 0; i < n; ++i) order[i] = i;
    for (usize ii = 1; ii < n; ++ii) {
        const usize i = ii - 1;
        usize k = i;
        f64 p = evals[order[i]];
        for (usize j = ii; j < n; ++j) {
            if (evals[order[j]] < p) {
                k = j;
                p = evals[order[j]];
            }
        }
        if (k != i) keel::swap(order[i], order[k]);
    }

    for (usize k = 0; k < n; ++k) {
        const usize src = order[k];
        D[k] = evals[src];
        for (usize i = 0; i < n; ++i) V[i + k * n] = V_work[i + src * n];
    }
}

bool eig_sym_generalized(const f64* A, const f64* B, usize n, f64* V, f64* D, Allocator* alloc) noexcept {
    if (n == 0) return true;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> R(n * n, 0.0, scratch.allocator());

    // Case 1: B is SPD -> standard Cholesky reduction (R^-T A R^-1) Q = Q D, V = R^-1 Q.
    if (try_cholesky_upper_skyline(B, n, R.data(), scratch.allocator())) {
        Array<f64> C(n * n, 0.0, scratch.allocator());
        Array<f64> Q(n * n, 0.0, scratch.allocator());
        form_symmetric_reduced(A, R.data(), n, C.data(), scratch.allocator());
        eig_sym(C.data(), n, Q.data(), D, scratch.allocator());
        backsolve_r_mat(R.data(), Q.data(), n, V);
        return true;
    }

    // Case 2: B is indefinite/singular, A is SPD -> reciprocal reduction.
    if (try_cholesky_upper_skyline(A, n, R.data(), scratch.allocator())) {
        Array<f64> C(n * n, 0.0, scratch.allocator());
        Array<f64> Q(n * n, 0.0, scratch.allocator());
        Array<f64> dmu(n, 0.0, scratch.allocator());
        form_symmetric_reduced(B, R.data(), n, C.data(), scratch.allocator());
        eig_sym(C.data(), n, Q.data(), dmu.data(), scratch.allocator());

        Array<f64> Vraw(n * n, 0.0, scratch.allocator());
        backsolve_r_mat(R.data(), Q.data(), n, Vraw.data());

        f64 mu_max = 0.0;
        for (usize j = 0; j < n; ++j) mu_max = linalg_detail::linalg_max(mu_max, keel::abs(dmu[j]));
        const f64 mu_tol = 1e-11 * linalg_detail::linalg_max(1.0, mu_max);

        Array<f64> lam(n, 0.0, scratch.allocator());
        Array<usize> order(n, usize{0}, scratch.allocator());
        for (usize j = 0; j < n; ++j) {
            order[j] = j;
            const f64 mu = dmu[j];
            lam[j] = (keel::abs(mu) > mu_tol) ? (1.0 / mu) : Limits<f64>::infinity();
            usize i_max = 0;
            f64 max_abs = 0.0;
            for (usize i = 0; i < n; ++i) {
                const f64 a = keel::abs(Vraw[i + j * n]);
                if (a > max_abs) {
                    max_abs = a;
                    i_max = i;
                }
            }
            if (max_abs > 0.0) {
                const f64 scale = Vraw[i_max + j * n];
                for (usize i = 0; i < n; ++i) Vraw[i + j * n] /= scale;
            }
        }

        // Ascending sort of eigenvalues.
        for (usize i = 0; i + 1 < n; ++i) {
            usize best = i;
            for (usize j = i + 1; j < n; ++j) {
                if (lam[order[j]] < lam[order[best]]) best = j;
            }
            if (best != i) keel::swap(order[i], order[best]);
        }

        for (usize col = 0; col < n; ++col) {
            const usize src = order[col];
            D[col] = lam[src];
            for (usize row = 0; row < n; ++row) V[row + col * n] = Vraw[row + src * n];
        }
        return true;
    }

    return false;
}

bool eigs_sym_generalized_smallest_abs(const f64* A, const f64* B, usize n, usize k, f64* V, f64* D,
                                       Allocator* alloc) noexcept {
    if (n == 0 || k == 0 || k > n) return false;

    ArenaTemp scratch = get_scratch(alloc);

    Array<usize> a_first_nz(n, usize{0}, scratch.allocator());
    for (usize j = 0; j < n; ++j) {
        const f64* aj = A + j * n;
        usize fnz = j;
        for (usize i = 0; i < j; ++i) {
            if (aj[i] != 0.0) {
                fnz = i;
                break;
            }
        }
        a_first_nz[j] = fnz;
    }

    Array<f64> RA(n * n, 0.0, scratch.allocator());
    if (!try_cholesky_upper_skyline(A, n, RA.data(), scratch.allocator())) return false;

    Array<usize> b_first(n, usize{0}, scratch.allocator());
    Array<usize> b_last(n, usize{0}, scratch.allocator());
    bool b_diag_all_pos = true;
    for (usize j = 0; j < n; ++j) {
        const f64* bj = B + j * n;
        if (!(bj[j] > 0.0)) b_diag_all_pos = false;
        usize fnz = n;
        usize lnz = 0;
        for (usize i = 0; i < n; ++i) {
            if (bj[i] != 0.0) {
                if (fnz == n) fnz = i;
                lnz = i + 1;
            }
        }
        b_first[j] = fnz;
        b_last[j] = lnz;
    }

    const f64* R = RA.data();
    auto solve_rt = [&](const f64* rhs, f64* y) noexcept {
        for (usize i = 0; i < n; ++i) {
            const f64* ri = R + i * n;
            f64 s = rhs[i];
            for (usize p = a_first_nz[i]; p < i; ++p) s -= ri[p] * y[p];
            y[i] = s / ri[i];
        }
    };
    auto solve_r = [&](const f64* rhs, f64* x) noexcept {
        keel::mem_copy(x, rhs, n * sizeof(f64));
        for (usize ii = n; ii > 0; --ii) {
            const usize i = ii - 1;
            const f64* ri = R + i * n;
            const f64 xi = x[i] / ri[i];
            x[i] = xi;
            if (xi != 0.0) {
                for (usize p = a_first_nz[i]; p < i; ++p) x[p] -= ri[p] * xi;
            }
        }
    };
    auto matvec_b = [&](const f64* z, f64* u) noexcept {
        for (usize i = 0; i < n; ++i) u[i] = 0.0;
        for (usize j = 0; j < n; ++j) {
            const f64 zj = z[j];
            if (zj == 0.0) continue;
            const usize f = b_first[j];
            const usize l = b_last[j];
            const f64* bj = B + j * n;
            for (usize i = f; i < l; ++i) u[i] += bj[i] * zj;
        }
    };

    const usize m = linalg_detail::linalg_min(n, linalg_detail::linalg_max(3 * k + 24, usize{64}));
    Array<f64> Q(n * m, 0.0, scratch.allocator());
    Array<f64> alpha(m, 0.0, scratch.allocator());
    Array<f64> beta(m, 0.0, scratch.allocator());
    Array<f64> z_buf(n, 0.0, scratch.allocator());
    Array<f64> u_buf(n, 0.0, scratch.allocator());
    Array<f64> w_buf(n, 0.0, scratch.allocator());

    f64* q0 = Q.data();
    u64 seed = 0x9E3779B97F4A7C15ULL;
    f64 norm0 = 0.0;
    for (usize i = 0; i < n; ++i) {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        const f64 val =
            static_cast<f64>(static_cast<i64>(seed * 0x2545F4914F6CDD1DULL)) * (1.0 / 9223372036854775808.0);
        q0[i] = val;
        norm0 += val * val;
    }
    norm0 = keel::sqrt(norm0);
    for (usize i = 0; i < n; ++i) q0[i] /= norm0;

    usize m_actual = 0;
    for (usize j = 0; j < m; ++j) {
        const f64* qj = Q.data() + j * n;
        solve_r(qj, z_buf.data());
        matvec_b(z_buf.data(), u_buf.data());
        solve_rt(u_buf.data(), w_buf.data());

        f64 aj = 0.0;
        for (usize i = 0; i < n; ++i) aj += qj[i] * w_buf[i];
        alpha[j] = aj;

        for (usize pass = 0; pass < 2; ++pass) {
            for (usize r = 0; r <= j; ++r) {
                const f64* qr = Q.data() + r * n;
                f64 dp = 0.0;
                for (usize i = 0; i < n; ++i) dp += qr[i] * w_buf[i];
                for (usize i = 0; i < n; ++i) w_buf[i] -= dp * qr[i];
            }
        }

        f64 bnorm = 0.0;
        for (usize i = 0; i < n; ++i) bnorm += w_buf[i] * w_buf[i];
        bnorm = keel::sqrt(bnorm);
        beta[j] = bnorm;
        m_actual = j + 1;

        if (j + 1 < m) {
            f64* q_next = Q.data() + (j + 1) * n;
            if (bnorm > 1e-14) {
                const f64 inv_b = 1.0 / bnorm;
                for (usize i = 0; i < n; ++i) q_next[i] = w_buf[i] * inv_b;
            } else {
                for (usize i = 0; i < n; ++i) q_next[i] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
                for (usize pass = 0; pass < 2; ++pass) {
                    for (usize r = 0; r <= j; ++r) {
                        const f64* qr = Q.data() + r * n;
                        f64 dp = 0.0;
                        for (usize i = 0; i < n; ++i) dp += qr[i] * q_next[i];
                        for (usize i = 0; i < n; ++i) q_next[i] -= dp * qr[i];
                    }
                }
                f64 nrm = 0.0;
                for (usize i = 0; i < n; ++i) nrm += q_next[i] * q_next[i];
                nrm = keel::sqrt(nrm);
                if (nrm <= 1e-14) break;
                for (usize i = 0; i < n; ++i) q_next[i] /= nrm;
            }
        }
    }

    if (m_actual < k) return false;

    Array<f64> S(m_actual * m_actual, 0.0, scratch.allocator());
    for (usize i = 0; i < m_actual; ++i) {
        S[i + i * m_actual] = alpha[i];
        if (i + 1 < m_actual) {
            S[(i + 1) + i * m_actual] = beta[i];
            S[i + (i + 1) * m_actual] = beta[i];
        }
    }
    Array<f64> d_trid(m_actual, 0.0, scratch.allocator());
    sym_tridiag_ql(S.data(), m_actual, d_trid.data(), scratch.allocator());

    f64 mu_max = 0.0;
    for (usize i = 0; i < m_actual; ++i) mu_max = linalg_detail::linalg_max(mu_max, keel::abs(d_trid[i]));
    const f64 mu_tol = 1e-12 * linalg_detail::linalg_max(1.0, mu_max);

    Array<usize> idx(m_actual, usize{0}, scratch.allocator());
    for (usize i = 0; i < m_actual; ++i) idx[i] = i;
    // Select k largest |mu|.
    for (usize i = 0; i < k; ++i) {
        usize best = i;
        for (usize j = i + 1; j < m_actual; ++j) {
            if (keel::abs(d_trid[idx[j]]) > keel::abs(d_trid[idx[best]])) best = j;
        }
        if (best != i) keel::swap(idx[i], idx[best]);
    }

    auto lambda_of = [&](usize r) noexcept -> f64 {
        const f64 mu = d_trid[r];
        return (keel::abs(mu) > mu_tol) ? (1.0 / mu) : Limits<f64>::infinity();
    };
    // Sort selected k eigenvalues ascending algebraically.
    for (usize i = 0; i + 1 < k; ++i) {
        usize best = i;
        for (usize j = i + 1; j < k; ++j) {
            if (lambda_of(idx[j]) < lambda_of(idx[best])) best = j;
        }
        if (best != i) keel::swap(idx[i], idx[best]);
    }

    for (usize col = 0; col < k; ++col) {
        const usize r = idx[col];
        const f64 mu = d_trid[r];
        D[col] = lambda_of(r);

        const f64* sr = S.data() + r * m_actual;
        for (usize i = 0; i < n; ++i) z_buf[i] = 0.0;
        for (usize j = 0; j < m_actual; ++j) {
            const f64 sj = sr[j];
            const f64* qj = Q.data() + j * n;
            for (usize i = 0; i < n; ++i) z_buf[i] += qj[i] * sj;
        }
        f64* vcol = V + col * n;
        solve_r(z_buf.data(), vcol);

        if (b_diag_all_pos && mu > mu_tol) {
            matvec_b(vcol, u_buf.data());
            f64 b_norm_sq = 0.0;
            for (usize i = 0; i < n; ++i) b_norm_sq += vcol[i] * u_buf[i];
            if (b_norm_sq > 0.0) {
                const f64 inv_norm = 1.0 / keel::sqrt(b_norm_sq);
                for (usize i = 0; i < n; ++i) vcol[i] *= inv_norm;
            }
        } else {
            usize i_max = 0;
            f64 max_abs = 0.0;
            for (usize i = 0; i < n; ++i) {
                const f64 a = keel::abs(vcol[i]);
                if (a > max_abs) {
                    max_abs = a;
                    i_max = i;
                }
            }
            if (max_abs > 0.0) {
                const f64 scale = vcol[i_max];
                for (usize i = 0; i < n; ++i) vcol[i] /= scale;
            }
        }
    }
    return true;
}

void hess_reduce(f64* H, usize n, f64* P, Allocator* alloc) noexcept {
    hess_reduce_impl(H, n, P, alloc);
}

void hess_reduce(c64* H, usize n, c64* P, Allocator* alloc) noexcept {
    hess_reduce_impl(H, n, P, alloc);
}

void schur_real(const f64* A, usize n, f64* U, f64* T, Allocator* alloc) noexcept {
    if (n == 0) return;
    keel::mem_copy(T, A, n * n * sizeof(f64));
    hess_reduce_impl(T, n, U, alloc);
    francis_schur(T, U, n);
    standardize_schur_2x2(T, U, n);
}

bool schur_complex(const c64* A, usize n, c64* U, c64* T, Allocator* alloc) noexcept {
    if (n == 0) return true;
    keel::mem_copy(T, A, n * n * sizeof(c64));
    hess_reduce_impl(T, n, U, alloc);
    return complex_schur_qr(T, U, n);
}

void schur_eigenvectors(const c64* U, const c64* T, usize n, c64* V, c64* evals, Allocator* alloc) noexcept {
    if (n == 0) return;

    f64 t_max = 0.0;
    for (usize i = 0; i < n * n; ++i) t_max = linalg_detail::linalg_max(t_max, keel::abs(T[i]));
    const f64 eps = Limits<f64>::kEpsilon;
    const f64 back_tol = 64.0 * static_cast<f64>(n) * eps * linalg_detail::linalg_max(1.0, t_max);
    const f64 smin = linalg_detail::linalg_max(eps * linalg_detail::linalg_max(1.0, t_max),
                                               Limits<f64>::kMinPositive * (static_cast<f64>(n) / eps));

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> Y(n * n, c64(0.0, 0.0), scratch.allocator());

    for (usize k = 0; k < n; ++k) {
        const c64 lam = T[k + k * n];
        Y[k + k * n] = c64(1.0, 0.0);
        f64 max_y = 1.0;
        for (isize i = static_cast<isize>(k) - 1; i >= 0; --i) {
            const usize ui = static_cast<usize>(i);
            c64 sum(0.0, 0.0);
            for (usize j = ui + 1; j <= k; ++j) sum += T[ui + j * n] * Y[j + k * n];
            c64 denom = T[ui + ui * n] - lam;
            if (keel::abs(denom) <= back_tol && keel::abs(sum) <= back_tol * max_y) {
                Y[ui + k * n] = c64(0.0, 0.0);
            } else {
                if (keel::abs(denom) < smin) denom = c64(smin, 0.0);
                Y[ui + k * n] = -sum / denom;
                max_y = linalg_detail::linalg_max(max_y, keel::abs(Y[ui + k * n]));
            }
        }
    }

    for (usize k = 0; k < n; ++k) {
        evals[k] = T[k + k * n];
        f64 norm_sq = 0.0;
        for (usize i = 0; i < n; ++i) {
            c64 s(0.0, 0.0);
            for (usize j = 0; j <= k; ++j) s += U[i + j * n] * Y[j + k * n];
            V[i + k * n] = s;
            norm_sq += keel::norm(s);
        }
        const f64 norm = keel::sqrt(norm_sq);
        if (norm > 0.0) {
            for (usize i = 0; i < n; ++i) V[i + k * n] /= norm;
        }
    }
}

void ordschur_complex(c64* U, c64* T, usize n, const u8* select, Allocator* alloc) noexcept {
    if (n <= 1) return;
    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> sel(n, u8{0}, scratch.allocator());
    for (usize i = 0; i < n; ++i) sel[i] = (select[i] != 0) ? u8{1} : u8{0};

    for (usize i = 0; i < n; ++i) {
        for (isize j = static_cast<isize>(n) - 2; j >= static_cast<isize>(i); --j) {
            const usize uj = static_cast<usize>(j);
            if (sel[uj + 1] != 0 && sel[uj] == 0) {
                swap_adjacent_schur(T, U, n, uj);
                keel::swap(sel[uj], sel[uj + 1]);
            }
        }
    }
}

void rsf2csf(const f64* UR, const f64* TR, usize n, c64* UC, c64* TC) noexcept {
    if (n == 0) return;
    constexpr f64 kPairTol = 1e-10;

    for (usize i = 0; i < n * n; ++i) {
        TC[i] = c64(TR[i], 0.0);
        UC[i] = c64(UR[i], 0.0);
    }

    usize i = 0;
    while (i + 1 < n) {
        const c64 sub = TC[(i + 1) + i * n];
        if (keel::abs(sub) <= kPairTol) {
            ++i;
            continue;
        }
        const c64 a = TC[i + i * n];
        const c64 ap = TC[(i + 1) + (i + 1) * n];
        const c64 c = TC[i + (i + 1) * n];
        const c64 d = sub;
        const c64 disc = keel::sqrt((a - ap) * (a - ap) + c64(4.0, 0.0) * c * d);
        const c64 lam1 = (a + ap + disc) * c64(0.5, 0.0);

        c64 x0 = c;
        c64 x1 = lam1 - a;
        if (keel::abs(x0) < keel::abs(x1) * 1e-300) {
            x0 = lam1 - ap;
            x1 = d;
        }
        const f64 nrm = keel::sqrt(keel::norm(x0) + keel::norm(x1));
        if (nrm == 0.0) {
            ++i;
            continue;
        }
        x0 /= nrm;
        x1 /= nrm;
        const c64 y0 = -keel::conj(x1);
        const c64 y1 = keel::conj(x0);

        for (usize r = 0; r < n; ++r) {
            const c64 t0 = TC[r + i * n];
            const c64 t1 = TC[r + (i + 1) * n];
            TC[r + i * n] = t0 * x0 + t1 * x1;
            TC[r + (i + 1) * n] = t0 * y0 + t1 * y1;
        }
        for (usize k = 0; k < n; ++k) {
            const c64 t0 = TC[i + k * n];
            const c64 t1 = TC[(i + 1) + k * n];
            TC[i + k * n] = keel::conj(x0) * t0 + keel::conj(x1) * t1;
            TC[(i + 1) + k * n] = keel::conj(y0) * t0 + keel::conj(y1) * t1;
        }
        for (usize r = 0; r < n; ++r) {
            const c64 u0 = UC[r + i * n];
            const c64 u1 = UC[r + (i + 1) * n];
            UC[r + i * n] = u0 * x0 + u1 * x1;
            UC[r + (i + 1) * n] = u0 * y0 + u1 * y1;
        }
        TC[(i + 1) + i * n] = c64(0.0, 0.0);
        i += 2;
    }
}

bool cdf2rdf(const c64* V, const c64* D, usize n, f64* VR, f64* DR) noexcept {
    constexpr f64 kPairTol = 1e-10;
    for (usize i = 0; i < n * n; ++i) {
        VR[i] = 0.0;
        DR[i] = 0.0;
    }

    usize i = 0;
    while (i < n) {
        const c64 lam = D[i + i * n];
        if (keel::abs(lam.imag()) <= kPairTol * (1.0 + keel::abs(lam.real()))) {
            DR[i + i * n] = lam.real();
            for (usize r = 0; r < n; ++r) VR[r + i * n] = V[r + i * n].real();
            ++i;
            continue;
        }
        if (i + 1 >= n) return false;
        const c64 lam2 = D[(i + 1) + (i + 1) * n];
        if (!is_approx_conj(lam, lam2, kPairTol)) return false;

        const f64 a = lam.real();
        const f64 b = lam.imag();
        const f64 bb = (b >= 0.0) ? b : -b;
        DR[i + i * n] = a;
        DR[i + (i + 1) * n] = bb;
        DR[(i + 1) + i * n] = -bb;
        DR[(i + 1) + (i + 1) * n] = a;
        for (usize r = 0; r < n; ++r) {
            const c64 v = V[r + i * n];
            VR[r + i * n] = v.real();
            VR[r + (i + 1) * n] = (b >= 0.0) ? v.imag() : -v.imag();
        }
        i += 2;
    }
    return true;
}

} // namespace keel::linalg
