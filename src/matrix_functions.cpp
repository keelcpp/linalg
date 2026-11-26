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
        for (usize i = 0; i < n; ++i) {
            const f64 v = keel::abs(A[i + j * n]);
            if (!keel::is_finite(v)) return v;
            s += v;
        }
        if (s > mx) mx = s;
    }
    return mx;
}

template <typename T>
inline void square_gemm(const T* A, const T* B, T* C, usize n) noexcept {
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans, n, n, n, T(1), A, n, B, n, T(0), C, n);
}

// Higham (2005) scaling-and-squaring with optimal Padé approximants of degree 3, 5, 7, 9, 13.
template <typename T>
bool expm_impl(const T* A, usize n, T* X, Allocator* alloc) noexcept {
    if (n == 0) return true;

    const f64 a_norm = mat_1norm(A, n);
    if (!keel::is_finite(a_norm)) return false;

    static constexpr f64 kTheta3 = 1.495585217958292e-2;
    static constexpr f64 kTheta5 = 2.539398330063230e-1;
    static constexpr f64 kTheta7 = 9.504178996162932e-1;
    static constexpr f64 kTheta9 = 2.097847961257068e0;
    static constexpr f64 kTheta13 = 5.371920351148152e0;

    static constexpr f64 b3[4] = {120.0, 60.0, 12.0, 1.0};
    static constexpr f64 b5[6] = {30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0};
    static constexpr f64 b7[8] = {17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0};
    static constexpr f64 b9[10] = {
        17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
        2162160.0,     110880.0,     3960.0,       90.0,        1.0,
    };
    static constexpr f64 b13[14] = {
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0, 129060195264000.0,
        10559470521600.0,    670442572800.0,      33522128640.0,      1323241920.0,       40840800.0,
        960960.0,            16380.0,             182.0,              1.0,
    };

    const usize nn = n * n;
    ArenaTemp scratch = get_scratch(alloc);
    Array<T> A_s(nn, T(0), scratch.allocator());
    keel::mem_copy(A_s.data(), A, nn * sizeof(T));

    i32 s = 0;
    if (a_norm > kTheta13) {
        const f64 lg = keel::log2(a_norm / kTheta13);
        if (!keel::is_finite(lg) || lg > 1000.0) return false;
        s = keel::ceil_i32(lg);
        if (s < 0) s = 0;
        const f64 scale = keel::ldexp(1.0, -s);
        for (usize i = 0; i < nn; ++i) A_s[i] *= scale;
    }

    Array<T> U(nn, T(0), scratch.allocator());
    Array<T> V(nn, T(0), scratch.allocator());
    Array<T> A2(nn, T(0), scratch.allocator());
    square_gemm(A_s.data(), A_s.data(), A2.data(), n);

    if (s == 0 && a_norm <= kTheta9) {
        const f64* b = nullptr;
        i32 deg = 9;
        if (a_norm <= kTheta3) {
            b = b3;
            deg = 3;
        } else if (a_norm <= kTheta5) {
            b = b5;
            deg = 5;
        } else if (a_norm <= kTheta7) {
            b = b7;
            deg = 7;
        } else {
            b = b9;
            deg = 9;
        }

        Array<T> W(nn, T(0), scratch.allocator());
        for (usize i = 0; i < n; ++i) {
            W[i + i * n] = T(b[1]);
            V[i + i * n] = T(b[0]);
        }
        for (usize i = 0; i < nn; ++i) {
            W[i] += b[3] * A2[i];
            V[i] += b[2] * A2[i];
        }
        if (deg >= 5) {
            Array<T> A4(nn, T(0), scratch.allocator());
            square_gemm(A2.data(), A2.data(), A4.data(), n);
            for (usize i = 0; i < nn; ++i) {
                W[i] += b[5] * A4[i];
                V[i] += b[4] * A4[i];
            }
            if (deg >= 7) {
                Array<T> A6(nn, T(0), scratch.allocator());
                square_gemm(A2.data(), A4.data(), A6.data(), n);
                for (usize i = 0; i < nn; ++i) {
                    W[i] += b[7] * A6[i];
                    V[i] += b[6] * A6[i];
                }
                if (deg >= 9) {
                    Array<T> A8(nn, T(0), scratch.allocator());
                    square_gemm(A4.data(), A4.data(), A8.data(), n);
                    for (usize i = 0; i < nn; ++i) {
                        W[i] += b[9] * A8[i];
                        V[i] += b[8] * A8[i];
                    }
                }
            }
        }
        square_gemm(A_s.data(), W.data(), U.data(), n);
    } else {
        // Padé [13/13] using 4 extra GEMMs: A4, A6, A6*(...), A*(...).
        Array<T> A4(nn, T(0), scratch.allocator());
        Array<T> A6(nn, T(0), scratch.allocator());
        Array<T> T1(nn, T(0), scratch.allocator());
        Array<T> T2(nn, T(0), scratch.allocator());
        square_gemm(A2.data(), A2.data(), A4.data(), n);
        square_gemm(A2.data(), A4.data(), A6.data(), n);

        // W_u = A6 * (b13*A6 + b11*A4 + b9*A2) + b7*A6 + b5*A4 + b3*A2 + b1*I
        for (usize i = 0; i < nn; ++i) {
            T1[i] = b13[13] * A6[i] + b13[11] * A4[i] + b13[9] * A2[i];
        }
        square_gemm(A6.data(), T1.data(), T2.data(), n);
        for (usize i = 0; i < nn; ++i) {
            T2[i] += b13[7] * A6[i] + b13[5] * A4[i] + b13[3] * A2[i];
        }
        for (usize i = 0; i < n; ++i) T2[i + i * n] += T(b13[1]);
        square_gemm(A_s.data(), T2.data(), U.data(), n);

        // V = A6 * (b12*A6 + b10*A4 + b8*A2) + b6*A6 + b4*A4 + b2*A2 + b0*I
        for (usize i = 0; i < nn; ++i) {
            T1[i] = b13[12] * A6[i] + b13[10] * A4[i] + b13[8] * A2[i];
        }
        square_gemm(A6.data(), T1.data(), V.data(), n);
        for (usize i = 0; i < nn; ++i) {
            V[i] += b13[6] * A6[i] + b13[4] * A4[i] + b13[2] * A2[i];
        }
        for (usize i = 0; i < n; ++i) V[i + i * n] += T(b13[0]);
    }

    // Solve (V - U) * X = (V + U)
    Array<T> P(nn, T(0), scratch.allocator());
    Array<T> Q(nn, T(0), scratch.allocator());
    for (usize i = 0; i < nn; ++i) {
        P[i] = V[i] + U[i];
        Q[i] = V[i] - U[i];
    }

    if (!la_solve(Q.data(), n, n, P.data(), n, X, scratch.allocator())) return false;

    if (s > 0) {
        Array<T> tmp(nn, T(0), scratch.allocator());
        for (i32 k = 0; k < s; ++k) {
            square_gemm(X, X, tmp.data(), n);
            keel::mem_copy(X, tmp.data(), nn * sizeof(T));
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

// Solves upper-triangular system (I + t * E) * Y = E where E is upper-triangular.
void solve_upper_ip_te(const c64* E, usize n, f64 t, c64* Y) noexcept {
    for (usize i = 0; i < n * n; ++i) Y[i] = c64(0.0, 0.0);
    for (usize j = 0; j < n; ++j) {
        for (isize i = as<isize>(j); i >= 0; --i) {
            const usize ui = as<usize>(i);
            c64 rhs = E[ui + j * n];
            for (usize k = ui + 1; k <= j; ++k) {
                rhs -= (t * E[ui + k * n]) * Y[k + j * n];
            }
            const c64 diag = c64(1.0, 0.0) + t * E[ui + ui * n];
            Y[ui + j * n] = rhs / diag;
        }
    }
}

// Evaluates f(T_blk) for a confluent/clustered upper-triangular block T_blk of size m x m
// using Cauchy contour integral Taylor coefficients around cluster mean sigma.
void eval_confluent_block(const c64* T_blk, usize m, FunctionRef<c64(c64)> f, c64* F_blk,
                          Allocator* scratch_alloc) noexcept {
    for (usize i = 0; i < m * m; ++i) F_blk[i] = c64(0.0, 0.0);
    if (m == 1) {
        F_blk[0] = f(T_blk[0]);
        return;
    }

    c64 sigma(0.0, 0.0);
    for (usize i = 0; i < m; ++i) sigma += T_blk[i + i * m];
    sigma /= as<f64>(m);

    f64 max_dev = 0.0;
    for (usize i = 0; i < m; ++i) {
        const f64 d = keel::abs(T_blk[i + i * m] - sigma);
        if (d > max_dev) max_dev = d;
    }

    // Choose safe analytic contour radius enclosing all block eigenvalues while staying away from origin singularities.
    const f64 abs_sig = keel::abs(sigma);
    f64 r = (abs_sig > 0.0) ? linalg_detail::linalg_min(0.05 * linalg_detail::linalg_max(1.0, abs_sig), 0.25 * abs_sig)
                            : 0.05;
    if (2.5 * max_dev > r) r = 2.5 * max_dev;
    if (r < 1e-4) r = 1e-4;

    constexpr usize kNumPts = 64;
    const usize max_order = linalg_detail::linalg_min(m + 10, usize{24});
    Array<c64> fvals(kNumPts, c64(0.0, 0.0), scratch_alloc);
    for (usize q = 0; q < kNumPts; ++q) {
        const f64 theta = (2.0 * kPi * as<f64>(q)) / as<f64>(kNumPts);
        const c64 z = sigma + r * c64(keel::cos(theta), keel::sin(theta));
        fvals[q] = f(z);
    }

    // M = T_blk - sigma * I (upper triangular with small diagonal)
    Array<c64> M(m * m, c64(0.0, 0.0), scratch_alloc);
    for (usize j = 0; j < m; ++j) {
        for (usize i = 0; i <= j; ++i) {
            M[i + j * m] = (i == j) ? (T_blk[i + j * m] - sigma) : T_blk[i + j * m];
        }
    }

    Array<c64> Mpow(m * m, c64(0.0, 0.0), scratch_alloc);
    Array<c64> Mnext(m * m, c64(0.0, 0.0), scratch_alloc);
    for (usize i = 0; i < m; ++i) Mpow[i + i * m] = c64(1.0, 0.0);

    f64 r_pow = 1.0;
    f64 f_norm = 0.0;
    for (usize p = 0; p <= max_order; ++p) {
        c64 cp(0.0, 0.0);
        for (usize q = 0; q < kNumPts; ++q) {
            const f64 ang = -(2.0 * kPi * as<f64>(q) * as<f64>(p)) / as<f64>(kNumPts);
            cp += fvals[q] * c64(keel::cos(ang), keel::sin(ang));
        }
        cp /= (as<f64>(kNumPts) * r_pow);

        f64 term_norm = 0.0;
        for (usize i = 0; i < m * m; ++i) {
            const c64 add = cp * Mpow[i];
            F_blk[i] += add;
            const f64 a = keel::abs(add);
            if (a > term_norm) term_norm = a;
            const f64 fn = keel::abs(F_blk[i]);
            if (fn > f_norm) f_norm = fn;
        }
        if (p >= m && term_norm <= 1e-15 * linalg_detail::linalg_max(1.0, f_norm)) break;

        if (p < max_order) {
            // Mnext = Mpow * M (both upper triangular)
            for (usize i = 0; i < m * m; ++i) Mnext[i] = c64(0.0, 0.0);
            for (usize j = 0; j < m; ++j) {
                for (usize i = 0; i <= j; ++i) {
                    c64 s(0.0, 0.0);
                    for (usize k = i; k <= j; ++k) s += Mpow[i + k * m] * M[k + j * m];
                    Mnext[i + j * m] = s;
                }
            }
            keel::mem_copy(Mpow.data(), Mnext.data(), m * m * sizeof(c64));
            r_pow *= r;
        }
    }

    // Exact diagonal when eigenvalues are distinct enough from sigma.
    for (usize i = 0; i < m; ++i) {
        F_blk[i + i * m] = f(T_blk[i + i * m]);
    }
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

    Array<c64> diag_orig(n, c64(0.0, 0.0), scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        diag_orig[i] = T[i + i * n];
        const f64 d = keel::abs(diag_orig[i]);
        if (d == 0.0 || !keel::is_finite(d)) return false;
    }

    i32 s_count = 0;
    Array<c64> T_next(n * n, c64(0.0, 0.0), scratch.allocator());
    while (s_count < 64 && mat_diff_i_1norm(T.data(), n) > 0.25) {
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

    // 7-point Gauss-Legendre quadrature on [0, 1] (equivalent to [7/7] Padé of log(I + E),
    // achieving < 2e-16 relative error for ||E||_1 <= 0.25).
    static constexpr f64 kGlNodes[7] = {
        0.025446043828620701,
        0.12923440720030277,
        0.29707742431130141,
        0.5,
        0.70292257568869854,
        0.87076559279969723,
        0.9745539561713793,
    };
    static constexpr f64 kGlWeights[7] = {
        0.064742483084434865,
        0.13985269574463843,
        0.19091502525255935,
        0.20897959183673465,
        0.19091502525255935,
        0.13985269574463843,
        0.064742483084434865,
    };

    Array<c64> log_t(n * n, c64(0.0, 0.0), scratch.allocator());
    Array<c64> Y(n * n, c64(0.0, 0.0), scratch.allocator());
    for (usize q = 0; q < 7; ++q) {
        solve_upper_ip_te(E.data(), n, kGlNodes[q], Y.data());
        const f64 w = kGlWeights[q];
        for (usize i = 0; i < n * n; ++i) log_t[i] += w * Y[i];
    }

    const f64 scale = keel::ldexp(1.0, s_count);
    for (usize i = 0; i < n * n; ++i) log_t[i] *= scale;
    // Restore exact principal-branch diagonal log(T_ii) to avoid 2^s rounding drift.
    for (usize i = 0; i < n; ++i) log_t[i + i * n] = keel::log(diag_orig[i]);

    schur_reconstruct(U.data(), log_t.data(), n, X, scratch.allocator());
    return true;
}

bool funm(const c64* A, usize n, FunctionRef<c64(c64)> f, c64* X, Allocator* alloc) noexcept {
    if (n == 0) return true;

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> U(n * n, c64(0.0, 0.0), scratch.allocator());
    Array<c64> T(n * n, c64(0.0, 0.0), scratch.allocator());
    if (!schur_complex(A, n, U.data(), T.data(), scratch.allocator())) return false;

    // Davies-Higham Schur-Parlett blocking: cluster eigenvalues within delta = 0.05.
    constexpr f64 kDelta = 0.05;
    Array<usize> cluster_id(n, usize{0}, scratch.allocator());
    for (usize i = 0; i < n; ++i) cluster_id[i] = i;

    auto find_root = [&](usize x) noexcept -> usize {
        while (cluster_id[x] != x) {
            cluster_id[x] = cluster_id[cluster_id[x]];
            x = cluster_id[x];
        }
        return x;
    };

    for (usize i = 0; i < n; ++i) {
        for (usize j = i + 1; j < n; ++j) {
            const f64 scale = linalg_detail::linalg_max(1.0, linalg_detail::linalg_max(keel::abs(T[i + i * n]),
                                                                                        keel::abs(T[j + j * n])));
            if (keel::abs(T[i + i * n] - T[j + j * n]) <= kDelta * scale) {
                const usize ri = find_root(i);
                const usize rj = find_root(j);
                if (ri != rj) cluster_id[rj] = ri;
            }
        }
    }

    // Relabel clusters densely 0..num_blocks-1.
    Array<usize> root_to_blk(n, n, scratch.allocator());
    Array<usize> blk_sizes(scratch.allocator());
    usize num_blocks = 0;
    for (usize i = 0; i < n; ++i) {
        const usize r = find_root(i);
        if (root_to_blk[r] == n) {
            root_to_blk[r] = num_blocks++;
            blk_sizes.push_back(0);
        }
        cluster_id[i] = root_to_blk[r];
        ++blk_sizes[cluster_id[i]];
    }

    // Reorder Schur form so clusters are contiguous: pull cluster c to front for c = num_blocks-1 down to 0.
    if (num_blocks > 1) {
        Array<u8> sel(n, u8{0}, scratch.allocator());
        Array<usize> cur_cid(n, usize{0}, scratch.allocator());
        for (usize i = 0; i < n; ++i) cur_cid[i] = cluster_id[i];
        for (isize c = as<isize>(num_blocks) - 1; c >= 0; --c) {
            const usize uc = as<usize>(c);
            for (usize i = 0; i < n; ++i) sel[i] = (cur_cid[i] == uc) ? u8{1} : u8{0};
            ordschur_complex(U.data(), T.data(), n, sel.data(), scratch.allocator());
            // Mirror exact bubble-reorder on cur_cid.
            for (usize i = 0; i < n; ++i) {
                for (isize j = as<isize>(n) - 2; j >= as<isize>(i); --j) {
                    const usize uj = as<usize>(j);
                    if (sel[uj + 1] != 0 && sel[uj] == 0) {
                        keel::swap(sel[uj], sel[uj + 1]);
                        keel::swap(cur_cid[uj], cur_cid[uj + 1]);
                    }
                }
            }
        }
    }

    Array<usize> blk_off(num_blocks + 1, usize{0}, scratch.allocator());
    for (usize b = 0; b < num_blocks; ++b) blk_off[b + 1] = blk_off[b] + blk_sizes[b];

    Array<c64> F(n * n, c64(0.0, 0.0), scratch.allocator());

    // 1. Evaluate each diagonal block F_{bb} = f(T_{bb}).
    for (usize b = 0; b < num_blocks; ++b) {
        const usize s0 = blk_off[b];
        const usize mb = blk_sizes[b];
        if (mb == 1) {
            F[s0 + s0 * n] = f(T[s0 + s0 * n]);
        } else {
            Array<c64> T_blk(mb * mb, c64(0.0, 0.0), scratch.allocator());
            Array<c64> F_blk(mb * mb, c64(0.0, 0.0), scratch.allocator());
            for (usize j = 0; j < mb; ++j) {
                for (usize i = 0; i <= j; ++i) {
                    T_blk[i + j * mb] = T[(s0 + i) + (s0 + j) * n];
                }
            }
            eval_confluent_block(T_blk.data(), mb, f, F_blk.data(), scratch.allocator());
            for (usize j = 0; j < mb; ++j) {
                for (usize i = 0; i <= j; ++i) {
                    F[(s0 + i) + (s0 + j) * n] = F_blk[i + j * mb];
                }
            }
        }
    }

    // 2. Block Parlett recurrence: solve T_{aa}*F_{ab} - F_{ab}*T_{bb} = R_{ab}.
    for (usize b = 1; b < num_blocks; ++b) {
        const usize sb = blk_off[b];
        const usize nb = blk_sizes[b];
        for (isize a = as<isize>(b) - 1; a >= 0; --a) {
            const usize ua = as<usize>(a);
            const usize sa = blk_off[ua];
            const usize na = blk_sizes[ua];

            // Compute R_{ab} = F_{aa}*T_{ab} - T_{ab}*F_{bb} + sum_{k=a+1}^{b-1} (F_{ak}*T_{kb} - T_{ak}*F_{kb})
            for (usize j = 0; j < nb; ++j) {
                for (isize i = as<isize>(na) - 1; i >= 0; --i) {
                    const usize ui = as<usize>(i);
                    const usize gi = sa + ui;
                    const usize gj = sb + j;
                    c64 rhs(0.0, 0.0);
                    // F_{aa} * T_{ab}
                    for (usize k = ui; k < na; ++k) rhs += F[gi + (sa + k) * n] * T[(sa + k) + gj * n];
                    // - T_{ab} * F_{bb}
                    for (usize k = 0; k <= j; ++k) rhs -= T[gi + (sb + k) * n] * F[(sb + k) + gj * n];
                    // Intermediate blocks k in (a, b)
                    for (usize k = sa + na; k < sb; ++k) {
                        rhs += F[gi + k * n] * T[k + gj * n] - T[gi + k * n] * F[k + gj * n];
                    }
                    // Triangular Sylvester back-sub terms within (a, b)
                    for (usize k = ui + 1; k < na; ++k) rhs -= T[gi + (sa + k) * n] * F[(sa + k) + gj * n];
                    for (usize k = 0; k < j; ++k) rhs += F[gi + (sb + k) * n] * T[(sb + k) + gj * n];

                    const c64 denom = T[gi + gi * n] - T[gj + gj * n];
                    F[gi + gj * n] = rhs / denom;
                }
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

    f64 scale = 1.0;
    for (usize i = 0; i < m; ++i) scale = linalg_detail::linalg_max(scale, keel::abs(Ta[i + i * m]));
    for (usize j = 0; j < n; ++j) scale = linalg_detail::linalg_max(scale, keel::abs(Tb[j + j * n]));
    const f64 sing_tol = 1e-14 * scale;

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
            if (keel::abs(denom) <= sing_tol) return false;
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
    if (!keel::is_finite(t)) return false;

    if (t == 0.0) {
        keel::mem_copy(out, v, n * sizeof(f64));
        return true;
    }

    const f64 a_norm = mat_1norm(A, n);
    if (!keel::is_finite(a_norm)) return false;

    const f64 beta0 = nrm2(n, v);
    if (!keel::is_finite(beta0)) return false;
    if (beta0 == 0.0) {
        for (usize i = 0; i < n; ++i) out[i] = 0.0;
        return true;
    }

    // Step-splitting keeps ||dt * A||_1 <= 2.0 so m <= 30 Arnoldi converges to machine precision
    // even for large |t| * ||A||.
    const f64 ta_norm = keel::abs(t) * a_norm;
    const usize m = linalg_detail::linalg_min(usize{30}, n);
    const usize num_steps = (m == n || ta_norm <= 2.0) ? usize{1} : as<usize>(keel::ceil(ta_norm / 2.0));
    const f64 dt = t / as<f64>(num_steps);

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> cur(n, 0.0, scratch.allocator());
    keel::mem_copy(cur.data(), v, n * sizeof(f64));

    Array<f64> V(n * (m + 1), 0.0, scratch.allocator());
    Array<f64> H(m * m, 0.0, scratch.allocator());
    Array<f64> work(n, 0.0, scratch.allocator());
    Array<f64> Hsq(m * m, 0.0, scratch.allocator());
    Array<f64> F(m * m, 0.0, scratch.allocator());

    for (usize step = 0; step < num_steps; ++step) {
        const f64 beta = nrm2(n, cur.data());
        if (beta == 0.0) {
            for (usize i = 0; i < n; ++i) out[i] = 0.0;
            return true;
        }
        if (!keel::is_finite(beta)) return false;

        for (usize i = 0; i < m * m; ++i) H[i] = 0.0;
        for (usize i = 0; i < n; ++i) V[i] = cur[i] / beta;

        usize m_eff = m;
        for (usize j = 0; j < m; ++j) {
            gemv(MatrixTranspose::NoTrans, n, n, 1.0, A, n, V.data() + j * n, 1, 0.0, work.data(), 1);

            // Double-pass modified Gram-Schmidt for full orthogonality.
            for (usize pass = 0; pass < 2; ++pass) {
                for (usize i = 0; i <= j; ++i) {
                    const f64 hij = dot(n, V.data() + i * n, work.data());
                    H[i + j * m] += hij;
                    axpy(n, -hij, V.data() + i * n, work.data());
                }
            }

            const f64 hjp1 = nrm2(n, work.data());
            if (hjp1 <= 1e-14 * linalg_detail::linalg_max(1.0, a_norm)) {
                m_eff = j + 1;
                break;
            }
            if (j + 1 < m) H[(j + 1) + j * m] = hjp1;
            const f64 inv_h = 1.0 / hjp1;
            for (usize k = 0; k < n; ++k) V[k + (j + 1) * n] = work[k] * inv_h;
        }

        for (usize j = 0; j < m_eff; ++j) {
            for (usize i = 0; i < m_eff; ++i) {
                Hsq[i + j * m_eff] = dt * H[i + j * m];
            }
        }

        if (!expm(Hsq.data(), m_eff, F.data(), scratch.allocator())) return false;
        gemv(MatrixTranspose::NoTrans, n, m_eff, beta, V.data(), n, F.data(), 1, 0.0, cur.data(), 1);
    }

    keel::mem_copy(out, cur.data(), n * sizeof(f64));
    return true;
}

} // namespace keel::linalg
