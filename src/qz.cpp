#include "keel/arena.h"
#include "keel/array.h"
#include "keel/complex.h"
#include "keel/core.h"
#include "keel/linalg.h"
#include "keel/math.h"
#include "linalg_internal.h"

namespace keel::linalg {

namespace {

void make_givens_left(c64 x, c64 y, f64& c, c64& s) noexcept {
    const f64 norm_y = keel::abs(y);
    if (norm_y == 0.0) {
        c = 1.0;
        s = c64(0.0, 0.0);
        return;
    }
    const f64 norm_x = keel::abs(x);
    const f64 r = keel::hypot(norm_x, norm_y);
    c = norm_x / r;
    const c64 phase = (norm_x > 0.0) ? (x / norm_x) : c64(1.0, 0.0);
    s = phase * keel::conj(y) / r;
}

void make_givens_right(c64 x, c64 y, f64& c, c64& s) noexcept {
    const f64 norm_x = keel::abs(x);
    if (norm_x == 0.0) {
        c = 1.0;
        s = c64(0.0, 0.0);
        return;
    }
    const f64 norm_y = keel::abs(y);
    const f64 r = keel::hypot(norm_x, norm_y);
    c = norm_y / r;
    const c64 phase = (norm_y > 0.0) ? (y / norm_y) : c64(1.0, 0.0);
    s = -phase * keel::conj(x) / r;
}

void apply_givens_left(c64* M, usize n, usize row1, usize row2, f64 c, c64 s, usize col_start = 0) noexcept {
    const c64 g00(c, 0.0);
    const c64 g01 = s;
    const c64 g10 = -keel::conj(s);
    const c64 g11(c, 0.0);

    for (usize k = col_start; k < n; ++k) {
        const c64 r0 = M[row1 + k * n];
        const c64 r1 = M[row2 + k * n];
        M[row1 + k * n] = g00 * r0 + g01 * r1;
        M[row2 + k * n] = g10 * r0 + g11 * r1;
    }
}

void apply_givens_right(c64* M, usize n, usize col1, usize col2, f64 c, c64 s, usize row_end = 0) noexcept {
    if (row_end == 0) row_end = n;
    const c64 gh00(c, 0.0);
    const c64 gh01 = -s;
    const c64 gh10 = keel::conj(s);
    const c64 gh11(c, 0.0);

    for (usize i = 0; i < row_end; ++i) {
        const c64 c0 = M[i + col1 * n];
        const c64 c1 = M[i + col2 * n];
        M[i + col1 * n] = c0 * gh00 + c1 * gh10;
        M[i + col2 * n] = c0 * gh01 + c1 * gh11;
    }
}

void swap_adjacent_pencil(c64* AA, c64* BB, c64* Q, c64* Z, usize n, usize j) noexcept {
    const c64 a11 = AA[j + j * n];
    const c64 a22 = AA[j + 1 + (j + 1) * n];
    const c64 a12 = AA[j + (j + 1) * n];

    const c64 b11 = BB[j + j * n];
    const c64 b22 = BB[j + 1 + (j + 1) * n];
    const c64 b12 = BB[j + (j + 1) * n];

    // Right null vector of (b22 * A - a22 * B) satisfies v1 * z0 + v2 * z1 = 0, so [z0; z1] ~ [v2; -v1].
    const c64 v1 = a11 * b22 - a22 * b11;
    const c64 v2 = a12 * b22 - b12 * a22;
    const f64 norm_v1 = keel::abs(v1);
    const f64 norm_v2 = keel::abs(v2);
    const f64 r_z = keel::hypot(norm_v2, norm_v1);
    if (r_z == 0.0) return;

    const f64 c_z = norm_v2 / r_z;
    const c64 phase_z = (norm_v2 > 0.0) ? (v2 / norm_v2) : c64(1.0, 0.0);
    const c64 s_z = phase_z * keel::conj(v1) / r_z;

    const c64 z00(c_z, 0.0);
    const c64 z01 = s_z;
    const c64 z10 = -keel::conj(s_z);
    const c64 z11(c_z, 0.0);

    for (usize i = 0; i < n; ++i) {
        const c64 c0a = AA[i + j * n];
        const c64 c1a = AA[i + (j + 1) * n];
        AA[i + j * n] = c0a * z00 + c1a * z10;
        AA[i + (j + 1) * n] = c0a * z01 + c1a * z11;

        const c64 c0b = BB[i + j * n];
        const c64 c1b = BB[i + (j + 1) * n];
        BB[i + j * n] = c0b * z00 + c1b * z10;
        BB[i + (j + 1) * n] = c0b * z01 + c1b * z11;

        const c64 c0z = Z[i + j * n];
        const c64 c1z = Z[i + (j + 1) * n];
        Z[i + j * n] = c0z * z00 + c1z * z10;
        Z[i + (j + 1) * n] = c0z * z01 + c1z * z11;
    }

    // Left unitary rotation to eliminate the parallel subdiagonal entries AA(j+1,j) & BB(j+1,j).
    const f64 a_col_norm = keel::abs(AA[j + j * n]) + keel::abs(AA[j + 1 + j * n]);
    const f64 b_col_norm = keel::abs(BB[j + j * n]) + keel::abs(BB[j + 1 + j * n]);
    f64 c_q = 1.0;
    c64 s_q(0.0, 0.0);
    if (b_col_norm >= a_col_norm) {
        make_givens_left(BB[j + j * n], BB[j + 1 + j * n], c_q, s_q);
    } else {
        make_givens_left(AA[j + j * n], AA[j + 1 + j * n], c_q, s_q);
    }

    const c64 q00(c_q, 0.0);
    const c64 q01 = s_q;
    const c64 q10 = -keel::conj(s_q);
    const c64 q11(c_q, 0.0);

    for (usize k = 0; k < n; ++k) {
        const c64 r0a = AA[j + k * n];
        const c64 r1a = AA[j + 1 + k * n];
        AA[j + k * n] = q00 * r0a + q01 * r1a;
        AA[j + 1 + k * n] = q10 * r0a + q11 * r1a;

        const c64 r0b = BB[j + k * n];
        const c64 r1b = BB[j + 1 + k * n];
        BB[j + k * n] = q00 * r0b + q01 * r1b;
        BB[j + 1 + k * n] = q10 * r0b + q11 * r1b;

        const c64 r0q = Q[j + k * n];
        const c64 r1q = Q[j + 1 + k * n];
        Q[j + k * n] = q00 * r0q + q01 * r1q;
        Q[j + 1 + k * n] = q10 * r0q + q11 * r1q;
    }
    AA[j + 1 + j * n] = c64(0.0, 0.0);
    BB[j + 1 + j * n] = c64(0.0, 0.0);
}

} // namespace

bool qz_factor(const c64* A, const c64* B, usize n, c64* AA, c64* BB, c64* Q, c64* Z) noexcept {
    if (n == 0) return true;

    keel::mem_copy(AA, A, n * n * sizeof(c64));
    keel::mem_copy(BB, B, n * n * sizeof(c64));
    for (usize i = 0; i < n * n; ++i) {
        Q[i] = c64(0.0, 0.0);
        Z[i] = c64(0.0, 0.0);
    }
    for (usize i = 0; i < n; ++i) {
        Q[i + i * n] = c64(1.0, 0.0);
        Z[i + i * n] = c64(1.0, 0.0);
    }

    // Step 1: Reduce BB to upper-triangular via left Givens rotations.
    for (usize k = 0; k < n; ++k) {
        for (isize i = static_cast<isize>(n) - 1; i > static_cast<isize>(k); --i) {
            const usize ui = static_cast<usize>(i);
            if (BB[ui + k * n] != c64(0.0, 0.0)) {
                f64 c = 0.0;
                c64 s(0.0, 0.0);
                make_givens_left(BB[ui - 1 + k * n], BB[ui + k * n], c, s);
                apply_givens_left(BB, n, ui - 1, ui, c, s, k);
                apply_givens_left(AA, n, ui - 1, ui, c, s, 0);
                apply_givens_left(Q, n, ui - 1, ui, c, s, 0);
                BB[ui + k * n] = c64(0.0, 0.0);
            }
        }
    }

    // Step 2: Reduce AA to upper Hessenberg while preserving upper-triangular BB.
    for (usize j = 0; j < n; ++j) {
        for (isize i = static_cast<isize>(n) - 1; i >= static_cast<isize>(j) + 2; --i) {
            const usize ui = static_cast<usize>(i);
            if (AA[ui + j * n] != c64(0.0, 0.0)) {
                f64 c = 0.0;
                c64 s(0.0, 0.0);
                make_givens_left(AA[ui - 1 + j * n], AA[ui + j * n], c, s);
                apply_givens_left(AA, n, ui - 1, ui, c, s, j);
                apply_givens_left(BB, n, ui - 1, ui, c, s, ui - 1);
                apply_givens_left(Q, n, ui - 1, ui, c, s, 0);
                AA[ui + j * n] = c64(0.0, 0.0);

                if (BB[ui + (ui - 1) * n] != c64(0.0, 0.0)) {
                    f64 cr = 0.0;
                    c64 sr(0.0, 0.0);
                    make_givens_right(BB[ui + (ui - 1) * n], BB[ui + ui * n], cr, sr);
                    apply_givens_right(BB, n, ui - 1, ui, cr, sr, ui + 1);
                    apply_givens_right(AA, n, ui - 1, ui, cr, sr, n);
                    apply_givens_right(Z, n, ui - 1, ui, cr, sr, n);
                    BB[ui + (ui - 1) * n] = c64(0.0, 0.0);
                }
            }
        }
    }

    // Step 3: Moler-Stewart single-shift complex QZ bulge-chasing with zero-B deflation.
    if (n > 1) {
        f64 norm_a = 0.0;
        f64 norm_b = 0.0;
        for (usize i = 0; i < n * n; ++i) {
            norm_a = keel::hypot(norm_a, keel::abs(AA[i]));
            norm_b = keel::hypot(norm_b, keel::abs(BB[i]));
        }
        const f64 eps = Limits<f64>::kEpsilon;
        const f64 a_zero_tol = 4.0 * eps * linalg_detail::linalg_max(1.0, norm_a);
        const f64 b_zero_tol = 4.0 * eps * linalg_detail::linalg_max(1.0, norm_b);

        usize ihi = n - 1;
        usize iter = 0;
        usize local_iter = 0;
        const usize max_iter = 200 * n;

        while (ihi > 0 && iter < max_iter) {
            while (ihi > 0) {
                const f64 diag = keel::abs(AA[(ihi - 1) + (ihi - 1) * n]) + keel::abs(AA[ihi + ihi * n]);
                const f64 thresh = linalg_detail::linalg_max(a_zero_tol, 4.0 * eps * diag);
                if (keel::abs(AA[ihi + (ihi - 1) * n]) <= thresh) {
                    AA[ihi + (ihi - 1) * n] = c64(0.0, 0.0);
                    --ihi;
                    local_iter = 0;
                } else {
                    break;
                }
            }
            if (ihi == 0) break;

            usize ilo = ihi;
            while (ilo > 0) {
                const f64 diag = keel::abs(AA[(ilo - 1) + (ilo - 1) * n]) + keel::abs(AA[ilo + ilo * n]);
                const f64 thresh = linalg_detail::linalg_max(a_zero_tol, 4.0 * eps * diag);
                if (keel::abs(AA[ilo + (ilo - 1) * n]) <= thresh) {
                    AA[ilo + (ilo - 1) * n] = c64(0.0, 0.0);
                    break;
                }
                --ilo;
            }

            bool deflated_zero_b = false;
            for (usize p = ihi + 1; p > ilo; --p) {
                const usize idx = p - 1;
                if (keel::abs(BB[idx + idx * n]) <= b_zero_tol) {
                    BB[idx + idx * n] = c64(0.0, 0.0);
                    for (usize j = idx; j < ihi; ++j) {
                        f64 c = 0.0;
                        c64 s(0.0, 0.0);
                        make_givens_left(BB[j + (j + 1) * n], BB[(j + 1) + (j + 1) * n], c, s);
                        apply_givens_left(BB, n, j, j + 1, c, s, j + 1);
                        apply_givens_left(AA, n, j, j + 1, c, s, (j > ilo ? j - 1 : ilo));
                        apply_givens_left(Q, n, j, j + 1, c, s, 0);
                        BB[(j + 1) + (j + 1) * n] = c64(0.0, 0.0);
                        if (j > ilo) {
                            f64 cr = 0.0;
                            c64 sr(0.0, 0.0);
                            make_givens_right(AA[(j + 1) + (j - 1) * n], AA[(j + 1) + j * n], cr, sr);
                            apply_givens_right(AA, n, j - 1, j, cr, sr, j + 2);
                            apply_givens_right(BB, n, j - 1, j, cr, sr, j);
                            apply_givens_right(Z, n, j - 1, j, cr, sr, n);
                            AA[(j + 1) + (j - 1) * n] = c64(0.0, 0.0);
                        }
                    }
                    f64 cr = 0.0;
                    c64 sr(0.0, 0.0);
                    make_givens_right(AA[ihi + (ihi - 1) * n], AA[ihi + ihi * n], cr, sr);
                    apply_givens_right(AA, n, ihi - 1, ihi, cr, sr, ihi + 1);
                    apply_givens_right(BB, n, ihi - 1, ihi, cr, sr, ihi);
                    apply_givens_right(Z, n, ihi - 1, ihi, cr, sr, n);
                    AA[ihi + (ihi - 1) * n] = c64(0.0, 0.0);
                    BB[ihi + ihi * n] = c64(0.0, 0.0);
                    --ihi;
                    local_iter = 0;
                    deflated_zero_b = true;
                    break;
                }
            }
            if (deflated_zero_b) continue;

            ++iter;
            ++local_iter;
            const usize m1 = ihi - 1;
            const c64 a11 = AA[m1 + m1 * n] / BB[m1 + m1 * n];
            const c64 a21 = AA[ihi + m1 * n] / BB[m1 + m1 * n];
            const c64 a12 = (AA[m1 + ihi * n] - a11 * BB[m1 + ihi * n]) / BB[ihi + ihi * n];
            const c64 a22 = (AA[ihi + ihi * n] - a21 * BB[m1 + ihi * n]) / BB[ihi + ihi * n];

            c64 shift(0.0, 0.0);
            if (local_iter % 15 == 0) {
                const f64 mag = keel::abs(a22) + keel::abs(a21) + 1.0;
                const f64 ang = 0.7 * static_cast<f64>(local_iter / 15);
                shift = a22 + keel::polar(0.75 * mag, ang);
            } else {
                const c64 half_diff = 0.5 * (a11 - a22);
                const c64 disc = keel::sqrt(half_diff * half_diff + a12 * a21);
                const c64 r1 = a22 + half_diff + disc;
                const c64 r2 = a22 + half_diff - disc;
                shift = (keel::abs(r1 - a22) < keel::abs(r2 - a22)) ? r1 : r2;
            }

            // Introduce bulge at ilo.
            {
                const c64 x = AA[ilo + ilo * n] - shift * BB[ilo + ilo * n];
                const c64 y = AA[(ilo + 1) + ilo * n];
                f64 c = 0.0;
                c64 s(0.0, 0.0);
                make_givens_left(x, y, c, s);
                apply_givens_left(AA, n, ilo, ilo + 1, c, s, ilo);
                apply_givens_left(BB, n, ilo, ilo + 1, c, s, ilo);
                apply_givens_left(Q, n, ilo, ilo + 1, c, s, 0);

                f64 cr = 0.0;
                c64 sr(0.0, 0.0);
                make_givens_right(BB[(ilo + 1) + ilo * n], BB[(ilo + 1) + (ilo + 1) * n], cr, sr);
                apply_givens_right(BB, n, ilo, ilo + 1, cr, sr, ilo + 2);
                apply_givens_right(AA, n, ilo, ilo + 1, cr, sr, linalg_detail::linalg_min(n, ilo + 3));
                apply_givens_right(Z, n, ilo, ilo + 1, cr, sr, n);
                BB[(ilo + 1) + ilo * n] = c64(0.0, 0.0);
            }

            // Chase bulge down to ihi.
            for (usize k = ilo + 1; k < ihi; ++k) {
                f64 c = 0.0;
                c64 s(0.0, 0.0);
                make_givens_left(AA[k + (k - 1) * n], AA[(k + 1) + (k - 1) * n], c, s);
                apply_givens_left(AA, n, k, k + 1, c, s, k - 1);
                apply_givens_left(BB, n, k, k + 1, c, s, k);
                apply_givens_left(Q, n, k, k + 1, c, s, 0);
                AA[(k + 1) + (k - 1) * n] = c64(0.0, 0.0);

                f64 cr = 0.0;
                c64 sr(0.0, 0.0);
                make_givens_right(BB[(k + 1) + k * n], BB[(k + 1) + (k + 1) * n], cr, sr);
                apply_givens_right(BB, n, k, k + 1, cr, sr, k + 2);
                apply_givens_right(AA, n, k, k + 1, cr, sr, linalg_detail::linalg_min(n, k + 3));
                apply_givens_right(Z, n, k, k + 1, cr, sr, n);
                BB[(k + 1) + k * n] = c64(0.0, 0.0);
            }
        }

        if (ihi > 0) return false;
    }

    for (usize i = 0; i < n; ++i) {
        for (usize j = 0; j < i; ++j) {
            AA[i + j * n] = c64(0.0, 0.0);
            BB[i + j * n] = c64(0.0, 0.0);
        }
    }
    return true;
}

void ordqz_complex(c64* AA, c64* BB, c64* Q, c64* Z, usize n, const u8* select, Allocator* alloc) noexcept {
    if (n <= 1) return;
    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> sel(n, u8{0}, scratch.allocator());
    for (usize i = 0; i < n; ++i) sel[i] = (select[i] != 0) ? u8{1} : u8{0};

    usize target = 0;
    for (usize i = 0; i < n; ++i) {
        if (sel[i] != 0) {
            for (usize j = i; j > target; --j) {
                swap_adjacent_pencil(AA, BB, Q, Z, n, j - 1);
                keel::swap(sel[j], sel[j - 1]);
            }
            ++target;
        }
    }
}

void qz_eigenvectors(const c64* AA, const c64* BB, const c64* Z, usize n, c64* V, c64* evals,
                     Allocator* alloc) noexcept {
    if (n == 0) return;

    f64 a_max = 0.0;
    f64 b_max = 0.0;
    for (usize i = 0; i < n * n; ++i) {
        a_max = linalg_detail::linalg_max(a_max, keel::abs(AA[i]));
        b_max = linalg_detail::linalg_max(b_max, keel::abs(BB[i]));
    }
    const f64 eps = Limits<f64>::kEpsilon;
    const f64 ab_scale = linalg_detail::linalg_max(1.0, linalg_detail::linalg_max(a_max, b_max));
    const f64 back_tol = 64.0 * static_cast<f64>(n) * eps * ab_scale;
    const f64 smin = linalg_detail::linalg_max(eps * ab_scale, Limits<f64>::kMinPositive * (static_cast<f64>(n) / eps));
    const f64 b_zero_tol = 64.0 * static_cast<f64>(n) * eps * linalg_detail::linalg_max(1.0, b_max);
    const f64 a_zero_tol = 64.0 * static_cast<f64>(n) * eps * linalg_detail::linalg_max(1.0, a_max);

    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> Y(n * n, c64(0.0, 0.0), scratch.allocator());

    for (usize k = 0; k < n; ++k) {
        c64 alpha = AA[k + k * n];
        c64 beta = BB[k + k * n];
        const f64 pair_scale = linalg_detail::linalg_max(keel::abs(alpha), keel::abs(beta));
        if (pair_scale > 0.0) {
            alpha /= pair_scale;
            beta /= pair_scale;
        }
        Y[k + k * n] = c64(1.0, 0.0);
        f64 max_y = 1.0;
        for (isize i = static_cast<isize>(k) - 1; i >= 0; --i) {
            const usize ui = static_cast<usize>(i);
            c64 sum(0.0, 0.0);
            for (usize j = ui + 1; j <= k; ++j) {
                const c64 mij = beta * AA[ui + j * n] - alpha * BB[ui + j * n];
                sum += mij * Y[j + k * n];
            }
            c64 denom = beta * AA[ui + ui * n] - alpha * BB[ui + ui * n];
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
        const c64 alpha = AA[k + k * n];
        const c64 beta = BB[k + k * n];
        if (keel::abs(beta) > b_zero_tol) {
            evals[k] = alpha / beta;
        } else if (keel::abs(alpha) > a_zero_tol) {
            evals[k] = c64(Limits<f64>::infinity(), 0.0);
        } else {
            evals[k] = c64(Limits<f64>::quiet_nan(), 0.0);
        }

        f64 norm_sq = 0.0;
        for (usize i = 0; i < n; ++i) {
            c64 s(0.0, 0.0);
            for (usize j = 0; j <= k; ++j) s += Z[i + j * n] * Y[j + k * n];
            V[i + k * n] = s;
            norm_sq += keel::norm(s);
        }
        const f64 norm = keel::sqrt(norm_sq);
        if (norm > 0.0) {
            for (usize i = 0; i < n; ++i) V[i + k * n] /= norm;
        }
    }
}

} // namespace keel::linalg
