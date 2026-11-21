#include "keel/linalg.h"
#include "keel/math.h"
#include "linalg_internal.h"

namespace keel::linalg {

usize linalg_last_gemm_threads_used() noexcept {
    return linalg_detail::g_last_gemm_threads_used.load();
}

void axpy(usize n, c64 alpha, const c64* x, c64* y) noexcept {
    if (n == 0 || (alpha.real() == 0.0 && alpha.imag() == 0.0)) return;
    for (usize i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

f64 dot(usize n, const f64* x, const f64* y) noexcept {
    f64 s0 = 0.0;
    f64 s1 = 0.0;
    f64 s2 = 0.0;
    f64 s3 = 0.0;
    usize i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += x[i + 0] * y[i + 0];
        s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2];
        s3 += x[i + 3] * y[i + 3];
    }
    f64 sum = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

c64 dotc(usize n, const c64* x, const c64* y) noexcept {
    f64 re = 0.0;
    f64 im = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 xr = x[i].real();
        const f64 xi = x[i].imag();
        const f64 yr = y[i].real();
        const f64 yi = y[i].imag();
        re += xr * yr + xi * yi;
        im += xr * yi - xi * yr;
    }
    return c64(re, im);
}

c64 dotu(usize n, const c64* x, const c64* y) noexcept {
    f64 re = 0.0;
    f64 im = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 xr = x[i].real();
        const f64 xi = x[i].imag();
        const f64 yr = y[i].real();
        const f64 yi = y[i].imag();
        re += xr * yr - xi * yi;
        im += xr * yi + xi * yr;
    }
    return c64(re, im);
}

f64 nrm2(usize n, const f64* x) noexcept {
    if (n == 0) return 0.0;
    f64 scale = 0.0;
    f64 ssq = 1.0;
    for (usize i = 0; i < n; ++i) {
        const f64 ax = keel::abs(x[i]);
        if (ax != 0.0) {
            if (scale < ax) {
                const f64 r = scale / ax;
                ssq = 1.0 + ssq * r * r;
                scale = ax;
            } else {
                const f64 r = ax / scale;
                ssq += r * r;
            }
        }
    }
    return scale * keel::sqrt(ssq);
}

f64 nrm2(usize n, const c64* x) noexcept {
    if (n == 0) return 0.0;
    f64 scale = 0.0;
    f64 ssq = 1.0;
    auto accum = [&](f64 val) noexcept {
        const f64 ax = keel::abs(val);
        if (ax != 0.0) {
            if (scale < ax) {
                const f64 r = scale / ax;
                ssq = 1.0 + ssq * r * r;
                scale = ax;
            } else {
                const f64 r = ax / scale;
                ssq += r * r;
            }
        }
    };
    for (usize i = 0; i < n; ++i) {
        accum(x[i].real());
        accum(x[i].imag());
    }
    return scale * keel::sqrt(ssq);
}

void scal(usize n, f64 alpha, f64* x) noexcept {
    if (n == 0 || alpha == 1.0) return;
    if (alpha == 0.0) {
        for (usize i = 0; i < n; ++i) x[i] = 0.0;
        return;
    }
    for (usize i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

void scal(usize n, c64 alpha, c64* x) noexcept {
    if (n == 0) return;
    if (alpha.real() == 1.0 && alpha.imag() == 0.0) return;
    if (alpha.real() == 0.0 && alpha.imag() == 0.0) {
        for (usize i = 0; i < n; ++i) x[i] = c64(0.0, 0.0);
        return;
    }
    for (usize i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

void gemv(MatrixTranspose trans, usize m, usize n, f64 alpha, const f64* A, usize lda, const f64* x, usize incx,
          f64 beta, f64* y, usize incy) noexcept {
    if (trans == MatrixTranspose::NoTrans) {
        gemv(m, n, alpha, A, lda, x, incx, beta, y, incy);
        return;
    }
    // op(A) = A^T is n x m, y has length n, x has length m.
    if (n == 0) return;
    if (beta == 0.0) {
        for (usize j = 0; j < n; ++j) y[j * incy] = 0.0;
    } else if (beta != 1.0) {
        for (usize j = 0; j < n; ++j) y[j * incy] *= beta;
    }
    if (alpha == 0.0 || m == 0) return;
    for (usize j = 0; j < n; ++j) {
        const f64* aj = A + j * lda;
        f64 sum = 0.0;
        if (incx == 1) {
            sum = dot(m, aj, x);
        } else {
            for (usize i = 0; i < m; ++i) {
                sum += aj[i] * x[i * incx];
            }
        }
        y[j * incy] += alpha * sum;
    }
}

} // namespace keel::linalg
