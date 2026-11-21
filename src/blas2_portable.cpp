#include "keel/linalg.h"
#include "linalg_internal.h"

namespace keel::linalg {

void gemv(usize m, usize n, f64 alpha, const f64* A, usize lda, const f64* x, usize incx, f64 beta, f64* y,
          usize incy) noexcept {
    if (m == 0) return;

    if (beta == 0.0) {
        for (usize i = 0; i < m; ++i) y[i * incy] = 0.0;
    } else if (beta != 1.0) {
        for (usize i = 0; i < m; ++i) y[i * incy] *= beta;
    }

    if (alpha == 0.0 || n == 0) return;

    constexpr usize kRowThreshold = 1024;
    linalg_detail::parallel_for(m, kRowThreshold, [=](usize i_start, usize i_end) {
        for (usize j = 0; j < n; ++j) {
            const f64 xj = alpha * x[j * incx];
            if (xj == 0.0) continue;
            const f64* aj = A + j * lda;
            for (usize i = i_start; i < i_end; ++i) {
                y[i * incy] += aj[i] * xj;
            }
        }
    });
}

void ger(usize m, usize n, f64 alpha, const f64* x, usize incx, const f64* y, usize incy, f64* A, usize lda) noexcept {
    if (alpha == 0.0 || m == 0 || n == 0) return;

    constexpr usize kParallelThreshold = 256;
    linalg_detail::parallel_for(n, kParallelThreshold, [=](usize j_start, usize j_end) {
        for (usize j = j_start; j < j_end; ++j) {
            const f64 yj = alpha * y[j * incy];
            if (yj == 0.0) continue;
            f64* aj = A + j * lda;
            if (incx == 1) {
                for (usize i = 0; i < m; ++i) aj[i] += x[i] * yj;
            } else {
                for (usize i = 0; i < m; ++i) aj[i] += x[i * incx] * yj;
            }
        }
    });
}

} // namespace keel::linalg
