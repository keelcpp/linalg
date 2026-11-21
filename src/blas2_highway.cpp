#include "keel/linalg.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "blas2_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace keel {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void GemvDoubleKernel(usize m, usize n, f64 alpha, const f64* A, usize lda, const f64* x, usize incx, f64 beta, f64* y,
                      usize incy) {
    if (m == 0) return;
    const hn::ScalableTag<f64> d;
    const usize lanes = hn::Lanes(d);

    if (beta == 0.0) {
        for (usize i = 0; i < m; ++i) y[i * incy] = 0.0;
    } else if (beta != 1.0) {
        for (usize i = 0; i < m; ++i) y[i * incy] *= beta;
    }

    if (alpha == 0.0 || n == 0) return;

    if (incy == 1) {
        for (usize j = 0; j < n; ++j) {
            const f64 xj = alpha * x[j * incx];
            if (xj == 0.0) continue;
            const f64* aj = A + j * lda;
            const auto v_x = hn::Set(d, xj);

            usize i = 0;
            for (; i + lanes <= m; i += lanes) {
                const auto v_y = hn::MulAdd(hn::LoadU(d, aj + i), v_x, hn::LoadU(d, y + i));
                hn::StoreU(v_y, d, y + i);
            }
            for (; i < m; ++i) y[i] += aj[i] * xj;
        }
    } else {
        for (usize j = 0; j < n; ++j) {
            const f64 xj = alpha * x[j * incx];
            if (xj == 0.0) continue;
            const f64* aj = A + j * lda;
            for (usize i = 0; i < m; ++i) y[i * incy] += aj[i] * xj;
        }
    }
}

void GerDoubleKernel(usize m, usize n, f64 alpha, const f64* x, usize incx, const f64* y, usize incy, f64* A,
                     usize lda) {
    if (alpha == 0.0 || m == 0 || n == 0) return;
    const hn::ScalableTag<f64> d;
    const usize lanes = hn::Lanes(d);

    for (usize j = 0; j < n; ++j) {
        const f64 yj = alpha * y[j * incy];
        if (yj == 0.0) continue;
        f64* aj = A + j * lda;
        const auto v_y = hn::Set(d, yj);

        if (incx == 1) {
            usize i = 0;
            for (; i + lanes <= m; i += lanes) {
                const auto v_a = hn::MulAdd(hn::LoadU(d, x + i), v_y, hn::LoadU(d, aj + i));
                hn::StoreU(v_a, d, aj + i);
            }
            for (; i < m; ++i) aj[i] += x[i] * yj;
        } else {
            for (usize i = 0; i < m; ++i) aj[i] += x[i * incx] * yj;
        }
    }
}

} // namespace HWY_NAMESPACE
} // namespace keel
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace keel {

HWY_EXPORT(GemvDoubleKernel);
HWY_EXPORT(GerDoubleKernel);

void gemv(usize m, usize n, f64 alpha, const f64* A, usize lda, const f64* x, usize incx, f64 beta, f64* y,
          usize incy) noexcept {
    HWY_DYNAMIC_DISPATCH(GemvDoubleKernel)(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void ger(usize m, usize n, f64 alpha, const f64* x, usize incx, const f64* y, usize incy, f64* A, usize lda) noexcept {
    HWY_DYNAMIC_DISPATCH(GerDoubleKernel)(m, n, alpha, x, incx, y, incy, A, lda);
}

} // namespace keel
#endif
