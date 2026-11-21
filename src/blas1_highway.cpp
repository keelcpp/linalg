#include "keel/linalg.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "blas1_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace keel::linalg {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void AxpyDoubleKernel(usize n, f64 alpha, const f64* x, f64* y) {
    if (alpha == 0.0 || n == 0) return;
    const hn::ScalableTag<f64> d;
    const usize lanes = hn::Lanes(d);
    const auto v_a = hn::Set(d, alpha);

    usize i = 0;
    for (; i + lanes <= n; i += lanes) {
        const auto v_y = hn::MulAdd(hn::LoadU(d, x + i), v_a, hn::LoadU(d, y + i));
        hn::StoreU(v_y, d, y + i);
    }
    for (; i < n; ++i) {
        y[i] += x[i] * alpha;
    }
}

} // namespace HWY_NAMESPACE
} // namespace keel::linalg
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace keel::linalg {

HWY_EXPORT(AxpyDoubleKernel);

void axpy(usize n, f64 alpha, const f64* x, f64* y) noexcept {
    HWY_DYNAMIC_DISPATCH(AxpyDoubleKernel)(n, alpha, x, y);
}

} // namespace keel::linalg
#endif
