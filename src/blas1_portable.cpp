#include "keel/linalg.h"
#include "linalg_internal.h"

namespace keel {

void axpy(usize n, f64 alpha, const f64* x, f64* y) noexcept {
    if (alpha == 0.0 || n == 0) return;
    constexpr usize kParallelThreshold = 256 * 1024;
    linalg_detail::parallel_for(n, kParallelThreshold, [=](usize start, usize end) {
        for (usize i = start; i < end; ++i) {
            y[i] += alpha * x[i];
        }
    });
}

} // namespace keel
