#pragma once

#include "keel/core.h"
#include "keel/linalg.h"

#if defined(KEEL_LINALG_WITH_THREADS)
#include "keel/thread_pool.h"
#endif

#include <atomic>

namespace keel::linalg::linalg_detail {

inline std::atomic<usize> g_last_gemm_threads_used{0};

template <typename F>
inline void parallel_for(usize n, usize threshold, F&& fn, u32 max_workers = 0) {
#if defined(KEEL_LINALG_WITH_THREADS)
    keel::parallel_for(n, threshold, keel::forward<F>(fn), max_workers);
#else
    (void)threshold;
    (void)max_workers;
    if (n > 0) {
        fn(as<usize>(0), n);
    }
#endif
}

template <typename T>
[[nodiscard]] inline T linalg_min(T a, T b) noexcept {
    return (a < b) ? a : b;
}

template <typename T>
[[nodiscard]] inline T linalg_max(T a, T b) noexcept {
    return (a > b) ? a : b;
}

} // namespace keel::linalg::linalg_detail
