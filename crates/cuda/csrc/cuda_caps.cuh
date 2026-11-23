// Per-device CUDA capability probes shared by the host launchers.
#pragma once
#include <cuda_runtime.h>
#include <atomic>

namespace gbx_cuda {

// major * 100 + minor (sm_89 = 809, sm_121 = 1201), or -1 when the current
// device or either attribute cannot be queried. The attribute pair is probed
// at most once per device. Concurrent first calls may duplicate the probe, but
// the atomic publication is race-free and every caller observes a whole value.
inline int compute_capability() {
    static std::atomic<int> cached[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return -1;
    int cc = cached[dev].load(std::memory_order_relaxed);
    if (cc == 0) {
        int major = 0, minor = 0;
        const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        cc = ok ? major * 100 + minor : -1;
        cached[dev].store(cc, std::memory_order_relaxed);
    }
    return cc;
}

inline int compute_capability_major() {
    const int cc = compute_capability();
    return cc > 0 ? cc / 100 : -1;
}

// Both the running device and the compiled device code must support PDL. The
// build is single-architecture (crates/cuda/build.rs), so __CUDA_ARCH_LIST__
// is one number in the host pass too. In particular, sm_89 PTX JIT-run on
// sm_90+ must not enable launches whose JIT-ed waits are compiled out.
inline bool device_supports_pdl() {
#if !defined(__CUDA_ARCH_LIST__) || __CUDA_ARCH_LIST__ < 900
    return false;
#else
    return compute_capability_major() >= 9;
#endif
}

}  // namespace gbx_cuda
