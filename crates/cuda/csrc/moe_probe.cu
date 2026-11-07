// Streaming-read bandwidth probe: the roofline the expert GEMV is measured against on this box today (clocks, other tenants).
// Not part of any serving path.

#include <stdint.h>

#include "common.cuh"

namespace {

__global__ void __launch_bounds__(256) read_probe_kernel(const uint4* __restrict__ p, size_t n16, uint32_t* __restrict__ out) {
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n16; i += 4 * stride) {
        const uint4 a = __ldcs(p + i), b = __ldcs(p + i + stride), c = __ldcs(p + i + 2 * stride), d = __ldcs(p + i + 3 * stride);
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n16; i += stride) {
        const uint4 a = __ldcs(p + i);
        acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    if (acc == 0x9E3779B9u) out[0] = acc;   // keeps the loads alive
}

}  // namespace

// Reads `bytes` (a multiple of 16) from `p` once with `blocks` x 256 threads and discards them. `out` is one device word.
GB_EXPORT int gb_moe_read_probe(const void* p, size_t bytes, void* out, int blocks, void* stream) {
    if (!p || !out || bytes < 16 || bytes % 16 || blocks < 1) return (int)cudaErrorInvalidValue;
    read_probe_kernel<<<blocks, 256, 0, gb_stream(stream)>>>(static_cast<const uint4*>(p), bytes / 16, static_cast<uint32_t*>(out));
    return GB_CHECK_LAUNCH();
}
