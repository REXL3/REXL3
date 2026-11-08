#include "common.cuh"

// The local half of the one-shot collective of the tensor-parallel Nemotron engine (docs/NEMOTRON_TP.md, "one-shot collective"): every
// rank has received the other ranks' buffers into staging slots and sums the R operands of one range in FIXED RANK ORDER,
//
//     out[i] = (((s[0][i] + s[1][i]) + s[2][i]) + ... + s[R-1][i])
//
// where s[r] is rank r's operand (the staging slot of rank r, or the rank's own buffer in place of its own slot). `out` may alias one of
// the operands (in place): every thread reads all of the operands of element i before it writes out[i], and no thread touches another
// element, so the alias is safe.
//
// Exactness: the engine's buffers have DISJOINT SUPPORT (each rank holds its own elements and exact -0.0 elsewhere), and
// `x + (-0.0) == x` for EVERY x (the sign of a zero and every subnormal included), so the sum is a copy of the owner's element in ANY
// order and for any R. With two operands it is `a + b`, bit-identical to the ring's. On dense data (no disjoint support) the fixed
// order makes every rank compute the same bits, which a ring does not guarantee for R > 2.
//
// No flush-to-zero anywhere: `x + (-0.0) == x` fails for a subnormal `x` under FTZ. The add is therefore written as PTX `add.rn.f32`
// WITHOUT `.ftz` (the same as `tp_loopback.cu`), which keeps subnormals whatever --ftz / --use_fast_math this file is built with, and
// no fused multiply-add can contract an add that has no multiply. Do not replace it with `a + b`, `__fadd_rn` or `fadd`.

namespace {

constexpr int kMaxSlots = 16;  // ranks of a group: the operand pointers travel in the kernel's parameters

struct SlotPtrs {
    const void* p[kMaxSlots];
};

__device__ __forceinline__ float add_rn_keep_denormals(float a, float b) {
    float r;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__global__ void tp_sum_slots_vec_k(SlotPtrs s, int n_slots, float4* out, size_t n4) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 acc = static_cast<const float4*>(s.p[0])[i];
        for (int r = 1; r < n_slots; ++r) {
            const float4 v = static_cast<const float4*>(s.p[r])[i];
            acc.x = add_rn_keep_denormals(acc.x, v.x);
            acc.y = add_rn_keep_denormals(acc.y, v.y);
            acc.z = add_rn_keep_denormals(acc.z, v.z);
            acc.w = add_rn_keep_denormals(acc.w, v.w);
        }
        out[i] = acc;
    }
}

__global__ void tp_sum_slots_k(SlotPtrs s, int n_slots, float* out, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float acc = static_cast<const float*>(s.p[0])[i];
        for (int r = 1; r < n_slots; ++r) acc = add_rn_keep_denormals(acc, static_cast<const float*>(s.p[r])[i]);
        out[i] = acc;
    }
}

}  // namespace

// out[0..n) = srcs[0][0..n) + srcs[1][0..n) + ... + srcs[n_srcs-1][0..n), left to right, f32, one IEEE add per element per operand.
// `srcs` is a HOST array of n_srcs device pointers (1 to 16; the kernel receives them by value, so the array may be a temporary).
// `out` may equal any of the srcs (in place); the ranges must otherwise be disjoint. Returns cudaErrorInvalidValue without launching for
// n_srcs outside 1..16, a null pointer or a negative n.
GB_EXPORT int gb_tp_sum_slots_f32(const void* const* srcs, int n_srcs, void* out, long long n, void* stream) {
    if (n < 0 || n_srcs < 1 || n_srcs > kMaxSlots || !srcs || (n > 0 && !out)) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;
    SlotPtrs s;
    size_t any = (size_t)out;
    for (int r = 0; r < kMaxSlots; ++r) s.p[r] = nullptr;
    for (int r = 0; r < n_srcs; ++r) {
        if (!srcs[r]) return (int)cudaErrorInvalidValue;
        s.p[r] = srcs[r];
        any |= (size_t)srcs[r];
    }
    const bool vec = n % 4 == 0 && (any & 15) == 0;
    const size_t units = vec ? (size_t)n / 4 : (size_t)n;
    size_t blocks = (units + 255) / 256;
    if (blocks > 1u << 16) blocks = 1u << 16;
    if (vec)
        tp_sum_slots_vec_k<<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>(s, n_srcs, static_cast<float4*>(out), units);
    else
        tp_sum_slots_k<<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>(s, n_srcs, static_cast<float*>(out), units);
    return GB_CHECK_LAUNCH();
}
