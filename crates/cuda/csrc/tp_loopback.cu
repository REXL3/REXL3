#include "common.cuh"

// The reduction of the in-process loopback TpComm (crates/model/.../nemotron_h_omni/tp_comm.rs): out[i] = a[i] + b[i] over n f32,
// `out` may alias `a`. One IEEE add per element, no fma, no reordering: a + b == b + a bit for bit, so the two ranks of a loopback group
// end with identical bits, and with disjoint support (one operand exactly 0.0) the result is the other operand.
//
// Denormals: x + 0.0 == x must hold for a subnormal x too, which a flush-to-zero add breaks. The add is therefore written as PTX
// `add.rn.f32` WITHOUT `.ftz`, which keeps subnormals whatever -ftz / --use_fast_math the file is compiled with (the crate's flags have
// neither; this keeps it so if they ever change). Do not replace it with `a + b`, `__fadd_rn` or `fadd`.
__device__ __forceinline__ float add_rn_keep_denormals(float a, float b) {
    float r;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__global__ void tp_add_f32_vec_k(const float4* a, const float4* b, float4* out, size_t n4) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        const float4 x = a[i], y = b[i];
        float4 o;
        o.x = add_rn_keep_denormals(x.x, y.x);
        o.y = add_rn_keep_denormals(x.y, y.y);
        o.z = add_rn_keep_denormals(x.z, y.z);
        o.w = add_rn_keep_denormals(x.w, y.w);
        out[i] = o;
    }
}

__global__ void tp_add_f32_k(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = add_rn_keep_denormals(a[i], b[i]);
}

GB_EXPORT int gb_tp_add_f32(const void* a, const void* b, void* out, long long n, void* stream) {
    if (n < 0) return 1;  // cudaErrorInvalidValue
    if (n == 0) return 0;
    const bool vec = n % 4 == 0 && ((size_t)a | (size_t)b | (size_t)out) % 16 == 0;
    const size_t units = vec ? (size_t)n / 4 : (size_t)n;
    size_t blocks = (units + 255) / 256;
    if (blocks > 1u << 16) blocks = 1u << 16;
    if (vec)
        tp_add_f32_vec_k<<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>((const float4*)a, (const float4*)b, (float4*)out, units);
    else
        tp_add_f32_k<<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>((const float*)a, (const float*)b, (float*)out, units);
    return GB_CHECK_LAUNCH();
}
