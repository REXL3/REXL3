// gb-nvcc: --fmad=false
// DSpark draft helpers (MiniCPM-V-4.7's draft, `docs/MINICPMV4_7_INTEGRATION_PLAN.md` E4a):
//   * the confidence head: sigmoid(proj . [h_k ; markov_w1[x_{k-1}]] + b) in fp32, one block per row;
//   * bf16 <-> fp16 activation copies for the dense (cuBLAS) reference path, with a counter of values the fp16 range cannot hold.
//
// Conventions as glue.cu: bf16 activations stored as uint16_t, fp32 math, compiled with --fmad=false (every fused multiply-add is an
// explicit fmaf so a host loop with the same order reproduces the bits), fixed reduction order that depends on the row length only.
#include "common.cuh"
#include <math.h>

namespace {

typedef uint16_t u16;
typedef uint32_t u32;

constexpr int ERR_INVALID = 1;  // cudaErrorInvalidValue

__device__ __forceinline__ float bf2f(u16 u) { return __uint_as_float((u32)u << 16); }
__device__ __forceinline__ u16 f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float h2f(u16 u) { return __half2float(__ushort_as_half(u)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

template <int NT>
__device__ __forceinline__ float block_sum(float v, float* sm) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    float s = sm[0];
#pragma unroll
    for (int w = 1; w < NT / 32; ++w) s = s + sm[w];
    __syncthreads();
    return s;
}

constexpr int CONF_NT = 256;

// Row r = seq * rows + j of the block rows: x_{j-1} is the anchor for j == 0 and the walk's pick of row r - 1 otherwise.
//   out[r] = sigmoid( sum_i h[r, i] * w[i] + sum_i w1[x_{j-1}, i] * w[hidden + i] + b )
// Thread t owns h elements t, t + 256, ... then the Markov elements t, t + 256, ...; fmaf in that order; then the block butterfly.
__global__ void __launch_bounds__(CONF_NT)
    confidence_k(const u16* __restrict__ h, const int* __restrict__ tokens, const int* __restrict__ anchor, const u16* __restrict__ w1,
                 const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ out, int rows, int hidden, int rank,
                 int vocab) {
    __shared__ float sm[CONF_NT / 32];
    const int r = blockIdx.x;
    const int seq = r / rows, j = r - seq * rows;
    const int prev = j == 0 ? anchor[seq] : tokens[r - 1];
    const bool ok = (unsigned)prev < (unsigned)vocab;
    float acc = 0.f;
    for (int i = threadIdx.x; i < hidden; i += CONF_NT) acc = fmaf(bf2f(h[(size_t)r * hidden + i]), w[i], acc);
    for (int i = threadIdx.x; i < rank; i += CONF_NT)
        acc = fmaf(ok ? h2f(w1[(size_t)(unsigned)prev * rank + i]) : 0.f, w[hidden + i], acc);
    acc = block_sum<CONF_NT>(acc, sm);
    if (threadIdx.x == 0) out[r] = sigmoid_f(acc + b[0]);
}

// dst = fp16(src); *bad += 1 for every element whose fp16 value is not finite (an input past 65504, inf or NaN).
__global__ void bf16_to_f16_k(const u16* __restrict__ src, u16* __restrict__ dst, long long n, int* __restrict__ bad) {
    int local_bad = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const __half hv = __float2half_rn(bf2f(src[i]));
        if (!isfinite(__half2float(hv))) ++local_bad;
        dst[i] = __half_as_ushort(hv);
    }
    if (bad && local_bad) atomicAdd(bad, local_bad);
}

__global__ void f16_to_bf16_k(const u16* __restrict__ src, u16* __restrict__ dst, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        dst[i] = f2bf(h2f(src[i]));
}

inline unsigned blocks_for(long long n) {
    long long b = (n + 255) / 256;
    if (b > (1 << 16)) b = 1 << 16;
    return (unsigned)(b < 1 ? 1 : b);
}

}  // namespace

// h: bf16 [n_seq * rows, hidden] (the final-normed block rows); tokens: i32 [n_seq * rows] (the walk's picks); anchor: i32 [n_seq];
// w1: f16 [vocab, rank] (markov_w1); w: f32 [hidden + rank]; b: f32 [1]; out: f32 [n_seq * rows].
GB_EXPORT int gb_dspark_confidence(const void* h, const void* tokens, const void* anchor, const void* w1, const void* w, const void* b,
                                   void* out, int n_seq, int rows, int hidden, int rank, int vocab, void* stream) {
    if (n_seq < 0 || rows <= 0 || hidden <= 0 || rank <= 0 || vocab <= 0) return ERR_INVALID;
    if (n_seq == 0) return 0;
    confidence_k<<<n_seq * rows, CONF_NT, 0, gb_stream(stream)>>>((const u16*)h, (const int*)tokens, (const int*)anchor, (const u16*)w1,
                                                                  (const float*)w, (const float*)b, (float*)out, rows, hidden, rank, vocab);
    return GB_CHECK_LAUNCH();
}

// dst (fp16) = src (bf16) over n elements; bad (nullable device i32, zeroed by the caller) counts the non-finite results.
GB_EXPORT int gb_dspark_bf16_to_f16(const void* src, void* dst, long long n, void* bad, void* stream) {
    if (n < 0) return ERR_INVALID;
    if (n == 0) return 0;
    bf16_to_f16_k<<<blocks_for(n), 256, 0, gb_stream(stream)>>>((const u16*)src, (u16*)dst, n, (int*)bad);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dspark_f16_to_bf16(const void* src, void* dst, long long n, void* stream) {
    if (n < 0) return ERR_INVALID;
    if (n == 0) return 0;
    f16_to_bf16_k<<<blocks_for(n), 256, 0, gb_stream(stream)>>>((const u16*)src, (u16*)dst, n);
    return GB_CHECK_LAUNCH();
}
