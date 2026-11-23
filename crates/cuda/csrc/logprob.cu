// gb-nvcc: --fmad=false
// Log-probability normaliser of bf16 logits rows (`src/logprob.rs`): per row the log-sum-exp over the valid columns and
// the logit of one chosen column. One block per row and nothing shared between rows, so a row's bits depend on that row
// and on `valid` alone: not on the row's index, on how many rows the launch has, or on how a prompt was cut into launches.
//
// The reduction order is fixed by LP_NT and `valid`: thread t sums exp(x[i] - max) over its columns i = t, t + LP_NT, ...
// in ascending order in fp32, then the LP_NT partial sums are folded pairwise (t with t + w for w = LP_NT/2 ... 1). The
// maximum is exact in any order. No fused multiply-add is contracted (the line above), so the sum is the one written.
#include "common.cuh"

namespace {

constexpr int LP_NT = 1024;

__device__ __forceinline__ float lp_bf16(unsigned short v) { return __uint_as_float((unsigned int)v << 16); }
__device__ __forceinline__ float lp_f16(unsigned short v) { return __half2float(__ushort_as_half(v)); }

// a NaN anywhere makes the maximum NaN (fmaxf alone would drop it)
__device__ __forceinline__ float lp_max(float a, float b) { return (isnan(a) || isnan(b)) ? NAN : fmaxf(a, b); }

template <bool F16>
__global__ void __launch_bounds__(LP_NT) logprob_norm_k(const unsigned short* __restrict__ logits, const int* __restrict__ chosen,
                                                        float* __restrict__ out, int stride, int valid) {
    __shared__ float sm[LP_NT];
    const int row = blockIdx.x, t = threadIdx.x;
    const unsigned short* x = logits + (size_t)row * stride;
    const auto load = [](unsigned short v) { return F16 ? lp_f16(v) : lp_bf16(v); };
    float m = -INFINITY;
    for (int i = t; i < valid; i += LP_NT) m = lp_max(m, load(x[i]));
    sm[t] = m;
    __syncthreads();
    for (int w = LP_NT / 2; w > 0; w >>= 1) {
        if (t < w) sm[t] = lp_max(sm[t], sm[t + w]);
        __syncthreads();
    }
    m = sm[0];
    __syncthreads();
    // a non-finite maximum (NaN, +inf, or -inf for a row of -inf) is the answer itself: there is nothing to shift by
    const bool finite = !isnan(m) && !isinf(m);
    float s = 0.0f;
    if (finite)
        for (int i = t; i < valid; i += LP_NT) s += expf(load(x[i]) - m);
    sm[t] = s;
    __syncthreads();
    for (int w = LP_NT / 2; w > 0; w >>= 1) {
        if (t < w) sm[t] += sm[t + w];
        __syncthreads();
    }
    if (t == 0) {
        const int c = chosen[row];
        out[2 * (size_t)row] = finite ? m + logf(sm[0]) : m;
        out[2 * (size_t)row + 1] = (c >= 0 && c < valid) ? load(x[c]) : -INFINITY;
    }
}

}  // namespace

// Per row r of bf16 logits [rows, stride], over columns [0, valid): out[2r] = log(sum_i exp(x[r][i])) in fp32 (max-shifted;
// NaN when the row holds a NaN, +inf when it holds +inf, -inf when every column is -inf) and out[2r + 1] = x[r][chosen[r]]
// widened to fp32 (-inf when chosen[r] is outside [0, valid): a column that is not in the row has probability zero).
// chosen i32 [rows]; out f32 [rows, 2]. Zero rows is a no-op.
GB_EXPORT int gb_logprob_norm(const void* logits, int rows, int stride, int valid, const void* chosen, void* out, void* stream) {
    if (rows < 0 || valid < 1 || valid > stride) return 1;
    if (rows == 0) return 0;
    if (!logits || !chosen || !out) return 1;
    logprob_norm_k<false><<<rows, LP_NT, 0, gb_stream(stream)>>>((const unsigned short*)logits, (const int*)chosen, (float*)out, stride,
                                                                  valid);
    return GB_CHECK_LAUNCH();
}

// The fp16 counterpart, used by the GLM and DeepSeek heads. Identical launch and reduction order to gb_logprob_norm.
GB_EXPORT int gb_logprob_norm_f16(const void* logits, int rows, int stride, int valid, const void* chosen, void* out, void* stream) {
    if (rows < 0 || valid < 1 || valid > stride) return 1;
    if (rows == 0) return 0;
    if (!logits || !chosen || !out) return 1;
    logprob_norm_k<true><<<rows, LP_NT, 0, gb_stream(stream)>>>((const unsigned short*)logits, (const int*)chosen, (float*)out, stride,
                                                                 valid);
    return GB_CHECK_LAUNCH();
}
