// HCG: the multi-HyperConnection pre-mix for large row counts (prefill chunks) as two GEMMs. Original REXL3 code,
// Victor Cruz. Same math as HCX (hcx.cu) / gb_hc_pre:
//   normed[t][s][d] = x[t][s][d] * rms[t][s] * (1 + nw[s * H + d]),   rms = rsqrt(mean_d x^2 + eps)
//   low[t]  = silu(normed[t] . down^T / hc)          (down [rank][hc * H])
//   inj[t]  = 2 sigmoid(normed[t] . inject^T / hc)    (inject [hc][hc * H], optional)
//   y[t][d] = (1 / hc) sum_s sigmoid((low[t] . up^T)[s * H + d]) * normed[t][s][d]     (up [hc * H][rank])
// The two products run on cuBLAS tensor cores with fp16 operands (normed and low are O(1) after the norm) and fp32
// accumulation; the final mix recomputes normed in fp32 from the streams.
// Weight layout contract: every operand is read exactly at its declared stride, never past its own buffer. down is
// [rank][hc*H] and up is [hc*H][rank], both at row stride `rank`; the inject weights are their own [hc][hc*H] buffer
// read by the separate inject GEMMs, never trailing columns of down/up. The low-row scratch (L, low hi+lo) is stride
// `rank` too. Nothing reads past the end of a tensor: a caller that wants spare columns must size its buffers for them.
#include "common.cuh"
#include <cublas_v2.h>

namespace hcg {

__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w];
    __syncthreads();
    return t;
}

// one block a row: per stream the rms, and normed as an fp16 hi+lo split for the GEMMs. The split is the error-free
// Dekker pairing of the fp32 value: hi carries the top 11 mantissa bits, lo the next 11, so hi+lo reconstructs the fp32
// value to ~22 bits. The tensor-core GEMMs (fp16 operands, fp32 accumulate) are then exact in the products; the only
// remaining error is the ~2^-22 residual of the split, not the ~2^-11 of a single fp16 rounding.
__global__ void __launch_bounds__(256) norm_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                   half* __restrict__ normed_hi, half* __restrict__ normed_lo,
                                                   float* __restrict__ rms, int H, int hc, float eps) {
    __shared__ float red[8];
    const int t = blockIdx.x;
    for (int s = 0; s < hc; ++s) {
        const float* xr = x + ((size_t)t * hc + s) * H;
        float a = 0.f;
        for (int d = threadIdx.x * 4; d < H; d += blockDim.x * 4) {
            const float4 v = *reinterpret_cast<const float4*>(xr + d);
            a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float r = rsqrtf(block_sum(a, red) / (float)H + eps);
        if (threadIdx.x == 0) rms[t * hc + s] = r;
        half* hi = normed_hi + ((size_t)t * hc + s) * H;
        half* lo = normed_lo + ((size_t)t * hc + s) * H;
        const half* w = nw + (size_t)s * H;
        for (int d = threadIdx.x * 4; d < H; d += blockDim.x * 4) {
            const float4 v = *reinterpret_cast<const float4*>(xr + d);
            const float2 w01 = __half22float2(*reinterpret_cast<const half2*>(w + d));
            const float2 w23 = __half22float2(*reinterpret_cast<const half2*>(w + d + 2));
            const float n0 = v.x * r * (1.f + w01.x), n1 = v.y * r * (1.f + w01.y);
            const float n2 = v.z * r * (1.f + w23.x), n3 = v.w * r * (1.f + w23.y);
            const float h0 = __half2float(__float2half_rn(n0)), h1 = __half2float(__float2half_rn(n1));
            const float h2 = __half2float(__float2half_rn(n2)), h3 = __half2float(__float2half_rn(n3));
            *reinterpret_cast<half2*>(hi + d) = __floats2half2_rn(h0, h1);
            *reinterpret_cast<half2*>(hi + d + 2) = __floats2half2_rn(h2, h3);
            *reinterpret_cast<half2*>(lo + d) = __floats2half2_rn(n0 - h0, n1 - h1);
            *reinterpret_cast<half2*>(lo + d + 2) = __floats2half2_rn(n2 - h2, n3 - h3);
        }
    }
}

// low (fp16 hi+lo split) = silu(L / hc); inj = 2 sigmoid(Ij / hc). `rank` is both the L/low row stride and the width
// of the low columns (the inject columns are the separate Ij block, not trailing low columns).
__global__ void low_kernel(const float* __restrict__ L, const float* __restrict__ Ij, half* __restrict__ low_hi,
                           half* __restrict__ low_lo, float* __restrict__ inj, int T, int rank, int hc) {
    const float inv = 1.f / (float)hc;
    const size_t nl = (size_t)T * rank, ni = Ij ? (size_t)T * hc : 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nl + ni; i += (size_t)gridDim.x * blockDim.x) {
        if (i < nl) {
            const float v = L[i] * inv;
            const float sv = v / (1.f + __expf(-v));
            const float h = __half2float(__float2half_rn(sv));
            low_hi[i] = __float2half_rn(sv);
            low_lo[i] = __float2half_rn(sv - h);
        } else {
            const size_t j = i - nl;
            inj[j] = 2.f / (1.f + __expf(-Ij[j] * inv));
        }
    }
}

// y[t][d] = bf16((1 / hc) sum_s sigmoid(G[t][s * H + d]) * x[t][s][d] * rms[t][s] * (1 + nw[s * H + d]))
__global__ void __launch_bounds__(256) mix_kernel(const float* __restrict__ x, const float* __restrict__ G,
                                                  const float* __restrict__ rms, const half* __restrict__ nw,
                                                  __nv_bfloat16* __restrict__ y, int H, int hc) {
    const int t = blockIdx.x;
    const float inv = 1.f / (float)hc;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < hc; ++s) {
            const size_t i = ((size_t)t * hc + s) * H + d;
            const float g = 1.f / (1.f + __expf(-G[i]));
            acc += g * x[i] * rms[t * hc + s] * (1.f + __half2float(nw[(size_t)s * H + d]));
        }
        y[(size_t)t * H + d] = __float2bfloat16_rn(acc * inv);
    }
}

// One cuBLAS handle per host thread, created on first use and never destroyed (destroying it at thread exit can run
// after the CUDA context is gone). The static handle it replaces was a cross-thread cublasSetStream race if the server
// ever overlaps prefill and decode threads (same fix as exl3_linear.cu's cublas_handle).
cublasHandle_t handle() {
    static thread_local cublasHandle_t h = nullptr;
    if (!h && cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) h = nullptr;
    if (h) cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    return h;
}

// row-major C [T][N] fp32 = beta*C + alpha*A [T][K] fp16 . B^T, B row-major [N][K] fp16
int gemm_abt(cublasHandle_t h, const half* A, const half* B, float* C, int T, int N, int K, float beta) {
    const float one = 1.f;
    const cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, B, CUDA_R_16F, K, A, CUDA_R_16F, K,
                                          &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}

}  // namespace hcg

using namespace hcg;

// Scratch bytes for up to `rows` rows (normed hi+lo, L, Ij, low hi+lo, G, rms; each 256-aligned). `rank` is the low
// row stride; nothing here depends on any other layout choice.
GB_EXPORT size_t gb_hcg_scratch_bytes(int rows, int H, int hc, int rank) {
    auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
    const size_t K = (size_t)hc * H;
    const size_t padded = (size_t)((rows + 63) / 64) * 64;
    return 2 * al(padded * K * 2) + al(padded * rank * 4) + al(padded * hc * 4) + 2 * al(padded * rank * 2) +
           al(padded * K * 4) + 2 * al(padded * hc * 4);
}

// y [T][H] bf16 (+ inj [T][hc] f32 when inject_w is non-null) from streams [T][hc][H] f32. scratch holds
// gb_hcg_scratch_bytes(T, H, hc, rank). H % 4 == 0, rank >= 1. The weight operands are read exactly at their declared
// stride `rank` (down [rank][hc*H], up [hc*H][rank], inject [hc][hc*H]); no read extends past a weight buffer.
GB_EXPORT int gb_hcg_pre(const void* streams, const void* norm_w, const void* down_w, const void* up_w,
                         const void* inject_w, void* y_out, float* inj_out, void* scratch, int H, int hc, int rank,
                         float eps, int T, void* stream) {
    if (T < 1 || H % 4 || hc < 1 || rank < 1 || !scratch || (inject_w != nullptr) != (inj_out != nullptr))
        return (int)cudaErrorInvalidValue;
    cublasHandle_t h = handle();
    if (!h) return (int)cudaErrorUnknown;
    cudaStream_t st = gb_stream(stream);
    if (cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
    const int K = hc * H;
    const int padded = ((T + 63) / 64) * 64;
    char* p = static_cast<char*>(scratch);
    half* normed_hi = (half*)p;     p += al((size_t)padded * K * 2);
    half* normed_lo = (half*)p;     p += al((size_t)padded * K * 2);
    float* L = (float*)p;           p += al((size_t)padded * rank * 4);
    float* Ij = (float*)p;          p += al((size_t)padded * hc * 4);
    half* low_hi = (half*)p;        p += al((size_t)padded * rank * 2);
    half* low_lo = (half*)p;        p += al((size_t)padded * rank * 2);
    float* G = (float*)p;           p += al((size_t)padded * K * 4);
    float* rms = (float*)p;         p += al((size_t)padded * hc * 4);
    // inj_pad occupies the trailing hc*4 block already counted by gb_hcg_scratch_bytes
    // (the second `2 * al(padded*hc*4)` term); make the reservation explicit.
    float* inj_pad = (float*)p;     p += al((size_t)padded * hc * 4);

    // A fixed 64-row GEMM tile gives the same accumulation for a row in every decode window.
    // Initialize only the unused tail of the final tile; valid rows are overwritten by norm_kernel.
    if (padded != T) {
        if (cudaMemsetAsync(normed_hi + (size_t)T * K, 0, (size_t)(padded - T) * K * sizeof(half), st) != cudaSuccess)
            return (int)cudaErrorUnknown;
        if (cudaMemsetAsync(normed_lo + (size_t)T * K, 0, (size_t)(padded - T) * K * sizeof(half), st) != cudaSuccess)
            return (int)cudaErrorUnknown;
    }
    norm_kernel<<<T, 256, 0, st>>>((const float*)streams, (const half*)norm_w, normed_hi, normed_lo, rms, H, hc, eps);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    for (int row = 0; row < padded; row += 64) {
        if ((rc = gemm_abt(h, normed_hi + (size_t)row * K, (const half*)down_w, L + (size_t)row * rank, 64, rank, K, 0.f))) return rc;
        if ((rc = gemm_abt(h, normed_lo + (size_t)row * K, (const half*)down_w, L + (size_t)row * rank, 64, rank, K, 1.f))) return rc;
        if (inject_w) {
            if ((rc = gemm_abt(h, normed_hi + (size_t)row * K, (const half*)inject_w,
                                Ij + (size_t)row * hc, 64, hc, K, 0.f))) return rc;
            if ((rc = gemm_abt(h, normed_lo + (size_t)row * K, (const half*)inject_w,
                                Ij + (size_t)row * hc, 64, hc, K, 1.f))) return rc;
        }
    }
    const size_t n = (size_t)padded * rank + (inject_w ? (size_t)padded * hc : 0);
    low_kernel<<<(unsigned)((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, st>>>(
        L, inject_w ? Ij : nullptr, low_hi, low_lo, inject_w ? inj_pad : nullptr, padded, rank, hc);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    if (inject_w && cudaMemcpyAsync(inj_out, inj_pad, (size_t)T * hc * sizeof(float), cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        return (int)cudaErrorUnknown;
    for (int row = 0; row < padded; row += 64) {
        if ((rc = gemm_abt(h, low_hi + (size_t)row * rank, (const half*)up_w, G + (size_t)row * K, 64, K, rank, 0.f))) return rc;
        if ((rc = gemm_abt(h, low_lo + (size_t)row * rank, (const half*)up_w, G + (size_t)row * K, 64, K, rank, 1.f))) return rc;
    }
    mix_kernel<<<T, 256, 0, st>>>((const float*)streams, G, rms, (const half*)norm_w, (__nv_bfloat16*)y_out, H, hc);
    return GB_CHECK_LAUNCH();
}

// ------------------------------------------------------------------------------------------------ small linear GEMM
// y [rows][n] bf16 = x [rows][k] bf16 . w^T (w fp16 [n][k]), for prefill chunks: x widened to fp16 (exact for the
// O(1) mixed rows), one tensor-core GEMM with fp32 accumulation, rounded to bf16. scratch: rows * k * 2 + rows * n * 4.
namespace hcg {
__global__ void bf16_to_f16_kernel(const __nv_bfloat16* __restrict__ x, half* __restrict__ y, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = __float2half_rn(__bfloat162float(x[i]));
}
__global__ void f32_to_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = __float2bfloat16_rn(x[i]);
}
static unsigned grid_for(size_t n) { return (unsigned)((n + 255) / 256 < 8192 ? (n + 255) / 256 : 8192); }
}  // namespace hcg

GB_EXPORT size_t gb_lin16_scratch_bytes(int rows, int k, int n) {
    auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
    return al((size_t)rows * k * 2) + al((size_t)rows * n * 4);
}

GB_EXPORT int gb_lin16_gemm(const void* x, const void* w, void* y, int rows, int k, int n, void* scratch, void* stream) {
    if (rows < 1 || k < 1 || n < 1) return (int)cudaErrorInvalidValue;
    cublasHandle_t h = handle();
    if (!h) return (int)cudaErrorUnknown;
    cudaStream_t st = gb_stream(stream);
    if (cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    auto al = [](size_t v) { return (v + 255) & ~(size_t)255; };
    half* x16 = static_cast<half*>(scratch);
    float* yf = reinterpret_cast<float*>(static_cast<char*>(scratch) + al((size_t)rows * k * 2));
    const size_t nx = (size_t)rows * k, ny = (size_t)rows * n;
    bf16_to_f16_kernel<<<grid_for(nx), 256, 0, st>>>((const __nv_bfloat16*)x, x16, nx);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    if ((rc = gemm_abt(h, x16, (const half*)w, yf, rows, n, k, 0.f))) return rc;
    f32_to_bf16_kernel<<<grid_for(ny), 256, 0, st>>>(yf, (__nv_bfloat16*)y, ny);
    return GB_CHECK_LAUNCH();
}
