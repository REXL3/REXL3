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
#include <cublasLt.h>

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

// H=2560, hc=4 is exactly ten floats per thread per stream. Keep them in registers across the
// reduction so the normalization pass reads each input element once instead of once per pass.
// The three float4 accumulations below deliberately retain norm_kernel's operation and addition
// order (including the absent final float4 in threads 128..255), making this a bit-exact fast path.
__global__ void __launch_bounds__(256) norm_kernel_2560_4(const float* __restrict__ x,
                                                          const half* __restrict__ nw,
                                                          half* __restrict__ normed_hi,
                                                          half* __restrict__ normed_lo,
                                                          float* __restrict__ rms, float eps) {
    constexpr int H = 2560, hc = 4;
    __shared__ float red[8];
    const int t = blockIdx.x;
    const int d0 = threadIdx.x * 4;
#pragma unroll
    for (int s = 0; s < hc; ++s) {
        const float* xr = x + ((size_t)t * hc + s) * H;
        const float4 v0 = *reinterpret_cast<const float4*>(xr + d0);
        const float4 v1 = *reinterpret_cast<const float4*>(xr + d0 + 1024);
        float4 v2 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (d0 < 512) v2 = *reinterpret_cast<const float4*>(xr + d0 + 2048);
        float a = 0.f;
        a += v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w;
        a += v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w;
        if (d0 < 512) a += v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w;
        const float r = rsqrtf(block_sum(a, red) / (float)H + eps);
        if (threadIdx.x == 0) rms[t * hc + s] = r;
        half* hi = normed_hi + ((size_t)t * hc + s) * H;
        half* lo = normed_lo + ((size_t)t * hc + s) * H;
        const half* w = nw + (size_t)s * H;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int d = d0 + j * 1024;
            if (j == 2 && d0 >= 512) break;
            const float4 v = j == 0 ? v0 : (j == 1 ? v1 : v2);
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

// The same fixed 64-row products, with all full tiles resident in one launch. Column-major A is the shared weight,
// so strideA=0; row-major activations and outputs advance by one 64-row tile per batch. Keeping n=64 (rather than
// making the whole chunk one GEMM) preserves each row's reduction schedule and chunk-independence contract.
int gemm_abt_tiles(cublasHandle_t h, const half* A, const half* B, float* C, int N, int K, float beta, int batches) {
    cudaStream_t st = nullptr;
    if (cublasGetStream(h, &st) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    static thread_local cublasLtHandle_t lt = nullptr;
    if (!lt && cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    struct LtPlan {
        int n = 0, k = 0;
        bool valid = false;
        cublasLtMatmulDesc_t op = nullptr;
        cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr, dd = nullptr;
        cublasLtMatmulAlgo_t algo{};
    };
    // Descriptors intentionally live as long as the handle to avoid repeated host setup. Plans are thread-local,
    // so overlapping prefill threads cannot race batch-count updates.
    static thread_local LtPlan plans[16];
    LtPlan* plan = nullptr;
    LtPlan* empty = nullptr;
    for (LtPlan& p : plans) {
        if (p.valid && p.n == N && p.k == K) {
            plan = &p;
            break;
        }
        if (!empty && !p.valid) empty = &p;
    }
    if (!plan) {
        if (!empty) return (int)cudaErrorUnknown;
        LtPlan p;
        p.n = N;
        p.k = K;
        bool ok = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS &&
                  cublasLtMatrixLayoutCreate(&p.ad, CUDA_R_16F, K, N, K) == CUBLAS_STATUS_SUCCESS &&
                  cublasLtMatrixLayoutCreate(&p.bd, CUDA_R_16F, K, 64, K) == CUBLAS_STATUS_SUCCESS &&
                  cublasLtMatrixLayoutCreate(&p.cd, CUDA_R_32F, N, 64, N) == CUBLAS_STATUS_SUCCESS &&
                  cublasLtMatrixLayoutCreate(&p.dd, CUDA_R_32F, N, 64, N) == CUBLAS_STATUS_SUCCESS;
        const cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
        ok = ok && cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)) ==
                       CUBLAS_STATUS_SUCCESS &&
             cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)) ==
                       CUBLAS_STATUS_SUCCESS;
        const long long astride = 0, bstride = (long long)64 * K, cstride = (long long)64 * N;
        const int one_batch = 1;
        auto set_layout = [&](cublasLtMatrixLayout_t d, long long stride) {
            return cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &one_batch,
                                                    sizeof(one_batch)) == CUBLAS_STATUS_SUCCESS &&
                   cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                                                    sizeof(stride)) == CUBLAS_STATUS_SUCCESS;
        };
        ok = ok && set_layout(p.ad, astride) && set_layout(p.bd, bstride) && set_layout(p.cd, cstride) &&
             set_layout(p.dd, cstride);
        cublasLtMatmulPreference_t pref = nullptr;
        const size_t no_workspace = 0;
        ok = ok && cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS &&
             cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &no_workspace,
                                                  sizeof(no_workspace)) == CUBLAS_STATUS_SUCCESS;
        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;
        // Optimize the target 4096-row shape, then pin that complete splitK=1 descriptor for smaller chunks too.
        // The layouts are reset to the actual batch count below before every launch.
        const int heuristic_batches = 64;
        for (cublasLtMatrixLayout_t d : {p.ad, p.bd, p.cd, p.dd})
            ok = ok && cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &heuristic_batches,
                                                        sizeof(heuristic_batches)) == CUBLAS_STATUS_SUCCESS;
        ok = ok && cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.ad, p.bd, p.cd, p.dd, pref, 1, &heuristic, &returned) ==
                       CUBLAS_STATUS_SUCCESS && returned == 1;
        if (ok) p.algo = heuristic.algo;
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (!ok) {
            if (p.op) cublasLtMatmulDescDestroy(p.op);
            for (cublasLtMatrixLayout_t d : {p.ad, p.bd, p.cd, p.dd})
                if (d) cublasLtMatrixLayoutDestroy(d);
            return (int)cudaErrorUnknown;
        }
        p.valid = true;
        *empty = p;
        plan = empty;
    }
    auto set_batches = [&](cublasLtMatrixLayout_t d) {
        return cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches)) ==
               CUBLAS_STATUS_SUCCESS;
    };
    if (!set_batches(plan->ad) || !set_batches(plan->bd) || !set_batches(plan->cd) || !set_batches(plan->dd))
        return (int)cudaErrorUnknown;
    const float one = 1.f;
    const cublasStatus_t status = cublasLtMatmul(lt, plan->op, &one, B, plan->ad, A, plan->bd, &beta, C, plan->cd, C,
                                                  plan->dd, &plan->algo, nullptr, 0, st);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
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

    // Keep each strip's G and normed hi/lo resident in L2 between the dependent up/mix launches.
    // The strip is a whole number of the same pinned 64-row GEMM tiles, so splitting the outer row
    // loop cannot change any reduction schedule or output bit. Scratch is deliberately reused at
    // offset zero; only caller-owned input/output rows advance.
    constexpr int strip_rows = 512;
    int rc = 0;
    for (int r0 = 0; r0 < T; r0 += strip_rows) {
        const int rows = T - r0 < strip_rows ? T - r0 : strip_rows;
        const int strip_padded = ((rows + 63) / 64) * 64;
        const float* strip_streams = (const float*)streams + (size_t)r0 * K;
        __nv_bfloat16* strip_y = (__nv_bfloat16*)y_out + (size_t)r0 * H;
        // A fixed 64-row GEMM tile gives the same accumulation for a row in every decode window.
        // Initialize only the unused tail of the final tile; valid rows are overwritten by norm.
        if (strip_padded != rows) {
            if (cudaMemsetAsync(normed_hi + (size_t)rows * K, 0, (size_t)(strip_padded - rows) * K * sizeof(half), st) !=
                cudaSuccess)
                return (int)cudaErrorUnknown;
            if (cudaMemsetAsync(normed_lo + (size_t)rows * K, 0, (size_t)(strip_padded - rows) * K * sizeof(half), st) !=
                cudaSuccess)
                return (int)cudaErrorUnknown;
        }
        if (H == 2560 && hc == 4)
            norm_kernel_2560_4<<<rows, 256, 0, st>>>(strip_streams, (const half*)norm_w, normed_hi, normed_lo, rms, eps);
        else
            norm_kernel<<<rows, 256, 0, st>>>(strip_streams, (const half*)norm_w, normed_hi, normed_lo, rms, H, hc, eps);
        if ((rc = GB_CHECK_LAUNCH())) return rc;
        const int tiles = strip_padded / 64;
        if ((rc = gemm_abt_tiles(h, normed_hi, (const half*)down_w, L, rank, K, 0.f, tiles))) return rc;
        if ((rc = gemm_abt_tiles(h, normed_lo, (const half*)down_w, L, rank, K, 1.f, tiles))) return rc;
        if (inject_w) {
            if ((rc = gemm_abt_tiles(h, normed_hi, (const half*)inject_w, Ij, hc, K, 0.f, tiles))) return rc;
            if ((rc = gemm_abt_tiles(h, normed_lo, (const half*)inject_w, Ij, hc, K, 1.f, tiles))) return rc;
        }
        const size_t n = (size_t)strip_padded * rank + (inject_w ? (size_t)strip_padded * hc : 0);
        low_kernel<<<(unsigned)((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, st>>>(
            L, inject_w ? Ij : nullptr, low_hi, low_lo, inject_w ? inj_pad : nullptr, strip_padded, rank, hc);
        if ((rc = GB_CHECK_LAUNCH())) return rc;
        if (inject_w &&
            cudaMemcpyAsync(inj_out + (size_t)r0 * hc, inj_pad, (size_t)rows * hc * sizeof(float),
                            cudaMemcpyDeviceToDevice, st) != cudaSuccess)
            return (int)cudaErrorUnknown;
        if ((rc = gemm_abt_tiles(h, low_hi, (const half*)up_w, G, K, rank, 0.f, tiles))) return rc;
        if ((rc = gemm_abt_tiles(h, low_lo, (const half*)up_w, G, K, rank, 1.f, tiles))) return rc;
        mix_kernel<<<rows, 256, 0, st>>>(strip_streams, G, rms, (const half*)norm_w, strip_y, H, hc);
        if ((rc = GB_CHECK_LAUNCH())) return rc;
    }
    return 0;
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
