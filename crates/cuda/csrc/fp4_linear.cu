// gb-nvcc: --fmad=false
// MXFP4 (E2M1 x E8M0-per-32) linear: y[m, n] = sum_k x[m, k] * W[n, k], fp32 accumulate, fp16 out.
//
// ---------------------------------------------------------------------------------------------------------------
// THE FORMAT, AS THE PACK STORES IT
//
// `mtp.{0,1,2}.ffn.experts.e.{w1,w3}.weight` is I8 `[2304, 2560]` -- 2560 bytes a row for 5120 values, two per
// byte -- with `...w1.scale` F8_E8M0 `[2304, 160]`, i.e. one scale a row per 32 columns. `w2` is the same at
// `[5120, 1152]` / `[5120, 72]`. So the scale tile is (one output row) x (32 inputs), NOT the 32x32 tile the fp8
// trunk path uses: `Mxfp4Weights` therefore indexes scales `[n, k/32]`, one byte each.
//
// A byte holds two E2M1 values, element 2i in the low nibble and 2i+1 in the high one -- the
// `torch.float4_e2m1fn_x2` convention, and the one `dsv4_kvq4.cu` already decodes for the q4 KV plane
// (`nibble_value(byte & 0xF)` for the even element, `byte >> 4` for the odd). E2M1 is
// {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit, so a nibble is `mag | (negative << 3)`.
//
// E8M0 is exponent-only: scale = 2^(b - 127), b = 0xFF is NaN. `ldexpf(1, b - 127)` is exact for every b, including
// b = 0 (2^-127, a subnormal f32 that an exponent-field insert would get wrong).
//
// ---------------------------------------------------------------------------------------------------------------
// WHY THIS IS A SEPARATE KERNEL FROM fp8_linear.cu
//
// The fp8 kernel reads ONE scale a block because a 32x32 tile is one scale cell. Here every output column of a block
// carries its own scale per k tile, so the scale is a per-thread scalar (broadcast out of shared) and the k loop is
// 32 wide -- the scale tile and the k tile have to agree or a k tile straddles two scales.
//
// Two shapes of the same kernel: ROWS = 8 (decode, m <= 8, grid.y = 1) and ROWS = 32 (the verify window: a draft
// round's `block_size` rows plus any speculative rows a caller packs). No split-K, no atomics, no workspace, and a
// fixed reduction order per output, so an element's bits do not depend on m.
#include "common.cuh"

namespace {

typedef unsigned int u32;

/// 2^(b - 127); 0xFF is NaN in E8M0.
__device__ __forceinline__ float e8m0_scale(uint8_t b) {
    return (b == 0xFFu) ? __int_as_float(0x7fc00000) : ldexpf(1.0f, (int)b - 127);
}

/// The 16 E2M1 values a nibble can hold, in code order (bit 3 = sign, bits 2:0 = magnitude).
__device__ __constant__ float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

/// x[i] as f32, x_dtype 0 = fp16, 1 = bf16 (the ABI codes of `exl3_linear::Dtype`).
__device__ __forceinline__ float ld_x(const void* x, int x_dtype, size_t i) {
    if (x_dtype == 0) {
        return __half2float(static_cast<const __half*>(x)[i]);
    }
    return __bfloat162float(static_cast<const __nv_bfloat16*>(x)[i]);
}

constexpr int TILE = 32;  // output columns per block == the k tile width == the scale cell width
constexpr int THREADS = 256;

/// y [m, n] = x [m, k] @ W [n, k]^T, W packed two E2M1 a byte, scales [n, k/32] E8M0.
/// Block: (32, 8). Grid: (n/32, ceil(m/ROWS)). Thread (tx, ty) owns output column n0 + tx for rows r0 + ty + 8i.
template <int ROWS>
__global__ void __launch_bounds__(THREADS) fp4_gemm_kernel(const void* __restrict__ x,
                                                           int x_dtype,
                                                           const uint8_t* __restrict__ w,
                                                           const uint8_t* __restrict__ s,
                                                           __half* __restrict__ y,
                                                           int m,
                                                           int k,
                                                           int n) {
    constexpr int RPT = ROWS / 8;  // output rows per thread
    __shared__ float sx[ROWS][TILE + 1];
    __shared__ float ssc[TILE];

    const int n0 = blockIdx.x * TILE;
    const int r0 = blockIdx.y * ROWS;
    const int tx = threadIdx.x;  // output column inside the tile
    const int ty = threadIdx.y;  // row group
    const int tid = ty * TILE + tx;
    const int kt = k / TILE;
    const int nc = n0 + tx;
    const uint8_t* __restrict__ wc = w + (size_t)nc * (k / 2);

    float acc[RPT];
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        acc[i] = 0.0f;
    }

    for (int kb = 0; kb < kt; ++kb) {
        // x tile, zero filled past m so a padding row is inert rather than out of bounds.
#pragma unroll
        for (int j = 0; j < (ROWS * TILE) / THREADS; ++j) {
            const int idx = tid + j * THREADS;
            const int rn = idx / TILE;
            const int kk = idx % TILE;
            const int rc = r0 + rn;
            sx[rn][kk] = (rc < m) ? ld_x(x, x_dtype, (size_t)rc * k + kb * TILE + kk) : 0.0f;
        }
        if (tid < TILE) {
            ssc[tid] = e8m0_scale(s[(size_t)(n0 + tid) * kt + kb]);
        }
        __syncthreads();

        // 16 bytes = 32 nibbles for this thread's column, in this k tile.
        const uint4 raw = *reinterpret_cast<const uint4*>(wc + (size_t)kb * (TILE / 2));
        const u32 words[4] = {raw.x, raw.y, raw.z, raw.w};
        const float sc = ssc[tx];
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
            const int rn = ty + i * 8;
            float dot = 0.0f;
#pragma unroll
            for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const u32 byte = (words[wd] >> (8 * b)) & 0xFFu;
                    dot = fmaf(sx[rn][wd * 8 + 2 * b], kE2M1[byte & 0x0Fu], dot);
                    dot = fmaf(sx[rn][wd * 8 + 2 * b + 1], kE2M1[byte >> 4], dot);
                }
            }
            acc[i] = fmaf(dot, sc, acc[i]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        const int rc = r0 + ty + i * 8;
        if (rc < m && nc < n) {
            y[(size_t)rc * n + nc] = __float2half(acc[i]);
        }
    }
}

/// out[i] = W[i] * scale[i / k][(i % k) / 32] in fp16, for the tests and for a host-side check of the layout.
__global__ void fp4_dequant_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                                   __half* __restrict__ out, long long total, int k, int kt) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const long long nrow = i / k;
    const int kcol = (int)(i % k);
    const uint8_t byte = w[i >> 1];
    const int nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    out[i] = __float2half(kE2M1[nib] * e8m0_scale(s[(size_t)nrow * kt + kcol / 32]));
}

/// Shared checks: m >= 1, k and n positive multiples of the scale tile.
inline int check_args(int m, int k, int n, int x_dtype) {
    if (m < 1 || k < TILE || n < TILE || (k % TILE) != 0 || (n % TILE) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (x_dtype != 0 && x_dtype != 1) {
        return (int)cudaErrorInvalidValue;
    }
    return 0;
}

}  // namespace

/// y [m, n] fp16 = x [m, k] @ W [n, k]^T with the [n, k/32] E8M0 scale. Asynchronous on `stream`.
GB_EXPORT int gb_fp4_linear(const void* x,
                            int x_dtype,
                            const uint8_t* w,
                            const uint8_t* s,
                            void* y,
                            int m,
                            int k,
                            int n,
                            void* stream) {
    if (x == nullptr || w == nullptr || s == nullptr || y == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    const int bad = check_args(m, k, n, x_dtype);
    if (bad != 0) {
        return bad;
    }
    __half* out = static_cast<__half*>(y);
    const int ntiles = n / TILE;
    if (m <= 8) {
        fp4_gemm_kernel<8><<<dim3(ntiles, 1), dim3(TILE, 8), 0, gb_stream(stream)>>>(x, x_dtype, w, s, out, m, k, n);
    } else {
        const int rtiles = (m + 31) / 32;
        fp4_gemm_kernel<32><<<dim3(ntiles, rtiles), dim3(TILE, 8), 0, gb_stream(stream)>>>(x, x_dtype, w, s, out, m, k, n);
    }
    return GB_CHECK_LAUNCH();
}

/// out [n, k] fp16 = dequantised W. Asynchronous on `stream`.
GB_EXPORT int gb_fp4_dequant(const uint8_t* w, const uint8_t* s, void* out, int n, int k, void* stream) {
    if (w == nullptr || s == nullptr || out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (n < 1 || k < TILE || (k % TILE) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    const long long total = (long long)n * (long long)k;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    fp4_dequant_kernel<<<(unsigned)blocks, threads, 0, gb_stream(stream)>>>(w, s, static_cast<__half*>(out), total, k,
                                                                          k / TILE);
    return GB_CHECK_LAUNCH();
}