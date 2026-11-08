// Kernels of the Nemotron 3.5 Super VL vision path: the RADIO ViT-H tower (32 blocks, hidden 1280, 16 heads of 80), the
// pixel-shuffle projector (ps_version v2, ratio 2) and mlp1 (RMSNorm, relu^2 MLP). The reference is the pack's own code run
// in bf16 (tools/dump_radio_reference.py); this file reproduces it from the pixels.
//
// Conventions: row-major. The residual stream is fp32; GEMM operands and activations are fp16 with fp32 accumulation;
// LayerNorm, softmax and RMSNorm statistics are fp32. Weights are the checkpoint's [out, in] layout, so every GEMM is
// C[m, n] = A[m, k] * B[n, k]^T, which is exactly the mma.sync "row.col" operand layout (both operands K-contiguous).
// Every kernel is deterministic (fixed reduction order, no atomics, no split-K).
//
// The tower's arithmetic is not a bitwise contract (it has to sit at the bf16 noise floor against the torch dumps), so the
// tensor-core kernels below reorder the accumulation freely:
//   * gemm: mma.sync m16n8k16 fp16 in / fp32 accumulate, 128x128x32 CTA tiles, a 3-stage cp.async pipeline, ldmatrix fragments
//     from XOR-swizzled shared tiles, and the epilogues (bias, exact GELU, relu^2, fp32 residual add) fused on the accumulators.
//     Shapes the fast path does not take (N or K not a multiple of 8 / 32, unaligned C) run the simple kernel (gemm_simple_kernel).
//   * attention: flash-style, head dim 80, 64-key tiles through cp.async double or triple buffers, fp32 online softmax (exp2
//     domain), the probabilities rounded to fp16 for the P.V product (as FlashAttention does; the reference is a bf16 eager
//     attention, so its own probabilities are coarser than that).
//   * layernorm: one warp per row, the row held in registers, fp32 statistics (two-pass).
// Helpers are copies of the ones in vision.cu / glm_vision.cu, so this file does not depend on the other towers' kernels.

#include <stdint.h>

#include "common.cuh"

namespace {

constexpr int HID = 1280;      // hidden size of the tower and of the LayerNorms
constexpr int HEADS = 16;      // attention heads
constexpr int HD = 80;         // head dim (HEADS * HD == HID)
constexpr int PATCH_PX = 256;  // pixels of one 16 x 16 patch plane
constexpr int PREFIX = 10;     // cls and register tokens, dropped from the features
constexpr int POS_GRID = 128;  // pack's position-embedding grid (128 x 128)
constexpr int PS_C = 4 * HID;  // pixel-shuffle channels (5120)
constexpr int THREADS = 256;

// ---------------------------------------------------------------------------------------------------------------------
// Tensor-core helpers.
// ---------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// 16 bytes global -> shared; `valid` false zero-fills the 16 bytes without reading global memory.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    const int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// ---------------------------------------------------------------------------------------------------------------------
// Row kernels.
// ---------------------------------------------------------------------------------------------------------------------

// Sum over the block; every thread gets the total. `sh` is 32 floats of shared memory. The leading barrier orders this call's
// reads after the previous call's, so one buffer serves any number of reductions.
__device__ __forceinline__ float block_sum(float v, float* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    const int nw = blockDim.x >> 5;
    for (int i = 0; i < nw; ++i) r += sh[i];
    return r;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * w + b over `dim` (biased variance, fp32 statistics). x fp32, y fp16. Any dim.
__global__ void __launch_bounds__(THREADS) layernorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                            const float* __restrict__ b, half* __restrict__ y, int dim,
                                                            float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t)blockIdx.x * dim;
    half* yr = y + (size_t)blockIdx.x * dim;
    float s = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) s += xr[c];
    const float mean = block_sum(s, sh) / (float)dim;
    float v = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const float d = xr[c] - mean;
        v += d * d;
    }
    const float inv = rsqrtf(block_sum(v, sh) / (float)dim + eps);
    for (int c = threadIdx.x; c < dim; c += blockDim.x) yr[c] = __float2half_rn((xr[c] - mean) * inv * w[c] + b[c]);
}

// The same for dim == HID (1280): one warp a row, the row in registers (10 float4 a lane), the row read from memory once.
// Eight rows a block. x must be 16-byte aligned (rows are 5,120 bytes).
constexpr int LN_ROWS = 8;
__global__ void __launch_bounds__(LN_ROWS * 32) layernorm_warp_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                                      const float* __restrict__ b, half* __restrict__ y,
                                                                      int rows, float eps) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * LN_ROWS + warp;
    if (row >= rows) return;
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * HID);
    float4 v[10];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        v[j] = xr[lane + 32 * j];
        s += (v[j].x + v[j].y) + (v[j].z + v[j].w);
    }
    const float mean = warp_sum(s) / (float)HID;
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        const float a = v[j].x - mean, c = v[j].y - mean, d = v[j].z - mean, e = v[j].w - mean;
        q += (a * a + c * c) + (d * d + e * e);
    }
    const float inv = rsqrtf(warp_sum(q) / (float)HID + eps);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    uint2* yr = reinterpret_cast<uint2*>(y + (size_t)row * HID);
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        const float4 wv = __ldg(w4 + lane + 32 * j), bv = __ldg(b4 + lane + 32 * j);
        uint2 o;
        o.x = pack_h2((v[j].x - mean) * inv * wv.x + bv.x, (v[j].y - mean) * inv * wv.y + bv.y);
        o.y = pack_h2((v[j].z - mean) * inv * wv.z + bv.z, (v[j].w - mean) * inv * wv.w + bv.w);
        yr[lane + 32 * j] = o;
    }
}

// y[r, :] = w * x[r, :] / sqrt(mean(x[r, :]^2) + eps), the mlp1.0 RMSNorm (weight applied to the normalized value, one rounding).
__global__ void __launch_bounds__(THREADS) rmsnorm_kernel(const half* __restrict__ x, const float* __restrict__ w,
                                                          half* __restrict__ y, int dim, float eps) {
    __shared__ float sh[32];
    const half* xr = x + (size_t)blockIdx.x * dim;
    half* yr = y + (size_t)blockIdx.x * dim;
    float s = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const float v = __half2float(xr[c]);
        s += v * v;
    }
    const float inv = rsqrtf(block_sum(s, sh) / (float)dim + eps);
    for (int c = threadIdx.x; c < dim; c += blockDim.x) yr[c] = __float2half_rn(w[c] * (__half2float(xr[c]) * inv));
}

// Patch embedding input: pix fp16 [C, H, W] (already normalized) -> patches fp16 [gh * gw, C * 256]. The column order inside a
// patch is the pack's (channel, row, column): index c * 256 + py * 16 + px, which is the order of the embedder's weight. C is 3
// for an image; a two-frame video patch is C = 6 (frame 0's three planes, then frame 1's).
__global__ void patchify_kernel(const half* __restrict__ pix, half* __restrict__ out, int H, int W, int gw, int C) {
    const int p = blockIdx.x;
    const int gy = p / gw, gx = p % gw;
    const int K = C * PATCH_PX;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const int c = i >> 8, py = (i >> 4) & 15, px = i & 15;
        out[(size_t)p * K + i] = pix[((size_t)c * H + gy * 16 + py) * W + gx * 16 + px];
    }
}

// The pack's square-then-crop position resample, one coordinate. The table has POS_GRID samples per side; the resample is
// bilinear to an m x m grid (align_corners=False, no antialias, PyTorch's source-index rule), and the patch at index i of the
// crop reads the resampled grid at index i. Returns the two taps and the weight of the second one.
__device__ __forceinline__ void resample_tap(int i, int m, int& i0, int& i1, float& lam) {
    const float scale = (float)POS_GRID / (float)m;
    float src = scale * ((float)i + 0.5f) - 0.5f;
    if (src < 0.f) src = 0.f;
    const int lo = (int)src;
    i0 = lo;
    i1 = lo < POS_GRID - 1 ? lo + 1 : lo;
    lam = src - (float)lo;
}

// h fp32 [PREFIX + gh * gw, HID]. Rows 0..PREFIX-1 are the cls and register tokens (copied from `cls`). Patch row PREFIX + p
// (the GEMM's output, already in h) gets pos(p) added, where pos is the bilinear resample of `pos` [128 * 128, HID] (fp32, the
// pack's bf16 table widened) at the square max(gh, gw) grid, read at the patch's own (row, column).
__global__ void __launch_bounds__(THREADS) pos_embed_kernel(float* __restrict__ h, const float* __restrict__ pos,
                                                            const float* __restrict__ cls, int gh, int gw) {
    const int row = blockIdx.x;
    float* out = h + (size_t)row * HID;
    if (row < PREFIX) {
        for (int c = threadIdx.x; c < HID; c += blockDim.x) out[c] = cls[(size_t)row * HID + c];
        return;
    }
    const int p = row - PREFIX;
    const int gy = p / gw, gx = p % gw;
    const int m = gh > gw ? gh : gw;
    int y0, y1, x0, x1;
    float ly, lx;
    resample_tap(gy, m, y0, y1, ly);
    resample_tap(gx, m, x0, x1, lx);
    const float* t00 = pos + (size_t)(y0 * POS_GRID + x0) * HID;
    const float* t01 = pos + (size_t)(y0 * POS_GRID + x1) * HID;
    const float* t10 = pos + (size_t)(y1 * POS_GRID + x0) * HID;
    const float* t11 = pos + (size_t)(y1 * POS_GRID + x1) * HID;
    for (int c = threadIdx.x; c < HID; c += blockDim.x) {
        const float top = (1.f - lx) * t00[c] + lx * t01[c];
        const float bot = (1.f - lx) * t10[c] + lx * t11[c];
        out[c] += (1.f - ly) * top + ly * bot;
    }
}

// ps_version v2 pixel shuffle, ratio 2. in fp16 [gh * gw, HID] token-major; out fp16 [(gh / 2) * (gw / 2), 4 * HID] with the
// channel order (row inside the 2 x 2 block, column inside it, channel). One block per output token.
__global__ void __launch_bounds__(THREADS) pixel_shuffle_kernel(const half* __restrict__ in, half* __restrict__ out, int gw) {
    const int o = blockIdx.x;
    const int ow = gw / 2;
    const int R = o / ow, C = o % ow;
    for (int ch = threadIdx.x; ch < PS_C; ch += blockDim.x) {
        const int blk = ch / HID, d = ch % HID;
        const int r = 2 * R + (blk >> 1), c = 2 * C + (blk & 1);
        out[(size_t)o * PS_C + ch] = in[((size_t)r * gw + c) * HID + d];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// GEMM. C[M, N] = A[M, K] * B[N, K]^T with fp16 A and B, fp32 accumulation, and the epilogue applied once to each element.
// ---------------------------------------------------------------------------------------------------------------------

enum { EPI_BIAS_F16 = 0, EPI_BIAS_GELU_F16 = 1, EPI_BIAS_ADD_F32 = 2, EPI_STORE_F32 = 3, EPI_RELU2_F16 = 4 };

template <int EPI>
struct EpiOut {
    static constexpr bool f16 = EPI == EPI_BIAS_F16 || EPI == EPI_BIAS_GELU_F16 || EPI == EPI_RELU2_F16;
};

// v = acc + bias (bias may be absent) and the fp16 epilogues' nonlinearity, as one float.
template <int EPI>
__device__ __forceinline__ float epi_value(float acc, float bias) {
    const float v = acc + bias;
    if (EPI == EPI_BIAS_GELU_F16) return 0.5f * v * (1.f + erff(v * 0.7071067811865476f));  // exact GELU
    if (EPI == EPI_RELU2_F16) {                                                              // relu(v)^2 (mlp1.1: no bias)
        const float r = v > 0.f ? v : 0.f;
        return r * r;
    }
    return v;
}

// Reference-simple kernel for the shapes the tensor-core path does not take. Block tile 64 x 64 x 16, 4 x 4 outputs a thread.
template <int EPI>
__global__ void __launch_bounds__(THREADS) gemm_simple_kernel(const half* __restrict__ A, const half* __restrict__ B,
                                                              const float* __restrict__ bias, void* __restrict__ C, int M,
                                                              int N, int K, int ldc) {
    __shared__ float As[16][64 + 4];  // [k][m]
    __shared__ float Bs[16][64 + 4];  // [k][n]
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + i * THREADS;  // 0..1023 = 64 rows x 16 k
            const int row = idx >> 4, kk = idx & 15;
            const int gm = m0 + row, gn = n0 + row, gk = k0 + kk;
            As[kk][row] = (gm < M && gk < K) ? __half2float(A[(size_t)gm * K + gk]) : 0.f;
            Bs[kk][row] = (gn < N && gk < K) ? __half2float(B[(size_t)gn * K + gk]) : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            const float v = epi_value<EPI>(acc[i][j], bias ? bias[n] : 0.f);
            const size_t at = (size_t)m * ldc + n;
            if (EPI == EPI_BIAS_ADD_F32) {
                ((float*)C)[at] += v;  // residual add, in place
            } else if (EPI == EPI_STORE_F32) {
                ((float*)C)[at] = v;
            } else {
                ((half*)C)[at] = __float2half_rn(v);
            }
        }
    }
}

// The tensor-core kernel. CTA tile BM x BN, k step BK (a multiple of 32), STAGES cp.async stages, WARPS_M x WARPS_N warps each
// owning a (BM / WARPS_M) x (BN / WARPS_N) tile of m16n8 fragments. Requires N % 8 == 0 and K % BK == 0; M is arbitrary (rows
// past M are zero-filled on load and skipped on store); C is 16-byte aligned with ldc % 8 == 0 for the fp16 epilogues and
// 8-byte aligned with ldc even for the fp32 ones.
//
// Shared tiles are [rows][BK] halves, each 16-byte chunk c of row r stored at chunk c ^ swz(r), so cp.async stores and
// ldmatrix reads (8 consecutive rows of one logical chunk) are bank-conflict free. The fp16 epilogues stage the tile through
// the pipeline's shared memory for 16-byte coalesced stores; the fp32 ones store the fragments' float2 (a quad of lanes covers
// one 32-byte sector of a row) directly.
template <int BM, int BN, int BK, int STAGES, int WARPS_M, int WARPS_N>
struct GemmCfg {
    static constexpr int NT = WARPS_M * WARPS_N * 32;
    static constexpr int WTM = BM / WARPS_M, WTN = BN / WARPS_N;
    static constexpr int MI = WTM / 16, NI = WTN / 8;
    static constexpr int CH = BK / 8;  // 16-byte chunks of a shared row
    static constexpr int PIPE_BYTES = STAGES * (BM + BN) * BK * 2;
    static constexpr int C_BYTES = BM * (BN + 8) * 2;
    static constexpr int SMEM = PIPE_BYTES > C_BYTES ? PIPE_BYTES : C_BYTES;
    static_assert(BM % (WARPS_M * 16) == 0 && BN % (WARPS_N * 16) == 0, "warp tile");
    static_assert(CH == 4 || CH == 8, "swizzle");
    static_assert(BK % 16 == 0, "k step");
};

template <int CH>
__device__ __forceinline__ int tile_swz(int r) {
    return CH == 8 ? (r & 7) : ((r >> 1) & 3);
}

template <int EPI, int BM, int BN, int BK, int STAGES, int WARPS_M, int WARPS_N, int MINB>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32, MINB)
    gemm_mma_kernel(const half* __restrict__ A, const half* __restrict__ B, const float* __restrict__ bias, void* __restrict__ C,
                    int M, int N, int K, int ldc, int mt, int nt, int group) {
    using Cfg = GemmCfg<BM, BN, BK, STAGES, WARPS_M, WARPS_N>;
    constexpr int NT = Cfg::NT, MI = Cfg::MI, NI = Cfg::NI, CH = Cfg::CH;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const uint32_t smem_base = smem_u32(smem_raw);
    const uint32_t a_base = smem_base;                          // [STAGES][BM][BK]
    const uint32_t b_base = smem_base + STAGES * BM * BK * 2;  // [STAGES][BN][BK]

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
    // grouped tile order: `group` row tiles side by side, then the next column tile, so the CTAs resident together (about 2 a
    // SM) read a compact patch of A and B (an L2-friendly walk when either operand is larger than the L2)
    int bm, bn;
    {
        const int pid = blockIdx.x, per_group = group * nt;
        const int gi = pid / per_group, first = gi * group;
        const int gsz = min(mt - first, group);
        const int r = pid - gi * per_group;
        bm = first + r % gsz;
        bn = r / gsz;
    }
    const int m0 = bm * BM, n0 = bn * BN;
    const int KT = K / BK;

    // The in-place residual add reads its C tile in the epilogue. Every CTA of a wave reaches its epilogue together, so that DRAM
    // traffic would be exposed after the main loop; asking for the lines PF_AHEAD iterations before the end (a whole wave's tiles
    // at DRAM rate, and short enough that the streamed A and B do not evict them) hides it under the loop.
    constexpr int PF_AHEAD = 32;
    const int pf_at = KT > PF_AHEAD ? KT - PF_AHEAD : 0;

    auto load_stage = [&](int stage, int k0) {
        const uint32_t as = a_base + stage * (BM * BK * 2), bs = b_base + stage * (BN * BK * 2);
        for (int i = tid; i < BM * CH; i += NT) {
            const int r = i / CH, c = i % CH, gm = m0 + r;
            const bool ok = gm < M;
            cp_async16(as + r * (BK * 2) + ((c ^ tile_swz<CH>(r)) << 4), A + (size_t)(ok ? gm : 0) * K + k0 + c * 8, ok);
        }
        for (int i = tid; i < BN * CH; i += NT) {
            const int r = i / CH, c = i % CH, gn = n0 + r;
            const bool ok = gn < N;
            cp_async16(bs + r * (BK * 2) + ((c ^ tile_swz<CH>(r)) << 4), B + (size_t)(ok ? gn : 0) * K + k0 + c * 8, ok);
        }
    };

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_stage(s, s * BK);
        cp_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        if (EPI == EPI_BIAS_ADD_F32 && kt == pf_at) {
            constexpr int LINES = BN / 32;  // 128-byte lines of one fp32 tile row
            for (int i = tid; i < BM * LINES; i += NT) {
                const int gm = m0 + i / LINES, gn = n0 + (i % LINES) * 32;
                if (gm < M && gn < N)
                    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const float*>(C) + (size_t)gm * ldc + gn));
            }
        }
        cp_wait<STAGES - 2>();
        __syncthreads();  // stage kt has landed for everyone; everyone is past iteration kt - 1's reads of the slot refilled below
        if (kt + STAGES - 1 < KT) load_stage((kt + STAGES - 1) % STAGES, (kt + STAGES - 1) * BK);
        cp_commit();
        const int stage = kt % STAGES;
        const uint32_t as = a_base + stage * (BM * BK * 2), bs = b_base + stage * (BN * BK * 2);
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t af[MI][4], bf[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const int row = warp_m * Cfg::WTM + mi * 16 + (lane & 15);
                const int chunk = ks * 2 + (lane >> 4);
                ldsm_x4(af[mi], as + row * (BK * 2) + ((chunk ^ tile_swz<CH>(row)) << 4));
            }
#pragma unroll
            for (int nj = 0; nj < NI / 2; ++nj) {
                const int row = warp_n * Cfg::WTN + nj * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = ks * 2 + ((lane >> 3) & 1);
                uint32_t r[4];
                ldsm_x4(r, bs + row * (BK * 2) + ((chunk ^ tile_swz<CH>(row)) << 4));
                bf[2 * nj][0] = r[0];
                bf[2 * nj][1] = r[1];
                bf[2 * nj + 1][0] = r[2];
                bf[2 * nj + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int ni = 0; ni < NI; ++ni) mma16816(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
    }

    // epilogue. Column of fragment (ni, element pair): n0 + warp_n * WTN + ni * 8 + 2t; rows g and g + 8 of each m16 tile.
    float2 bv[NI];
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
        const int n = n0 + warp_n * Cfg::WTN + ni * 8 + 2 * t;
        bv[ni] = (bias != nullptr && n < N) ? *reinterpret_cast<const float2*>(bias + n) : make_float2(0.f, 0.f);
    }
    if (EpiOut<EPI>::f16) {
        cp_wait<0>();
        __syncthreads();  // the pipeline's shared memory is free: stage the fp16 tile [BM][BN + 8]
        half* cs = reinterpret_cast<half*>(smem_raw);
        constexpr int CS = BN + 8;
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) {
                const int r = warp_m * Cfg::WTM + mi * 16 + g, c = warp_n * Cfg::WTN + ni * 8 + 2 * t;
                *reinterpret_cast<uint32_t*>(cs + r * CS + c) =
                    pack_h2(epi_value<EPI>(acc[mi][ni][0], bv[ni].x), epi_value<EPI>(acc[mi][ni][1], bv[ni].y));
                *reinterpret_cast<uint32_t*>(cs + (r + 8) * CS + c) =
                    pack_h2(epi_value<EPI>(acc[mi][ni][2], bv[ni].x), epi_value<EPI>(acc[mi][ni][3], bv[ni].y));
            }
        __syncthreads();
        constexpr int CPR = BN / 8;  // 16-byte chunks a row
        for (int i = tid; i < BM * CPR; i += NT) {
            const int r = i / CPR, c = i % CPR;
            const int gm = m0 + r, gn = n0 + c * 8;
            if (gm < M && gn < N)
                *reinterpret_cast<uint4*>(reinterpret_cast<half*>(C) + (size_t)gm * ldc + gn) =
                    *reinterpret_cast<const uint4*>(cs + r * CS + c * 8);
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) {
                const int n = n0 + warp_n * Cfg::WTN + ni * 8 + 2 * t;
                if (n >= N) continue;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int m = m0 + warp_m * Cfg::WTM + mi * 16 + g + h * 8;
                    if (m >= M) continue;
                    float2* p = reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + (size_t)m * ldc + n);
                    const float v0 = epi_value<EPI>(acc[mi][ni][2 * h], bv[ni].x);
                    const float v1 = epi_value<EPI>(acc[mi][ni][2 * h + 1], bv[ni].y);
                    if (EPI == EPI_BIAS_ADD_F32) {
                        float2 o = *p;
                        o.x += v0;  // residual add, in place
                        o.y += v1;
                        *p = o;
                    } else {
                        *p = make_float2(v0, v1);
                    }
                }
            }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Attention. Scaled dot-product attention over all T tokens of one image, for every head. qkv fp16 [T, 3 * HID] (the fused qkv
// GEMM's output: q at column head * HD, k at HID + head * HD, v at 2 * HID + head * HD); out fp16 [T, HID].
// One CTA is NW warps = 16 * NW queries of one head; keys and values stream through shared memory in tiles of 64 (cp.async,
// STAGES buffers). Per warp: S = Q K^T (16 x 64, fp32, mma), the running max and sum in fp32 (log2 domain), P rounded to fp16
// and multiplied into the fp32 O accumulators (mma). Shared rows are padded to 88 halves (176 bytes) so that ldmatrix of 8
// consecutive rows is bank-conflict free without a swizzle (80 halves is 10 chunks, not a power of two).
// ---------------------------------------------------------------------------------------------------------------------

constexpr int AT_BN = 64;       // keys a tile
constexpr int AT_LD = HD + 8;   // shared row stride in halves

template <int NW, int STAGES>
constexpr int attn_smem_bytes() {
    return STAGES * 2 * AT_BN * AT_LD * 2;
}

template <int NW, int STAGES, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB)
    attention_mma_kernel(const half* __restrict__ qkv, half* __restrict__ out, int T, float scale_log2) {
    constexpr int NT = NW * 32;
    constexpr int LD3 = 3 * HID;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const uint32_t k_base = smem_u32(smem_raw);
    const uint32_t v_base = k_base + STAGES * AT_BN * AT_LD * 2;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int head = blockIdx.y;
    const half* qh = qkv + head * HD;
    const half* kh = qkv + HID + head * HD;
    const half* vh = qkv + 2 * HID + head * HD;
    const int q0 = blockIdx.x * (NW * 16) + warp * 16;
    const int r0 = q0 + g, r1 = q0 + g + 8;

    uint32_t qa[5][4];
#pragma unroll
    for (int ks = 0; ks < 5; ++ks) {
        const int c0 = ks * 16 + 2 * t, c1 = c0 + 8;
        qa[ks][0] = r0 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r0 * LD3 + c0) : 0u;
        qa[ks][1] = r1 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r1 * LD3 + c0) : 0u;
        qa[ks][2] = r0 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r0 * LD3 + c1) : 0u;
        qa[ks][3] = r1 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r1 * LD3 + c1) : 0u;
    }

    float o[10][4];
#pragma unroll
    for (int i = 0; i < 10; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int tiles = (T + AT_BN - 1) / AT_BN;
    auto load_tile = [&](int kt, int stage) {
        const uint32_t kb = k_base + stage * (AT_BN * AT_LD * 2), vb = v_base + stage * (AT_BN * AT_LD * 2);
        for (int i = tid; i < AT_BN * 10; i += NT) {
            const int key = i / 10, c = i % 10, gk = kt * AT_BN + key;
            const bool ok = gk < T;
            const size_t src = (size_t)(ok ? gk : 0) * LD3 + c * 8;
            const uint32_t off = key * (AT_LD * 2) + c * 16;
            cp_async16(kb + off, kh + src, ok);
            cp_async16(vb + off, vh + src, ok);
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < tiles) load_tile(s, s);
        cp_commit();
    }
    for (int kt = 0; kt < tiles; ++kt) {
        cp_wait<STAGES - 2>();
        __syncthreads();  // tile kt is resident; everyone is done with the slot refilled below
        if (kt + STAGES - 1 < tiles) load_tile(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
        cp_commit();
        const int stage = kt % STAGES;
        const uint32_t kb = k_base + stage * (AT_BN * AT_LD * 2), vb = v_base + stage * (AT_BN * AT_LD * 2);

        // The tile's 64 keys go through the online softmax as two halves of 32, which keeps S and P at 16 + 8 registers (a CTA of
        // 8 warps with two CTAs a SM has 128 registers a thread).
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            // S = Q K^T: 4 key n-tiles of 8, 5 k steps of 16 over the head dim; one ldmatrix gives two n-tiles' fragments
            float s[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int ks = 0; ks < 5; ++ks) {
#pragma unroll
                for (int jp = 0; jp < 2; ++jp) {
                    const int row = hf * 32 + jp * 16 + (lane & 7) + ((lane >> 4) << 3);
                    const int chunk = ks * 2 + ((lane >> 3) & 1);
                    uint32_t b[4];
                    ldsm_x4(b, kb + row * (AT_LD * 2) + chunk * 16);
                    mma16816(s[2 * jp], qa[ks], b[0], b[1]);
                    mma16816(s[2 * jp + 1], qa[ks], b[2], b[3]);
                }
            }
            // keys past T (the last tile only) take no part
            if (kt == tiles - 1 && (T & (AT_BN - 1)) != 0) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int key = kt * AT_BN + hf * 32 + nt * 8 + 2 * t;
                    if (key >= T) {
                        s[nt][0] = -INFINITY;
                        s[nt][2] = -INFINITY;
                    }
                    if (key + 1 >= T) {
                        s[nt][1] = -INFINITY;
                        s[nt][3] = -INFINITY;
                    }
                }
            }
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                tm0 = fmaxf(tm0, fmaxf(s[nt][0], s[nt][1]));
                tm1 = fmaxf(tm1, fmaxf(s[nt][2], s[nt][3]));
            }
#pragma unroll
            for (int o2 = 1; o2 <= 2; o2 <<= 1) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, o2));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, o2));
            }
            // running max in the scaled (log2) domain; the first half always holds a key, so n0 and n1 are finite
            const float n0 = fmaxf(m0, tm0 * scale_log2), n1 = fmaxf(m1, tm1 * scale_log2);
            const float c0 = fast_exp2(m0 - n0), c1 = fast_exp2(m1 - n1);  // exp2(-inf) = 0 on the first half
            m0 = n0;
            m1 = n1;
            l0 *= c0;
            l1 *= c1;
            // the O accumulators are rescaled only when some row's max moved (after the first tiles it almost never does)
            if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
#pragma unroll
                for (int i = 0; i < 10; ++i) {
                    o[i][0] *= c0;
                    o[i][1] *= c0;
                    o[i][2] *= c1;
                    o[i][3] *= c1;
                }
            }
            // P = exp2(s * scale - max), rounded to fp16 as the A operand of P.V (the sum l stays in fp32 on the unrounded values)
            uint32_t pk[4][2];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float p0 = fast_exp2(fmaf(s[nt][0], scale_log2, -n0));
                const float p1 = fast_exp2(fmaf(s[nt][1], scale_log2, -n0));
                const float p2 = fast_exp2(fmaf(s[nt][2], scale_log2, -n1));
                const float p3 = fast_exp2(fmaf(s[nt][3], scale_log2, -n1));
                l0 += p0 + p1;
                l1 += p2 + p3;
                pk[nt][0] = pack_h2(p0, p1);
                pk[nt][1] = pack_h2(p2, p3);
            }
            // O += P V: 2 k steps of 16 keys, 10 n-tiles of 8 head dims; V fragments by ldmatrix.trans (V is [key][d])
#pragma unroll
            for (int kc = 0; kc < 2; ++kc) {
                const uint32_t pa[4] = {pk[2 * kc][0], pk[2 * kc][1], pk[2 * kc + 1][0], pk[2 * kc + 1][1]};
                const int row = hf * 32 + kc * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
                for (int dp = 0; dp < 5; ++dp) {
                    const int chunk = dp * 2 + (lane >> 4);
                    uint32_t b[4];
                    ldsm_x4_t(b, vb + row * (AT_LD * 2) + chunk * 16);
                    mma16816(o[2 * dp], pa, b[0], b[1]);
                    mma16816(o[2 * dp + 1], pa, b[2], b[3]);
                }
            }
        }
    }
#pragma unroll
    for (int o2 = 1; o2 <= 2; o2 <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, o2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, o2);
    }
    const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
    for (int dn = 0; dn < 10; ++dn) {
        const int col = head * HD + dn * 8 + 2 * t;
        if (r0 < T) *reinterpret_cast<uint32_t*>(out + (size_t)r0 * HID + col) = pack_h2(o[dn][0] * i0, o[dn][1] * i0);
        if (r1 < T) *reinterpret_cast<uint32_t*>(out + (size_t)r1 * HID + col) = pack_h2(o[dn][2] * i1, o[dn][3] * i1);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Launch plumbing.
// ---------------------------------------------------------------------------------------------------------------------

template <typename K>
cudaError_t set_dyn_smem(K kernel, int bytes) {
    return bytes > 48 * 1024 ? cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) : cudaSuccess;
}

template <int EPI, int BM, int BN, int BK, int STAGES, int WM, int WN, int MINB>
int launch_gemm_mma(const half* a, const half* b, const float* bias, void* c, int M, int N, int K, int ldc, cudaStream_t s) {
    using Cfg = GemmCfg<BM, BN, BK, STAGES, WM, WN>;
    const dim3 block(Cfg::NT);
    const int nt = (N + BN - 1) / BN, mt = (M + BM - 1) / BM;
    auto kernel = gemm_mma_kernel<EPI, BM, BN, BK, STAGES, WM, WN, MINB>;
    static const cudaError_t configured = set_dyn_smem(kernel, Cfg::SMEM);
    if (configured != cudaSuccess) return (int)configured;
    const int group = mt < 8 ? mt : 8;
    kernel<<<dim3(mt * nt), block, Cfg::SMEM, s>>>(a, b, bias, c, M, N, K, ldc, mt, nt, group);
    return GB_CHECK_LAUNCH();
}

template <int EPI>
int launch_gemm(const half* a, const half* b, const float* bias, void* c, int M, int N, int K, int ldc, cudaStream_t s) {
    constexpr bool f16 = EpiOut<EPI>::f16;
    const bool fast = N % 8 == 0 && K % 32 == 0 && ldc % 8 == 0 && ((uintptr_t)c & (f16 ? 15 : 7)) == 0 &&
                      ((uintptr_t)a & 15) == 0 && ((uintptr_t)b & 15) == 0 && ((uintptr_t)bias & 7) == 0;
    if (fast) return launch_gemm_mma<EPI, 128, 128, 32, 3, 2, 4, 2>(a, b, bias, c, M, N, K, ldc, s);
    const dim3 grid((N + 63) / 64, (M + 63) / 64);
    gemm_simple_kernel<EPI><<<grid, THREADS, 0, s>>>(a, b, bias, c, M, N, K, ldc);
    return GB_CHECK_LAUNCH();
}

}  // namespace

// ----------------------------------------------------------------------------------------------------------------------------
// Launchers. Each returns 0 or a cudaError_t from the launch. Pointers are device addresses; `stream` is a cudaStream_t.
// ----------------------------------------------------------------------------------------------------------------------------

GB_EXPORT int gb_nvit_layernorm(const void* x, const void* w, const void* b, void* y, int rows, int dim, float eps,
                                void* stream) {
    if (rows < 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    if (dim == HID && ((uintptr_t)x & 15) == 0 && ((uintptr_t)w & 15) == 0 && ((uintptr_t)b & 15) == 0 && ((uintptr_t)y & 7) == 0) {
        layernorm_warp_kernel<<<(rows + LN_ROWS - 1) / LN_ROWS, LN_ROWS * 32, 0, gb_stream(stream)>>>(
            (const float*)x, (const float*)w, (const float*)b, (half*)y, rows, eps);
        return GB_CHECK_LAUNCH();
    }
    layernorm_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>((const float*)x, (const float*)w, (const float*)b, (half*)y, dim,
                                                              eps);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nvit_rmsnorm(const void* x, const void* w, void* y, int rows, int dim, float eps, void* stream) {
    if (rows < 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    rmsnorm_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>((const half*)x, (const float*)w, (half*)y, dim, eps);
    return GB_CHECK_LAUNCH();
}

// channels C = 3 (an image) or 6 (two stacked frames): pix [C, H, W] -> out [gh * gw, C * 256].
GB_EXPORT int gb_nvit_patchify_c(const void* pix, void* out, int H, int W, int gh, int gw, int C, void* stream) {
    if (H != gh * 16 || W != gw * 16 || gh <= 0 || gw <= 0 || C <= 0) return (int)cudaErrorInvalidValue;
    patchify_kernel<<<gh * gw, THREADS, 0, gb_stream(stream)>>>((const half*)pix, (half*)out, H, W, gw, C);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nvit_patchify(const void* pix, void* out, int H, int W, int gh, int gw, void* stream) {
    return gb_nvit_patchify_c(pix, out, H, W, gh, gw, 3, stream);
}

GB_EXPORT int gb_nvit_pos_embed(void* h, const void* pos, const void* cls, int gh, int gw, void* stream) {
    if (gh <= 0 || gw <= 0 || gh > POS_GRID * 4 || gw > POS_GRID * 4) return (int)cudaErrorInvalidValue;
    pos_embed_kernel<<<PREFIX + gh * gw, THREADS, 0, gb_stream(stream)>>>((float*)h, (const float*)pos, (const float*)cls, gh,
                                                                          gw);
    return GB_CHECK_LAUNCH();
}

// epi selects the epilogue (see the EPI_ enum). bias may be null. A is [M, K] fp16, B is [N, K] fp16, C is [M, N] at row
// stride ldc (fp16 for the EPI_*_F16 modes, fp32 for the others).
GB_EXPORT int gb_nvit_gemm(int epi, const void* A, const void* B, const void* bias, void* C, int M, int N, int K, int ldc,
                           void* stream) {
    if (M < 0 || N <= 0 || K <= 0 || ldc < N) return (int)cudaErrorInvalidValue;
    if (M == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    const half* a = (const half*)A;
    const half* b = (const half*)B;
    const float* bi = (const float*)bias;
    switch (epi) {
        case EPI_BIAS_F16: return launch_gemm<EPI_BIAS_F16>(a, b, bi, C, M, N, K, ldc, s);
        case EPI_BIAS_GELU_F16: return launch_gemm<EPI_BIAS_GELU_F16>(a, b, bi, C, M, N, K, ldc, s);
        case EPI_BIAS_ADD_F32: return launch_gemm<EPI_BIAS_ADD_F32>(a, b, bi, C, M, N, K, ldc, s);
        case EPI_STORE_F32: return launch_gemm<EPI_STORE_F32>(a, b, bi, C, M, N, K, ldc, s);
        case EPI_RELU2_F16: return launch_gemm<EPI_RELU2_F16>(a, b, bi, C, M, N, K, ldc, s);
        default: return (int)cudaErrorInvalidValue;
    }
}

GB_EXPORT int gb_nvit_pixel_shuffle(const void* in, void* out, int gh, int gw, void* stream) {
    if (gh <= 0 || gw <= 0 || (gh & 1) || (gw & 1)) return (int)cudaErrorInvalidValue;
    pixel_shuffle_kernel<<<(gh / 2) * (gw / 2), THREADS, 0, gb_stream(stream)>>>((const half*)in, (half*)out, gw);
    return GB_CHECK_LAUNCH();
}

// qkv fp16 [T, 3 * 1280] -> out fp16 [T, 1280]. Full attention among the T tokens of one image (prefix tokens included).
GB_EXPORT int gb_nvit_attention(const void* qkv, void* out, int T, void* stream) {
    if (T < 0) return (int)cudaErrorInvalidValue;
    if (T == 0) return 0;
    if (((uintptr_t)qkv & 15) != 0 || ((uintptr_t)out & 3) != 0) return (int)cudaErrorInvalidValue;
    constexpr int NW = 8, STAGES = 2;
    constexpr int SMEM = attn_smem_bytes<NW, STAGES>();
    auto kernel = attention_mma_kernel<NW, STAGES, 2>;
    static const cudaError_t configured = set_dyn_smem(kernel, SMEM);
    if (configured != cudaSuccess) return (int)configured;
    const float scale_log2 = 1.4426950408889634f / sqrtf((float)HD);
    kernel<<<dim3((T + NW * 16 - 1) / (NW * 16), HEADS), NW * 32, SMEM, gb_stream(stream)>>>((const half*)qkv, (half*)out, T,
                                                                                             scale_log2);
    return GB_CHECK_LAUNCH();
}
