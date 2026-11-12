// Kernels of the MiniCPM-V-4.7 vision path: the NaViT-style SigLIP tower (27 layers, hidden 1152, 16 heads of 72, gelu_tanh MLP
// 4304, LayerNorm with bias, eps 1e-6, no RoPE, a learned 70 x 70 position table read by nearest bucket), the 2 x 2 window
// merger after layer 6 (`vit_merger`: LN, window self-attention + residual, concat 4 x 1152, LN(4608), Linear 4608 -> 17216,
// gelu_tanh, Linear -> 1152, plus the 4-patch mean) and the projector (`merger.mlp.0`: 2 x 2 concat, LN(4608), Linear, exact
// GELU, Linear -> 2048). The reference is the HF `modeling_minicpmv4_7.py` (transformers 5.18) run in bf16.
//
// Conventions follow `nemotron_vit.cu` (this file does not depend on it: the helpers and the GEMM are copies): row-major, fp32
// residual stream, fp16 GEMM operands and activations with fp32 accumulation, LayerNorm and softmax statistics in fp32,
// weights in the checkpoint's [out, in] layout so that every GEMM is C[m, n] = A[m, k] * B[n, k]^T (the mma.sync "row.col"
// layout). Every kernel is deterministic (fixed reduction order, no atomics except the diagnostic absmax, no split-K).
//
// Row and crop invariance by construction: every GEMM output element accumulates over k alone in a fixed order whatever
// the rows around it are; LayerNorm is one row; attention reads only the keys of its own crop (a crop table, one grid z per
// crop) and the window attention only its own four tokens. A crop's rows therefore do not depend on which other crops share
// the launch (tests/mcv_vision.rs `a_crop_does_not_depend_on_the_crops_beside_it`).
//
// Shapes the fast GEMM path needs: N % 8 == 0, K % 32 == 0. The patch embedding (K = 588) pads K to 608 and the MLP (4304)
// pads to 4320 (135 x 32) with zero rows / columns of the weights (gelu_tanh(0 + 0) == 0, so the padded activations are exact zeros).

#include <stdint.h>

#include "common.cuh"

namespace {

constexpr int HID = 1152;       // hidden size of the tower, of every LayerNorm but two, and of the features
constexpr int HEADS = 16;       // attention heads
constexpr int HD = 72;          // head dim (HEADS * HD == HID)
constexpr int PATCH_DIM = 588;  // 3 * 14 * 14
constexpr int PATCH_K = 608;    // PATCH_DIM padded to a multiple of 32
constexpr int WIN_C = 4 * HID;  // concat of a 2 x 2 window (4608)
constexpr int THREADS = 256;

// ---------------------------------------------------------------------------------------------------------------------
// Tensor-core helpers.
// ---------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// m16n8k8: the last, half-width k step of the head dimension 72 = 4 * 16 + 8.
__device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0));
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
__device__ __forceinline__ void ldsm_x2_t(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
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

// y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * w + b over `dim` (biased variance, fp32 two-pass statistics). x fp32, y fp16.
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

// The same for dim == HID (1152): one warp a row, the row in registers (9 float4 a lane), one read of x. Eight rows a block.
// x must be 16-byte aligned (rows are 4,608 bytes).
constexpr int LN_ROWS = 8;
__global__ void __launch_bounds__(LN_ROWS * 32) layernorm_warp_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                                      const float* __restrict__ b, half* __restrict__ y,
                                                                      int rows, float eps) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * LN_ROWS + warp;
    if (row >= rows) return;
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * HID);
    float4 v[9];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        v[j] = xr[lane + 32 * j];
        s += (v[j].x + v[j].y) + (v[j].z + v[j].w);
    }
    const float mean = warp_sum(s) / (float)HID;
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        const float a = v[j].x - mean, c = v[j].y - mean, d = v[j].z - mean, e = v[j].w - mean;
        q += (a * a + c * c) + (d * d + e * e);
    }
    const float inv = rsqrtf(warp_sum(q) / (float)HID + eps);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    uint2* yr = reinterpret_cast<uint2*>(y + (size_t)row * HID);
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        const float4 wv = __ldg(w4 + lane + 32 * j), bv = __ldg(b4 + lane + 32 * j);
        uint2 o;
        o.x = pack_h2((v[j].x - mean) * inv * wv.x + bv.x, (v[j].y - mean) * inv * wv.y + bv.y);
        o.y = pack_h2((v[j].z - mean) * inv * wv.z + bv.z, (v[j].w - mean) * inv * wv.w + bv.w);
        yr[lane + 32 * j] = o;
    }
}

// The same LayerNorm over fp16 input (the projector's pre_norm reads the merged fp16 concat).
__global__ void __launch_bounds__(THREADS) layernorm_f16_kernel(const half* __restrict__ x, const float* __restrict__ w,
                                                                const float* __restrict__ b, half* __restrict__ y, int dim,
                                                                float eps) {
    __shared__ float sh[32];
    const half* xr = x + (size_t)blockIdx.x * dim;
    half* yr = y + (size_t)blockIdx.x * dim;
    float s = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) s += __half2float(xr[c]);
    const float mean = block_sum(s, sh) / (float)dim;
    float v = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const float d = __half2float(xr[c]) - mean;
        v += d * d;
    }
    const float inv = rsqrtf(block_sum(v, sh) / (float)dim + eps);
    for (int c = threadIdx.x; c < dim; c += blockDim.x)
        yr[c] = __float2half_rn((__half2float(xr[c]) - mean) * inv * w[c] + b[c]);
}

// Patch matrix [P, PATCH_K] fp16 from the packed pixel tensor the processor returns, bf16 bits [3, 14, W] with W = 14 * P:
// element [c, i, 14 * l + j] is channel c, row i, column j of patch l. The column order of a patch row is the conv weight's
// (channel, row, column); columns 588..607 are zero. bf16 -> fp16 is exact for the normalized pixels (|v| in [2^-8, 1]).
__global__ void __launch_bounds__(THREADS) patchify_kernel(const uint16_t* __restrict__ pix, half* __restrict__ out, int W) {
    const int p = blockIdx.x;
    for (int k = threadIdx.x; k < PATCH_K; k += blockDim.x) {
        float v = 0.f;
        if (k < PATCH_DIM) {
            const int c = k / 196, r = k % 196, i = r / 14, j = r % 14;
            v = __uint_as_float((uint32_t)pix[((size_t)c * 14 + i) * W + (size_t)p * 14 + j] << 16);
        }
        out[(size_t)p * PATCH_K + k] = __float2half_rn(v);
    }
}

// h[row, :] += pos[idx[row], :] (the nearest-bucket position embedding; the table is the checkpoint's bf16 widened to fp32).
__global__ void __launch_bounds__(THREADS) pos_add_kernel(float* __restrict__ h, const float* __restrict__ pos,
                                                          const int* __restrict__ idx) {
    const int row = blockIdx.x;
    const float* t = pos + (size_t)idx[row] * HID;
    float* o = h + (size_t)row * HID;
    for (int c = threadIdx.x; c < HID; c += blockDim.x) o[c] += t[c];
}

// Window reorder: hw[i, :] = h[src[i], :] (fp32 stream) and xw[i, :] = xn[src[i], :] (fp16 LayerNorm output).
__global__ void __launch_bounds__(THREADS) window_gather_kernel(const float* __restrict__ h, const half* __restrict__ xn,
                                                                const int* __restrict__ src, float* __restrict__ hw,
                                                                half* __restrict__ xw) {
    const int i = blockIdx.x, s = src[i];
    for (int c = threadIdx.x; c < HID; c += blockDim.x) {
        hw[(size_t)i * HID + c] = h[(size_t)s * HID + c];
        xw[(size_t)i * HID + c] = xn[(size_t)s * HID + c];
    }
}

// out[w, :] = (hw[4w] + hw[4w + 1] + hw[4w + 2] + hw[4w + 3]) / 4: the 4-patch mean of the window merger's residual.
__global__ void __launch_bounds__(THREADS) mean4_kernel(const float* __restrict__ hw, float* __restrict__ out) {
    const int w = blockIdx.x;
    const float* r = hw + (size_t)w * 4 * HID;
    for (int c = threadIdx.x; c < HID; c += blockDim.x)
        out[(size_t)w * HID + c] = (((r[c] + r[HID + c]) + r[2 * HID + c]) + r[3 * HID + c]) * 0.25f;
}

// 2 x 2 concat: out[o, k * HID + d] = in[src[4 * o + k], d] (fp16), the channel order (row in block, column in block, channel).
__global__ void __launch_bounds__(THREADS) merge4_kernel(const half* __restrict__ in, const int* __restrict__ src,
                                                         half* __restrict__ out) {
    const int o = blockIdx.x;
    for (int i = threadIdx.x; i < 4 * (HID / 8); i += blockDim.x) {
        const int k = i / (HID / 8), c = i % (HID / 8);
        reinterpret_cast<uint4*>(out + (size_t)o * WIN_C + k * HID)[c] =
            reinterpret_cast<const uint4*>(in + (size_t)src[4 * o + k] * HID)[c];
    }
}

// Window self-attention of the merger: windows of four tokens, 16 heads, bias-free here (the biases are in the GEMM). qkv fp16
// [4 * nwin, 3 * HID] (window-major rows), out fp16 [4 * nwin, HID]. One block per window, one warp a head; a lane owns the
// dims lane, lane + 32 and lane + 64 (< 72). fp32 throughout.
__global__ void __launch_bounds__(HEADS * 32) window_attention_kernel(const half* __restrict__ qkv, half* __restrict__ out,
                                                                      float scale) {
    const int win = blockIdx.x, head = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float q[4][3], k[4][3], v[4][3];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const half* row = qkv + (size_t)(win * 4 + t) * 3 * HID + head * HD;
#pragma unroll
        for (int e = 0; e < 3; ++e) {
            const int d = lane + 32 * e;
            const bool ok = d < HD;
            q[t][e] = ok ? __half2float(row[d]) : 0.f;
            k[t][e] = ok ? __half2float(row[HID + d]) : 0.f;
            v[t][e] = ok ? __half2float(row[2 * HID + d]) : 0.f;
        }
    }
    float s[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = q[i][0] * k[j][0] + q[i][1] * k[j][1] + q[i][2] * k[j][2];
            s[i][j] = warp_sum(a) * scale;
        }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float m = fmaxf(fmaxf(s[i][0], s[i][1]), fmaxf(s[i][2], s[i][3]));
        float p[4], l = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            p[j] = __expf(s[i][j] - m);
            l += p[j];
        }
        const float inv = 1.f / l;
#pragma unroll
        for (int e = 0; e < 3; ++e) {
            const int d = lane + 32 * e;
            if (d < HD) {
                const float o = (p[0] * v[0][e] + p[1] * v[1][e] + p[2] * v[2][e] + p[3] * v[3][e]) * inv;
                out[(size_t)(win * 4 + i) * HID + head * HD + d] = __float2half_rn(o);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// GEMM. C[M, N] = A[M, K] * B[N, K]^T with fp16 A and B, fp32 accumulation, and the epilogue applied once to each element.
// (A copy of nemotron_vit.cu's, with the tanh GELU added.)
// ---------------------------------------------------------------------------------------------------------------------

enum { EPI_BIAS_F16 = 0, EPI_BIAS_GELU_F16 = 1, EPI_BIAS_ADD_F32 = 2, EPI_STORE_F32 = 3, EPI_BIAS_GELU_TANH_F16 = 4 };

template <int EPI>
struct EpiOut {
    static constexpr bool f16 = EPI == EPI_BIAS_F16 || EPI == EPI_BIAS_GELU_F16 || EPI == EPI_BIAS_GELU_TANH_F16;
};

// v = acc + bias (bias may be absent) and the fp16 epilogues' nonlinearity, as one float.
template <int EPI>
__device__ __forceinline__ float epi_value(float acc, float bias) {
    const float v = acc + bias;
    if (EPI == EPI_BIAS_GELU_F16) return 0.5f * v * (1.f + erff(v * 0.7071067811865476f));  // exact GELU
    if (EPI == EPI_BIAS_GELU_TANH_F16) {  // gelu_pytorch_tanh
        const float t = tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v));
        return 0.5f * v * (1.f + t);
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
// the pipeline's shared memory for 16-byte coalesced stores; the fp32 ones store the fragments' float2 directly.
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
    // grouped tile order: `group` row tiles side by side, then the next column tile, so the CTAs resident together read a
    // compact patch of A and B
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

    // The in-place residual add reads its C tile in the epilogue; asking for the lines PF_AHEAD iterations before the end hides it
    // under the main loop.
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
// Attention. Scaled dot-product attention among the tokens of one crop, every head, one grid z a crop. qkv fp16 [T, 3 * HID]
// (the fused qkv GEMM's output: q at column head * HD, k at HID + head * HD, v at 2 * HID + head * HD); out fp16 [T, HID].
// crops[z] = (first row, rows) of crop z. One CTA is NW warps = 16 * NW queries of one head; keys and values stream through
// shared memory in tiles of 64 (cp.async, STAGES buffers). Per warp: S = Q K^T (16 x 64, fp32; the head dim 72 is four m16n8k16
// steps and one m16n8k8), the running max and sum in fp32 (log2 domain), P rounded to fp16 and multiplied into the fp32 O
// accumulators (nine n-tiles of 8). Shared rows are 144 bytes (9 chunks of 16): 36 words a row, 4 mod 8, so ldmatrix of 8
// consecutive rows is bank-conflict free without padding.
// ---------------------------------------------------------------------------------------------------------------------

constexpr int AT_BN = 64;  // keys a tile
constexpr int AT_LD = HD;  // shared row stride in halves

template <int NW, int STAGES>
constexpr int attn_smem_bytes() {
    return STAGES * 2 * AT_BN * AT_LD * 2;
}

template <int NW, int STAGES, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB)
    attention_mma_kernel(const half* __restrict__ qkv, half* __restrict__ out, const int2* __restrict__ crops, float scale_log2) {
    constexpr int NT = NW * 32;
    constexpr int LD3 = 3 * HID;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const uint32_t k_base = smem_u32(smem_raw);
    const uint32_t v_base = k_base + STAGES * AT_BN * AT_LD * 2;

    const int2 crop = crops[blockIdx.z];
    const int row0 = crop.x, T = crop.y;
    if ((int)blockIdx.x * (NW * 16) >= T) return;  // this crop has fewer query tiles than the grid

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int head = blockIdx.y;
    const half* qh = qkv + (size_t)row0 * LD3 + head * HD;
    const half* kh = qkv + (size_t)row0 * LD3 + HID + head * HD;
    const half* vh = qkv + (size_t)row0 * LD3 + 2 * HID + head * HD;
    half* oh = out + (size_t)row0 * HID + head * HD;
    const int q0 = blockIdx.x * (NW * 16) + warp * 16;
    const int r0 = q0 + g, r1 = q0 + g + 8;

    uint32_t qa[4][4], qa8[2];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int c0 = ks * 16 + 2 * t, c1 = c0 + 8;
        qa[ks][0] = r0 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r0 * LD3 + c0) : 0u;
        qa[ks][1] = r1 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r1 * LD3 + c0) : 0u;
        qa[ks][2] = r0 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r0 * LD3 + c1) : 0u;
        qa[ks][3] = r1 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r1 * LD3 + c1) : 0u;
    }
    qa8[0] = r0 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r0 * LD3 + 64 + 2 * t) : 0u;
    qa8[1] = r1 < T ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r1 * LD3 + 64 + 2 * t) : 0u;

    float o[9][4];
#pragma unroll
    for (int i = 0; i < 9; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int tiles = (T + AT_BN - 1) / AT_BN;
    auto load_tile = [&](int kt, int stage) {
        const uint32_t kb = k_base + stage * (AT_BN * AT_LD * 2), vb = v_base + stage * (AT_BN * AT_LD * 2);
        for (int i = tid; i < AT_BN * 9; i += NT) {
            const int key = i / 9, c = i % 9, gk = kt * AT_BN + key;
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

        // The tile's 64 keys go through the online softmax as two halves of 32 (S and P stay at 16 + 8 registers).
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            // S = Q K^T: 4 key n-tiles of 8; one ldmatrix gives two n-tiles' fragments of a 16-wide k step
            float s[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
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
            {   // the last 8 dims (chunk 8): one x4 gives the four key n-tiles of this half
                const int row = hf * 32 + lane;
                uint32_t b[4];
                ldsm_x4(b, kb + row * (AT_LD * 2) + 8 * 16);
                // matrix i of the x4 holds keys 8i..8i+7 (lanes 8i..8i+7 supplied their row addresses)
                mma1688(s[0], qa8[0], qa8[1], b[0]);
                mma1688(s[1], qa8[0], qa8[1], b[1]);
                mma1688(s[2], qa8[0], qa8[1], b[2]);
                mma1688(s[3], qa8[0], qa8[1], b[3]);
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
            // running max in the scaled (log2) domain. A key half past T has no key for these rows: tm is -inf, the max stays m
            // (finite after the first half, which always holds a key) and p = exp2(-inf) = 0.
            const float n0 = fmaxf(m0, tm0 * scale_log2), n1 = fmaxf(m1, tm1 * scale_log2);
            const float c0 = (n0 == -INFINITY) ? 1.f : fast_exp2(m0 - n0);
            const float c1 = (n1 == -INFINITY) ? 1.f : fast_exp2(m1 - n1);
            m0 = n0;
            m1 = n1;
            l0 *= c0;
            l1 *= c1;
            if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
#pragma unroll
                for (int i = 0; i < 9; ++i) {
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
                const float p0 = (n0 == -INFINITY) ? 0.f : fast_exp2(fmaf(s[nt][0], scale_log2, -n0));
                const float p1 = (n0 == -INFINITY) ? 0.f : fast_exp2(fmaf(s[nt][1], scale_log2, -n0));
                const float p2 = (n1 == -INFINITY) ? 0.f : fast_exp2(fmaf(s[nt][2], scale_log2, -n1));
                const float p3 = (n1 == -INFINITY) ? 0.f : fast_exp2(fmaf(s[nt][3], scale_log2, -n1));
                l0 += p0 + p1;
                l1 += p2 + p3;
                pk[nt][0] = pack_h2(p0, p1);
                pk[nt][1] = pack_h2(p2, p3);
            }
            // O += P V: 2 k steps of 16 keys; V fragments by ldmatrix.trans (V is [key][d]): 4 pairs of n-tiles and the ninth
#pragma unroll
            for (int kc = 0; kc < 2; ++kc) {
                const uint32_t pa[4] = {pk[2 * kc][0], pk[2 * kc][1], pk[2 * kc + 1][0], pk[2 * kc + 1][1]};
                const int row = hf * 32 + kc * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
                for (int dp = 0; dp < 4; ++dp) {
                    const int chunk = dp * 2 + (lane >> 4);
                    uint32_t b[4];
                    ldsm_x4_t(b, vb + row * (AT_LD * 2) + chunk * 16);
                    mma16816(o[2 * dp], pa, b[0], b[1]);
                    mma16816(o[2 * dp + 1], pa, b[2], b[3]);
                }
                uint32_t b2[2];
                ldsm_x2_t(b2, vb + row * (AT_LD * 2) + 8 * 16);
                mma16816(o[8], pa, b2[0], b2[1]);
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
    for (int dn = 0; dn < 9; ++dn) {
        const int col = dn * 8 + 2 * t;
        if (r0 < T) *reinterpret_cast<uint32_t*>(oh + (size_t)r0 * HID + col) = pack_h2(o[dn][0] * i0, o[dn][1] * i0);
        if (r1 < T) *reinterpret_cast<uint32_t*>(oh + (size_t)r1 * HID + col) = pack_h2(o[dn][2] * i1, o[dn][3] * i1);
    }
}

// max |x| of n floats or halves into `out` (a uint32 holding the float bits of a non-negative value; zero it first). Diagnostic
// only: the overflow check of the fp16 path (`gb_mcv_absmax`), not part of any forward.
template <bool F32>
__global__ void absmax_kernel(const void* __restrict__ x, size_t n, unsigned int* __restrict__ out) {
    float m = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = F32 ? ((const float*)x)[i] : __half2float(((const half*)x)[i]);
        // NaN compares false: count it as infinity so that it is never missed
        m = fmaxf(m, (v != v) ? INFINITY : fabsf(v));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(m));
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
    // Fewer 128 x 128 tiles than the SMs can hold two of (96 on the GB10's 48): a 64 x 64 tile makes four times the CTAs. Every
    // output element accumulates the same k steps in the same order in both, so the choice never changes a value.
    const int ctas128 = ((M + 127) / 128) * ((N + 127) / 128);
    if (fast && ctas128 < 96) return launch_gemm_mma<EPI, 64, 64, 32, 4, 2, 2, 4>(a, b, bias, c, M, N, K, ldc, s);
    if (fast) return launch_gemm_mma<EPI, 128, 128, 32, 3, 2, 4, 2>(a, b, bias, c, M, N, K, ldc, s);
    const dim3 grid((N + 63) / 64, (M + 63) / 64);
    gemm_simple_kernel<EPI><<<grid, THREADS, 0, s>>>(a, b, bias, c, M, N, K, ldc);
    return GB_CHECK_LAUNCH();
}

}  // namespace

// ----------------------------------------------------------------------------------------------------------------------------
// Launchers. Each returns 0 or a cudaError_t from the launch. Pointers are device addresses; `stream` is a cudaStream_t.
// ----------------------------------------------------------------------------------------------------------------------------

// y fp16 [rows, dim] = LayerNorm(x fp32 [rows, dim]) * w + b (fp32 w, b), fp32 statistics, biased variance.
GB_EXPORT int gb_mcv_layernorm(const void* x, const void* w, const void* b, void* y, int rows, int dim, float eps, void* stream) {
    if (rows < 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    if (dim == HID && ((uintptr_t)x & 15) == 0 && ((uintptr_t)w & 15) == 0 && ((uintptr_t)b & 15) == 0 && ((uintptr_t)y & 7) == 0) {
        layernorm_warp_kernel<<<(rows + LN_ROWS - 1) / LN_ROWS, LN_ROWS * 32, 0, gb_stream(stream)>>>(
            (const float*)x, (const float*)w, (const float*)b, (half*)y, rows, eps);
        return GB_CHECK_LAUNCH();
    }
    layernorm_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>((const float*)x, (const float*)w, (const float*)b, (half*)y, dim, eps);
    return GB_CHECK_LAUNCH();
}

// The same over fp16 x (the projector's pre_norm).
GB_EXPORT int gb_mcv_layernorm_f16(const void* x, const void* w, const void* b, void* y, int rows, int dim, float eps,
                                   void* stream) {
    if (rows < 0 || dim <= 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    layernorm_f16_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>((const half*)x, (const float*)w, (const float*)b, (half*)y, dim,
                                                                  eps);
    return GB_CHECK_LAUNCH();
}

// pix bf16 bits [3, 14, 14 * patches] -> out fp16 [patches, 608] (patch vector in (channel, row, column) order, zero padded).
GB_EXPORT int gb_mcv_patchify(const void* pix, void* out, int patches, void* stream) {
    if (patches < 0) return (int)cudaErrorInvalidValue;
    if (patches == 0) return 0;
    patchify_kernel<<<patches, THREADS, 0, gb_stream(stream)>>>((const uint16_t*)pix, (half*)out, 14 * patches);
    return GB_CHECK_LAUNCH();
}

// h fp32 [rows, 1152] += pos fp32 [*, 1152][idx[row]].
GB_EXPORT int gb_mcv_pos_add(void* h, const void* pos, const void* idx, int rows, void* stream) {
    if (rows < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    pos_add_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>((float*)h, (const float*)pos, (const int*)idx);
    return GB_CHECK_LAUNCH();
}

// hw[i] = h[src[i]] (fp32), xw[i] = xn[src[i]] (fp16), rows of 1152.
GB_EXPORT int gb_mcv_window_gather(const void* h, const void* xn, const void* src, void* hw, void* xw, int rows, void* stream) {
    if (rows < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    window_gather_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>((const float*)h, (const half*)xn, (const int*)src, (float*)hw,
                                                                   (half*)xw);
    return GB_CHECK_LAUNCH();
}

// out fp32 [nwin, 1152] = mean of the four rows of each window of hw fp32 [4 * nwin, 1152].
GB_EXPORT int gb_mcv_mean4(const void* hw, void* out, int nwin, void* stream) {
    if (nwin < 0) return (int)cudaErrorInvalidValue;
    if (nwin == 0) return 0;
    mean4_kernel<<<nwin, THREADS, 0, gb_stream(stream)>>>((const float*)hw, (float*)out);
    return GB_CHECK_LAUNCH();
}

// out fp16 [n, 4608] = concat over k of in fp16 [*, 1152] row src[4n + k].
GB_EXPORT int gb_mcv_merge4(const void* in, const void* src, void* out, int n, void* stream) {
    if (n < 0) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;
    merge4_kernel<<<n, THREADS, 0, gb_stream(stream)>>>((const half*)in, (const int*)src, (half*)out);
    return GB_CHECK_LAUNCH();
}

// qkv fp16 [4 * nwin, 3456] -> out fp16 [4 * nwin, 1152]: attention among the four tokens of each window.
GB_EXPORT int gb_mcv_window_attention(const void* qkv, void* out, int nwin, void* stream) {
    if (nwin < 0) return (int)cudaErrorInvalidValue;
    if (nwin == 0) return 0;
    window_attention_kernel<<<nwin, HEADS * 32, 0, gb_stream(stream)>>>((const half*)qkv, (half*)out,
                                                                        1.0f / sqrtf((float)HD));
    return GB_CHECK_LAUNCH();
}

// epi selects the epilogue (see the EPI_ enum). bias may be null. A is [M, K] fp16, B is [N, K] fp16, C is [M, N] at row
// stride ldc (fp16 for the EPI_*_F16 modes, fp32 for the others).
GB_EXPORT int gb_mcv_gemm(int epi, const void* A, const void* B, const void* bias, void* C, int M, int N, int K, int ldc,
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
        case EPI_BIAS_GELU_TANH_F16: return launch_gemm<EPI_BIAS_GELU_TANH_F16>(a, b, bi, C, M, N, K, ldc, s);
        default: return (int)cudaErrorInvalidValue;
    }
}

// qkv fp16 [rows, 3 * 1152] -> out fp16 [rows, 1152]. crops: device int2 [ncrops] = (first row, rows); max_rows is the largest
// `rows` of any crop (it sizes the grid). Attention never crosses a crop.
GB_EXPORT int gb_mcv_attention(const void* qkv, void* out, const void* crops, int ncrops, int max_rows, void* stream) {
    if (ncrops < 0 || max_rows < 0) return (int)cudaErrorInvalidValue;
    if (ncrops == 0 || max_rows == 0) return 0;
    if (((uintptr_t)qkv & 15) != 0 || ((uintptr_t)out & 3) != 0) return (int)cudaErrorInvalidValue;
    constexpr int NW = 8, STAGES = 2;
    constexpr int SMEM = attn_smem_bytes<NW, STAGES>();
    auto kernel = attention_mma_kernel<NW, STAGES, 2>;
    static const cudaError_t configured = set_dyn_smem(kernel, SMEM);
    if (configured != cudaSuccess) return (int)configured;
    const float scale_log2 = 1.4426950408889634f / sqrtf((float)HD);
    kernel<<<dim3((max_rows + NW * 16 - 1) / (NW * 16), HEADS, ncrops), NW * 32, SMEM, gb_stream(stream)>>>(
        (const half*)qkv, (half*)out, (const int2*)crops, scale_log2);
    return GB_CHECK_LAUNCH();
}

// max |x| over n fp32 (is_f32) or fp16 values into the uint32 `out` (the float bits of a non-negative value; the caller zeroes it).
GB_EXPORT int gb_mcv_absmax(const void* x, int is_f32, size_t n, void* out, void* stream) {
    if (n == 0) return 0;
    const unsigned blocks = (unsigned)((n + 255) / 256 < 1024 ? (n + 255) / 256 : 1024);
    if (is_f32)
        absmax_kernel<true><<<blocks, 256, 0, gb_stream(stream)>>>(x, n, (unsigned int*)out);
    else
        absmax_kernel<false><<<blocks, 256, 0, gb_stream(stream)>>>(x, n, (unsigned int*)out);
    return GB_CHECK_LAUNCH();
}
