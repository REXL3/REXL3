// DeepSeek-V4.1-Flash vision tower: the kernels behind src/dsv4_vision.rs.
//
// The reference is /tmp/ds41ref/vision.py. Every kernel here is a stage of `ViT.forward` / `Block.forward` /
// `Aligner.forward` and nothing else: there is no learned position table (positions enter only through the 2-D RoPE of q
// and k), no attention mask (the tower is bidirectional over one image), and no dropout.
//
// Conventions, matching the rest of csrc/:
//   * every launcher launches on `s` and returns without synchronising;
//   * a short buffer is a host-side assert in the Rust wrapper, not a device check;
//   * the residual stream is fp32 ([P, 1024]); GEMM operands and their outputs are fp16 with fp32 accumulation. That is
//     the reference's own split: its `x` is a bf16 tensor and `x.float()`/`.to(dtype)` bracket every norm, so the norms
//     are f32 and the matmuls are bf16 x bf16 -> bf16. This build holds the stream in f32 and narrows at each GEMM
//     operand, which is the same arithmetic with a wider stream (the pack is bf16 and is converted to f16 once, at load).
//
// K in the patch embedding is 588 = 3 * 14 * 14, which is not a multiple of 8, so `gemm_k` predicates its last K tile
// instead of the caller padding the pack's [1024, 588] weight.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define GB_CHECK(rc)                                                                                                   \
    do {                                                                                                               \
        cudaError_t e_ = (rc);                                                                                          \
        if (e_ != cudaSuccess) return (int)e_;                                                                          \
    } while (0)

namespace {

__device__ __forceinline__ float h2f(const __half h) { return __half2float(h); }
__device__ __forceinline__ __half f2h(const float f) { return __float2half_rn(f); }

// -------------------------------------------------------------------------------------------------------------------
// conversions and elementwise
// -------------------------------------------------------------------------------------------------------------------

// bf16 bits -> f16 in place; a value that is not finite in f16 is counted in `bad` and written as 0.
__global__ void bf16_to_f16_k(uint16_t* p, size_t n, int* bad) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&p[i]));
        if (!isfinite(v) || fabsf(v) > 65504.0f) {
            atomicAdd(bad, 1);
            p[i] = __half_as_ushort(f2h(0.0f));
        } else {
            p[i] = __half_as_ushort(f2h(v));
        }
    }
}

// fp32 x [rows, dim] -> fp16 y [rows, dim] = RMSNorm(x) * w. The stream is fp32 (the residual), the norm's output feeds
// a GEMM operand and the aligner gather, both fp16 — the narrow happens here, once, exactly where the reference's
// `RMSNorm.forward` does `.to(dtype)` after computing in f32.
__global__ void rmsnorm_k(const float* __restrict__ x, const float* __restrict__ w, __half* __restrict__ y, int dim,
                          float eps) {
    const int r = blockIdx.x;
    __shared__ float s;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = x[(size_t)r * dim + i];
        acc += v * v;
    }
    // block reduction: every step ADDS the shuffled partial; the first step must not overwrite the lane's own (a `=`
    // there sums only the warp's upper half — a sqrt(2) scale error on every row that random-weight end-to-end tests
    // hide inside small sublayer outputs)
    __shared__ float red[32];
    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; ++i) t += red[i];
        s = rsqrtf(t / (float)dim + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        y[(size_t)r * dim + i] = f2h(w[i] * (x[(size_t)r * dim + i] * s));
}

// fp32 [rows, n] += fp16 bias [n]
__global__ void bias_add_f32_k(float* __restrict__ x, const __half* __restrict__ bias, int n, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x)
        x[i] += h2f(bias[i % (size_t)n]);
}

// fp16 [rows, n] += fp16 bias [n]
__global__ void bias_add_f16_k(__half* __restrict__ x, const __half* __restrict__ bias, int n, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x)
        x[i] = f2h(h2f(x[i]) + h2f(bias[i % (size_t)n]));
}

// gu[.., :inter] = silu(gu[.., :inter]) * gu[.., inter:2*inter], in place, row stride `ld`.
__global__ void silu_mul_k(__half* __restrict__ gu, int inter, int ld, int rows) {
    const int r = blockIdx.y;
    if (r >= rows) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inter; i += gridDim.x * blockDim.x) {
        const float g = h2f(gu[(size_t)r * ld + i]);
        const float u = h2f(gu[(size_t)r * ld + inter + i]);
        gu[(size_t)r * ld + i] = f2h((g / (1.0f + __expf(-g))) * u);
    }
}

// gelu_erf, the `F.gelu` default
__global__ void gelu_erf_k(__half* __restrict__ x, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = h2f(x[i]);
        x[i] = f2h(0.5f * v * (1.0f + erff(v * 0.70710678118654752440f)));
    }
}

// -------------------------------------------------------------------------------------------------------------------
// gemm: C[m, n] = A[m, k] * B[n, k]^T, fp16 operands, fp32 accumulate. Arbitrary k (the patch embedding's 588).
// -------------------------------------------------------------------------------------------------------------------

constexpr int GBM = 64, GBN = 64, GBK = 16;

__device__ __forceinline__ void gemm_store(int epi, float acc, const __half* __restrict__ bias, int col,
                                           __half* __restrict__ ch, float* __restrict__ cf) {
    const float b = bias ? h2f(bias[col]) : 0.0f;
    switch (epi) {
        case 0: ch[col] = f2h(acc + b); break;
        case 1: cf[col] = acc + b; break;
        case 2: cf[col] += acc + b; break;
        default: {
            const float v = acc + b;
            ch[col] = f2h(0.5f * v * (1.0f + erff(v * 0.70710678118654752440f)));
        }
    }
}

__global__ void __launch_bounds__(256)
    gemm_k(int epi, const __half* __restrict__ a, const __half* __restrict__ b, const __half* __restrict__ bias,
           void* __restrict__ c, int m, int n, int k, int lda, int ldc) {
    __shared__ __align__(16) __half As[GBM][GBK + 4];
    __shared__ __align__(16) __half Bs[GBN][GBK + 4];

    const int row0 = blockIdx.y * GBM;
    const int col0 = blockIdx.x * GBN;
    const int tx = threadIdx.x & 15;   // 16 columns of the tile
    const int ty = threadIdx.x >> 4;   // 16 rows of the tile

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < k; k0 += GBK) {
        // A tile [GBM][GBK] and B tile [GBN][GBK]: 1024 values each, 4 per thread, index = r * GBK + c. The A tile's
        // rows come from the row block (`row0`) at row stride `lda` (the MLP keeps its activation in the first half of
        // a 2*INTER-strided row), the B tile's from the COLUMN block (`col0`) — b is [n, k], so a tile computing output
        // columns col0..col0+GBN reads b's rows col0..col0+GBN. Loading b from `row0` instead reads valid but unrelated
        // weight rows on every off-diagonal tile: finite, deterministic garbage.
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int idx = threadIdx.x + t * 256;
            const int r = idx / GBK, cc = idx % GBK;
            const int gc = k0 + cc;
            const int ar = row0 + r, br = col0 + r;
            As[r][cc] = (ar < m && gc < k) ? a[(size_t)ar * lda + gc] : f2h(0.0f);
            Bs[r][cc] = (br < n && gc < k) ? b[(size_t)br * k + gc] : f2h(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < GBK; ++kk) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float av = h2f(As[ty + i * 16][kk]);
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av * h2f(Bs[tx + j * 16][kk]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gr = row0 + ty + i * 16;
        if (gr >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gc = col0 + tx + j * 16;
            if (gc >= n) continue;
            if (epi <= 0 || epi == 3) {
                gemm_store(epi, acc[i][j], bias, gc, (__half*)c + (size_t)gr * ldc, nullptr);
            } else {
                gemm_store(epi, acc[i][j], bias, gc, nullptr, (float*)c + (size_t)gr * ldc);
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------
// the 2-D rope: `stack([hpos, wpos], -1).reshape(-1, 2, 1) * inv_freq` flattened, so the first half of a head's channels
// turns by the row angle and the second half by the column angle.
// -------------------------------------------------------------------------------------------------------------------

__global__ void rope_table_k(float* __restrict__ ang, int n_w, int dim, float theta) {
    const int half = dim / 2;
    // one row per (h, w); `half` frequencies per axis, the row angle in the first half and the column angle in the second
    const int p = blockIdx.x;
    const int h = p / n_w, w = p % n_w;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        const int axis = j / half, f = j % half;
        const float inv = 1.0f / powf(theta, (float)(2 * f) / (float)dim);
        ang[(size_t)p * dim + j] = (axis == 0 ? (float)h : (float)w) * inv;
    }
}

// One angle per channel of the first `rope` channels of a head: `apply_rotary` chunks the head in half and pairs `c` with
// `rope + c`, both turned by `ang[c]`.
__global__ void rope_k(__half* __restrict__ qkv, const float* __restrict__ ang, int heads, int rope) {
    const int p = blockIdx.x;
    const int dim = 64;  // HEAD_DIM
    const int stride = 3 * heads * dim;
    for (int t = threadIdx.x; t < 2 * heads * rope; t += blockDim.x) {
        // t -> (which of q|k, head, channel)
        const int which = t / (heads * rope);
        const int rem = t % (heads * rope);
        const int h = rem / rope, j = rem % rope;
        const float a = ang[(size_t)p * rope + j];
        const float c = __cosf(a), s = __sinf(a);
        __half* base = qkv + (size_t)p * stride + (size_t)which * heads * dim + (size_t)h * dim;
        const float x1 = h2f(base[j]);
        const float x2 = h2f(base[rope + j]);
        base[j] = f2h(x1 * c - x2 * s);
        base[rope + j] = f2h(x2 * c + x1 * s);
    }
}

// -------------------------------------------------------------------------------------------------------------------
// bidirectional attention over the P patches of one image, head dim 64. One warp per query row, K/V streamed through
// shared memory in tiles of 64 keys, online softmax in fp32.
// -------------------------------------------------------------------------------------------------------------------

constexpr int ATTN_KS = 64 + 8;

__global__ void __launch_bounds__(128)
    attention_k(const __half* __restrict__ qkv, __half* __restrict__ out, int p) {
    __shared__ __align__(16) __half Ks[64][ATTN_KS];
    __shared__ __align__(16) __half Vs[64][ATTN_KS];

    const int head = blockIdx.y;
    const int warp = threadIdx.x >> 5;          // 4 warps
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;      // one query row per warp
    const bool live = row < p;                  // every thread must reach __syncthreads, so no early return

    const int D = 64;
    const int stride = 3 * 16 * D;
    const __half* q = qkv + (size_t)(live ? row : 0) * stride + (size_t)head * D;

    // this lane's two channels
    float q0 = h2f(q[lane]);
    float q1 = h2f(q[lane + 32]);
    float acc0 = 0.0f, acc1 = 0.0f, m = -INFINITY, l = 0.0f;

    const float scale = 1.0f / 8.0f;  // 1 / sqrt(64)
    for (int j0 = 0; j0 < p; j0 += 64) {
        __syncthreads();
        for (int t = threadIdx.x; t < 64 * 64; t += blockDim.x) {
            const int j = t >> 6, c = t & 63;
            const int gj = j0 + j;
            // K lives in the middle third of each patch's qkv row (q | k | v), V in the last: +16*D and +2*16*D.
            // Reading K from the q section scores the queries against themselves.
            Ks[j][c] = (gj < p) ? qkv[(size_t)gj * stride + 16 * D + (size_t)head * D + c] : f2h(0.0f);
            Vs[j][c] = (gj < p) ? qkv[(size_t)gj * stride + 2 * 16 * D + (size_t)head * D + c] : f2h(0.0f);
        }
        __syncthreads();
        const int jn = min(64, p - j0);
        for (int j = 0; j < jn; ++j) {
            float s = q0 * h2f(Ks[j][lane]) + q1 * h2f(Ks[j][lane + 32]);
            s += __shfl_xor_sync(0xffffffffu, s, 16);
            s += __shfl_xor_sync(0xffffffffu, s, 8);
            s += __shfl_xor_sync(0xffffffffu, s, 4);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s *= scale;
            const float mn = fmaxf(m, s);
            const float e = __expf(s - mn);
            const float corr = __expf(m - mn);
            acc0 = acc0 * corr + e * h2f(Vs[j][lane]);
            acc1 = acc1 * corr + e * h2f(Vs[j][lane + 32]);
            l = l * corr + e;
            m = mn;
        }
    }
    const float inv = 1.0f / l;
    if (!live) return;
    __half* o = out + (size_t)row * (16 * D) + (size_t)head * D;
    o[lane] = f2h(acc0 * inv);
    o[lane + 32] = f2h(acc1 * inv);
}

// -------------------------------------------------------------------------------------------------------------------
// the aligner's F.pad + F.unfold(3, stride=3): [n_h * n_w, dim] -> [oh * ow, dim * 9], window (C, kH, kW) with C outer.
// -------------------------------------------------------------------------------------------------------------------

__global__ void aligner_gather_k(const __half* __restrict__ x, __half* __restrict__ out, int n_h, int n_w, int ow,
                                 int dim) {
    const int r = blockIdx.x;  // output row = (oy, ox)
    const int oy = r / ow, ox = r % ow;
    __half* dst = out + (size_t)r * dim * 9;
    for (int t = threadIdx.x; t < dim * 9; t += blockDim.x) {
        const int c = t / 9, k = t % 9;
        const int kh = k / 3, kw = k % 3;
        const int y = oy * 3 + kh, xx = ox * 3 + kw;
        dst[t] = (y < n_h && xx < n_w) ? x[((size_t)y * n_w + xx) * dim + c] : f2h(0.0f);
    }
}

// -------------------------------------------------------------------------------------------------------------------
// the span splice: [START] ([IMAGE] * w [NEW_LINE]) * h [END] over the text stream's rows.
// -------------------------------------------------------------------------------------------------------------------

__global__ void splice_k(__half* __restrict__ dst, const __half* __restrict__ feats, int start, int n_llm_h, int n_llm_w,
                         int dim, const __half* __restrict__ start_emb, const __half* __restrict__ end_emb,
                         const __half* __restrict__ nl_emb) {
    const int t = blockIdx.x;  // span offset
    // where this offset is in the layout
    const __half* src;
    int span = 1 + n_llm_h * (n_llm_w + 1);  // first offset past the IMAGE/NEW_LINE grid
    if (t == 0) {
        src = start_emb;
    } else if (t == span) {
        src = end_emb;
    } else {
        const int i = t - 1;
        const int row = i / (n_llm_w + 1);
        const int col = i % (n_llm_w + 1);
        src = (col < n_llm_w) ? feats + ((size_t)row * n_llm_w + col) * dim : nl_emb;
    }
    __half* d = dst + ((size_t)start + t) * dim;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) d[c] = src[c];
}

}  // namespace

// -------------------------------------------------------------------------------------------------------------------
// launchers
// -------------------------------------------------------------------------------------------------------------------

// grid size for a flat `n`-element kernel, capped at the 65535 blocks a grid may have on this arch
static int blocks_for(size_t n, size_t per_block, size_t cap) {
    const size_t b = (n + per_block - 1) / per_block;
    return (int)(b < cap ? b : cap);
}

extern "C" int gb_dsv4vis_bf16_to_f16(void* p, size_t n, int* bad, cudaStream_t s) {
    if (n == 0) return 0;
    const int blocks = blocks_for(n, 256, 65535);
    GB_CHECK(cudaMemsetAsync(bad, 0, sizeof(int), s));
    bf16_to_f16_k<<<blocks, 256, 0, s>>>((uint16_t*)p, n, bad);
    return 0;
}

extern "C" int gb_dsv4vis_gemm(int epi, const void* a, const void* b, const void* bias, void* c, int m, int n, int k,
                               int lda, int ldc, cudaStream_t s) {
    if (m == 0 || n == 0 || k == 0) return 0;
    const dim3 grid((unsigned)((n + GBN - 1) / GBN), (unsigned)((m + GBM - 1) / GBM));
    gemm_k<<<grid, 256, 0, s>>>(epi, (const __half*)a, (const __half*)b, (const __half*)bias, c, m, n, k, lda, ldc);
    return 0;
}

extern "C" int gb_dsv4vis_rmsnorm(const void* x, const void* w, void* y, int rows, int dim, float eps, cudaStream_t s) {
    if (rows == 0) return 0;
    rmsnorm_k<<<rows, 256, 0, s>>>((const float*)x, (const float*)w, (__half*)y, dim, eps);
    return 0;
}

extern "C" int gb_dsv4vis_bias_add_f32(void* x, const void* bias, int n, size_t total, cudaStream_t s) {
    if (total == 0) return 0;
    const int blocks = blocks_for(total, 256, 65535);
    bias_add_f32_k<<<blocks, 256, 0, s>>>((float*)x, (const __half*)bias, n, total);
    return 0;
}

extern "C" int gb_dsv4vis_bias_add_f16(void* x, const void* bias, int n, size_t total, cudaStream_t s) {
    if (total == 0) return 0;
    const int blocks = blocks_for(total, 256, 65535);
    bias_add_f16_k<<<blocks, 256, 0, s>>>((__half*)x, (const __half*)bias, n, total);
    return 0;
}

extern "C" int gb_dsv4vis_rope_table(void* ang, int n_h, int n_w, int dim, float theta, cudaStream_t s) {
    const int p = n_h * n_w;
    if (p == 0) return 0;
    rope_table_k<<<p, 128, 0, s>>>((float*)ang, n_w, dim, theta);
    return 0;
}

extern "C" int gb_dsv4vis_rope(void* qkv, const void* ang, int rows, int n_w, int heads, cudaStream_t s) {
    if (rows == 0) return 0;
    (void)n_w;
    rope_k<<<rows, 128, 0, s>>>((__half*)qkv, (const float*)ang, heads, 32);
    return 0;
}

extern "C" int gb_dsv4vis_attention(const void* qkv, void* out, int p, cudaStream_t s) {
    if (p == 0) return 0;
    const dim3 grid((unsigned)((p + 3) / 4), 16);  // 16 = HEADS, one block.y per head
    attention_k<<<grid, 128, 0, s>>>((const __half*)qkv, (__half*)out, p);
    return 0;
}

extern "C" int gb_dsv4vis_silu_mul(void* gu, int inter, int ld, int rows, cudaStream_t s) {
    if (rows == 0 || inter == 0) return 0;
    const int blocks = blocks_for((size_t)inter, 256, 4096);
    silu_mul_k<<<dim3((unsigned)blocks, (unsigned)rows), 256, 0, s>>>((__half*)gu, inter, ld, rows);
    return 0;
}

extern "C" int gb_dsv4vis_gelu_erf(void* x, size_t n, cudaStream_t s) {
    if (n == 0) return 0;
    const int blocks = blocks_for(n, 256, 65535);
    gelu_erf_k<<<blocks, 256, 0, s>>>((__half*)x, n);
    return 0;
}

extern "C" int gb_dsv4vis_aligner_gather(const void* x, void* out, int n_h, int n_w, int dim, cudaStream_t s) {
    const int oh = (n_h + 2) / 3, ow = (n_w + 2) / 3;
    if (oh == 0 || ow == 0) return 0;
    aligner_gather_k<<<oh * ow, 256, 0, s>>>((const __half*)x, (__half*)out, n_h, n_w, ow, dim);
    return 0;
}

extern "C" int gb_dsv4vis_splice(void* dst, const void* feats, int start, int n_llm_h, int n_llm_w, int dim,
                                 const void* start_emb, const void* end_emb, const void* nl_emb, cudaStream_t s) {
    const int n = 1 + n_llm_h * (n_llm_w + 1) + 1;
    if (n <= 2) return 0;
    splice_k<<<n, 256, 0, s>>>((__half*)dst, (const __half*)feats, start, n_llm_h, n_llm_w, dim,
                               (const __half*)start_emb, (const __half*)end_emb, (const __half*)nl_emb);
    return 0;
}