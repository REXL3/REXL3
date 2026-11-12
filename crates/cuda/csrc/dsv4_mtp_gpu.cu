// gb-nvcc: --fmad=false
// The DSpark / MTP draft chain's device pieces that have no trunk counterpart: the block embedding, the
// hyper-connection mixer at an arbitrary width, the bidirectional block attention over `window ++ block`, the markov
// walk's argmax, and the MXFP4 routed-expert pair.
//
// ---------------------------------------------------------------------------------------------------------------
// WHY EACH OF THESE IS NEW
//
//   embed       The trunk's embedding is read by the residual stream; here it seeds `block_size` draft rows.
//   hc_*        `dsv4_moe.cu`'s mixer is the 5120/20480/40-atom instantiation. The toy config is 128 wide, so this
//               file carries the same arithmetic over (streams, dim) at run time. Every formula, epsilon and the
//               Sinkhorn iteration count are the ones in `Block.hc_mixes` / `hc_pre` / `hc_post`.
//   attn        `DSparkAttention.forward`: ONE kv source (the trunk tap) in a ring of `window`, plus the block's own
//               `block_size` rows appended; every query row attends to the SAME slot list (the whole block, not a
//               causal prefix -- `get_dspark_topk_idxs` broadcasts one row), plus the head's `attn_sink` logit.
//   moe_*       Routed experts as packed MXFP4 with an E8M0 scale a row per 32 inputs. Grid is (n tile, expert): the
//               expert loop lives on the device, so a draft round issues a fixed number of launches whatever the
//               routing was and the whole chain is CUDA-graph capturable.
//
// Everything numeric that the trunk already does (RMSNorm, RoPE, the E4M3 block quantise-dequantise, the fp8 GEMM,
// swiglu, group/gather/scatter) is called, not re-derived.
#include "common.cuh"
#include "keyed.cuh"
#include "dsv4_round_params.cuh"
#include <cuda_fp8.h>
#include <math.h>

namespace {

typedef uint16_t u16;
typedef uint32_t u32;

constexpr int ERR_INVALID = 1;  // cudaErrorInvalidValue
constexpr int NS = 4;           // hc streams
constexpr int NM = 24;          // hc fn rows: 4 pre, 4 post, 16 comb
constexpr int NQ = NM + 1;      // + the sum of squares
constexpr int TILE = 32;
constexpr int THREADS = 256;
constexpr int MAX_KV = 160;  // window 128 + block 5 with room; checked by the launcher
constexpr int WARPS_A = 4;   // heads a block of the attention kernel runs

enum { T_F16 = 0, T_BF16 = 1, T_F32 = 2 };

__device__ __forceinline__ float bf2f(u16 b) { return __uint_as_float((u32)b << 16); }

template <int T>
__device__ __forceinline__ float ldx(const void* p, size_t i) {
    if constexpr (T == T_F32) {
        return ((const float*)p)[i];
    } else if constexpr (T == T_F16) {
        return __half2float(((const __half*)p)[i]);
    } else {
        return bf2f(((const u16*)p)[i]);
    }
}

__device__ __forceinline__ float e8m0_scale(uint8_t b) {
    return (b == 0xFFu) ? __int_as_float(0x7fc00000) : ldexpf(1.0f, (int)b - 127);
}

__device__ __forceinline__ float e4m3_to_f32(uint8_t b) { return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)); }

__device__ __constant__ float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float sqrtsoftplus_f(float x) {
    const float m = fabsf(x);
    return sqrtf(log1pf(expf(-m)) + fmaxf(x, 0.0f));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// ------------------------------------------------------------------------------------------------ 1. embedding

// h[r][c][d] = table[ids[r]][d], one block a (row, stream copy). `dtype` is the table's; the stream is fp32.
template <int T>
__global__ void embed_k(float* __restrict__ h, const int* __restrict__ ids, const void* __restrict__ table, int nc,
                        int dim, int vocab) {
    const int r = blockIdx.x / nc;
    const int c = blockIdx.x - r * nc;
    const int t = ids[r];
    const bool ok = (unsigned)t < (unsigned)vocab;  // caller-validated; a bad id would fault the table read
    const size_t src = (size_t)(ok ? t : 0) * dim;
    float* dst = h + ((size_t)r * nc + c) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        dst[d] = (T == T_F32) ? ((const float*)table)[src + d] : ldx<T>(table, src + d);
    }
}

// ------------------------------------------------------------------------------------------------ 2. hyper-connections

// pre[4], post[4], comb[16] of one row, exactly `hc_split_sinkhorn_kernel`.
__device__ void hc_split(float* pre, float* post, float* comb, const float* mixes, const float* scale, const float* base,
                         float hc_eps, int iters) {
    for (int h = 0; h < NS; ++h) {
        pre[h] = sigmoid_f(fmaf(mixes[h], scale[0], base[h])) + hc_eps;
        post[h] = 2.0f * sigmoid_f(fmaf(mixes[h + NS], scale[1], base[h + NS]));
    }
    for (int j = 0; j < 16; ++j) {
        comb[j] = fmaf(mixes[NS * 2 + j], scale[2], base[NS * 2 + j]);
    }
    // Sinkhorn: softmax rows, + eps, one column normalisation, then iters-1 rounds of (row, column).
    for (int j = 0; j < NS; ++j) {
        float mx = comb[j * NS];
        for (int k = 1; k < NS; ++k) mx = fmaxf(mx, comb[j * NS + k]);
        float s = 0.0f;
        for (int k = 0; k < NS; ++k) {
            comb[j * NS + k] = expf(comb[j * NS + k] - mx);
            s += comb[j * NS + k];
        }
        for (int k = 0; k < NS; ++k) comb[j * NS + k] = comb[j * NS + k] / s + hc_eps;
    }
    float col[NS] = {0, 0, 0, 0};
    for (int k = 0; k < NS; ++k) {
        for (int j = 0; j < NS; ++j) col[k] += comb[j * NS + k];
        for (int j = 0; j < NS; ++j) comb[j * NS + k] /= col[k] + hc_eps;
    }
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < NS; ++j) {
            float s = 0.0f;
            for (int k = 0; k < NS; ++k) s += comb[j * NS + k];
            for (int k = 0; k < NS; ++k) comb[j * NS + k] /= s + hc_eps;
        }
        for (int k = 0; k < NS; ++k) {
            float s = 0.0f;
            for (int j = 0; j < NS; ++j) s += comb[j * NS + k];
            for (int j = 0; j < NS; ++j) comb[j * NS + k] /= s + hc_eps;
        }
    }
}

// One block a row: the 24 dots over the flattened (stream, d) row, then the split.
__global__ void __launch_bounds__(1024) hc_mixes_k(const float* __restrict__ x, const float* __restrict__ fnw,
                                                   const float* __restrict__ scale, const float* __restrict__ base,
                                                   float* __restrict__ pre, float* __restrict__ post,
                                                   float* __restrict__ comb, int dim, float rms_eps, float hc_eps,
                                                   int iters) {
    const int row = blockIdx.x;
    const int row_n = NS * dim;
    const float* xr = x + (size_t)row * row_n;
    __shared__ float sh[32][NM];
    // Each thread walks a strided slice of the row and keeps 24 partial accumulators in registers.
    __shared__ float sh_ss[32];
    float acc[NM];
    float my_ss = 0.0f;
#pragma unroll
    for (int q = 0; q < NM; ++q) acc[q] = 0.0f;
    for (int j = threadIdx.x; j < row_n; j += blockDim.x) {
        const float v = xr[j];
        my_ss = fmaf(v, v, my_ss);
        for (int q = 0; q < NM; ++q) acc[q] = fmaf(fnw[(size_t)q * row_n + j], v, acc[q]);
    }
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nw = blockDim.x >> 5;
    for (int q = 0; q < NM; ++q) {
        float v = acc[q];
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0) sh[w][q] = v;
    }
    {
        float v = my_ss;
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0) sh_ss[w] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        if (row_n >= 4096) { for (int wi = 0; wi < nw; ++wi) ss += sh_ss[wi]; } else { for (int j = 0; j < row_n; ++j) ss = fmaf(xr[j], xr[j], ss); }
        const float rmr = rsqrtf(ss / (float)row_n + rms_eps);
        float mixes[NM];
        for (int q = 0; q < NM; ++q) {
            float s = 0.0f;
            for (int wi = 0; wi < nw; ++wi) s += sh[wi][q];
            mixes[q] = s * rmr;
        }
        hc_split(pre + row * NS, post + row * NS, comb + row * 16, mixes, scale, base, hc_eps, iters);
    }
}

// y[r][d] = sum_s pre[r][s] * x[r][s][d], fp16 out.
__global__ void hc_pre_k(const float* __restrict__ x, const float* __restrict__ pre, __half* __restrict__ y, int dim) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (d >= dim) return;
    const float* xr = x + (size_t)row * NS * dim;
    const float* pr = pre + row * NS;
    __half* yr = y + (size_t)row * dim;
    float acc = pr[0] * xr[d];
    for (int s = 1; s < NS; ++s) acc = fmaf(pr[s], xr[s * dim + d], acc);
    yr[d] = __float2half(acc);
}

// x[r][h][d] <- post[r][h] * y[r][d] + sum_g comb[r][g][h] * x[r][g][d], in place; one thread a column.
__global__ void hc_post_k(float* __restrict__ x, const void* __restrict__ y, int y_dtype, const float* __restrict__ post,
                          const float* __restrict__ comb, int dim) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (d >= dim) return;
    float* xr = x + (size_t)row * NS * dim;
    const float* po = post + row * NS;
    const float* co = comb + row * 16;
    const float yv = ldx<T_F16>(y, (size_t)row * dim + d);
    float xv[NS];
    for (int s = 0; s < NS; ++s) xv[s] = xr[s * dim + d];
    for (int h = 0; h < NS; ++h) {
        float acc = po[h] * yv;
        for (int g = 0; g < NS; ++g) acc = fmaf(co[g * NS + h], xv[g], acc);
        xr[h * dim + d] = acc;
    }
}
__device__ __forceinline__ void mtp_f16_load16(const __half* row, int i0, float* v) {
    const uint4* p4 = reinterpret_cast<const uint4*>(row + i0);
    const uint4 u0 = __ldg(p4 + 0);
    const uint4 u1 = __ldg(p4 + 1);
    const u32 w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

__global__ void __launch_bounds__(128)
    mtp_attn_512_k(const __half* __restrict__ q, const __half* __restrict__ kv, const float* __restrict__ sink,
                   const int* __restrict__ idxs, __half* __restrict__ o, int n_heads, int n_kv, float scale) {
    constexpr int WARPS = 4;
    __shared__ float sm_m[WARPS];
    __shared__ float sm_l[WARPS];
    __shared__ float sm_acc[WARPS][512];

    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int i0 = lane * 16;

    const size_t qbase = ((size_t)row * n_heads + head) * 512;
    float qv[16];
    mtp_f16_load16(q + qbase, i0, qv);

    float m = (warp == 0) ? sink[head] : -INFINITY;
    float l = (warp == 0) ? 1.0f : 0.0f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

#pragma unroll 2
    for (int j = warp; j < n_kv; j += WARPS) {
        const int e = __ldg(idxs + j);
        if (e < 0) continue;
        float kv_v[16];
        mtp_f16_load16(kv + (size_t)e * 512, i0, kv_v);
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) d = fmaf(qv[i], kv_v[i], d);
        d = warp_sum(d) * scale;
        if (d > m) {
            const float c = (l > 0.0f) ? __expf(m - d) : 0.0f;
            m = d;
            l = l * c + 1.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = fmaf(acc[i], c, kv_v[i]);
        } else {
            const float p = __expf(d - m);
            l += p;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = fmaf(p, kv_v[i], acc[i]);
        }
    }
    if (lane == 0) {
        sm_m[warp] = m;
        sm_l[warp] = l;
    }
    __syncthreads();

    float m_max = sm_m[0];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) m_max = fmaxf(m_max, sm_m[w]);
    float l_sum = 0.0f;
    float w_scale[WARPS];
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        const float sw = (sm_l[w] > 0.0f) ? __expf(sm_m[w] - m_max) : 0.0f;
        w_scale[w] = sw;
        l_sum += sm_l[w] * sw;
    }
    const float factor = (l_sum > 0.0f) ? (w_scale[warp] / l_sum) : 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        sm_acc[warp][i0 + i] = acc[i] * factor;
    }
    __syncthreads();

    for (int d0 = tid * 2; d0 < 512; d0 += WARPS * 32 * 2) {
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            s0 += sm_acc[w][d0];
            s1 += sm_acc[w][d0 + 1];
        }
        *reinterpret_cast<__half2*>(o + qbase + d0) = __floats2half2_rn(s0, s1);
    }
}

__global__ void __launch_bounds__(128)
    mtp_attn_512_params_k(const __half* __restrict__ q, const __half* __restrict__ kv, const float* __restrict__ sink,
                   const int* __restrict__ idxs, __half* __restrict__ o, int n_heads, float scale,
                   const Dsv4MtpRoundParams* p) {
    const int n_kv = p->n_kv;
    constexpr int WARPS = 4;
    __shared__ float sm_m[WARPS];
    __shared__ float sm_l[WARPS];
    __shared__ float sm_acc[WARPS][512];

    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int i0 = lane * 16;

    const size_t qbase = ((size_t)row * n_heads + head) * 512;
    float qv[16];
    mtp_f16_load16(q + qbase, i0, qv);

    float m = (warp == 0) ? sink[head] : -INFINITY;
    float l = (warp == 0) ? 1.0f : 0.0f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

#pragma unroll 2
    for (int j = warp; j < n_kv; j += WARPS) {
        const int e = __ldg(idxs + j);
        if (e < 0) continue;
        float kv_v[16];
        mtp_f16_load16(kv + (size_t)e * 512, i0, kv_v);
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) d = fmaf(qv[i], kv_v[i], d);
        d = warp_sum(d) * scale;
        if (d > m) {
            const float c = (l > 0.0f) ? __expf(m - d) : 0.0f;
            m = d;
            l = l * c + 1.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = fmaf(acc[i], c, kv_v[i]);
        } else {
            const float p = __expf(d - m);
            l += p;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = fmaf(p, kv_v[i], acc[i]);
        }
    }
    if (lane == 0) {
        sm_m[warp] = m;
        sm_l[warp] = l;
    }
    __syncthreads();

    float m_max = sm_m[0];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) m_max = fmaxf(m_max, sm_m[w]);
    float l_sum = 0.0f;
    float w_scale[WARPS];
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        const float sw = (sm_l[w] > 0.0f) ? __expf(sm_m[w] - m_max) : 0.0f;
        w_scale[w] = sw;
        l_sum += sm_l[w] * sw;
    }
    const float factor = (l_sum > 0.0f) ? (w_scale[warp] / l_sum) : 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        sm_acc[warp][i0 + i] = acc[i] * factor;
    }
    __syncthreads();

    for (int d0 = tid * 2; d0 < 512; d0 += WARPS * 32 * 2) {
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            s0 += sm_acc[w][d0];
            s1 += sm_acc[w][d0 + 1];
        }
        *reinterpret_cast<__half2*>(o + qbase + d0) = __floats2half2_rn(s0, s1);
    }
}


// ------------------------------------------------------------------------------------------------ 3. block attention

// DSparkAttention.forward for start_pos > 0. `kv` is the combined plane [window + block_size][hd]; `idxs` is the one
// slot row the reference broadcasts over every query; `o` is [block_size][n_heads][hd] fp16.
__global__ void mtp_attn_k(const __half* __restrict__ q, const __half* __restrict__ kv, const float* __restrict__ sink,
                           const int* __restrict__ idxs, __half* __restrict__ o, int n_heads, int hd, int n_kv,
                           float scale) {
    __shared__ float sc[WARPS_A][MAX_KV];
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int head = blockIdx.x * WARPS_A + w;
    const int row = blockIdx.y;
    const __half* qh = q + ((size_t)row * n_heads + head) * hd;
    const int nv = hd / 32;

    float mx = sink[head];
    for (int j = 0; j < n_kv; ++j) {
        const int e = idxs[j];
        const __half* kh = kv + (size_t)e * hd;
        float d = 0.0f;
#pragma unroll 4
        for (int c = 0; c < nv; ++c) d = fmaf(__half2float(qh[lane + 32 * c]), __half2float(kh[lane + 32 * c]), d);
        d = warp_sum(d) * scale;
        if (lane == 0) sc[w][j] = d;
        mx = fmaxf(mx, d);  // warp_sum leaves every lane with the same bits
    }
    __syncwarp();
    float sum = 0.0f;
    for (int j = 0; j < n_kv; ++j) {
        const float e = expf(sc[w][j] - mx);
        if (lane == 0) sc[w][j] = e;
        sum += e;
    }
    __syncwarp();
    // the sink is one more logit with no value: it only renormalises
    sum += expf(sink[head] - mx);
    const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    float acc[32];
#pragma unroll
    for (int c = 0; c < 32; ++c) acc[c] = 0.0f;
    for (int j = 0; j < n_kv; ++j) {
        const float e = sc[w][j] * inv;
        const __half* kh = kv + (size_t)idxs[j] * hd;
        for (int c = 0; c < nv; ++c) acc[c] = fmaf(e, __half2float(kh[lane + 32 * c]), acc[c]);
    }
    __half* oh = o + ((size_t)row * n_heads + head) * hd;
    for (int c = 0; c < nv; ++c) oh[lane + 32 * c] = __float2half(acc[c]);
}

// Device-params variant: `n_kv` comes from the round params block (same launch config, same math) so a graph
// captured at one `start_pos` replays as it advances.
__global__ void mtp_attn_params_k(const __half* __restrict__ q, const __half* __restrict__ kv,
                                  const float* __restrict__ sink, const int* __restrict__ idxs,
                                  __half* __restrict__ o, int n_heads, int hd, float scale,
                                  const Dsv4MtpRoundParams* p) {
    const int n_kv = p->n_kv;
    __shared__ float sc[WARPS_A][MAX_KV];
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int head = blockIdx.x * WARPS_A + w;
    const int row = blockIdx.y;
    const __half* qh = q + ((size_t)row * n_heads + head) * hd;
    const int nv = hd / 32;

    float mx = sink[head];
    for (int j = 0; j < n_kv; ++j) {
        const int e = idxs[j];
        const __half* kh = kv + (size_t)e * hd;
        float d = 0.0f;
#pragma unroll 4
        for (int c = 0; c < nv; ++c) d = fmaf(__half2float(qh[lane + 32 * c]), __half2float(kh[lane + 32 * c]), d);
        d = warp_sum(d) * scale;
        if (lane == 0) sc[w][j] = d;
        mx = fmaxf(mx, d);  // warp_sum leaves every lane with the same bits
    }
    __syncwarp();
    float sum = 0.0f;
    for (int j = 0; j < n_kv; ++j) {
        const float e = expf(sc[w][j] - mx);
        if (lane == 0) sc[w][j] = e;
        sum += e;
    }
    __syncwarp();
    // the sink is one more logit with no value: it only renormalises
    sum += expf(sink[head] - mx);
    const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    float acc[32];
#pragma unroll
    for (int c = 0; c < 32; ++c) acc[c] = 0.0f;
    for (int j = 0; j < n_kv; ++j) {
        const float e = sc[w][j] * inv;
        const __half* kh = kv + (size_t)idxs[j] * hd;
        for (int c = 0; c < nv; ++c) acc[c] = fmaf(e, __half2float(kh[lane + 32 * c]), acc[c]);
    }
    __half* oh = o + ((size_t)row * n_heads + head) * hd;
    for (int c = 0; c < nv; ++c) oh[lane + 32 * c] = __float2half(acc[c]);
}

// ------------------------------------------------------------------------------------------------ 4. argmax (the markov walk)

// out[r] = argmax_v logits[r][v], lowest index on a tie (the reference's argmax over a fp32 row).
__global__ void __launch_bounds__(1024) argmax_k(const float* __restrict__ logits, int* __restrict__ out, int vocab) {
    const int r = blockIdx.x;
    const float* row = logits + (size_t)r * vocab;
    float bv = -INFINITY;
    int bi = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        const float x = row[v];
        if (x > bv) {
            bv = x;
            bi = v;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, m);
        const int oi = __shfl_down_sync(0xffffffffu, bi, m);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        sv[wid] = bv;
        si[wid] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int best = 0;
        for (int w = 1; w < 32; ++w) {
            if (sv[w] > sv[best] || (sv[w] == sv[best] && si[w] < si[best])) best = w;
        }
        out[r] = si[best];
    }
}

// ------------------------------------------------------------------------------------------------ 5. fp8 logits (f32 out)

// y [m, n] f32 = x [m, k] @ W [n, k]^T, W E4M3 [n, k] with an E8M0 [n/32, k/32] scale. Same tile math as
// `fp8_linear.cu`, fp32 output: the head's logits are 129,280 wide and the markov bias is added to them, so an fp16
// rounding there is a rounding on the sampled token.
__device__ __forceinline__ float mtp_dot4_e4m3(uint32_t u, float4 xv) {
    const float w0 = e4m3_to_f32(static_cast<uint8_t>(u & 0xFFu));
    const float w1 = e4m3_to_f32(static_cast<uint8_t>((u >> 8) & 0xFFu));
    const float w2 = e4m3_to_f32(static_cast<uint8_t>((u >> 16) & 0xFFu));
    const float w3 = e4m3_to_f32(static_cast<uint8_t>(u >> 24));
    return fmaf(w0, xv.x, fmaf(w1, xv.y, fmaf(w2, xv.z, w3 * xv.w)));
}

__device__ __forceinline__ float mtp_dot16_e4m3(uint4 w16, float4 x0, float4 x1, float4 x2, float4 x3) {
    return mtp_dot4_e4m3(w16.x, x0) + mtp_dot4_e4m3(w16.y, x1) + mtp_dot4_e4m3(w16.z, x2) + mtp_dot4_e4m3(w16.w, x3);
}

__device__ __forceinline__ float ld_any(const void* p, int dtype, size_t i) {
    if (dtype == T_F32) return ((const float*)p)[i];
    if (dtype == T_BF16) return bf2f(((const u16*)p)[i]);
    return __half2float(((const __half*)p)[i]);
}

template <int ROWS>
__global__ void __launch_bounds__(THREADS) fp8_logits_k(const void* __restrict__ x, int x_dtype,
                                                        const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                                                        float* __restrict__ y, int m, int k, int n) {
    constexpr int CHUNK_K = 512;
    __shared__ __align__(16) float sx[ROWS][CHUNK_K];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n0 = blockIdx.x * TILE + warp * 4;
    const int r0 = blockIdx.y * ROWS;
    const int ktiles = k / TILE;
    const uint8_t* s_tile_row = s + (size_t)blockIdx.x * ktiles;

    float acc[ROWS][4];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.0f;

    for (int k_base = 0; k_base < k; k_base += CHUNK_K) {
        const int rem = k - k_base;
        for (int idx = tid; idx < ROWS * CHUNK_K; idx += THREADS) {
            const int rn = idx / CHUNK_K;
            const int kk = idx % CHUNK_K;
            const int rc = r0 + rn;
            sx[rn][kk] = (rc < m && kk < rem) ? ld_any(x, x_dtype, (size_t)rc * k + k_base + kk) : 0.0f;
        }
        __syncthreads();
        const int k_off = lane * 16;
        if (k_off < rem) {
            const float sc = e8m0_scale(__ldg(s_tile_row + (k_base >> 5) + (lane >> 1)));
            uint4 w_vec[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int nc = n0 + c;
                w_vec[c] = (nc < n) ? __ldg(reinterpret_cast<const uint4*>(w + (size_t)nc * k + k_base + k_off))
                                    : make_uint4(0u, 0u, 0u, 0u);
            }
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const float4* sx4 = reinterpret_cast<const float4*>(&sx[r][k_off]);
                const float4 x0 = sx4[0], x1 = sx4[1], x2 = sx4[2], x3 = sx4[3];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    acc[r][c] = fmaf(mtp_dot16_e4m3(w_vec[c], x0, x1, x2, x3), sc, acc[r][c]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int rc = r0 + r;
        if (rc < m) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float v = warp_sum(acc[r][c]);
                const int nc = n0 + c;
                if (lane == 0 && nc < n) y[(size_t)rc * n + nc] = v;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ 6. router

// scores[r][e] = sqrtsoftplus(x[r] . gate[e] / temp); one block a row, a warp an expert tile.
__global__ void __launch_bounds__(256) router_k(const __half* __restrict__ x, const u16* __restrict__ gate,
                                                float* __restrict__ logits, int dim, int experts, float temp) {
    const int r = blockIdx.x;
    const __half* xr = x + (size_t)r * dim;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int e = blockIdx.y * 8 + warp; e < experts; e += gridDim.y * 8) {
        const u16* gr = gate + (size_t)e * dim;
        float acc = 0.0f;
        for (int i = lane; i < dim; i += 32) acc = fmaf(__half2float(xr[i]), bf2f(gr[i]), acc);
        acc = warp_sum(acc);
        if (lane == 0) logits[(size_t)r * experts + e] = sqrtsoftplus_f(acc / temp);
    }
}

template <int ROWS>
__global__ void __launch_bounds__(256) router_rows_k(const __half* __restrict__ x, const u16* __restrict__ gate,
                                                     float* __restrict__ logits, int rows, int dim, int experts, float temp) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e = blockIdx.x * 8 + warp;
    if (e >= experts) return;
    const u16* gr = gate + (size_t)e * dim;
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        const float g = bf2f(__ldg(gr + i));
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r < rows) {
                acc[r] = fmaf(__half2float(__ldg(x + (size_t)r * dim + i)), g, acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        if (r < rows) {
            const float s = warp_sum(acc[r]);
            if (lane == 0) logits[(size_t)r * experts + e] = sqrtsoftplus_f(s / temp);
        }
    }
}

// noaux_tc top-k: the bias steers selection only, the weights come from the unbiased scores, normalised over the top-k
// and scaled by route_scale. One thread a row.
__global__ void select_k(const float* __restrict__ scores, const float* __restrict__ bias, int* __restrict__ picks,
                         __half* __restrict__ wts, int rows, int experts, int topk, float scale) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* sr = scores + (size_t)r * experts;
    int chosen[16];
    for (int k = 0; k < topk; ++k) {
        int best = -1;
        float bv = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            bool used = false;
            for (int j = 0; j < k; ++j) used = used || (chosen[j] == e);
            if (used) continue;
            const float v = sr[e] + bias[e];
            if (v > bv) {
                bv = v;
                best = e;
            }
        }
        chosen[k] = best < 0 ? 0 : best;
    }
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) sum += sr[chosen[k]];
    const float den = sum + 1e-20f;
    for (int k = 0; k < topk; ++k) {
        picks[(size_t)r * topk + k] = chosen[k];
        wts[(size_t)r * topk + k] = __float2half((sr[chosen[k]] / den) * scale);
    }
}

__global__ void __launch_bounds__(128) select_fast_k(const float* __restrict__ scores, const float* __restrict__ bias,
                                                     int* __restrict__ picks, __half* __restrict__ wts,
                                                     int rows, int experts, int topk, float scale) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;
    const float* sr = scores + (size_t)r * experts;
    __shared__ float s_sc[256];
    __shared__ float s_key[256];
    __shared__ uint8_t s_used[256];
    for (int e = tid; e < experts; e += blockDim.x) {
        const float sc = sr[e];
        s_sc[e] = sc;
        s_key[e] = sc + bias[e];
        s_used[e] = 0;
    }
    __syncthreads();
    if (tid < 32) {
        int chosen[16];
        for (int k = 0; k < topk; ++k) {
            float bv = -INFINITY;
            int bi = -1;
            for (int e = tid; e < experts; e += 32) {
                if (!s_used[e]) {
                    const float v = s_key[e];
                    if (v > bv || (v == bv && (bi < 0 || e < bi))) {
                        bv = v;
                        bi = e;
                    }
                }
            }
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1) {
                const float ov = __shfl_down_sync(0xffffffffu, bv, m);
                const int oi = __shfl_down_sync(0xffffffffu, bi, m);
                if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) {
                    bv = ov;
                    bi = oi;
                }
            }
            bi = __shfl_sync(0xffffffffu, bi, 0);
            if (bi < 0) bi = 0;
            chosen[k] = bi;
            if (tid == 0) s_used[bi] = 1;
            __syncwarp();
        }
        if (tid == 0) {
            float sum = 0.0f;
            for (int k = 0; k < topk; ++k) sum += s_sc[chosen[k]];
            const float den = sum + 1e-20f;
            for (int k = 0; k < topk; ++k) {
                picks[(size_t)r * topk + k] = chosen[k];
                wts[(size_t)r * topk + k] = __float2half((s_sc[chosen[k]] / den) * scale);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ 7. MXFP4 experts, grouped

// act[p][j] = swiglu(x[p] . w1[e]^T, x[p] . w3[e]^T) * wpack[p] for the slots p of expert e = blockIdx.y.
// grid (inter/32, experts), block (32, 8).
__global__ void __launch_bounds__(THREADS) fp4_gateup_k(const __half* __restrict__ x, const uint8_t* __restrict__ w1,
                                                        const uint8_t* __restrict__ s1, const uint8_t* __restrict__ w3,
                                                        const uint8_t* __restrict__ s3, const int* __restrict__ offsets,
                                                        const __half* __restrict__ wpack, __half* __restrict__ act,
                                                        int dim, int inter, float limit) {
    __shared__ float sx[8][TILE + 1];
    __shared__ float ssc[2][TILE];
    const int e = blockIdx.y;
    const int a = offsets[e], b = offsets[e + 1];
    const int cnt = b - a;
    if (cnt <= 0) return;
    const int n0 = blockIdx.x * TILE;
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * TILE + tx;
    const int kt = dim / TILE;
    const int nc = n0 + tx;  // column inside this expert's [inter, dim]
    // the expert's own slab: `w1` / `w3` are [experts][inter][dim]
    const uint8_t* w1c = w1 + ((size_t)e * inter + nc) * (dim / 2);
    const uint8_t* w3c = w3 + ((size_t)e * inter + nc) * (dim / 2);
    const uint8_t* s1c = s1 + ((size_t)e * inter + nc) * kt;
    const uint8_t* s3c = s3 + ((size_t)e * inter + nc) * kt;
    float g = 0.0f, u = 0.0f;
    for (int kb = 0; kb < kt; ++kb) {
        float dg = 0.0f, du = 0.0f;
#pragma unroll
        for (int j = 0; j < (8 * TILE) / THREADS; ++j) {
            const int idx = tid + j * THREADS;
            const int rn = idx / TILE;
            const int kk = idx % TILE;
            sx[rn][kk] = (rn < cnt) ? __half2float(x[(size_t)(a + rn) * dim + kb * TILE + kk]) : 0.0f;
        }
        if (tid < TILE) {
            // `s1c` / `s3c` already point at this thread's (expert, column) row, so the tile is the only index
            ssc[0][tid] = e8m0_scale(s1c[kb]);
            ssc[1][tid] = e8m0_scale(s3c[kb]);
        }
        __syncthreads();
        const uint4 r1 = *reinterpret_cast<const uint4*>(w1c + (size_t)kb * 16);
        const uint4 r3 = *reinterpret_cast<const uint4*>(w3c + (size_t)kb * 16);
        const u32 v1[4] = {r1.x, r1.y, r1.z, r1.w};
        const u32 v3[4] = {r3.x, r3.y, r3.z, r3.w};
#pragma unroll
        for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
            for (int by = 0; by < 4; ++by) {
                const u32 b1 = (v1[wd] >> (8 * by)) & 0xFFu;
                const u32 b3 = (v3[wd] >> (8 * by)) & 0xFFu;
                dg = fmaf(sx[ty][wd * 8 + 2 * by], kE2M1[b1 & 0x0Fu], dg);
                dg = fmaf(sx[ty][wd * 8 + 2 * by + 1], kE2M1[b1 >> 4], dg);
                du = fmaf(sx[ty][wd * 8 + 2 * by], kE2M1[b3 & 0x0Fu], du);
                du = fmaf(sx[ty][wd * 8 + 2 * by + 1], kE2M1[b3 >> 4], du);
            }
        }
        __syncthreads();
        g = fmaf(dg, ssc[0][tx], g);
        u = fmaf(du, ssc[1][tx], u);
    }
    if (ty < cnt && nc < inter) {
        const float gc = fminf(g, limit);
        const float uc = fminf(fmaxf(u, -limit), limit);
        const float silu = gc / (1.0f + expf(-gc));
        act[(size_t)(a + ty) * inter + nc] = __float2half(silu * uc * __half2float(wpack[a + ty]));
    }
}

// y[p][d] = act[p] . w2[e]^T for the slots p of expert e. grid (dim/32, experts), block (32, 8).
__global__ void __launch_bounds__(THREADS) fp4_down_k(const __half* __restrict__ act, const uint8_t* __restrict__ w2,
                                                      const uint8_t* __restrict__ s2, const int* __restrict__ offsets,
                                                      __half* __restrict__ y, int dim, int inter) {
    __shared__ float sx[8][TILE + 1];
    __shared__ float ssc[TILE];
    const int e = blockIdx.y;
    const int a = offsets[e], b = offsets[e + 1];
    const int cnt = b - a;
    if (cnt <= 0) return;
    const int n0 = blockIdx.x * TILE;
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * TILE + tx;
    const int kt = inter / TILE;
    const int nc = n0 + tx;  // column inside this expert's [dim, inter]
    const uint8_t* wc = w2 + ((size_t)e * dim + nc) * (inter / 2);
    const uint8_t* sc = s2 + ((size_t)e * dim + nc) * kt;
    float acc = 0.0f;
    for (int kb = 0; kb < kt; ++kb) {
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < (8 * TILE) / THREADS; ++j) {
            const int idx = tid + j * THREADS;
            const int rn = idx / TILE;
            const int kk = idx % TILE;
            sx[rn][kk] = (rn < cnt) ? __half2float(act[(size_t)(a + rn) * inter + kb * TILE + kk]) : 0.0f;
        }
        if (tid < TILE) ssc[tid] = e8m0_scale(sc[kb]);
        __syncthreads();
        const uint4 raw = *reinterpret_cast<const uint4*>(wc + (size_t)kb * 16);
        const u32 v[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
            for (int by = 0; by < 4; ++by) {
                const u32 byte = (v[wd] >> (8 * by)) & 0xFFu;
                dot = fmaf(sx[ty][wd * 8 + 2 * by], kE2M1[byte & 0x0Fu], dot);
                dot = fmaf(sx[ty][wd * 8 + 2 * by + 1], kE2M1[byte >> 4], dot);
            }
        }
        __syncthreads();
        acc = fmaf(dot, ssc[tx], acc);
    }
    if (ty < cnt && nc < dim) y[(size_t)(a + ty) * dim + nc] = __float2half(acc);
}

// ------------------------------------------------------------------------------------------------ 8. small plumbing

// f16 -> f32, for the confidence head (which the reference evaluates in fp32 over the collapsed stream).
__global__ void widen_k(const __half* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// m[r][:] = table[ids[r]][:], fp32 -- the markov embeddings the confidence head is scored against.
__global__ void gather_rows_k(float* __restrict__ dst, const int* __restrict__ ids, const float* __restrict__ table,
                              int rows, int width, int vocab) {
    const int r = blockIdx.x;
    const int t = ids[r];
    const float* src = table + (size_t)((unsigned)t < (unsigned)vocab ? t : 0) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[(size_t)r * width + i] = src[i];
}

// ids[0] = anchor, ids[1..] = noise: `forward_embed`'s `draft_input_ids`, built on the device so a captured round has
// no host-to-device copy in it. Also resets `pre_mix` [block, 4] to `[1, 0, 0, 0]` for stage 0.
__global__ void fill_ids_k(int* __restrict__ ids, float* __restrict__ pre_mix, int anchor, int noise, int block) {
    const int i = threadIdx.x;
    if (i < block) ids[i] = i == 0 ? anchor : noise;
    if (pre_mix && i < block * 4) pre_mix[i] = (i & 3) == 0 ? 1.0f : 0.0f;
}

// Device-params variant: the anchor comes from the round params block, so a CUDA-graph replay can advance it
// without re-capturing.
__global__ void fill_ids_params_k(int* __restrict__ ids, float* __restrict__ pre_mix,
                                  const Dsv4MtpRoundParams* p, int noise, int block) {
    const int i = threadIdx.x;
    if (i < block) ids[i] = i == 0 ? p->anchor : noise;
    if (pre_mix && i < block * 4) pre_mix[i] = (i & 3) == 0 ? 1.0f : 0.0f;
}

}  // namespace

extern "C" {

GB_EXPORT int gb_dsv4_mtp_embed(void* h, const int* ids, const void* table, int rows, int nc, int dim, int vocab,
                                int dtype, void* stream) {
    if (!h || !ids || !table || rows <= 0 || nc <= 0 || dim <= 0 || vocab <= 0) return ERR_INVALID;
    if (dtype == T_F32) {
        embed_k<T_F32><<<rows * nc, 256, 0, gb_stream(stream)>>>((float*)h, ids, table, nc, dim, vocab);
    } else if (dtype == T_F16) {
        embed_k<T_F16><<<rows * nc, 256, 0, gb_stream(stream)>>>((float*)h, ids, table, nc, dim, vocab);
    } else if (dtype == T_BF16) {
        embed_k<T_BF16><<<rows * nc, 256, 0, gb_stream(stream)>>>((float*)h, ids, table, nc, dim, vocab);
    } else {
        return ERR_INVALID;
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_hc_mixes(const float* x, const float* fnw, const float* scale, const float* base, float* pre,
                                   float* post, float* comb, int rows, int dim, float rms_eps, float hc_eps,
                                   int iters, void* stream) {
    if (!x || !fnw || !scale || !base || !pre || !post || !comb || rows <= 0 || dim <= 0) return ERR_INVALID;
    hc_mixes_k<<<rows, 1024, 0, gb_stream(stream)>>>(x, fnw, scale, base, pre, post, comb, dim, rms_eps, hc_eps, iters);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_hc_pre(const float* x, const float* pre, void* y, int rows, int dim, void* stream) {
    if (!x || !pre || !y || rows <= 0 || dim <= 0) return ERR_INVALID;
    hc_pre_k<<<dim3((dim + 255) / 256, rows), 256, 0, gb_stream(stream)>>>(x, pre, (__half*)y, dim);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_hc_post(float* x, const void* y, int y_dtype, const float* post, const float* comb, int rows,
                                  int dim, void* stream) {
    if (!x || !y || !post || !comb || rows <= 0 || dim <= 0 || y_dtype != T_F16) return ERR_INVALID;
    hc_post_k<<<dim3((dim + 255) / 256, rows), 256, 0, gb_stream(stream)>>>(x, y, y_dtype, post, comb, dim);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_attn(const void* q, const void* kv, const float* sink, const int* idxs, void* o, int block,
                               int n_heads, int hd, int n_kv, float scale, void* stream) {
    if (!q || !kv || !sink || !idxs || !o || block <= 0 || n_heads <= 0 || hd <= 0) return ERR_INVALID;
    if (n_kv <= 0 || n_kv > MAX_KV || (n_heads % WARPS_A) != 0 || (hd % 32) != 0) return ERR_INVALID;
    if (hd == 512) {
        mtp_attn_512_k<<<dim3(n_heads, block), 128, 0, gb_stream(stream)>>>(
            (const __half*)q, (const __half*)kv, sink, idxs, (__half*)o, n_heads, n_kv, scale);
        return GB_CHECK_LAUNCH();
    }
    mtp_attn_k<<<dim3(n_heads / WARPS_A, block), 32 * WARPS_A, 0, gb_stream(stream)>>>(
        (const __half*)q, (const __half*)kv, sink, idxs, (__half*)o, n_heads, hd, n_kv, scale);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_attn_params(const void* q, const void* kv, const float* sink, const int* idxs, void* o,
                                      int block, int n_heads, int hd, float scale, const void* params, void* stream) {
    if (!q || !kv || !sink || !idxs || !o || !params || block <= 0 || n_heads <= 0 || hd <= 0) return ERR_INVALID;
    // NOTE: `params` is a DEVICE pointer; never dereference it on the host (the kernels read it on the device).
    const Dsv4MtpRoundParams* p = (const Dsv4MtpRoundParams*)params;
    if ((n_heads % WARPS_A) != 0 || (hd % 32) != 0) return ERR_INVALID;
    if (hd == 512) {
        mtp_attn_512_params_k<<<dim3(n_heads, block), 128, 0, gb_stream(stream)>>>(
            (const __half*)q, (const __half*)kv, sink, idxs, (__half*)o, n_heads, scale, p);
        return GB_CHECK_LAUNCH();
    }
    mtp_attn_params_k<<<dim3(n_heads / WARPS_A, block), 32 * WARPS_A, 0, gb_stream(stream)>>>(
        (const __half*)q, (const __half*)kv, sink, idxs, (__half*)o, n_heads, hd, scale, p);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_argmax(const float* logits, int* out, int rows, int vocab, void* stream) {
    if (!logits || !out || rows <= 0 || vocab <= 0) return ERR_INVALID;
    argmax_k<<<rows, 1024, 0, gb_stream(stream)>>>(logits, out, vocab);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_logits_typed(const void* x, int x_dtype, const uint8_t* w, const uint8_t* s, float* y, int m, int k, int n,
                                       void* stream) {
    if (!x || !w || !s || !y || m <= 0 || k <= 0 || n <= 0) return ERR_INVALID;
    if ((k % TILE) != 0 || (n % TILE) != 0) return ERR_INVALID;
    const int ntiles = n / TILE;
    if (m == 1) {
        fp8_logits_k<1><<<dim3(ntiles, 1), THREADS, 0, gb_stream(stream)>>>(x, x_dtype, w, s, y, m, k, n);
    } else if (m <= 4) {
        fp8_logits_k<4><<<dim3(ntiles, 1), THREADS, 0, gb_stream(stream)>>>(x, x_dtype, w, s, y, m, k, n);
    } else {
        fp8_logits_k<8><<<dim3(ntiles, (m + 7) / 8), THREADS, 0, gb_stream(stream)>>>(x, x_dtype, w, s, y, m, k, n);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_logits(const void* x, const uint8_t* w, const uint8_t* s, float* y, int m, int k, int n,
                                 void* stream) {
    return gb_dsv4_mtp_logits_typed(x, T_F16, w, s, y, m, k, n, stream);
}

GB_EXPORT int gb_dsv4_mtp_router(const void* x, const u16* gate, float* logits, int rows, int dim, int experts,
                                 float temp, void* stream) {
    if (!x || !gate || !logits || rows <= 0 || dim <= 0 || experts <= 0) return ERR_INVALID;
    const int yt = (experts + 7) / 8;
    cudaStream_t st = gb_stream(stream);
    if (rows == 1) {
        router_rows_k<1><<<yt, 256, 0, st>>>((const __half*)x, gate, logits, rows, dim, experts, temp);
    } else if (rows <= 5) {
        router_rows_k<5><<<yt, 256, 0, st>>>((const __half*)x, gate, logits, rows, dim, experts, temp);
    } else if (rows <= 8) {
        router_rows_k<8><<<yt, 256, 0, st>>>((const __half*)x, gate, logits, rows, dim, experts, temp);
    } else {
        router_k<<<dim3(rows, yt), 256, 0, st>>>((const __half*)x, gate, logits, dim, experts, temp);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_select(const float* scores, const float* bias, int* picks, void* wts, int rows, int experts,
                                 int topk, float scale, void* stream) {
    if (!scores || !bias || !picks || !wts || rows <= 0 || experts <= 0 || topk <= 0 || topk > 16) return ERR_INVALID;
    cudaStream_t st = gb_stream(stream);
    if (experts <= 256) {
        select_fast_k<<<(unsigned)rows, 128, 0, st>>>(scores, bias, picks, (__half*)wts, rows, experts, topk, scale);
        return GB_CHECK_LAUNCH();
    }
    const int nt = 128;
    select_k<<<(rows + nt - 1) / nt, nt, 0, st>>>(scores, bias, picks, (__half*)wts, rows, experts, topk, scale);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_fp4_gateup(const void* x, const uint8_t* w1, const uint8_t* s1, const uint8_t* w3,
                                     const uint8_t* s3, const int* offsets, const void* wpack, void* act, int experts,
                                     int dim, int inter, float limit, void* stream) {
    if (!x || !w1 || !s1 || !w3 || !s3 || !offsets || !wpack || !act) return ERR_INVALID;
    if (dim <= 0 || inter <= 0 || (dim % TILE) != 0 || (inter % TILE) != 0) return ERR_INVALID;
    fp4_gateup_k<<<dim3(inter / TILE, experts), dim3(TILE, 8), 0, gb_stream(stream)>>>(
        (const __half*)x, w1, s1, w3, s3, offsets, (const __half*)wpack, (__half*)act, dim, inter, limit);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_fp4_down(const void* act, const uint8_t* w2, const uint8_t* s2, const int* offsets, void* y,
                                   int experts, int dim, int inter, void* stream) {
    if (!act || !w2 || !s2 || !offsets || !y) return ERR_INVALID;
    if (dim <= 0 || inter <= 0 || (dim % TILE) != 0 || (inter % TILE) != 0) return ERR_INVALID;
    fp4_down_k<<<dim3(dim / TILE, experts), dim3(TILE, 8), 0, gb_stream(stream)>>>((const __half*)act, w2, s2, offsets,
                                                                                   (__half*)y, dim, inter);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_widen(const void* src, float* dst, int n, void* stream) {
    if (!src || !dst || n <= 0) return ERR_INVALID;
    widen_k<<<(n + 255) / 256, 256, 0, gb_stream(stream)>>>((const __half*)src, dst, n);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_gather_rows(float* dst, const int* ids, const float* table, int rows, int width, int vocab,
                                      void* stream) {
    if (!dst || !ids || !table || rows <= 0 || width <= 0 || vocab <= 0) return ERR_INVALID;
    gather_rows_k<<<rows, 128, 0, gb_stream(stream)>>>(dst, ids, table, rows, width, vocab);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_fill_ids(int* ids, float* pre_mix, int anchor, int noise, int block, void* stream) {
    if (!ids || block <= 0 || block > 256) return ERR_INVALID;
    fill_ids_k<<<1, 1024, 0, gb_stream(stream)>>>(ids, pre_mix, anchor, noise, block);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_fill_ids_params(int* ids, float* pre_mix, const void* params, int noise, int block,
                                          void* stream) {
    if (!ids || !params || block <= 0 || block > 256) return ERR_INVALID;
    fill_ids_params_k<<<1, 1024, 0, gb_stream(stream)>>>(
        ids, pre_mix, (const Dsv4MtpRoundParams*)params, noise, block);
    return GB_CHECK_LAUNCH();
}

}  // extern "C"

namespace {

__global__ void __launch_bounds__(256) markov_bias_bf16_4_k(float* __restrict__ logits,
                                                            const int* __restrict__ tokens,
                                                            const u16* __restrict__ embed,
                                                            const u16* __restrict__ head,
                                                            int vocab,
                                                            int dir) {
    const int row = blockIdx.y;
    const int t = __ldg(tokens + row);
    const bool ok = (unsigned)t < (unsigned)vocab;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int bx = dir ? ((int)gridDim.x - 1 - (int)blockIdx.x) : (int)blockIdx.x;
    const int v_base = (bx * 8 + warp) * 4;
    if (v_base >= vocab) return;

    float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f, e4 = 0.f, e5 = 0.f, e6 = 0.f, e7 = 0.f;
    if (ok) {
        const uint4 eu = __ldg(reinterpret_cast<const uint4*>(embed + (size_t)t * 256 + lane * 8));
        e0 = bf2f(eu.x & 0xFFFFu); e1 = bf2f(eu.x >> 16);
        e2 = bf2f(eu.y & 0xFFFFu); e3 = bf2f(eu.y >> 16);
        e4 = bf2f(eu.z & 0xFFFFu); e5 = bf2f(eu.z >> 16);
        e6 = bf2f(eu.w & 0xFFFFu); e7 = bf2f(eu.w >> 16);
    }

#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int v = v_base + c;
        if (v < vocab) {
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(head + (size_t)v * 256 + lane * 8));
            float acc = 0.0f;
            acc = fmaf(e0, bf2f(u.x & 0xFFFFu), acc);
            acc = fmaf(e1, bf2f(u.x >> 16), acc);
            acc = fmaf(e2, bf2f(u.y & 0xFFFFu), acc);
            acc = fmaf(e3, bf2f(u.y >> 16), acc);
            acc = fmaf(e4, bf2f(u.z & 0xFFFFu), acc);
            acc = fmaf(e5, bf2f(u.z >> 16), acc);
            acc = fmaf(e6, bf2f(u.w & 0xFFFFu), acc);
            acc = fmaf(e7, bf2f(u.w >> 16), acc);
            acc = warp_sum(acc);
            if (lane == 0) logits[(size_t)row * vocab + v] += acc;
        }
    }
}

__global__ void __launch_bounds__(256) markov_bias_bf16_k(float* __restrict__ logits,
                                                          const int* __restrict__ tokens,
                                                          const u16* __restrict__ embed,
                                                          const u16* __restrict__ head,
                                                          int vocab,
                                                          int rank) {
    __shared__ float se[256];
    const int row = blockIdx.y;
    const int t = tokens[row];
    const bool ok = (unsigned)t < (unsigned)vocab;
    for (int i = threadIdx.x; i < rank; i += 256) {
        se[i] = ok ? bf2f(embed[(size_t)t * rank + i]) : 0.0f;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * 8 + warp;
    if (v >= vocab) return;
    const u16* hr = head + (size_t)v * rank;
    float acc = 0.0f;
    if (rank == 256) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(hr + lane * 8));
        const u16 h0 = u.x & 0xFFFFu, h1 = u.x >> 16;
        const u16 h2 = u.y & 0xFFFFu, h3 = u.y >> 16;
        const u16 h4 = u.z & 0xFFFFu, h5 = u.z >> 16;
        const u16 h6 = u.w & 0xFFFFu, h7 = u.w >> 16;
        const float* sp = se + lane * 8;
        acc = fmaf(sp[0], bf2f(h0), acc);
        acc = fmaf(sp[1], bf2f(h1), acc);
        acc = fmaf(sp[2], bf2f(h2), acc);
        acc = fmaf(sp[3], bf2f(h3), acc);
        acc = fmaf(sp[4], bf2f(h4), acc);
        acc = fmaf(sp[5], bf2f(h5), acc);
        acc = fmaf(sp[6], bf2f(h6), acc);
        acc = fmaf(sp[7], bf2f(h7), acc);
    } else {
        for (int i = lane; i < rank; i += 32) {
            acc = fmaf(se[i], bf2f(hr[i]), acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[(size_t)row * vocab + v] += acc;
}

__global__ void gather_rows_bf16_k(float* __restrict__ dst,
                                   const int* __restrict__ ids,
                                   const u16* __restrict__ table,
                                   int rows,
                                   int width,
                                   int vocab) {
    const int r = blockIdx.x;
    const int t = ids[r];
    const u16* src = table + (size_t)((unsigned)t < (unsigned)vocab ? t : 0) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        dst[(size_t)r * width + i] = bf2f(src[i]);
    }
}

__global__ void __launch_bounds__(256) confidence_bf16_k(const float* __restrict__ h,
                                                         const float* __restrict__ m,
                                                         const u16* __restrict__ w,
                                                         const float* __restrict__ b,
                                                         float* __restrict__ out,
                                                         int dim,
                                                         int rank,
                                                         int combine_gumbel_rows) {
    __shared__ float sm[8];
    const int r = blockIdx.x;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += 256) acc = fmaf(h[(size_t)r * dim + i], bf2f(w[i]), acc);
    for (int i = threadIdx.x; i < rank; i += 256) acc = fmaf(m[(size_t)r * rank + i], bf2f(w[dim + i]), acc);
    acc = warp_sum(acc);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int wi = 0; wi < 8; ++wi) s += sm[wi];
        const float c_head = 1.0f / (1.0f + expf(-(s + b[0])));
        if (combine_gumbel_rows > 0) {
            out[r] = (r < combine_gumbel_rows) ? (c_head * out[r]) : 0.0f;
        } else {
            out[r] = c_head;
        }
    }
}

__global__ void tap_mean_bf16_k(const float* __restrict__ streams,
                                u16* __restrict__ taps_bf16,
                                int rows,
                                int dim,
                                int tap_idx,
                                int n_taps) {
    const int r = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || d >= dim) return;
    const float* sr = streams + (size_t)r * 4 * dim;
    const float m = 0.25f * ((sr[d] + sr[dim + d]) + (sr[2 * dim + d] + sr[3 * dim + d]));
    const __nv_bfloat16 b = __float2bfloat16_rn(m);
    taps_bf16[(size_t)r * (n_taps * dim) + (size_t)tap_idx * dim + d] = *reinterpret_cast<const u16*>(&b);
}

__global__ void __launch_bounds__(256) mtp_topk_candidates_4_k(const float* __restrict__ logits,
                                                               int* __restrict__ cand_ids,
                                                               int vocab) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lr = logits + (size_t)row * vocab;
    float bv0 = -INFINITY, bv1 = -INFINITY, bv2 = -INFINITY, bv3 = -INFINITY;
    int bi0 = 0, bi1 = 0, bi2 = 0, bi3 = 0;
    for (int v = tid; v < vocab; v += 256) {
        const float x = __ldg(lr + v);
        if (x > bv3) {
            if (x > bv0) {
                bv3 = bv2; bi3 = bi2;
                bv2 = bv1; bi2 = bi1;
                bv1 = bv0; bi1 = bi0;
                bv0 = x;   bi0 = v;
            } else if (x > bv1) {
                bv3 = bv2; bi3 = bi2;
                bv2 = bv1; bi2 = bi1;
                bv1 = x;   bi1 = v;
            } else if (x > bv2) {
                bv3 = bv2; bi3 = bi2;
                bv2 = x;   bi2 = v;
            } else {
                bv3 = x;   bi3 = v;
            }
        }
    }
    *reinterpret_cast<int4*>(cand_ids + (size_t)row * 1024 + tid * 4) = make_int4(bi0, bi1, bi2, bi3);
}

__global__ void __launch_bounds__(256) markov_bias_cand_bf16_4_k(float* __restrict__ logits,
                                                                 const int* __restrict__ tokens,
                                                                 const u16* __restrict__ embed,
                                                                 const u16* __restrict__ head,
                                                                 const int* __restrict__ cand_ids,
                                                                 float* __restrict__ cand_scores,
                                                                 int vocab) {
    const int t = __ldg(tokens);
    const bool ok = (unsigned)t < (unsigned)vocab;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c_base = ((int)blockIdx.x * 8 + warp) * 4;

    float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f, e4 = 0.f, e5 = 0.f, e6 = 0.f, e7 = 0.f;
    if (ok) {
        const uint4 eu = __ldg(reinterpret_cast<const uint4*>(embed + (size_t)t * 256 + lane * 8));
        e0 = bf2f(eu.x & 0xFFFFu); e1 = bf2f(eu.x >> 16);
        e2 = bf2f(eu.y & 0xFFFFu); e3 = bf2f(eu.y >> 16);
        e4 = bf2f(eu.z & 0xFFFFu); e5 = bf2f(eu.z >> 16);
        e6 = bf2f(eu.w & 0xFFFFu); e7 = bf2f(eu.w >> 16);
    }

#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int idx = c_base + c;
        const int v = __ldg(cand_ids + idx);
        if ((unsigned)v < (unsigned)vocab) {
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(head + (size_t)v * 256 + lane * 8));
            float acc = 0.0f;
            acc = fmaf(e0, bf2f(u.x & 0xFFFFu), acc);
            acc = fmaf(e1, bf2f(u.x >> 16), acc);
            acc = fmaf(e2, bf2f(u.y & 0xFFFFu), acc);
            acc = fmaf(e3, bf2f(u.y >> 16), acc);
            acc = fmaf(e4, bf2f(u.z & 0xFFFFu), acc);
            acc = fmaf(e5, bf2f(u.z >> 16), acc);
            acc = fmaf(e6, bf2f(u.w & 0xFFFFu), acc);
            acc = fmaf(e7, bf2f(u.w >> 16), acc);
            acc = warp_sum(acc);
            if (lane == 0) {
                const float s = logits[v] + acc;
                logits[v] = s;
                cand_scores[idx] = s;
            }
        } else if (lane == 0) {
            cand_scores[idx] = -INFINITY;
        }
    }
}

__global__ void __launch_bounds__(256) argmax_cand_1024_k(const float* __restrict__ cand_scores,
                                                          const int* __restrict__ cand_ids,
                                                          int* __restrict__ out) {
    const int tid = threadIdx.x;
    const float4 sc4 = __ldg(reinterpret_cast<const float4*>(cand_scores + tid * 4));
    const int4 id4 = __ldg(reinterpret_cast<const int4*>(cand_ids + tid * 4));
    float vals[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
    int ids[4] = {id4.x, id4.y, id4.z, id4.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (!isfinite(vals[j])) {
            vals[j] = -INFINITY;
            ids[j] = 0x7fffffff;
        }
    }
    float bv = vals[0];
    int bi = ids[0];
#pragma unroll
    for (int j = 1; j < 4; ++j) {
        if (vals[j] > bv || (vals[j] == bv && ids[j] < bi)) {
            bv = vals[j];
            bi = ids[j];
        }
    }
    __shared__ float sv[8];
    __shared__ int si[8];
    const int lane = tid & 31, wid = tid >> 5;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, m);
        const int oi = __shfl_down_sync(0xffffffffu, bi, m);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        sv[wid] = bv;
        si[wid] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        int best = 0;
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > sv[best] || (sv[w] == sv[best] && si[w] < si[best])) best = w;
        }
        out[0] = (si[best] >= 0 && si[best] != 0x7fffffff) ? si[best] : 0;
    }
}

__global__ void __launch_bounds__(256) keyed_cand_1024_k(const float* __restrict__ cand_scores,
                                                         const int* __restrict__ cand_ids,
                                                         int* __restrict__ out_id,
                                                         float* __restrict__ out_conf,
                                                         int cand_k,
                                                         unsigned long long seed,
                                                         double temperature,
                                                         double top_p,
                                                         double min_p,
                                                         unsigned long long position) {
    const int tid = threadIdx.x;
    const float4 sc4 = __ldg(reinterpret_cast<const float4*>(cand_scores + tid * 4));
    const int4 id4 = __ldg(reinterpret_cast<const int4*>(cand_ids + tid * 4));
    float vals[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
    int ids[4] = {id4.x, id4.y, id4.z, id4.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (!isfinite(vals[j])) {
            vals[j] = -INFINITY;
            ids[j] = 0x7fffffff;
        }
    }

    __shared__ float sv[8];
    __shared__ int si[8];
    __shared__ int sidx[8];
    __shared__ float top_scores[64];
    __shared__ int top_ids[64];
    __shared__ int win_idx;

    const int lane = tid & 31, wid = tid >> 5;
    const int k = cand_k < 1 ? 1 : (cand_k > 64 ? 64 : cand_k);

    for (int step = 0; step < k; ++step) {
        float bv = vals[0];
        int bi = ids[0];
        int bidx = (tid << 2) | 0;
#pragma unroll
        for (int j = 1; j < 4; ++j) {
            if (vals[j] > bv || (vals[j] == bv && ids[j] < bi)) {
                bv = vals[j];
                bi = ids[j];
                bidx = (tid << 2) | j;
            }
        }
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, m);
            const int oi = __shfl_down_sync(0xffffffffu, bi, m);
            const int oidx = __shfl_down_sync(0xffffffffu, bidx, m);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
                bidx = oidx;
            }
        }
        if (lane == 0) {
            sv[wid] = bv;
            si[wid] = bi;
            sidx[wid] = bidx;
        }
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            for (int w = 1; w < 8; ++w) {
                if (sv[w] > sv[best] || (sv[w] == sv[best] && si[w] < si[best])) {
                    best = w;
                }
            }
            top_scores[step] = sv[best];
            top_ids[step] = si[best];
            win_idx = sidx[best];
        }
        __syncthreads();
        if (tid == (win_idx >> 2)) {
            vals[win_idx & 3] = -INFINITY;
            ids[win_idx & 3] = 0x7fffffff;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int valid_k = 0;
        while (valid_k < k && isfinite(top_scores[valid_k]) && top_ids[valid_k] != 0x7fffffff) {
            ++valid_k;
        }
        if (valid_k == 0) {
            out_id[0] = 0;
            if (out_conf) {
                out_conf[0] = 0.0f;
            }
        } else {
            float margin = 30.0f;
            const int chosen = gb_keyed::keyed_choose(
                top_scores, top_ids, valid_k, seed, temperature, top_p, min_p, position, &margin);
            if (!isfinite(margin)) margin = -30.0f;
            out_id[0] = top_ids[chosen] != 0x7fffffff ? top_ids[chosen] : 0;
            if (out_conf) {
                out_conf[0] = 1.0f / (1.0f + expf(-margin));
            }
        }
    }
}

// Device-params variant: `cand_k` / `seed` / `temperature` / `top_p` / `min_p` come from the round params block;
// `position` is `base_pos + step` with `step` baked per launch site (0..walk_rows-1), so a graph captured once
// replays with advancing `base_pos` / `seed` without re-capturing.
__global__ void __launch_bounds__(256) keyed_cand_1024_params_k(const float* __restrict__ cand_scores,
                                                                const int* __restrict__ cand_ids,
                                                                int* __restrict__ out_id,
                                                                float* __restrict__ out_conf,
                                                                const Dsv4MtpRoundParams* p,
                                                                unsigned long long step) {
    const int tid = threadIdx.x;
    const float4 sc4 = __ldg(reinterpret_cast<const float4*>(cand_scores + tid * 4));
    const int4 id4 = __ldg(reinterpret_cast<const int4*>(cand_ids + tid * 4));
    float vals[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
    int ids[4] = {id4.x, id4.y, id4.z, id4.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (!isfinite(vals[j])) {
            vals[j] = -INFINITY;
            ids[j] = 0x7fffffff;
        }
    }

    __shared__ float sv[8];
    __shared__ int si[8];
    __shared__ int sidx[8];
    __shared__ float top_scores[64];
    __shared__ int top_ids[64];
    __shared__ int win_idx;

    const int lane = tid & 31, wid = tid >> 5;
    const int k = p->cand_k < 1 ? 1 : (p->cand_k > 64 ? 64 : p->cand_k);
    const unsigned long long seed = p->seed;
    const double temperature = p->temperature;
    const double top_p = p->top_p;
    const double min_p = p->min_p;
    const unsigned long long position = p->base_pos + step;

    for (int step_i = 0; step_i < k; ++step_i) {
        float bv = vals[0];
        int bi = ids[0];
        int bidx = (tid << 2) | 0;
#pragma unroll
        for (int j = 1; j < 4; ++j) {
            if (vals[j] > bv || (vals[j] == bv && ids[j] < bi)) {
                bv = vals[j];
                bi = ids[j];
                bidx = (tid << 2) | j;
            }
        }
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, m);
            const int oi = __shfl_down_sync(0xffffffffu, bi, m);
            const int oidx = __shfl_down_sync(0xffffffffu, bidx, m);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
                bidx = oidx;
            }
        }
        if (lane == 0) {
            sv[wid] = bv;
            si[wid] = bi;
            sidx[wid] = bidx;
        }
        __syncthreads();
        if (tid == 0) {
            int best = 0;
            for (int w = 1; w < 8; ++w) {
                if (sv[w] > sv[best] || (sv[w] == sv[best] && si[w] < si[best])) {
                    best = w;
                }
            }
            top_scores[step_i] = sv[best];
            top_ids[step_i] = si[best];
            win_idx = sidx[best];
        }
        __syncthreads();
        if (tid == (win_idx >> 2)) {
            vals[win_idx & 3] = -INFINITY;
            ids[win_idx & 3] = 0x7fffffff;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int valid_k = 0;
        while (valid_k < k && isfinite(top_scores[valid_k]) && top_ids[valid_k] != 0x7fffffff) {
            ++valid_k;
        }
        if (valid_k == 0) {
            out_id[0] = 0;
            if (out_conf) {
                out_conf[0] = 0.0f;
            }
        } else {
            float margin = 30.0f;
            const int chosen = gb_keyed::keyed_choose(
                top_scores, top_ids, valid_k, seed, temperature, top_p, min_p, position, &margin);
            if (!isfinite(margin)) margin = -30.0f;
            out_id[0] = top_ids[chosen] != 0x7fffffff ? top_ids[chosen] : 0;
            if (out_conf) {
                out_conf[0] = 1.0f / (1.0f + expf(-margin));
            }
        }
    }
}

}  // namespace

extern "C" {

GB_EXPORT int gb_dsv4_mtp_markov_bias_bf16(void* logits, const void* tokens, const void* embed, const void* head,
                                           int rows, int vocab, int rank, void* stream) {
    if (!logits || !tokens || !embed || !head || rows <= 0 || vocab <= 0 || rank <= 0 || rank > 256) return ERR_INVALID;
    if (rank == 256) {
        // Use pointer alignment / step parity from tokens pointer ((uintptr_t)tokens >> 2) & 1 for boustrophedon L2 reuse!
        const int dir = (int)(((uintptr_t)tokens >> 2) & 1u);
        const dim3 grid((vocab + 31) / 32, rows);
        markov_bias_bf16_4_k<<<grid, 256, 0, gb_stream(stream)>>>((float*)logits, (const int*)tokens, (const u16*)embed,
                                                                  (const u16*)head, vocab, dir);
        return GB_CHECK_LAUNCH();
    }
    const dim3 grid((vocab + 7) / 8, rows);
    markov_bias_bf16_k<<<grid, 256, 0, gb_stream(stream)>>>((float*)logits, (const int*)tokens, (const u16*)embed,
                                                            (const u16*)head, vocab, rank);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_gather_rows_bf16(float* dst, const int* ids, const void* table, int rows, int width,
                                           int vocab, void* stream) {
    if (!dst || !ids || !table || rows <= 0 || width <= 0 || vocab <= 0) return ERR_INVALID;
    gather_rows_bf16_k<<<rows, 128, 0, gb_stream(stream)>>>(dst, ids, (const u16*)table, rows, width, vocab);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_confidence_bf16(const void* h, const void* m, const void* w, const void* b, void* out,
                                          int rows, int dim, int rank, void* stream) {
    if (!h || !m || !w || !b || !out || rows <= 0 || dim <= 0 || rank <= 0) return ERR_INVALID;
    confidence_bf16_k<<<rows, 256, 0, gb_stream(stream)>>>((const float*)h, (const float*)m, (const u16*)w,
                                                           (const float*)b, (float*)out, dim, rank, 0);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_confidence_keyed_bf16(const void* h, const void* m, const void* w, const void* b, void* out,
                                                int rows, int dim, int rank, int walk_rows, void* stream) {
    if (!h || !m || !w || !b || !out || rows <= 0 || dim <= 0 || rank <= 0) return ERR_INVALID;
    confidence_bf16_k<<<rows, 256, 0, gb_stream(stream)>>>((const float*)h, (const float*)m, (const u16*)w,
                                                           (const float*)b, (float*)out, dim, rank, walk_rows);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_tap_mean_bf16(const float* streams, void* taps_bf16, int rows, int dim, int tap_idx,
                                        int n_taps, void* stream) {
    if (!streams || !taps_bf16 || rows <= 0 || dim <= 0 || tap_idx < 0 || tap_idx >= n_taps) return ERR_INVALID;
    const dim3 grid((dim + 255) / 256, rows);
    tap_mean_bf16_k<<<grid, 256, 0, gb_stream(stream)>>>(streams, (u16*)taps_bf16, rows, dim, tap_idx, n_taps);
    return GB_CHECK_LAUNCH();
}

// The host reference's rate-optimal verify-length schedule (`DsparkGpu::schedule_verify_len`), as a single
// device-side pass over the round's GPU-resident confidences. The loop below is the host function's loop with the
// same clamp, the same hard floor break, the same expected-token accumulation order, the same cycle table and the
// same `>=` tie-break, so the result is bit-for-bit the host's. One thread; the loop is ≤ 8 iterations.
__global__ void schedule_verify_len_k(const float* __restrict__ conf, int conf_len, float min_conf,
                                      int max_drafts, const float* __restrict__ cycle_ms, int* __restrict__ out) {
    int cap = conf_len < max_drafts ? conf_len : max_drafts;
    if (cap <= 0) {
        out[0] = 0;
        return;
    }
    float cum_prod = 1.0f;
    float expected_tokens = 1.0f;
    int best_l = 1;
    float best_rate = 0.0f;
    for (int l = 1; l <= cap; ++l) {
        float c = conf[l - 1];
        c = fminf(fmaxf(c, 0.0f), 1.0f);
        if (l > 1 && c < min_conf) break;
        cum_prod *= c;
        expected_tokens += cum_prod;
        int ti = l < 8 ? l : 7;  // VERIFY_CYCLE_MS[l.min(len - 1)] with len = 8
        float rate = expected_tokens / cycle_ms[ti];
        if (l == 1 || rate >= best_rate) {
            best_rate = rate;
            best_l = l;
        }
    }
    out[0] = best_l;
}

GB_EXPORT int gb_dsv4_mtp_schedule_verify_len(const float* conf, int conf_len, float min_conf, int max_drafts,
                                              const float* cycle_ms, int* out, void* stream) {
    if (!conf || !cycle_ms || !out || conf_len < 0 || max_drafts < 0) return ERR_INVALID;
    schedule_verify_len_k<<<1, 1, 0, gb_stream(stream)>>>(conf, conf_len, min_conf, max_drafts, cycle_ms, out);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_markov_walk_fast_bf16(void* logits, int* ids, const void* embed, const void* head,
                                                void* cand_ids_ws, void* cand_scores_ws,
                                                int walk_rows, int vocab, int rank, void* stream) {
    if (!logits || !ids || !embed || !head || !cand_ids_ws || !cand_scores_ws) return ERR_INVALID;
    if (walk_rows <= 0 || vocab < 1024 || rank != 256) return ERR_INVALID;
    cudaStream_t st = gb_stream(stream);
    float* log_f = static_cast<float*>(logits);
    int* cand_ids = static_cast<int*>(cand_ids_ws);
    float* cand_scores = static_cast<float*>(cand_scores_ws);
    mtp_topk_candidates_4_k<<<(unsigned)walk_rows, 256, 0, st>>>(log_f, cand_ids, vocab);
    for (int k = 0; k < walk_rows; ++k) {
        markov_bias_cand_bf16_4_k<<<32, 256, 0, st>>>(
            log_f + (size_t)k * vocab,
            ids + k,
            static_cast<const u16*>(embed),
            static_cast<const u16*>(head),
            cand_ids + (size_t)k * 1024,
            cand_scores,
            vocab);
        argmax_cand_1024_k<<<1, 256, 0, st>>>(cand_scores, cand_ids + (size_t)k * 1024, ids + k + 1);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_markov_walk_keyed_bf16(void* logits, int* ids, float* conf,
                                                 const void* embed, const void* head,
                                                 void* cand_ids_ws, void* cand_scores_ws,
                                                 int walk_rows, int vocab, int rank,
                                                 int cand_k, unsigned long long seed,
                                                 double temperature, double top_p, double min_p,
                                                 unsigned long long base_pos, void* stream) {
    if (!logits || !ids || !embed || !head || !cand_ids_ws || !cand_scores_ws) return ERR_INVALID;
    if (walk_rows <= 0 || vocab < 1024 || rank != 256 || cand_k < 1) return ERR_INVALID;
    cudaStream_t st = gb_stream(stream);
    float* log_f = static_cast<float*>(logits);
    int* cand_ids = static_cast<int*>(cand_ids_ws);
    float* cand_scores = static_cast<float*>(cand_scores_ws);
    mtp_topk_candidates_4_k<<<(unsigned)walk_rows, 256, 0, st>>>(log_f, cand_ids, vocab);
    for (int k = 0; k < walk_rows; ++k) {
        markov_bias_cand_bf16_4_k<<<32, 256, 0, st>>>(
            log_f + (size_t)k * vocab,
            ids + k,
            static_cast<const u16*>(embed),
            static_cast<const u16*>(head),
            cand_ids + (size_t)k * 1024,
            cand_scores,
            vocab);
        keyed_cand_1024_k<<<1, 256, 0, st>>>(
            cand_scores,
            cand_ids + (size_t)k * 1024,
            ids + k + 1,
            conf ? (conf + k) : nullptr,
            cand_k,
            seed,
            temperature,
            top_p,
            min_p,
            base_pos + (unsigned long long)k);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_mtp_markov_walk_keyed_bf16_params(void* logits, int* ids, float* conf,
                                                        const void* embed, const void* head,
                                                        void* cand_ids_ws, void* cand_scores_ws,
                                                        int walk_rows, int vocab, int rank,
                                                        const void* params, void* stream) {
    if (!logits || !ids || !embed || !head || !cand_ids_ws || !cand_scores_ws || !params) return ERR_INVALID;
    if (walk_rows <= 0 || vocab < 1024 || rank != 256) return ERR_INVALID;
    // NOTE: `params` is a DEVICE pointer; never dereference it on the host (the kernels read it on the device).
    const Dsv4MtpRoundParams* p = (const Dsv4MtpRoundParams*)params;
    cudaStream_t st = gb_stream(stream);
    float* log_f = static_cast<float*>(logits);
    int* cand_ids = static_cast<int*>(cand_ids_ws);
    float* cand_scores = static_cast<float*>(cand_scores_ws);
    mtp_topk_candidates_4_k<<<(unsigned)walk_rows, 256, 0, st>>>(log_f, cand_ids, vocab);
    for (int k = 0; k < walk_rows; ++k) {
        markov_bias_cand_bf16_4_k<<<32, 256, 0, st>>>(
            log_f + (size_t)k * vocab,
            ids + k,
            static_cast<const u16*>(embed),
            static_cast<const u16*>(head),
            cand_ids + (size_t)k * 1024,
            cand_scores,
            vocab);
        keyed_cand_1024_params_k<<<1, 256, 0, st>>>(
            cand_scores,
            cand_ids + (size_t)k * 1024,
            ids + k + 1,
            conf ? (conf + k) : nullptr,
            p,
            (unsigned long long)k);
    }
    return GB_CHECK_LAUNCH();
}

}  // extern "C"
