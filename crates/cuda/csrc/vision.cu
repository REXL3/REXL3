// gb-nvcc: --expt-relaxed-constexpr
// Kernels of the Qwen3.8 vision tower (docs/VISION.md): position-embedding add, LayerNorm with bias, axial 2D RoPE, residual
// add, tanh GELU, fp32 -> bf16 export, and a bidirectional flash attention for head_dim 72.
//
// Conventions: row-major; fp32 residual stream `x`, fp16 GEMM operands and activations. The numerics follow ExLlamaV3's
// vision modules (MIT) so the comparison against its dumps is tight: the position-embedding sum runs in bf16 exactly as
// `Qwen3VLPosEmbedding` does (weights rounded to bf16, each product and each add rounded to bf16, then added to the fp32
// stream), LayerNorm uses fp32 statistics, RoPE is NEOX (pairs d, d+36) in fp32, softmax is fp32 with fp16 P.
// Nothing here has to be row invariant: the encoder runs one image at a time, and its output is not part of the
// serial-equals-drafted contract (docs/CUDA_BACKEND.md), but every kernel is deterministic (fixed order, no atomics).

#include <stdint.h>

#include "common.cuh"

namespace {

constexpr int HD = 72;          // vision head dim
constexpr int HDP = 80;         // padded to 5 mma k-steps of 16

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// x [P, H] fp32 += bias [H] (fp16), then += the bf16 bilinear position embedding of the patch:
// ((t0 + t1) + t2) + t3 with t_i = bf16(table[idx_i] * w_i), every operation rounded to bf16 as in ExLlamaV3.
// table: fp32 values that are exactly bf16 (the pack's bf16 table widened); idx i32 [P, 4]; w fp32 [P, 4] already bf16-rounded.
__global__ void pos_add_kernel(float* __restrict__ x, const half* __restrict__ bias, const float* __restrict__ table,
                               const int* __restrict__ idx, const float* __restrict__ w, int H) {
    const int p = blockIdx.x;
    const int i0 = idx[p * 4], i1 = idx[p * 4 + 1], i2 = idx[p * 4 + 2], i3 = idx[p * 4 + 3];
    const float w0 = w[p * 4], w1 = w[p * 4 + 1], w2 = w[p * 4 + 2], w3 = w[p * 4 + 3];
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        const float t0 = bf16r(table[(size_t)i0 * H + c] * w0);
        const float t1 = bf16r(table[(size_t)i1 * H + c] * w1);
        const float t2 = bf16r(table[(size_t)i2 * H + c] * w2);
        const float t3 = bf16r(table[(size_t)i3 * H + c] * w3);
        const float s = bf16r(bf16r(bf16r(t0 + t1) + t2) + t3);
        float v = x[(size_t)p * H + c];
        if (bias) v += __half2float(bias[c]);
        x[(size_t)p * H + c] = v + s;
    }
}

// y [rows, dim] fp16 = LayerNorm(x) * w + b, fp32 statistics (two passes over registers), biased variance. x fp32 or fp16.
template <bool XF32>
__global__ void layernorm_kernel(const void* __restrict__ xv, const half* __restrict__ w, const half* __restrict__ b,
                                 half* __restrict__ y, int dim, float eps) {
    const int row = blockIdx.x;
    extern __shared__ float red[];  // 2 * (blockDim / 32)
    const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    auto load = [&](int c) -> float {
        return XF32 ? ((const float*)xv)[(size_t)row * dim + c] : __half2float(((const half*)xv)[(size_t)row * dim + c]);
    };
    float s = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) s += load(c);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    float mean = 0.f;
    for (int i = 0; i < nw; ++i) mean += red[i];
    mean /= (float)dim;
    float v = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const float d = load(c) - mean;
        v += d * d;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[nw + warp] = v;
    __syncthreads();
    float var = 0.f;
    for (int i = 0; i < nw; ++i) var += red[nw + i];
    const float inv = rsqrtf(var / (float)dim + eps);
    for (int c = threadIdx.x; c < dim; c += blockDim.x)
        y[(size_t)row * dim + c] = __float2half_rn((load(c) - mean) * inv * __half2float(w[c]) + __half2float(b[c]));
}

// NEOX RoPE over all 72 dims of every head of q and k [P, heads*72] fp16, in place: for d < 36,
// (x[d], x[d+36]) -> (x[d]*cos - x[d+36]*sin, x[d+36]*cos + x[d]*sin) with cs [P, 2, 36] = cos then sin of the patch's angle.
__global__ void rope2d_kernel(half* __restrict__ q, half* __restrict__ k, const float* __restrict__ cs, int heads) {
    const int p = blockIdx.x;
    const int stride = heads * HD;
    for (int i = threadIdx.x; i < heads * 36; i += blockDim.x) {
        const int h = i / 36, d = i % 36;
        const float c = cs[p * 72 + d], s = cs[p * 72 + 36 + d];
        const size_t at = (size_t)p * stride + h * HD + d;
        float a = __half2float(q[at]), b = __half2float(q[at + 36]);
        q[at] = __float2half_rn(a * c - b * s);
        q[at + 36] = __float2half_rn(b * c + a * s);
        a = __half2float(k[at]);
        b = __half2float(k[at + 36]);
        k[at] = __float2half_rn(a * c - b * s);
        k[at + 36] = __float2half_rn(b * c + a * s);
    }
}

__global__ void add_f16_kernel(float* __restrict__ x, const half* __restrict__ y, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        x[i] += __half2float(y[i]);
}

// tanh-approximate GELU in place on fp16, computed in fp32 (F.gelu(x, approximate="tanh") on a half tensor).
__global__ void gelu_tanh_kernel(half* __restrict__ x, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = __half2float(x[i]);
        const float t = tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v));
        x[i] = __float2half_rn(0.5f * v * (1.f + t));
    }
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = __float2bfloat16_rn(x[i]);
}

// ------------------------------------------------------------------------------------------------------------------
// Bidirectional flash attention, head_dim 72 (padded to 80 for the mma k-steps), fp16 in, fp32 accumulate.
// q, k, v: [P, heads*72] fp16 (row stride heads*72); out: same layout. One CTA = 4 warps = 64 query rows of one head;
// keys stream through shared memory in tiles of 64 (K row-major, V transposed so the P.V B-fragments are contiguous).
// ------------------------------------------------------------------------------------------------------------------
constexpr int BM = 64, BN = 64, KS = HDP + 8, VS = BN + 8;

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__global__ void __launch_bounds__(128) attn_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                                   const half* __restrict__ v, half* __restrict__ out, int P, int heads,
                                                   float scale_log2) {
    __shared__ __align__(16) half Ks[BN * KS];
    __shared__ __align__(16) half Vt[HDP * VS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int stride = heads * HD;
    const int head = blockIdx.y;
    const int q0 = blockIdx.x * BM + warp * 16;
    const int r0 = q0 + g, r1 = q0 + g + 8;

    // Q fragments for the 5 k-steps (zero past d = 72 and past the last row)
    uint32_t qa[5][4];
#pragma unroll
    for (int ks = 0; ks < 5; ++ks) {
        const int c0 = ks * 16 + 2 * t, c1 = c0 + 8;
        auto ld = [&](int row, int col) -> uint32_t {
            if (row >= P || col >= HD) return 0u;
            return *reinterpret_cast<const uint32_t*>(q + (size_t)row * stride + head * HD + col);
        };
        qa[ks][0] = ld(r0, c0);
        qa[ks][1] = ld(r1, c0);
        qa[ks][2] = ld(r0, c1);
        qa[ks][3] = ld(r1, c1);
    }

    float o[10][4];
#pragma unroll
    for (int i = 0; i < 10; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int tiles = (P + BN - 1) / BN;
    for (int kt = 0; kt < tiles; ++kt) {
        __syncthreads();  // the previous tile's readers are done
        // load K [64][80] row-major and V transposed [80][64]; pad columns and keys past P with zeros
        for (int i = tid; i < BN * (HDP / 2); i += 128) {
            const int key = i / (HDP / 2), c2 = i % (HDP / 2), gk = kt * BN + key, d = c2 * 2;
            uint32_t kv = 0u, vv = 0u;
            if (gk < P && d < HD) {
                kv = *reinterpret_cast<const uint32_t*>(k + (size_t)gk * stride + head * HD + d);
                vv = *reinterpret_cast<const uint32_t*>(v + (size_t)gk * stride + head * HD + d);
            }
            *reinterpret_cast<uint32_t*>(Ks + key * KS + d) = kv;
            const __half2 vh = *reinterpret_cast<const __half2*>(&vv);
            Vt[d * VS + key] = __low2half(vh);
            Vt[(d + 1) * VS + key] = __high2half(vh);
        }
        __syncthreads();

        // S = Q K^T over 8 key groups of 8
        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int ks = 0; ks < 5; ++ks) {
                const half* kr = Ks + (nt * 8 + g) * KS + ks * 16 + 2 * t;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr + 8);
                mma16816(s[nt], qa[ks], b0, b1);
            }
        }
        // scale (folded with log2 e), mask keys past P, tile row maxima
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int key = kt * BN + nt * 8 + 2 * t;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool valid = key + (e & 1) < P;
                s[nt][e] = valid ? s[nt][e] * scale_log2 : -INFINITY;
            }
            tm0 = fmaxf(tm0, fmaxf(s[nt][0], s[nt][1]));
            tm1 = fmaxf(tm1, fmaxf(s[nt][2], s[nt][3]));
        }
#pragma unroll
        for (int o2 = 1; o2 <= 2; o2 <<= 1) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, o2));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, o2));
        }
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);  // exp2(-inf) = 0 on the first tile
        m0 = n0;
        m1 = n1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int i = 0; i < 10; ++i) {
            o[i][0] *= c0;
            o[i][1] *= c0;
            o[i][2] *= c1;
            o[i][3] *= c1;
        }
        // P = exp2(S - m), row sums (per-thread partials, reduced over the quad at the end)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - n0);
            s[nt][1] = exp2f(s[nt][1] - n0);
            s[nt][2] = exp2f(s[nt][2] - n1);
            s[nt][3] = exp2f(s[nt][3] - n1);
            l0 += s[nt][0] + s[nt][1];
            l1 += s[nt][2] + s[nt][3];
        }
        // O += P V: P's C-fragment layout is the A-fragment layout of the next mma (two key groups a 16-key step)
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
            uint32_t pa[4];
            pa[0] = pack_h2(s[2 * kc][0], s[2 * kc][1]);
            pa[1] = pack_h2(s[2 * kc][2], s[2 * kc][3]);
            pa[2] = pack_h2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[3] = pack_h2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
#pragma unroll
            for (int dn = 0; dn < 10; ++dn) {
                const half* vr = Vt + (dn * 8 + g) * VS + kc * 16 + 2 * t;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(vr);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(vr + 8);
                mma16816(o[dn], pa, b0, b1);
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
    for (int dn = 0; dn < 9; ++dn) {  // 9 x 8 = 72 real columns
        const int col = head * HD + dn * 8 + 2 * t;
        if (r0 < P) *reinterpret_cast<uint32_t*>(out + (size_t)r0 * stride + col) = pack_h2(o[dn][0] * i0, o[dn][1] * i0);
        if (r1 < P) *reinterpret_cast<uint32_t*>(out + (size_t)r1 * stride + col) = pack_h2(o[dn][2] * i1, o[dn][3] * i1);
    }
}

int launch_cfg_ok(int rows, int dim) { return rows >= 0 && dim > 0; }

}  // namespace

GB_EXPORT int gb_vis_pos_add(void* x, const void* bias, const void* table, const void* idx, const void* w, int P, int H,
                             void* stream) {
    if (P < 0 || H <= 0) return (int)cudaErrorInvalidValue;
    if (P == 0) return 0;
    pos_add_kernel<<<P, 256, 0, gb_stream(stream)>>>((float*)x, (const half*)bias, (const float*)table, (const int*)idx,
                                                     (const float*)w, H);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_vis_layernorm(const void* x, int x_is_f32, const void* w, const void* b, void* y, int rows, int dim,
                               float eps, void* stream) {
    if (!launch_cfg_ok(rows, dim)) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const int threads = 256;
    const size_t smem = 2 * (threads / 32) * sizeof(float);
    if (x_is_f32)
        layernorm_kernel<true><<<rows, threads, smem, gb_stream(stream)>>>(x, (const half*)w, (const half*)b, (half*)y, dim, eps);
    else
        layernorm_kernel<false><<<rows, threads, smem, gb_stream(stream)>>>(x, (const half*)w, (const half*)b, (half*)y, dim, eps);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_vis_rope2d(void* q, void* k, const void* cs, int P, int heads, void* stream) {
    if (P < 0 || heads <= 0) return (int)cudaErrorInvalidValue;
    if (P == 0) return 0;
    rope2d_kernel<<<P, 256, 0, gb_stream(stream)>>>((half*)q, (half*)k, (const float*)cs, heads);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_vis_add_f16(void* x, const void* y, size_t n, void* stream) {
    if (n == 0) return 0;
    const unsigned blocks = (unsigned)((n + 255) / 256 < 65535 ? (n + 255) / 256 : 65535);
    add_f16_kernel<<<blocks, 256, 0, gb_stream(stream)>>>((float*)x, (const half*)y, n);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_vis_gelu_tanh(void* x, size_t n, void* stream) {
    if (n == 0) return 0;
    const unsigned blocks = (unsigned)((n + 255) / 256 < 65535 ? (n + 255) / 256 : 65535);
    gelu_tanh_kernel<<<blocks, 256, 0, gb_stream(stream)>>>((half*)x, n);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_vis_f32_to_bf16(const void* x, void* y, size_t n, void* stream) {
    if (n == 0) return 0;
    const unsigned blocks = (unsigned)((n + 255) / 256 < 65535 ? (n + 255) / 256 : 65535);
    f32_to_bf16_kernel<<<blocks, 256, 0, gb_stream(stream)>>>((const float*)x, (__nv_bfloat16*)y, n);
    return GB_CHECK_LAUNCH();
}

// q, k, v, out: [P, heads*72] fp16, 4-byte aligned. Full (non-causal) attention among the P patches of one image.
GB_EXPORT int gb_vis_attention(const void* q, const void* k, const void* v, void* out, int P, int heads, void* stream) {
    if (P < 0 || heads <= 0) return (int)cudaErrorInvalidValue;
    if (P == 0) return 0;
    const float scale_log2 = 1.4426950408889634f / sqrtf((float)HD);
    attn_kernel<<<dim3((P + BM - 1) / BM, heads), 128, 0, gb_stream(stream)>>>((const half*)q, (const half*)k, (const half*)v,
                                                                              (half*)out, P, heads, scale_log2);
    return GB_CHECK_LAUNCH();
}
