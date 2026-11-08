// Kernels of the GLM-5.3-Flash vision tower (docs/VISION.md, "GLM-5.3-Flash vision tower"): hidden 1024, 16 heads of 64, a SwiGLU MLP of 4096, a 2x2
// merge to 4096 and a merger with an MLP of 10240. The numerics follow the ExLlamaV3 fork's `glm5_next_vision` branch (MIT), the oracle
// (`tools/dump_glm_vision_reference.py`): where the fork does a step in fp16 so do we, rounding where it rounds, so that the per-stage
// comparison against its dumps measures noise and not a convention.
//
//   * Linear and Conv weights and biases are fp16 (the fork casts the pack's bf16 to fp16 at load); RMSNorm gammas stay bf16 and are applied in
//     fp32. The residual stream is fp32. GEMM outputs are fp16 and a bias is added in fp16 afterwards (`y += bias` on an fp16 tensor), here fused
//     into the next kernel that reads the output.
//   * Head RMSNorm of q and k and the NEOX rope: the fork's rope kernel (`exllamav3_ext/rope.cu`): fp16 in, RMSNorm in fp32 with `(x * rmf) * w`
//     rounded once to fp16, then the rotation in fp32 with `__sinf` / `__cosf` of the angle table, rounded once.
//   * Block MLP: `silu(g) * u` in half intrinsics (`h2exp`, `h2rcp`, `__hmul2`), silu(g) clamped to at most the limit, u to +-limit
//     (`activation_kernels.cuh`, `act_mul_kernel_h`). Merger MLP: the HF order, g clamped BEFORE the silu, which runs in fp32 (ATen) and rounds
//     to fp16, then u * s rounded to fp16; GELU is exact (erf).
//   * Attention: bidirectional flash attention, head dim 64, fp16 in, fp32 accumulate, P rounded to fp16 for the second product.
//
// Everything is deterministic (fixed order, no atomics on data) and is run one image at a time: nothing here is row invariant across batchings and
// nothing has to be (an image's features are a function of that image alone, docs/CUDA_BACKEND.md). Default nvcc flags: the fork's extension is
// built with --use_fast_math, so the few fast intrinsics it relies on (`__sinf`, `__cosf`, `rsqrtf`, the half2 math) are named here explicitly and
// everything that ATen computes (the LayerNorm, GELU and the merger's silu) uses the accurate functions.

#include <cublas_v2.h>
#include <math.h>
#include <stdint.h>

#include "common.cuh"

namespace {

constexpr int D = 1024;       // hidden
constexpr int HEADS = 16;
constexpr int HD = 64;        // head dim
constexpr int QKV = 3 * D;    // fused qkv row

constexpr int ERR_INVALID = (int)cudaErrorInvalidValue;

__device__ __forceinline__ float h2f(half h) { return __half2float(h); }
__device__ __forceinline__ float bf2f(__nv_bfloat16 b) { return __bfloat162float(b); }
// fp16 `a + b` as torch does it on half tensors: both exact in fp32, one rounding
__device__ __forceinline__ half hadd(half a, half b) { return __float2half_rn(__half2float(a) + __half2float(b)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// ------------------------------------------------------------------------------------------------------------------
// Weight preparation (run once at load, in place where the byte counts allow).
// ------------------------------------------------------------------------------------------------------------------

// bf16 -> fp16 in place (round to nearest even, as `tensor.to(torch.float16)`); counts results that are not finite.
__global__ void bf16_to_f16_k(uint16_t* p, size_t n, int* bad) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float f = __uint_as_float(((uint32_t)p[i]) << 16);
        const half h = __float2half_rn(f);
        const float back = __half2float(h);
        if (!isfinite(back)) atomicAdd(bad, 1);
        p[i] = __half_as_ushort(h);
    }
}

// The patch embedding conv3d weight [O, 3, 2, 14, 14] (bf16) folded over the two temporal slots into [O, KP] fp16 (K = 3 * 196 = 588 real columns,
// zero to KP = 592 so the GEMM's leading dimension is a multiple of 8): w = fp16(w0) + fp16(w1), summed in fp32 and rounded once. `slots`: 2 folds,
// 1 takes slot 0 only (a mutation of the tests).
__global__ void fold_patch_k(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int O, int KP, int slots) {
    const int K = 3 * 196;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < O * KP; i += gridDim.x * blockDim.x) {
        const int o = i / KP, k = i % KP;
        float acc = 0.f;
        if (k < K) {
            const int c = k / 196, r = k % 196;
            const size_t base = (size_t)o * (3 * 2 * 196) + (size_t)c * (2 * 196) + r;
            for (int t = 0; t < slots; ++t) {
                const float f = __uint_as_float(((uint32_t)src[base + t * 196]) << 16);
                acc += __half2float(__float2half_rn(f));
            }
        }
        dst[i] = __half_as_ushort(__float2half_rn(acc));
    }
}

// The downsample conv2d weight [O, C, 2, 2] (bf16) as a GEMM weight [O, 4C] fp16 over four consecutive block-major patches. Row of the GEMM input:
// (patch j = kh * 2 + kw) * C + c. `mode` 0: that order; 1: the conv's own flattening (c * 4 + j: wrong for this input, a mutation of the tests);
// 2: the two patches of the diagonal swapped (kh and kw exchanged: a transposed merge block, a mutation).
__global__ void permute_downsample_k(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int O, int C, int mode, int* bad) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)O * C * 4; i += (size_t)gridDim.x * blockDim.x) {
        const int o = (int)(i / (C * 4));
        const int rem = (int)(i % (C * 4));
        const int c = rem / 4, j = rem % 4;
        const int kh = j / 2, kw = j % 2;
        int dstk;
        if (mode == 0) dstk = j * C + c;
        else if (mode == 1) dstk = c * 4 + j;
        else dstk = (kw * 2 + kh) * C + c;
        const float f = __uint_as_float(((uint32_t)src[i]) << 16);
        const half h = __float2half_rn(f);
        if (!isfinite(__half2float(h))) atomicAdd(bad, 1);
        dst[(size_t)o * C * 4 + dstk] = __half_as_ushort(h);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// RMSNorm of the residual stream, optionally fused with the sublayer's residual add.
// ------------------------------------------------------------------------------------------------------------------

// y[row] = f16((x * w) * rmf), x fp32 [rows, D], w bf16 [D], rmf = rsqrt(mean(x^2) + eps) (exllamav3_ext/norm.cu: `x4.x * w4.x * rmf`, the sum of
// squares accumulated with fma). With ADD: first x += f16(raw + bias) (the fork's `y = mlp.forward(); x += y`, `y` an fp16 tensor with the bias already
// added in fp16), x written back, then normalised. 256 threads, one float4 each.
template <bool ADD>
__global__ void __launch_bounds__(256)
    rms_k(float* __restrict__ x, const half* __restrict__ raw, const half* __restrict__ bias, const __nv_bfloat16* __restrict__ w,
          half* __restrict__ y, float eps) {
    const int row = blockIdx.x, t = threadIdx.x;
    float4 v = *(const float4*)(x + (size_t)row * D + t * 4);
    if (ADD) {
        const uint2 r = *(const uint2*)(raw + (size_t)row * D + t * 4);
        const half2* r2 = (const half2*)&r;
        float rv[4] = {h2f(__low2half(r2[0])), h2f(__high2half(r2[0])), h2f(__low2half(r2[1])), h2f(__high2half(r2[1]))};
        if (bias) {
            const uint2 b = *(const uint2*)(bias + t * 4);
            const half2* b2 = (const half2*)&b;
            const float bv[4] = {h2f(__low2half(b2[0])), h2f(__high2half(b2[0])), h2f(__low2half(b2[1])), h2f(__high2half(b2[1]))};
#pragma unroll
            for (int i = 0; i < 4; ++i) rv[i] = h2f(__float2half_rn(rv[i] + bv[i]));
        }
        v.x += rv[0];
        v.y += rv[1];
        v.z += rv[2];
        v.w += rv[3];
        *(float4*)(x + (size_t)row * D + t * 4) = v;
    }
    float ss = fmaf(v.x, v.x, 0.f);
    ss = fmaf(v.y, v.y, ss);
    ss = fmaf(v.z, v.z, ss);
    ss = fmaf(v.w, v.w, ss);
    ss = warp_sum(ss);
    __shared__ float red[8];
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float rmf = rsqrtf(tot / (float)D + eps);
    const uint2 wr = *(const uint2*)(w + t * 4);
    const __nv_bfloat162* w2 = (const __nv_bfloat162*)&wr;
    const float wv[4] = {bf2f(__low2bfloat16(w2[0])), bf2f(__high2bfloat16(w2[0])), bf2f(__low2bfloat16(w2[1])), bf2f(__high2bfloat16(w2[1]))};
    const half o0 = __float2half_rn(v.x * wv[0] * rmf), o1 = __float2half_rn(v.y * wv[1] * rmf), o2 = __float2half_rn(v.z * wv[2] * rmf),
               o3 = __float2half_rn(v.w * wv[3] * rmf);
    uint2 out;
    half2* o2p = (half2*)&out;
    o2p[0] = __halves2half2(o0, o1);
    o2p[1] = __halves2half2(o2, o3);
    *(uint2*)(y + (size_t)row * D + t * 4) = out;
}

// ------------------------------------------------------------------------------------------------------------------
// The qkv GEMM's output (fp16 [P, 3D], q | k | v, no bias yet) -> q, k, v in place: bias in fp16, then for q and k the per-head RMSNorm and the NEOX
// rope with the 2-D angle table, for v the bias only. One CTA per patch, one warp per head, lane l holds dims l and l + 32 (the rope pair).
// flags: 1 swap the row and column axes of the angles, 2 skip the bias, 4 skip the head norm, 8 skip the rope.
// ------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(128)
    qkv_prep_k(half* __restrict__ qkv, const half* __restrict__ bias, const __nv_bfloat16* __restrict__ qn, const __nv_bfloat16* __restrict__ kn,
               const float* __restrict__ inv_freq, int gw, float eps, int flags) {
    const int p = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    half* row = qkv + (size_t)p * QKV;
    // block-major patch p: merge block b = p / 4 and the patch inside it
    const int bw = gw / 2;
    const int blk = p >> 2, w4 = p & 3;
    const int r = (blk / bw) * 2 + (w4 >> 1), c = (blk % bw) * 2 + (w4 & 1);
    const int ra = (flags & 1) ? c : r, ca = (flags & 1) ? r : c;
    const bool use_bias = !(flags & 2);
    for (int head = warp; head < 2 * HEADS; head += 4) {
        const bool is_q = head < HEADS;
        half* hp = row + (is_q ? head : head) * HD;  // q: [0, D), k: [D, 2D): head * HD already spans both
        const __nv_bfloat16* nw = is_q ? qn : kn;
        float a1 = h2f(hp[lane]), a2 = h2f(hp[lane + 32]);
        if (use_bias) {
            a1 = h2f(__float2half_rn(a1 + h2f(bias[head * HD + lane])));
            a2 = h2f(__float2half_rn(a2 + h2f(bias[head * HD + lane + 32])));
        }
        float n1 = a1, n2 = a2;
        if (!(flags & 4)) {
            const float ss = warp_sum(a1 * a1 + a2 * a2);
            const float rmf = rsqrtf(ss / (float)HD + eps);
            n1 = h2f(__float2half_rn(a1 * rmf * bf2f(nw[lane])));
            n2 = h2f(__float2half_rn(a2 * rmf * bf2f(nw[lane + 32])));
            // (x * rmf) * w: the fork's `v1 *= rmf; v1 *= w`
        }
        float o1 = n1, o2 = n2;
        if (!(flags & 8)) {
            const float ang = lane < 16 ? (float)ra * inv_freq[lane] : (float)ca * inv_freq[lane - 16];
            const float sn = __sinf(ang), cs = __cosf(ang);
            o1 = n1 * cs - n2 * sn;
            o2 = n2 * cs + n1 * sn;
        }
        hp[lane] = __float2half_rn(o1);
        hp[lane + 32] = __float2half_rn(o2);
    }
    // v: the bias alone
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        half v = row[2 * D + i];
        if (use_bias) v = hadd(v, bias[2 * D + i]);
        row[2 * D + i] = v;
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Block MLP activation in place on gu [P, ld]: g = cols [0, I), u = cols [I, 2I) of the two GEMMs, biases added in fp16, then the half-precision
// `act_mul_kernel_h<ACT_SILU>` of the fork: silu in half2 intrinsics, silu(g) = min(silu(g), limit), u clamped to +-limit, product in half2. The result
// overwrites g. flags 1: skip the clamps; 2: clamp g before the silu (the merger's order, a mutation here); 4: skip the biases (a mutation).
// ------------------------------------------------------------------------------------------------------------------
__global__ void swiglu_block_k(half* __restrict__ gu, const half* __restrict__ bg, const half* __restrict__ bu, int I, int ld, float limit, int flags,
                               size_t rows) {
    const size_t n2 = rows * (size_t)(I / 2);
    const half2 one = __float2half2_rn(1.0f), lim = __float2half2_rn(limit), nlim = __float2half2_rn(-limit);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / (I / 2);
        const int c2 = (int)(i % (I / 2));
        half2* gp = (half2*)(gu + r * ld) + c2;
        half2* up = (half2*)(gu + r * ld + I) + c2;
        half2 x2 = (flags & 4) ? *gp : __hadd2(*gp, ((const half2*)bg)[c2]);
        half2 y2 = (flags & 4) ? *up : __hadd2(*up, ((const half2*)bu)[c2]);
        if (flags & 2) x2 = __hmin2(x2, lim);
        const half2 e = h2exp(__hneg2(x2));
        const half2 s = __hadd2(one, e);
        const half2 rcp = h2rcp(s);
        x2 = __hmul2(x2, rcp);
        if (!(flags & 1)) {
            y2 = __hmax2(y2, nlim);
            y2 = __hmin2(y2, lim);
            if (!(flags & 2)) x2 = __hmin2(x2, lim);
        }
        *gp = __hmul2(x2, y2);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Merger pieces on [N, 4096] fp16: LayerNorm (fp16 weight and bias, fp32 statistics, biased variance; ATen's `layer_norm` on a half tensor), exact GELU
// in place (ATen computes in fp32 and rounds), and the merger's gated activation in place on gu [N, ld] (g = cols [0, M), u = [M, 2M)): g clamped to
// at most the limit BEFORE the silu, u to +-limit, s = f16(silu_fp32(g)), result f16(u * s). flags 1: GELU by the tanh formula (a mutation).
// ------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
    layernorm_f16_k(const half* __restrict__ x, const half* __restrict__ w, const half* __restrict__ b, half* __restrict__ y, int dim, float eps) {
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * dim;
    __shared__ float red[8];
    float s = 0.f;
    for (int c = threadIdx.x; c < dim; c += 256) s += h2f(xr[c]);
    s = warp_sum(s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float mean = tot / (float)dim;
    __syncthreads();
    float v = 0.f;
    for (int c = threadIdx.x; c < dim; c += 256) {
        const float d = h2f(xr[c]) - mean;
        v += d * d;
    }
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    tot = 0.f;
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float rstd = rsqrtf(tot / (float)dim + eps);
    for (int c = threadIdx.x; c < dim; c += 256)
        y[(size_t)row * dim + c] = __float2half_rn((h2f(xr[c]) - mean) * rstd * h2f(w[c]) + h2f(b[c]));
}

__global__ void gelu_f16_k(half* __restrict__ x, size_t n, int flags) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = h2f(x[i]);
        float r;
        if (flags & 1) r = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        else r = v * 0.5f * (1.f + erff(v * 0.70710678118654752440f));
        x[i] = __float2half_rn(r);
    }
}

__global__ void swiglu_merger_k(half* __restrict__ gu, int M, int ld, float limit, size_t rows, int flags) {
    const size_t n = rows * (size_t)M;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / M;
        const int c = (int)(i % M);
        float g = h2f(gu[r * ld + c]), u = h2f(gu[r * ld + M + c]);
        if (!(flags & 1)) {
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        const float s = h2f(__float2half_rn(g / (1.f + expf(-g))));
        gu[r * ld + c] = __float2half_rn(u * s);
    }
}

// bias add in place: x [rows, n] fp16 += bias [n] (fp16 add, as `y += bias` on a half tensor)
__global__ void bias_add_f16_k(half* __restrict__ x, const half* __restrict__ bias, int n, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x)
        x[i] = hadd(x[i], bias[i % n]);
}

// fp32 [rows, n] += fp16 bias [n] (the patch embedding's `y += self.bias` on an fp32 tensor)
__global__ void bias_add_f32_k(float* __restrict__ x, const half* __restrict__ bias, int n, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) x[i] += h2f(bias[i % n]);
}

// The image's rows replace the embedding rows of its placeholder tokens in the four mHC streams: streams [rows][4][4096] f32 <- feats [n][4096] f32.
// An embedding is the same row in all four streams (docs/GLM53.md).
__global__ void splice_k(float* __restrict__ streams, const float* __restrict__ feats, int dst_row, int src_row, int n, int hidden, int nstreams) {
    const int r = blockIdx.x;
    const float4* f = (const float4*)(feats + (size_t)(src_row + r) * hidden);
    for (int c = threadIdx.x; c < hidden / 4; c += blockDim.x) {
        const float4 v = f[c];
        for (int s = 0; s < nstreams; ++s) *((float4*)(streams + ((size_t)(dst_row + r) * nstreams + s) * hidden) + c) = v;
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Bidirectional flash attention, head dim 64. q, k, v live in one fp16 buffer qkv [P, 3D] (q | k | v, each [P][16][64]); out [P, D].
// One CTA = 4 warps = 64 query rows of one head; keys stream through shared memory in tiles of 64 (K row-major, V transposed so the P.V
// B-fragments are contiguous). fp32 softmax statistics, P rounded to fp16 for the second product.
// mask: 0 none; 1 causal (a mutation of the tests: the tower is bidirectional).
// ------------------------------------------------------------------------------------------------------------------
constexpr int BM = 64, BN = 64, KS = HD + 8, VS = BN + 8;

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__global__ void __launch_bounds__(128)
    attn64_k(const half* __restrict__ qkv, half* __restrict__ out, int P, float scale_log2, int mask) {
    __shared__ __align__(16) half Ks[BN * KS];
    __shared__ __align__(16) half Vt[HD * VS];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int head = blockIdx.y;
    const half* qh = qkv + head * HD;
    const half* kh = qkv + D + head * HD;
    const half* vh = qkv + 2 * D + head * HD;
    const int q0 = blockIdx.x * BM + warp * 16;
    const int r0 = q0 + g, r1 = q0 + g + 8;

    uint32_t qa[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int c0 = ks * 16 + 2 * t, c1 = c0 + 8;
        qa[ks][0] = r0 < P ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r0 * QKV + c0) : 0u;
        qa[ks][1] = r1 < P ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r1 * QKV + c0) : 0u;
        qa[ks][2] = r0 < P ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r0 * QKV + c1) : 0u;
        qa[ks][3] = r1 < P ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r1 * QKV + c1) : 0u;
    }

    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int tiles = (P + BN - 1) / BN;
    for (int kt = 0; kt < tiles; ++kt) {
        __syncthreads();
        // K tile [64 keys][64] row-major (16 bytes a go), V tile transposed into Vt[d][key]; keys past P are zeros
        for (int i = tid; i < BN * (HD / 8); i += 128) {
            const int key = i / (HD / 8), c8 = i % (HD / 8), gk = kt * BN + key;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (gk < P) {
                kv = *reinterpret_cast<const uint4*>(kh + (size_t)gk * QKV + c8 * 8);
                vv = *reinterpret_cast<const uint4*>(vh + (size_t)gk * QKV + c8 * 8);
            }
            *reinterpret_cast<uint4*>(Ks + key * KS + c8 * 8) = kv;
            const half* vhh = reinterpret_cast<const half*>(&vv);
#pragma unroll
            for (int j = 0; j < 8; ++j) Vt[(c8 * 8 + j) * VS + key] = vhh[j];
        }
        __syncthreads();

        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const half* kr = Ks + (nt * 8 + g) * KS + ks * 16 + 2 * t;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr + 8);
                mma16816(s[nt], qa[ks], b0, b1);
            }
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int key = kt * BN + nt * 8 + 2 * t;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int kk = key + (e & 1);
                const int rr = e < 2 ? r0 : r1;
                bool valid = kk < P;
                if (mask == 1) valid = valid && kk <= rr;
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
        // a row whose keys are all masked so far keeps m = -inf: use 0 as the reference so exp2(-inf - 0) = 0 and nothing is NaN
        const float z0 = n0 == -INFINITY ? 0.f : n0, z1 = n1 == -INFINITY ? 0.f : n1;
        const float c0 = exp2f(m0 - z0), c1 = exp2f(m1 - z1);
        m0 = n0;
        m1 = n1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[i][0] *= c0;
            o[i][1] *= c0;
            o[i][2] *= c1;
            o[i][3] *= c1;
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - z0);
            s[nt][1] = exp2f(s[nt][1] - z0);
            s[nt][2] = exp2f(s[nt][2] - z1);
            s[nt][3] = exp2f(s[nt][3] - z1);
            l0 += s[nt][0] + s[nt][1];
            l1 += s[nt][2] + s[nt][3];
        }
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
            uint32_t pa[4];
            pa[0] = pack_h2(s[2 * kc][0], s[2 * kc][1]);
            pa[1] = pack_h2(s[2 * kc][2], s[2 * kc][3]);
            pa[2] = pack_h2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[3] = pack_h2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
#pragma unroll
            for (int dn = 0; dn < 8; ++dn) {
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
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int dn = 0; dn < 8; ++dn) {
        const int col = head * HD + dn * 8 + 2 * t;
        if (r0 < P) *reinterpret_cast<uint32_t*>(out + (size_t)r0 * D + col) = pack_h2(o[dn][0] * i0, o[dn][1] * i0);
        if (r1 < P) *reinterpret_cast<uint32_t*>(out + (size_t)r1 * D + col) = pack_h2(o[dn][2] * i1, o[dn][3] * i1);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Attention v2: the same arithmetic as attn64_k (the same tiles of 64 keys in the same order, the same mma sequence into every accumulator, the same
// softmax), so its output is bit-identical to v1's (a kernel test), with the loads of the next K and V tile in flight while this one is computed
// (cp.async, two stages), the fragments read with ldmatrix (K as stored, V transposed on the way in) from XOR-swizzled shared tiles (no bank conflicts), and
// 8 warps = 128 query rows a CTA (half the K and V traffic per query of v1).
// ------------------------------------------------------------------------------------------------------------------
constexpr int BM2 = 128;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    const int sz = valid ? 16 : 0;  // a source size of 0 zero-fills the 16 bytes without reading them
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__global__ void __launch_bounds__(256)
    attn64v2_k(const half* __restrict__ qkv, half* __restrict__ out, int P, float scale_log2, int mask) {
    // K and V tiles [64 keys][64 d], 128 bytes a row, the 16-byte chunk c of row r stored at chunk c ^ (r & 7)
    __shared__ __align__(128) half Ks[2][BN * HD];
    __shared__ __align__(128) half Vs[2][BN * HD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int head = blockIdx.y;
    const half* qh = qkv + head * HD;
    const half* kh = qkv + D + head * HD;
    const half* vh = qkv + 2 * D + head * HD;
    const int q0 = blockIdx.x * BM2 + warp * 16;
    const int r0 = q0 + g, r1 = q0 + g + 8;

    uint32_t qa[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int c0 = ks * 16 + 2 * t, c1 = c0 + 8;
        qa[ks][0] = r0 < P ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r0 * QKV + c0) : 0u;
        qa[ks][1] = r1 < P ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r1 * QKV + c0) : 0u;
        qa[ks][2] = r0 < P ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r0 * QKV + c1) : 0u;
        qa[ks][3] = r1 < P ? *reinterpret_cast<const uint32_t*>(qh + (size_t)r1 * QKV + c1) : 0u;
    }

    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int tiles = (P + BN - 1) / BN;
    auto load_tile = [&](int kt, int stage) {
        const uint32_t kb = smem_u32(&Ks[stage][0]), vb = smem_u32(&Vs[stage][0]);
        for (int i = tid; i < BN * 8; i += 256) {
            const int key = i >> 3, c = i & 7, gk = kt * BN + key;
            const bool valid = gk < P;
            const size_t src = (size_t)(valid ? gk : 0) * QKV + c * 8;
            const uint32_t off = key * 128 + ((c ^ (key & 7)) << 4);
            cp_async16(kb + off, kh + src, valid);
            cp_async16(vb + off, vh + src, valid);
        }
    };
    load_tile(0, 0);
    cp_commit();
    for (int kt = 0; kt < tiles; ++kt) {
        const int stage = kt & 1;
        if (kt + 1 < tiles) {
            load_tile(kt + 1, stage ^ 1);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const uint32_t kb = smem_u32(&Ks[stage][0]), vb = smem_u32(&Vs[stage][0]);

        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
            const int row = nt * 8 + (lane & 7);
#pragma unroll
            for (int kp = 0; kp < 2; ++kp) {
                uint32_t b[4];
                const int chunk = kp * 4 + (lane >> 3);
                ldsm_x4(b, kb + row * 128 + ((chunk ^ (row & 7)) << 4));
                mma16816(s[nt], qa[2 * kp], b[0], b[1]);
                mma16816(s[nt], qa[2 * kp + 1], b[2], b[3]);
            }
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int key = kt * BN + nt * 8 + 2 * t;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int kk = key + (e & 1);
                const int rr = e < 2 ? r0 : r1;
                bool valid = kk < P;
                if (mask == 1) valid = valid && kk <= rr;
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
        const float z0 = n0 == -INFINITY ? 0.f : n0, z1 = n1 == -INFINITY ? 0.f : n1;
        const float c0 = exp2f(m0 - z0), c1 = exp2f(m1 - z1);
        m0 = n0;
        m1 = n1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[i][0] *= c0;
            o[i][1] *= c0;
            o[i][2] *= c1;
            o[i][3] *= c1;
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - z0);
            s[nt][1] = exp2f(s[nt][1] - z0);
            s[nt][2] = exp2f(s[nt][2] - z1);
            s[nt][3] = exp2f(s[nt][3] - z1);
            l0 += s[nt][0] + s[nt][1];
            l1 += s[nt][2] + s[nt][3];
        }
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) {
            uint32_t pa[4];
            pa[0] = pack_h2(s[2 * kc][0], s[2 * kc][1]);
            pa[1] = pack_h2(s[2 * kc][2], s[2 * kc][3]);
            pa[2] = pack_h2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[3] = pack_h2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
            const int row = kc * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
            for (int dp = 0; dp < 4; ++dp) {
                uint32_t b[4];
                const int chunk = dp * 2 + (lane >> 4);
                ldsm_x4_t(b, vb + row * 128 + ((chunk ^ (row & 7)) << 4));
                mma16816(o[2 * dp], pa, b[0], b[1]);
                mma16816(o[2 * dp + 1], pa, b[2], b[3]);
            }
        }
        __syncthreads();  // everyone is done with this stage before the next iteration's loads overwrite it
    }
#pragma unroll
    for (int o2 = 1; o2 <= 2; o2 <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, o2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, o2);
    }
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int dn = 0; dn < 8; ++dn) {
        const int col = head * HD + dn * 8 + 2 * t;
        if (r0 < P) *reinterpret_cast<uint32_t*>(out + (size_t)r0 * D + col) = pack_h2(o[dn][0] * i0, o[dn][1] * i0);
        if (r1 < P) *reinterpret_cast<uint32_t*>(out + (size_t)r1 * D + col) = pack_h2(o[dn][2] * i1, o[dn][3] * i1);
    }
}

// The mma.sync m16n8k16 (fp16 in, fp32 accumulate) peak of the device, for the roofline of the tower: every warp runs `iters` rounds of 8 independent mma
// into 8 accumulator sets and nothing else. FLOP = blocks * warps * iters * 8 * 4096.
__global__ void __launch_bounds__(256) mma_peak_k(float* sink, int iters) {
    const uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
    const uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
    float d[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) d[i][0] = d[i][1] = d[i][2] = d[i][3] = 0.f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 8; ++i) mma16816(d[i], a, b0, b1);
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += d[i][0] + d[i][1] + d[i][2] + d[i][3];
    if (s == 123456.f) sink[threadIdx.x] = s;  // keeps the loop alive
}

// ------------------------------------------------------------------------------------------------------------------
// cuBLAS: one handle per host thread (created on first use, never destroyed: a destructor at thread exit can run after the context is gone).
// ------------------------------------------------------------------------------------------------------------------
cublasHandle_t handle() {
    static thread_local cublasHandle_t h = nullptr;
    if (!h && cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) h = nullptr;
    return h;
}

unsigned grid_for(size_t n, int threads = 256) {
    const size_t b = (n + threads - 1) / threads;
    return (unsigned)(b < 65535 ? b : 65535);
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------------ launchers

// bf16 -> fp16 in place over n elements; *bad (device int, zeroed by the caller) counts results that are not finite.
GB_EXPORT int gb_glmvis_bf16_to_f16(void* p, size_t n, int* bad, void* stream) {
    if (!p || !bad) return ERR_INVALID;
    if (n == 0) return 0;
    bf16_to_f16_k<<<grid_for(n), 256, 0, gb_stream(stream)>>>((uint16_t*)p, n, bad);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glmvis_fold_patch(const void* src, void* dst, int O, int KP, int slots, void* stream) {
    if (!src || !dst || O <= 0 || KP < 588 || (slots != 1 && slots != 2)) return ERR_INVALID;
    fold_patch_k<<<grid_for((size_t)O * KP), 256, 0, gb_stream(stream)>>>((const uint16_t*)src, (uint16_t*)dst, O, KP, slots);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glmvis_permute_downsample(const void* src, void* dst, int O, int C, int mode, int* bad, void* stream) {
    if (!src || !dst || !bad || O <= 0 || C <= 0 || mode < 0 || mode > 2) return ERR_INVALID;
    permute_downsample_k<<<grid_for((size_t)O * C * 4), 256, 0, gb_stream(stream)>>>((const uint16_t*)src, (uint16_t*)dst, O, C, mode, bad);
    return GB_CHECK_LAUNCH();
}

// 2-D copy of host rows into a device matrix with another pitch (the [P, 588] patches into the GEMM's [P, 592] input).
GB_EXPORT int gb_glmvis_upload_rows(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t rows, void* stream) {
    if (!dst || !src) return ERR_INVALID;
    if (rows == 0) return 0;
    return (int)cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, rows, cudaMemcpyHostToDevice, gb_stream(stream));
}

// x fp32 [rows, 1024] (+= f16(raw + bias) when raw != null), y fp16 = RMSNorm(x) * w (w bf16 [1024]).
GB_EXPORT int gb_glmvis_rmsnorm(void* x, const void* raw, const void* bias, const void* w, void* y, int rows, float eps, void* stream) {
    if (rows < 0 || !x || !w || !y) return ERR_INVALID;
    if (rows == 0) return 0;
    if (raw)
        rms_k<true><<<rows, 256, 0, gb_stream(stream)>>>((float*)x, (const half*)raw, (const half*)bias, (const __nv_bfloat16*)w, (half*)y, eps);
    else
        rms_k<false><<<rows, 256, 0, gb_stream(stream)>>>((float*)x, nullptr, nullptr, (const __nv_bfloat16*)w, (half*)y, eps);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glmvis_qkv_prep(void* qkv, const void* bias, const void* qn, const void* kn, const void* inv_freq, int P, int gw, float eps, int flags,
                                 void* stream) {
    if (P < 0 || gw <= 0 || gw % 2 || !qkv || !bias || !qn || !kn || !inv_freq) return ERR_INVALID;
    if (P == 0) return 0;
    qkv_prep_k<<<P, 128, 0, gb_stream(stream)>>>((half*)qkv, (const half*)bias, (const __nv_bfloat16*)qn, (const __nv_bfloat16*)kn, (const float*)inv_freq, gw,
                                                 eps, flags);
    return GB_CHECK_LAUNCH();
}

// A benchmark of the device's mma.sync peak (see mma_peak_k): `blocks` CTAs of 8 warps, `iters` rounds each; `sink` is 256 floats.
GB_EXPORT int gb_glmvis_mma_peak(void* sink, int blocks, int iters, void* stream) {
    if (!sink || blocks <= 0 || iters <= 0) return ERR_INVALID;
    mma_peak_k<<<blocks, 256, 0, gb_stream(stream)>>>((float*)sink, iters);
    return GB_CHECK_LAUNCH();
}

// variant 2 (the default): attn64v2_k; 1: the first version, attn64_k (kept as the reference the second is checked bit for bit against)
GB_EXPORT int gb_glmvis_attention(const void* qkv, void* out, int P, int mask, int variant, void* stream) {
    if (P < 0 || !qkv || !out || (variant != 1 && variant != 2)) return ERR_INVALID;
    if (P == 0) return 0;
    const float scale_log2 = 1.4426950408889634f / sqrtf((float)HD);
    if (variant == 1)
        attn64_k<<<dim3((P + BM - 1) / BM, HEADS), 128, 0, gb_stream(stream)>>>((const half*)qkv, (half*)out, P, scale_log2, mask);
    else
        attn64v2_k<<<dim3((P + BM2 - 1) / BM2, HEADS), 256, 0, gb_stream(stream)>>>((const half*)qkv, (half*)out, P, scale_log2, mask);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glmvis_swiglu_block(void* gu, const void* bg, const void* bu, int I, int ld, float limit, int flags, size_t rows, void* stream) {
    if (!gu || !bg || !bu || I <= 0 || I % 2 || ld < 2 * I) return ERR_INVALID;
    if (rows == 0) return 0;
    swiglu_block_k<<<grid_for(rows * (size_t)(I / 2)), 256, 0, gb_stream(stream)>>>((half*)gu, (const half*)bg, (const half*)bu, I, ld, limit, flags, rows);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glmvis_layernorm(const void* x, const void* w, const void* b, void* y, int rows, int dim, float eps, void* stream) {
    if (rows < 0 || dim <= 0 || !x || !w || !b || !y) return ERR_INVALID;
    if (rows == 0) return 0;
    layernorm_f16_k<<<rows, 256, 0, gb_stream(stream)>>>((const half*)x, (const half*)w, (const half*)b, (half*)y, dim, eps);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glmvis_gelu(void* x, size_t n, int flags, void* stream) {
    if (!x) return ERR_INVALID;
    if (n == 0) return 0;
    gelu_f16_k<<<grid_for(n), 256, 0, gb_stream(stream)>>>((half*)x, n, flags);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glmvis_swiglu_merger(void* gu, int M, int ld, float limit, size_t rows, int flags, void* stream) {
    if (!gu || M <= 0 || ld < 2 * M) return ERR_INVALID;
    if (rows == 0) return 0;
    swiglu_merger_k<<<grid_for(rows * (size_t)M), 256, 0, gb_stream(stream)>>>((half*)gu, M, ld, limit, rows, flags);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glmvis_bias_add_f16(void* x, const void* bias, int n, size_t rows, void* stream) {
    if (!x || !bias || n <= 0) return ERR_INVALID;
    if (rows == 0) return 0;
    bias_add_f16_k<<<grid_for(rows * (size_t)n), 256, 0, gb_stream(stream)>>>((half*)x, (const half*)bias, n, rows * (size_t)n);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_glmvis_bias_add_f32(void* x, const void* bias, int n, size_t rows, void* stream) {
    if (!x || !bias || n <= 0) return ERR_INVALID;
    if (rows == 0) return 0;
    bias_add_f32_k<<<grid_for(rows * (size_t)n), 256, 0, gb_stream(stream)>>>((float*)x, (const half*)bias, n, rows * (size_t)n);
    return GB_CHECK_LAUNCH();
}

// feats f32 [.., hidden] rows src_row .. src_row + n -> the four mHC streams of rows dst_row ..: streams f32 [rows][nstreams][hidden].
GB_EXPORT int gb_glmvis_splice(void* streams, const void* feats, int dst_row, int src_row, int n, int hidden, int nstreams, void* stream) {
    if (!streams || !feats || dst_row < 0 || src_row < 0 || n < 0 || hidden <= 0 || hidden % 4 || nstreams <= 0) return ERR_INVALID;
    if (n == 0) return 0;
    splice_k<<<n, 256, 0, gb_stream(stream)>>>((float*)streams, (const float*)feats, dst_row, src_row, n, hidden, nstreams);
    return GB_CHECK_LAUNCH();
}

// C [m, n] = A [m, k] (row stride lda) @ W [n, k]^T, fp16 inputs, fp32 accumulate; C fp16 or fp32 (row stride ldc). The cuBLAS workspace is the
// caller's (counted in the tower's memory plan).
GB_EXPORT int gb_glmvis_gemm(const void* a, int lda, const void* w, void* c, int ldc, int m, int k, int n, int c_is_f32, void* ws, size_t ws_bytes,
                             void* stream) {
    if (m < 1 || k <= 0 || n <= 0 || lda < k || ldc < n || !a || !w || !c) return ERR_INVALID;
    cublasHandle_t h = handle();
    if (!h) return (int)cudaErrorUnknown;
    if (cublasSetStream(h, gb_stream(stream)) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    if (ws && ws_bytes && cublasSetWorkspace(h, ws, ws_bytes) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    // row-major C = A W^T  is, column-major, C^T [n, m] = W_cm^T [n, k] A_cm [k, m]  (W row-major [n, k] is column-major [k, n] with ld k)
    const float one = 1.f, zero = 0.f;
    const cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, w, CUDA_R_16F, k, a, CUDA_R_16F, lda, &zero, c,
                                          c_is_f32 ? CUDA_R_32F : CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}
