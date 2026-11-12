// ling_kda_proj_fused.cu — Ling-owned fused decode projections for two or three EXL3 layers that take the SAME input x.
//
// The KDA decode path runs qkv, f and g as three separate `gb_exl3_linear_k2` launches on the same `x` (the layer
// input norm), each doing its own `rot_in` (Walsh-Hadamard input rotation) on the identical input and then a linear
// kernel over its column strip. `f` and `g` are 3-bit 4096x8192 layers; `qkv` a 3.5-bit 4096x24576 one; at 200+
// registers one CTA fits per SM, so a 64-CTA launch runs 48 CTAs in wave 1 and 16 (67% of the SMs idle) in wave 2.
//
// This file provides one launch computing the whole qkv+f+g triple together, and one computing f+g together:
//   * the input rotation runs once per projection in ONE launch (same kernels, same inputs -> the identical xh bytes
//     per half; each projection's suh gets its own buffer);
//   * the linear kernel's grid covers all projections' columns (192+64+64 = 320 CTAs) while every output element keeps
//     the exact arithmetic, k order (ascending tile order, warp-order sum) and finish (svh, no bias) of the original
//     `linear_kernel` per 128-column strip. Per-element bits are therefore unchanged: the fusion is timing-only.
//
// The pair also serves the MLA `q_a + kv_a` projection (k2 = 10, narrow n, the decode path splits K to fill the SMs:
// the fused pair keeps the same plan and finishes from the split partials in split order via `gb_exl3_rotate_out_sk`,
// exactly as the separate launches do) and the dense FFN `gate + up` pair (k2 = 6, bf16 out).
//
// The code below is the decode machinery of `exl3_linear.cu` (load4/store4/finish, step helpers, rot_in kernels,
// linear_kernel) verbatim except for the per-projection column indirection. It is registered as a Ling-owned file
// (bailing_moe_v3 scope), so `exl3_linear.cu` itself is untouched for other families.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "common.cuh"
#include "pdl.cuh"
#include "exl3_linear_decode.cuh"

namespace lingproj {

using namespace gbx;   // the decode helpers (tile_words, lane_start, fwht128, HAD_SCALE, mma16816, ...)
using gbp_pdl::pdl_wait;
using gbp_pdl::pdl_trigger;
using gbp_pdl::prefetch_l2;

// The rmsnorm math of glue.cu's rmsnorm_k<false> (compiled there with --fmad=false, so here every multiply and
// add is an explicit RN intrinsic to keep the bits identical under this TU's default fmad=true).
__device__ __forceinline__ float lpf_bf2f(uint16_t u) { return __uint_as_float((uint32_t)u << 16); }
__device__ __forceinline__ uint16_t lpf_f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ void lpf_unpack8(const uint4& u, float (&f)[8]) {
    f[0] = __uint_as_float(u.x << 16); f[1] = __uint_as_float(u.x & 0xFFFF0000u);
    f[2] = __uint_as_float(u.y << 16); f[3] = __uint_as_float(u.y & 0xFFFF0000u);
    f[4] = __uint_as_float(u.z << 16); f[5] = __uint_as_float(u.z & 0xFFFF0000u);
    f[6] = __uint_as_float(u.w << 16); f[7] = __uint_as_float(u.w & 0xFFFF0000u);
}
__device__ __forceinline__ uint32_t lpf_pack2(float a, float b) {
    return (uint32_t)lpf_f2bf(a) | ((uint32_t)lpf_f2bf(b) << 16);
}
__device__ __forceinline__ uint4 lpf_pack8(const float (&f)[8]) {
    uint4 u;
    u.x = lpf_pack2(f[0], f[1]); u.y = lpf_pack2(f[2], f[3]);
    u.z = lpf_pack2(f[4], f[5]); u.w = lpf_pack2(f[6], f[7]);
    return u;
}
__device__ __forceinline__ float lpf_warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
template <int NT>
__device__ __forceinline__ float lpf_block_sum(float v, float* sm) {
    v = lpf_warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    float s = sm[0];
#pragma unroll
    for (int w = 1; w < NT / 32; ++w) s = s + sm[w];
    __syncthreads();  // sm may be reused by the caller
    return s;
}

// The C ABI dtype codes (exl3_linear.rs Dtype discriminants).
enum DtypeCode : int { F16 = 0, BF16 = 1, F32 = 2 };

constexpr int kMaxRows = 128;

    // Forward declarations of the separately-compiled exl3_linear entries the pair's split-K finish and fallback use.
    // (Indented so release_scope.py's line-anchored C-definition regex does not count them as definitions of this file.)
    extern "C" int gb_exl3_linear_k2(const void* x, int x_dtype, const void* trellis, const void* suh,
                                     const void* svh, const void* bias, void* y, int y_dtype, int m, int k, int n,
                                     int bits, int codebook, void* workspace, size_t workspace_bytes, void* stream);
    extern "C" int gb_exl3_rotate_out_sk(const void* c, const void* svh, const void* bias, void* y, int y_dtype,
                                          int m, int n, int sk, void* stream);

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        if (bias) v[j] += __half2float(__ldg(bias + col + j));
    }
}

template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2, int JN = 8>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? JN * tile_words<K2>() / 32 : JN * lane_words<K2>();
}

template <int K2, int JN = 8>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2, JN>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < step_regs<K2, JN>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < JN; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

template <int K2, int JN = 8>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2, JN>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;                  // lanes whose windows end in the same word
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;                      // a window inside one word never reads w[1]
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

// xh [M, K] fp16 = (x * suh) H / sqrt(128), fp32 inside, fp16 out. One warp a 128-wide block of one row.
__global__ void __launch_bounds__(128) ling_rot_in_kernel(const void* __restrict__ x, int x_dtype,
                                                          const half* __restrict__ suh, half* __restrict__ xh, int K) {
    pdl_wait();                                  // first, in every block, early exits included (see pdl.cuh)
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(suh, F16, k, s);
    load4(x, x_dtype, (size_t)row * K + k, v);
    pdl_trigger();
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(xh, F16, (size_t)row * K + k, v);
}

__global__ void __launch_bounds__(128) ling_rot_in_frag_kernel(const void* __restrict__ x, int x_dtype,
                                                              const half* __restrict__ suh, uint32_t* __restrict__ xf, int K,
                                                              int M) {
    pdl_wait();
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(suh, F16, k, s);
    load4(x, x_dtype, (size_t)min(row, M - 1) * K + k, v);
    pdl_trigger();
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    const half2 pair[2] = {__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3])};
    const int mt = row >> 4, g = row & 7, rh = (row >> 3) & 1, KT = K >> 4;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int col = k + 2 * q, kt = col >> 4, kk = col & 15;
        xf[(((size_t)mt * KT + kt) * 32 + g * 4 + ((kk & 7) >> 1)) * 4 + 2 * (kk >> 3) + rh] =
            *reinterpret_cast<const uint32_t*>(&pair[q]);
    }
}

// Paired input rotation: one launch rotating the same x with suh0 into xh0 AND with suh1 into xh1 (grid.y is
// twice the single-projection height; each half keeps the exact arithmetic of ling_rot_in_kernel).
__global__ void __launch_bounds__(128) ling_rot_in_pair_kernel(const void* __restrict__ x, int x_dtype,
                                                              const half* __restrict__ suh0, half* __restrict__ xh0,
                                                              const half* __restrict__ suh1, half* __restrict__ xh1,
                                                              int K, int gy_single) {
    pdl_wait();                                  // first, in every block, early exits included (see pdl.cuh)
    const bool second = blockIdx.y >= gy_single;
    const int blk = (blockIdx.y - (second ? gy_single : 0)) * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(second ? suh1 : suh0, F16, k, s);
    load4(x, x_dtype, (size_t)row * K + k, v);
    pdl_trigger();
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(second ? xh1 : xh0, F16, (size_t)row * K + k, v);
}

__global__ void __launch_bounds__(128) ling_rot_in_frag_pair_kernel(const void* __restrict__ x, int x_dtype,
                                                                    const half* __restrict__ suh0, uint32_t* __restrict__ xf0,
                                                                    const half* __restrict__ suh1, uint32_t* __restrict__ xf1,
                                                                    int K, int M, int gy_single) {
    pdl_wait();
    const bool second = blockIdx.y >= gy_single;
    const int blk = (blockIdx.y - (second ? gy_single : 0)) * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(second ? suh1 : suh0, F16, k, s);
    load4(x, x_dtype, (size_t)min(row, M - 1) * K + k, v);
    pdl_trigger();
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    const half2 pair[2] = {__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3])};
    const int mt = row >> 4, g = row & 7, rh = (row >> 3) & 1, KT = K >> 4;
    uint32_t* xf = second ? xf1 : xf0;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int col = k + 2 * q, kt = col >> 4, kk = col & 15;
        xf[(((size_t)mt * KT + kt) * 32 + g * 4 + ((kk & 7) >> 1)) * 4 + 2 * (kk >> 3) + rh] =
            *reinterpret_cast<const uint32_t*>(&pair[q]);
    }
}

// Triple input rotation: one launch rotating the same x with suh0 / suh1 / suh2 into xh0 / xh1 / xh2 (grid.y is
// three times the single-projection height).
__global__ void __launch_bounds__(128) ling_rot_in_triple_kernel(const void* __restrict__ x, int x_dtype,
                                                                const half* __restrict__ suh0, half* __restrict__ xh0,
                                                                const half* __restrict__ suh1, half* __restrict__ xh1,
                                                                const half* __restrict__ suh2, half* __restrict__ xh2,
                                                                int K, int gy_single) {
    pdl_wait();
    const int pidx = blockIdx.y / gy_single;
    const int blk = (blockIdx.y - pidx * gy_single) * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(pidx == 0 ? suh0 : (pidx == 1 ? suh1 : suh2), F16, k, s);
    load4(x, x_dtype, (size_t)row * K + k, v);
    pdl_trigger();
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(pidx == 0 ? xh0 : (pidx == 1 ? xh1 : xh2), F16, (size_t)row * K + k, v);
}

__global__ void __launch_bounds__(128) ling_rot_in_frag_triple_kernel(const void* __restrict__ x, int x_dtype,
                                                                      const half* __restrict__ suh0, uint32_t* __restrict__ xf0,
                                                                      const half* __restrict__ suh1, uint32_t* __restrict__ xf1,
                                                                      const half* __restrict__ suh2, uint32_t* __restrict__ xf2,
                                                                      int K, int M, int gy_single) {
    pdl_wait();
    const int pidx = blockIdx.y / gy_single;
    const int blk = (blockIdx.y - pidx * gy_single) * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(pidx == 0 ? suh0 : (pidx == 1 ? suh1 : suh2), F16, k, s);
    load4(x, x_dtype, (size_t)min(row, M - 1) * K + k, v);
    pdl_trigger();
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    const half2 pair[2] = {__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3])};
    const int mt = row >> 4, g = row & 7, rh = (row >> 3) & 1, KT = K >> 4;
    uint32_t* xf = pidx == 0 ? xf0 : (pidx == 1 ? xf1 : xf2);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int col = k + 2 * q, kt = col >> 4, kk = col & 15;
        xf[(((size_t)mt * KT + kt) * 32 + g * 4 + ((kk & 7) >> 1)) * 4 + 2 * (kk >> 3) + rh] =
            *reinterpret_cast<const uint32_t*>(&pair[q]);
    }
}

// The linear path of one 128-column block, factored from the (pair / triple) kernels: `linear_kernel` of
// exl3_linear.cu verbatim, with the projection's resolved tensors as arguments. Each projection's strip keeps the
// exact k sweep, per-warp sums, split partials and finish of the original single-layer kernel, so every output
// element's bits are the separate launches'.
template <int K2, int CB, int NT, int H, int WK>
__device__ __forceinline__ void run_linear_cta(
    const half* xh, const uint32_t* T, long long stride_k, long long stride_nb, const half* svh,
    void* y, int y_dtype, int N, int col0,
    float* Z, int M, int K, int SK, int ZM,
    float* red, float* stage, int rh) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr bool FRAG = NT >= 2;                            // the A fragments come in fragment order
    constexpr int JN = 8 / H;                                 // 16-column tiles of a sweep
    constexpr int RC = JN * 16;                               // columns of a sweep
    const int split = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + (col0 >> 7) * stride_nb;
    bool prev = false;
    int lane_word = 0, lane_offset = 0;
    lane_start<K2>(lane, lane_word, lane_offset);
    if constexpr (step_shuffled<K2>()) {
        prev = lane_word != lane / (8 / K2);
    }

    for (int m0 = 0; m0 < M; m0 += 16 * NT) {
        const half* x0[FRAG ? 1 : NT];
        const half* x1[FRAG ? 1 : NT];
        const uint32_t* xp[FRAG ? NT : 1];
        if constexpr (FRAG) {
#pragma unroll
            for (int p = 0; p < NT; ++p)
                xp[p] = reinterpret_cast<const uint32_t*>(xh) + (size_t)min((m0 >> 4) + p, (M - 1) >> 4) * (K >> 4) * 128 + lane * 4;
        } else {
#pragma unroll
            for (int p = 0; p < NT; ++p) {
                x0[p] = xh + (size_t)min(m0 + 16 * p + g, M - 1) * K;
                x1[p] = xh + (size_t)min(m0 + 16 * p + g + 8, M - 1) * K;
            }
        }
#pragma unroll 1
        for (int hh = 0; hh < H; ++hh) {
            float acc[NT][JN][2][4];
#pragma unroll
            for (int p = 0; p < NT; ++p)
#pragma unroll
                for (int i = 0; i < JN; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) acc[p][i][h][c] = 0.f;

            const uint32_t* tile = tiles + (size_t)kt0 * stride_k + (size_t)(hh * JN) * TW;
            constexpr bool PF = K2 <= 12;
            constexpr int SR = step_regs<K2, JN>();
            uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
            if constexpr (PF) {
                load_step<K2, JN>(tile, lane, cur);
            } else {
                constexpr int kLines = JN * TW * 4 / 128;
                static_assert(kLines <= 32, "a step's tiles fit one line a lane");
#pragma unroll
                for (int s = 0; s < 4; ++s)
                    if (s < per_warp && lane < kLines) prefetch_l2(tile + (size_t)s * stride_k + lane * 32);
            }
            pdl_wait();                                          // the rotated input (xh) is the predecessor's output
#pragma unroll 1
            for (int i = 0; i < per_warp; ++i) {
                const int kt = kt0 + i;
                uint32_t a[NT][4];
#pragma unroll
                for (int p = 0; p < NT; ++p) {
                    if constexpr (FRAG) {
                        const uint4 v = __ldg(reinterpret_cast<const uint4*>(xp[p] + (size_t)kt * 128));
                        a[p][0] = v.x; a[p][1] = v.y; a[p][2] = v.z; a[p][3] = v.w;
                    } else {
                        a[p][0] = __ldg(reinterpret_cast<const uint32_t*>(x0[p] + kt * 16 + 2 * t));
                        a[p][1] = __ldg(reinterpret_cast<const uint32_t*>(x1[p] + kt * 16 + 2 * t));
                        a[p][2] = __ldg(reinterpret_cast<const uint32_t*>(x0[p] + kt * 16 + 2 * t + 8));
                        a[p][3] = __ldg(reinterpret_cast<const uint32_t*>(x1[p] + kt * 16 + 2 * t + 8));
                    }
                }
                if constexpr (PF) {
                    if (i + 1 < per_warp) load_step<K2, JN>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
                }
#pragma unroll
                for (int j = 0; j < JN; ++j) {
                    uint32_t w[LW];
                    if constexpr (PF) step_lane_words<K2, JN>(cur, j, lane, prev, w);
                    else ldg_lane_words_at<K2>(tile + (size_t)i * stride_k + j * TW, lane_word, w);
                    uint32_t b0[2], b1[2];
                    decode_lane_at<K2, CB>(w, lane_offset, b0, b1);
#pragma unroll
                    for (int p = 0; p < NT; ++p) {
                        mma16816(acc[p][j][0], a[p], b0);
                        mma16816(acc[p][j][1], a[p], b1);
                    }
                }
                if constexpr (PF) {
#pragma unroll
                    for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
                }
            }

            pdl_trigger();                                           // xh is read: the successor may launch

#pragma unroll
            for (int p = 0; p < NT; ++p) {
                const int mp = m0 + 16 * p;
                if (mp >= M) break;
                const int R = min(16, M - mp);
                for (int rlo = 0; rlo < R; rlo += 8) {
                    const int rn = min(R - rlo, 8);
                    __syncthreads();                     // red is reused by every half and pass
                    if (g < rh) {
#pragma unroll
                        for (int i = 0; i < JN; ++i)
#pragma unroll
                            for (int h = 0; h < 2; ++h) {
                                const int col = i * 16 + h * 8 + 2 * t;
                                *reinterpret_cast<float2*>(red + (warp * rh + g) * RC + col) =
                                    rlo ? make_float2(acc[p][i][h][2], acc[p][i][h][3]) : make_float2(acc[p][i][h][0], acc[p][i][h][1]);
                            }
                    }
                    __syncthreads();

                    if constexpr (H == 1) {
                        if (SK == 1 && !ZM) {
                            for (int r = warp; r < rn; r += WK) {
                                float v[4];
                                const float4 u = *reinterpret_cast<const float4*>(red + r * RC + 4 * lane);
                                v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                                for (int w = 1; w < WK; ++w) {
                                    const float4 q = *reinterpret_cast<const float4*>(red + (w * rh + r) * RC + 4 * lane);
                                    v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                                }
                                finish(v, lane, svh, nullptr, col0 + 4 * lane);
                                store4(y, y_dtype, (size_t)(mp + rlo + r) * N + col0 + 4 * lane, v);
                            }
                            continue;
                        }
                    }
                    for (int idx = threadIdx.x; idx < rn * (RC / 4); idx += WK * 32) {
                        const int r = idx / (RC / 4), c = 4 * (idx % (RC / 4));
                        float4 s = *reinterpret_cast<const float4*>(red + r * RC + c);
#pragma unroll
                        for (int w = 1; w < WK; ++w) {
                            const float4 q = *reinterpret_cast<const float4*>(red + (w * rh + r) * RC + c);
                            s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                        }
                        if (SK > 1 || ZM)
                            *reinterpret_cast<float4*>(Z + ((size_t)split * M + mp + rlo + r) * N + col0 + hh * RC + c) = s;
                        else
                            *reinterpret_cast<float4*>(stage + (16 * p + rlo + r) * 128 + hh * RC + c) = s;
                    }
                }
            }
        }
        if constexpr (H > 1) {
            if (SK == 1 && !ZM) {
                __syncthreads();                         // every sweep's sums are in stage
                const int rows = min(16 * NT, M - m0);
                for (int r = warp; r < rows; r += WK) {
                    const float4 u = *reinterpret_cast<const float4*>(stage + r * 128 + 4 * lane);
                    float v[4] = {u.x, u.y, u.z, u.w};
                    finish(v, lane, svh, nullptr, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
            }
        }
        __syncthreads();                             // red and stage are reused in the next group
    }
}

// The exact `ling_silu_mul_rot_in_kernel` math for one 128-column block (`col_blk`), run by the second of the two
// pair CTAs (columns `col_blk * 128 .. + 128` of the gate strip = CTA `col_blk`, of the up strip = CTA
// `col_blk + n0 / 128`) after both wrote their strips: g/u are read straight from the pair's own output buffers
// (bf16, complete by the per-block atomic handshake), so the standalone swiglu-rotation launch disappears. Every
// expression is the source kernel's (this TU compiles without --fmad=false, so the silu and bf16 rounding are
// explicit RN intrinsics).
__device__ __forceinline__ void silu_mul_rot_block(const __nv_bfloat16* g, const __nv_bfloat16* u,
                                                   const half* suh, half* xh, int M, int K,
                                                   int col_blk, int warp_id, int wk, int lane) {
    const int k = col_blk * 128 + 4 * lane;
    float s[4];
    load4(suh, F16, k, s);
    for (int row = warp_id; row < M; row += wk) {
        const size_t at = (size_t)row * K + k;
        const uint2 gu = *reinterpret_cast<const uint2*>(g + at);
        const uint2 uu = *reinterpret_cast<const uint2*>(u + at);
        const float2 ga = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gu.x));
        const float2 gb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gu.y));
        const float2 ua = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uu.x));
        const float2 ub = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uu.y));
        const float sg0 = __fdiv_rn(ga.x, __fadd_rn(1.0f, expf(-ga.x)));
        const float sg1 = __fdiv_rn(ga.y, __fadd_rn(1.0f, expf(-ga.y)));
        const float sg2 = __fdiv_rn(gb.x, __fadd_rn(1.0f, expf(-gb.x)));
        const float sg3 = __fdiv_rn(gb.y, __fadd_rn(1.0f, expf(-gb.y)));
        float v[4];
        v[0] = lpf_bf2f(lpf_f2bf(__fmul_rn(sg0, ua.x)));
        v[1] = lpf_bf2f(lpf_f2bf(__fmul_rn(sg1, ua.y)));
        v[2] = lpf_bf2f(lpf_f2bf(__fmul_rn(sg2, ub.x)));
        v[3] = lpf_bf2f(lpf_f2bf(__fmul_rn(sg3, ub.y)));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= s[j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
        store4(xh, F16, at, v);
    }
}

// The exact `ling_mla_rot_out_qnorm_qb_rot_in_kernel` math for one row (the q_a/kv_a split-K finishes, the q_a
// RMSNorm with qa_norm_w, and the q_b input rotation with qb_suh into qb_xh), run by the last CTA of
// `ling_pair_linear_kernel` on its native 128-thread block (WK = 4 warps) after every CTA's split partials are
// visible. Step A stages the bf16 finish codes straight into y0/y1 (the q_a/kv_a output buffers; y0 doubles as
// the step-B scratch, exactly like the standalone kernel's s_qa), step B runs the 256-thread rmsnorm reduction as
// two halves per physical warp (t = tid and t = tid + 128 per half), so only float sm[8] (inside the linear's
// `red`) is shared memory, and step C rotates y0 with qb_suh into qb_xh using fwht128 (bit-identical to the
// standalone kernel's shared-memory butterfly: the same distances 1..64, the same add/sub selection, the same
// operand order). Every expression is the source kernel's verbatim.
// Step A (rot_out finish) of ONE 128-column block of the fused MLA epilogue: `Z [sk][m][n] -> y` for the block's
// own columns, run by the last split CTA of that column block once all SK splits' Z strips of the block are visible.
// Every row's elements are independent (fwht128 and the bf16 finish are pure per-lane math), so the M rows split
// over the 4 warps; each element's expression sequence is the standalone
// `ling_mla_rot_out_qnorm_qb_rot_in_kernel`'s verbatim (explicit RN intrinsics, as `ling_single_fused_column_block`).
__device__ __forceinline__ void ling_pair_mla_a_block(
    const float* __restrict__ Z, const half* __restrict__ svh,
    __nv_bfloat16* __restrict__ y, int m, int n, int sk, int col0, int warp, int lane) {
    const int col = col0 + 4 * lane;
    for (int row = warp; row < m; row += 4) {
        const size_t at = (size_t)row * n + col;
        float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
        for (int q0 = 1; q0 < sk; q0 += 8) {
            float4 u[8];
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) u[d] = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)(q0 + d) * m * n + at));
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) {
                    s.x = __fadd_rn(s.x, u[d].x);
                    s.y = __fadd_rn(s.y, u[d].y);
                    s.z = __fadd_rn(s.z, u[d].z);
                    s.w = __fadd_rn(s.w, u[d].w);
                }
        }
        float v[4] = {s.x, s.y, s.z, s.w};
        gbx::fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __fmul_rn(__fmul_rn(v[j], gbx::HAD_SCALE), __half2float(__ldg(svh + col + j)));
        store4(y, BF16, at, v);
    }
}

// Step B (q_a rmsnorm over y0, staging the rounded bf16 codes into `s_qa`) + Step C (q_b input rotation from
// `s_qa`) for one row: the exact `ling_mla_rot_out_qnorm_qb_rot_in_kernel` math, run by the last q_a column block
// after all n0/128 blocks' Step-A2 finishes are visible. `s_qa` holds the same bits as y0's normalized row (still
// written in place, as the source kernel does), so Step C reads shared instead of re-reading global L2.
__device__ __forceinline__ void ling_pair_mla_bc_row(
    const float* __restrict__ qa_w, const half* __restrict__ qb_suh, half* __restrict__ qb_xh,
    __nv_bfloat16* __restrict__ y0, int m, int n0, float eps, int row, float* sm, uint16_t* s_qa) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t base0 = (size_t)row * n0;

    // Step B: the exact ling_rmsnorm_rot_in_kernel step-1 body over y0 (explicit RN ops, lpf_block_sum), with the
    // 256-thread reduction simulated on the 128-thread block: each physical warp w carries virtual warps w (t =
    // tid) and w + 4 (t = tid + 128), so the per-thread v-set and per-warp butterfly sums are exactly the
    // 256-thread kernel's, and only sm[0..7] is shared.
    const int nvec = n0 / 8;
#pragma unroll
    for (int hb = 0; hb < 2; ++hb) {
        const int t = threadIdx.x + 128 * hb;
        float ss = 0.0f;
        for (int v = t; v < nvec; v += 256) {
            float f[8];
            lpf_unpack8(*(const uint4*)(y0 + base0 + (size_t)v * 8), f);
#pragma unroll
            for (int i = 0; i < 8; ++i) ss = __fadd_rn(ss, __fmul_rn(f[i], f[i]));
        }
        ss = lpf_warp_sum(ss);
        if (lane == 0) sm[warp + 4 * hb] = ss;
    }
    __syncthreads();
    float s = sm[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) s = s + sm[w];
    const float inv = 1.0f / sqrtf(s / (float)n0 + eps);
    __syncthreads();  // sm may be reused by the next row
    for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
        float f[8], o[8];
        lpf_unpack8(*(const uint4*)(y0 + base0 + (size_t)v * 8), f);
        const float4 w0 = *(const float4*)(qa_w + (size_t)v * 8), w1 = *(const float4*)(qa_w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(f[i], inv), wv[i]);
        const uint4 pk = lpf_pack8(o);
        *(uint4*)(y0 + base0 + (size_t)v * 8) = pk;
        *(uint4*)(s_qa + (size_t)v * 8) = pk;
    }
    __syncthreads();

    // Step C: the exact step-2 rotation of the normalized row with qb_suh into qb_xh (fwht128's fixed butterfly
    // order), read from the staged `s_qa` (same bits as the y0 row just written).
    for (int blk = warp; blk < n0 / 128; blk += 4) {
        const int col = blk * 128 + 4 * lane;
        float v[4], s[4];
        v[0] = lpf_bf2f(s_qa[col]);
        v[1] = lpf_bf2f(s_qa[col + 1]);
        v[2] = lpf_bf2f(s_qa[col + 2]);
        v[3] = lpf_bf2f(s_qa[col + 3]);
        load4(qb_suh, F16, col, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= s[j];
        gbx::fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= gbx::HAD_SCALE;
        store4(qb_xh, F16, base0 + col, v);
    }
}

// The two-projection linear kernel: one grid over both layers' columns ([0, n0) are projection 0, [n0, n0+n1)
// projection 1; the two layers share K, K2, CB, the plan and M).
template <int K2, int CB, int WK, int NT, int H>
__global__ void __launch_bounds__(WK * 32, 1) ling_pair_linear_kernel(
    const half* __restrict__ xh0,
    const half* __restrict__ xh1,
    const uint32_t* __restrict__ T0, long long stride_k0, long long stride_nb0, const half* __restrict__ svh0,
    void* __restrict__ y0, int y_dtype0, int n0,
    const uint32_t* __restrict__ T1, long long stride_k1, long long stride_nb1, const half* __restrict__ svh1,
    void* __restrict__ y1, int y_dtype1, int n1,
    float* __restrict__ Z0, float* __restrict__ Z1, int M, int K, int SK, int ZM,
    const half* __restrict__ down_suh, half* __restrict__ down_xh, int* __restrict__ col_cnt,
    const float* __restrict__ mla_qa_w, const half* __restrict__ mla_qb_suh, half* __restrict__ mla_qb_xh,
    int* __restrict__ pair_cnt, float eps) {
    constexpr int RC = (8 / H) * 16;
    extern __shared__ __align__(16) float red[];              // WK * RH * RC floats, then (H > 1, SK == 1) 16 NT * 128 floats
    const int RH = min(M, 8);                                 // rows of red a warp
    float* stage = red + WK * RH * RC;

    const int colg = blockIdx.x * 128;
    const bool second = colg >= n0;
    const int col0 = second ? colg - n0 : colg;
    const half* xh = second ? xh1 : xh0;
    const uint32_t* T = second ? T1 : T0;
    const long long stride_k = second ? stride_k1 : stride_k0;
    const long long stride_nb = second ? stride_nb1 : stride_nb0;
    const half* svh = second ? svh1 : svh0;
    void* y = second ? y1 : y0;
    const int y_dtype = second ? y_dtype1 : y_dtype0;
    const int N = second ? n1 : n0;

    run_linear_cta<K2, CB, NT, H, WK>(xh, T, stride_k, stride_nb, svh, y, y_dtype, N, col0,
                            second ? Z1 : Z0, M, K, SK, ZM, red, stage, RH);

    // Fused SwiGLU input rotation of a following single projection (the dense FFN down_proj) over this 128-column
    // block: exactly two CTAs write the block's gate and up strips (col_blk and col_blk + n0 / 128), so the second
    // to arrive (per-block self-resetting counter) computes `silu(g) * u` rotated by `down_suh` into `down_xh`
    // straight from the strips just written. Bit-identical per element to the separate
    // `ling_silu_mul_rot_in_kernel` launch (same operands, same expressions, same store).
    if (down_suh && down_xh && col_cnt && SK == 1 && !ZM && n0 == n1 &&
        y_dtype0 == BF16 && y_dtype1 == BF16) {
        const int col_blk = col0 >> 7;
        __syncthreads();
        __threadfence();
        __shared__ int s_second;
        if (threadIdx.x == 0) {
            const int prev = atomicAdd(&col_cnt[col_blk], 1);
            if (prev == 1) {
                col_cnt[col_blk] = 0;
                s_second = 1;
            } else {
                s_second = 0;
            }
        }
        __syncthreads();
        if (s_second) {
            __threadfence();
            const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
            silu_mul_rot_block((const __nv_bfloat16*)y0, (const __nv_bfloat16*)y1,
                               down_suh, down_xh, M, n0, col_blk, warp, WK, lane);
        }
    }

    // Fused MLA q_a/kv_a split finishes + q_a norm + q_b input rotation: per-column-block self-resetting counters
    // (`pair_cnt[0..32)`, one per 128-column block of both projections, then pair_cnt[32] for the q_a handshake)
    // let each column block's last split CTA run its OWN Step A finish (kv_a: A1 only; q_a: A2, then the 12th q_a
    // block runs Step B/C), so the serial last-CTA epilogue of the old fuse (all columns, every row, one CTA)
    // becomes parallel across the column blocks. Bit-identical per element to the standalone
    // `ling_mla_rot_out_qnorm_qb_rot_in_kernel` launch (same operands, same expressions, same stores). Requires the
    // native 128-thread block (WK = 4; the caller leaves these null when the plan's wk is 8 and the separate launch
    // runs).
    if (mla_qa_w && mla_qb_suh && mla_qb_xh && pair_cnt && blockDim.x == 128) {
        const int col_blk = blockIdx.x;
        __syncthreads();
        __threadfence();
        __shared__ int s_split_last;
        if (threadIdx.x == 0) {
            const int prev_s = atomicAdd(&pair_cnt[col_blk], 1);
            if (prev_s == (int)gridDim.y - 1) {
                pair_cnt[col_blk] = 0;
                s_split_last = 1;
            } else {
                s_split_last = 0;
            }
        }
        __syncthreads();
        if (!s_split_last) return;
        __threadfence();

        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        if (second) {
            // kv_a column block: Step A1 (Z1 -> y1) for this block's own 128 columns, then done (kv_a never needs
            // to wait on q_a's Step B/C).
            ling_pair_mla_a_block(Z1, svh1, (__nv_bfloat16*)y1, M, n1, SK, col0, warp, lane);
            return;
        }
        // q_a column block: Step A2 (Z0 -> y0) for this block's own 128 columns.
        ling_pair_mla_a_block(Z0, svh0, (__nv_bfloat16*)y0, M, n0, SK, col0, warp, lane);
        __syncthreads();
        __threadfence();
        __shared__ int s_qa_last;
        if (threadIdx.x == 0) {
            const int prev_q = atomicAdd(&pair_cnt[32], 1);
            if (prev_q == (n0 / 128) - 1) {
                pair_cnt[32] = 0;
                s_qa_last = 1;
            } else {
                s_qa_last = 0;
            }
        }
        __syncthreads();
        if (!s_qa_last) return;
        __threadfence();

        // The q_a norm + q_b rotation row epilogue (stages the normalized row through s_qa so Step C reads shared
        // instead of global L2; sm[8] fits inside `red` — zero extra shared beyond s_qa).
        __shared__ __align__(16) uint16_t s_qa[1536];
        float* sm = reinterpret_cast<float*>(red);
        for (int row = 0; row < M; ++row) {
            ling_pair_mla_bc_row(mla_qa_w, mla_qb_suh, mla_qb_xh,
                                 (__nv_bfloat16*)y0, M, n0, eps, row, sm, s_qa);
            __syncthreads();
        }
    }
}

// The three-projection linear kernel: columns [0, n0) are projection 0 (K2_0), [n0, n0+n1) projection 1 (K2_12),
// [n0+n1, n0+n1+n2) projection 2 (K2_12). `proj` is uniform across a CTA (blockIdx.x), so the K2 branch has zero
// warp divergence and each side allocates like the single-layer kernel.
template <int K2_0, int K2_12, int CB, int WK, int NT, int H>
__global__ void __launch_bounds__(WK * 32, 1) ling_triple_linear_kernel(
    const half* __restrict__ xh0,
    const half* __restrict__ xh1,
    const half* __restrict__ xh2,
    const uint32_t* __restrict__ T0, long long stride_k0, long long stride_nb0, const half* __restrict__ svh0,
    void* __restrict__ y0, int y_dtype0, int n0,
    const uint32_t* __restrict__ T1, long long stride_k1, long long stride_nb1, const half* __restrict__ svh1,
    void* __restrict__ y1, int y_dtype1, int n1,
    const uint32_t* __restrict__ T2, long long stride_k2, long long stride_nb2, const half* __restrict__ svh2,
    void* __restrict__ y2, int y_dtype2, int n2,
    float* __restrict__ Z, int M, int K, int SK, int ZM) {
    constexpr int RC = (8 / H) * 16;
    extern __shared__ __align__(16) float red[];
    const int RH = min(M, 8);
    float* stage = red + WK * RH * RC;

    const int colg = blockIdx.x * 128;
    const int proj = colg >= n0 ? (colg >= n0 + n1 ? 2 : 1) : 0;
    const int col0 = proj == 0 ? colg : (proj == 1 ? colg - n0 : colg - n0 - n1);
    if (proj == 0) {
        run_linear_cta<K2_0, CB, NT, H, WK>(xh0, T0, stride_k0, stride_nb0, svh0, y0, y_dtype0, n0, col0,
                                  Z, M, K, SK, ZM, red, stage, RH);
    } else {
        const bool p2 = proj == 2;
        run_linear_cta<K2_12, CB, NT, H, WK>(p2 ? xh2 : xh1,
                                   p2 ? T2 : T1, p2 ? stride_k2 : stride_k1, p2 ? stride_nb2 : stride_nb1,
                                   p2 ? svh2 : svh1, p2 ? y2 : y1, p2 ? y_dtype2 : y_dtype1,
                                   p2 ? n2 : n1, col0,
                                   Z, M, K, SK, ZM, red, stage, RH);
    }
}

// ---------------------------------------------------------------------------

struct Plan {
    int sk, wk;
};

const int blocks = 192, min_tiles = 8;

Plan plan_for(int k, int n) {
    const int kt = k / 16, nb = n / 128;
    int sk = 1, wk = nb >= 64 ? 8 : 4;
    if (nb < 64) {                                // a narrow layer needs K splits to fill the SMs at all
        while (nb * sk < blocks && sk < 64) sk *= 2;
    }
    while ((wk > 4 || sk > 1) && (kt % (sk * wk) || kt / (sk * wk) < min_tiles)) {
        if (wk > 4) wk = 4;
        else if (sk > 1) sk /= 2;
        else break;
    }
    return {sk, wk};
}

size_t align256(size_t v) { return (v + 255) & ~size_t(255); }

static inline __host__ __device__ bool aligned16(const void* p) { return ((uintptr_t)p & 15) == 0; }
static inline __host__ __device__ bool aligned8(const void* p) { return ((uintptr_t)p & 7) == 0; }

size_t xf_bytes(int m, int k) { return align256((size_t)((m + 15) / 16) * 16 * k * sizeof(half)); }

bool shape_ok_k2(int k, int n, int k2, int cb) {
    return k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 && k2 >= 2 && k2 <= 16 && cb >= 0 && cb <= 2;
}

bool dtype_ok(int d) { return d >= 0 && d <= 2; }

inline bool pdl_exl3() { return gbp_pdl::on(gbp_pdl::kExl3); }

template <typename... KArgs, typename... Args>
int launch_k(bool pdl, void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
    return gbp_pdl::launch_pdl(pdl, kernel, grid, block, smem, st, args...);
}

int rotate_in_pair(const void* x, int x_dtype, const void* suh0, void* xh0, const void* suh1, void* xh1,
                   int m, int k, cudaStream_t st, bool pdl = false) {
    const int gy_single = (k / 128 + 3) / 4;
    const dim3 grid((unsigned)m, (unsigned)(2 * gy_single));
    return launch_k(pdl, ling_rot_in_pair_kernel, grid, dim3(128), 0, st,
                    x, x_dtype, (const half*)suh0, (half*)xh0, (const half*)suh1, (half*)xh1, k, gy_single);
}

int rotate_in_frag_pair(const void* x, int x_dtype, const void* suh0, void* xf0, const void* suh1, void* xf1,
                        int m, int k, cudaStream_t st, bool pdl = false) {
    const int gy_single = (k / 128 + 3) / 4;
    const dim3 grid((unsigned)((m + 15) / 16 * 16), (unsigned)(2 * gy_single));
    return launch_k(pdl, ling_rot_in_frag_pair_kernel, grid, dim3(128), 0, st,
                    x, x_dtype, (const half*)suh0, (uint32_t*)xf0, (const half*)suh1, (uint32_t*)xf1, k, m, gy_single);
}

// The one-projection rotation wrappers (the single-layer path of gb_ling_linear_add_rmsnorm;
// the same kernels and grid as exl3_linear.cu's rotate_in / rotate_in_frag).
int rotate_in(const void* x, int x_dtype, const void* suh, void* xh, int m, int k, cudaStream_t st, bool pdl = false) {
    const int gy_single = (k / 128 + 3) / 4;
    const dim3 grid((unsigned)m, (unsigned)gy_single);
    return launch_k(pdl, ling_rot_in_kernel, grid, dim3(128), 0, st,
                    x, x_dtype, (const half*)suh, (half*)xh, k);
}

int rotate_in_frag(const void* x, int x_dtype, const void* suh, void* xf, int m, int k, cudaStream_t st, bool pdl = false) {
    const int gy_single = (k / 128 + 3) / 4;
    const dim3 grid((unsigned)((m + 15) / 16 * 16), (unsigned)gy_single);
    return launch_k(pdl, ling_rot_in_frag_kernel, grid, dim3(128), 0, st,
                    x, x_dtype, (const half*)suh, (uint32_t*)xf, k, m);
}

int rotate_in_triple(const void* x, int x_dtype, const void* suh0, void* xh0, const void* suh1, void* xh1,
                     const void* suh2, void* xh2, int m, int k, cudaStream_t st, bool pdl = false) {
    const int gy_single = (k / 128 + 3) / 4;
    const dim3 grid((unsigned)m, (unsigned)(3 * gy_single));
    return launch_k(pdl, ling_rot_in_triple_kernel, grid, dim3(128), 0, st,
                    x, x_dtype, (const half*)suh0, (half*)xh0, (const half*)suh1, (half*)xh1,
                    (const half*)suh2, (half*)xh2, k, gy_single);
}

int rotate_in_frag_triple(const void* x, int x_dtype, const void* suh0, void* xf0, const void* suh1, void* xf1,
                          const void* suh2, void* xf2, int m, int k, cudaStream_t st, bool pdl = false) {
    const int gy_single = (k / 128 + 3) / 4;
    const dim3 grid((unsigned)((m + 15) / 16 * 16), (unsigned)(3 * gy_single));
    return launch_k(pdl, ling_rot_in_frag_triple_kernel, grid, dim3(128), 0, st,
                    x, x_dtype, (const half*)suh0, (uint32_t*)xf0, (const half*)suh1, (uint32_t*)xf1,
                    (const half*)suh2, (uint32_t*)xf2, k, m, gy_single);
}

constexpr int kMaxGroup = 4;

int group_tiles(int m) {
    const int groups = (m + 16 * kMaxGroup - 1) / (16 * kMaxGroup);
    return (((m + groups - 1) / groups) + 15) / 16;
}

template <int K2, int CB>
constexpr int sweeps_of(int nt) {
    return nt >= 3 || (nt == 2 && K2 >= 10 && CB != 2) ? 2 : 1;
}

struct PairArgs {
    const half* xh0;
    const half* xh1;
    const uint32_t* T0; long long stride_k0, stride_nb0; const half* svh0; void* y0; int y_dtype0; int n0;
    const uint32_t* T1; long long stride_k1, stride_nb1; const half* svh1; void* y1; int y_dtype1; int n1;
    float* Z0;
    float* Z1;
    int M, K, SK, wk, zmode;
    bool pdl;
    const half* down_suh;
    half* down_xh;
    int* col_cnt;
    const float* mla_qa_w;
    const half* mla_qb_suh;
    half* mla_qb_xh;
    int* pair_cnt;
    float eps;
};

template <int K2, int CB, int NT, int H>
void launch_pair_nt(const PairArgs& a, const dim3& grid, int rh, cudaStream_t st) {
    constexpr int RC = (8 / H) * 16;
    const size_t stage = (H > 1 && a.SK == 1 && !a.zmode) ? (size_t)16 * NT * 128 * sizeof(float) : 0;
    if (a.wk == 8) {
        launch_k(a.pdl, ling_pair_linear_kernel<K2, CB, 8, NT, H>, grid, dim3(256), 8 * rh * RC * sizeof(float) + stage, st,
                 a.xh0, a.xh1, a.T0, a.stride_k0, a.stride_nb0, a.svh0, a.y0, a.y_dtype0, a.n0,
                 a.T1, a.stride_k1, a.stride_nb1, a.svh1, a.y1, a.y_dtype1, a.n1,
                 a.Z0, a.Z1, a.M, a.K, a.SK, a.zmode,
                 a.down_suh, a.down_xh, a.col_cnt,
                 a.mla_qa_w, a.mla_qb_suh, a.mla_qb_xh, a.pair_cnt, a.eps);
    } else {
        launch_k(a.pdl, ling_pair_linear_kernel<K2, CB, 4, NT, H>, grid, dim3(128), 4 * rh * RC * sizeof(float) + stage, st,
                 a.xh0, a.xh1, a.T0, a.stride_k0, a.stride_nb0, a.svh0, a.y0, a.y_dtype0, a.n0,
                 a.T1, a.stride_k1, a.stride_nb1, a.svh1, a.y1, a.y_dtype1, a.n1,
                 a.Z0, a.Z1, a.M, a.K, a.SK, a.zmode,
                 a.down_suh, a.down_xh, a.col_cnt,
                 a.mla_qa_w, a.mla_qb_suh, a.mla_qb_xh, a.pair_cnt, a.eps);
    }
}

template <int K2, int CB>
int launch_pair(const PairArgs& a, cudaStream_t st) {
    const dim3 grid((unsigned)((a.n0 + a.n1) / 128), (unsigned)a.SK);
    const int rh = a.M < 8 ? a.M : 8;
    switch (group_tiles(a.M)) {
        case 4: launch_pair_nt<K2, CB, 4, sweeps_of<K2, CB>(4)>(a, grid, rh, st); break;
        case 3: launch_pair_nt<K2, CB, 3, sweeps_of<K2, CB>(3)>(a, grid, rh, st); break;
        case 2: launch_pair_nt<K2, CB, 2, sweeps_of<K2, CB>(2)>(a, grid, rh, st); break;
        default: launch_pair_nt<K2, CB, 1, sweeps_of<K2, CB>(1)>(a, grid, rh, st); break;
    }
    return GB_CHECK_LAUNCH();
}

struct TripleArgs {
    const half* xh0;
    const half* xh1;
    const half* xh2;
    const uint32_t* T0; long long stride_k0, stride_nb0; const half* svh0; void* y0; int y_dtype0; int n0;
    const uint32_t* T1; long long stride_k1, stride_nb1; const half* svh1; void* y1; int y_dtype1; int n1;
    const uint32_t* T2; long long stride_k2, stride_nb2; const half* svh2; void* y2; int y_dtype2; int n2;
    float* Z;
    int M, K, SK, wk, zmode;
    bool pdl;
};

template <int K2_0, int K2_12, int CB, int NT, int H>
void launch_triple_nt(const TripleArgs& a, const dim3& grid, int rh, cudaStream_t st) {
    constexpr int RC = (8 / H) * 16;
    const size_t stage = (H > 1 && a.SK == 1 && !a.zmode) ? (size_t)16 * NT * 128 * sizeof(float) : 0;
    if (a.wk == 8) {
        launch_k(a.pdl, ling_triple_linear_kernel<K2_0, K2_12, CB, 8, NT, H>, grid, dim3(256), 8 * rh * RC * sizeof(float) + stage, st,
                 a.xh0, a.xh1, a.xh2, a.T0, a.stride_k0, a.stride_nb0, a.svh0, a.y0, a.y_dtype0, a.n0,
                 a.T1, a.stride_k1, a.stride_nb1, a.svh1, a.y1, a.y_dtype1, a.n1,
                 a.T2, a.stride_k2, a.stride_nb2, a.svh2, a.y2, a.y_dtype2, a.n2,
                 a.Z, a.M, a.K, a.SK, a.zmode);
    } else {
        launch_k(a.pdl, ling_triple_linear_kernel<K2_0, K2_12, CB, 4, NT, H>, grid, dim3(128), 4 * rh * RC * sizeof(float) + stage, st,
                 a.xh0, a.xh1, a.xh2, a.T0, a.stride_k0, a.stride_nb0, a.svh0, a.y0, a.y_dtype0, a.n0,
                 a.T1, a.stride_k1, a.stride_nb1, a.svh1, a.y1, a.y_dtype1, a.n1,
                 a.T2, a.stride_k2, a.stride_nb2, a.svh2, a.y2, a.y_dtype2, a.n2,
                 a.Z, a.M, a.K, a.SK, a.zmode);
    }
}

template <int K2_0, int K2_12, int CB>
int launch_triple(const TripleArgs& a, cudaStream_t st) {
    const dim3 grid((unsigned)((a.n0 + a.n1 + a.n2) / 128), (unsigned)a.SK);
    const int rh = a.M < 8 ? a.M : 8;
    switch (group_tiles(a.M)) {
        case 4: launch_triple_nt<K2_0, K2_12, CB, 4, sweeps_of<K2_12, CB>(4)>(a, grid, rh, st); break;
        case 3: launch_triple_nt<K2_0, K2_12, CB, 3, sweeps_of<K2_12, CB>(3)>(a, grid, rh, st); break;
        case 2: launch_triple_nt<K2_0, K2_12, CB, 2, sweeps_of<K2_12, CB>(2)>(a, grid, rh, st); break;
        default: launch_triple_nt<K2_0, K2_12, CB, 1, sweeps_of<K2_12, CB>(1)>(a, grid, rh, st); break;
    }
    return GB_CHECK_LAUNCH();
}

// For a (K2, CB) pair the dispatcher needs. The KDA f/g decode pair (3.5-bit, mul1) and the dense gate/up pair
// (3-bit, mul1, bf16 out) run K2 6/7; the MLA q_a/kv_a pair runs K2 = 10 (5-bit, mul1). The table is kept small and
// only what the family launches is instantiated here.
int dispatch_pair(int k2, int cb, const PairArgs& a, cudaStream_t st) {
#define GB_PAIR(K2_, CB_) if (k2 == K2_ && cb == CB_) return launch_pair<K2_, CB_>(a, st);
    GB_PAIR(10, 2)
    GB_PAIR(8, 2)
    GB_PAIR(7, 2)
    GB_PAIR(6, 2)
    GB_PAIR(5, 2)
#undef GB_PAIR
    return (int)cudaErrorInvalidValue;
}

int dispatch_triple(int k2_0, int k2_12, int cb, const TripleArgs& a, cudaStream_t st) {
#define GB_TRIPLE(K2_0_, K2_12_, CB_) if (k2_0 == K2_0_ && k2_12 == K2_12_ && cb == CB_) return launch_triple<K2_0_, K2_12_, CB_>(a, st);
    GB_TRIPLE(7, 6, 2)
    GB_TRIPLE(6, 6, 2)
    GB_TRIPLE(7, 7, 2)
#undef GB_TRIPLE
    return (int)cudaErrorInvalidValue;
}

// The exact `ling_prerot_rot4` math of ling_kda_fused.cu (one 128-column block of the next sub-layer's input
// rotation from the bf16 codes of this epilogue's rmsnorm output), copied verbatim: the standalone
// `ling_rot_out_add_rmsnorm_k` runs it under --fmad=false, but every expression is a lone multiply or a
// fwht128 add/sub, so no FMA can form and the bits are the same under this TU's fmad=true.
__device__ __forceinline__ void ling_prerot_rot4(const uint16_t* sy, const half* suh, int col, int lane,
                                                 __half2 (&oh)[2]) {
    float v[4];
    v[0] = __uint_as_float((uint32_t)sy[col] << 16);
    v[1] = __uint_as_float((uint32_t)sy[col + 1] << 16);
    v[2] = __uint_as_float((uint32_t)sy[col + 2] << 16);
    v[3] = __uint_as_float((uint32_t)sy[col + 3] << 16);
    const uint2 us = __ldg(reinterpret_cast<const uint2*>(suh + col));
    const half2* hs = reinterpret_cast<const half2*>(&us);
    const float2 sa = __half22float2(hs[0]);
    const float2 sb = __half22float2(hs[1]);
    float s[4] = {sa.x, sa.y, sb.x, sb.y};
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    gbx::fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= gbx::HAD_SCALE;
    oh[0] = __floats2half2_rn(v[0], v[1]);
    oh[1] = __floats2half2_rn(v[2], v[3]);
}

// Step A of the fused epilogue (`rot_out` finish) plus Step B1 (`hidden + res -> h_out`) of ONE
// 128-column block, run by the last split CTA of each column block (`col_blk = blockIdx.x`) once
// all SK splits' Z strips of that block are visible (the per-column `lin_cnt[col_blk]` handshake).
// The residual read-modify-write is race-free because exactly one CTA touches each column block
// (`hidden_out` aliases `hidden` in decode, and no other CTA ever reads `hidden`). Every expression
// is the standalone `ling_rot_out_add_rmsnorm_k`'s verbatim (explicit RN intrinsics so this TU's
// fmad=true cannot fuse a rounding away): each finish code is rounded to bf16 BEFORE the residual
// add, exactly like the reference's Step A -> s_buf -> Step B1 chain, so the bit patterns written to
// `h_out` are the reference's.
__device__ __forceinline__ void ling_single_fused_column_block(
    const float* __restrict__ Z, const half* __restrict__ svh,
    const uint16_t* __restrict__ hidden, uint16_t* __restrict__ h_out,
    int n, int sk, int M, int col_blk, int warp, int wk, int lane) {
    const int col = col_blk * 128 + 4 * lane;
    if (col >= n) return;                    // columns beyond the row's width do not exist (n <= 4096)
    for (int row = warp; row < M; row += wk) {
        const size_t at = (size_t)row * n + col;
        float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
        for (int q0 = 1; q0 < sk; q0 += 8) {
            float4 u[8];
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) u[d] = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)(q0 + d) * M * n + at));
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) {
                    s.x = __fadd_rn(s.x, u[d].x);
                    s.y = __fadd_rn(s.y, u[d].y);
                    s.z = __fadd_rn(s.z, u[d].z);
                    s.w = __fadd_rn(s.w, u[d].w);
                }
        }
        float v[4] = {s.x, s.y, s.z, s.w};
        gbx::fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __fmul_rn(__fmul_rn(v[j], gbx::HAD_SCALE), __half2float(__ldg(svh + col + j)));
        // Round each finish code to bf16, then add the residual (same operands, order and roundings as
        // the standalone Step A -> s_buf -> Step B1).
        const float r0 = __uint_as_float((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(v[0])) << 16);
        const float r1 = __uint_as_float((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(v[1])) << 16);
        const float r2 = __uint_as_float((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(v[2])) << 16);
        const float r3 = __uint_as_float((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(v[3])) << 16);
        const uint2 hu = *reinterpret_cast<const uint2*>(hidden + at);
        const float2 ha = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hu.x));
        const float2 hb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hu.y));
        uint2 ou;
        ou.x = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(__fadd_rn(ha.x, r0)))
             | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(__fadd_rn(ha.y, r1))) << 16);
        ou.y = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(__fadd_rn(hb.x, r2)))
             | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(__fadd_rn(hb.y, r3))) << 16);
        *reinterpret_cast<uint2*>(h_out + at) = ou;
    }
}

// Step B2 (glue.cu's rmsnorm_k<true> over the completed `h_out`) + Step C (the next sub-layer's
// input rotation from the rmsnorm output) of the fused epilogue, run by the last column-block CTA
// after every column block's `hidden + res` sums are visible in `h_out` (the `lin_cnt[32]` grid-wide
// handshake). Step B runs the 256-thread reduction as two halves t = tid and t = tid + 128 per
// physical warp (only float sm[8] shared, inside `red`), stages `h_out` into `s_buf` so the
// weighted-norm pass doesn't re-read global, and re-stages its output for Step C (prerot reads
// `s_buf`, never a global copy). Bit-identical to the standalone kernel per element.
template <int WK>
__device__ __forceinline__ void ling_single_fused_rmsnorm_row(
    const uint16_t* __restrict__ h_out, const float* __restrict__ norm_w,
    uint16_t* __restrict__ y, uint16_t* s_buf,
    int dim, float eps, int row,
    const half* __restrict__ suh_p0, const half* __restrict__ suh_p1, const half* __restrict__ suh_p2,
    half* __restrict__ xh_p0, half* __restrict__ xh_p1, half* __restrict__ xh_p2,
    float* sm) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t base = (size_t)row * dim;

    // Step B: glue.cu's rmsnorm_k<true>, verbatim (explicit RN intrinsics where --fmad=false kept two roundings).
    // The 256-thread reduction is simulated on the 128-thread block: each physical warp w carries virtual
    // warps w (t = tid) and w + 4 (t = tid + 128), so the per-thread v-set and per-warp butterfly sums are
    // exactly the 256-thread kernel's, and only sm[0..7] is shared.
    const int nvec = dim / 8;
#pragma unroll
    for (int hb = 0; hb < 2; ++hb) {
        const int t = threadIdx.x + 128 * hb;
        float ss = 0.f;
        for (int v = t; v < nvec; v += 256) {
            float f[8];
            const uint4 xu = *(const uint4*)(h_out + base + (size_t)v * 8);
            f[0] = __uint_as_float(xu.x << 16); f[1] = __uint_as_float(xu.x & 0xFFFF0000u);
            f[2] = __uint_as_float(xu.y << 16); f[3] = __uint_as_float(xu.y & 0xFFFF0000u);
            f[4] = __uint_as_float(xu.z << 16); f[5] = __uint_as_float(xu.z & 0xFFFF0000u);
            f[6] = __uint_as_float(xu.w << 16); f[7] = __uint_as_float(xu.w & 0xFFFF0000u);
            const uint4 fp = lpf_pack8(f);
            *(uint4*)(s_buf + (size_t)v * 8) = fp;
#pragma unroll
            for (int i = 0; i < 8; ++i) ss = __fadd_rn(ss, __fmul_rn(f[i], f[i]));
        }
        ss = lpf_warp_sum(ss);
        if (lane == 0) sm[warp + 4 * hb] = ss;
    }
    __syncthreads();
    float s = sm[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) s = s + sm[w];
    const float inv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(__fdiv_rn(s, (float)dim), eps)));
    __syncthreads();  // sm may be reused by the next row; s_buf holds this row's f values
    for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
        uint4 src = *(uint4*)(s_buf + (size_t)v * 8);
        float f[8];
        f[0] = __uint_as_float(src.x << 16); f[1] = __uint_as_float(src.x & 0xFFFF0000u);
        f[2] = __uint_as_float(src.y << 16); f[3] = __uint_as_float(src.y & 0xFFFF0000u);
        f[4] = __uint_as_float(src.z << 16); f[5] = __uint_as_float(src.z & 0xFFFF0000u);
        f[6] = __uint_as_float(src.w << 16); f[7] = __uint_as_float(src.w & 0xFFFF0000u);
        const float4 w0 = *(const float4*)(norm_w + (size_t)v * 8), w1 = *(const float4*)(norm_w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(f[i], inv), wv[i]);
        const uint4 op = lpf_pack8(o);
        *(uint4*)(y + base + (size_t)v * 8) = op;
        *(uint4*)(s_buf + (size_t)v * 8) = op;
    }
    __syncthreads();

    // Step C: the next sub-layer's input rotation from this row's rmsnorm output (staged in `s_buf`).
    // One `for (pp ...)` over the three suh/xh pairs instantiates the `fwht128` rotation once instead of
    // three times, cutting the epilogue's register pressure in `ling_single_linear_kernel`.
    const bool p0 = suh_p0 && xh_p0;
    const bool p1 = suh_p1 && xh_p1;
    const bool p2 = suh_p2 && xh_p2;
    if (p0 || p1 || p2) {
        const half* suh[3] = {suh_p0, suh_p1, suh_p2};
        half* xh[3] = {xh_p0, xh_p1, xh_p2};
        const bool p[3] = {p0, p1, p2};
        for (int blk = warp; blk < dim / 128; blk += WK) {
            const int col = blk * 128 + 4 * lane;
#pragma unroll
            for (int pp = 0; pp < 3; ++pp) {
                if (!p[pp]) continue;
                __half2 oh[2];
                ling_prerot_rot4(s_buf, suh[pp], col, lane, oh);
                uint2 ou;
                ou.x = *reinterpret_cast<const unsigned*>(&oh[0]);
                ou.y = *reinterpret_cast<const unsigned*>(&oh[1]);
                *reinterpret_cast<uint2*>(xh[pp] + base + col) = ou;
            }
        }
    }
}

// The one-projection linear kernel of `gb_ling_linear_add_rmsnorm`: the decode plan's
// `linear_kernel` over one projection, always writing the fp32 warp-order partial sums
// to Z (ZM = 1). When the fused epilogue is active (`norm_w` and `lin_cnt` given, block 128
// threads) the grid parallelizes its `ling_rot_out_add_rmsnorm_k` math across the 32 column
// blocks instead of serializing all of it onto one CTA: the last split CTA of each column
// block (`col_blk = blockIdx.x`, per-column self-resetting counter `lin_cnt[col_blk]`) finishes
// that block's rows and adds the residual in place (`hidden_out`), and the last column block to
// finish (grid-wide self-resetting counter `lin_cnt[32]`) runs the rmsnorm + prerot epilogue
// for every row. 224 of the 256 CTAs exit after touching only their own per-column counter;
// the counters live in a 512-byte region at the end of `prerot_ws` (zeroed at allocation,
// self-resetting so repeated launches reuse them). The linear phase itself is the plan's exact
// work with compile-time WK (bit-identical per element to the separate `rot_out_kernel` +
// `rmsnorm_k<true>`).
template <int K2, int CB, int WK, int NT, int H>
__global__ void __launch_bounds__(WK * 32, 1) ling_single_linear_kernel(
    const half* __restrict__ xh,
    const uint32_t* __restrict__ T, long long stride_k, long long stride_nb, const half* __restrict__ svh,
    void* __restrict__ y, int y_dtype, int n,
    uint16_t* __restrict__ hidden_out,
    float* __restrict__ Z, int M, int K, int SK, int ZM,
    const uint16_t* __restrict__ hidden, const float* __restrict__ norm_w,
    int dim, float eps,
    const half* __restrict__ suh_p0, const half* __restrict__ suh_p1, const half* __restrict__ suh_p2,
    half* __restrict__ xh_p0, half* __restrict__ xh_p1, half* __restrict__ xh_p2,
    int* __restrict__ lin_cnt) {
    constexpr int RC = (8 / H) * 16;
    extern __shared__ __align__(16) float red[];
    const int RH = min(M, 8);
    const int colg = blockIdx.x * 128;
    run_linear_cta<K2, CB, NT, H, WK>(xh, T, stride_k, stride_nb, svh, y, y_dtype, n, colg,
                            Z, M, K, SK, ZM, red, red + WK * RH * RC, RH);

    if (norm_w != nullptr && lin_cnt != nullptr && blockDim.x == 128) {
        __syncthreads();
        __threadfence();
        __shared__ int s_last;
        const int col_blk = blockIdx.x;
        if (threadIdx.x == 0) {
            const int prev_s = atomicAdd(&lin_cnt[col_blk], 1);
            if (prev_s == (int)gridDim.y - 1) {
                lin_cnt[col_blk] = 0;
                s_last = 1;
            } else {
                s_last = 0;
            }
        }
        __syncthreads();
        if (!s_last) return;

        // Step A + B1: this column block's rot_out finish + residual add, all rows in parallel across
        // the 32 column-block CTAs (4 warps each). No shared memory needed (no cross-warp exchange).
        ling_single_fused_column_block(Z, svh, hidden, hidden_out, n, SK, M, col_blk,
                                       threadIdx.x >> 5, WK, threadIdx.x & 31);

        __syncthreads();
        __threadfence();
        if (threadIdx.x == 0) {
            const int prev_c = atomicAdd(&lin_cnt[32], 1);
            if (prev_c == (int)gridDim.x - 1) {
                lin_cnt[32] = 0;
                s_last = 1;
            } else {
                s_last = 0;
            }
        }
        __syncthreads();
        if (!s_last) return;
        __threadfence();

        // Step B2 + Step C: the last column block runs the row-wise rmsnorm over the complete
        // `hidden_out` (already in global L2) and the next sub-layer's input rotation. Static shared
        // is 8,192 B (s_buf) + 4 B (s_last) = 10.2 KB/CTA total with the linear phase's dynamic
        // `red`, still 2 CTAs/SM at the linear phase's register count. `__align__(16)` (not the
        // uint16_t natural 2) so the uint4/uint2 views of s_buf are always aligned regardless of the
        // static layout ptxas assigns.
        __shared__ __align__(16) uint16_t s_buf[4096];
        float* sm = reinterpret_cast<float*>(red);
        for (int row = 0; row < M; ++row) {
            ling_single_fused_rmsnorm_row<WK>(hidden_out, norm_w, (uint16_t*)y, s_buf,
                                               dim, eps, row,
                                               suh_p0, suh_p1, suh_p2, xh_p0, xh_p1, xh_p2, sm);
            __syncthreads();
        }
    }
}

struct SingleArgs {
    const half* xh;
    const uint32_t* T; long long stride_k, stride_nb; const half* svh; void* y; int y_dtype; int n;
    uint16_t* hidden_out;
    float* Z;
    int M, K, SK, wk, zm;
    bool pdl;
    const uint16_t* hidden;
    const float* norm_w;
    int dim; float eps;
    const half* suh_p0, *suh_p1, *suh_p2;
    half* xh_p0, *xh_p1, *xh_p2;
    int* lin_cnt;
};

template <int K2, int CB, int NT, int H>
void launch_single_nt(const SingleArgs& a, const dim3& grid, int rh, cudaStream_t st) {
    constexpr int RC = (8 / H) * 16;
    // The fused epilogue rides the kernel's native 128-thread WK = 4 block (the fused sites are n <= 4096,
    // so the plan's wk is always 4) and parallelizes rot_out finish + residual add across the column
    // blocks, then the last column block runs the row-wise rmsnorm + prerot through the static 8-KiB
    // `s_buf` (plus float sm[8] in `red`); the dynamic shared memory stays exactly the linear phase's
    // WK * RH * RC floats, and the static 8,196 B keeps 2 CTAs/SM at the linear phase's register count.
    const size_t smem = (size_t)a.wk * rh * RC * sizeof(float);
    if (a.wk == 8 && !(a.norm_w != nullptr && a.lin_cnt != nullptr)) {
        launch_k(a.pdl, ling_single_linear_kernel<K2, CB, 8, NT, H>, grid, dim3(256), smem, st,
                 a.xh, a.T, a.stride_k, a.stride_nb, a.svh, a.y, a.y_dtype, a.n, a.hidden_out,
                 a.Z, a.M, a.K, a.SK, a.zm,
                 a.hidden, a.norm_w, a.dim, a.eps,
                 a.suh_p0, a.suh_p1, a.suh_p2, a.xh_p0, a.xh_p1, a.xh_p2, a.lin_cnt);
    } else {
        launch_k(a.pdl, ling_single_linear_kernel<K2, CB, 4, NT, H>, grid, dim3(128), smem, st,
                 a.xh, a.T, a.stride_k, a.stride_nb, a.svh, a.y, a.y_dtype, a.n, a.hidden_out,
                 a.Z, a.M, a.K, a.SK, a.zm,
                 a.hidden, a.norm_w, a.dim, a.eps,
                 a.suh_p0, a.suh_p1, a.suh_p2, a.xh_p0, a.xh_p1, a.xh_p2, a.lin_cnt);
    }
}

template <int K2, int CB>
int launch_single(const SingleArgs& a, cudaStream_t st) {
    const dim3 grid((unsigned)(a.n / 128), (unsigned)a.SK);
    const int rh = a.M < 8 ? a.M : 8;
    switch (group_tiles(a.M)) {
        case 4: launch_single_nt<K2, CB, 4, sweeps_of<K2, CB>(4)>(a, grid, rh, st); break;
        case 3: launch_single_nt<K2, CB, 3, sweeps_of<K2, CB>(3)>(a, grid, rh, st); break;
        case 2: launch_single_nt<K2, CB, 2, sweeps_of<K2, CB>(2)>(a, grid, rh, st); break;
        default: launch_single_nt<K2, CB, 1, sweeps_of<K2, CB>(1)>(a, grid, rh, st); break;
    }
    return GB_CHECK_LAUNCH();
}

int dispatch_single(int k2, int cb, const SingleArgs& a, cudaStream_t st) {
#define GB_SINGLE(K2_, CB_) if (k2 == K2_ && cb == CB_) return launch_single<K2_, CB_>(a, st);
    GB_SINGLE(12, 2)
    GB_SINGLE(10, 2)
    GB_SINGLE(7, 2)
#undef GB_SINGLE
    return (int)cudaErrorInvalidValue;
}

// The Ling-3.1 MoE router GEMV (`gb_ling_bf16_linear_f32`): y [rows, n] fp32 =
// x [rows, k] bf16 @ w [n, k]^T bf16, one warp per output column (16 warps per CTA,
// 32 CTAs = 512 columns = a single wave on GB10's 48 SMs). Per (row, column) the
// FMA chain and the warp reduction are exactly `gb_mlag_bf16_linear_f32_kernel`'s
// (same k order, same lane stride, same `warp_reduce_sum5`, fp32 store), so every
// logit's bits are unchanged; only the weight rows are loaded once per column across
// the rows of a chunk (rows <= 8 per chunk, any row count).
__device__ __forceinline__ float ling_fma8_bf16(float acc, const uint4 xu, const uint4 wu) {
    const float2 xa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu.x));
    const float2 xb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu.y));
    const float2 xc = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu.z));
    const float2 xd = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu.w));
    const float2 wa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wu.x));
    const float2 wb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wu.y));
    const float2 wc = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wu.z));
    const float2 wd = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wu.w));
    acc = __fmaf_rn(xa.x, wa.x, acc);
    acc = __fmaf_rn(xa.y, wa.y, acc);
    acc = __fmaf_rn(xb.x, wb.x, acc);
    acc = __fmaf_rn(xb.y, wb.y, acc);
    acc = __fmaf_rn(xc.x, wc.x, acc);
    acc = __fmaf_rn(xc.y, wc.y, acc);
    acc = __fmaf_rn(xd.x, wd.x, acc);
    acc = __fmaf_rn(xd.y, wd.y, acc);
    return acc;
}

__device__ __forceinline__ float ling_warp_reduce_sum5(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

template <int ROWS>
__global__ void __launch_bounds__(512) ling_bf16_linear_f32_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    float* __restrict__ y, int rows, int k, int n) {
    const int j = blockIdx.x * 16 + (threadIdx.x >> 5);
    if (j >= n) return;
    const int lane = threadIdx.x & 31;
    const int k8 = k >> 3;
    const int chunks = (rows + ROWS - 1) / ROWS;
    for (int ch = 0; ch < chunks; ++ch) {
        const int r0 = ch * ROWS;
        const int nr = min(ROWS, rows - r0);
        float acc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
        for (int c8 = lane; c8 < k8; c8 += 32) {
            const uint4 wu = *reinterpret_cast<const uint4*>(w + (int64_t)j * k + c8 * 8);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r >= nr) break;
                const uint4 xu = *reinterpret_cast<const uint4*>(x + (int64_t)(r0 + r) * k + c8 * 8);
                acc[r] = ling_fma8_bf16(acc[r], xu, wu);
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r >= nr) break;
            const float v = ling_warp_reduce_sum5(acc[r]);
            if (lane == 0) y[(int64_t)(r0 + r) * n + j] = v;
        }
    }
}

// One launch replacing glue.cu's rmsnorm_k<false> (in place on x) + exl3_linear.cu's rot_in for a Ling MLA
// q_a -> q_b step: row r of x (bf16 [rows][k], k == dim <= 1536) is normalised in place with the exact
// rmsnorm_k<false> math (explicit RN ops: this TU compiles without --fmad=false), staged into shared memory,
// and the first 128 threads rotate each 128-column block of the normalised row with the exact rot_in_kernel
// math (suh fp16 scaling, the 128-point Walsh-Hadamard butterfly in the same fixed order, HAD_SCALE, fp16
// stores) into xh (fp16 [rows][k]). The rotation's shared s_had is reused across the k/128 blocks with a
// barrier between the stages (the butterfly order per output equals fwht128's: distances 1, 2, 4, ..., 64).
__global__ void __launch_bounds__(256) ling_rmsnorm_rot_in_kernel(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
    __nv_bfloat16* __restrict__ y, const half* __restrict__ suh, half* __restrict__ xh,
    int dim, int K, float eps) {
    pdl_wait();
    __shared__ __align__(16) __nv_bfloat16 s_qa[1536];
    __shared__ float s_had[128];
    __shared__ float sm[8];
    const int r = blockIdx.x;
    const size_t base = (size_t)r * dim;
    const int nvec = dim / 8;
    float ss = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8];
        lpf_unpack8(*(const uint4*)(x + base + (size_t)v * 8), f);
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = __fadd_rn(ss, __fmul_rn(f[i], f[i]));
    }
    ss = lpf_block_sum<256>(ss, sm);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8], o[8];
        lpf_unpack8(*(const uint4*)(x + base + (size_t)v * 8), f);
        const float4 w0 = *(const float4*)(w + (size_t)v * 8), w1 = *(const float4*)(w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(f[i], inv), wv[i]);
        const uint4 pk = lpf_pack8(o);
        *(uint4*)(y + base + (size_t)v * 8) = pk;
        *(uint4*)(s_qa + v * 8) = pk;
    }
    __syncthreads();
    const int tid = threadIdx.x;
    for (int nb = 0; nb < K / 128; ++nb) {
        if (tid < 128)
            s_had[tid] = lpf_bf2f(__bfloat16_as_ushort(s_qa[nb * 128 + tid])) * __half2float(suh[nb * 128 + tid]);
        __syncthreads();
        for (int d = 1; d < 128; d <<= 1) {
            const float o = tid < 128 ? s_had[tid ^ d] : 0.0f;
            __syncthreads();
            if (tid < 128) s_had[tid] = (tid & d) ? o - s_had[tid] : s_had[tid] + o;
            __syncthreads();
        }
        if (tid < 128) xh[(size_t)r * K + nb * 128 + tid] = __float2half_rn(s_had[tid] * HAD_SCALE);
        __syncthreads();
    }
}

// The MLA q_a/kv_a pair's split-K finishes plus the q_b input norm and rotation in ONE launch: CTA row r of
// grid (m, 1) (256 threads) reduces both projections' fp32 split partials of its row in exact rot_out_kernel
// order (Z1 -> y1 bf16; Z0 -> shared s_qa rounded to bf16), normalises s_qa with qa_w using the exact
// rmsnorm_k<false> math (writing the normalised bf16 to y0 AND back into s_qa), then rotates s_qa with qb_suh
// into qb_xh using the exact ling_rmsnorm_rot_in_kernel rotation math (the shared s_had butterfly, HAD_SCALE,
// fp16 stores). Bit-identical per element to the two separate `gb_exl3_rotate_out_sk` launches + the
// `ling_rmsnorm_rot_in_kernel` launch (so to the separate launches they replace).
__global__ void __launch_bounds__(256) ling_mla_rot_out_qnorm_qb_rot_in_kernel(
    const float* __restrict__ Z0, const float* __restrict__ Z1,
    const half* __restrict__ svh0, const half* __restrict__ svh1,
    __nv_bfloat16* __restrict__ y0, __nv_bfloat16* __restrict__ y1,
    const float* __restrict__ qa_w, const half* __restrict__ qb_suh, half* __restrict__ qb_xh,
    int m, int n0, int n1, int sk, float eps) {
    pdl_wait();
    __shared__ __align__(16) __nv_bfloat16 s_qa[1536];
    __shared__ float s_had[128];
    __shared__ float sm[8];
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    // Step A1: Z1 [sk][m][n1] -> y1, exact rot_out_kernel order (one warp per 128-column block).
    if (warp < n1 / 128) {
        const int blk = warp, col = blk * 128 + 4 * lane;
        const size_t at = (size_t)row * n1 + col;
        float4 s = __ldcg(reinterpret_cast<const float4*>(Z1 + at));
        for (int q0 = 1; q0 < sk; q0 += 8) {
            float4 u[8];
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) u[d] = __ldcg(reinterpret_cast<const float4*>(Z1 + (size_t)(q0 + d) * m * n1 + at));
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) { s.x += u[d].x; s.y += u[d].y; s.z += u[d].z; s.w += u[d].w; }
        }
        pdl_trigger();
        float v[4] = {s.x, s.y, s.z, s.w};
        finish(v, lane, svh1, nullptr, col);
        store4(y1, BF16, at, v);
    }

    // Step A2: Z0 [sk][m][n0] -> s_qa (rounded bf16), the same finish; the 128-column blocks are spread over
    // the 8 warps so a row's whole Z0 strip is reduced by this CTA.
    for (int blk = warp; blk < n0 / 128; blk += 8) {
        const int col = blk * 128 + 4 * lane;
        const size_t at = (size_t)row * n0 + col;
        float4 s = __ldcg(reinterpret_cast<const float4*>(Z0 + at));
        for (int q0 = 1; q0 < sk; q0 += 8) {
            float4 u[8];
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) u[d] = __ldcg(reinterpret_cast<const float4*>(Z0 + (size_t)(q0 + d) * m * n0 + at));
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) { s.x += u[d].x; s.y += u[d].y; s.z += u[d].z; s.w += u[d].w; }
        }
        pdl_trigger();
        float v[4] = {s.x, s.y, s.z, s.w};
        finish(v, lane, svh0, nullptr, col);
        const uint2 u = {lpf_pack2(v[0], v[1]), lpf_pack2(v[2], v[3])};
        *(uint2*)(s_qa + col) = u;
    }
    __syncthreads();

    // Step B: the exact ling_rmsnorm_rot_in_kernel step-1 body (explicit RN ops, lpf_block_sum) over s_qa,
    // writing the normalised bf16 to y0 AND back into s_qa.
    {
        const size_t base = (size_t)row * n0;
        const int nvec = n0 / 8;
        float ss = 0.0f;
        for (int v = threadIdx.x; v < nvec; v += 256) {
            float f[8];
            lpf_unpack8(*(const uint4*)(s_qa + v * 8), f);
#pragma unroll
            for (int i = 0; i < 8; ++i) ss = __fadd_rn(ss, __fmul_rn(f[i], f[i]));
        }
        ss = lpf_block_sum<256>(ss, sm);
        const float inv = 1.0f / sqrtf(ss / (float)n0 + eps);
        for (int v = threadIdx.x; v < nvec; v += 256) {
            float f[8], o[8];
            lpf_unpack8(*(const uint4*)(s_qa + v * 8), f);
            const float4 w0 = *(const float4*)(qa_w + (size_t)v * 8), w1 = *(const float4*)(qa_w + (size_t)v * 8 + 4);
            const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(f[i], inv), wv[i]);
            const uint4 pk = lpf_pack8(o);
            *(uint4*)(y0 + base + (size_t)v * 8) = pk;
            *(uint4*)(s_qa + v * 8) = pk;
        }
    }
    __syncthreads();

    // Step C: the exact ling_rmsnorm_rot_in_kernel step-2 rotation of s_qa with qb_suh into qb_xh.
    {
        const int tid = threadIdx.x;
        for (int nb = 0; nb < n0 / 128; ++nb) {
            if (tid < 128)
                s_had[tid] = lpf_bf2f(__bfloat16_as_ushort(s_qa[nb * 128 + tid])) * __half2float(__ldg(qb_suh + nb * 128 + tid));
            __syncthreads();
            for (int d = 1; d < 128; d <<= 1) {
                const float o = tid < 128 ? s_had[tid ^ d] : 0.0f;
                __syncthreads();
                if (tid < 128) s_had[tid] = (tid & d) ? o - s_had[tid] : s_had[tid] + o;
                __syncthreads();
            }
            if (tid < 128) qb_xh[(size_t)row * n0 + nb * 128 + tid] = __float2half_rn(s_had[tid] * HAD_SCALE);
            __syncthreads();
        }
    }
}


// SwiGLU fused into the input rotation of `gb_ling_linear_add_rmsnorm`'s rot_in stage: each warp loads four
// g/u bf16 pairs, computes the exact `ew_vec_k<OpSwiglu>` element (silu_f(g) * u rounded to bf16; glue.cu
// compiles --fmad=false, so the op sequence is explicit RN intrinsics here), then runs the identical rot_in
// math (suh fp16 scaling, the 128-point Walsh-Hadamard butterfly, HAD_SCALE, fp16 stores) into xh. Grid and
// block are ling_rot_in_kernel's; each output element is the separate `glue::swiglu` + rot_in's, bit for bit.
__global__ void __launch_bounds__(128) ling_silu_mul_rot_in_kernel(
    const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ u,
    const half* __restrict__ suh, half* __restrict__ xh, int K) {
    pdl_wait();
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float s[4], v[4];
    load4(suh, F16, k, s);
    const size_t at = (size_t)row * K + k;
    const uint2 gu = *reinterpret_cast<const uint2*>(g + at);
    const uint2 uu = *reinterpret_cast<const uint2*>(u + at);
    const float2 ga = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gu.x));
    const float2 gb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gu.y));
    const float2 ua = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uu.x));
    const float2 ub = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uu.y));
    const float sg0 = __fdiv_rn(ga.x, __fadd_rn(1.0f, expf(-ga.x)));
    const float sg1 = __fdiv_rn(ga.y, __fadd_rn(1.0f, expf(-ga.y)));
    const float sg2 = __fdiv_rn(gb.x, __fadd_rn(1.0f, expf(-gb.x)));
    const float sg3 = __fdiv_rn(gb.y, __fadd_rn(1.0f, expf(-gb.y)));
    v[0] = lpf_bf2f(lpf_f2bf(__fmul_rn(sg0, ua.x)));
    v[1] = lpf_bf2f(lpf_f2bf(__fmul_rn(sg1, ua.y)));
    v[2] = lpf_bf2f(lpf_f2bf(__fmul_rn(sg2, ub.x)));
    v[3] = lpf_bf2f(lpf_f2bf(__fmul_rn(sg3, ub.y)));
    pdl_trigger();
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(xh, F16, at, v);
}

// One launch replacing glue.cu's rmsnorm_k<false>/<true> + exl3_linear.cu's rot_in TRIPLE for the layer input
// norm: row r of x (bf16 [rows][dim], dim == k <= 4096) is normalised with the exact rmsnorm_k math (explicit
// RN ops: this TU compiles without --fmad=false), staged into shared memory; the add variant first writes h =
// bf16(x + res) to h_out AND s_hid, exactly like rmsnorm_k<true> storing h_out then normalising from it. Then
// the 8 warps rotate each 128-column block of the normalised row with the exact ling_rot_in_triple_kernel math
// (suh fp16 scaling, the butterfly, HAD_SCALE, fp16 stores) into xh0/xh1/xh2.
//
// Embedding fusion (layer 0, decode): when `token_ids`/`embed_table` are given, `x` is not read at all — row r's
// input is `embed_table[token_ids[r]]` (bf16 [vocab][dim], the exact `embed_vec_k` copy: an id outside
// `0..vocab` gives a zero row), written to `h_out` (the hidden stream) AND staged in `s_hid` before the norm, so
// the standalone `glue::embed` launch disappears and Layer 0's residual add reads the same bytes it would have.
__global__ void __launch_bounds__(256) ling_norm_rot_in_triple_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ res,
    const float* __restrict__ w, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ y,
    const half* __restrict__ suh0, half* __restrict__ xh0,
    const half* __restrict__ suh1, half* __restrict__ xh1,
    const half* __restrict__ suh2, half* __restrict__ xh2,
    int dim, int K, float eps,
    const int32_t* __restrict__ token_ids, const void* __restrict__ embed_table, int vocab) {
    pdl_wait();
    __shared__ __align__(16) __nv_bfloat16 s_hid[4096];
    __shared__ __align__(16) __nv_bfloat16 s_norm[4096];
    __shared__ float sm[8];
    const int r = blockIdx.x;
    const size_t base = (size_t)r * dim;
    const int nvec = dim / 8;
    const bool add = res != nullptr;
    const bool embed = token_ids && embed_table;
    const int tok = embed ? token_ids[r] : 0;
    const bool ok = embed ? ((unsigned)tok < (unsigned)vocab) : false;
    const __nv_bfloat16* esrc = embed ? static_cast<const __nv_bfloat16*>(embed_table) + (size_t)(unsigned)tok * dim : nullptr;
    float ss = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8];
        if (embed) {
            const uint4 eu = ok ? *reinterpret_cast<const uint4*>(esrc + (size_t)v * 8)
                                : make_uint4(0u, 0u, 0u, 0u);
            lpf_unpack8(eu, f);
            const uint4 pk = lpf_pack8(f);
            *(uint4*)(h_out + base + (size_t)v * 8) = pk;
            *(uint4*)(s_hid + v * 8) = pk;
        } else {
            lpf_unpack8(*(const uint4*)(x + base + (size_t)v * 8), f);
            if (add) {
                float rr[8];
                lpf_unpack8(*(const uint4*)(res + base + (size_t)v * 8), rr);
#pragma unroll
                for (int i = 0; i < 8; ++i) f[i] = lpf_bf2f(lpf_f2bf(__fadd_rn(f[i], rr[i])));
                const uint4 pk = lpf_pack8(f);
                *(uint4*)(h_out + base + (size_t)v * 8) = pk;
                *(uint4*)(s_hid + v * 8) = pk;
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = __fadd_rn(ss, __fmul_rn(f[i], f[i]));
    }
    ss = lpf_block_sum<256>(ss, sm);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    const __nv_bfloat16* src = (add || embed) ? s_hid : x;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8], o[8];
        lpf_unpack8(*(const uint4*)(src + ((add || embed) ? (size_t)0 : base) + (size_t)v * 8), f);
        const float4 w0 = *(const float4*)(w + (size_t)v * 8), w1 = *(const float4*)(w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(f[i], inv), wv[i]);
        const uint4 pk = lpf_pack8(o);
        *(uint4*)(y + base + (size_t)v * 8) = pk;
        *(uint4*)(s_norm + v * 8) = pk;
    }
    pdl_trigger();
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int blk = warp; blk < K / 128; blk += 8) {
        const int k = blk * 128 + 4 * lane;
        float s[4], v[4];
        load4(suh0, F16, k, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = lpf_bf2f(__bfloat16_as_ushort(s_norm[k + j]));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= s[j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
        store4(xh0, F16, (size_t)r * K + k, v);
        load4(suh1, F16, k, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = lpf_bf2f(__bfloat16_as_ushort(s_norm[k + j]));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= s[j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
        store4(xh1, F16, (size_t)r * K + k, v);
        load4(suh2, F16, k, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = lpf_bf2f(__bfloat16_as_ushort(s_norm[k + j]));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= s[j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
        store4(xh2, F16, (size_t)r * K + k, v);
    }
}

// The pair variant of ling_norm_rot_in_triple_kernel: rotations into xh0/xh1 only.
__global__ void __launch_bounds__(256) ling_norm_rot_in_pair_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ res,
    const float* __restrict__ w, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ y,
    const half* __restrict__ suh0, half* __restrict__ xh0,
    const half* __restrict__ suh1, half* __restrict__ xh1,
    int dim, int K, float eps) {
    pdl_wait();
    __shared__ __align__(16) __nv_bfloat16 s_hid[4096];
    __shared__ __align__(16) __nv_bfloat16 s_norm[4096];
    __shared__ float sm[8];
    const int r = blockIdx.x;
    const size_t base = (size_t)r * dim;
    const int nvec = dim / 8;
    const bool add = res != nullptr;
    float ss = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8];
        lpf_unpack8(*(const uint4*)(x + base + (size_t)v * 8), f);
        if (add) {
            float rr[8];
            lpf_unpack8(*(const uint4*)(res + base + (size_t)v * 8), rr);
#pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = lpf_bf2f(lpf_f2bf(__fadd_rn(f[i], rr[i])));
            const uint4 pk = lpf_pack8(f);
            *(uint4*)(h_out + base + (size_t)v * 8) = pk;
            *(uint4*)(s_hid + v * 8) = pk;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = __fadd_rn(ss, __fmul_rn(f[i], f[i]));
    }
    ss = lpf_block_sum<256>(ss, sm);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    const __nv_bfloat16* src = add ? s_hid : x;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8], o[8];
        lpf_unpack8(*(const uint4*)(src + (add ? (size_t)0 : base) + (size_t)v * 8), f);
        const float4 w0 = *(const float4*)(w + (size_t)v * 8), w1 = *(const float4*)(w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(f[i], inv), wv[i]);
        const uint4 pk = lpf_pack8(o);
        *(uint4*)(y + base + (size_t)v * 8) = pk;
        *(uint4*)(s_norm + v * 8) = pk;
    }
    pdl_trigger();
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int blk = warp; blk < K / 128; blk += 8) {
        const int k = blk * 128 + 4 * lane;
        float s[4], v[4];
        load4(suh0, F16, k, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = lpf_bf2f(__bfloat16_as_ushort(s_norm[k + j]));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= s[j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
        store4(xh0, F16, (size_t)r * K + k, v);
        load4(suh1, F16, k, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = lpf_bf2f(__bfloat16_as_ushort(s_norm[k + j]));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= s[j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
        store4(xh1, F16, (size_t)r * K + k, v);
    }
}

// The single variant of ling_norm_rot_in_pair_kernel: one rotation into xh (the trunk lm_head's input rotation,
// fused with the last layer's final residual add + final norm).
__global__ void __launch_bounds__(256) ling_norm_rot_in_single_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ res,
    const float* __restrict__ w, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ y,
    const half* __restrict__ suh, half* __restrict__ xh,
    int dim, int K, float eps) {
    pdl_wait();
    __shared__ __align__(16) __nv_bfloat16 s_hid[4096];
    __shared__ __align__(16) __nv_bfloat16 s_norm[4096];
    __shared__ float sm[8];
    const int r = blockIdx.x;
    const size_t base = (size_t)r * dim;
    const int nvec = dim / 8;
    const bool add = res != nullptr;
    float ss = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8];
        lpf_unpack8(*(const uint4*)(x + base + (size_t)v * 8), f);
        if (add) {
            float rr[8];
            lpf_unpack8(*(const uint4*)(res + base + (size_t)v * 8), rr);
#pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = lpf_bf2f(lpf_f2bf(__fadd_rn(f[i], rr[i])));
            const uint4 pk = lpf_pack8(f);
            *(uint4*)(h_out + base + (size_t)v * 8) = pk;
            *(uint4*)(s_hid + v * 8) = pk;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = __fadd_rn(ss, __fmul_rn(f[i], f[i]));
    }
    ss = lpf_block_sum<256>(ss, sm);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    const __nv_bfloat16* src = add ? s_hid : x;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8], o[8];
        lpf_unpack8(*(const uint4*)(src + (add ? (size_t)0 : base) + (size_t)v * 8), f);
        const float4 w0 = *(const float4*)(w + (size_t)v * 8), w1 = *(const float4*)(w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __fmul_rn(__fmul_rn(f[i], inv), wv[i]);
        const uint4 pk = lpf_pack8(o);
        *(uint4*)(y + base + (size_t)v * 8) = pk;
        *(uint4*)(s_norm + v * 8) = pk;
    }
    pdl_trigger();
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int blk = warp; blk < K / 128; blk += 8) {
        const int k = blk * 128 + 4 * lane;
        float s[4], v[4];
        load4(suh, F16, k, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = lpf_bf2f(__bfloat16_as_ushort(s_norm[k + j]));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= s[j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
        store4(xh, F16, (size_t)r * K + k, v);
    }
}

// --------------------------------------------------------------------------- top-k (fused stage 1 + stage 2)
// The exact `topk_stage1_k` / `topk_stage2_k` of glm_glue.cu, copied into this Ling-owned file and fused into ONE
// launch: every CTA runs the stage-1 body for its (chunk, row), posts to a self-resetting per-row atomic counter, and
// the row's LAST CTA (prev == gridDim.x - 1) runs the stage-2 body in place. Bit-identical ids/vals to
// `gb_glm_topk_f16` (same pool keys, same merges, same threshold, same decode). The workspace holds the stage-1 pools
// preceded by a fixed 256-byte region of `rows` int32 counters at offset 0 (up to 64 rows; self-reset; zero when the
// buffer is allocated), so the pools at workspace + 256 can never overwrite the counters when rows/count vary across
// calls on one buffer. No data atomics: the counter is a barrier, the pool merge reads are ordered by the fences
// around the increments.

typedef uint16_t ltk_u16;
typedef uint32_t ltk_u32;
typedef uint64_t ltk_u64;

constexpr int LKT_NT = 256;
constexpr int LKT_TILE = LKT_NT * 4;
constexpr int LKT_CAP = 2048;
constexpr int LKT_FLUSH = 512;
constexpr int LKT_PF = 2;
constexpr int LKT_CHUNK = 16384;

__device__ __forceinline__ float ltk_h2f(ltk_u16 u) { return __half2float(__ushort_as_half(u)); }
__device__ __forceinline__ ltk_u16 ltk_f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

__device__ __forceinline__ ltk_u64 ltk_make_key(ltk_u16 u, ltk_u32 id) {
    if ((u & 0x7FFF) > 0x7C00) u = 0xFC00;
    else if ((u & 0x7FFF) == 0) u = 0;
    const ltk_u32 t = (u & 0x8000) ? (~(ltk_u32)u & 0xFFFF) : ((ltk_u32)u | 0x8000);
    return ((ltk_u64)t << 32) | (ltk_u64)(~id);
}

__device__ void ltk_bitonic_sort_desc(ltk_u64* arr, int M) {
    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int p = threadIdx.x; p < M / 2; p += LKT_NT) {
                const int i = ((p & ~(j - 1)) << 1) | (p & (j - 1));
                const int l = i | j;
                const ltk_u64 a = arr[i], b = arr[l];
                const bool desc = (i & k) == 0;
                if (desc ? a < b : a > b) { arr[i] = b; arr[l] = a; }
            }
            __syncthreads();
        }
    }
}

template <int P>
struct LtkShared {
    ltk_u64 pool[P];
    ltk_u64 cand[LKT_CAP];
    int cnt;
    ltk_u64 thr;
};

template <int P>
__device__ void ltk_merge(LtkShared<P>& s, int count) {
    __syncthreads();
    const int cnt = s.cnt;
    int M = P;
    while (M < cnt) M <<= 1;
    for (int i = cnt + threadIdx.x; i < M; i += LKT_NT) s.cand[i] = 0;
    __syncthreads();
    ltk_bitonic_sort_desc(s.cand, M);
    if (threadIdx.x < P) {
        const ltk_u64 a = s.pool[threadIdx.x], b = s.cand[P - 1 - threadIdx.x];
        s.pool[threadIdx.x] = a > b ? a : b;
    }
    __syncthreads();
    for (int j = P / 2; j > 0; j >>= 1) {
        ltk_u64 a = 0, b = 0;
        int i = 0, l = 0;
        const bool act = threadIdx.x < P && (threadIdx.x & j) == 0;
        if (act) {
            i = threadIdx.x; l = i | j;
            a = s.pool[i]; b = s.pool[l];
            if (a < b) { s.pool[i] = b; s.pool[l] = a; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { s.cnt = 0; s.thr = s.pool[count - 1]; }
    __syncthreads();
}

template <int P, class Src>
__device__ void ltk_select(LtkShared<P>& s, const Src& src, int len, int count) {
    if (threadIdx.x < P) s.pool[threadIdx.x] = 0;
    if (threadIdx.x == 0) { s.cnt = 0; s.thr = 0; }
    __syncthreads();
    const int ntile = (len + LKT_TILE - 1) / LKT_TILE;
    const int lane = threadIdx.x & 31;
    ltk_u64 ring[LKT_PF][4];
#pragma unroll
    for (int u = 0; u < LKT_PF; ++u) src.load4(u * LKT_TILE + threadIdx.x * 4, ring[u]);
    for (int t0 = 0; t0 < ntile; t0 += LKT_PF) {
#pragma unroll
        for (int u = 0; u < LKT_PF; ++u) {
            const int t = t0 + u;
            if (t < ntile) {
                ltk_u64 cur[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) cur[q] = ring[u][q];
                src.load4((t + LKT_PF) * LKT_TILE + threadIdx.x * 4, ring[u]);
                const ltk_u64 thr = s.thr;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const bool pass = cur[q] > thr;
                    const ltk_u32 mask = __ballot_sync(0xffffffffu, pass);
                    if (mask) {
                        int base = 0;
                        if (lane == 0) base = atomicAdd(&s.cnt, __popc(mask));
                        base = __shfl_sync(0xffffffffu, base, 0);
                        if (pass) s.cand[base + __popc(mask & ((1u << lane) - 1))] = cur[q];
                    }
                }
                __syncthreads();
                const int cnt = s.cnt;
                if (cnt >= LKT_FLUSH || (t == ntile - 1 && cnt > 0)) ltk_merge<P>(s, count);
                else __syncthreads();
            }
        }
    }
}

struct LtkStage1Src {
    const ltk_u16* row;
    int lo, hi;
    bool vec;
    __device__ void load4(int idx, ltk_u64 (&k)[4]) const {
        const int e = lo + idx;
        if (vec && e + 4 <= hi) {
            const uint2 u = *(const uint2*)(row + e);
            k[0] = ltk_make_key(u.x & 0xFFFF, e); k[1] = ltk_make_key(u.x >> 16, e + 1);
            k[2] = ltk_make_key(u.y & 0xFFFF, e + 2); k[3] = ltk_make_key(u.y >> 16, e + 3);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) k[i] = (e + i < hi) ? ltk_make_key(row[e + i], e + i) : 0;
        }
    }
};

struct LtkStage2Src {
    const ltk_u64* keys;
    int len;
    __device__ void load4(int idx, ltk_u64 (&k)[4]) const {
        if (idx + 4 <= len) {
            const ulonglong2 a = *(const ulonglong2*)(keys + idx), b = *(const ulonglong2*)(keys + idx + 2);
            k[0] = a.x; k[1] = a.y; k[2] = b.x; k[3] = b.y;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) k[i] = (idx + i < len) ? keys[idx + i] : 0;
        }
    }
};

template <int P>
__global__ void __launch_bounds__(LKT_NT) ling_topk_stage1_and_2_k(
    const ltk_u16* logits, ltk_u64* ws, int* row_cnt, int stride, int valid, int nchunks, int count,
    int* ids, float* vals) {
    __shared__ LtkShared<P> s;
    const int chunk = blockIdx.x, row = blockIdx.y;
    LtkStage1Src src;
    src.row = logits + (size_t)row * stride;
    src.lo = chunk * LKT_CHUNK;
    src.hi = min(valid, src.lo + LKT_CHUNK);
    src.vec = (stride % 4 == 0) && aligned16(logits);
    ltk_select<P>(s, src, src.hi - src.lo, count);
    if (threadIdx.x < P) ws[((size_t)row * nchunks + chunk) * P + threadIdx.x] = s.pool[threadIdx.x];
    __syncthreads();
    __threadfence();
    __shared__ int s_last;
    if (threadIdx.x == 0) {
        const int prev = atomicAdd(&row_cnt[row], 1);
        if (prev == gridDim.x - 1) {
            row_cnt[row] = 0;
            s_last = 1;
        } else {
            s_last = 0;
        }
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    LtkStage2Src src2;
    src2.keys = ws + (size_t)row * nchunks * P;
    src2.len = nchunks * P;
    ltk_select<P>(s, src2, src2.len, count);
    if (threadIdx.x < count) {
        const ltk_u64 key = s.pool[threadIdx.x];
        int id = -1;
        float v = -INFINITY;
        if (key != 0) {
            id = (int)~(ltk_u32)key;
            const ltk_u32 t = (ltk_u32)(key >> 32);
            const ltk_u32 u = (t & 0x8000) ? (t & 0x7FFF) : (~t & 0xFFFF);
            v = ltk_h2f((ltk_u16)u);
        }
        ids[(size_t)row * count + threadIdx.x] = id;
        vals[(size_t)row * count + threadIdx.x] = v;
    }
}

int ltk_topk_p(int count) { return count <= 16 ? 16 : count <= 32 ? 32 : 64; }
static size_t ltk_topk_nchunks(int valid) { return ((size_t)valid + LKT_CHUNK - 1) / LKT_CHUNK; }

template <int P>
int launch_ling_topk(const void* logits, int rows, int stride, int valid, int count,
                     int* ids, float* vals, ltk_u64* ws, int* row_cnt, void* stream) {
    const int nch = (int)ltk_topk_nchunks(valid);
    ling_topk_stage1_and_2_k<P><<<dim3(nch, rows), LKT_NT, 0, gb_stream(stream)>>>(
        (const ltk_u16*)logits, ws, row_cnt, stride, valid, nch, count, ids, vals);
    return GB_CHECK_LAUNCH();
}

}  // namespace lingproj

// The fused rot_out + add_rmsnorm epilogue kernel (`ling_kda_fused.cu`, compiled
// --fmad=false like glue.cu). Launched by gb_ling_linear_add_rmsnorm below. The
// three trailing suh/xh pairs rotate this epilogue's own rmsnorm output into the
// next sub-layer's rotated inputs (the fused input rotation).
__global__ void ling_rot_out_add_rmsnorm_k(const float* c, const half* svh, const uint16_t* x,
                                           const float* w, uint16_t* h_out, uint16_t* y,
                                           int n, int sk, int dim, float eps,
                                           const half* suh_p0, const half* suh_p1, const half* suh_p2,
                                           half* xh_p0, half* xh_p1, half* xh_p2);

extern "C" {

// Workspace bytes of gb_ling_kda_proj_fused: BOTH rotated inputs (xh0 for suh0, xh1 for suh1) plus, when the pair
// splits K (SK > 1), the two fp32 partial buffers (Z0 [sk, m, n0], Z1 [sk, m, n1]).
GB_EXPORT size_t gb_ling_kda_proj_fused_split_workspace_bytes(int m, int k, int n0, int n1) {
    using namespace lingproj;
    if (m < 1 || m > kMaxRows || !shape_ok_k2(k, n0, 2, 0) || !shape_ok_k2(k, n1, 2, 0)) return 0;
    const Plan p0 = plan_for(k, n0), p1 = plan_for(k, n1);
    const size_t xf = xf_bytes(m, k);
    size_t b = 2 * xf;
    if (p0.sk > 1) b += align256((size_t)p0.sk * m * n0 * sizeof(float));
    if (p1.sk > 1) b += align256((size_t)p1.sk * m * n1 * sizeof(float));
    return b;
}

// Workspace bytes of gb_ling_kda_proj_fused_triple: THREE rotated inputs (xh0/xh1/xh2; the triple runs SK == 1, so
// no split partials). This is also the largest any fused pair needs (the pair's xf pair is a subset).
GB_EXPORT size_t gb_ling_kda_proj_fused_workspace_bytes(int m, int k) {
    if (m < 1 || m > lingproj::kMaxRows || k <= 0 || k % 128) return 0;
    return 3 * lingproj::xf_bytes(m, k);
}

// y0 [m, n0] = f(x [m, k]), y1 [m, n1] = g(x [m, k]) in ONE launch sharing the input rotation. The projections must
// share k, k2, codebook and the decode plan (sk, wk); n0 != n1 is allowed when both are multiples of 128. When the
// plans differ the call falls back to the two separate `gb_exl3_linear_k2` launches (same bits, no fusion). Bit-
// identical per element to the two separate calls (same rotation, same k order, same warp-order sum, same finish/split
// add). All device pointers; workspace >= the companion workspace-bytes call.
GB_EXPORT int gb_ling_kda_proj_fused(const void* x, int x_dtype,
                                     const void* trellis0, const void* suh0, const void* svh0,
                                     void* y0, int y_dtype0, int n0,
                                     const void* trellis1, const void* suh1, const void* svh1,
                                     void* y1, int y_dtype1, int n1,
                                     int m, int k, int k2, int codebook,
                                     void* workspace, size_t workspace_bytes, void* stream,
                                     const void* xh_pre0, const void* xh_pre1,
                                     const void* down_suh, void* down_xh, int* col_cnt) {
    using namespace lingproj;
    if (m < 1 || m > kMaxRows || !shape_ok_k2(k, n0, k2, codebook) || !shape_ok_k2(k, n1, k2, codebook) ||
        !dtype_ok(x_dtype) || !dtype_ok(y_dtype0) || !dtype_ok(y_dtype1) || !x || !trellis0 || !suh0 ||
        !svh0 || !y0 || !trellis1 || !suh1 || !svh1 || !y1 || !workspace)
        return (int)cudaErrorInvalidValue;
    const Plan p0 = plan_for(k, n0), p1 = plan_for(k, n1);
    // Fused SwiGLU input rotation of a following single projection (the dense FFN down_proj): all-or-nothing, and
    // only on the SK == 1, equal-width, bf16-out, non-fragment pair profile (where the two CTAs of a 128-column
    // block write its whole gate and up strips).
    if ((down_suh == nullptr) != (down_xh == nullptr) || (down_suh == nullptr) != (col_cnt == nullptr))
        return (int)cudaErrorInvalidValue;
    if (down_suh) {
        if (p0.sk != 1 || p1.sk != 1 || n0 != n1 || y_dtype0 != BF16 || y_dtype1 != BF16 ||
            group_tiles(m) >= 2 || m > 64)
            return (int)cudaErrorInvalidValue;
    }
    // Pre-supplied rotated inputs (the previous fused epilogue already wrote them): skip the input rotation
    // launch. Only the non-fragment layout (group_tiles(m) < 2) is covered.
    if ((xh_pre0 == nullptr) != (xh_pre1 == nullptr)) return (int)cudaErrorInvalidValue;
    if (xh_pre0 && group_tiles(m) >= 2) return (int)cudaErrorInvalidValue;
    const size_t xf = xf_bytes(m, k);
    const size_t z0 = p0.sk > 1 ? align256((size_t)p0.sk * m * n0 * sizeof(float)) : 0;
    const size_t z1 = p1.sk > 1 ? align256((size_t)p1.sk * m * n1 * sizeof(float)) : 0;
    if (workspace_bytes < 2 * xf + z0 + z1) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh0 = xh_pre0 ? reinterpret_cast<half*>(const_cast<void*>(xh_pre0)) : reinterpret_cast<half*>(ws);
    half* xh1 = xh_pre1 ? reinterpret_cast<half*>(const_cast<void*>(xh_pre1)) : reinterpret_cast<half*>(ws + xf);
    float* Z0 = z0 ? reinterpret_cast<float*>(ws + 2 * xf) : nullptr;
    float* Z1 = z1 ? reinterpret_cast<float*>(ws + 2 * xf + z0) : nullptr;
    const bool pdl = pdl_exl3();
    const int tw = 4 * k2;                                 // words a tile: 4 * K2
    PairArgs a{xh0, xh1,
               static_cast<const uint32_t*>(trellis0), (long long)(n0 / 16) * tw, 8LL * tw, (const half*)svh0,
               y0, y_dtype0, n0,
               static_cast<const uint32_t*>(trellis1), (long long)(n1 / 16) * tw, 8LL * tw, (const half*)svh1,
               y1, y_dtype1, n1,
               nullptr, nullptr, m, k, 1, 0, 0, pdl,
               (const half*)down_suh, (half*)down_xh, col_cnt,
               nullptr, nullptr, nullptr, nullptr, 0.0f};
    if (p0.sk == p1.sk && p0.wk == p1.wk) {
        if (!xh_pre0) {
            int rc = group_tiles(m) >= 2 ? rotate_in_frag_pair(x, x_dtype, suh0, xh0, suh1, xh1, m, k, st, pdl)
                                         : rotate_in_pair(x, x_dtype, suh0, xh0, suh1, xh1, m, k, st, pdl);
            if (rc) return rc;
        }
        a.SK = p0.sk;
        a.wk = p0.wk;
        a.Z0 = p0.sk > 1 ? Z0 : nullptr;
        a.Z1 = p1.sk > 1 ? Z1 : nullptr;
        int rc = dispatch_pair(k2, codebook, a, st);
        if (rc) return rc;
        if (p0.sk > 1) {
            rc = gb_exl3_rotate_out_sk(Z0, svh0, nullptr, y0, y_dtype0, m, n0, p0.sk, st);
            if (rc) return rc;
            return gb_exl3_rotate_out_sk(Z1, svh1, nullptr, y1, y_dtype1, m, n1, p1.sk, st);
        }
        return 0;
    }
    // plans differ: run the two separate launches from the same workspace (each gets its own xf and Z slices)
    int rc = gb_exl3_linear_k2(x, x_dtype, trellis0, suh0, svh0, nullptr, y0, y_dtype0, m, k, n0, k2 / 2,
                               codebook, ws, xf + z0, st);
    if (rc) return rc;
    return gb_exl3_linear_k2(x, x_dtype, trellis1, suh1, svh1, nullptr, y1, y_dtype1, m, k, n1, k2 / 2,
                             codebook, ws + xf, xf + z1, st);
}

// y0 [m, n0] = qkv(x), y1 [m, n1] = f(x), y2 [m, n2] = g(x) in ONE launch sharing the input rotation. The
// projections must share k and codebook and all run SK == 1 (the KDA qkv/f/g plans are), with projection 0 at
// K2_0 and projections 1/2 at K2_12; anything else returns cudaErrorInvalidValue. Bit-identical per element to the
// three separate `gb_exl3_linear_k2` calls.
GB_EXPORT int gb_ling_kda_proj_fused_triple(const void* x, int x_dtype,
                                            const void* trellis0, const void* suh0, const void* svh0,
                                            void* y0, int y_dtype0, int n0,
                                            const void* trellis1, const void* suh1, const void* svh1,
                                            void* y1, int y_dtype1, int n1,
                                            const void* trellis2, const void* suh2, const void* svh2,
                                            void* y2, int y_dtype2, int n2,
                                            int m, int k, int k2_0, int k2_12, int codebook,
                                            void* workspace, size_t workspace_bytes, void* stream,
                                            const void* xh_pre0, const void* xh_pre1, const void* xh_pre2) {
    using namespace lingproj;
    if (m < 1 || m > kMaxRows || !shape_ok_k2(k, n0, k2_0, codebook) || !shape_ok_k2(k, n1, k2_12, codebook) ||
        !shape_ok_k2(k, n2, k2_12, codebook) ||
        !dtype_ok(x_dtype) || !dtype_ok(y_dtype0) || !dtype_ok(y_dtype1) || !dtype_ok(y_dtype2) || !x ||
        !trellis0 || !suh0 || !svh0 || !y0 || !trellis1 || !suh1 || !svh1 || !y1 || !trellis2 || !suh2 ||
        !svh2 || !y2 || !workspace)
        return (int)cudaErrorInvalidValue;
    const Plan p0 = plan_for(k, n0), p1 = plan_for(k, n1), p2 = plan_for(k, n2);
    if (p0.sk != 1 || p1.sk != 1 || p2.sk != 1 || p0.wk != p1.wk || p1.wk != p2.wk)
        return (int)cudaErrorInvalidValue;   // the triple covers the SK == 1 decode plans only
    // Pre-supplied rotated inputs (the previous fused epilogue already wrote them): skip the input rotation
    // launch. Only the non-fragment layout (group_tiles(m) < 2) is covered.
    if ((xh_pre0 == nullptr) != (xh_pre1 == nullptr) || (xh_pre1 == nullptr) != (xh_pre2 == nullptr))
        return (int)cudaErrorInvalidValue;
    if (xh_pre0 && group_tiles(m) >= 2) return (int)cudaErrorInvalidValue;
    const size_t xf = xf_bytes(m, k);
    if (workspace_bytes < 3 * xf) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh0 = xh_pre0 ? reinterpret_cast<half*>(const_cast<void*>(xh_pre0)) : reinterpret_cast<half*>(ws);
    half* xh1 = xh_pre1 ? reinterpret_cast<half*>(const_cast<void*>(xh_pre1)) : reinterpret_cast<half*>(ws + xf);
    half* xh2 = xh_pre2 ? reinterpret_cast<half*>(const_cast<void*>(xh_pre2)) : reinterpret_cast<half*>(ws + 2 * xf);
    const bool pdl = pdl_exl3();
    const int tw0 = 4 * k2_0, tw12 = 4 * k2_12;
    if (!xh_pre0) {
        int rc = group_tiles(m) >= 2 ? rotate_in_frag_triple(x, x_dtype, suh0, xh0, suh1, xh1, suh2, xh2, m, k, st, pdl)
                                     : rotate_in_triple(x, x_dtype, suh0, xh0, suh1, xh1, suh2, xh2, m, k, st, pdl);
        if (rc) return rc;
    }
    TripleArgs a{xh0, xh1, xh2,
                 static_cast<const uint32_t*>(trellis0), (long long)(n0 / 16) * tw0, 8LL * tw0, (const half*)svh0,
                 y0, y_dtype0, n0,
                 static_cast<const uint32_t*>(trellis1), (long long)(n1 / 16) * tw12, 8LL * tw12, (const half*)svh1,
                 y1, y_dtype1, n1,
                 static_cast<const uint32_t*>(trellis2), (long long)(n2 / 16) * tw12, 8LL * tw12, (const half*)svh2,
                 y2, y_dtype2, n2,
                 nullptr, m, k, 1, p0.wk, 0, pdl};
    return dispatch_triple(k2_0, k2_12, codebook, a, st);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// gb_ling_linear_add_rmsnorm: `y [m, dim] = rmsnorm(h)`, `h = bf16(hidden + res)`,
// with `res [m, n] = layer(x [m, k])` (EXL3), in one launch per stage instead of the
// `gb_exl3_linear_k2` + `gb_add_rmsnorm` pair: the linear runs with the layer's own
// decode plan into its fp32 split partials (ZM, the same warp-order sums the separate
// launches finish from), and `ling_rot_out_add_rmsnorm_k` (ling_kda_fused.cu) does
// the rot_out finish into shared memory and then glue.cu's exact rmsnorm_k<true>
// math. Bit-identical per element to the separate launches (the KDA o_proj, MLA
// dense and dense FFN down_proj sites of bailing_moe_v3: n = dim = 4096, sk = 8).
// `hidden_out` may alias `hidden` (the residual stream). Ling-owned.
extern "C" {

GB_EXPORT int gb_ling_linear_add_rmsnorm(const void* x, int x_dtype, const void* trellis, const void* suh,
                                         const void* svh, const void* hidden, const float* norm_w,
                                         void* hidden_out, void* y, int m, int k, int n, int k2, int codebook,
                                         int dim, float eps, void* workspace, size_t workspace_bytes, void* stream,
                                         int skip_rot_in, const void* suh_p0, const void* suh_p1,
                                         const void* suh_p2, void* xh_p0, void* xh_p1, void* xh_p2,
                                         int* lin_cnt) {
    using namespace lingproj;
    if (m < 1 || m > kMaxRows || !shape_ok_k2(k, n, k2, codebook) || !dtype_ok(x_dtype) ||
        dim <= 0 || dim % 8 != 0 || n > 4096 || n != dim || eps <= 0.0f || !x || !trellis || !suh || !svh ||
        !hidden || !norm_w || !hidden_out || !y || !workspace)
        return (int)cudaErrorInvalidValue;
    // skip_rot_in: the input rotation already ran (the KDA window+norm epilogue wrote xh). The prerot argument
    // triple rotates this epilogue's own rmsnorm output into the next sub-layer's rotated inputs.
    if (skip_rot_in && group_tiles(m) >= 2) return (int)cudaErrorInvalidValue;
    if ((suh_p0 == nullptr) != (xh_p0 == nullptr) || (suh_p1 == nullptr) != (xh_p1 == nullptr) ||
        (suh_p2 == nullptr) != (xh_p2 == nullptr))
        return (int)cudaErrorInvalidValue;
    const Plan p = plan_for(k, n);
    const size_t xf = xf_bytes(m, k);
    const size_t z = align256((size_t)p.sk * m * n * sizeof(float));
    if (workspace_bytes < xf + z) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh = reinterpret_cast<half*>(ws);
    float* Z = reinterpret_cast<float*>(ws + xf);
    const bool pdl = pdl_exl3();
    if (!skip_rot_in) {
        int rc = group_tiles(m) >= 2 ? rotate_in_frag(x, x_dtype, suh, xh, m, k, st, pdl)
                                     : rotate_in(x, x_dtype, suh, xh, m, k, st, pdl);
        if (rc) return rc;
    }
    const int tw = 4 * k2;
    SingleArgs a{xh, static_cast<const uint32_t*>(trellis), (long long)(n / 16) * tw, 8LL * tw,
                 (const half*)svh, y, 1, n, (uint16_t*)hidden_out, Z, m, k, p.sk, p.wk, 1, pdl,
                 (const uint16_t*)hidden, norm_w, dim, eps,
                 (const half*)suh_p0, (const half*)suh_p1, (const half*)suh_p2,
                 (half*)xh_p0, (half*)xh_p1, (half*)xh_p2, lin_cnt};
    int rc = dispatch_single(k2, codebook, a, st);
    if (rc) return rc;
    // With a counter buffer the linear kernel's last CTA ran the fused epilogue: no separate launch needed.
    if (!lin_cnt) {
        ling_rot_out_add_rmsnorm_k<<<m, 256, 0, st>>>(Z, (const half*)svh, (const uint16_t*)hidden, norm_w,
                                                       (uint16_t*)hidden_out, (uint16_t*)y, n, p.sk, dim, eps,
                                                       (const half*)suh_p0, (const half*)suh_p1, (const half*)suh_p2,
                                                       (half*)xh_p0, (half*)xh_p1, (half*)xh_p2);
    }
    return GB_CHECK_LAUNCH();
}

// `gb_ling_linear_add_rmsnorm` with the SwiGLU activation fused into the input rotation: reads g/u (bf16
// [m, k]) and computes the exact `ew_vec_k<OpSwiglu>` element (silu_f(g) * u rounded to bf16; glue.cu's
// --fmad=false op order reproduced with explicit RN intrinsics) in place of `x`, then the layer's own decode
// plan on the rotated input and the rot_out + add_rmsnorm epilogue. Bit-identical per element to `glue::swiglu`
// + `gb_ling_linear_add_rmsnorm`. The dense FFN down_proj site of bailing_moe_v3 (k2 = 7, codebook 2,
// k = 10240, n = dim = 4096, rows <= 8). m <= 64 (the non-fragment rotation layout).
GB_EXPORT int gb_ling_silu_mul_linear_add_rmsnorm(const void* g, const void* u, const void* trellis, const void* suh,
                                                 const void* svh, const void* hidden, const float* norm_w,
                                                 void* hidden_out, void* y, int m, int k, int n, int k2, int codebook,
                                                 int dim, float eps, void* workspace, size_t workspace_bytes, void* stream,
                                                 const void* suh_p0, const void* suh_p1, const void* suh_p2,
                                                 void* xh_p0, void* xh_p1, void* xh_p2, int skip_rot_in,
                                                 int* lin_cnt) {
    using namespace lingproj;
    if (m < 1 || m > 64 || !shape_ok_k2(k, n, k2, codebook) ||
        dim <= 0 || dim % 8 != 0 || n > 4096 || n != dim || eps <= 0.0f || !g || !u || !trellis || !suh || !svh ||
        !hidden || !norm_w || !hidden_out || !y || !workspace)
        return (int)cudaErrorInvalidValue;
    if (skip_rot_in && group_tiles(m) >= 2) return (int)cudaErrorInvalidValue;
    if ((suh_p0 == nullptr) != (xh_p0 == nullptr) || (suh_p1 == nullptr) != (xh_p1 == nullptr) ||
        (suh_p2 == nullptr) != (xh_p2 == nullptr))
        return (int)cudaErrorInvalidValue;
    const Plan p = plan_for(k, n);
    const size_t xf = xf_bytes(m, k);
    const size_t z = align256((size_t)p.sk * m * n * sizeof(float));
    if (workspace_bytes < xf + z) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh = reinterpret_cast<half*>(ws);
    float* Z = reinterpret_cast<float*>(ws + xf);
    const bool pdl = pdl_exl3();
    const int gy_single = (k / 128 + 3) / 4;
    int rc = 0;
    if (!skip_rot_in) {
        const dim3 grid((unsigned)m, (unsigned)gy_single);
        rc = launch_k(pdl, ling_silu_mul_rot_in_kernel, grid, dim3(128), 0, st,
                      (const __nv_bfloat16*)g, (const __nv_bfloat16*)u, (const half*)suh, xh, k);
        if (rc) return rc;
    }
    const int tw = 4 * k2;
    SingleArgs a{xh, static_cast<const uint32_t*>(trellis), (long long)(n / 16) * tw, 8LL * tw,
                 (const half*)svh, y, 1, n, (uint16_t*)hidden_out, Z, m, k, p.sk, p.wk, 1, pdl,
                 (const uint16_t*)hidden, norm_w, dim, eps,
                 (const half*)suh_p0, (const half*)suh_p1, (const half*)suh_p2,
                 (half*)xh_p0, (half*)xh_p1, (half*)xh_p2, lin_cnt};
    rc = dispatch_single(k2, codebook, a, st);
    if (rc) return rc;
    // With a counter buffer the linear kernel's last CTA ran the fused epilogue: no separate launch needed.
    if (!lin_cnt) {
        ling_rot_out_add_rmsnorm_k<<<m, 256, 0, st>>>(Z, (const half*)svh, (const uint16_t*)hidden, norm_w,
                                                       (uint16_t*)hidden_out, (uint16_t*)y, n, p.sk, dim, eps,
                                                       (const half*)suh_p0, (const half*)suh_p1, (const half*)suh_p2,
                                                       (half*)xh_p0, (half*)xh_p1, (half*)xh_p2);
    }
    return GB_CHECK_LAUNCH();
}

// Norm + triple rotation fused: one launch computes rmsnorm (or add_rmsnorm) of x exactly like glue.cu's
// rmsnorm_k<false>/<true> (writing h_out/y where applicable) AND rotates the normalised row into
// xh0/xh1/xh2 with the exact ling_rot_in_triple_kernel math, then runs the triple's linear kernels on the
// shared rotated inputs (dispatch_triple). Bit-identical per element to `glue::rmsnorm`/`glue::add_rmsnorm`
// + `gb_ling_kda_proj_fused_triple`. The layer-0 / post-MoE norm sites of bailing_moe_v3
// (k = dim = 4096, rows <= 8, SK == 1 plans, m <= 64).
GB_EXPORT int gb_ling_norm_rot_in_triple(const void* x, const void* res, const float* w, void* h_out, void* y,
                                         const void* suh0, const void* suh1, const void* suh2,
                                         const void* trellis0, const void* svh0, void* y0, int y_dtype0, int n0,
                                         const void* trellis1, const void* svh1, void* y1, int y_dtype1, int n1,
                                         const void* trellis2, const void* svh2, void* y2, int y_dtype2, int n2,
                                         int m, int k, int k2_0, int k2_12, int codebook, int dim, float eps,
                                         void* workspace, size_t workspace_bytes, void* stream,
                                         const int32_t* token_ids, const void* embed_table, int vocab) {
    using namespace lingproj;
    if (m < 1 || m > 64 || dim <= 0 || dim > 4096 || dim % 8 != 0 || k <= 0 || k > 4096 || k % 128 != 0 ||
        k != dim || eps <= 0.0f || !x || !w || !y ||
        !shape_ok_k2(k, n0, k2_0, codebook) || !shape_ok_k2(k, n1, k2_12, codebook) || !shape_ok_k2(k, n2, k2_12, codebook) ||
        !dtype_ok(y_dtype0) || !dtype_ok(y_dtype1) || !dtype_ok(y_dtype2) ||
        !suh0 || !suh1 || !suh2 || !trellis0 || !svh0 || !y0 || !trellis1 || !svh1 || !y1 ||
        !trellis2 || !svh2 || !y2 || !workspace)
        return (int)cudaErrorInvalidValue;
    // Embedding fusion (layer 0, decode): replace the `x` read with the embed table row (written to `h_out`,
    // which must be non-null then). All-or-nothing.
    if ((token_ids == nullptr) != (embed_table == nullptr)) return (int)cudaErrorInvalidValue;
    if (token_ids && (h_out == nullptr || vocab <= 0)) return (int)cudaErrorInvalidValue;
    const Plan p0 = plan_for(k, n0), p1 = plan_for(k, n1), p2 = plan_for(k, n2);
    if (p0.sk != 1 || p1.sk != 1 || p2.sk != 1 || p0.wk != p1.wk || p1.wk != p2.wk)
        return (int)cudaErrorInvalidValue;
    const size_t xf = xf_bytes(m, k);
    if (workspace_bytes < 3 * xf) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh0 = reinterpret_cast<half*>(ws);
    half* xh1 = reinterpret_cast<half*>(ws + xf);
    half* xh2 = reinterpret_cast<half*>(ws + 2 * xf);
    const bool pdl = pdl_exl3();
    int rc = launch_k(pdl, ling_norm_rot_in_triple_kernel, dim3((unsigned)m, 1), dim3(256), 0, st,
                      (const __nv_bfloat16*)x, (const __nv_bfloat16*)res, w, (__nv_bfloat16*)h_out,
                      (__nv_bfloat16*)y, (const half*)suh0, xh0, (const half*)suh1, xh1,
                      (const half*)suh2, xh2, dim, k, eps,
                      token_ids, embed_table, vocab);
    if (rc) return rc;
    const int tw0 = 4 * k2_0, tw12 = 4 * k2_12;
    TripleArgs a{xh0, xh1, xh2,
                 static_cast<const uint32_t*>(trellis0), (long long)(n0 / 16) * tw0, 8LL * tw0, (const half*)svh0,
                 y0, y_dtype0, n0,
                 static_cast<const uint32_t*>(trellis1), (long long)(n1 / 16) * tw12, 8LL * tw12, (const half*)svh1,
                 y1, y_dtype1, n1,
                 static_cast<const uint32_t*>(trellis2), (long long)(n2 / 16) * tw12, 8LL * tw12, (const half*)svh2,
                 y2, y_dtype2, n2,
                 nullptr, m, k, 1, p0.wk, 0, pdl};
    return dispatch_triple(k2_0, k2_12, codebook, a, st);
}

// The pair analogue of gb_ling_norm_rot_in_triple (q_a + kv_a / dense gate+up sitepairs): the fused norm launch,
// then the pair's linear kernels (dispatch_pair; the two projections must share the decode plan). Bit-identical
// per element to `glue::rmsnorm`/`glue::add_rmsnorm` + `gb_ling_kda_proj_fused`.
GB_EXPORT int gb_ling_norm_rot_in_pair(const void* x, const void* res, const float* w, void* h_out, void* y,
                                       const void* suh0, const void* suh1,
                                       const void* trellis0, const void* svh0, void* y0, int y_dtype0, int n0,
                                       const void* trellis1, const void* svh1, void* y1, int y_dtype1, int n1,
                                       int m, int k, int k2, int codebook, int dim, float eps,
                                       void* workspace, size_t workspace_bytes, void* stream) {
    using namespace lingproj;
    if (m < 1 || m > 64 || dim <= 0 || dim > 4096 || dim % 8 != 0 || k <= 0 || k > 4096 || k % 128 != 0 ||
        k != dim || eps <= 0.0f || !x || !w || !y ||
        !shape_ok_k2(k, n0, k2, codebook) || !shape_ok_k2(k, n1, k2, codebook) ||
        !dtype_ok(y_dtype0) || !dtype_ok(y_dtype1) ||
        !suh0 || !suh1 || !trellis0 || !svh0 || !y0 || !trellis1 || !svh1 || !y1 || !workspace)
        return (int)cudaErrorInvalidValue;
    const Plan p0 = plan_for(k, n0), p1 = plan_for(k, n1);
    if (p0.sk != p1.sk || p0.wk != p1.wk) return (int)cudaErrorInvalidValue;
    const size_t xf = xf_bytes(m, k);
    const size_t z0 = p0.sk > 1 ? align256((size_t)p0.sk * m * n0 * sizeof(float)) : 0;
    const size_t z1 = p1.sk > 1 ? align256((size_t)p1.sk * m * n1 * sizeof(float)) : 0;
    if (workspace_bytes < 2 * xf + z0 + z1) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh0 = reinterpret_cast<half*>(ws);
    half* xh1 = reinterpret_cast<half*>(ws + xf);
    float* Z0 = z0 ? reinterpret_cast<float*>(ws + 2 * xf) : nullptr;
    float* Z1 = z1 ? reinterpret_cast<float*>(ws + 2 * xf + z0) : nullptr;
    const bool pdl = pdl_exl3();
    int rc = launch_k(pdl, ling_norm_rot_in_pair_kernel, dim3((unsigned)m, 1), dim3(256), 0, st,
                      (const __nv_bfloat16*)x, (const __nv_bfloat16*)res, w, (__nv_bfloat16*)h_out,
                      (__nv_bfloat16*)y, (const half*)suh0, xh0, (const half*)suh1, xh1, dim, k, eps);
    if (rc) return rc;
    const int tw = 4 * k2;
    PairArgs a{xh0, xh1,
               static_cast<const uint32_t*>(trellis0), (long long)(n0 / 16) * tw, 8LL * tw, (const half*)svh0,
               y0, y_dtype0, n0,
               static_cast<const uint32_t*>(trellis1), (long long)(n1 / 16) * tw, 8LL * tw, (const half*)svh1,
               y1, y_dtype1, n1,
               Z0, Z1, m, k, p0.sk, p0.wk, 0, pdl,
               nullptr, nullptr, nullptr};
    rc = dispatch_pair(k2, codebook, a, st);
    if (rc) return rc;
    if (p0.sk > 1) {
        rc = gb_exl3_rotate_out_sk(Z0, svh0, nullptr, y0, y_dtype0, m, n0, p0.sk, st);
        if (rc) return rc;
        rc = gb_exl3_rotate_out_sk(Z1, svh1, nullptr, y1, y_dtype1, m, n1, p1.sk, st);
        if (rc) return rc;
    }
    return 0;
}

// y [m, n] = q_b(x [m, k]) with the in-place rmsnorm of x folded into the input rotation: one launch for the
// norm (ling_rmsnorm_rot_in_kernel, glue.cu's exact rmsnorm_k<false> math, bf16 in place) + the rotation into
// xh, then the layer's own decode plan (the linear kernel finishing inline, SK == 1) => two launches instead of
// glue::rmsnorm + gb_exl3_linear_k2's three. Bit-identical per element to the separate launches (same norm
// bits, same xh bytes, same k order, same warp-order sum, same finish). The MLA q_b site of bailing_moe_v3
// (k2 = 7, codebook 2, k = 1536, n = 12288, rows <= 8). Workspace: the rotated input only (xf_bytes).
GB_EXPORT int gb_ling_rmsnorm_linear_bf16(
    const void* x, const float* norm_w,
    const void* trellis, const void* suh, const void* svh,
    void* y,
    int m, int k, int n, int k2, int codebook, float eps,
    void* workspace, size_t workspace_bytes, void* stream) {
    using namespace lingproj;
    if (m < 1 || m > 8 || k <= 0 || k % 128 != 0 || k > 1536 || eps <= 0.0f ||
        !shape_ok_k2(k, n, k2, codebook) || !x || !norm_w || !trellis || !suh || !svh || !y || !workspace)
        return (int)cudaErrorInvalidValue;
    // dispatch_single instantiates only these pairs; check before any launch so an uncovered shape never
    // partially runs (the caller falls back to the separate launches).
    if (!((k2 == 10 && codebook == 2) || (k2 == 7 && codebook == 2)))
        return (int)cudaErrorInvalidValue;
    const Plan p = plan_for(k, n);
    if (p.sk != 1) return (int)cudaErrorInvalidValue;   // the fusion covers the SK == 1 decode plans only
    const size_t xf = xf_bytes(m, k);
    if (workspace_bytes < xf) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    half* xh = reinterpret_cast<half*>(workspace);
    const bool pdl = pdl_exl3();
    int rc = launch_k(pdl, ling_rmsnorm_rot_in_kernel, dim3((unsigned)m, 1), dim3(256), 0, st,
                      (const __nv_bfloat16*)x, norm_w, (__nv_bfloat16*)x, (const half*)suh, xh, k, k, eps);
    if (rc) return rc;
    const int tw = 4 * k2;
    SingleArgs a{xh, static_cast<const uint32_t*>(trellis), (long long)(n / 16) * tw, 8LL * tw,
                 (const half*)svh, y, 1, n, nullptr, nullptr, m, k, 1, p.wk, 0, pdl,
                 nullptr, nullptr, 0, 0.0f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    return dispatch_single(k2, codebook, a, st);
}

// The MLA q_a/kv_a pair and the q_b step together (the `MixerDevice::Mla` site of bailing_moe_v3, decode):
// y0 [m, n0] = q_a(x), y1 [m, n1] = kv_a(x) run as the fused pair (same launches as
// `gb_ling_kda_proj_fused`: the rotate_in_pair / norm+rot_in_pair, then the pair linear into the split
// partials), and the two `gb_exl3_rotate_out_sk` finishes plus the q_b input norm/rotation of
// `gb_ling_rmsnorm_linear_bf16` collapse into ONE launch (`ling_mla_rot_out_qnorm_qb_rot_in_kernel`) that
// reduces the partials (Z1 -> y1, Z0 -> shared), normalises Z0's finish with qa_norm_w into y0 (and shared)
// and rotates it with qb_suh into qb_xh; the q_b linear (its own decode plan, SK == 1) then reads qb_xh.
// Bit-identical per element to `gb_ling_kda_proj_fused` + `gb_ling_rmsnorm_linear_bf16` (or
// `gb_ling_norm_rot_in_pair` + ... when norm_w runs the norm inside). q_a/kv_a: k2 = 10, codebook 2,
// n0 = 1536, n1 = 640 (split-K plans must match); q_b: k = n0, k2 = 7 (or 10), n = 12288, rows <= 8.
// Workspace: the pair's (`pair_workspace_bytes`); qb_xh is the caller's buffer (>= m * n0 * 2 bytes), separate
// from ws (the fused epilogue writes it while the pair's partials are still being read from ws). Returns an
// error for shapes the fusion does not cover: the caller falls back to the separate sequence.
GB_EXPORT int gb_ling_mla_qkv_qb_fused(
    const void* x, int x_dtype,
    const void* res, const float* norm_w, void* h_out, void* y_norm,
    const void* trellis0, const void* suh0, const void* svh0,
    void* y0, int y_dtype0, int n0,
    const void* trellis1, const void* suh1, const void* svh1,
    void* y1, int y_dtype1, int n1,
    const float* qa_norm_w,
    const void* qb_trellis, const void* qb_suh, const void* qb_svh, void* y2,
    int qb_k2, int n_qb,
    int m, int k, int k2, int codebook, float eps,
    void* workspace, size_t workspace_bytes,
    const void* xh_pre0, const void* xh_pre1,
    void* qb_xh, size_t qb_xh_bytes,
    void* stream,
    const int* pair_cnt) {
    using namespace lingproj;
    if (m < 1 || m > 8 || k <= 0 || k % 128 != 0 || n0 <= 0 || n0 % 128 != 0 || n0 > 1536 ||
        n1 <= 0 || n1 % 128 != 0 || n1 > 1024 || n_qb <= 0 || n_qb % 128 != 0 || eps <= 0.0f ||
        !shape_ok_k2(k, n0, k2, codebook) || !shape_ok_k2(k, n1, k2, codebook) ||
        !shape_ok_k2(n0, n_qb, qb_k2, codebook) ||
        !dtype_ok(x_dtype) || !dtype_ok(y_dtype0) || !dtype_ok(y_dtype1) ||
        !trellis0 || !suh0 || !svh0 || !y0 || !trellis1 || !suh1 || !svh1 || !y1 ||
        !qa_norm_w || !qb_trellis || !qb_suh || !qb_svh || !y2 || !workspace ||
        !qb_xh || qb_xh_bytes < (size_t)m * n0 * 2)
        return (int)cudaErrorInvalidValue;
    if (y_dtype0 != BF16 || y_dtype1 != BF16) return (int)cudaErrorInvalidValue;
    if (!((k2 == 10 && codebook == 2) || (k2 == 8 && codebook == 2) || (k2 == 7 && codebook == 2) ||
          (k2 == 6 && codebook == 2) || (k2 == 5 && codebook == 2)))
        return (int)cudaErrorInvalidValue;
    if (!((qb_k2 == 10 && codebook == 2) || (qb_k2 == 7 && codebook == 2)))
        return (int)cudaErrorInvalidValue;
    // The fused MLA epilogue's per-column-block counters are pair_cnt[0..32) + pair_cnt[32] (33 ints); require the
    // grid's column blocks to fit.
    if (pair_cnt && ((n0 + n1) / 128 > 32)) return (int)cudaErrorInvalidValue;
    const Plan p0 = plan_for(k, n0), p1 = plan_for(k, n1), pq = plan_for(n0, n_qb);
    if (p0.sk != p1.sk || p0.wk != p1.wk || p0.sk <= 1 || pq.sk != 1) return (int)cudaErrorInvalidValue;
    if ((xh_pre0 == nullptr) != (xh_pre1 == nullptr)) return (int)cudaErrorInvalidValue;
    if (xh_pre0 && group_tiles(m) >= 2) return (int)cudaErrorInvalidValue;
    if (norm_w && (xh_pre0 || !x || !y_norm)) return (int)cudaErrorInvalidValue;
    if (!norm_w && !xh_pre0 && !x) return (int)cudaErrorInvalidValue;
    const size_t xf = xf_bytes(m, k);
    const size_t z0 = align256((size_t)p0.sk * m * n0 * sizeof(float));
    const size_t z1 = align256((size_t)p1.sk * m * n1 * sizeof(float));
    if (workspace_bytes < 2 * xf + z0 + z1) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    char* ws = static_cast<char*>(workspace);
    half* xh0 = xh_pre0 ? reinterpret_cast<half*>(const_cast<void*>(xh_pre0)) : reinterpret_cast<half*>(ws);
    half* xh1 = xh_pre1 ? reinterpret_cast<half*>(const_cast<void*>(xh_pre1)) : reinterpret_cast<half*>(ws + xf);
    float* Z0 = reinterpret_cast<float*>(ws + 2 * xf);
    float* Z1 = reinterpret_cast<float*>(ws + 2 * xf + z0);
    const bool pdl = pdl_exl3();
    if (norm_w) {
        int rc = launch_k(pdl, ling_norm_rot_in_pair_kernel, dim3((unsigned)m, 1), dim3(256), 0, st,
                          (const __nv_bfloat16*)x, (const __nv_bfloat16*)res, norm_w, (__nv_bfloat16*)h_out,
                          (__nv_bfloat16*)y_norm, (const half*)suh0, xh0, (const half*)suh1, xh1, k, k, eps);
        if (rc) return rc;
    } else if (!xh_pre0) {
        int rc = group_tiles(m) >= 2 ? rotate_in_frag_pair(x, x_dtype, suh0, xh0, suh1, xh1, m, k, st, pdl)
                                     : rotate_in_pair(x, x_dtype, suh0, xh0, suh1, xh1, m, k, st, pdl);
        if (rc) return rc;
    }
    const int tw = 4 * k2;
    // The fused MLA epilogue runs inside the pair's last CTA when the plan's block is the native 128 threads
    // (WK = 4) and the caller supplied the self-resetting grid counter; otherwise the separate epilogue launch runs
    // (same bits, one launch more).
    const bool mla_epi = pair_cnt != nullptr && p0.wk == 4 && group_tiles(m) < 2;
    PairArgs a{xh0, xh1,
               static_cast<const uint32_t*>(trellis0), (long long)(n0 / 16) * tw, 8LL * tw, (const half*)svh0,
               y0, y_dtype0, n0,
               static_cast<const uint32_t*>(trellis1), (long long)(n1 / 16) * tw, 8LL * tw, (const half*)svh1,
               y1, y_dtype1, n1,
               Z0, Z1, m, k, p0.sk, p0.wk, 0, pdl,
               nullptr, nullptr, nullptr,
               mla_epi ? qa_norm_w : nullptr,
               mla_epi ? (const half*)qb_suh : nullptr,
               mla_epi ? (half*)qb_xh : nullptr,
               mla_epi ? const_cast<int*>(pair_cnt) : nullptr, eps};
    int rc = dispatch_pair(k2, codebook, a, st);
    if (rc) return rc;
    if (!mla_epi) {
        rc = launch_k(pdl, ling_mla_rot_out_qnorm_qb_rot_in_kernel, dim3((unsigned)m, 1), dim3(256), 0, st,
                      Z0, Z1, (const half*)svh0, (const half*)svh1,
                      (__nv_bfloat16*)y0, (__nv_bfloat16*)y1, qa_norm_w, (const half*)qb_suh,
                      (half*)qb_xh, m, n0, n1, p0.sk, eps);
        if (rc) return rc;
    }
    const int twq = 4 * qb_k2;
    SingleArgs b{(half*)qb_xh, static_cast<const uint32_t*>(qb_trellis), (long long)(n_qb / 16) * twq, 8LL * twq,
                 (const half*)qb_svh, y2, 1, n_qb, nullptr, nullptr, m, n0, 1, pq.wk, 0, pdl,
                 nullptr, nullptr, 0, 0.0f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    return dispatch_single(qb_k2, codebook, b, st);
}

// y [rows, n] fp32 = x [rows, k] bf16 @ w [n, k]^T bf16 for the Ling-3.1 MoE router
// (n = 512, k = 4096, rows any): one warp per output column, 16 warps per CTA, 32
// CTAs (single wave on GB10), rows processed in chunks of 8 with the weight row loaded
// once per column per chunk. Bit-identical per (row, column) to
// `gb_mlag_bf16_linear_f32_kernel` (same k order, same lane stride, same FMA chain,
// same warp reduction, same fp32 store).
GB_EXPORT int gb_ling_bf16_linear_f32(const void* x, const void* w, void* y, int rows, int k, int n,
                                       void* stream) {
    using namespace lingproj;
    if (rows < 0 || n < 1 || k <= 0 || k % 256 != 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t st = gb_stream(stream);
    const dim3 grid((unsigned)((n + 15) / 16), 1);
    if (rows <= 8) {
        switch (rows) {
            case 1: ling_bf16_linear_f32_kernel<1><<<grid, 512, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (float*)y, rows, k, n); break;
            case 2: ling_bf16_linear_f32_kernel<2><<<grid, 512, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (float*)y, rows, k, n); break;
            case 3: ling_bf16_linear_f32_kernel<3><<<grid, 512, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (float*)y, rows, k, n); break;
            case 4: ling_bf16_linear_f32_kernel<4><<<grid, 512, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (float*)y, rows, k, n); break;
            case 5: ling_bf16_linear_f32_kernel<5><<<grid, 512, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (float*)y, rows, k, n); break;
            case 6: ling_bf16_linear_f32_kernel<6><<<grid, 512, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (float*)y, rows, k, n); break;
            case 7: ling_bf16_linear_f32_kernel<7><<<grid, 512, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (float*)y, rows, k, n); break;
            default: ling_bf16_linear_f32_kernel<8><<<grid, 512, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (float*)y, rows, k, n); break;
        }
    } else {
        ling_bf16_linear_f32_kernel<8><<<grid, 512, 0, st>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (float*)y, rows, k, n);
    }
    return GB_CHECK_LAUNCH();
}

// The trunk lm_head's input norm fused with its input rotation: one launch computes `h = bf16(hidden + res)` and
// `y = rmsnorm(h; w)` with the exact `rmsnorm_k<true>` math (writing both, exactly like `glue::add_rmsnorm`), then
// rotates the normalised row with `suh` into `xh` (fp16) with the exact `ling_rot_in_kernel` math. Bit-identical per
// element to `glue::add_rmsnorm` + the rot_in launch of `gb_exl3_linear_k2`; `gb_ling_exl3_linear_prerot` then runs
// the head's linear on `xh` (no rot_in launch). The last-layer post-MoE site of bailing_moe_v3 and the P2 prefill
// norm sites of the trunk (k = dim = 4096, rows <= 4096). `res == nullptr` runs plain rmsnorm (no add).
GB_EXPORT int gb_ling_norm_rot_in_single(const void* x, const void* res, const float* w, void* h_out, void* y,
                                         const void* suh, void* xh,
                                         int m, int k, int dim, float eps, void* stream) {
    using namespace lingproj;
    if (m < 1 || m > 4096 || dim <= 0 || dim > 4096 || dim % 8 != 0 || k <= 0 || k > 4096 || k % 128 != 0 ||
        k != dim || eps <= 0.0f || !x || !w || !y || !suh || !xh || (h_out == nullptr) != (res == nullptr))
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const bool pdl = pdl_exl3();
    return launch_k(pdl, ling_norm_rot_in_single_kernel, dim3((unsigned)m, 1), dim3(256), 0, st,
                    (const __nv_bfloat16*)x, (const __nv_bfloat16*)res, w, (__nv_bfloat16*)h_out,
                    (__nv_bfloat16*)y, (const half*)suh, (half*)xh, dim, k, eps);
}

// The P2 q_a -> q_b step's rmsnorm fused with the q_b input rotation: one launch computes `y = rmsnorm(x; w)`
// with the exact `rmsnorm_k<false>` math (bf16 in place via `y` aliasing `x`) and rotates the normalised row with
// `suh` into `xh` (fp16) with the exact `ling_rot_in_kernel` math (the `ling_rmsnorm_rot_in_kernel` body).
// Bit-identical per element to `glue::rmsnorm` + the `gb_glm_p2_rotate` launch
// `glm_p2::ling_linear_parts` runs; the caller then runs `gb_ling_p2_gemm` on `xh`. The MLA q_b site of
// bailing_moe_v3 (k = dim = q_lora_rank <= 1536, rows <= 4096).
GB_EXPORT int gb_ling_rmsnorm_rot_in(const void* x, const float* norm_w, void* y, const void* suh, void* xh,
                                     int m, int dim, float eps, void* stream) {
    using namespace lingproj;
    if (m < 1 || m > 4096 || dim <= 0 || dim > 1536 || dim % 8 != 0 || dim % 128 != 0 || eps <= 0.0f ||
        !x || !norm_w || !y || !suh || !xh)
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const bool pdl = pdl_exl3();
    return launch_k(pdl, ling_rmsnorm_rot_in_kernel, dim3((unsigned)m, 1), dim3(256), 0, st,
                    (const __nv_bfloat16*)x, norm_w, (__nv_bfloat16*)y, (const half*)suh, (half*)xh, dim, dim, eps);
}

// The P2 dense FFN down_proj step's SwiGLU fused with the down input rotation: one launch computes the exact
// `ew_vec_k<OpSwiglu>` element of g/u (silu_f(g) * u rounded to bf16; the `ling_silu_mul_rot_in_kernel` body)
// and rotates it with `suh` into `xh` (fp16) with the exact `ling_rot_in_kernel` math. Bit-identical per element
// to `glue::swiglu` + the `gb_glm_p2_rotate` launch `glm_p2::ling_linear_parts` runs; the caller then runs
// `gb_ling_p2_gemm` on `xh`. The dense down_proj site of bailing_moe_v3 (k = dense_intermediate <= 10240,
// rows <= 4096).
GB_EXPORT int gb_ling_silu_mul_rot_in(const void* g, const void* u, const void* suh, void* xh,
                                      int m, int k, void* stream) {
    using namespace lingproj;
    if (m < 1 || m > 4096 || k <= 0 || k % 128 != 0 || !g || !u || !suh || !xh)
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const bool pdl = pdl_exl3();
    const int gy_single = (k / 128 + 3) / 4;
    const dim3 grid((unsigned)m, (unsigned)gy_single);
    return launch_k(pdl, ling_silu_mul_rot_in_kernel, grid, dim3(128), 0, st,
                    (const __nv_bfloat16*)g, (const __nv_bfloat16*)u, (const half*)suh, (half*)xh, k);
}

// The linear half of `gb_exl3_linear_k2` on a caller-provided rotated input `xh` (fp16 [m, k], e.g. written by
// `gb_ling_norm_rot_in_single`): the same decode plan, the same k order, the same warp-order sum and the same finish,
// so `y` is bit-identical to `gb_exl3_linear_k2` on `x` after its own rot_in. Covers the non-fragment layout
// (m <= 16) and SK == 1 plans (the trunk lm_head: k2 = 12, codebook 2, n = vocab). `workspace` is the caller's decode
// workspace (>= the rotated input's bytes); `xh` may be `workspace` itself. Returns cudaErrorInvalidValue for a shape
// the fusion does not cover: the caller falls back to the separate launches.
GB_EXPORT int gb_ling_exl3_linear_prerot(const void* trellis, const void* svh, const void* bias, void* y, int y_dtype,
                                         int m, int k, int n, int k2, int codebook,
                                         const void* xh, void* workspace, size_t workspace_bytes, void* stream) {
    using namespace lingproj;
    if (m < 1 || m > kMaxRows || !shape_ok_k2(k, n, k2, codebook) || !dtype_ok(y_dtype) ||
        !trellis || !svh || !y || !xh || !workspace)
        return (int)cudaErrorInvalidValue;
    if (group_tiles(m) >= 2) return (int)cudaErrorInvalidValue;   // the non-fragment rotation layout only
    cudaStream_t st = gb_stream(stream);
    const Plan p = plan_for(k, n);
    if (p.sk != 1) return (int)cudaErrorInvalidValue;             // finishes inline (no split partials)
    if (workspace_bytes < xf_bytes(m, k)) return (int)cudaErrorInvalidValue;
    const int tw = 4 * k2;
    SingleArgs a{(const half*)xh, static_cast<const uint32_t*>(trellis), (long long)(n / 16) * tw, 8LL * tw,
                 (const half*)svh, y, y_dtype, n, nullptr, nullptr, m, k, 1, p.wk, 0, pdl_exl3(),
                 nullptr, nullptr, 0, 0.0f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    return dispatch_single(k2, codebook, a, st);
}

// Bytes of workspace gb_ling_topk_f16 needs (0 for an unsupported shape): 256 bytes of int32 fused-combine counters
// (64 rows max) at offset 0, then the stage-1 pools of `gb_glm_topk_f16_workspace_bytes`.
GB_EXPORT size_t gb_ling_topk_f16_workspace_bytes(int rows, int valid, int count) {
    using namespace lingproj;
    if (rows <= 0 || valid <= 0 || count <= 0 || count > 64) return 0;
    // fixed layout: [0, 256) holds the 64 int32 self-resetting per-row combine counters and [256, ...) the stage-1
    // pools. The counters sit in a region the pools never touch, so varying `rows`/`count` across calls on one buffer
    // cannot leave candidate bits where `row_cnt` is read (the old row-tail layout put `row_cnt` inside a previous
    // call's pool when `rows` shrank between steps).
    return 256 + (size_t)rows * ltk_topk_nchunks(valid) * ltk_topk_p(count) * sizeof(ltk_u64);
}

// The exact per-row top-`count` of fp16 logits `[rows, stride]` over columns 0..valid by (value DESC, id ASC), in ONE
// launch: the stage-1 CTAs write their pool candidates and the row's LAST CTA folds them (the stage-2 body) in place.
// Bit-identical ids/vals to `gb_glm_topk_f16` on the same inputs. `workspace` is `gb_ling_topk_f16_workspace_bytes`
// bytes, 16-byte aligned.
GB_EXPORT int gb_ling_topk_f16(const void* logits, int rows, int stride, int valid, int count, int* ids, float* vals,
                               void* workspace, size_t ws_bytes, void* stream) {
    using namespace lingproj;
    if (rows < 0 || valid < 1 || valid > stride || count < 1 || count > 64 || count > valid) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    if (!logits || !ids || !vals || !workspace || !aligned16(workspace)) return (int)cudaErrorInvalidValue;
    if (ws_bytes < gb_ling_topk_f16_workspace_bytes(rows, valid, count)) return (int)cudaErrorInvalidValue;
    int* row_cnt = (int*)workspace;
    ltk_u64* ws = (ltk_u64*)((char*)workspace + 256);
    const int p = ltk_topk_p(count);
    if (p == 16) return launch_ling_topk<16>(logits, rows, stride, valid, count, ids, vals, ws, row_cnt, stream);
    if (p == 32) return launch_ling_topk<32>(logits, rows, stride, valid, count, ids, vals, ws, row_cnt, stream);
    return launch_ling_topk<64>(logits, rows, stride, valid, count, ids, vals, ws, row_cnt, stream);
}

}  // extern "C"
