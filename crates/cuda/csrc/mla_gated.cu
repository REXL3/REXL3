// gb-nvcc: --fmad=false
// Ling-3.1-flash-1019 (`BailingMoeV3MultiLatentAttention`) Gated MLA: 64 heads, `qk_nope_head_dim = 128`,
// `qk_rope_head_dim = 64`, `v_head_dim = 128`, `kv_lora_rank = 512`, full causal attention (no DSA indexer),
// interleaved (adjacent-pair) RoPE with `theta = 6e6`, and a head-wise sigmoid output gate `g_proj` (`[64, 4096]`
// bf16) applied before `attention.dense`. This file is the CUDA side of `src/mla_gated.rs`.
//
// The math, exactly as the reference does it, for one row r with normalized latent c_kv (512), rotated rope keys
// k_rot (64), per-head nope queries q_pass (128) and rotated rope queries q_rot (64):
//
//   q_lat[h][c]  = sum_{d<128} q_pass[h][d] * W_UK[h][d][c]        (absorb; W_UK = kv_b_proj rows h*256 + d)
//   score[h][j]  = (q_lat[h] . c_kv[j] + q_rot[h] . k_rot[j]) * 192^-0.5
//   o_lat[h][c]  = sum_{j<=pos} softmax_j(score[h][j]) * c_kv[j][c]
//   out[h][n]    = sum_{c<512} o_lat[h][c] * W_UV[h][c][n]         (expand; W_UV = kv_b_proj rows h*256 + 128 + n)
//   gated[h][n]  = out[h][n] * sigmoid(x[r] . g_proj[h])           (gate, applied per head before dense)
//
// The KV cache holds, per token per layer, 576 T elements: the normalized 512-wide latent (col 0..511) followed by
// the 64 rotated rope key (col 512..575) — the compressed form, so a token stores 576 elements regardless of head
// count (spec §5.2). `kv_a_proj_with_mqa` is a padded EXL3 group `(4096, 640)`: cols 0..511 are the latent, cols
// 512..575 the rope, cols 576..639 are zero padding and are never read.
//
// RoPE (adjacent-pair, `rope_interleave = true`): the reference's `apply_rotary_pos_emb_interleave` rotates pair
// (2i, 2i+1) with frequency i but stores the results half-split:
//   y[i]       = x[2i] * c_i - x[2i+1] * s_i        (i < 32)
//   y[i + 32]  = x[2i] * s_i + x[2i+1] * c_i        (i < 32)
// with `inv_freq[i] = 1 / theta^(2i/64)`, `c_i = cos(pos * inv_freq[i])`, `s_i = sin(pos * inv_freq[i])`.
// The host builds the 32-wide cos/sin tables (one entry per position) — see `mla_gated::rope_tables`.
//
// ROW-INVARIANCE BY CONSTRUCTION. Every kernel is a per-row program: a row's output bits depend only on its own
// query row, its slot's keys 0..=pos[r] and its own position — never on how many rows a call has, their order, other
// slots, the block placement or the context. Absorb/expand outputs are one fixed-order fp32 FMA chain per output
// element (rounded once to T); attention is online softmax over keys in ascending order with fixed-order tree
// reductions; the gate is a fixed-order dot. No atomics, no split-K, no rows-derived tiling.
//
// Layouts (T = fp16 or bf16 for every activation, the cache and the packed weights; `g_proj` is always bf16):
//   latent cache   T [n_blocks][256 tokens][576]     block table i32 [slots][max_blocks], as the GQA cache
//   w_uk          T [64][128 d][512 c]              kv_b_proj rows h*256 + d, repacked contiguous
//   w_uvt         T [64][512 c][128 n]              kv_b_proj rows h*256 + 128 + n, transposed to [c][n]
//   q (full)      T [rows][row_stride]              head h at h*192: [0,128) nope, [128,192) rope
//   kv (padded)   T [rows][kv_stride >= 640]        cols [0,512) latent, [512,576) rope, [576,640) padding
//   q_lat         T [rows][64][512], q_rot  T [rows][64][64]
//   g_proj        bf16 [64][4096]
//   out / gated   T [rows][64*128]
#include "common.cuh"
#include <type_traits>

namespace {

constexpr int NH = 64;          // heads
constexpr int DC = 512;         // latent width (kv_lora_rank)
constexpr int DN = 128;         // qk_nope_head_dim == v_head_dim
constexpr int DR = 64;          // qk_rope_head_dim
constexpr int DV = 128;         // v_head_dim
constexpr int QK = DN + DR;     // qk_head_dim = 192
constexpr int CACHE_W = DC + DR;  // 576 elements per token per layer
constexpr int PAGE = 256;       // tokens per block
constexpr int HIDDEN = 4096;
constexpr int V_ROWS = NH * DV; // 8192
constexpr float SM_SCALE = 0.07216878364870322f;  // 192^-0.5

typedef __nv_bfloat16 bf16;
typedef __half f16;

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<bf16>(bf16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f32<f16>(f16 v) { return __half2float(v); }

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ bf16 from_f32<bf16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ f16 from_f32<f16>(float v) { return __float2half_rn(v); }

// ---------------------------------------------------------------- kv prepare: RMSNorm + rope, into the cache

// One block per row. Normalises the 512-wide latent (cols 0..511 of the padded kv row, `kv_a_layernorm`), rotates
// the 64-wide rope key (cols 512..575), and writes both to the cache at (slot, pos). Grid: rows blocks of 256.
template <typename T>
__global__ void __launch_bounds__(256)
kv_prep_kernel(const T* __restrict__ kv, int kv_stride, const T* __restrict__ norm_w,
               const float* __restrict__ cos, const float* __restrict__ sin, int table_stride,
               const int* __restrict__ table, int max_blocks, const int* __restrict__ row_slot,
               const int* __restrict__ row_pos, T* __restrict__ cache) {
    const int r = blockIdx.x;
    const int pos = row_pos[r];
    if (pos < 0) return;
    const int blk = table[(int64_t)row_slot[r] * max_blocks + pos / PAGE];
    const T* src = kv + (int64_t)r * kv_stride;
    T* dst = cache + ((int64_t)blk * PAGE + (pos % PAGE)) * CACHE_W;
    __shared__ float red[256];
    // sum of squares of the latent in ascending column order, fixed tree reduction
    float ss = 0.0f;
    for (int c = threadIdx.x; c < DC; c += 256) {
        const float v = to_f32<T>(src[c]);
        ss = __fmaf_rn(v, v, ss);
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    // 0-ULP bit-identical warp-0 tree reduction (same addition order as the 8-barrier loop).
    if (threadIdx.x < 32) {
        const float r0 = red[threadIdx.x] + red[threadIdx.x + 128];
        const float r2 = red[threadIdx.x + 64] + red[threadIdx.x + 192];
        const float r1 = red[threadIdx.x + 32] + red[threadIdx.x + 160];
        const float r3 = red[threadIdx.x + 96] + red[threadIdx.x + 224];
        float v = (r0 + r2) + (r1 + r3);
        v += __shfl_down_sync(0xffffffffu, v, 16);
        v += __shfl_down_sync(0xffffffffu, v, 8);
        v += __shfl_down_sync(0xffffffffu, v, 4);
        v += __shfl_down_sync(0xffffffffu, v, 2);
        v += __shfl_down_sync(0xffffffffu, v, 1);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)DC + 1e-6f);
    for (int c = threadIdx.x; c < DC; c += 256) {
        dst[c] = from_f32<T>(__fmul_rn(to_f32<T>(src[c]) * inv, to_f32<T>(norm_w[c])));
    }
    // rope: pair (2i, 2i+1) rotated with frequency i, stored half-split
    const float* cs = cos + (size_t)pos * table_stride;
    const float* sn = sin + (size_t)pos * table_stride;
    for (int i = threadIdx.x; i < DR / 2; i += 256) {
        const float a = to_f32<T>(src[DC + 2 * i]);
        const float b = to_f32<T>(src[DC + 2 * i + 1]);
        const float ci = cs[i], si = sn[i];
        dst[DC + i] = from_f32<T>(a * ci - b * si);
        dst[DC + DR / 2 + i] = from_f32<T>(b * ci + a * si);
    }
}

// ---------------------------------------------------------------- q prepare: absorb (nope) + rope

// y[r][h][j] = round_T( sum_{k=0..K-1} x[r][h][k] * W[h][k][j] ), k ascending, one __fmaf_rn chain per output element.
// One CTA = one warp = (head h, 32 * CPT consecutive columns j); each lane owns CPT consecutive columns; the head's
// nope part starts at h*head_stride in x. Weights are read once per pass of RB rows. The per-row operation sequence
// is independent of RB. This is the `mla.cu` skinny kernel with a head stride parameter.
template <typename T, int K, int J, int CPT, int RB, int UNROLL>
__global__ void __launch_bounds__(32) skinny_kernel(const T* __restrict__ x, int64_t x_stride, int head_stride,
                                                    const T* __restrict__ w, T* __restrict__ y,
                                                    int64_t y_stride, int rows) {
    static_assert(K % UNROLL == 0 && UNROLL % 4 == 0 && K % 8 == 0, "tiling");
    static_assert(CPT == 4 || CPT == 8, "16-byte or 8-byte weight loads");
    __shared__ __align__(16) float sx[RB * K];
    const int lane = threadIdx.x, h = blockIdx.y, j0 = blockIdx.x * (32 * CPT) + lane * CPT;
    const T* wp = w + (int64_t)h * K * J + j0;
    for (int r0 = 0; r0 < rows; r0 += RB) {
        __syncwarp();
        for (int i = lane; i < RB * (K / 8); i += 32) {
            const int rr = i / (K / 8), c8 = i % (K / 8);
            float v[8];
            if (r0 + rr < rows) {
                const uint4 u = *reinterpret_cast<const uint4*>(x + (int64_t)(r0 + rr) * x_stride
                                                               + (int64_t)h * head_stride + c8 * 8);
                const uint32_t uw[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint32_t lo = (uw[e] & 0xffffu) << 16;
                    const uint32_t hi = uw[e] & 0xffff0000u;
                    if constexpr (std::is_same<T, bf16>::value) {
                        v[2 * e] = __uint_as_float(lo);
                        v[2 * e + 1] = __uint_as_float(hi);
                    } else {
                        v[2 * e] = __half2float(__ushort_as_half((unsigned short)(uw[e] & 0xffffu)));
                        v[2 * e + 1] = __half2float(__ushort_as_half((unsigned short)(uw[e] >> 16)));
                    }
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = 0.0f;
            }
            float4* d = reinterpret_cast<float4*>(sx + rr * K + c8 * 8);
            d[0] = make_float4(v[0], v[1], v[2], v[3]);
            d[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
        __syncwarp();
        float acc[RB][CPT];
#pragma unroll
        for (int r = 0; r < RB; ++r)
#pragma unroll
            for (int c = 0; c < CPT; ++c) acc[r][c] = 0.0f;
        for (int k0 = 0; k0 < K; k0 += UNROLL) {
            uint32_t wv[UNROLL][CPT / 2];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const T* p = wp + (int64_t)(k0 + u) * J;
                if constexpr (CPT == 8) {
                    const uint4 t = *reinterpret_cast<const uint4*>(p);
                    wv[u][0] = t.x; wv[u][1] = t.y; wv[u][2] = t.z; wv[u][3] = t.w;
                } else {
                    const uint2 t = *reinterpret_cast<const uint2*>(p);
                    wv[u][0] = t.x; wv[u][1] = t.y;
                }
            }
#pragma unroll
            for (int u4 = 0; u4 < UNROLL; u4 += 4) {
                float wf[4][CPT];
#pragma unroll
                for (int u = 0; u < 4; ++u)
#pragma unroll
                    for (int e = 0; e < CPT / 2; ++e) {
                        const uint32_t w = wv[u4 + u][e];
                        const uint32_t lo = (w & 0xffffu) << 16;
                        const uint32_t hi = w & 0xffff0000u;
                        if constexpr (std::is_same<T, bf16>::value) {
                            wf[u][2 * e] = __uint_as_float(lo);
                            wf[u][2 * e + 1] = __uint_as_float(hi);
                        } else {
                            wf[u][2 * e] = __half2float(__ushort_as_half((unsigned short)(w & 0xffffu)));
                            wf[u][2 * e + 1] = __half2float(__ushort_as_half((unsigned short)(w >> 16)));
                        }
                    }
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    const float4 xv = *reinterpret_cast<const float4*>(sx + r * K + k0 + u4);
                    const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                    for (int u = 0; u < 4; ++u)
#pragma unroll
                        for (int c = 0; c < CPT; ++c) acc[r][c] = __fmaf_rn(xs[u], wf[u][c], acc[r][c]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            if (r0 + r >= rows) break;
            T* yp = y + (int64_t)(r0 + r) * y_stride + (int64_t)h * J + j0;
#pragma unroll
            for (int c = 0; c < CPT; c += 2) {
                const float lo = acc[r][c], hi = acc[r][c + 1];
                uint32_t packed;
                if constexpr (std::is_same<T, bf16>::value) {
                    __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
                    packed = *reinterpret_cast<uint32_t*>(&p);
                } else {
                    __half2 p = __floats2half2_rn(lo, hi);
                    packed = *reinterpret_cast<uint32_t*>(&p);
                }
                *reinterpret_cast<uint32_t*>(yp + c) = packed;
            }
        }
    }
}

// One block per (row, head): rotates the head's 64 rope query columns (q cols 128..191 of head h) and writes them to
// q_rot[row][h][0..64) in the half-split layout.
template <typename T>
__global__ void __launch_bounds__(64)
q_rope_kernel(const T* __restrict__ q, int64_t q_stride, const float* __restrict__ cos,
              const float* __restrict__ sin, int table_stride, const int* __restrict__ row_pos,
              T* __restrict__ q_rot, int rows) {
    const int r = blockIdx.x, h = blockIdx.y;
    const T* qh = q + (int64_t)r * q_stride + (int64_t)h * QK + DN;
    T* dst = q_rot + ((int64_t)r * NH + h) * DR;
    const int pos = row_pos[r];
    if (pos < 0) {
        for (int i = threadIdx.x; i < DR; i += blockDim.x) dst[i] = from_f32<T>(0.0f);
        return;
    }
    const float* cs = cos + (size_t)pos * table_stride;
    const float* sn = sin + (size_t)pos * table_stride;
    for (int i = threadIdx.x; i < DR / 2; i += blockDim.x) {
        const float a = to_f32<T>(qh[2 * i]);
        const float b = to_f32<T>(qh[2 * i + 1]);
        const float ci = cs[i], si = sn[i];
        dst[i] = from_f32<T>(a * ci - b * si);
        dst[DR / 2 + i] = from_f32<T>(b * ci + a * si);
    }
}

// ---------------------------------------------------------------- dense causal attention over the latent

// One block per (row, head): online softmax over the row's keys 0..=pos in ascending order, fp32 everywhere, one
// fixed-order tree reduction per key. Each thread owns latent columns tid and tid + 256 of the output accumulator
// (and, for tid < 64, the rope column tid of the score). A padding row (position < 0) writes zeros.
template <typename T>
__global__ void __launch_bounds__(256)
attend_kernel(const T* __restrict__ q_lat, const T* __restrict__ q_rot,
              const T* __restrict__ cache, const int* __restrict__ table, int max_blocks,
              const int* __restrict__ row_slot, const int* __restrict__ row_pos,
              T* __restrict__ out, int rows) {
    const int r = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x;
    __shared__ float red[256];
    const T* qlr = q_lat + ((int64_t)r * NH + h) * DC;
    const T* qrr = q_rot + ((int64_t)r * NH + h) * DR;
    // Each thread reads only its own latent/rope columns; keep them in registers.
    const float ql0 = to_f32<T>(qlr[tid]);
    const float ql1 = to_f32<T>(qlr[tid + 256]);
    const float qr0 = tid < DR ? to_f32<T>(qrr[tid]) : 0.0f;
    T* o = out + ((int64_t)r * NH + h) * DC;
    const int pos = row_pos[r];
    if (pos < 0) {
        o[tid] = from_f32<T>(0.0f);
        o[tid + 256] = from_f32<T>(0.0f);
        return;
    }
    const int* tbl = table + (int64_t)row_slot[r] * max_blocks;
    float o0 = 0.0f, o1 = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (int j = 0; j <= pos; ++j) {
        const int blk = tbl[j / PAGE];
        const T* kp = cache + ((int64_t)blk * PAGE + (j % PAGE)) * CACHE_W;
        const float ka = to_f32<T>(kp[tid]);
        const float kb = to_f32<T>(kp[tid + 256]);
        float p = __fmaf_rn(ql0, ka, 0.0f);
        p = __fmaf_rn(ql1, kb, p);
        if (tid < DR) p = __fmaf_rn(qr0, to_f32<T>(kp[DC + tid]), p);
        red[tid] = p;
        __syncthreads();
        // 0-ULP bit-identical warp-0 tree reduction: exactly the addition order of the
        // s = 128, 64, 32, 16, 8, 4, 2, 1 shared-memory loop, with 3 barriers per key.
        if (tid < 32) {
            const float r0 = red[tid] + red[tid + 128];
            const float r2 = red[tid + 64] + red[tid + 192];
            const float r1 = red[tid + 32] + red[tid + 160];
            const float r3 = red[tid + 96] + red[tid + 224];
            float v = (r0 + r2) + (r1 + r3);
            v += __shfl_down_sync(0xffffffffu, v, 16);
            v += __shfl_down_sync(0xffffffffu, v, 8);
            v += __shfl_down_sync(0xffffffffu, v, 4);
            v += __shfl_down_sync(0xffffffffu, v, 2);
            v += __shfl_down_sync(0xffffffffu, v, 1);
            if (tid == 0) red[0] = v;
        }
        __syncthreads();
        const float score = __fmul_rn(red[0], SM_SCALE);
        const float nm = fmaxf(m, score);
        const float a = m == -INFINITY ? 0.0f : expf(m - nm);
        const float e = expf(score - nm);
        o0 = __fmaf_rn(o0, a, __fmul_rn(e, ka));
        o1 = __fmaf_rn(o1, a, __fmul_rn(e, kb));
        l = __fmaf_rn(l, a, e);
        m = nm;
        __syncthreads();  // every thread is done reading `red` for this key before the next key overwrites it
    }
    const float il = l > 0.0f ? 1.0f / l : 0.0f;
    o[tid] = from_f32<T>(__fmul_rn(o0, il));
    o[tid + 256] = from_f32<T>(__fmul_rn(o1, il));
}

// ---------------------------------------------------------------- gate: sigmoid(x . g_proj[h]) applied per head

// One block per (row, head): the 4096-wide dot of the layer input row with g_proj[h] (bf16) in a fixed-order tree,
// sigmoid, then multiply the head's 128 output columns (threads 0..127).
template <typename T>
__global__ void __launch_bounds__(256)
gate_apply_kernel(const T* __restrict__ x, int64_t x_stride, const bf16* __restrict__ g,
                  const T* __restrict__ out, T* __restrict__ out_gated, int rows) {
    const int r = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x;
    __shared__ float red[256];
    const T* xr = x + (int64_t)r * x_stride;
    const bf16* gr = g + (int64_t)h * HIDDEN;
    float acc = 0.0f;
    for (int c = tid; c < HIDDEN; c += 256) {
        acc = __fmaf_rn(to_f32<T>(xr[c]), __bfloat162float(gr[c]), acc);
    }
    red[tid] = acc;
    __syncthreads();
    // 0-ULP bit-identical warp-0 tree reduction (same addition order as the 8-barrier loop).
    if (tid < 32) {
        const float r0 = red[tid] + red[tid + 128];
        const float r2 = red[tid + 64] + red[tid + 192];
        const float r1 = red[tid + 32] + red[tid + 160];
        const float r3 = red[tid + 96] + red[tid + 224];
        float v = (r0 + r2) + (r1 + r3);
        v += __shfl_down_sync(0xffffffffu, v, 16);
        v += __shfl_down_sync(0xffffffffu, v, 8);
        v += __shfl_down_sync(0xffffffffu, v, 4);
        v += __shfl_down_sync(0xffffffffu, v, 2);
        v += __shfl_down_sync(0xffffffffu, v, 1);
        if (tid == 0) red[0] = v;
    }
    __syncthreads();
    const float gate = 1.0f / (1.0f + expf(-red[0]));
    const T* oin = out + (int64_t)r * V_ROWS + h * DV;
    T* od = out_gated + (int64_t)r * V_ROWS + h * DV;
    if (tid < DV) od[tid] = from_f32<T>(__fmul_rn(to_f32<T>(oin[tid]), gate));
}

// ---------------------------------------------------------------- direct bf16 linear kernels
//
// `gb_mlag_bf16_linear`: y[r][j] = sum_{c<k} x[r][c] * w[j][c] for x bf16 [rows, k], w bf16 [n, k], y
// bf16 [rows, n]. One warp computes one (row, output column): 8 warps per block cover 8 columns
// j = blockIdx.x * 8 + warp; the 32 lanes stride over k/8 uint4 (8 bf16) loads with ascending __fmaf_rn
// chains; a __shfl_down_sync tree sums the lanes; lane 0 writes the rounded bf16 scalar.
__device__ __forceinline__ float fma8_bf16(float acc, const uint4 xu, const uint4 wu) {
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

__device__ __forceinline__ float warp_reduce_sum5(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

__global__ void __launch_bounds__(256)
gb_mlag_bf16_linear_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                           bf16* __restrict__ y, int rows, int k, int n) {
    const int j = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (j >= n) return;
    const int r = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int k8 = k >> 3;
    float acc = 0.0f;
    for (int c8 = lane; c8 < k8; c8 += 32) {
        const uint4 xu = *reinterpret_cast<const uint4*>(x + (int64_t)r * k + c8 * 8);
        const uint4 wu = *reinterpret_cast<const uint4*>(w + (int64_t)j * k + c8 * 8);
        acc = fma8_bf16(acc, xu, wu);
    }
    acc = warp_reduce_sum5(acc);
    if (lane == 0) y[(int64_t)r * n + j] = __float2bfloat16_rn(acc);
}

// `gb_mlag_eh_proj_cat_bf16`: y[r][j] = sum_{c<h} e_norm[r][c] * w[j][c] + sum_{c<h} h_norm[r][c] *
// w[j][h + c] for e_norm/h_norm bf16 [rows, h], w bf16 [h, 2h], y bf16 [rows, h]. Same warp-per-output
// scheme; the e dot completes before the h dot (both ascending in c).
__global__ void __launch_bounds__(256)
gb_mlag_eh_proj_cat_bf16_kernel(const bf16* __restrict__ e_norm, const bf16* __restrict__ h_norm,
                                const bf16* __restrict__ w, bf16* __restrict__ y, int rows, int h) {
    const int j = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (j >= h) return;
    const int r = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int h8 = h >> 3;
    const bf16* er = e_norm + (int64_t)r * h;
    const bf16* hr = h_norm + (int64_t)r * h;
    const bf16* wj = w + (int64_t)j * (2 * h);
    float acc = 0.0f;
    for (int c8 = lane; c8 < h8; c8 += 32) {
        const uint4 eu = *reinterpret_cast<const uint4*>(er + c8 * 8);
        const uint4 wu = *reinterpret_cast<const uint4*>(wj + c8 * 8);
        acc = fma8_bf16(acc, eu, wu);
    }
    for (int c8 = lane; c8 < h8; c8 += 32) {
        const uint4 hu = *reinterpret_cast<const uint4*>(hr + c8 * 8);
        const uint4 wu = *reinterpret_cast<const uint4*>(wj + h + c8 * 8);
        acc = fma8_bf16(acc, hu, wu);
    }
    acc = warp_reduce_sum5(acc);
    if (lane == 0) y[(int64_t)r * h + j] = __float2bfloat16_rn(acc);
}

// ---------------------------------------------------------------- launchers

template <typename T> int kv_prep_t(const void* kv, int kv_stride, const void* norm_w,
                                    const float* cos, const float* sin, int table_stride,
                                    const int* table, int max_blocks, const int* row_slot,
                                    const int* row_pos, void* cache, int rows, cudaStream_t s) {
    kv_prep_kernel<T><<<rows, 256, 0, s>>>((const T*)kv, kv_stride, (const T*)norm_w, cos, sin, table_stride,
                                            table, max_blocks, row_slot, row_pos, (T*)cache);
    return GB_CHECK_LAUNCH();
}

template <typename T> int q_prep_t(const void* q, int64_t qs, const void* w_uk,
                                   const float* cos, const float* sin, int table_stride,
                                   const int* row_pos, void* q_lat, void* q_rot, int rows,
                                   cudaStream_t s) {
    const T* qp = (const T*)q;
    T* ql = (T*)q_lat;
    T* qr = (T*)q_rot;
    const int64_t ys = (int64_t)NH * DC;
    if (rows <= 1) skinny_kernel<T, DN, DC, 8, 1, 8><<<dim3(DC / (32 * 8), NH), 32, 0, s>>>(qp, qs, QK, (const T*)w_uk, ql, ys, rows);
    else if (rows <= 4) skinny_kernel<T, DN, DC, 8, 4, 8><<<dim3(DC / (32 * 8), NH), 32, 0, s>>>(qp, qs, QK, (const T*)w_uk, ql, ys, rows);
    else skinny_kernel<T, DN, DC, 8, 16, 8><<<dim3(DC / (32 * 8), NH), 32, 0, s>>>(qp, qs, QK, (const T*)w_uk, ql, ys, rows);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    q_rope_kernel<T><<<dim3(rows, NH), 64, 0, s>>>(qp, qs, cos, sin, table_stride, row_pos, qr, rows);
    return GB_CHECK_LAUNCH();
}

template <typename T> int attend_t(const void* q_lat, const void* q_rot, const void* cache,
                                   const int* table, int max_blocks, const int* row_slot,
                                   const int* row_pos, void* out, int rows, cudaStream_t s) {
    attend_kernel<T><<<dim3(rows, NH), 256, 0, s>>>((const T*)q_lat, (const T*)q_rot, (const T*)cache, table,
                                                     max_blocks, row_slot, row_pos, (T*)out, rows);
    return GB_CHECK_LAUNCH();
}

template <typename T> int expand_t(const void* o_lat, const void* w_uvt, void* out, int rows,
                                   cudaStream_t s) {
    const T* x = (const T*)o_lat;
    T* y = (T*)out;
    const int64_t xs = (int64_t)NH * DC, ys = (int64_t)NH * DV;
    if (rows <= 1) skinny_kernel<T, DC, DV, 4, 1, 16><<<dim3(DV / (32 * 4), NH), 32, 0, s>>>(x, xs, DC, (const T*)w_uvt, y, ys, rows);
    else if (rows <= 4) skinny_kernel<T, DC, DV, 4, 4, 16><<<dim3(DV / (32 * 4), NH), 32, 0, s>>>(x, xs, DC, (const T*)w_uvt, y, ys, rows);
    else skinny_kernel<T, DC, DV, 4, 16, 8><<<dim3(DV / (32 * 4), NH), 32, 0, s>>>(x, xs, DC, (const T*)w_uvt, y, ys, rows);
    return GB_CHECK_LAUNCH();
}

template <typename T> int gate_apply_t(const void* x, int64_t xs, const void* g, const void* out,
                                       void* out_gated, int rows, cudaStream_t s) {
    gate_apply_kernel<T><<<dim3(rows, NH), 256, 0, s>>>((const T*)x, xs, (const bf16*)g, (const T*)out,
                                                         (T*)out_gated, rows);
    return GB_CHECK_LAUNCH();
}

}  // namespace

// ---------------------------------------------------------------- exported launchers (dtype: 0 = fp16, 1 = bf16)

// kv: T [rows][kv_stride >= 640], cols [0,512) latent, [512,576) rope, [576,640) padding ignored; norm_w: T [512];
// cos/sin: f32 [max_pos][table_stride >= 32]; cache: T [n_blocks][256][576]. Negative positions skip the row.
GB_EXPORT int gb_mlag_kv_prep(const void* kv, int kv_stride, const void* norm_w, const float* cos,
                              const float* sin, int table_stride, const int* block_table, int max_blocks,
                              const int* row_slot, const int* row_pos, void* cache, int rows, int dtype,
                              void* stream) {
    if (rows < 0 || kv_stride < 640 || table_stride < 32 || max_blocks <= 0 || dtype < 0 || dtype > 1)
        return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    return dtype == 1 ? kv_prep_t<bf16>(kv, kv_stride, norm_w, cos, sin, table_stride, block_table, max_blocks,
                                        row_slot, row_pos, cache, rows, s)
                      : kv_prep_t<f16>(kv, kv_stride, norm_w, cos, sin, table_stride, block_table, max_blocks,
                                       row_slot, row_pos, cache, rows, s);
}

// q: T [rows][q_row_stride] with head h at h*192 (nope 0..127, rope 128..191); w_uk: T [64][128][512]; q_lat:
// T [rows][64][512]; q_rot: T [rows][64][64].
GB_EXPORT int gb_mlag_q_prep(const void* q, int64_t q_row_stride, const void* w_uk, const float* cos,
                             const float* sin, int table_stride, const int* row_pos, void* q_lat,
                             void* q_rot, int rows, int dtype, void* stream) {
    if (rows < 0 || q_row_stride < (int64_t)NH * QK || table_stride < 32 || dtype < 0 || dtype > 1)
        return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    return dtype == 1 ? q_prep_t<bf16>(q, q_row_stride, w_uk, cos, sin, table_stride, row_pos, q_lat, q_rot,
                                       rows, s)
                      : q_prep_t<f16>(q, q_row_stride, w_uk, cos, sin, table_stride, row_pos, q_lat, q_rot,
                                      rows, s);
}

// Dense causal latent attention: row r attends to keys 0..=row_pos[r] of slot row_slot[r] (already written by
// gb_mlag_kv_prep); a negative position writes zeros. q_lat: T [rows][64][512]; q_rot: T [rows][64][64]; cache:
// T [n_blocks][256][576]; out: T [rows][64][512].
GB_EXPORT int gb_mlag_attend(const void* q_lat, const void* q_rot, const void* cache, const int* block_table,
                             int max_blocks, const int* row_slot, const int* row_pos, void* out, int rows,
                             int dtype, void* stream) {
    if (rows < 1 || max_blocks <= 0 || dtype < 0 || dtype > 1) return (int)cudaErrorInvalidValue;
    cudaStream_t s = gb_stream(stream);
    return dtype == 1 ? attend_t<bf16>(q_lat, q_rot, cache, block_table, max_blocks, row_slot, row_pos, out,
                                       rows, s)
                      : attend_t<f16>(q_lat, q_rot, cache, block_table, max_blocks, row_slot, row_pos, out,
                                      rows, s);
}

// o_lat: T [rows][64][512]; w_uvt: T [64][512][128]; out: T [rows][64 * 128].
GB_EXPORT int gb_mlag_expand(const void* o_lat, const void* w_uvt, void* out, int rows, int dtype, void* stream) {
    if (rows < 0 || dtype < 0 || dtype > 1) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    return dtype == 1 ? expand_t<bf16>(o_lat, w_uvt, out, rows, s)
                      : expand_t<f16>(o_lat, w_uvt, out, rows, s);
}

// gate: sigmoid(x[r] . g_proj[h]) * out[r][h*128 .. h*128+128). g: bf16 [64][4096]; x: T [rows][x_stride];
// out, out_gated: T [rows][64 * 128].
GB_EXPORT int gb_mlag_gate_apply(const void* x, int64_t x_stride, const void* g, const void* out,
                                 void* out_gated, int rows, int dtype, void* stream) {
    if (rows < 0 || x_stride < HIDDEN || dtype < 0 || dtype > 1) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    return dtype == 1 ? gate_apply_t<bf16>(x, x_stride, g, out, out_gated, rows, s)
                      : gate_apply_t<f16>(x, x_stride, g, out, out_gated, rows, s);
}

// Direct bf16 linear: x: bf16 [rows, k]; w: bf16 [n, k]; y: bf16 [rows, n]; k % 256 == 0, n >= 1.
// Row-invariant by construction: one warp computes one (row, output column).
GB_EXPORT int gb_mlag_bf16_linear(const void* x, const void* w, void* y, int rows, int k, int n,
                                   void* stream) {
    if (rows < 0 || n < 1 || k <= 0 || k % 256 != 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    gb_mlag_bf16_linear_kernel<<<dim3((n + 7) / 8, rows), 256, 0, s>>>((const bf16*)x, (const bf16*)w,
                                                                         (bf16*)y, rows, k, n);
    return GB_CHECK_LAUNCH();
}

// Fused eh_proj cat: e_norm, h_norm: bf16 [rows, h]; w: bf16 [h, 2h]; y: bf16 [rows, h]; h % 256 == 0.
GB_EXPORT int gb_mlag_eh_proj_cat_bf16(const void* e_norm, const void* h_norm, const void* w, void* y,
                                       int rows, int h, void* stream) {
    if (rows < 0 || h <= 0 || h % 256 != 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
    gb_mlag_eh_proj_cat_bf16_kernel<<<dim3((h + 7) / 8, rows), 256, 0, s>>>(
        (const bf16*)e_norm, (const bf16*)h_norm, (const bf16*)w, (bf16*)y, rows, h);
    return GB_CHECK_LAUNCH();
}
