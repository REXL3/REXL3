// HCX: fused multi-HyperConnection pre-mix for Flash-Next (Qwen4_Exp). Original REXL3 code, Victor Cruz.
//
// Same math as gb_hc_pre / gb_hc_head (mhc.cu), restructured so the 13 MB of fp16 mixer weights a call are read once
// for every row of the window and the per-stream RMS no longer needs its own pass:
//   normed[s][d] = x[s][d] * rms_s * (1 + nw[s][d]),   rms_s = rsqrt(mean_d x[s][d]^2 + eps)
//   low[r]  = silu(  (1/hc) sum_s rms_s * P[r][s] ),   P[r][s] = sum_d down[r][s*H + d] * (1 + nw[s*H + d]) * x[s][d]
//   inj[q]  = 2 sigmoid((1/hc) sum_s rms_s * P[rank + q][s])     (inject rows ride along as extra rows of down)
//   y[d]    = (1/hc) sum_s sigmoid(up[s*H + d] . low) * normed[s][d]
// K1 computes P (one warp a (row, stream), weights read once for all rows of the window) and the sums of squares;
// K2 turns them into low / inj in every block and applies up. fp32 throughout, weights fp16 as stored.
//
// Decode fast path (k1b/k2c): the same arithmetic, bit for bit, with launch shapes tuned for the 1..8-row window:
// - k1b uses 8-warp blocks (the same per-job math and reduction order);
// - k2c gives each block 16 warps (512 threads) covering 16 columns, so the k2 grid halves (320 -> 160 blocks) with
//   the reference's per-column body untouched;
// - hc_post is regridded over the whole [t, s, d] volume (elementwise, so any grid gives the same bits).
// The fast path is unconditional: the gate needs the bit-exact fast kernels always on. The reference launches are
// still reachable through gb_hcx_pre_ref / gb_hcx_pre_z_ref (used by the bit-exact GPU tests in tests/hcx_perf.rs).
#include "common.cuh"

namespace hcx {

constexpr int TMAX = 8;   // rows a pass (more rows loop in passes)

__device__ __forceinline__ void h8(const half* p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
    const half2* h = reinterpret_cast<const half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ void h8_u4(const uint4 u, float (&v)[8]) {
    const half2* h = reinterpret_cast<const half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

// Qwen4Exp decode shapes: H = 2560, hc = 4, rank = 320 (ninj = 4 when inject rows ride along, else 0). These are
// compile-time constants for the templated fast kernels below, so the per-row t-loop has no `if (t < nt)` guards,
// the acc/out arrays are exactly NT floats, and rank = 320 makes the k2c rank loop a 2-step static trip.
constexpr int QFN_H = 2560, QFN_HC = 4, QFN_RANK = 320;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// K1. Jobs: (s, r) for r < R (rank down rows, then ninj inject rows), r fastest; then (t, s) sum-of-squares jobs.
// P [T][R][hc] fp32, ss [T][hc] fp32. Rows t0 .. t0 + nt - 1 of x [*, hc, H] fp32.
__global__ void __launch_bounds__(128) k1_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                 const half* __restrict__ down, const half* __restrict__ inj,
                                                 float* __restrict__ P, float* __restrict__ ss, int t0, int nt, int H,
                                                 int hc, int rank, int ninj) {
    const int lane = threadIdx.x & 31;
    const int job = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int R = rank + ninj;
    const size_t HC = (size_t)hc * H;
    if (job < R * hc) {
        const int s = job / R, r = job % R;
        const half* w = (r < rank ? down + (size_t)r * HC : inj + (size_t)(r - rank) * HC) + (size_t)s * H;
        const half* n = nw + (size_t)s * H;
        float acc[TMAX];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
        for (int d = 8 * lane; d < H; d += 256) {
            float wv[8], nv[8];
            h8(w + d, wv);
            h8(n + d, nv);
#pragma unroll
            for (int i = 0; i < 8; ++i) wv[i] *= 1.f + nv[i];
#pragma unroll
            for (int t = 0; t < TMAX; ++t) {
                if (t < nt) {
                    const float* xr = x + (size_t)(t0 + t) * HC + (size_t)s * H + d;
                    const float4 a = __ldg(reinterpret_cast<const float4*>(xr));
                    const float4 b = __ldg(reinterpret_cast<const float4*>(xr + 4));
                    acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y +
                              wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t < nt) {
                const float v = warp_sum(acc[t]);
                if (lane == 0) P[((size_t)t * R + r) * hc + s] = v;
            }
        }
        return;
    }
    const int q = job - R * hc;
    if (q >= nt * hc) return;
    const int t = q / hc, s = q % hc;
    const float* xr = x + (size_t)(t0 + t) * HC + (size_t)s * H;
    float a = 0.f;
    for (int d = 4 * lane; d < H; d += 128) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(xr + d));
        a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    a = warp_sum(a);
    if (lane == 0) ss[t * hc + s] = a;
}

// K2. Block: 8 warps, one hidden column d a warp. Every block first rebuilds low [nt][rank] in shared memory
// (block 0 also writes inj [nt][ninj]). y [*, H] bf16 rows t0 .. t0 + nt - 1.
__global__ void __launch_bounds__(256) k2_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                 const half* __restrict__ up, const float* __restrict__ P,
                                                 const float* __restrict__ ss, __nv_bfloat16* __restrict__ y,
                                                 float* __restrict__ inj_out, int t0, int nt, int H, int hc, int rank,
                                                 int ninj, float eps, float* __restrict__ dbg_z) {
    extern __shared__ float s_low[];  // [nt][rank]
    __shared__ float s_rms[TMAX * 8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int R = rank + ninj;
    const float inv_hc = 1.f / (float)hc;
    if (tid < nt * hc) s_rms[tid] = rsqrtf(ss[tid] / (float)H + eps);
    __syncthreads();
    for (int i = tid; i < nt * rank; i += blockDim.x) {
        const int t = i / rank, r = i % rank;
        const float* p = P + ((size_t)t * R + r) * hc;
        float a = 0.f;
        for (int s = 0; s < hc; ++s) a += s_rms[t * hc + s] * p[s];
        a *= inv_hc;
        s_low[i] = a / (1.f + __expf(-a));
    }
    if (blockIdx.x == 0 && inj_out != nullptr) {
        for (int i = tid; i < nt * ninj; i += blockDim.x) {
            const int t = i / ninj, qq = i % ninj;
            const float* p = P + ((size_t)t * R + rank + qq) * hc;
            float a = 0.f;
            for (int s = 0; s < hc; ++s) a += s_rms[t * hc + s] * p[s];
            inj_out[(size_t)(t0 + t) * ninj + qq] = 2.f / (1.f + __expf(-a * inv_hc));
        }
    }
    __syncthreads();

    const int d = blockIdx.x * 8 + warp;
    if (d >= H) return;
    const size_t HC = (size_t)hc * H;
    float out[TMAX];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) out[t] = 0.f;
    for (int s = 0; s < hc; ++s) {
        const half* row = up + ((size_t)s * H + d) * rank;
        float acc[TMAX];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
        for (int c = 8 * lane; c < rank; c += 256) {
            float wv[8];
            h8(row + c, wv);
#pragma unroll
            for (int t = 0; t < TMAX; ++t) {
                if (t < nt) {
                    const float* l = s_low + t * rank + c;
                    const float4 a = *reinterpret_cast<const float4*>(l);
                    const float4 b = *reinterpret_cast<const float4*>(l + 4);
                    acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y +
                              wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
        const float nv = 1.f + __half2float(nw[(size_t)s * H + d]);
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t < nt) {
                const float z = warp_sum(acc[t]);
                if (dbg_z) dbg_z[((size_t)s * nt + t) * H + d] = z;
                const float xv = x[(size_t)(t0 + t) * HC + (size_t)s * H + d];
                out[t] += xv * s_rms[t * hc + s] * nv / (1.f + __expf(-z));
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < TMAX; ++t)
            if (t < nt) y[(size_t)(t0 + t) * H + d] = __float2bfloat16(out[t] * inv_hc);
    }
}

// k1b: the reference k1's jobs and reduction order, launched with 8 warps a block (the decode window's 1..8 rows
// and the real shapes make the grid small; fewer blocks cut the launch and drain tail). Per-job math is unchanged:
// the same per-thread add sequence and the same warp-shuffle reduction tree produce bit-identical P and ss.
__global__ void __launch_bounds__(256) k1b_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                  const half* __restrict__ down, const half* __restrict__ inj,
                                                  float* __restrict__ P, float* __restrict__ ss, int t0, int nt,
                                                  int H, int hc, int rank, int ninj) {
    const int lane = threadIdx.x & 31;
    const int job = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int R = rank + ninj;
    const size_t HC = (size_t)hc * H;
    if (job < R * hc) {
        const int s = job / R, r = job % R;
        const half* w = (r < rank ? down + (size_t)r * HC : inj + (size_t)(r - rank) * HC) + (size_t)s * H;
        const half* n = nw + (size_t)s * H;
        float acc[TMAX];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
        for (int d = 8 * lane; d < H; d += 256) {
            float wv[8], nv[8];
            h8(w + d, wv);
            h8(n + d, nv);
#pragma unroll
            for (int i = 0; i < 8; ++i) wv[i] *= 1.f + nv[i];
#pragma unroll
            for (int t = 0; t < TMAX; ++t) {
                if (t < nt) {
                    const float* xr = x + (size_t)(t0 + t) * HC + (size_t)s * H + d;
                    const float4 a = __ldg(reinterpret_cast<const float4*>(xr));
                    const float4 b = __ldg(reinterpret_cast<const float4*>(xr + 4));
                    acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y +
                              wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t < nt) {
                const float v = warp_sum(acc[t]);
                if (lane == 0) P[((size_t)t * R + r) * hc + s] = v;
            }
        }
        return;
    }
    const int q = job - R * hc;
    if (q >= nt * hc) return;
    const int t = q / hc, s = q % hc;
    const float* xr = x + (size_t)(t0 + t) * HC + (size_t)s * H;
    float a = 0.f;
    for (int d = 4 * lane; d < H; d += 128) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(xr + d));
        a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    a = warp_sum(a);
    if (lane == 0) ss[t * hc + s] = a;
}


// k1b_qfn: the Qwen4Exp specialization of k1b (H=2560, hc=4, rank=320, ninj=0/4). Identical per-job math and
// reduction order as k1b — the same per-thread FMA chain per (t, d) and the same warp-shuffle butterfly — so every
// output bit equals k1b/ref k1. `nt` is a compile-time constant (the dispatch guarantees NT == runtime nt), so
// there are no `if (t < nt)` guards in the inner loops and acc is exactly NT floats. The 10-step d-loop is fully
// unrolled and software-pipelined: the next iteration's raw fp16 weight/norm chunks are issued before the current
// iteration's math, keeping two 128-byte cache lines in flight per warp.
template <int NT, int NINJ>
__global__ void __launch_bounds__(256) k1b_qfn_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                      const half* __restrict__ down, const half* __restrict__ inj,
                                                      float* __restrict__ P, float* __restrict__ ss, int t0) {
    constexpr int H = QFN_H, HC = QFN_HC, RANK = QFN_RANK;
    constexpr int R = RANK + NINJ;
    constexpr int RHC = R * HC;
    constexpr size_t HC_H = (size_t)HC * H;
    const int lane = threadIdx.x & 31;
    const int job = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (job < RHC) {
        const int s = job / R, r = job % R;
        const half* w = (r < RANK ? down + (size_t)r * HC_H : inj + (size_t)(r - RANK) * HC_H) + (size_t)s * H;
        const half* n = nw + (size_t)s * H;
        const float* xb = x + (size_t)t0 * HC_H + (size_t)s * H;
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = 0.f;
        uint4 wu = __ldg(reinterpret_cast<const uint4*>(w + 8 * lane));
        uint4 nu = __ldg(reinterpret_cast<const uint4*>(n + 8 * lane));
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int d = 8 * lane + 256 * k;
            const uint4 wu_n = __ldg(reinterpret_cast<const uint4*>(w + d + 256));
            const uint4 nu_n = __ldg(reinterpret_cast<const uint4*>(n + d + 256));
            float wv[8], nv[8];
            h8_u4(wu, wv);
            h8_u4(nu, nv);
#pragma unroll
            for (int i = 0; i < 8; ++i) wv[i] *= 1.f + nv[i];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float* xr = xb + (size_t)t * HC_H + d;
                const float4 a = __ldg(reinterpret_cast<const float4*>(xr));
                const float4 b = __ldg(reinterpret_cast<const float4*>(xr + 4));
                acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x +
                          wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
            }
            wu = wu_n;
            nu = nu_n;
        }
        {
            const int d = 8 * lane + 2304;
            float wv[8], nv[8];
            h8_u4(wu, wv);
            h8_u4(nu, nv);
#pragma unroll
            for (int i = 0; i < 8; ++i) wv[i] *= 1.f + nv[i];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float* xr = xb + (size_t)t * HC_H + d;
                const float4 a = __ldg(reinterpret_cast<const float4*>(xr));
                const float4 b = __ldg(reinterpret_cast<const float4*>(xr + 4));
                acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x +
                          wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = warp_sum(acc[t]);
            if (lane == 0) P[((size_t)t * R + r) * HC + s] = v;
        }
        return;
    }
    const int q = job - RHC;
    if (q >= NT * HC) return;
    const int t = q / HC, s = q % HC;
    const float* xr = x + (size_t)(t0 + t) * HC_H + (size_t)s * H;
    float a = 0.f;
    for (int dd = 4 * lane; dd < H; dd += 128) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(xr + dd));
        a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    a = warp_sum(a);
    if (lane == 0) ss[t * HC + s] = a;
}


// k2c: the reference k2 body with 16 warps a block (512 threads) covering 16 columns, so the encode grid halves
// (320 -> 160 blocks) while every column's arithmetic is exactly the reference's (same per-column add order, same
// warp-shuffle reduction). Bit-identical to k2 by construction; the bit-exact GPU tests assert it.
__global__ void __launch_bounds__(512) k2c_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                  const half* __restrict__ up, const float* __restrict__ P,
                                                  const float* __restrict__ ss, __nv_bfloat16* __restrict__ y,
                                                  float* __restrict__ inj_out, int t0, int nt, int H, int hc,
                                                  int rank, int ninj, float eps, float* __restrict__ dbg_z) {
    extern __shared__ float s_low[];  // [nt][rank]
    __shared__ float s_rms[TMAX * 8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int R = rank + ninj;
    const float inv_hc = 1.f / (float)hc;
    if (tid < nt * hc) s_rms[tid] = rsqrtf(ss[tid] / (float)H + eps);
    __syncthreads();
    for (int i = tid; i < nt * rank; i += blockDim.x) {
        const int t = i / rank, r = i % rank;
        const float* p = P + ((size_t)t * R + r) * hc;
        float a = 0.f;
        for (int s = 0; s < hc; ++s) a += s_rms[t * hc + s] * p[s];
        a *= inv_hc;
        s_low[i] = a / (1.f + __expf(-a));
    }
    if (blockIdx.x == 0 && inj_out != nullptr) {
        for (int i = tid; i < nt * ninj; i += blockDim.x) {
            const int t = i / ninj, qq = i % ninj;
            const float* p = P + ((size_t)t * R + rank + qq) * hc;
            float a = 0.f;
            for (int s = 0; s < hc; ++s) a += s_rms[t * hc + s] * p[s];
            inj_out[(size_t)(t0 + t) * ninj + qq] = 2.f / (1.f + __expf(-a * inv_hc));
        }
    }
    __syncthreads();

    const int d = blockIdx.x * 16 + warp;
    if (d >= H) return;
    const size_t HC = (size_t)hc * H;
    float out[TMAX];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) out[t] = 0.f;
    for (int s = 0; s < hc; ++s) {
        const half* row = up + ((size_t)s * H + d) * rank;
        float acc[TMAX];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
        for (int c = 8 * lane; c < rank; c += 256) {
            float wv[8];
            h8(row + c, wv);
#pragma unroll
            for (int t = 0; t < TMAX; ++t) {
                if (t < nt) {
                    const float* l = s_low + t * rank + c;
                    const float4 a = *reinterpret_cast<const float4*>(l);
                    const float4 b = *reinterpret_cast<const float4*>(l + 4);
                    acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y +
                              wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
        const float nv = 1.f + __half2float(nw[(size_t)s * H + d]);
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t < nt) {
                const float z = warp_sum(acc[t]);
                if (dbg_z) dbg_z[((size_t)s * nt + t) * H + d] = z;
                const float xv = x[(size_t)(t0 + t) * HC + (size_t)s * H + d];
                out[t] += xv * s_rms[t * hc + s] * nv / (1.f + __expf(-z));
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < TMAX; ++t)
            if (t < nt) y[(size_t)(t0 + t) * H + d] = __float2bfloat16(out[t] * inv_hc);
    }
}

// k2c_qfn: the Qwen4Exp specialization of k2c (H=2560, hc=4, rank=320, ninj=0/4). Identical shared-memory stage
// and per-column arithmetic as k2c — same s_rms/rsqrtf, same low/inj silu sequence, same per-column FMA chains and
// warp_sum butterfly, same bf16 rounding — so every output bit equals k2c/ref k2. `nt` is a compile-time constant
// (dispatch guarantees NT == runtime nt): the per-row guards vanish and s_low is a static NT*RANK shared array.
// The first fp16 chunk of every `up` row and the nw scale for this column are issued BEFORE the first barrier:
// they depend only on (blockIdx, warp, lane), so the 6.55 MB up DRAM stream starts while s_rms + low still run in
// shared memory instead of waiting for both __syncthreads() barriers.
__device__ __forceinline__ float qfn_nv(const half* nw, int s, int d) {
    return 1.f + __half2float(nw[(size_t)s * QFN_H + d]);
}

template <int NT, bool PF>
__device__ __forceinline__ void k2c_qfn_stream(uint4 u0, const half* uprow, const float nv,
                                               const float* low, const float* xrow, const float* rms,
                                               int lane, int d, int s, float (&out)[NT], float* dbg_z) {
    constexpr int H = QFN_H, HC = QFN_HC, RANK = QFN_RANK;
    if constexpr (!PF) u0 = __ldg(reinterpret_cast<const uint4*>(uprow + 8 * lane));
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    float wv[8];
    h8_u4(u0, wv);
    {
        const int c = 8 * lane;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float* l = low + t * RANK + c;
            const float4 a = *reinterpret_cast<const float4*>(l);
            const float4 b = *reinterpret_cast<const float4*>(l + 4);
            acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x +
                      wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
        }
    }
    if (lane < 8) {
        const uint4 u1 = __ldg(reinterpret_cast<const uint4*>(uprow + 256 + 8 * lane));
        h8_u4(u1, wv);
        const int c = 256 + 8 * lane;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float* l = low + t * RANK + c;
            const float4 a = *reinterpret_cast<const float4*>(l);
            const float4 b = *reinterpret_cast<const float4*>(l + 4);
            acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x +
                      wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float z = warp_sum(acc[t]);
        if (dbg_z) dbg_z[((size_t)s * NT + t) * H + d] = z;
        const float xv = xrow[(size_t)t * (size_t)HC * H + d];
        out[t] += xv * rms[t * HC + s] * nv / (1.f + __expf(-z));
    }
}

template <int NT, int NINJ>
__global__ void __launch_bounds__(512) k2c_qfn_kernel(const float* __restrict__ x, const half* __restrict__ nw,
                                                      const half* __restrict__ up, const float* __restrict__ P,
                                                      const float* __restrict__ ss, __nv_bfloat16* __restrict__ y,
                                                      float* __restrict__ inj_out, int t0, float eps,
                                                      float* __restrict__ dbg_z) {
    constexpr int H = QFN_H, HC = QFN_HC, RANK = QFN_RANK;
    constexpr int R = RANK + NINJ;
    constexpr float INV_HC = 1.f / (float)HC;
    __shared__ float s_rms[NT * HC];
    __shared__ float s_low[NT * RANK];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int d = blockIdx.x * 16 + warp;
    constexpr bool PF = NT < 8;  // 8-row windows keep 2 CTAs/SM (≤64 regs): load up chunks lazily instead.
    const half* upbase = up + (size_t)d * RANK;
    // Pre-barrier prefetch: first fp16 chunk of each up row and the nw scale for this column (skipped for NT=8,
    // where the longer s_low stage would otherwise push register usage over the 2-CTA-per-SM threshold).
    const uint4 u0a = PF ? __ldg(reinterpret_cast<const uint4*>(upbase + 8 * lane)) : make_uint4(0, 0, 0, 0);
    const uint4 u1a = PF ? __ldg(reinterpret_cast<const uint4*>(upbase + (size_t)H * RANK + 8 * lane))
                         : make_uint4(0, 0, 0, 0);
    const uint4 u2a = PF ? __ldg(reinterpret_cast<const uint4*>(upbase + (size_t)(2 * H) * RANK + 8 * lane))
                         : make_uint4(0, 0, 0, 0);
    const uint4 u3a = PF ? __ldg(reinterpret_cast<const uint4*>(upbase + (size_t)(3 * H) * RANK + 8 * lane))
                         : make_uint4(0, 0, 0, 0);
    const float nv0 = qfn_nv(nw, 0, d);
    const float nv1 = qfn_nv(nw, 1, d);
    const float nv2 = qfn_nv(nw, 2, d);
    const float nv3 = qfn_nv(nw, 3, d);
    if (tid < NT * HC) s_rms[tid] = rsqrtf(ss[tid] / (float)H + eps);
    __syncthreads();
    for (int i = tid; i < NT * RANK; i += 512) {
        const int t = i / RANK, r = i % RANK;
        const float* p = P + ((size_t)t * R + r) * HC;
        float a = 0.f;
#pragma unroll
        for (int s = 0; s < HC; ++s) a += s_rms[t * HC + s] * p[s];
        a *= INV_HC;
        s_low[i] = a / (1.f + __expf(-a));
    }
    if (blockIdx.x == 0 && inj_out != nullptr) {
        for (int i = tid; i < NT * NINJ; i += 512) {
            const int t = i / NINJ, qq = i % NINJ;
            const float* p = P + ((size_t)t * R + RANK + qq) * HC;
            float a = 0.f;
#pragma unroll
            for (int s = 0; s < HC; ++s) a += s_rms[t * HC + s] * p[s];
            inj_out[(size_t)(t0 + t) * NINJ + qq] = 2.f / (1.f + __expf(-a * INV_HC));
        }
    }
    __syncthreads();
    float out[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) out[t] = 0.f;
    const float* xrow0 = x + (size_t)t0 * (size_t)HC * H;
    k2c_qfn_stream<NT, PF>(u0a, upbase, nv0, s_low, xrow0, s_rms, lane, d, 0, out, dbg_z);
    k2c_qfn_stream<NT, PF>(u1a, upbase + (size_t)H * RANK, nv1, s_low, xrow0 + (size_t)H, s_rms, lane, d, 1, out, dbg_z);
    k2c_qfn_stream<NT, PF>(u2a, upbase + (size_t)(2 * H) * RANK, nv2, s_low, xrow0 + (size_t)(2 * H), s_rms, lane, d, 2, out, dbg_z);
    k2c_qfn_stream<NT, PF>(u3a, upbase + (size_t)(3 * H) * RANK, nv3, s_low, xrow0 + (size_t)(3 * H), s_rms, lane, d, 3, out, dbg_z);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) y[(size_t)(t0 + t) * H + d] = __float2bfloat16(out[t] * INV_HC);
    }
}

}  // namespace hcx

using namespace hcx;

// Scratch floats: P [8][rank + ninj][hc] + ss [8][hc].
GB_EXPORT size_t gb_hcx_scratch_bytes(int hc, int rank, int ninj) {
    return ((size_t)TMAX * (rank + ninj) * hc + (size_t)TMAX * hc) * sizeof(float);
}

// y [T][H] bf16 (and inj [T][hc] when inject_w != null) from streams [T][hc][H] fp32. Same contract as gb_hc_pre.
// `ref=false` (gb_hcx_pre / gb_hcx_pre_z) launches the decode fast kernels k1b/k2c unconditionally, bit-identical to
// k1/k2; `ref=true` (gb_hcx_pre_ref / gb_hcx_pre_z_ref) launches the reference k1_kernel/k2_kernel directly for the
// bit-exact GPU tests.
// Qwen4Exp decode fast-path dispatch (used when H=2560, hc=4, rank=320 and ninj is 0 or 4): the templated
// k1b_qfn/k2c_qfn kernels run with NT == the runtime pass width (1..8), bit-identical to k1b/k2c.
#define HCX_K1B_QFN(NT)                                                                                    \
    do {                                                                                                   \
        const int grid = (R * hc + nt * hc + 7) / 8;                                                       \
        if (ninj == 0)                                                                                     \
            k1b_qfn_kernel<NT, 0><<<grid, 256, 0, st>>>((const float*)streams, (const half*)hc_norm_w,     \
                                                        (const half*)down_w, (const half*)inject_w, P, ss, \
                                                        t0);                                               \
        else                                                                                               \
            k1b_qfn_kernel<NT, 4><<<grid, 256, 0, st>>>((const float*)streams, (const half*)hc_norm_w,     \
                                                        (const half*)down_w, (const half*)inject_w, P, ss, \
                                                        t0);                                               \
    } while (0)

#define HCX_K2C_QFN(NT)                                                                                    \
    do {                                                                                                   \
        if (ninj == 0)                                                                                     \
            k2c_qfn_kernel<NT, 0><<<(H + 15) / 16, 512, 0, st>>>(                                         \
                (const float*)streams, (const half*)hc_norm_w, (const half*)up_w, P, ss,                   \
                (__nv_bfloat16*)y_out, inject_w ? inj_out : nullptr, t0, eps, dbg_z);                      \
        else                                                                                               \
            k2c_qfn_kernel<NT, 4><<<(H + 15) / 16, 512, 0, st>>>(                                         \
                (const float*)streams, (const half*)hc_norm_w, (const half*)up_w, P, ss,                   \
                (__nv_bfloat16*)y_out, inject_w ? inj_out : nullptr, t0, eps, dbg_z);                      \
    } while (0)

static int hcx_pre_impl(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                        const void* inject_w, void* y_out, float* inj_out, void* scratch, unsigned int hidden_size,
                        unsigned int hc_mult, unsigned int rank, float eps, unsigned int num_tokens, void* stream,
                        bool ref, float* dbg_z) {
    if (num_tokens == 0) return 0;
    const int H = (int)hidden_size, hc = (int)hc_mult, rk = (int)rank;
    if (H % 256 || rk % 8 || hc > 8) return (int)cudaErrorInvalidValue;
    const int ninj = inject_w ? hc : 0;
    const int R = rk + ninj;
    const bool qfn = !ref && H == QFN_H && hc == QFN_HC && rk == QFN_RANK && (ninj == 0 || ninj == 4);
    cudaStream_t st = gb_stream(stream);
    float* P = static_cast<float*>(scratch);
    float* ss = P + (size_t)TMAX * R * hc;
    for (int t0 = 0; t0 < (int)num_tokens; t0 += TMAX) {
        const int nt = min(TMAX, (int)num_tokens - t0);
        const int jobs = R * hc + nt * hc;
        const bool use_k1b = !ref;
        const bool use_k2b = !ref;
        if (qfn) {
            switch (nt) {
                case 1: HCX_K1B_QFN(1); break;
                case 2: HCX_K1B_QFN(2); break;
                case 3: HCX_K1B_QFN(3); break;
                case 4: HCX_K1B_QFN(4); break;
                case 5: HCX_K1B_QFN(5); break;
                case 6: HCX_K1B_QFN(6); break;
                case 7: HCX_K1B_QFN(7); break;
                default: HCX_K1B_QFN(8); break;
            }
            int rc = GB_CHECK_LAUNCH();
            if (rc) return rc;
            switch (nt) {
                case 1: HCX_K2C_QFN(1); break;
                case 2: HCX_K2C_QFN(2); break;
                case 3: HCX_K2C_QFN(3); break;
                case 4: HCX_K2C_QFN(4); break;
                case 5: HCX_K2C_QFN(5); break;
                case 6: HCX_K2C_QFN(6); break;
                case 7: HCX_K2C_QFN(7); break;
                default: HCX_K2C_QFN(8); break;
            }
            rc = GB_CHECK_LAUNCH();
            if (rc) return rc;
            continue;
        }
        if (use_k1b) {
            k1b_kernel<<<(jobs + 7) / 8, 256, 0, st>>>((const float*)streams, (const half*)hc_norm_w,
                                                       (const half*)down_w, (const half*)inject_w, P, ss, t0, nt, H,
                                                       hc, rk, ninj);
        } else {
            k1_kernel<<<(jobs + 3) / 4, 128, 0, st>>>((const float*)streams, (const half*)hc_norm_w,
                                                       (const half*)down_w, (const half*)inject_w, P, ss, t0, nt, H,
                                                       hc, rk, ninj);
        }
        int rc = GB_CHECK_LAUNCH();
        if (rc) return rc;
        if (use_k2b) {
            k2c_kernel<<<(H + 15) / 16, 512, (size_t)nt * rk * sizeof(float), st>>>(
                (const float*)streams, (const half*)hc_norm_w, (const half*)up_w, P, ss, (__nv_bfloat16*)y_out,
                inject_w ? inj_out : nullptr, t0, nt, H, hc, rk, ninj, eps, dbg_z);
        } else {
            k2_kernel<<<(H + 7) / 8, 256, (size_t)nt * rk * sizeof(float), st>>>(
                (const float*)streams, (const half*)hc_norm_w, (const half*)up_w, P, ss, (__nv_bfloat16*)y_out,
                inject_w ? inj_out : nullptr, t0, nt, H, hc, rk, ninj, eps, dbg_z);
        }
        rc = GB_CHECK_LAUNCH();
        if (rc) return rc;
    }
    return 0;
}

GB_EXPORT int gb_hcx_pre(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                         const void* inject_w, void* y_out, float* inj_out, void* scratch, unsigned int hidden_size,
                         unsigned int hc_mult, unsigned int rank, float eps, unsigned int num_tokens, void* stream) {
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult,
                        rank, eps, num_tokens, stream, false, nullptr);
}

GB_EXPORT int gb_hcx_pre_ref(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                             const void* inject_w, void* y_out, float* inj_out, void* scratch,
                             unsigned int hidden_size, unsigned int hc_mult, unsigned int rank, float eps,
                             unsigned int num_tokens, void* stream) {
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult,
                        rank, eps, num_tokens, stream, true, nullptr);
}

// z-dump entry points (debug aids, not a production path): run the new/ref k2 with dbg_z writes into `z_out`
// ([hc * nt, H] floats) and the same P/ss machinery, so the test can compare the pre-sigmoid sums bit for bit.
GB_EXPORT int gb_hcx_pre_z(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                           const void* inject_w, void* y_out, float* inj_out, void* scratch, float* z_out,
                           unsigned int hidden_size, unsigned int hc_mult, unsigned int rank, float eps,
                           unsigned int num_tokens, void* stream) {
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult,
                        rank, eps, num_tokens, stream, false, z_out);
}

GB_EXPORT int gb_hcx_pre_z_ref(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                               const void* inject_w, void* y_out, float* inj_out, void* scratch, float* z_out,
                               unsigned int hidden_size, unsigned int hc_mult, unsigned int rank, float eps,
                               unsigned int num_tokens, void* stream) {
    return hcx_pre_impl(streams, hc_norm_w, down_w, up_w, inject_w, y_out, inj_out, scratch, hidden_size, hc_mult,
                        rank, eps, num_tokens, stream, true, z_out);
}
