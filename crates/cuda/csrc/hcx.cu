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
static int hcx_pre_impl(const void* streams, const void* hc_norm_w, const void* down_w, const void* up_w,
                        const void* inject_w, void* y_out, float* inj_out, void* scratch, unsigned int hidden_size,
                        unsigned int hc_mult, unsigned int rank, float eps, unsigned int num_tokens, void* stream,
                        bool ref, float* dbg_z) {
    if (num_tokens == 0) return 0;
    const int H = (int)hidden_size, hc = (int)hc_mult, rk = (int)rank;
    if (H % 256 || rk % 8 || hc > 8) return (int)cudaErrorInvalidValue;
    const int ninj = inject_w ? hc : 0;
    const int R = rk + ninj;
    cudaStream_t st = gb_stream(stream);
    float* P = static_cast<float*>(scratch);
    float* ss = P + (size_t)TMAX * R * hc;
    for (int t0 = 0; t0 < (int)num_tokens; t0 += TMAX) {
        const int nt = min(TMAX, (int)num_tokens - t0);
        const int jobs = R * hc + nt * hc;
        const bool use_k1b = !ref;
        const bool use_k2b = !ref;
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
