// gb-nvcc: --expt-relaxed-constexpr --fmad=false
// The routed-expert MoE block: grouping, gather, and the weighted scatter-add. DeepSeek-V4.1-Flash, top-6 over 384 routed
// experts, hidden 5120, expert width 2304, fp16 activations.
//
// ---------------------------------------------------------------------------------------------------------------
// WHAT THIS REPLACES
//
// The routed experts used to run as a host-driven loop: the picks came down, the host scanned 384 experts per row,
// gathered each expert's rows with a `cudaMemcpy`-shaped gather, ran three GEMMs, downloaded the result and
// accumulated in f32 on the CPU. That is a sync and a round trip per expert per layer. Here every step that touches
// a row is a kernel on the device; the only host-visible number is the 385-entry `offsets` array (one 1,540-byte
// copy per layer, section 4).
//
// ---------------------------------------------------------------------------------------------------------------
// THE THREE KERNELS
//
//   group    picks [slots] i32, wts [slots] f16  ->  counts [E], offsets [E+1], perm [slots], pinv [slots], wpack [slots]
//            slots = rows * topk. `perm[p]` is the slot at packed position `p` and `pinv[s]` its inverse, so the
//            packed order is by ascending expert id and, inside an expert, by ascending slot. One block; the
//            histogram is a shared-memory array of E ints (E = 384 here, 1,536 B).
//   gather   x [rows, H] f16, perm  ->  xp [slots, H] f16,  xp[p] = x[perm[p] / topk]
//            (a slot is `r * topk + k`, so its row is `perm[p] / topk` — no division table, one shift.)
//   scatter  shared [rows, H] f16, yp [slots, H] f16, pinv, wpack  ->  out [rows, H] f32 and [rows, H] f16
//            out[r][d] = shared[r][d] + sum_k wpack[pinv[r*topk + k]] * yp[pinv[r*topk + k]][d]
//
// The per-expert GEMMs between `gather` and `scatter` are launched from the host, one segment
// `[offsets[e], offsets[e+1])` at a time, because each expert carries its own trellis width and codebook (the pack is
// mixed-K: w1/w3 of one expert can differ from each other and from w2). Their count and width are the only thing the
// host loop needs, and both come from the single `offsets` copy.
//
// ---------------------------------------------------------------------------------------------------------------
// NUMERICS
//
// Activations are fp16 (the EXL3 gemv's native operand), the accumulation is fp32, and the scatter adds the fp16
// product with an explicit `fmaf`. `--fmad=false` keeps the swiglu's `a * b` and the silu's division as two
// roundings, so the host mirror in `src/dsv4_moe.rs` (`moe_scatter_host`) is exact and the test can compare bits.
// `out[r][d]` is read once and written once by the same thread: the in-place f16 output is not a hazard.
#include <type_traits>

#include "common.cuh"
#include "exl3_linear_decode.cuh"
#include "exl3_mma.cuh"

namespace {

typedef uint16_t u16;
typedef uint32_t u32;

constexpr int ERR_INVALID = 1;  // cudaErrorInvalidValue
constexpr int MAX_EXPERTS = 1024;

// counts/offsets/perm/pinv/wpack from the router's picks. blockDim.x covers `slots` in one pass for every shape this
// pack runs (8 x 6 = 48 decode, 128 x 6 = 768 prefill); the grid is one block, so the shared histogram needs no
// atomics beyond the block's own.
__global__ void group_k(const int* picks, const u16* wts, int* counts, int* offsets, int* perm, int* pinv, u16* wpack,
                        int slots, int n_exp) {
    extern __shared__ int sh[];
    const int t = threadIdx.x, nt = blockDim.x;
    for (int e = t; e < n_exp; e += nt) sh[e] = 0;
    __syncthreads();
    for (int i = t; i < slots; i += nt) {
        const int e = picks[i];
        if (e >= 0 && e < n_exp) atomicAdd(&sh[e], 1);
    }
    __syncthreads();
    __shared__ int sh_total_packed;
    if (t < 32) {
        int acc = 0;
        for (int base = 0; base < n_exp; base += 32) {
            const int e = base + t;
            const int c = (e < n_exp) ? sh[e] : 0;
            int scan_c = c;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const int n = __shfl_up_sync(0xffffffffu, scan_c, d);
                if (t >= d) scan_c += n;
            }
            const int off = acc + scan_c - c;
            if (e < n_exp) {
                counts[e] = c;
                offsets[e] = off;
                sh[e] = off;
            }
            acc += __shfl_sync(0xffffffffu, scan_c, 31);
        }
        if (t == 0) {
            offsets[n_exp] = acc;
            sh_total_packed = acc;
        }
    }
    __syncthreads();
    const int total_packed = sh_total_packed;
    for (int p = total_packed + t; p < slots; p += nt) {
        perm[p] = -1;
        wpack[p] = 0;
    }
    if (t == 0) {
        for (int i = 0; i < slots; ++i) {
            const int e = picks[i];
            if (e < 0 || e >= n_exp) {
                pinv[i] = -1;
                continue;
            }
            const int p = sh[e]++;
            perm[p] = i;
            pinv[i] = p;
            wpack[p] = wts[i];
        }
    }
}

// xp[p][:] = x[perm[p] / topk][:], one block per packed slot, 8 halves a thread.
constexpr int GVT = 128;  // threads a block
__global__ void gather_k(const __half* __restrict__ x, const int* __restrict__ perm, __half* __restrict__ xp, int topk,
                         int h) {
    const int p = blockIdx.x;
    const int s = perm[p];
    if (s < 0) return;
    const int row = s / topk;
    const __half* src = x + (size_t)row * h;
    __half* dst = xp + (size_t)p * h;
    const int n8 = h / 8;
    for (int i = blockIdx.y * GVT + threadIdx.x; i < n8; i += GVT * gridDim.y) {
        *(uint4*)(dst + (size_t)i * 8) = *(const uint4*)(src + (size_t)i * 8);
    }
}

// out[r][d] = shared[r][d] + sum_k w[pinv[r*topk+k]] * yp[pinv[r*topk+k]][d], fp32 accumulate, both outputs written.
__global__ void scatter_k(const __half* __restrict__ shared, const __half* __restrict__ yp,
                          const int* __restrict__ pinv, const __half* __restrict__ wpack, float* __restrict__ out_f32,
                          __half* __restrict__ out_f16, int topk, int h) {
    const int r = blockIdx.x;
    const int* pv = pinv + (size_t)r * topk;
    if (topk == 6 && (h & 3) == 0) {
        int p0 = pv[0], p1 = pv[1], p2 = pv[2], p3 = pv[3], p4 = pv[4], p5 = pv[5];
        float w0 = (p0 >= 0) ? __half2float(wpack[p0]) : 0.0f;
        float w1 = (p1 >= 0) ? __half2float(wpack[p1]) : 0.0f;
        float w2 = (p2 >= 0) ? __half2float(wpack[p2]) : 0.0f;
        float w3 = (p3 >= 0) ? __half2float(wpack[p3]) : 0.0f;
        float w4 = (p4 >= 0) ? __half2float(wpack[p4]) : 0.0f;
        float w5 = (p5 >= 0) ? __half2float(wpack[p5]) : 0.0f;
        for (int d0 = (blockIdx.y * blockDim.x + threadIdx.x) * 4; d0 < h; d0 += gridDim.y * blockDim.x * 4) {
            const uint2 us = *reinterpret_cast<const uint2*>(shared + (size_t)r * h + d0);
            const float2 sa = __half22float2(*reinterpret_cast<const __half2*>(&us.x));
            const float2 sb = __half22float2(*reinterpret_cast<const __half2*>(&us.y));
            float acc[4] = {sa.x, sa.y, sb.x, sb.y};
            const int ps[6] = {p0, p1, p2, p3, p4, p5};
            const float ws[6] = {w0, w1, w2, w3, w4, w5};
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                if (ps[k] < 0) continue;
                const uint2 uy = __ldg(reinterpret_cast<const uint2*>(yp + (size_t)ps[k] * h + d0));
                const float2 ya = __half22float2(*reinterpret_cast<const __half2*>(&uy.x));
                const float2 yb = __half22float2(*reinterpret_cast<const __half2*>(&uy.y));
                acc[0] = fmaf(ws[k], ya.x, acc[0]);
                acc[1] = fmaf(ws[k], ya.y, acc[1]);
                acc[2] = fmaf(ws[k], yb.x, acc[2]);
                acc[3] = fmaf(ws[k], yb.y, acc[3]);
            }
            *reinterpret_cast<float4*>(out_f32 + (size_t)r * h + d0) = make_float4(acc[0], acc[1], acc[2], acc[3]);
            const __half2 ha = __floats2half2_rn(acc[0], acc[1]);
            const __half2 hb = __floats2half2_rn(acc[2], acc[3]);
            *reinterpret_cast<uint2*>(out_f16 + (size_t)r * h + d0) =
                make_uint2(*reinterpret_cast<const uint32_t*>(&ha), *reinterpret_cast<const uint32_t*>(&hb));
        }
        return;
    }
    for (int d = blockIdx.y * blockDim.x + threadIdx.x; d < h; d += gridDim.y * blockDim.x) {
        float acc = __half2float(shared[(size_t)r * h + d]);
        for (int k = 0; k < topk; ++k) {
            const int p = pv[k];
            if (p < 0) continue;
            acc = fmaf(__half2float(wpack[p]), __half2float(yp[(size_t)p * h + d]), acc);
        }
        out_f32[(size_t)r * h + d] = acc;
        out_f16[(size_t)r * h + d] = __float2half(acc);
    }
}

static inline bool al16(const void* p) { return ((uintptr_t)p & 15u) == 0; }

}  // namespace

// counts [E] i32, offsets [E+1] i32, perm [slots] i32, pinv [slots] i32, wpack [slots] f16 from picks [slots] i32 and
// wts [slots] f16. One block of min(1024, max(32, slots rounded up)) threads.
GB_EXPORT int gb_dsv4_moe_group(const int* picks, const u16* wts, int* counts, int* offsets, int* perm, int* pinv,
                                u16* wpack, int slots, int n_exp, void* stream) {
    if (slots < 0 || n_exp <= 0 || n_exp > MAX_EXPERTS) return ERR_INVALID;
    if (!al16(picks) || !al16(wts) || !al16(counts) || !al16(offsets) || !al16(perm) || !al16(pinv) || !al16(wpack))
        return ERR_INVALID;
    if (slots == 0) return 0;
    int threads = 32;
    while (threads < slots && threads < 1024) threads <<= 1;
    const size_t shbytes = (size_t)n_exp * sizeof(int);
    group_k<<<1, threads, shbytes, gb_stream(stream)>>>(picks, wts, counts, offsets, perm, pinv, wpack, slots, n_exp);
    return GB_CHECK_LAUNCH();
}

// xp [slots, h] f16 = x [rows, h] gathered by `perm` (slot -> packed position). grid (slots, g).
GB_EXPORT int gb_dsv4_moe_gather(const void* x, const int* perm, void* xp, int slots, int topk, int h, void* stream) {
    if (slots < 0 || topk <= 0 || h <= 0 || (h % 8) != 0) return ERR_INVALID;
    if (!al16(x) || !al16(perm) || !al16(xp)) return ERR_INVALID;
    if (slots == 0) return 0;
    const int n8 = h / 8;
    int g = (n8 + GVT - 1) / GVT;
    if (g > 8) g = 8;
    gather_k<<<dim3((unsigned)slots, (unsigned)g), GVT, 0, gb_stream(stream)>>>((const __half*)x, perm, (__half*)xp, topk,
                                                                              h);
    return GB_CHECK_LAUNCH();
}

// out [rows, h] f32 and f16 = shared [rows, h] f16 + sum_k w[pinv[r*topk+k]] yp[pinv[r*topk+k]][:] f16.
GB_EXPORT int gb_dsv4_moe_scatter(const void* shared, const void* yp, const int* pinv, const void* wpack, void* out_f32,
                                  void* out_f16, int rows, int topk, int h, void* stream) {
    if (rows < 0 || topk <= 0 || h <= 0) return ERR_INVALID;
    if (!al16(shared) || !al16(yp) || !al16(pinv) || !al16(wpack) || !al16(out_f32) || !al16(out_f16)) return ERR_INVALID;
    if (rows == 0) return 0;
    scatter_k<<<(unsigned)rows, 256, 0, gb_stream(stream)>>>((const __half*)shared, (const __half*)yp, pinv,
                                                             (const __half*)wpack, (float*)out_f32, (__half*)out_f16, topk,
                                                             h);
    return GB_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------------------------------------------
// Fast resident grouped EXL3 MoE pipeline (zero host sync, 6 launches per layer for all active experts)
// ---------------------------------------------------------------------------------------------------------------

using namespace gbx;

struct Dsv4ExpertDesc {
    uint64_t w1_trellis;
    uint64_t w1_suh;
    uint64_t w1_svh;
    int32_t  w1_bits;
    int32_t  w3_bits;
    uint64_t w3_trellis;
    uint64_t w3_suh;
    uint64_t w3_svh;
    int32_t  w2_bits;
    int32_t  pad;
    uint64_t w2_trellis;
    uint64_t w2_suh;
    uint64_t w2_svh;
};

namespace {

__device__ __forceinline__ void load4h(const __half* p, float (&v)[4]) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

__device__ __forceinline__ void store4h(__half* p, const float (&v)[4]) {
    __half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
}

template <int K2>
__host__ __device__ constexpr bool fast_step_shuffled() {
    return K2 == 2 || K2 == 4 || K2 == 6 || K2 == 8;
}

template <int K2>
__host__ __device__ constexpr int fast_step_regs() {
    return fast_step_shuffled<K2>() ? tile_words<K2>() / 4 : 8 * lane_words<K2>();
}

template <int K2>
__device__ __forceinline__ void fast_load_step(const uint32_t* step, int lane, uint32_t (&raw)[fast_step_regs<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (fast_step_shuffled<K2>()) {
#pragma unroll
        for (int c = 0; c < fast_step_regs<K2>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

template <int K2>
__device__ __forceinline__ void fast_step_lane_words(const uint32_t (&raw)[fast_step_regs<K2>()], int j, int lane, bool prev,
                                                     uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (K2 == 2 || K2 == 4 || K2 == 8) {
        constexpr int LPW = 8 / K2;
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;
    } else if constexpr (K2 == 6) {
        int word, offset;
        lane_start<6>(lane, word, offset);
        const int idx0 = j * 24 + word;
        const int idx1 = j * 24 + (word + 1 == 24 ? 0 : word + 1);
        switch (j) {
            case 0:
                w[0] = __shfl_sync(0xffffffffu, raw[0], idx0 & 31);
                w[1] = __shfl_sync(0xffffffffu, raw[0], idx1 & 31);
                break;
            case 1: {
                const uint32_t a0 = __shfl_sync(0xffffffffu, raw[0], idx0 & 31);
                const uint32_t b0 = __shfl_sync(0xffffffffu, raw[1], idx0 & 31);
                w[0] = (idx0 < 32) ? a0 : b0;
                const uint32_t a1 = __shfl_sync(0xffffffffu, raw[0], idx1 & 31);
                const uint32_t b1 = __shfl_sync(0xffffffffu, raw[1], idx1 & 31);
                w[1] = (idx1 < 32) ? a1 : b1;
                break;
            }
            case 2: {
                const uint32_t a0 = __shfl_sync(0xffffffffu, raw[1], idx0 & 31);
                const uint32_t b0 = __shfl_sync(0xffffffffu, raw[2], idx0 & 31);
                w[0] = (idx0 < 64) ? a0 : b0;
                const uint32_t a1 = __shfl_sync(0xffffffffu, raw[1], idx1 & 31);
                const uint32_t b1 = __shfl_sync(0xffffffffu, raw[2], idx1 & 31);
                w[1] = (idx1 < 64) ? a1 : b1;
                break;
            }
            case 3:
                w[0] = __shfl_sync(0xffffffffu, raw[2], idx0 & 31);
                w[1] = __shfl_sync(0xffffffffu, raw[2], idx1 & 31);
                break;
            case 4:
                w[0] = __shfl_sync(0xffffffffu, raw[3], idx0 & 31);
                w[1] = __shfl_sync(0xffffffffu, raw[3], idx1 & 31);
                break;
            case 5: {
                const uint32_t a0 = __shfl_sync(0xffffffffu, raw[3], idx0 & 31);
                const uint32_t b0 = __shfl_sync(0xffffffffu, raw[4], idx0 & 31);
                w[0] = (idx0 < 128) ? a0 : b0;
                const uint32_t a1 = __shfl_sync(0xffffffffu, raw[3], idx1 & 31);
                const uint32_t b1 = __shfl_sync(0xffffffffu, raw[4], idx1 & 31);
                w[1] = (idx1 < 128) ? a1 : b1;
                break;
            }
            case 6: {
                const uint32_t a0 = __shfl_sync(0xffffffffu, raw[4], idx0 & 31);
                const uint32_t b0 = __shfl_sync(0xffffffffu, raw[5], idx0 & 31);
                w[0] = (idx0 < 160) ? a0 : b0;
                const uint32_t a1 = __shfl_sync(0xffffffffu, raw[4], idx1 & 31);
                const uint32_t b1 = __shfl_sync(0xffffffffu, raw[5], idx1 & 31);
                w[1] = (idx1 < 160) ? a1 : b1;
                break;
            }
            default:
                w[0] = __shfl_sync(0xffffffffu, raw[5], idx0 & 31);
                w[1] = __shfl_sync(0xffffffffu, raw[5], idx1 & 31);
                break;
        }
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

__global__ void group_fast_k(const int* __restrict__ picks, const u16* __restrict__ wts,
                             int* __restrict__ counts, int* __restrict__ offsets,
                             int* __restrict__ perm, int* __restrict__ pinv, u16* __restrict__ wpack,
                             int* __restrict__ active_e, int* __restrict__ n_active, int* __restrict__ slot_e,
                             int slots, int n_exp, int step_r, int e_lo = 0, int e_hi = -1) {
    extern __shared__ int sh[];
    const int t = threadIdx.x, nt = blockDim.x;
    if (e_hi < 0) e_hi = n_exp;
    for (int e = t; e < n_exp; e += nt) sh[e] = 0;
    __syncthreads();
    for (int i = t; i < slots; i += nt) {
        const int e = picks[i];
        if (e >= 0 && e < n_exp) atomicAdd(&sh[e], 1);
    }
    __syncthreads();
    __shared__ int sh_total_packed;
    if (step_r < 32) {
        if (t < 32) {
            int acc = 0, na = 0;
            for (int base = 0; base < n_exp; base += 32) {
                const int e = base + t;
                const int c = (e < n_exp) ? sh[e] : 0;
                int scan_c = c;
#pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    const int n = __shfl_up_sync(0xffffffffu, scan_c, d);
                    if (t >= d) scan_c += n;
                }
                const int off = acc + scan_c - c;
                if (e < n_exp) {
                    counts[e] = c;
                    offsets[e] = off;
                    sh[e] = off;
                }
                acc += __shfl_sync(0xffffffffu, scan_c, 31);
                const bool owned = (e >= e_lo && e < e_hi);
                const uint32_t act_mask = __ballot_sync(0xffffffffu, c > 0 && owned);
                if (act_mask) {
                    if (__all_sync(0xffffffffu, !owned || c <= step_r)) {
                        if (c > 0 && owned) {
                            active_e[na + __popc(act_mask & ((1u << t) - 1u))] = e;
                        }
                        na += __popc(act_mask);
                    } else {
                        int nt_e = 0;
                        if (owned) {
                            for (int m0 = 0; m0 < c; m0 += step_r) ++nt_e;
                        }
                        int scan_t = nt_e;
#pragma unroll
                        for (int d = 1; d < 32; d <<= 1) {
                            const int n = __shfl_up_sync(0xffffffffu, scan_t, d);
                            if (t >= d) scan_t += n;
                        }
                        int pos = na + scan_t - nt_e;
                        if (owned) {
                            for (int m0 = 0; m0 < c; m0 += step_r) {
                                active_e[pos++] = e | (m0 << 12);
                            }
                        }
                        na += __shfl_sync(0xffffffffu, scan_t, 31);
                    }
                }
            }
            if (t == 0) {
                offsets[n_exp] = acc;
                n_active[0] = na;
                sh_total_packed = acc;
            }
        }
    } else if (t == 0) {
        int acc = 0, na = 0;
        for (int e = 0; e < n_exp; ++e) {
            const int c = sh[e];
            counts[e] = c;
            offsets[e] = acc;
            sh[e] = acc;
            acc += c;
        }
        offsets[n_exp] = acc;
        sh_total_packed = acc;
        for (int target_mt = (step_r >> 4); target_mt >= 1; --target_mt) {
            const int lo_r = (target_mt == 1) ? 9 : ((target_mt - 1) * 16 + 1);
            const int hi_r = target_mt * 16;
            for (int e = e_lo; e < e_hi; ++e) {
                const int c = counts[e];
                if (c <= 8) continue;
                for (int m0 = 0; m0 < c; m0 += step_r) {
                    const int rem = min(c - m0, step_r);
                    if (rem >= lo_r && rem <= hi_r) {
                        active_e[na++] = e | (m0 << 12);
                    }
                }
            }
        }
        for (int e = e_lo; e < e_hi; ++e) {
            const int c = counts[e];
            for (int m0 = 0; m0 < c; m0 += step_r) {
                const int rem = min(c - m0, step_r);
                if (rem >= 1 && rem <= 8) {
                    active_e[na++] = e | (m0 << 12);
                }
            }
        }
        n_active[0] = na;
    }
    __syncthreads();
    const int total_packed = sh_total_packed;
    for (int p = total_packed + t; p < slots; p += nt) {
        perm[p] = -1;
        wpack[p] = 0;
        slot_e[p] = -1;
    }
    if (t == 0) {
        for (int i = 0; i < slots; ++i) {
            const int e = picks[i];
            if (e < 0 || e >= n_exp) {
                pinv[i] = -1;
                continue;
            }
            const int p = sh[e]++;
            perm[p] = i;
            pinv[i] = p;
            wpack[p] = wts[i];
            slot_e[p] = e;
        }
    }
}

__global__ void __launch_bounds__(128) gather_rot_in_w13_k(const __half* __restrict__ hidden_in,
                                                           const int* __restrict__ perm,
                                                           const int* __restrict__ slot_e,
                                                           const Dsv4ExpertDesc* __restrict__ desc,
                                                           __half* __restrict__ xh_w13,
                                                           int slots,
                                                           int topk,
                                                           int H,
                                                           int e_lo = 0,
                                                           int e_hi = 0x7fffffff) {
    const int p = blockIdx.x;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= H) return;
    const int s = perm[p];
    const int e = slot_e[p];
    if (s < 0 || e < e_lo || e >= e_hi) return;
    const int row = s / topk;
    const int k = blk * 128 + 4 * lane;
    const Dsv4ExpertDesc d = desc[e];
    float vx[4], v1[4], v3[4], s1[4], s3[4];
    load4h(hidden_in + (size_t)row * H + k, vx);
    load4h(reinterpret_cast<const __half*>(d.w1_suh) + k, s1);
    load4h(reinterpret_cast<const __half*>(d.w3_suh) + k, s3);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v1[j] = vx[j] * s1[j];
        v3[j] = vx[j] * s3[j];
    }
    fwht128(v1, lane);
    fwht128(v3, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v1[j] *= HAD_SCALE;
        v3[j] *= HAD_SCALE;
    }
    store4h(xh_w13 + (size_t)p * H + k, v1);
    store4h(xh_w13 + ((size_t)slots + p) * H + k, v3);
}

template <int K2, int CB, int WK, int OUT_MODE, int MAX_R = 8>
__device__ __forceinline__ void gemv_packed_cta(const __half* __restrict__ X,
                                                const uint32_t* __restrict__ T,
                                                float* __restrict__ red,
                                                float* __restrict__ Z_base,
                                                __half* __restrict__ Y_base,
                                                const __half* __restrict__ svh_nb,
                                                const __half* __restrict__ suh2_nb,
                                                const __half* __restrict__ G_base,
                                                float limit,
                                                int cnt,
                                                int K,
                                                int N,
                                                int nb,
                                                int split,
                                                int SK) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const long long stride_k = (long long)(N >> 4) * TW;
    const uint32_t* tiles = T + (size_t)nb * 8 * TW;
    bool prev = false;
    if constexpr (K2 == 2 || K2 == 4 || K2 == 8) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    if constexpr (MAX_R <= 8) {
        const int ra = (g < cnt) ? g : 0;
        const __half* x0 = X + (size_t)ra * K;

        float acc[8][2][2];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                acc[i][h][0] = 0.f;
                acc[i][h][1] = 0.f;
            }

        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        constexpr bool DOUBLE_BUF = fast_step_shuffled<K2>();
        constexpr int SR = DOUBLE_BUF ? fast_step_regs<K2>() : 1;
        uint32_t cur[SR], nxt[SR];
        if constexpr (DOUBLE_BUF) fast_load_step<K2>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            const uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            const uint32_t a2 = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            if constexpr (DOUBLE_BUF) {
                if (i + 1 < per_warp) fast_load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (DOUBLE_BUF) {
                    fast_step_lane_words<K2>(cur, j, lane, prev, w);
                } else {
                    ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                }
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                {
                    float d2 = 0.f, d3 = 0.f;
                    const uint32_t z = 0u;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                 "{%0,%1,%2,%3};\n"
                                 : "+f"(acc[j][0][0]), "+f"(acc[j][0][1]), "+f"(d2), "+f"(d3)
                                 : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b0[0]), "r"(b0[1]));
                }
                {
                    float d2 = 0.f, d3 = 0.f;
                    const uint32_t z = 0u;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                 "{%0,%1,%2,%3};\n"
                                 : "+f"(acc[j][1][0]), "+f"(acc[j][1][1]), "+f"(d2), "+f"(d3)
                                 : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b1[0]), "r"(b1[1]));
                }
            }
#pragma unroll
            for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
        }

        const int rn = min(cnt, MAX_R);
        if (g < rn) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = i * 16 + h * 8 + 2 * t;
                    *reinterpret_cast<float2*>(red + (warp * MAX_R + g) * 128 + col) =
                        make_float2(acc[i][h][0], acc[i][h][1]);
                }
        }
        __syncthreads();
        if constexpr (OUT_MODE == 0) {
            for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                const int r = idx >> 5, c = 4 * (idx & 31);
                float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                for (int w = 1; w < WK; ++w) {
                    const float4 q = *reinterpret_cast<const float4*>(red + (w * MAX_R + r) * 128 + c);
                    s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                }
                *reinterpret_cast<float4*>(Z_base + (size_t)r * N + c) = s;
            }
        } else {
            const int c = 4 * lane;
            float sv[4];
            load4h(svh_nb + c, sv);
            for (int r = warp; r < rn; r += WK) {
                float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                for (int w = 1; w < WK; ++w) {
                    const float4 q = *reinterpret_cast<const float4*>(red + (w * MAX_R + r) * 128 + c);
                    s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                }
                float v[4] = {s.x, s.y, s.z, s.w};
                fwht128(v, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * sv[j];
                store4h(Y_base + (size_t)r * N + c, v);
            }
        }
    } else {
        for (int m0 = 0; m0 < cnt; m0 += 8) {
            const int ra = (m0 + g < cnt) ? (m0 + g) : m0;
            const __half* x0 = X + (size_t)ra * K;

            float acc[8][2][2];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    acc[i][h][0] = 0.f;
                    acc[i][h][1] = 0.f;
                }

            const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
            constexpr bool DOUBLE_BUF = fast_step_shuffled<K2>();
            constexpr int SR = DOUBLE_BUF ? fast_step_regs<K2>() : 1;
            uint32_t cur[SR], nxt[SR];
            if constexpr (DOUBLE_BUF) fast_load_step<K2>(tile, lane, cur);
#pragma unroll 1
            for (int i = 0; i < per_warp; ++i) {
                const int kt = kt0 + i;
                const uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
                const uint32_t a2 = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
                if constexpr (DOUBLE_BUF) {
                    if (i + 1 < per_warp) fast_load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    uint32_t w[LW];
                    if constexpr (DOUBLE_BUF) {
                        fast_step_lane_words<K2>(cur, j, lane, prev, w);
                    } else {
                        ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                    }
                    uint32_t b0[2], b1[2];
                    decode_lane<K2, CB>(w, lane, b0, b1);
                    {
                        float d2 = 0.f, d3 = 0.f;
                        const uint32_t z = 0u;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                     "{%0,%1,%2,%3};\n"
                                     : "+f"(acc[j][0][0]), "+f"(acc[j][0][1]), "+f"(d2), "+f"(d3)
                                     : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b0[0]), "r"(b0[1]));
                    }
                    {
                        float d2 = 0.f, d3 = 0.f;
                        const uint32_t z = 0u;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                     "{%0,%1,%2,%3};\n"
                                     : "+f"(acc[j][1][0]), "+f"(acc[j][1][1]), "+f"(d2), "+f"(d3)
                                     : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b1[0]), "r"(b1[1]));
                    }
                }
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }

            const int rn = min(cnt - m0, 8);
            if (m0 > 0) __syncthreads();
            if (g < rn) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * 8 + g) * 128 + col) =
                            make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();
            if constexpr (OUT_MODE == 0) {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * 8 + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z_base + (size_t)(m0 + r) * N + c) = s;
                }
            } else if constexpr (OUT_MODE == 1 || OUT_MODE == 3) {
                const int c = 4 * lane;
                const int out_stride = (OUT_MODE == 3) ? 128 : N;
                float sv[4];
                load4h(svh_nb + c, sv);
                for (int r = warp; r < rn; r += WK) {
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * 8 + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    fwht128(v, lane);
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * sv[j];
                    store4h(Y_base + (size_t)(m0 + r) * out_stride + c, v);
                }
            } else {
                const int c = 4 * lane;
                float sv[4], sd4[4];
                load4h(svh_nb + c, sv);
                load4h(suh2_nb + c, sd4);
                for (int r = warp; r < rn; r += WK) {
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * 8 + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    float uv[4] = {s.x, s.y, s.z, s.w};
                    fwht128(uv, lane);
                    float gv[4], v[4];
                    load4h(G_base + (size_t)(m0 + r) * 128 + c, gv);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float g_val = gv[j];
                        float u_val = __half2float(__float2half(uv[j] * HAD_SCALE * sv[j]));
                        if (limit > 0.f) {
                            u_val = fminf(fmaxf(u_val, -limit), limit);
                            g_val = fminf(g_val, limit);
                        }
                        const float act_val = __half2float(__float2half((g_val / (1.0f + expf(-g_val))) * u_val));
                        v[j] = act_val * sd4[j];
                    }
                    fwht128(v, lane);
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
                    store4h(Y_base + (size_t)(m0 + r) * N + c, v);
                }
            }
        }
    }
}

template <int MAX_R, bool FUSE_W13 = false>
__global__ void __launch_bounds__(128) grouped_gemv_w13_k(const __half* __restrict__ xh_w13,
                                                          const Dsv4ExpertDesc* __restrict__ desc,
                                                          const int* __restrict__ active_e,
                                                          const int* __restrict__ n_active,
                                                          const int* __restrict__ offsets,
                                                          const int* __restrict__ counts,
                                                          float* __restrict__ z_w13,
                                                          __half* __restrict__ xh_w2,
                                                          float limit,
                                                          int slots,
                                                          int H,
                                                          int I,
                                                          int SK) {
    const int u = blockIdx.z;
    if (u >= n_active[0]) return;
    constexpr int RED_R = (MAX_R <= 8) ? MAX_R : 8;
    __shared__ __align__(16) float red[4 * RED_R * 128];
    __shared__ __align__(16) __half sh_gate[FUSE_W13 ? (MAX_R * 128) : 8];
    const int packed = active_e[u];
    const int e = packed & 0xFFFu;
    const int m0_tile = packed >> 12;
    const int a = offsets[e] + m0_tile;
    const int cnt = min(counts[e] - m0_tile, MAX_R);
    const int nb = blockIdx.x;
    const Dsv4ExpertDesc d = desc[e];
    if constexpr (FUSE_W13) {
        const uint32_t* T1 = reinterpret_cast<const uint32_t*>(d.w1_trellis);
        const __half* X1 = xh_w13 + (size_t)a * H;
        const __half* svh1 = reinterpret_cast<const __half*>(d.w1_svh) + (size_t)nb * 128;
        switch (d.w1_bits) {
            case 1: gemv_packed_cta<2, 2, 4, 3, MAX_R>(X1, T1, red, nullptr, sh_gate, svh1, nullptr, nullptr, 0.f, cnt, H, I, nb, 0, 1); break;
            case 2: gemv_packed_cta<4, 2, 4, 3, MAX_R>(X1, T1, red, nullptr, sh_gate, svh1, nullptr, nullptr, 0.f, cnt, H, I, nb, 0, 1); break;
            case 3: gemv_packed_cta<6, 2, 4, 3, MAX_R>(X1, T1, red, nullptr, sh_gate, svh1, nullptr, nullptr, 0.f, cnt, H, I, nb, 0, 1); break;
            case 4: gemv_packed_cta<8, 2, 4, 3, MAX_R>(X1, T1, red, nullptr, sh_gate, svh1, nullptr, nullptr, 0.f, cnt, H, I, nb, 0, 1); break;
            default: break;
        }
        __syncthreads();
        const uint32_t* T3 = reinterpret_cast<const uint32_t*>(d.w3_trellis);
        const __half* X3 = xh_w13 + ((size_t)slots + a) * H;
        const __half* svh3 = reinterpret_cast<const __half*>(d.w3_svh) + (size_t)nb * 128;
        const __half* suh2 = reinterpret_cast<const __half*>(d.w2_suh) + (size_t)nb * 128;
        __half* Y2_base = xh_w2 + (size_t)a * I + (size_t)nb * 128;
        switch (d.w3_bits) {
            case 1: gemv_packed_cta<2, 2, 4, 2, MAX_R>(X3, T3, red, nullptr, Y2_base, svh3, suh2, sh_gate, limit, cnt, H, I, nb, 0, 1); break;
            case 2: gemv_packed_cta<4, 2, 4, 2, MAX_R>(X3, T3, red, nullptr, Y2_base, svh3, suh2, sh_gate, limit, cnt, H, I, nb, 0, 1); break;
            case 3: gemv_packed_cta<6, 2, 4, 2, MAX_R>(X3, T3, red, nullptr, Y2_base, svh3, suh2, sh_gate, limit, cnt, H, I, nb, 0, 1); break;
            case 4: gemv_packed_cta<8, 2, 4, 2, MAX_R>(X3, T3, red, nullptr, Y2_base, svh3, suh2, sh_gate, limit, cnt, H, I, nb, 0, 1); break;
            default: break;
        }
        return;
    }
    const int split = blockIdx.y % SK, mat = blockIdx.y / SK;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? d.w3_trellis : d.w1_trellis);
    const int bits = mat ? d.w3_bits : d.w1_bits;
    const __half* X = xh_w13 + ((size_t)mat * slots + a) * H;
    float* Z_base = z_w13 + ((size_t)(mat * SK + split) * slots + a) * I + (size_t)nb * 128;
    switch (bits) {
        case 1: gemv_packed_cta<2, 2, 4, 0, MAX_R>(X, T, red, Z_base, nullptr, nullptr, nullptr, nullptr, 0.f, cnt, H, I, nb, split, SK); break;
        case 2: gemv_packed_cta<4, 2, 4, 0, MAX_R>(X, T, red, Z_base, nullptr, nullptr, nullptr, nullptr, 0.f, cnt, H, I, nb, split, SK); break;
        case 3: gemv_packed_cta<6, 2, 4, 0, MAX_R>(X, T, red, Z_base, nullptr, nullptr, nullptr, nullptr, 0.f, cnt, H, I, nb, split, SK); break;
        case 4: gemv_packed_cta<8, 2, 4, 0, MAX_R>(X, T, red, Z_base, nullptr, nullptr, nullptr, nullptr, 0.f, cnt, H, I, nb, split, SK); break;
        default: break;
    }
}

__global__ void __launch_bounds__(128) rot_out_swiglu_rot_in_w2_k(const float* __restrict__ z_w13,
                                                                  const int* __restrict__ slot_e,
                                                                  const Dsv4ExpertDesc* __restrict__ desc,
                                                                  __half* __restrict__ xh_w2,
                                                                  int slots,
                                                                  int I,
                                                                  int SK,
                                                                  float limit,
                                                                  int e_lo = 0,
                                                                  int e_hi = 0x7fffffff) {
    const int p = blockIdx.x;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= I) return;
    const int e = slot_e[p];
    if (e < e_lo || e >= e_hi) return;
    const Dsv4ExpertDesc d = desc[e];
    const int col = blk * 128 + 4 * lane;

    const float* zg = z_w13 + (size_t)p * I + col;
    const float* zu = z_w13 + ((size_t)SK * slots + p) * I + col;
    float4 sg = *reinterpret_cast<const float4*>(zg);
    float4 su = *reinterpret_cast<const float4*>(zu);
    for (int s = 1; s < SK; ++s) {
        const float4 a = *reinterpret_cast<const float4*>(zg + (size_t)s * slots * I);
        const float4 b = *reinterpret_cast<const float4*>(zu + (size_t)s * slots * I);
        sg.x += a.x; sg.y += a.y; sg.z += a.z; sg.w += a.w;
        su.x += b.x; su.y += b.y; su.z += b.z; su.w += b.w;
    }
    float gv[4] = {sg.x, sg.y, sg.z, sg.w};
    float uv[4] = {su.x, su.y, su.z, su.w};
    fwht128(gv, lane);
    fwht128(uv, lane);

    float sg4[4], su4[4], sd4[4], v[4];
    load4h(reinterpret_cast<const __half*>(d.w1_svh) + col, sg4);
    load4h(reinterpret_cast<const __half*>(d.w3_svh) + col, su4);
    load4h(reinterpret_cast<const __half*>(d.w2_suh) + col, sd4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float g_val = __half2float(__float2half(gv[j] * HAD_SCALE * sg4[j]));
        float u_val = __half2float(__float2half(uv[j] * HAD_SCALE * su4[j]));
        if (limit > 0.f) {
            u_val = fminf(fmaxf(u_val, -limit), limit);
            g_val = fminf(g_val, limit);
        }
        const float act_val = __half2float(__float2half((g_val / (1.0f + expf(-g_val))) * u_val));
        v[j] = act_val * sd4[j];
    }
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4h(xh_w2 + (size_t)p * I + col, v);
}

__global__ void __launch_bounds__(128) swiglu_rot_in_w2_half_k(const __half* __restrict__ gu_w13,
                                                               const int* __restrict__ slot_e,
                                                               const Dsv4ExpertDesc* __restrict__ desc,
                                                               __half* __restrict__ xh_w2,
                                                               int slots,
                                                               int I,
                                                               float limit,
                                                               int e_lo = 0,
                                                               int e_hi = 0x7fffffff) {
    const int p = blockIdx.x;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= I) return;
    const int e = slot_e[p];
    if (e < e_lo || e >= e_hi) return;
    const Dsv4ExpertDesc d = desc[e];
    const int col = blk * 128 + 4 * lane;

    float gv[4], uv[4], sd4[4], v[4];
    load4h(gu_w13 + (size_t)p * I + col, gv);
    load4h(gu_w13 + ((size_t)slots + p) * I + col, uv);
    load4h(reinterpret_cast<const __half*>(d.w2_suh) + col, sd4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float g_val = gv[j];
        float u_val = uv[j];
        if (limit > 0.f) {
            u_val = fminf(fmaxf(u_val, -limit), limit);
            g_val = fminf(g_val, limit);
        }
        const float act_val = __half2float(__float2half((g_val / (1.0f + expf(-g_val))) * u_val));
        v[j] = act_val * sd4[j];
    }
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4h(xh_w2 + (size_t)p * I + col, v);
}

template <int K2, int CB>
__device__ __noinline__ void gemv_packed_cta_wk8(const __half* __restrict__ X,
                                                 const uint32_t* __restrict__ T,
                                                 float* __restrict__ red,
                                                 __half* __restrict__ Y_base,
                                                 const __half* __restrict__ svh_nb,
                                                 int cnt,
                                                 int K,
                                                 int N,
                                                 int nb) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr int WK = 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / WK;
    const int kt0 = warp * per_warp;
    const long long stride_k = (long long)(N >> 4) * TW;
    const uint32_t* tiles = T + (size_t)nb * 8 * TW;
    bool prev = false;
    if constexpr (K2 == 2 || K2 == 4 || K2 == 8) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    const int ra = (g < cnt) ? g : 0;
    const __half* x0 = X + (size_t)ra * K;

    float acc[8][2][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            acc[i][h][0] = 0.f;
            acc[i][h][1] = 0.f;
        }

    const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
    constexpr bool DOUBLE_BUF = fast_step_shuffled<K2>();
    constexpr int SR = DOUBLE_BUF ? fast_step_regs<K2>() : 1;
    uint32_t cur[SR], nxt[SR];
    if constexpr (DOUBLE_BUF) fast_load_step<K2>(tile, lane, cur);
#pragma unroll 1
    for (int i = 0; i < per_warp; ++i) {
        const int kt = kt0 + i;
        const uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
        const uint32_t a2 = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
        if constexpr (DOUBLE_BUF) {
            if (i + 1 < per_warp) fast_load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t w[LW];
            if constexpr (DOUBLE_BUF) {
                fast_step_lane_words<K2>(cur, j, lane, prev, w);
            } else {
                ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
            }
            uint32_t b0[2], b1[2];
            decode_lane<K2, CB>(w, lane, b0, b1);
            const uint32_t z = 0u;
            float d2 = 0.f, d3 = 0.f;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(acc[j][0][0]), "+f"(acc[j][0][1]), "+f"(d2), "+f"(d3)
                         : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b0[0]), "r"(b0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(acc[j][1][0]), "+f"(acc[j][1][1]), "+f"(d2), "+f"(d3)
                         : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b1[0]), "r"(b1[1]));
        }
#pragma unroll
        for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
    }

    if (g < cnt) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                *reinterpret_cast<float2*>(red + (warp * 8 + g) * 128 + col) =
                    make_float2(acc[i][h][0], acc[i][h][1]);
            }
    }
    __syncthreads();
    if (warp < cnt) {
        const int r = warp;
        const int c = 4 * lane;
        float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
        for (int w = 1; w < WK; ++w) {
            const float4 q = *reinterpret_cast<const float4*>(red + (w * 8 + r) * 128 + c);
            s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
        }
        float sv[4];
        load4h(svh_nb + c, sv);
        float v[4] = {s.x, s.y, s.z, s.w};
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * sv[j];
        store4h(Y_base + (size_t)r * N + c, v);
    }
}

template <int K2, int CB, int MT>
__device__ __noinline__ void mma_packed_cta(const __half* __restrict__ X,
                                            const uint32_t* __restrict__ T,
                                            unsigned char* __restrict__ ring_smem,
                                            __half* __restrict__ Y_base,
                                            const __half* __restrict__ svh_nb,
                                            int cnt,
                                            int K,
                                            int N,
                                            int nb) {
    using gbm::a_chunk;
    using gbm::a_frag_addr;
    using gbm::cp_async16;
    using gbm::cp_async16_z;
    using gbm::cp_commit;
    using gbm::cp_wait;
    using gbm::ldmatrix_x4;

    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr int kAStages = 2;
    constexpr int kWStages = (K2 <= 4) ? 5 : (K2 == 6 ? 4 : 3);
    constexpr int kAStageBytes = 16 * MT * 64 * 2;
    constexpr int kWStageBytes = 4 * 8 * TW * 4;
    constexpr int kEStride = 136;

    unsigned char* aring = ring_smem;
    unsigned char* wring = ring_smem + kAStages * kAStageBytes;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, tq = lane & 3;
    const int mt = (cnt + 15) >> 4;
    const int NS = K / 64;
    const size_t stride_k = (size_t)(N >> 4) * TW;
    const uint32_t* tiles = T + (size_t)nb * 8 * TW;

    int word0 = 0, offset0 = 0;
    lane_start<K2>(lane, word0, offset0);
    const int w0_idx = word0;
    const int w1_idx = (word0 + 1) % TW;
    const int w2_idx = (LW > 2) ? ((word0 + 2) % TW) : 0;

    constexpr int kWChunksPerKt = 8 * K2;
    constexpr int kWTotalChunks = 4 * kWChunksPerKt;
    const bool w_active = tid < kWTotalChunks;
    const int w_kt = tid / kWChunksPerKt;
    const int w_off = (tid % kWChunksPerKt) * 4;
    const uint32_t* w_src = tiles + (size_t)w_kt * stride_k + w_off;
    const int w_dst = w_kt * (8 * TW) + w_off;

    auto load_w = [&](int s) {
        if (w_active) {
            uint32_t* ts = reinterpret_cast<uint32_t*>(wring + (s % kWStages) * kWStageBytes);
            cp_async16(ts + w_dst, w_src + (size_t)s * 4 * stride_k);
        }
    };

    float acc[MT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][h][c] = 0.f;

    auto kloop = [&](auto ma_c) {
        constexpr int MA = decltype(ma_c)::value;
        constexpr int kACh = (16 * MA * 8 + 255) / 256;
        const __half* a_src[kACh];
        int a_dst[kACh];
        bool a_issue[kACh], a_valid[kACh];
#pragma unroll
        for (int j = 0; j < kACh; ++j) {
            const int i = tid + 256 * j;
            const int r = i >> 3, c = i & 7;
            a_issue[j] = (i < 16 * MA * 8);
            a_valid[j] = (r < cnt);
            a_src[j] = X + (a_issue[j] && a_valid[j] ? (size_t)r * K + c * 8 : 0);
            a_dst[j] = a_chunk(r, c);
        }
        auto load_a = [&](int s) {
            uint16_t* as = reinterpret_cast<uint16_t*>(aring + (s % kAStages) * kAStageBytes);
#pragma unroll
            for (int j = 0; j < kACh; ++j) {
                if (a_issue[j]) {
                    cp_async16_z(as + a_dst[j], a_src[j] + (a_valid[j] ? s * 64 : 0), a_valid[j]);
                }
            }
        };

        load_w(0);
        load_a(0);
        cp_commit();
#pragma unroll
        for (int ws = 1; ws < kWStages - 1; ++ws) {
            if (ws < NS) load_w(ws);
        }
        cp_commit();

#pragma unroll 1
        for (int s = 0; s < NS; ++s) {
            cp_wait<1>();
            __syncthreads();
            if (s + 1 < NS) load_a(s + 1);
            cp_commit();
            if (s + kWStages - 1 < NS) load_w(s + kWStages - 1);
            cp_commit();

            const uint16_t* as = reinterpret_cast<const uint16_t*>(aring + (s % kAStages) * kAStageBytes);
            const uint32_t* ts = reinterpret_cast<const uint32_t*>(wring + (s % kWStages) * kWStageBytes);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const uint32_t* tile_s = ts + (kk * 8 + warp) * TW;
                uint32_t w[LW];
                w[0] = tile_s[w0_idx];
                w[1] = tile_s[w1_idx];
                if constexpr (LW > 2) w[2] = tile_s[w2_idx];
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
                for (int m = 0; m < MA; ++m) {
                    uint32_t af[4];
                    ldmatrix_x4(af, a_frag_addr(as, m, kk, lane));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(acc[m][0][0]), "+f"(acc[m][0][1]), "+f"(acc[m][0][2]), "+f"(acc[m][0][3])
                                 : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(b0[0]), "r"(b0[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(acc[m][1][0]), "+f"(acc[m][1][1]), "+f"(acc[m][1][2]), "+f"(acc[m][1][3])
                                 : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(b1[0]), "r"(b1[1]));
                }
            }
        }
    };

    switch (mt) {
        case 1: kloop(std::integral_constant<int, 1>{}); break;
        case 2: if constexpr (MT >= 2) kloop(std::integral_constant<int, MT >= 2 ? 2 : 1>{}); break;
        case 3: if constexpr (MT >= 3) kloop(std::integral_constant<int, MT >= 3 ? 3 : 1>{}); break;
        default: kloop(std::integral_constant<int, MT>{}); break;
    }
    cp_wait<0>();
    __syncthreads();

    float* E = reinterpret_cast<float*>(aring);
    const int c = 4 * lane;
    float sv[4];
    load4h(svh_nb + c, sv);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m >= mt) break;
        if (m > 0) __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = 16 * warp + 8 * h + 2 * tq;
            *reinterpret_cast<float2*>(E + gq * kEStride + col) = make_float2(acc[m][h][0], acc[m][h][1]);
            *reinterpret_cast<float2*>(E + (gq + 8) * kEStride + col) = make_float2(acc[m][h][2], acc[m][h][3]);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int rl = warp + 8 * j;
            const int r = 16 * m + rl;
            if (r < cnt) {
                const float4 s = *reinterpret_cast<const float4*>(E + rl * kEStride + c);
                float v[4] = {s.x, s.y, s.z, s.w};
                fwht128(v, lane);
#pragma unroll
                for (int q = 0; q < 4; ++q) v[q] = v[q] * HAD_SCALE * sv[q];
                store4h(Y_base + (size_t)r * N + c, v);
            }
        }
    }
}

template <int K2, int CB, int MT>
__device__ __forceinline__ void proj_packed_cta(const __half* __restrict__ X,
                                                const uint32_t* __restrict__ T,
                                                unsigned char* __restrict__ ring_smem,
                                                __half* __restrict__ Y_base,
                                                const __half* __restrict__ svh_nb,
                                                int cnt,
                                                int K,
                                                int N,
                                                int nb) {
    mma_packed_cta<K2, CB, MT>(X, T, ring_smem, Y_base, svh_nb, cnt, K, N, nb);
}

template <int MT>
__global__ void __launch_bounds__(256, 2) grouped_gemm_w13_k(const __half* __restrict__ xh_w13,
                                                             const Dsv4ExpertDesc* __restrict__ desc,
                                                             const int* __restrict__ active_e,
                                                             const int* __restrict__ n_active,
                                                             const int* __restrict__ offsets,
                                                             const int* __restrict__ counts,
                                                             __half* __restrict__ gu_w13,
                                                             int slots,
                                                             int H,
                                                             int I) {
    const int u = blockIdx.z;
    if (u >= n_active[0]) return;
    constexpr int MAX_R = 16 * MT;
    __shared__ __align__(16) unsigned char ring_smem[32768];

    const int packed = active_e[u];
    const int e = packed & 0xFFFu;
    const int m0_tile = packed >> 12;
    const int a = offsets[e] + m0_tile;
    const int cnt = min(counts[e] - m0_tile, MAX_R);
    if (cnt <= 0) return;
    const int nb = blockIdx.x;
    const int mat = blockIdx.y;
    const Dsv4ExpertDesc d = desc[e];

    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? d.w3_trellis : d.w1_trellis);
    const int bits = mat ? d.w3_bits : d.w1_bits;
    const __half* svh_nb = reinterpret_cast<const __half*>(mat ? d.w3_svh : d.w1_svh) + (size_t)nb * 128;
    const __half* X = xh_w13 + ((size_t)mat * slots + a) * H;
    __half* Y_base = gu_w13 + ((size_t)mat * slots + a) * I + (size_t)nb * 128;
    switch (bits) {
        case 1: proj_packed_cta<2, 2, MT>(X, T, ring_smem, Y_base, svh_nb, cnt, H, I, nb); break;
        case 2: proj_packed_cta<4, 2, MT>(X, T, ring_smem, Y_base, svh_nb, cnt, H, I, nb); break;
        case 3: proj_packed_cta<6, 2, MT>(X, T, ring_smem, Y_base, svh_nb, cnt, H, I, nb); break;
        case 4: proj_packed_cta<8, 2, MT>(X, T, ring_smem, Y_base, svh_nb, cnt, H, I, nb); break;
        default: break;
    }
}

template <int MT>
__global__ void __launch_bounds__(256, 2) grouped_gemm_w2_k(const __half* __restrict__ xh_w2,
                                                            const Dsv4ExpertDesc* __restrict__ desc,
                                                            const int* __restrict__ active_e,
                                                            const int* __restrict__ n_active,
                                                            const int* __restrict__ offsets,
                                                            const int* __restrict__ counts,
                                                            __half* __restrict__ y,
                                                            int I,
                                                            int H) {
    const int u = blockIdx.z;
    if (u >= n_active[0]) return;
    constexpr int MAX_R = 16 * MT;
    __shared__ __align__(16) unsigned char ring_smem[32768];

    const int packed = active_e[u];
    const int e = packed & 0xFFFu;
    const int m0_tile = packed >> 12;
    const int a = offsets[e] + m0_tile;
    const int cnt = min(counts[e] - m0_tile, MAX_R);
    if (cnt <= 0) return;
    const int nb = blockIdx.x;
    const Dsv4ExpertDesc d = desc[e];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(d.w2_trellis);
    const __half* svh_nb = reinterpret_cast<const __half*>(d.w2_svh) + (size_t)nb * 128;
    const __half* X = xh_w2 + (size_t)a * I;
    __half* Y_base = y + (size_t)a * H + (size_t)nb * 128;
    switch (d.w2_bits) {
        case 1: proj_packed_cta<2, 2, MT>(X, T, ring_smem, Y_base, svh_nb, cnt, I, H, nb); break;
        case 2: proj_packed_cta<4, 2, MT>(X, T, ring_smem, Y_base, svh_nb, cnt, I, H, nb); break;
        case 3: proj_packed_cta<6, 2, MT>(X, T, ring_smem, Y_base, svh_nb, cnt, I, H, nb); break;
        case 4: proj_packed_cta<8, 2, MT>(X, T, ring_smem, Y_base, svh_nb, cnt, I, H, nb); break;
        default: break;
    }
}

template <int MAX_R>
__global__ void __launch_bounds__(128) grouped_gemv_w2_k(const __half* __restrict__ xh_w2,
                                                         const Dsv4ExpertDesc* __restrict__ desc,
                                                         const int* __restrict__ active_e,
                                                         const int* __restrict__ n_active,
                                                         const int* __restrict__ offsets,
                                                         const int* __restrict__ counts,
                                                         __half* __restrict__ y,
                                                         int I,
                                                         int H) {
    const int u = blockIdx.z;
    if (u >= n_active[0]) return;
    constexpr int RED_R = (MAX_R <= 8) ? MAX_R : 8;
    __shared__ __align__(16) float red[4 * RED_R * 128];
    const int packed = active_e[u];
    const int e = packed & 0xFFFu;
    const int m0_tile = packed >> 12;
    const int a = offsets[e] + m0_tile;
    const int cnt = min(counts[e] - m0_tile, MAX_R);
    const int nb = blockIdx.x;
    const Dsv4ExpertDesc d = desc[e];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(d.w2_trellis);
    const __half* svh_nb = reinterpret_cast<const __half*>(d.w2_svh) + (size_t)nb * 128;
    const __half* X = xh_w2 + (size_t)a * I;
    __half* Y_base = y + (size_t)a * H + (size_t)nb * 128;
    switch (d.w2_bits) {
        case 1: gemv_packed_cta<2, 2, 4, 1, MAX_R>(X, T, red, nullptr, Y_base, svh_nb, nullptr, nullptr, 0.f, cnt, I, H, nb, 0, 1); break;
        case 2: gemv_packed_cta<4, 2, 4, 1, MAX_R>(X, T, red, nullptr, Y_base, svh_nb, nullptr, nullptr, 0.f, cnt, I, H, nb, 0, 1); break;
        case 3: gemv_packed_cta<6, 2, 4, 1, MAX_R>(X, T, red, nullptr, Y_base, svh_nb, nullptr, nullptr, 0.f, cnt, I, H, nb, 0, 1); break;
        case 4: gemv_packed_cta<8, 2, 4, 1, MAX_R>(X, T, red, nullptr, Y_base, svh_nb, nullptr, nullptr, 0.f, cnt, I, H, nb, 0, 1); break;
        default: break;
    }
}

static inline size_t align256_sz(size_t x) { return (x + 255) & ~size_t(255); }

__global__ void scatter_tp_k(const __half* __restrict__ shared,
                             const __half* __restrict__ yp,
                             const __half* __restrict__ yp_peer,
                             const int* __restrict__ offsets,
                             const int* __restrict__ pinv,
                             const __half* __restrict__ wpack,
                             float* __restrict__ out_f32,
                             __half* __restrict__ out_f16,
                             int topk,
                             int h,
                             int e_lo,
                             int e_hi) {
    const int r = blockIdx.x;
    const int own_lo = offsets[e_lo];
    const int own_hi = offsets[e_hi];
    const int* pv = pinv + (size_t)r * topk;
    if (topk == 6 && (h & 3) == 0) {
        int p0 = pv[0], p1 = pv[1], p2 = pv[2], p3 = pv[3], p4 = pv[4], p5 = pv[5];
        float w0 = (p0 >= 0) ? __half2float(wpack[p0]) : 0.0f;
        float w1 = (p1 >= 0) ? __half2float(wpack[p1]) : 0.0f;
        float w2 = (p2 >= 0) ? __half2float(wpack[p2]) : 0.0f;
        float w3 = (p3 >= 0) ? __half2float(wpack[p3]) : 0.0f;
        float w4 = (p4 >= 0) ? __half2float(wpack[p4]) : 0.0f;
        float w5 = (p5 >= 0) ? __half2float(wpack[p5]) : 0.0f;
        for (int d0 = threadIdx.x * 4; d0 < h; d0 += blockDim.x * 4) {
            const uint2 us = *reinterpret_cast<const uint2*>(shared + (size_t)r * h + d0);
            const float2 sa = __half22float2(*reinterpret_cast<const __half2*>(&us.x));
            const float2 sb = __half22float2(*reinterpret_cast<const __half2*>(&us.y));
            float acc[4] = {sa.x, sa.y, sb.x, sb.y};
            const int ps[6] = {p0, p1, p2, p3, p4, p5};
            const float ws[6] = {w0, w1, w2, w3, w4, w5};
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                if (ps[k] < 0) continue;
                const __half* src = (ps[k] >= own_lo && ps[k] < own_hi) ? yp : yp_peer;
                const uint2 uy = __ldg(reinterpret_cast<const uint2*>(src + (size_t)ps[k] * h + d0));
                const float2 ya = __half22float2(*reinterpret_cast<const __half2*>(&uy.x));
                const float2 yb = __half22float2(*reinterpret_cast<const __half2*>(&uy.y));
                acc[0] = fmaf(ws[k], ya.x, acc[0]);
                acc[1] = fmaf(ws[k], ya.y, acc[1]);
                acc[2] = fmaf(ws[k], yb.x, acc[2]);
                acc[3] = fmaf(ws[k], yb.y, acc[3]);
            }
            *reinterpret_cast<float4*>(out_f32 + (size_t)r * h + d0) = make_float4(acc[0], acc[1], acc[2], acc[3]);
            const __half2 ha = __floats2half2_rn(acc[0], acc[1]);
            const __half2 hb = __floats2half2_rn(acc[2], acc[3]);
            *reinterpret_cast<uint2*>(out_f16 + (size_t)r * h + d0) =
                make_uint2(*reinterpret_cast<const uint32_t*>(&ha), *reinterpret_cast<const uint32_t*>(&hb));
        }
        return;
    }
    for (int d = threadIdx.x; d < h; d += blockDim.x) {
        float acc = __half2float(shared[(size_t)r * h + d]);
        for (int k = 0; k < topk; ++k) {
            const int p = pv[k];
            if (p < 0) continue;
            const __half* src = (p >= own_lo && p < own_hi) ? yp : yp_peer;
            acc = fmaf(__half2float(wpack[p]), __half2float(src[(size_t)p * h + d]), acc);
        }
        out_f32[(size_t)r * h + d] = acc;
        out_f16[(size_t)r * h + d] = __float2half(acc);
    }
}

__global__ void interleave_o_halves_k(const uint4* __restrict__ left,
                                      const uint4* __restrict__ right,
                                      uint4* __restrict__ dst,
                                      int rows,
                                      int half_u4) {
    const int r = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || i >= half_u4) return;
    dst[(size_t)r * (2 * half_u4) + i] = left[(size_t)r * half_u4 + i];
    dst[(size_t)r * (2 * half_u4) + half_u4 + i] = right[(size_t)r * half_u4 + i];
}

}  // namespace

GB_EXPORT size_t gb_dsv4_moe_fast_ws_bytes(int slots, int h, int inter) {
    const size_t z_slots = (slots <= 96) ? (size_t)slots : 96;
    return align256_sz((size_t)slots * 4) + 256 + align256_sz((size_t)slots * 4)
         + align256_sz((size_t)2 * slots * h * 2)
         + align256_sz((size_t)4 * z_slots * inter * 4);
}

GB_EXPORT int gb_dsv4_moe_fast_experts(const void* hidden_in,
                                       void* shared_f16,
                                       const int* picks,
                                       const void* wts,
                                       void* moe_out_f32,
                                       const void* descs,
                                       int* counts,
                                       int* offsets,
                                       int* perm,
                                       int* pinv,
                                       void* wpack,
                                       void* xh_w2,
                                       void* y,
                                       void* ws,
                                       size_t ws_bytes,
                                       int rows,
                                       int topk,
                                       int h,
                                       int inter,
                                       int n_exp,
                                       int gemm_only,
                                       float limit,
                                       void* stream) {
    const int slots = rows * topk;
    if (rows <= 0 || topk <= 0 || h <= 0 || inter <= 0 || (h % 128) != 0 || (inter % 128) != 0) return ERR_INVALID;
    if (((h / 16) % 8) != 0 || ((inter / 16) % 4) != 0) return ERR_INVALID;
    const bool use_gemv = (rows <= 16) && !gemm_only;
    const size_t z_slots = use_gemv ? (size_t)slots : 0;
    const size_t min_ws = align256_sz((size_t)slots * 4) + 256 + align256_sz((size_t)slots * 4)
                        + align256_sz((size_t)2 * slots * h * 2)
                        + align256_sz(4 * z_slots * (size_t)inter * 4);
    if (ws_bytes < min_ws) return ERR_INVALID;

    cudaStream_t st = gb_stream(stream);
    char* wptr = static_cast<char*>(ws);
    int* active_e = reinterpret_cast<int*>(wptr);
    wptr += align256_sz((size_t)slots * 4);
    int* n_active = reinterpret_cast<int*>(wptr);
    wptr += 256;
    int* slot_e = reinterpret_cast<int*>(wptr);
    wptr += align256_sz((size_t)slots * 4);
    __half* xh_w13 = reinterpret_cast<__half*>(wptr);
    wptr += align256_sz((size_t)2 * slots * h * 2);
    float* z_w13 = reinterpret_cast<float*>(wptr);

    int threads = 32;
    while (threads < slots && threads < 1024) threads <<= 1;
    const size_t shbytes = (size_t)n_exp * sizeof(int);
    const int step_r = use_gemv ? ((rows == 1) ? 1 : (rows <= 4) ? 4 : 8) : 64;
    group_fast_k<<<1, threads, shbytes, st>>>(picks, (const u16*)wts, counts, offsets, perm, pinv, (u16*)wpack,
                                              active_e, n_active, slot_e, slots, n_exp, step_r);

    const Dsv4ExpertDesc* d_ptr = reinterpret_cast<const Dsv4ExpertDesc*>(descs);
    const dim3 g_rot13((unsigned)slots, (unsigned)((h / 128 + 3) / 4));
    gather_rot_in_w13_k<<<g_rot13, 128, 0, st>>>((const __half*)hidden_in, perm, slot_e, d_ptr, xh_w13, slots, topk, h);

    const int sk_13 = use_gemv ? 2 : 1;
    const unsigned max_act_e = (unsigned)min(slots, n_exp + slots / step_r);
    const dim3 g_gemv13((unsigned)(inter / 128), (unsigned)(2 * sk_13), max_act_e);
    const dim3 g_gemm13((unsigned)(inter / 128), 2u, max_act_e);
    const dim3 g_act((unsigned)slots, (unsigned)((inter / 128 + 3) / 4));
    const dim3 g_gemv2((unsigned)(h / 128), 1, max_act_e);
    if (use_gemv && rows == 1) {
        grouped_gemv_w13_k<1><<<g_gemv13, 128, 0, st>>>(xh_w13, d_ptr, active_e, n_active, offsets, counts, z_w13, nullptr, limit, slots, h, inter, sk_13);
        rot_out_swiglu_rot_in_w2_k<<<g_act, 128, 0, st>>>(z_w13, slot_e, d_ptr, (__half*)xh_w2, slots, inter, sk_13, limit);
        grouped_gemv_w2_k<1><<<g_gemv2, 128, 0, st>>>((const __half*)xh_w2, d_ptr, active_e, n_active, offsets, counts, (__half*)y, inter, h);
    } else if (use_gemv && rows <= 4) {
        grouped_gemv_w13_k<4><<<g_gemv13, 128, 0, st>>>(xh_w13, d_ptr, active_e, n_active, offsets, counts, z_w13, nullptr, limit, slots, h, inter, sk_13);
        rot_out_swiglu_rot_in_w2_k<<<g_act, 128, 0, st>>>(z_w13, slot_e, d_ptr, (__half*)xh_w2, slots, inter, sk_13, limit);
        grouped_gemv_w2_k<4><<<g_gemv2, 128, 0, st>>>((const __half*)xh_w2, d_ptr, active_e, n_active, offsets, counts, (__half*)y, inter, h);
    } else if (use_gemv && rows <= 16) {
        grouped_gemv_w13_k<8><<<g_gemv13, 128, 0, st>>>(xh_w13, d_ptr, active_e, n_active, offsets, counts, z_w13, nullptr, limit, slots, h, inter, sk_13);
        rot_out_swiglu_rot_in_w2_k<<<g_act, 128, 0, st>>>(z_w13, slot_e, d_ptr, (__half*)xh_w2, slots, inter, sk_13, limit);
        grouped_gemv_w2_k<8><<<g_gemv2, 128, 0, st>>>((const __half*)xh_w2, d_ptr, active_e, n_active, offsets, counts, (__half*)y, inter, h);
    } else {
        grouped_gemm_w13_k<4><<<g_gemm13, 256, 0, st>>>(xh_w13, d_ptr, active_e, n_active, offsets, counts, (__half*)y, slots, h, inter);
        swiglu_rot_in_w2_half_k<<<g_act, 128, 0, st>>>((const __half*)y, slot_e, d_ptr, (__half*)xh_w2, slots, inter, limit);
        grouped_gemm_w2_k<4><<<g_gemv2, 256, 0, st>>>((const __half*)xh_w2, d_ptr, active_e, n_active, offsets, counts, (__half*)y, inter, h);
    }

    const dim3 g_scat((unsigned)rows, use_gemv ? (unsigned)((h + 1023) / 1024) : 1u);
    scatter_k<<<g_scat, 256, 0, st>>>((const __half*)shared_f16, (const __half*)y, pinv, (const __half*)wpack,
                                      (float*)moe_out_f32, (__half*)shared_f16, topk, h);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_moe_fast_experts_tp_pre(const void* hidden_in,
                                              const int* picks,
                                              const void* wts,
                                              const void* descs,
                                              int* counts,
                                              int* offsets,
                                              int* perm,
                                              int* pinv,
                                              void* wpack,
                                              void* xh_w2,
                                              void* y,
                                              void* ws,
                                              size_t ws_bytes,
                                              int rows,
                                              int topk,
                                              int h,
                                              int inter,
                                              int n_exp,
                                              int e_lo,
                                              int e_hi,
                                              int gemm_only,
                                              float limit,
                                              void* stream) {
    const int slots = rows * topk;
    if (rows <= 0 || topk <= 0 || h <= 0 || inter <= 0 || (h % 128) != 0 || (inter % 128) != 0) return ERR_INVALID;
    if (((h / 16) % 8) != 0 || ((inter / 16) % 4) != 0) return ERR_INVALID;
    if (e_lo < 0 || e_hi < e_lo || e_hi > n_exp) return ERR_INVALID;
    const bool use_gemv = (rows <= 16) && !gemm_only;
    const size_t z_slots = use_gemv ? (size_t)slots : 0;
    const size_t min_ws = align256_sz((size_t)slots * 4) + 256 + align256_sz((size_t)slots * 4)
                        + align256_sz((size_t)2 * slots * h * 2)
                        + align256_sz(4 * z_slots * (size_t)inter * 4);
    if (ws_bytes < min_ws) return ERR_INVALID;

    cudaStream_t st = gb_stream(stream);
    char* wptr = static_cast<char*>(ws);
    int* active_e = reinterpret_cast<int*>(wptr);
    wptr += align256_sz((size_t)slots * 4);
    int* n_active = reinterpret_cast<int*>(wptr);
    wptr += 256;
    int* slot_e = reinterpret_cast<int*>(wptr);
    wptr += align256_sz((size_t)slots * 4);
    __half* xh_w13 = reinterpret_cast<__half*>(wptr);
    wptr += align256_sz((size_t)2 * slots * h * 2);
    float* z_w13 = reinterpret_cast<float*>(wptr);

    int threads = 32;
    while (threads < slots && threads < 1024) threads <<= 1;
    const size_t shbytes = (size_t)n_exp * sizeof(int);
    const int step_r = use_gemv ? ((rows == 1) ? 1 : (rows <= 4) ? 4 : 8) : 64;
    group_fast_k<<<1, threads, shbytes, st>>>(picks, (const u16*)wts, counts, offsets, perm, pinv, (u16*)wpack,
                                              active_e, n_active, slot_e, slots, n_exp, step_r, e_lo, e_hi);

    const Dsv4ExpertDesc* d_ptr = reinterpret_cast<const Dsv4ExpertDesc*>(descs);
    const dim3 g_rot13((unsigned)slots, (unsigned)((h / 128 + 3) / 4));
    gather_rot_in_w13_k<<<g_rot13, 128, 0, st>>>((const __half*)hidden_in, perm, slot_e, d_ptr, xh_w13, slots, topk, h, e_lo, e_hi);

    const int sk_13 = use_gemv ? 2 : 1;
    const int own_exp = max(1, e_hi - e_lo);
    const unsigned max_act_e = (unsigned)min(slots, own_exp + slots / step_r);
    const dim3 g_gemv13((unsigned)(inter / 128), (unsigned)(2 * sk_13), max_act_e);
    const dim3 g_gemm13((unsigned)(inter / 128), 2u, max_act_e);
    const dim3 g_act((unsigned)slots, (unsigned)((inter / 128 + 3) / 4));
    const dim3 g_gemv2((unsigned)(h / 128), 1, max_act_e);
    if (use_gemv && rows == 1) {
        grouped_gemv_w13_k<1><<<g_gemv13, 128, 0, st>>>(xh_w13, d_ptr, active_e, n_active, offsets, counts, z_w13, nullptr, limit, slots, h, inter, sk_13);
        rot_out_swiglu_rot_in_w2_k<<<g_act, 128, 0, st>>>(z_w13, slot_e, d_ptr, (__half*)xh_w2, slots, inter, sk_13, limit, e_lo, e_hi);
        grouped_gemv_w2_k<1><<<g_gemv2, 128, 0, st>>>((const __half*)xh_w2, d_ptr, active_e, n_active, offsets, counts, (__half*)y, inter, h);
    } else if (use_gemv && rows <= 4) {
        grouped_gemv_w13_k<4><<<g_gemv13, 128, 0, st>>>(xh_w13, d_ptr, active_e, n_active, offsets, counts, z_w13, nullptr, limit, slots, h, inter, sk_13);
        rot_out_swiglu_rot_in_w2_k<<<g_act, 128, 0, st>>>(z_w13, slot_e, d_ptr, (__half*)xh_w2, slots, inter, sk_13, limit, e_lo, e_hi);
        grouped_gemv_w2_k<4><<<g_gemv2, 128, 0, st>>>((const __half*)xh_w2, d_ptr, active_e, n_active, offsets, counts, (__half*)y, inter, h);
    } else if (use_gemv && rows <= 16) {
        grouped_gemv_w13_k<8><<<g_gemv13, 128, 0, st>>>(xh_w13, d_ptr, active_e, n_active, offsets, counts, z_w13, nullptr, limit, slots, h, inter, sk_13);
        rot_out_swiglu_rot_in_w2_k<<<g_act, 128, 0, st>>>(z_w13, slot_e, d_ptr, (__half*)xh_w2, slots, inter, sk_13, limit, e_lo, e_hi);
        grouped_gemv_w2_k<8><<<g_gemv2, 128, 0, st>>>((const __half*)xh_w2, d_ptr, active_e, n_active, offsets, counts, (__half*)y, inter, h);
    } else {
        grouped_gemm_w13_k<4><<<g_gemm13, 256, 0, st>>>(xh_w13, d_ptr, active_e, n_active, offsets, counts, (__half*)y, slots, h, inter);
        swiglu_rot_in_w2_half_k<<<g_act, 128, 0, st>>>((const __half*)y, slot_e, d_ptr, (__half*)xh_w2, slots, inter, limit, e_lo, e_hi);
        grouped_gemm_w2_k<4><<<g_gemv2, 256, 0, st>>>((const __half*)xh_w2, d_ptr, active_e, n_active, offsets, counts, (__half*)y, inter, h);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_moe_fast_experts_tp_post(void* shared_f16,
                                               const void* y,
                                               const void* y_peer,
                                               const int* offsets,
                                               const int* pinv,
                                               const void* wpack,
                                               void* moe_out_f32,
                                               int rows,
                                               int topk,
                                               int h,
                                               int e_lo,
                                               int e_hi,
                                               void* stream) {
    if (rows <= 0 || topk <= 0 || h <= 0) return ERR_INVALID;
    cudaStream_t st = gb_stream(stream);
    scatter_tp_k<<<(unsigned)rows, 256, 0, st>>>((const __half*)shared_f16, (const __half*)y, (const __half*)y_peer,
                                                 offsets, pinv, (const __half*)wpack, (float*)moe_out_f32,
                                                 (__half*)shared_f16, topk, h, e_lo, e_hi);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_interleave_o_halves(const void* left,
                                          const void* right,
                                          void* dst,
                                          int rows,
                                          int half_f16,
                                          void* stream) {
    if (rows <= 0 || half_f16 <= 0 || (half_f16 & 7) != 0) return ERR_INVALID;
    if (!al16(left) || !al16(right) || !al16(dst)) return ERR_INVALID;
    const int half_u4 = half_f16 / 8;
    const dim3 grid((unsigned)((half_u4 + 255) / 256), (unsigned)rows);
    interleave_o_halves_k<<<grid, 256, 0, gb_stream(stream)>>>(
        reinterpret_cast<const uint4*>(left),
        reinterpret_cast<const uint4*>(right),
        reinterpret_cast<uint4*>(dst),
        rows,
        half_u4);
    return GB_CHECK_LAUNCH();
}
