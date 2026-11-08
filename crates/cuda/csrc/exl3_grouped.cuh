// The inner loop shared by the large-row EXL3 GEMMs (moe_prefill.cu for the routed experts, exl3_big.cu for the dense layers). Both reproduce
// the DECODE kernels' arithmetic bit for bit (exl3_linear.cu's linear_kernel, moe_experts.cu's moe_gemv_kernel), so what they share is exactly
// what those kernels do per warp:
//
//   * a warp owns `per_warp` consecutive 16-wide k tiles of one K split; each of its accumulators is one fresh fp32 m16n8k16 chain from zero
//     over those k tiles in order;
//   * NT 16-row mma tiles share every tile decode (a tile's B fragments are decoded once and applied to all NT tiles of A): a tile's rows only
//     ever meet their own accumulators, so a row's chain does not depend on NT, on H (column sweeps) or on the other rows;
//   * the warps' chains are added in warp order and the splits' sums in split order (`reduce_sweep`, with the running sum in `stage`).
//
// Header-only, templates only; included by exactly the translation units that name `using namespace gbp`.

#pragma once

#include <stdint.h>

#include "common.cuh"
#include "exl3_linear_decode.cuh"

namespace gbp {

using namespace gbx;

template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

// Registers a lane holds for one k step of JN tiles (a column half of the 8-tile block row, or the whole of it).
template <int K2, int JN>
__host__ __device__ constexpr int step_regs() {
    return step_shuffled<K2>() ? JN * tile_words<K2>() / 32 : JN * lane_words<K2>();
}

template <int K2, int JN>
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

template <int K2, int JN>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2, JN>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

// Column sweeps a group of NT tiles takes (a warp's accumulators are NT x 8 / H tiles x 8 floats; 3 and 4 tiles keep half a block row at a
// time, as the decode path's linear_kernel does, so that no instantiation spills).
template <int K2, int CB>
constexpr int sweeps_of(int nt) {
    return nt >= 3 || (nt == 2 && K2 >= 10 && CB != 2) ? 2 : 1;
}

// Whether a lane's first window starts in the word before its own (the 1- and 2-bit shuffle scheme).
template <int K2>
__device__ __forceinline__ bool lane_prev(int lane) {
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }
    return prev;
}

// One warp's k sweep: acc[p][j][h] += sum over its `per_warp` k tiles (kt0 .. kt0 + per_warp - 1) of A tile p times B tile j (columns 16 j ..,
// halves h) for the active tiles p < nt. `tile`: the stripe's first word at k tile kt0 and this sweep's first column tile; `stride_k`: words
// between k tile rows; `xp[p]`: tile p's A fragments for k tile 0 plus this lane's 16 bytes (the layout of rot_in_frag_kernel).
//
// The loads run PD steps ahead (PD = 1: the next step's words are loaded while this one is decoded) in a ring of registers, and with AP = 1 the next
// step's A fragments too: neither changes an operand, only when it was read, so the bits do not depend on them.
template <int K2, int CB, int NT, int JN, int PD = 1, int AP = 0>
__device__ __forceinline__ void ksweep(float (&acc)[NT][JN][2][4], const uint32_t* tile, long long stride_k,
                                       const uint32_t* const (&xp)[NT], int kt0, int per_warp, int nt, int lane, bool prev) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr bool PF = K2 <= 12;
    constexpr int SR = step_regs<K2, JN>();
    constexpr int RD = PF ? PD + 1 : 1;
    uint32_t ring[RD][PF ? SR : 1];
    uint4 anx[AP ? NT : 1];
    if constexpr (PF) {
#pragma unroll
        for (int d = 0; d < PD; ++d)
            if (d < per_warp) load_step<K2, JN>(tile + (size_t)d * stride_k, lane, ring[d]);
    }
    if constexpr (AP) {
#pragma unroll
        for (int p = 0; p < NT; ++p)
            if (p < nt) anx[p] = __ldg(reinterpret_cast<const uint4*>(xp[p] + (size_t)kt0 * 128));
    }
#pragma unroll 1
    for (int i = 0; i < per_warp; ++i) {
        const int kt = kt0 + i;
        uint32_t a[NT][4];
#pragma unroll
        for (int p = 0; p < NT; ++p) {
            if (p < nt) {
                uint4 v;
                if constexpr (AP) v = anx[p];
                else v = __ldg(reinterpret_cast<const uint4*>(xp[p] + (size_t)kt * 128));
                a[p][0] = v.x; a[p][1] = v.y; a[p][2] = v.z; a[p][3] = v.w;
            }
        }
        if constexpr (AP) {
            if (i + 1 < per_warp) {
#pragma unroll
                for (int p = 0; p < NT; ++p)
                    if (p < nt) anx[p] = __ldg(reinterpret_cast<const uint4*>(xp[p] + (size_t)(kt + 1) * 128));
            }
        }
        if constexpr (PF) {
            if (i + PD < per_warp) load_step<K2, JN>(tile + (size_t)(i + PD) * stride_k, lane, ring[PD]);
        }
#pragma unroll
        for (int j = 0; j < JN; ++j) {
            uint32_t w[LW];
            if constexpr (PF) step_lane_words<K2, JN>(ring[0], j, lane, prev, w);
            else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
            uint32_t b0[2], b1[2];
            decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
            for (int p = 0; p < NT; ++p) {
                if (p < nt) {
                    mma16816(acc[p][j][0], a[p], b0);
                    mma16816(acc[p][j][1], a[p], b1);
                }
            }
        }
        if constexpr (PF) {
#pragma unroll
            for (int d = 0; d < PD; ++d)
#pragma unroll
                for (int q = 0; q < SR; ++q) ring[d][q] = ring[d + 1][q];
        }
    }
}

// The warps' sums of a sweep in warp order, one 16-row tile at a time (rows 0-7 and then rows 8-15), added to the running sum over the
// splits in `stage` ([NT * 16][128] fp32, columns hh * RC ..): the first split starts it, every later one is added (the decode epilogue's
// `sg = Z[0]; sg += Z[1]`). red: WK * 8 * RC floats of shared memory. `tt`: the thread's index in its team of WK warps (threadIdx.x for a block that is
// one team; a block of two teams computing two matrices at once runs the same code in lockstep, barriers included).
template <int WK, int NT, int JN>
__device__ __forceinline__ void reduce_sweep(const float (&acc)[NT][JN][2][4], float* red, float* stage, int hh, bool first_split, int nt,
                                             int warp, int g, int t, int tt) {
    constexpr int RC = JN * 16;
#pragma unroll
    for (int p = 0; p < NT; ++p) {
        if (p >= nt) break;
#pragma unroll
        for (int rlo = 0; rlo < 16; rlo += 8) {
            __syncthreads();                           // red is reused by every half and tile
#pragma unroll
            for (int i = 0; i < JN; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = i * 16 + h * 8 + 2 * t;
                    *reinterpret_cast<float2*>(red + (warp * 8 + g) * RC + col) =
                        rlo ? make_float2(acc[p][i][h][2], acc[p][i][h][3]) : make_float2(acc[p][i][h][0], acc[p][i][h][1]);
                }
            __syncthreads();
            for (int idx = tt; idx < 8 * (RC / 4); idx += WK * 32) {
                const int r = idx / (RC / 4), c = 4 * (idx % (RC / 4));
                float4 s = *reinterpret_cast<const float4*>(red + r * RC + c);
#pragma unroll
                for (int w = 1; w < WK; ++w) {
                    const float4 q = *reinterpret_cast<const float4*>(red + (w * 8 + r) * RC + c);
                    s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                }
                float4* dst = reinterpret_cast<float4*>(stage + (16 * p + rlo + r) * 128 + hh * RC + c);
                if (first_split) {
                    *dst = s;
                } else {
                    float4 d = *dst;
                    d.x += s.x; d.y += s.y; d.z += s.z; d.w += s.w;
                    *dst = d;
                }
            }
        }
    }
}

// The whole of one block's work for one group of NT tiles over all K splits and column sweeps: leaves the merged fp32 sums of the group's
// rows x 128 columns in `stage`, after a final __syncthreads. WK warps a block (the decode plan's: 4 for the experts, 4 or 8 for a dense layer).
template <int WK, int K2, int CB, int NT, int H, int PD = 1, int AP = 0>
__device__ __forceinline__ void group_sums(const uint32_t* tiles, long long stride_k, const uint32_t* const (&xp)[NT], int per_warp, int SK,
                                           int nt, float* red, float* stage, int tt) {
    constexpr int TW = tile_words<K2>();
    constexpr int JN = 8 / H;
    const int warp = tt >> 5, lane = tt & 31;
    const int g = lane >> 2, t = lane & 3;
    const bool prev = lane_prev<K2>(lane);
    for (int split = 0; split < SK; ++split) {
        const int kt0 = split * (per_warp * WK) + warp * per_warp;
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
            ksweep<K2, CB, NT, JN, PD, AP>(acc, tiles + (size_t)kt0 * stride_k + (size_t)(hh * JN) * TW, stride_k, xp, kt0, per_warp, nt, lane, prev);
            reduce_sweep<WK, NT, JN>(acc, red, stage, hh, split == 0, nt, warp, g, t, tt);
        }
    }
    __syncthreads();                                           // every sum is in stage
}

}  // namespace gbp
