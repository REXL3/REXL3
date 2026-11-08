// Tensor-core primitives for EXL3 GEMMs under prefill contract P2 (docs/NEMOTRON_PREFILL_MMA.md): the dequant-and-MMA tile shared by
// the MoE prefill kernel (nemotron_moe_mma.cu) and, for lane S3, the dense linears.
//
// Contract P2 in one paragraph (the doc has the full text): a row's output depends on that row only (not on the other rows, the slice,
// its width, the padding or the order of rows); the k reduction order is a fixed function of K; activations enter the MMA as fp16 (or
// bf16, for the quality comparison only), weights are the EXL3 trellis values exactly as the decode path decodes them (fp16 codebook
// values), accumulation is fp32; deterministic run to run (no atomics, no split K).
//
// What this header provides:
//   * `ScaledRows`: the A operand format. A row of K activations is stored as K 16-bit values (fp16, or bf16) plus one fp32
//     power-of-two factor per 128-k block (`inv`): value = stored * inv, exactly. The producer picks, per (row, 128-block), the factor
//     that brings the block's largest magnitude to [2^14, 2^15) (`block_factor`), so relu2 activations of any size neither overflow
//     fp16 nor lose bits to its subnormals. The consumer multiplies each 128-k block's fp32 partial by `inv` (an exact scaling) and adds
//     it to the running sum in block order: `tot = fma(part_b, inv_b, tot)`, b = 0, 1, ..: the k order is a function of K alone.
//   * the A stage layout (rows of 64 k = 8 chunks of 16 B, chunk c of row r at c ^ (r & 7): ldmatrix is conflict-free) and its
//     ldmatrix.x4 addressing;
//   * the B side: a 16 x 16 EXL3 tile's 32 words in shared memory (cp.async'd raw, as streamed) decoded by one warp straight into the
//     two m16n8k16 B fragments of the tile (exl3_linear_decode.cuh's decode_lane: the decode path's codebook values, bit for bit);
//   * mma.sync m16n8k16 with fp32 accumulation, f16 or bf16 operands; cp.async with zero fill; the per-128-block convert of a warp's
//     128 fp32 values (4 a lane) to ScaledRows.
//
// Structural ideas taken from upstream ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp), written anew here: raw trellis tiles are streamed
// into shared memory with cp.async and each warp decodes its own column tiles into registers (exl3_gemm_inner.cuh, `sh_b` stages and the
// per-warp FRAGS_N dequant), and A is XOR-swizzled by 16-byte chunk for conflict-free fragment loads (exl3_gemm_inner.cuh, A_SWIZZLE).
// The scaled-row format and the fixed-order block fold are this engine's (contract P2).

#pragma once

#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_linear_decode.cuh"

namespace gbm {

using namespace gbx;

constexpr int kKStage = 64;       // k of one A / B pipeline stage (4 k tiles)
constexpr int kScaleBlock = 128;  // k of one scale block of ScaledRows

// ---- async copies -----------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
}

// 16 bytes, or 16 zero bytes when !valid (gmem must still be a valid address: the copy reads 0 bytes of it).
__device__ __forceinline__ void cp_async16_z(void* smem, const void* gmem, bool valid) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(n) : "memory");
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// ---- MMA ----------------------------------------------------------------------------------------------------------------------------

// d += a @ b, m16n8k16, fp32 accumulators; BF: bf16 operands, else fp16.
template <bool BF>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    if constexpr (BF) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

// ---- A stage layout ---------------------------------------------------------------------------------------------------------------

// The half offset of 16-byte chunk c (0..7) of row r in an A stage (rows of kKStage = 64 values, 128 bytes).
__device__ __forceinline__ int a_chunk(int r, int c) { return r * kKStage + ((c ^ (r & 7)) << 3); }

// This lane's ldmatrix.x4 address for m tile m (rows 16 m ..) and k tile kk (0..3) of a stage: matrices (rows 0-7, k 0-7), (rows 8-15,
// k 0-7), (rows 0-7, k 8-15), (rows 8-15, k 8-15) are the m16n8k16 A fragments a0..a3.
__device__ __forceinline__ const uint16_t* a_frag_addr(const uint16_t* stage, int m, int kk, int lane) {
    return stage + a_chunk(16 * m + (lane & 15), 2 * kk + (lane >> 4));
}

// ---- B side: a tile's words in shared memory to the B fragments ------------------------------------------------------------------

// The B fragments of one 16 x 16 tile (tile: its tile_words<K2>() words in shared memory, as stored in the checkpoint): columns 0-7 in
// b0, 8-15 in b1; the values are the decode path's fp16 codebook values (BF: then rounded to bf16, for the bf16 variant only).
template <int K2, int CB, bool BF>
__device__ __forceinline__ void tile_frags(const uint32_t* tile, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t w[lane_words<K2>()];
    load_lane_words<K2>(tile, lane, w);
    decode_lane<K2, CB>(w, lane, b0, b1);
    if constexpr (BF) {
        uint32_t* v[4] = {&b0[0], &b0[1], &b1[0], &b1[1]};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(*reinterpret_cast<const half2*>(v[i]));
            const __nv_bfloat162 q = __floats2bfloat162_rn(f.x, f.y);
            *v[i] = *reinterpret_cast<const uint32_t*>(&q);
        }
    }
}

// ---- ScaledRows -------------------------------------------------------------------------------------------------------------------

// The power-of-two factor f for a 128-block whose largest magnitude is m (m * f in [2^14, 2^15); 1 for a zero or non-finite m), and
// its inverse. Both are normal floats, so x * f and p * inv are exact (barring underflow of x * f below the fp16 subnormals, which is
// at most 2^-24 of the block's largest value).
__device__ __forceinline__ float block_factor(float m, float& inv) {
    if (!(m > 0.f) || !(m <= 3.402823466e38f)) {
        inv = 1.f;
        return 1.f;
    }
    int s = 14 - ilogbf(m);
    s = s > 127 ? 127 : (s < -126 ? -126 : s);
    inv = __int_as_float((127 - s) << 23);
    return __int_as_float((127 + s) << 23);
}

// A warp's 128 fp32 values (4 a lane: values 4 lane .. 4 lane + 3) to ScaledRows: dst16 is the block's first value (this lane writes
// 4 x 16 bits at dst16 + 4 lane), *inv_dst the block's factor (lane 0 writes it).
template <bool BF>
__device__ __forceinline__ void store_scaled_block(const float (&v)[4], int lane, uint16_t* dst16, float* inv_dst) {
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float inv;
    const float f = block_factor(m, inv);
    uint2 u;
    if constexpr (BF) {
        const __nv_bfloat162 a = __floats2bfloat162_rn(v[0] * f, v[1] * f), b = __floats2bfloat162_rn(v[2] * f, v[3] * f);
        u.x = *reinterpret_cast<const uint32_t*>(&a);
        u.y = *reinterpret_cast<const uint32_t*>(&b);
    } else {
        const half2 a = __floats2half2_rn(v[0] * f, v[1] * f), b = __floats2half2_rn(v[2] * f, v[3] * f);
        u.x = *reinterpret_cast<const uint32_t*>(&a);
        u.y = *reinterpret_cast<const uint32_t*>(&b);
    }
    *reinterpret_cast<uint2*>(dst16 + 4 * lane) = u;
    if (lane == 0) *inv_dst = inv;
}

// The fold of one 128-k block's partial sums into the running sums: tot = fma(part, inv, tot), exact scaling then one rounding.
template <int MT>
__device__ __forceinline__ void fold_block(float (&acc)[MT][2][4], float (&tot)[MT][2][4], const float (&lo)[MT], const float (&hi)[MT],
                                           int mt) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                tot[m][h][0] = __fmaf_rn(acc[m][h][0], lo[m], tot[m][h][0]);
                tot[m][h][1] = __fmaf_rn(acc[m][h][1], lo[m], tot[m][h][1]);
                tot[m][h][2] = __fmaf_rn(acc[m][h][2], hi[m], tot[m][h][2]);
                tot[m][h][3] = __fmaf_rn(acc[m][h][3], hi[m], tot[m][h][3]);
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][h][c] = 0.f;
            }
        }
    }
}

// One k tile of a warp: decode the warp's 16-column B tile (tile words in shared memory) and multiply the mt active 16-row A tiles of
// the stage by it. acc[m][h]: rows 16 m + (g, g + 8), columns 8 h + 2 t (+1) of the warp's 16 columns.
template <int K2, int CB, int MT, bool BF>
__device__ __forceinline__ void warp_ktile(float (&acc)[MT][2][4], const uint16_t* a_stage, const uint32_t* tile, int kk, int mt,
                                           int lane) {
    uint32_t b0[2], b1[2];
    tile_frags<K2, CB, BF>(tile, lane, b0, b1);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m < mt) {
            uint32_t a[4];
            ldmatrix_x4(a, a_frag_addr(a_stage, m, kk, lane));
            mma<BF>(acc[m][0], a, b0);
            mma<BF>(acc[m][1], a, b1);
        }
    }
}

}  // namespace gbm
