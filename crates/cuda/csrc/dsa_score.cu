// gb-nvcc: --fmad=false
// DSA indexer scoring: S[r, p] = fp16( 2^-6 * sum_h relu(q[r,h] . K_p) * w[r,h] ), pools p < (pos_r + 1) / 4.
//
// The kernel is one code path for every call shape (1..128 rows, any mix of slots), built so that S[r, p] is a pure function of
// (q[r], w[r], K_p):
//   * a block owns one pool page (64 pools = 16 KB, one block id of the page table), staged in shared memory with coalesced cp.async
//     copies; warp w scores pools 16 w .. 16 w + 15. For a row, head tile t (heads 8t..8t+7) and 16-dim step ks it issues one
//     mma.m16n8k16 (M = 16 pools, N = 8 heads, K = 16 dims), chained over ks = 0..7 from a zero accumulator: the fp32 dot of every
//     (pool, head) is accumulated in the same order, and a tensor-core output element depends on its own operands only, never on
//     the other columns (heads) or rows (pools) of the instruction
//   * the head sum is the same fixed tree for every pool: per thread over its 8 heads in (t, column) order, then the four threads
//     of a quad by an xor butterfly; relu(dot) * w is rounded alone and the sum starts at +0.0 (so the result is never -0)
//   * the 2^-6 = (128 * 32)^-1/2 scale is exact, then one round to fp16 (nearest even). Pools at or past the row's bound, inside a
//     computed page, get -inf; whole pages past the bound are not touched (the selection never reads them)
//   * the rows of a call only form a loop: the page stays in shared memory while consecutive rows share a slot, and is the same
//     bytes whether it was loaded for this row or an earlier one. Rows of the dense regime (pos + 1 <= 2048) are skipped.
// The 128 dims of the dot product are permuted identically in both operands (thread tg of a quad owns the 16-byte chunks 4c + tg, c = 0..3,
// i.e. dims 8 (4c + tg) .. +7): a fixed permutation of a sum, so operand loads are 16-byte vectors and the shared-memory reads are free of
// bank conflicts (chunks are XOR-swizzled by row parity).
#include "dsa.cuh"

namespace {

constexpr int ROW_GROUP = 16;  // rows one block serves (a verify window, or the rows of a few slots)

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float relu_mul(float c, float w) { return __fmul_rn(c > 0.f ? c : 0.f, w); }

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

// NP pages per block (NP x 64 pools, NP x 16 KB of shared memory): the head operand is loaded once per row for all NP tiles of a warp, and the
// 4 x NP accumulation chains of a row run interleaved (each chain is 8 dependent mma steps).
template <int NP>
__global__ __launch_bounds__(128) void dsa_score_kernel(
    const half* __restrict__ q, const half* __restrict__ w, int w_stride, const half* __restrict__ pool,
    const int* __restrict__ block_table, int max_blocks, const int* __restrict__ row_slot, const int* __restrict__ row_pos, int rows,
    int s_stride, half* __restrict__ scores) {
    __shared__ uint4 tile[NP][DSA_POOLS_PER_PAGE][16];  // 64 pools x 16 chunks of 16 bytes per page, chunk index XOR 4 * (row & 1)
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, tg = lane & 3;
    const int page0 = blockIdx.x * NP;
    const int r0 = blockIdx.y * ROW_GROUP, r1 = min(rows, r0 + ROW_GROUP);
    uint4 A[NP][2][4];  // [page][pool g, g + 8][chunk c]
    int cur_slot = -1;
    for (int r = r0; r < r1; ++r) {
        const int pos = row_pos[r];
        if (!dsa_row_is_sparse(pos)) continue;
        const int T = dsa_row_pools(pos);
        if (page0 * DSA_POOLS_PER_PAGE >= T) continue;  // uniform over the block
        const int slot = row_slot[r];
        if (slot != cur_slot) {
            if (cur_slot >= 0) __syncthreads();  // everyone is done reading the previous slot's tiles
            cur_slot = slot;
#pragma unroll
            for (int pp = 0; pp < NP; ++pp) {
                const int page = page0 + pp;
                const int blk = page < max_blocks ? block_table[(size_t)slot * max_blocks + page] : 0;
                const uint4* src = reinterpret_cast<const uint4*>(pool + (size_t)blk * DSA_POOLS_PER_PAGE * DSA_D);
#pragma unroll
                for (int it = 0; it < 8; ++it) {
                    const int L = it * 128 + threadIdx.x;  // 16-byte chunk of the page, in memory order
                    const int row = L >> 4, ch = L & 15;
                    cp_async16(&tile[pp][row][ch ^ (4 * (row & 1))], src + L);
                }
            }
            asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
#pragma unroll
            for (int pp = 0; pp < NP; ++pp)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = 16 * warp + g + 8 * h;
#pragma unroll
                    for (int c = 0; c < 4; ++c) A[pp][h][c] = tile[pp][row][(4 * c + tg) ^ (4 * (row & 1))];
                }
        }
        // head operand for all four head tiles: head 8t + g, chunks 4c + tg
        uint4 B[4][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint4* qs = reinterpret_cast<const uint4*>(q + ((size_t)r * DSA_H + 8 * t + g) * DSA_D);
#pragma unroll
            for (int c = 0; c < 4; ++c) B[t][c] = __ldg(qs + 4 * c + tg);
        }
        // head weights of this thread: heads 8t + 2tg, 8t + 2tg + 1 for t = 0..3
        float wf[8];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(w + (size_t)r * w_stride + 8 * t + 2 * tg));
            wf[2 * t] = f.x;
            wf[2 * t + 1] = f.y;
        }
        float acc[NP][4][4];  // [page][head tile][fp32 accumulator fragment]
#pragma unroll
        for (int pp = 0; pp < NP; ++pp)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[pp][t][0] = acc[pp][t][1] = acc[pp][t][2] = acc[pp][t][3] = 0.f;
        // 8 dependent k16 steps per chain; the 4 NP chains are independent, so they interleave
#pragma unroll
        for (int c = 0; c < 4; ++c) {
#pragma unroll
            for (int half_step = 0; half_step < 2; ++half_step)  // step 2c: dims 0..3 of chunk c, step 2c + 1: dims 4..7
#pragma unroll
                for (int pp = 0; pp < NP; ++pp)
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        if (half_step == 0)
                            mma16816(acc[pp][t], A[pp][0][c].x, A[pp][1][c].x, A[pp][0][c].y, A[pp][1][c].y, B[t][c].x, B[t][c].y);
                        else
                            mma16816(acc[pp][t], A[pp][0][c].z, A[pp][1][c].z, A[pp][0][c].w, A[pp][1][c].w, B[t][c].z, B[t][c].w);
                    }
        }
#pragma unroll
        for (int pp = 0; pp < NP; ++pp) {
            // c[0], c[1]: pool g, heads 8t + 2tg, +1; c[2], c[3]: pool g + 8. The head sum runs in t order, then over the quad
            float s_lo = 0.f, s_hi = 0.f;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                s_lo = __fadd_rn(__fadd_rn(s_lo, relu_mul(acc[pp][t][0], wf[2 * t])), relu_mul(acc[pp][t][1], wf[2 * t + 1]));
                s_hi = __fadd_rn(__fadd_rn(s_hi, relu_mul(acc[pp][t][2], wf[2 * t])), relu_mul(acc[pp][t][3], wf[2 * t + 1]));
            }
            s_lo = __fadd_rn(s_lo, __shfl_xor_sync(0xffffffffu, s_lo, 1));
            s_hi = __fadd_rn(s_hi, __shfl_xor_sync(0xffffffffu, s_hi, 1));
            s_lo = __fadd_rn(s_lo, __shfl_xor_sync(0xffffffffu, s_lo, 2));
            s_hi = __fadd_rn(s_hi, __shfl_xor_sync(0xffffffffu, s_hi, 2));
            // lane l < 16 stores pool 16 warp + l of the page: pools 0..7 are the quad leaders' s_lo, pools 8..15 their s_hi
            const float vlo = __shfl_sync(0xffffffffu, s_lo, (lane & 7) * 4);
            const float vhi = __shfl_sync(0xffffffffu, s_hi, (lane & 7) * 4);
            float v = __fmul_rn(lane < 8 ? vlo : vhi, 0.015625f);  // 2^-6, exact
            v = __fadd_rn(v, 0.0f);                                  // -0 -> +0
            if (!(v == v)) v = -INFINITY;
            const int base = (page0 + pp) * DSA_POOLS_PER_PAGE + 16 * warp;  // first pool of this warp's tile
            const int p = base + lane;
            if (lane < 16 && p < s_stride && base < T) scores[(size_t)r * s_stride + p] = p < T ? __float2half_rn(v) : __float2half_rn(-INFINITY);
        }
    }
}

}  // namespace

#define DSA_SCORE_NP 2
GB_EXPORT int gb_dsa_score_pools_per_block() { return DSA_POOLS_PER_PAGE * DSA_SCORE_NP; }

GB_EXPORT int gb_dsa_score(const void* q, const void* w, int w_stride, const void* pool, const int* block_table, int max_blocks,
                           const int* row_slot, const int* row_pos, int rows, int n_pools_max, int s_stride, void* scores,
                           void* stream) {
    if (rows <= 0 || n_pools_max <= 0) return 0;
    if (rows > DSA_MAX_ROWS || s_stride < n_pools_max || s_stride % 8 || w_stride < DSA_H) return (int)cudaErrorInvalidValue;
    dim3 grid((n_pools_max + DSA_POOLS_PER_PAGE * DSA_SCORE_NP - 1) / (DSA_POOLS_PER_PAGE * DSA_SCORE_NP), (rows + ROW_GROUP - 1) / ROW_GROUP);
    dsa_score_kernel<DSA_SCORE_NP><<<grid, 128, 0, gb_stream(stream)>>>((const half*)q, (const half*)w, w_stride, (const half*)pool, block_table,
                                                         max_blocks, row_slot, row_pos, rows, s_stride, (half*)scores);
    return GB_CHECK_LAUNCH();
}
