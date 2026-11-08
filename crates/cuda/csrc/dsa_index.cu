// gb-nvcc: --fmad=false
// DSA indexer, key side: the index-key LayerNorm with the raw-row ring, and the pooled-key update.
//
// Planes (per indexer layer; the MTP layer has its own):
//   ring  f16 [slots][ring_len][256]          the last ring_len tokens of each slot as [LayerNorm(k) 128 | gate 128] rows, token t at
//                                              row t % ring_len. It exists only so a pool that straddles two calls can be completed:
//                                              a pool's raw rows are read once, when it completes, and never again (scoring reads the
//                                              pool plane only), so the oracle's per-token plane (256 f16 per token, 128 MiB per layer
//                                              at 262,144 tokens) is not kept. ring_len >= DSA_RING_MIN (a full window plus a pool).
//   pool  f16 [n_blocks][64][128]             pooled keys, pool p of slot s at block_table[s][p / 64], row p % 64. A block is a
//                                              256-token page, the same block ids as the latent cache.
//
// Both kernels take per-row (slot, absolute position) arrays. The rows of one window are consecutive positions of one slot, in
// order; windows of different slots may share a call.
//
// norm_ring is the "append": it runs for every row in every regime (the planes must be complete before the context first
// exceeds 2,048). pool runs after it, on the same stream: the row whose position is 3 mod 4 completes pool pos/4 and
// (re)computes it from its four members, always, with no "already written" shortcut: that one rule covers pools that straddle calls
// and the rollback of rejected draft rows (the pool is recomputed from the accepted rows when its last member arrives again).
//
// Numerics follow the fork (fp16 planes, fp32 math): the LayerNorm is a two-pass fp32 one with a fixed reduction tree; the pool is
// softmax(g + ape) in fp32 (max, exp, sum in member order, one division per weight) then the weighted sum in member order, each
// product and each add rounded alone (no FMA), one round to fp16 at the end.
#include "dsa.cuh"

namespace {

__device__ __forceinline__ float warp_sum32(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, m));
    return x;  // an xor butterfly: every lane ends with the same bits
}

__device__ __forceinline__ void unpack4(uint2 v, float (&x)[4]) {
    __half2 a = *reinterpret_cast<__half2*>(&v.x), b = *reinterpret_cast<__half2*>(&v.y);
    float2 fa = __half22float2(a), fb = __half22float2(b);
    x[0] = fa.x; x[1] = fa.y; x[2] = fb.x; x[3] = fb.y;
}

// One warp per row; lane l owns dims 4l..4l+3. Writes [LayerNorm(k_raw) | gate] to the staging row and to the ring.
__global__ __launch_bounds__(128) void dsa_norm_ring_kernel(
    const half* __restrict__ k_raw, const half* __restrict__ gate, const half* __restrict__ ln_w, const half* __restrict__ ln_b,
    float eps, const int* __restrict__ row_slot, const int* __restrict__ row_pos, int rows, half* __restrict__ kg,
    half* __restrict__ ring, int ring_len) {
    const int r = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const uint2 kv = *reinterpret_cast<const uint2*>(k_raw + (size_t)r * DSA_D + 4 * lane);
    float x[4];
    unpack4(kv, x);
    float mean = warp_sum32(__fadd_rn(__fadd_rn(__fadd_rn(x[0], x[1]), x[2]), x[3]));
    mean = __fmul_rn(mean, 1.0f / DSA_D);  // 1/128 is exact
    float d[4], sq = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = __fsub_rn(x[i], mean);
    sq = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(d[0], d[0]), __fmul_rn(d[1], d[1])), __fmul_rn(d[2], d[2])), __fmul_rn(d[3], d[3]));
    const float var = __fmul_rn(warp_sum32(sq), 1.0f / DSA_D);
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
    float w[4], b[4];
    unpack4(*reinterpret_cast<const uint2*>(ln_w + 4 * lane), w);
    unpack4(*reinterpret_cast<const uint2*>(ln_b + 4 * lane), b);
    __half2 o01 = __floats2half2_rn(__fadd_rn(__fmul_rn(__fmul_rn(d[0], rstd), w[0]), b[0]),
                                    __fadd_rn(__fmul_rn(__fmul_rn(d[1], rstd), w[1]), b[1]));
    __half2 o23 = __floats2half2_rn(__fadd_rn(__fmul_rn(__fmul_rn(d[2], rstd), w[2]), b[2]),
                                    __fadd_rn(__fmul_rn(__fmul_rn(d[3], rstd), w[3]), b[3]));
    uint2 ok;
    ok.x = *reinterpret_cast<uint32_t*>(&o01);
    ok.y = *reinterpret_cast<uint32_t*>(&o23);
    const uint2 gv = *reinterpret_cast<const uint2*>(gate + (size_t)r * DSA_D + 4 * lane);
    half* row = kg + (size_t)r * 2 * DSA_D;
    *reinterpret_cast<uint2*>(row + 4 * lane) = ok;
    *reinterpret_cast<uint2*>(row + DSA_D + 4 * lane) = gv;
    const int pos = row_pos[r];
    if (pos < 0) return;
    half* rg = ring + ((size_t)row_slot[r] * ring_len + (size_t)(pos % ring_len)) * 2 * DSA_D;
    *reinterpret_cast<uint2*>(rg + 4 * lane) = ok;
    *reinterpret_cast<uint2*>(rg + DSA_D + 4 * lane) = gv;
}

// One block per row, one thread per dimension. Only the row that completes a pool (pos % 4 == 3) works.
__global__ __launch_bounds__(DSA_D) void dsa_pool_kernel(
    const half* __restrict__ kg, const half* __restrict__ ring, int ring_len, const float* __restrict__ ape,
    const int* __restrict__ row_slot, const int* __restrict__ row_pos, int rows, half* __restrict__ pool,
    const int* __restrict__ block_table, int max_blocks) {
    const int r = blockIdx.x;
    const int d = threadIdx.x;
    const int pos = row_pos[r];
    if (pos < 0 || (pos + 1) % DSA_P != 0) return;
    const int slot = row_slot[r];
    const int p = (pos + 1) / DSA_P - 1;
    float g[DSA_P], k[DSA_P];
#pragma unroll
    for (int j = 0; j < DSA_P; ++j) {
        const int back = DSA_P - 1 - j;  // member j is the token `back` positions before this row's
        const int rb = r - back;
        // a member that is a row of this call (same slot, the consecutive position) comes from the staging rows; an earlier one
        // from the ring. Rows are never mixed up across slots: both slot and position must match
        const bool in_call = rb >= 0 && row_slot[rb] == slot && row_pos[rb] == pos - back;
        const half* src = in_call ? kg + (size_t)rb * 2 * DSA_D
                                  : ring + ((size_t)slot * ring_len + (size_t)((pos - back) % ring_len)) * 2 * DSA_D;
        g[j] = __fadd_rn(__half2float(src[DSA_D + d]), ape[j * DSA_D + d]);
        k[j] = __half2float(src[d]);
    }
    const float m = fmaxf(fmaxf(g[0], g[1]), fmaxf(g[2], g[3]));
    float e[DSA_P];
#pragma unroll
    for (int j = 0; j < DSA_P; ++j) e[j] = expf(__fsub_rn(g[j], m));
    const float s = __fadd_rn(__fadd_rn(__fadd_rn(e[0], e[1]), e[2]), e[3]);
    float acc = __fmul_rn(__fdiv_rn(e[0], s), k[0]);
#pragma unroll
    for (int j = 1; j < DSA_P; ++j) acc = __fadd_rn(acc, __fmul_rn(__fdiv_rn(e[j], s), k[j]));
    const int blk = block_table[(size_t)slot * max_blocks + p / DSA_POOLS_PER_PAGE];
    pool[((size_t)blk * DSA_POOLS_PER_PAGE + p % DSA_POOLS_PER_PAGE) * DSA_D + d] = __float2half_rn(acc);
}

}  // namespace

GB_EXPORT int gb_dsa_norm_ring(const void* k_raw, const void* gate, const void* ln_w, const void* ln_b, float eps, const int* row_slot,
                               const int* row_pos, int rows, void* kg, void* ring, int ring_len, void* stream) {
    if (rows <= 0) return 0;
    if (rows > DSA_MAX_ROWS || ring_len < DSA_RING_MIN || ring_len % DSA_P) return (int)cudaErrorInvalidValue;
    dsa_norm_ring_kernel<<<(rows + 3) / 4, 128, 0, gb_stream(stream)>>>(
        (const half*)k_raw, (const half*)gate, (const half*)ln_w, (const half*)ln_b, eps, row_slot, row_pos, rows, (half*)kg,
        (half*)ring, ring_len);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsa_pool(const void* kg, const void* ring, int ring_len, const float* ape, const int* row_slot, const int* row_pos,
                          int rows, void* pool, const int* block_table, int max_blocks, void* stream) {
    if (rows <= 0) return 0;
    if (rows > DSA_MAX_ROWS || ring_len < DSA_RING_MIN || ring_len % DSA_P) return (int)cudaErrorInvalidValue;
    dsa_pool_kernel<<<rows, DSA_D, 0, gb_stream(stream)>>>((const half*)kg, (const half*)ring, ring_len, ape, row_slot, row_pos, rows,
                                                           (half*)pool, block_table, max_blocks);
    return GB_CHECK_LAUNCH();
}
