// gb-nvcc: --fmad=false
// DSA indexer selection: exact top-512 pools of a row's fp16 scores, ties to the lowest pool index, expanded to tokens, plus the tail.
//
// One block per row; the result is a pure function of that row's score row and position, so it cannot depend on the other rows of the
// call (nothing is shared between blocks, and the emission order is the index order). There are no atomics anywhere.
//
// Exactness: every score becomes a 16-bit order key (IEEE order; -0 == +0; -inf and NaN are invalid and map to 0, +inf is the largest).
// The row's keys live in registers (1024 threads x 64 keys = 65,536 pools = 262,144 tokens of context; longer rows re-read the scores in
// super-tiles). The k-th largest key v* is found by bisection over the 16 key bits: step b counts, with one block reduction (integer
// sums: exact in any order), the keys >= prefix | 2^b and keeps the bit when at least k keys remain; then n_gt = #(keys > v*) and the
// ties still to take are k - n_gt. A last pass walks the row in index order and emits every key above v* plus the first (k - n_gt)
// keys equal to v*, i.e. the lowest-indexed ties, each at its closed-form output position (#above before it) + min(#ties before it,
// ties to take) from a single packed block scan per tile. Selecting from the ordered keys instead of a score-sorted structure is what
// makes the tie rule an index rule. (The ExLlamaV3 fork's dsa_topk.cu, MIT, uses a two-pass 8-bit radix histogram with shared atomics
// and a two-run emission; the contract here is the same set, emitted in one ascending run.)
//
// Output row (int32, stride DSA_K_STRIDE): the selected pools' tokens ascending (pool * 4 + j), then the tail tokens (the 0..3 raw
// tokens of the incomplete last pool, ascending), then -1. `meta[r] = (sparse, count)`: a dense row (pos + 1 <= 2048) is not touched
// and gets (0, 0); a sparse row gets (1, 4 * pools_selected + tail), 2048 + tail whenever the row has >= 512 valid scores (always, for
// finite scores: a sparse row sees >= 512 pools). A row with fewer valid scores selects all of them.
#include "dsa.cuh"

namespace {

constexpr int THREADS = 1024;
constexpr int NWARPS = THREADS / 32;
constexpr int EPT = 8;                        // elements per thread per tile: one 16-byte load
constexpr int TILE = THREADS * EPT;           // 8192 pools
constexpr int NTILE = 8;                      // tiles held in registers
constexpr int SUPER = TILE * NTILE;           // 65,536 pools

// order key of fp16 bits; invalid (-inf, any NaN) maps to 0, every valid key is >= 0x0400
__device__ __forceinline__ uint32_t order_key(uint32_t u) {
    u &= 0xffffu;
    if (u == 0x8000u) u = 0;                       // -0 == +0
    if ((u & 0x7fffu) > 0x7c00u || u == 0xfc00u) return 0;  // NaN or -inf
    return (u & 0x8000u) ? (~u & 0xffffu) : (u | 0x8000u);
}
__device__ __forceinline__ uint32_t order_key2(uint32_t pair) { return order_key(pair) | (order_key(pair >> 16) << 16); }

__device__ __forceinline__ int warp_incl_scan(int v, int lane) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    return v;
}
__device__ __forceinline__ int warp_sum_i(int v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Loads one tile of this thread's 8 elements as 4 packed pairs of order keys (padding past the row is invalid).
__device__ __forceinline__ void load_tile(const uint16_t* row, int T, int i0, uint4& k) {
    uint32_t w[4];
    if (i0 + EPT <= T) {
        const uint4 v = *reinterpret_cast<const uint4*>(row + i0);
        w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t lo = (i0 + 2 * j < T) ? row[i0 + 2 * j] : 0xfc00u;
            const uint32_t hi = (i0 + 2 * j + 1 < T) ? row[i0 + 2 * j + 1] : 0xfc00u;
            w[j] = lo | (hi << 16);
        }
    }
    k.x = order_key2(w[0]); k.y = order_key2(w[1]); k.z = order_key2(w[2]); k.w = order_key2(w[3]);
}

// Number of this thread's keys >= cand (both halves of each packed pair).
__device__ __forceinline__ int count_ge(const uint4& k, uint32_t cand) {
    const uint32_t w[4] = {k.x, k.y, k.z, k.w};
    int c = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) c += (int)((w[i] & 0xffffu) >= cand) + (int)((w[i] >> 16) >= cand);
    return c;
}

__global__ __launch_bounds__(THREADS) void dsa_select_kernel(const half* __restrict__ scores, int s_stride, const int* __restrict__ row_pos,
                                                           int* __restrict__ out, int2* __restrict__ meta) {
    const int r = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int pos = row_pos[r];
    if (!dsa_row_is_sparse(pos)) {
        if (t == 0) meta[r] = make_int2(0, 0);
        return;
    }
    const int T = dsa_row_pools(pos);
    const int tail = dsa_row_tail(pos);
    const uint16_t* row = reinterpret_cast<const uint16_t*>(scores) + (size_t)r * s_stride;
    int* orow = out + (size_t)r * DSA_K_STRIDE;
    const int nsup = (T + SUPER - 1) / SUPER;

    __shared__ int wsum[NWARPS], wpre[NWARPS], tile_total;
    __shared__ int red[2][NWARPS];  // double-buffered block-reduction scratch: one barrier per reduction
    int par = 0;

    uint4 keys[NTILE];  // this thread's keys of the current super-tile: tile j holds elements j * 8192 + 8 t .. +7
    auto load_super = [&](int sup) {
#pragma unroll
        for (int j = 0; j < NTILE; ++j) {
            const int i0 = sup * SUPER + j * TILE + t * EPT;
            if (i0 < T) load_tile(row, T, i0, keys[j]);
            else keys[j] = make_uint4(0, 0, 0, 0);
        }
    };
    // Block-wide count of keys >= cand over the whole row (an exact integer sum).
    auto block_count_ge = [&](uint32_t cand) -> int {
        int c = 0;
        for (int sup = 0; sup < nsup; ++sup) {
            if (nsup > 1) load_super(sup);
            const int ntiles = min(NTILE, (T - sup * SUPER + TILE - 1) / TILE);
#pragma unroll
            for (int j = 0; j < NTILE; ++j)
                if (j < ntiles) c += count_ge(keys[j], cand);
        }
        // hardware warp reduction, then one barrier: the 32 per-warp partials are summed again by every warp (NWARPS == 32 lanes)
        c = (int)__reduce_add_sync(0xffffffffu, (unsigned)c);
        if (lane == 0) red[par][warp] = c;
        __syncthreads();
        const int total = (int)__reduce_add_sync(0xffffffffu, (unsigned)red[par][lane]);
        par ^= 1;  // the next reduction writes the other buffer, so no second barrier is needed
        return total;
    };

    if (nsup == 1) load_super(0);
    const int valid_total = block_count_ge(1);
    const int k_eff = min(DSA_SEL_POOLS, valid_total);
    uint32_t thresh = 0;  // with <= k valid keys every valid key is above it
    int n_eq_take = 0;
    if (valid_total > DSA_SEL_POOLS) {
        for (int b = 15; b >= 0; --b) {
            const uint32_t cand = thresh | (1u << b);
            if (block_count_ge(cand) >= DSA_SEL_POOLS) thresh = cand;
        }
        const int n_gt = block_count_ge(thresh + 1);
        n_eq_take = DSA_SEL_POOLS - n_gt;
    }

    // ---- index-ordered emission
    int gt_base = 0, eq_base = 0;
    for (int sup = 0; sup < nsup; ++sup) {
        if (nsup > 1) load_super(sup);
        const int ntiles = min(NTILE, (T - sup * SUPER + TILE - 1) / TILE);
#pragma unroll
        for (int j = 0; j < NTILE; ++j) {
            if (j >= ntiles) break;
            const int i0 = sup * SUPER + j * TILE + t * EPT;
            const uint32_t kw[4] = {keys[j].x, keys[j].y, keys[j].z, keys[j].w};
            unsigned gtm = 0, eqm = 0;
#pragma unroll
            for (int e = 0; e < EPT; ++e) {
                const uint32_t key = (kw[e >> 1] >> (16 * (e & 1))) & 0xffffu;
                if (key > thresh) gtm |= 1u << e;
                else if (key == thresh && key != 0) eqm |= 1u << e;
            }
            // packed counts: above in the low 16 bits, equal in the high 16 (each at most 8192 per tile)
            const int packed = __popc(gtm) | (__popc(eqm) << 16);
            const int incl = warp_incl_scan(packed, lane);
            if (lane == 31) wsum[warp] = incl;
            __syncthreads();
            if (warp == 0) {
                const int x = wsum[lane];
                const int wi = warp_incl_scan(x, lane);
                wpre[lane] = wi - x;
                if (lane == 31) tile_total = wi;
            }
            __syncthreads();
            const int excl = wpre[warp] + incl - packed;
            int gt_before = gt_base + (excl & 0xffff), eq_before = eq_base + (excl >> 16);
            if (gtm | eqm) {
#pragma unroll
                for (int e = 0; e < EPT; ++e) {
                    if ((gtm >> e) & 1) {
                        const int pos_out = gt_before + min(eq_before, n_eq_take);
                        const int id = (i0 + e) * DSA_P;
                        *reinterpret_cast<int4*>(orow + pos_out * DSA_P) = make_int4(id, id + 1, id + 2, id + 3);
                        ++gt_before;
                    } else if ((eqm >> e) & 1) {
                        if (eq_before < n_eq_take) {
                            const int pos_out = gt_before + eq_before;
                            const int id = (i0 + e) * DSA_P;
                            *reinterpret_cast<int4*>(orow + pos_out * DSA_P) = make_int4(id, id + 1, id + 2, id + 3);
                        }
                        ++eq_before;
                    }
                }
            }
            gt_base += tile_total & 0xffff;
            eq_base += tile_total >> 16;
        }
    }
    __syncthreads();
    // ---- tail and padding: the entries after the selected pools
    const int found = min(gt_base + min(eq_base, n_eq_take), k_eff);
    const int first = found * DSA_P;
    const int tail0 = pos + 1 - tail;
    for (int i = t; i < DSA_K_STRIDE - first; i += THREADS) orow[first + i] = i < tail ? tail0 + i : -1;
    if (t == 0) meta[r] = make_int2(1, first + tail);
}

}  // namespace

GB_EXPORT int gb_dsa_select(const void* scores, int s_stride, const int* row_pos, int rows, int* out, int2* meta, void* stream) {
    if (rows <= 0) return 0;
    if (rows > DSA_MAX_ROWS || s_stride % 8) return (int)cudaErrorInvalidValue;
    dsa_select_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>((const half*)scores, s_stride, row_pos, out, meta);
    return GB_CHECK_LAUNCH();
}
