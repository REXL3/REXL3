// Dense bf16 GEMV of the Nemotron 3.5 Super VL engine: y[m, n] = x[m, k] . W[n, k]^T, W bf16 row-major (the HF [out, in]
// layout), x and y fp32, fp32 accumulation. No tensor cores, no cuBLAS.
//
// The contract is one fixed order of f32 operations per output (the reductions below), the same on every launch path and for every
// m, so that a row of y is bitwise the same whatever else is in the call. tests/nemotron_dense.rs writes the order out on the CPU
// (`exact_dot`) and compares every output with it.
//
// Lanes. A warp covers G = 2 column groups of C = 8 output columns each. Lane l = (g, s): g = l / S is the column group and
// s = l % S (S = 16) a k-stripe. Stripe s owns the 8-element chunks q = s + S * i (i = 0 .. k / (8 S)), so iteration i covers
// 8 S = 128 consecutive elements of a row. Column j of group g is column col0 + j * G + g.
//
// Reduction. Per row and column, a lane adds its chunks in increasing i (each chunk as two chains of four products, the sum
// added to the accumulator once). Every RED iterations the stripes of each (row, column) are combined by an xor tree (offsets 8,
// 4, 2, 1) and the total is kept by stripe 0 (the others restart at zero), so no serial chain spans more than RED chunks. The final
// combination is the same xor tree. The order of every operation depends only on k and the stripe, never on m, on the row tile,
// or on the column set.
//
// Two launch paths, one arithmetic:
//
//   * m <= 4, "register path" (`dense_gemv_kernel`): operands go straight from global memory to registers. One to four rows is a
//     pure weight stream; this path reads it at 225 to 231 GB/s. That is the ceiling of this layout on this part: the weight is
//     read in 256 B runs per column, every access pattern of that shape that was tried (runs of 256 B to 2 KB, whole columns per
//     warp, block-private sweeps, L2 prefetch, 1 to 16 warps per block, 1 to 8 columns per lane) lands at 225 to 234 GB/s, and only
//     a whole-buffer interleaved sweep, which a per-column GEMV cannot be, read 241 to 247.
//
//   * m >= 5, "pipeline path" (`dense_pipe_kernel`): a block of 8 warps owns 16 rows x 64 columns, the warp tile of the register
//     path (8 rows x 16 columns). A three-stage cp.async ring stages one iteration (128 elements of k) of the block's W columns and
//     x rows in shared memory, so a warp reads its operands with LDS and the latency of both streams is hidden behind the FMA
//     work. The register path's loads of an iteration are issued at its top and waited for before its FMAs, and its row groups
//     read 64 rows of x per 16 columns (8x the weight bytes through L2 at m = 64); the 16 x 64 tile halves that, and the W lines
//     shared by the row blocks of one column block (blockIdx.x is the fast grid axis) are served by L2 after the first.
//     * The x tile in shared memory is rotated inside each group of 8 pieces of 16 B, so that the cp.async writes (consecutive
//       pieces of a row) and the lanes' two 16 B loads per row (stride 32 B) both hit every bank of a quarter warp once; the
//       plain layouts conflict on one side or the other and run at 130 GB/s at m = 64.
//     * The cross-stripe reduction of a segment is formed at the start of the next segment, pair by pair, in front of the first
//       chunk add (so its shuffles overlap the FMAs of other pairs), and as a reduce-scatter: lane s keeps the value of the pair
//       (column j ^ (s >> 1), row r ^ ((s & 1) << 2)) in its register (j, r) (a per-lane permutation of the shared-memory
//       addresses), so that "v[p] += shfl_xor(v[p + h], off)" for h = 32, 16, 8, 4 and off = 8, 4, 2, 1 is the xor tree for
//       all 64 pairs of a lane with 60 shuffles instead of 256, and the totals of the pairs 4 s .. 4 s + 3 end up in lane s.
//       They go to stripe 0 (the other stripes get zeros) through 16 bytes of shared memory per lane. The tree is the same one:
//       fp add commutes, and the pair at each level is the same pair of partial sums.
//
// Any m >= 5 takes the pipeline path (m larger than 16 re-reads W per row block of 16 rows, from L2 when the blocks run together).

#include <stdint.h>

#include "common.cuh"

namespace {

// bf16 -> f32 is exact: element 2i is the low half of word i, element 2i + 1 the high half (little endian).
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// acc += x[0..8] . w[0..8]. The eight products are summed as two chains of four (fma within a chain, in element order), and the
// sum of the chains is added to acc once. A lane's serial chain then has one addition per chunk, as in a tree over the chunk.
__device__ __forceinline__ float chunk_add(float acc, const float4 a, const float4 b, const uint4 w) {
    float p = __fmul_rn(a.x, bf_lo(w.x));
    p = __fmaf_rn(a.y, bf_hi(w.x), p);
    p = __fmaf_rn(a.z, bf_lo(w.y), p);
    p = __fmaf_rn(a.w, bf_hi(w.y), p);
    float q = __fmul_rn(b.x, bf_lo(w.z));
    q = __fmaf_rn(b.y, bf_hi(w.z), q);
    q = __fmaf_rn(b.z, bf_lo(w.w), q);
    q = __fmaf_rn(b.w, bf_hi(w.w), q);
    return __fadd_rn(acc, __fadd_rn(p, q));
}

constexpr int S = 16;      // k-stripes per column (lanes of one column group)
constexpr int G = 32 / S;  // column groups per warp
constexpr int C = 8;       // output columns per lane
constexpr int RED = 8;     // iterations between cross-stripe reductions (a power of two)

// ---------------------------------------------------------------------------------------------------------------------------
// Register path (m <= 4).

// Every RED iterations the stripes of each (column, row) are combined. Stripe 0 keeps the total and the others restart at zero;
// the schedule depends only on k. Lanes of one column group are the 32 / G consecutive lanes sharing g.
template <int MT>
__device__ __forceinline__ void stripe_reduce(float (&acc)[C][MT], int s) {
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            float v = acc[j][r];
            for (int off = S / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            acc[j][r] = s == 0 ? v : 0.f;
        }
}

// Iterations i0 .. i0 + UU - 1 of a stripe for the MT rows r0 .. r0 + MT - 1. Lane (g, s) reads chunk s of column col0 + j * G
// for each j; columns past n read column n - 1 (a valid row) and are not stored. Every (row, column) pair sees the same chunk
// sequence whatever MT, the batch size or the row tile is.
template <int MT, int UU, bool CLAMP>
__device__ __forceinline__ void iter_batch(const uint4* __restrict__ w, int chunks, int col0, int n, const float4* __restrict__ x,
                                           int m, int r0, int xstride, int s, int i0, float (&acc)[C][MT]) {
    uint4 wv[UU][C];
#pragma unroll
    for (int u = 0; u < UU; ++u)
#pragma unroll
        for (int j = 0; j < C; ++j)
            wv[u][j] = __ldg(w + (size_t)min(col0 + j * G, n - 1) * chunks + s + (size_t)S * (i0 + u));
#pragma unroll
    for (int u = 0; u < UU; ++u) {
        const int q2 = 2 * (s + S * (i0 + u));  // float4 index of this lane's chunk within a row
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            const int rr = CLAMP ? min(r0 + r, m - 1) : r0 + r;
            const float4* xr = x + (size_t)rr * xstride + q2;
            const float4 a = __ldg(xr), b = __ldg(xr + 1);
#pragma unroll
            for (int j = 0; j < C; ++j) acc[j][r] = chunk_add(acc[j][r], a, b, wv[u][j]);
        }
    }
}

template <int MT, int UU, bool CLAMP>
__device__ __forceinline__ void stripe_pass(const uint4* __restrict__ w, int chunks, int col0, int n, const float4* __restrict__ x,
                                            int m, int r0, int s, float (&acc)[C][MT]) {
    const int steps = chunks / S;
    const int xstride = 2 * chunks;  // float4 per x row (a row is 8 * chunks floats)
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int r = 0; r < MT; ++r) acc[j][r] = 0.f;
    int i = 0;
    for (; i + UU <= steps; i += UU) {
        iter_batch<MT, UU, CLAMP>(w, chunks, col0, n, x, m, r0, xstride, s, i, acc);
        if (((i + UU) & (RED - 1)) == 0) stripe_reduce<MT>(acc, s);
    }
    for (; i < steps; ++i) {
        iter_batch<MT, 1, CLAMP>(w, chunks, col0, n, x, m, r0, xstride, s, i, acc);
        if (((i + 1) & (RED - 1)) == 0) stripe_reduce<MT>(acc, s);
    }
}

// Block (blockIdx.x, blockIdx.y): column sets blockIdx.x * CS + set, row groups blockIdx.y * RG + rg. Warp wi = set * RG + rg.
template <int MT, int WARPS, int UU, int MINB>
__global__ void __launch_bounds__(32 * WARPS, MINB) dense_gemv_kernel(const uint4* __restrict__ w, const float4* __restrict__ x,
                                                                       float* __restrict__ y, int m, int n, int chunks, int rg, int cs) {
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
    if (wi >= rg * cs) return;  // whole warp
    const int set = wi / rg, row_grp = wi % rg;
    const int r0 = (blockIdx.y * rg + row_grp) * MT;
    if (r0 >= m) return;  // whole warp: r0 is the same for all its lanes
    const int s = lane % S, g = lane / S;
    const int col0 = blockIdx.x * (cs * G * C) + set * (G * C) + g;
    float acc[C][MT];
    if (r0 + MT <= m) {
        stripe_pass<MT, UU, false>(w, chunks, col0, n, x, m, r0, s, acc);
    } else {
        stripe_pass<MT, UU, true>(w, chunks, col0, n, x, m, r0, s, acc);
    }
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int r = 0; r < MT; ++r)
            for (int off = S / 2; off > 0; off >>= 1) acc[j][r] += __shfl_xor_sync(0xffffffffu, acc[j][r], off);
    if (s == 0) {
#pragma unroll
        for (int j = 0; j < C; ++j) {
            const int col = col0 + j * G;
            if (col < n) {
#pragma unroll
                for (int r = 0; r < MT; ++r)
                    if (r0 + r < m) y[(size_t)(r0 + r) * n + col] = acc[j][r];
            }
        }
    }
}

template <int MT, int WARPS, int UU, int MINB>
int launch(const void* w, const void* x, void* y, int m, int n, int chunks, void* stream) {
    const int rgn = (m + MT - 1) / MT;         // row tiles
    const int rg = rgn < WARPS ? rgn : WARPS;  // row tiles per block
    const int cs = WARPS / rg;                 // column sets per block
    const int cols = cs * G * C;               // columns per block
    const dim3 grid((unsigned)((n + cols - 1) / cols), (unsigned)((rgn + rg - 1) / rg));
    dense_gemv_kernel<MT, WARPS, UU, MINB><<<grid, 32 * WARPS, 0, gb_stream(stream)>>>((const uint4*)w, (const float4*)x,
                                                                                      (float*)y, m, n, chunks, rg, cs);
    return GB_CHECK_LAUNCH();
}

// One row tile per warp (m <= 4): a warp covers 16 columns, a block of 8 warps 128. Small weights (the MTP head's 4,096-wide
// projections are 32 and 64 MB) would give a block of 8 warps fewer blocks than the part has SMs, so the block shrinks until there
// are at least two per SM; the columns of a lane do not depend on the block shape, so no bit moves.
template <int MT, int UU>
int launch_reg(const void* w, const void* x, void* y, int m, int n, int chunks, void* stream) {
    if (n >= 96 * 128) return launch<MT, 8, UU, 1>(w, x, y, m, n, chunks, stream);
    if (n >= 96 * 64) return launch<MT, 4, UU, 1>(w, x, y, m, n, chunks, stream);
    if (n >= 96 * 32) return launch<MT, 2, UU, 1>(w, x, y, m, n, chunks, stream);
    return launch<MT, 1, UU, 1>(w, x, y, m, n, chunks, stream);
}

// ---------------------------------------------------------------------------------------------------------------------------
// Pipeline path (m >= 5).
//
// Block (blockIdx.x, blockIdx.y): rows blockIdx.x * PIPE_ROWS .., columns blockIdx.y * PIPE_COLS ..; blockIdx.x is the fast axis of
// the grid, so the row blocks that read the same W columns run together and their W lines are served by L2 after the first.
// Warp wi = set * PIPE_RG + row_grp owns rows row_grp * 8 .. + 8 and the 16 columns of column set `set`, the register path's
// warp tile (MT = 8, C = 8, G = 2).
//
// Shared memory, per stage (one iteration = 128 elements of k):
//   W: PIPE_COLS columns x 256 B, column c at c * 256, the 16 chunks of the iteration in order (a lane's chunk s at s * 16);
//   x: PIPE_ROWS rows x 512 B, row r at r * 512. A row's 32 16-byte pieces e = 0..31 (chunk s is pieces 2 s and 2 s + 1) are stored
//      at piece position (e & 24) | ((e + (e >> 3)) & 7): a rotation inside each aligned group of 8 pieces.
// After the ring: 16 float4 of totals per warp group (8 warps x 2 groups) and 16 float4 of zeros shared by all lanes.

constexpr int PIPE_WARPS = 8;
constexpr int PIPE_RG = 2;                                  // row groups (of 8 rows) per block
constexpr int PIPE_CS = PIPE_WARPS / PIPE_RG;               // column sets (of 16 columns) per block
constexpr int PIPE_ROWS = 8 * PIPE_RG;                      // 16
constexpr int PIPE_COLS = 16 * PIPE_CS;                     // 64
constexpr int PIPE_NST = 3;                                 // stages in the ring
constexpr int PIPE_XB = PIPE_ROWS * 512;                    // x bytes per stage
constexpr int PIPE_STAGE = PIPE_XB + PIPE_COLS * 256;       // bytes per stage
constexpr int PIPE_RING = PIPE_NST * PIPE_STAGE;            // 73,728 B
constexpr int PIPE_ZERO = PIPE_RING + PIPE_WARPS * 2 * 256; // the zero area follows the totals
constexpr int PIPE_SMEM = PIPE_ZERO + 256;                  // 78,080 B (the opt-in limit of the part is 101,376 B)
constexpr int PIPE_WPT = PIPE_COLS / 16;                    // W pieces a thread loads per stage (256 threads x 16 B)
constexpr int PIPE_XPT = PIPE_ROWS * 32 / 256;              // x pieces a thread loads per stage

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <bool B>
struct BoolC {
    static constexpr bool value = B;
};

// byte offset of 16-byte piece e of an x row in shared memory
__device__ __forceinline__ uint32_t swz(int e) { return (uint32_t)(((e & 24) | ((e + (e >> 3)) & 7)) * 16); }

// Reduce-scatter of the 64 values of a lane over the 16 stripes of its group, in the order of the xor tree (offsets 8, 4, 2, 1). The
// value in register p of lane s belongs to the pair p ^ perm(s), perm(s) = (s >> 1) << 3 | (s & 1) << 2 (flat pair index 8 j + r);
// each level adds the partner's upper half to the lane's lower half, which for both lanes of a pair is the same pair of partial
// sums. Afterwards v[0..3] of lane s hold the totals of the pairs 4 s .. 4 s + 3.
__device__ __forceinline__ void scatter(float (&a)[C][8]) {
    float* v = &a[0][0];
#pragma unroll
    for (int p = 0; p < 32; ++p) v[p] += __shfl_xor_sync(0xffffffffu, v[p + 32], 8);
#pragma unroll
    for (int p = 0; p < 16; ++p) v[p] += __shfl_xor_sync(0xffffffffu, v[p + 16], 4);
#pragma unroll
    for (int p = 0; p < 8; ++p) v[p] += __shfl_xor_sync(0xffffffffu, v[p + 8], 2);
#pragma unroll
    for (int p = 0; p < 4; ++p) v[p] += __shfl_xor_sync(0xffffffffu, v[p + 4], 1);
}

__global__ void __launch_bounds__(32 * PIPE_WARPS, 1) dense_pipe_kernel(const uint4* __restrict__ w, const float4* __restrict__ x,
                                                                         float* __restrict__ y, int m, int n, int chunks) {
    constexpr int MT = 8;
    extern __shared__ __align__(16) unsigned char smem[];
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    const int tid = threadIdx.x, lane = tid & 31, wi = tid >> 5;
    const int row_grp = wi % PIPE_RG, set = wi / PIPE_RG;
    const int row0 = blockIdx.x * PIPE_ROWS, col0b = blockIdx.y * PIPE_COLS;
    const int steps = chunks / S;
    const int xstride = 2 * chunks;  // float4 per x row

    // this thread's cp.async pieces: W piece (column tid / 16 + 16 u, chunk tid % 16) and x pieces (row (tid + 256 u) / 32, e = tid % 32).
    // Columns past n and rows past m read the last valid one; they are never stored.
    const uint4* wsrc[PIPE_WPT];
    uint32_t wdst[PIPE_WPT];
#pragma unroll
    for (int u = 0; u < PIPE_WPT; ++u) {
        const int col = (tid >> 4) + 16 * u, q = tid & 15;
        wsrc[u] = w + (size_t)min(col0b + col, n - 1) * chunks + q;
        wdst[u] = PIPE_XB + col * 256 + q * 16;
    }
    const float4* xsrc[PIPE_XPT];
    uint32_t xdst[PIPE_XPT];
#pragma unroll
    for (int u = 0; u < PIPE_XPT; ++u) {
        const int p = tid + 256 * u;
        const int row = p >> 5, e = p & 31;
        xsrc[u] = x + (size_t)min(row0 + row, m - 1) * xstride + e;
        xdst[u] = row * 512 + swz(e);
    }
    auto issue = [&](int it, int slot) {
        const uint32_t sb = sbase + slot * PIPE_STAGE;
#pragma unroll
        for (int u = 0; u < PIPE_WPT; ++u) cp_async16(sb + wdst[u], wsrc[u] + (size_t)16 * it);
#pragma unroll
        for (int u = 0; u < PIPE_XPT; ++u) cp_async16(sb + xdst[u], xsrc[u] + (size_t)32 * it);
    };

    // This lane: stripe s, column group g. Its register (j, r) works on column j ^ t and row r ^ (u << 2) of the warp tile, with
    // t = s >> 1 and u = s & 1 (the permutation the reduce-scatter wants); rows 0..3 / 4..7 are shifted by +- 4 u rows.
    const int s = lane & 15, g = lane >> 4;
    const int t = s >> 1, u1 = s & 1;
    const uint32_t xrow = (row_grp * MT) * 512;
    const uint32_t xa_lo = xrow + swz(2 * s) + u1 * 2048, xa_hi = xrow + swz(2 * s) - u1 * 2048;          // x chunk s, elements 0..3
    const uint32_t xb_lo = xrow + swz(2 * s + 1) + u1 * 2048, xb_hi = xrow + swz(2 * s + 1) - u1 * 2048;  // elements 4..7
    uint32_t wo[C];
#pragma unroll
    for (int j = 0; j < C; ++j) wo[j] = PIPE_XB + (set * 16 + g + 2 * (j ^ t)) * 256 + s * 16;

    // totals of this lane's pairs go to scr + s * 16; stripe 0 reads all 16 float4 of its group from scr, the others zeros
    const uint32_t scr = PIPE_RING + (wi * 2 + g) * 256;
    if (tid < 16) *(float4*)(smem + PIPE_ZERO + tid * 16) = make_float4(0.f, 0.f, 0.f, 0.f);

    float acc[C][MT];
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int r = 0; r < MT; ++r) acc[j][r] = 0.f;

#pragma unroll
    for (int st = 0; st < PIPE_NST - 1; ++st) {
        if (st < steps) issue(st, st);
        cp_async_commit();
    }
    int slot = 0;

    // One iteration. RF: the cross-stripe totals of the segment that just ended are formed before the first chunk add of the new
    // one (stripe 0 starts from the total, the other stripes from zero); this is the reduction of the register path at the end of a
    // segment, performed later, with the same tree and the same operands.
    auto iteration = [&](int i, auto rf) {
        constexpr bool RF = decltype(rf)::value;
        cp_async_wait<PIPE_NST - 2>();
        __syncthreads();  // stage i has landed; the slot about to be refilled was read by every warp in iteration i - 1
        {
            const int nx = i + PIPE_NST - 1;
            if (nx < steps) issue(nx, nx % PIPE_NST);
            cp_async_commit();
        }
        if (RF) {
            scatter(acc);
            *(float4*)(smem + scr + s * 16) = make_float4(acc[0][0], acc[0][1], acc[0][2], acc[0][3]);
            __syncwarp();
            const uint32_t rd = (s == 0) ? scr : (uint32_t)PIPE_ZERO;
#pragma unroll
            for (int q = 0; q < 16; ++q) {  // pairs 4 q .. 4 q + 3 are acc[q >> 1][(q & 1) * 4 ..]
                const float4 f = *(const float4*)(smem + rd + q * 16);
                acc[q >> 1][(q & 1) * 4 + 0] = f.x;
                acc[q >> 1][(q & 1) * 4 + 1] = f.y;
                acc[q >> 1][(q & 1) * 4 + 2] = f.z;
                acc[q >> 1][(q & 1) * 4 + 3] = f.w;
            }
        }
        const unsigned char* sb = smem + slot * PIPE_STAGE;
        uint4 wv[C];
#pragma unroll
        for (int j = 0; j < C; ++j) wv[j] = *(const uint4*)(sb + wo[j]);
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            const float4 a = *(const float4*)(sb + (r < 4 ? xa_lo : xa_hi) + r * 512);
            const float4 b = *(const float4*)(sb + (r < 4 ? xb_lo : xb_hi) + r * 512);
#pragma unroll
            for (int j = 0; j < C; ++j) acc[j][r] = chunk_add(acc[j][r], a, b, wv[j]);
        }
        slot = slot + 1 == PIPE_NST ? 0 : slot + 1;
    };

    for (int i = 0; i < steps; ++i) {
        if (i > 0 && (i & (RED - 1)) == 0)
            iteration(i, BoolC<true>{});
        else
            iteration(i, BoolC<false>{});
    }

    // final combination: the same xor tree over all stripes (when k / 128 is a multiple of RED the last segment's totals are formed
    // here, which is the same tree over the same operands as the end-of-segment reduction followed by a sum with zeros). Lane s
    // ends with the totals of the pairs 4 s .. 4 s + 3: column (s >> 1) of its tile, rows ((s & 1) << 2) + 0..3.
    scatter(acc);
    const int col = col0b + set * 16 + (s >> 1) * G + g;
    if (col < n) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int row = row0 + row_grp * MT + ((s & 1) << 2) + p;
            if (row < m) y[(size_t)row * n + col] = acc[0][p];
        }
    }
}

int launch_pipe(const void* w, const void* x, void* y, int m, int n, int chunks, void* stream) {
    // one-time opt-in to the dynamic shared memory above 48 KB
    static const cudaError_t configured =
        cudaFuncSetAttribute(dense_pipe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, PIPE_SMEM);
    if (configured != cudaSuccess) return (int)configured;
    const unsigned gx = (unsigned)((m + PIPE_ROWS - 1) / PIPE_ROWS), gy = (unsigned)((n + PIPE_COLS - 1) / PIPE_COLS);
    if (gy > 65535u) return (int)cudaErrorInvalidValue;  // gridDim.y limit (n above 4.19 M columns)
    dense_pipe_kernel<<<dim3(gx, gy), 32 * PIPE_WARPS, PIPE_SMEM, gb_stream(stream)>>>((const uint4*)w, (const float4*)x, (float*)y,
                                                                                         m, n, chunks);
    return GB_CHECK_LAUNCH();
}

}  // namespace

// w bf16 [n, k], x f32 [m, k], y f32 [m, n], all 16-byte aligned. k must be a multiple of 8 and one of the five supported
// widths (1024, 2688, 4096, 5376, 8192); anything else returns cudaErrorInvalidValue without launching.
GB_EXPORT int gb_nemo_dense_gemv(const void* w, const void* x, void* y, int m, int n, int k, void* stream) {
    if (m < 0 || n < 0 || k <= 0 || k % 8 != 0) return (int)cudaErrorInvalidValue;
    if (m == 0 || n == 0) return 0;
    const int chunks = k / 8;
    switch (chunks) {
        case 128:   // k = 1024
        case 336:   // k = 2688
        case 512:   // k = 4096
        case 672:   // k = 5376
        case 1024:  // k = 8192
            break;
        default:
            return (int)cudaErrorInvalidValue;
    }
    if (m >= 5) return launch_pipe(w, x, y, m, n, chunks, stream);
    switch (m) {  // rows per warp: the smallest power of two >= m
        case 1: return launch_reg<1, 2>(w, x, y, m, n, chunks, stream);
        case 2: return launch_reg<2, 2>(w, x, y, m, n, chunks, stream);
        default: return launch_reg<4, 1>(w, x, y, m, n, chunks, stream);
    }
}
