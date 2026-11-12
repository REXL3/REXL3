// gb-nvcc: --fmad=false
// Glue kernels of the Nemotron 3.5 Super VL layer loop: the embedding gather, the strided RMSNorm with a bf16 weight, the Mamba2
// in_proj split, the load-time widens, the f32 top-k over the lm_head logits, and the image-feature splice.
//
// Numerics. Every float operation is an explicit IEEE-754 binary32 operation (__fmul_rn, __fadd_rn, __fdiv_rn, __fsqrt_rn): no
// multiply-add is contracted (the file is also built with --fmad=false), there is no rsqrt and no fast-math, so a CPU model that
// does the same operations in the same order matches bit for bit. Subnormals are kept (no flush to zero). The copies are exact.
//
// rmsnorm_bf16w_f32. One block of 256 threads per row. The tree below does not depend on `rows`, on the strides or on `d`, so a
// row has the same bits in a 1-row and in a 64-row call:
//   1. thread t (0 .. 255) accumulates its strided elements in ascending order: acc = 0; for c = t, t + 256, t + 512, ... < d:
//      acc = acc + x[c] * x[c]   (the product and the sum each rounded to f32). A thread with t >= d keeps 0.
//   2. warp tree: for off = 16, 8, 4, 2, 1: v = v + shfl_xor(v, off). Both lanes of a pair add the same two numbers, so every lane
//      of the warp ends with the same bits; s_w is that value of warp w (w = 0 .. 7).
//   3. across warps, a fixed balanced tree: total = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)).
//   4. mean = total / (float)d; inv = 1.0f / sqrtf(mean + eps)   (IEEE division and square root).
//   5. out[c] = (x[c] * inv) * w[c], with w widened from bf16 (exact), two f32 products.
//
// split_inproj. z is rounded to the bf16 grid, round to nearest even (the bf16 model's cast at the in_proj output) and stored as
// f32; a NaN becomes the quiet NaN 0x7fc00000 (what torch's BFloat16 cast gives). xBC and dt are exact copies.
//
// topk_f32. The order is a strict total order on (value desc, id asc): the value is mapped to a u32 that sorts like the float
// (-0.0 and +0.0 map to the same u32, NaN to 0, below -inf), and the key is that u32 << 32 | (0xffffffff - id). Because the keys are
// distinct, every selection algorithm returns the same set in the same order, so the result is the stable sort's, whatever the
// split into blocks. Key 0 is reserved for "no candidate": no real element has it.
// Stage 1 (grid [blocks per row, rows], 8 warps per block): each warp scans its contiguous slice of the row 128 elements at a
// time and keeps the best 64 keys in registers, sorted descending, lane l holding entries l and 32 + l. A key enters when it
// beats entry `count - 1` (the lowest entry that can still be in the answer). A window with a few passing keys inserts them one
// by one (ballots to rank, shuffles to shift); a window with more is gathered, ranked against itself and merged into the list by
// ranks (every element's place is its index plus the number of greater elements on the other side, found by binary search in
// shared memory). Stage 2 (one block per row): each warp merges its share of the row's lists the same way, and warp 0 merges the
// 8 results, then writes ids and values (re-read from the logits, so a NaN payload and the sign of a zero come back as stored).
// The workspace is the stage-1 lists: rows * blocks * 8 warps * count u64.

#include <math.h>
#include <stdint.h>

#include "common.cuh"

namespace {

constexpr int HIDDEN = 4096;
constexpr int INPROJ_COLS = 18560;
constexpr int Z_COLS = 8192;
constexpr int XBC_COLS = 10240;
constexpr int DT_COLS = 128;
constexpr uint32_t FULL = 0xffffffffu;

// bf16 -> f32 is exact: element 2i is the low half of word i, element 2i + 1 the high half (little endian).
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// Eight bf16 (one 16-byte load) as two float4.
__device__ __forceinline__ void widen8_bf16(const uint4 w, float4* __restrict__ dst) {
    dst[0] = make_float4(bf_lo(w.x), bf_hi(w.x), bf_lo(w.y), bf_hi(w.y));
    dst[1] = make_float4(bf_lo(w.z), bf_hi(w.z), bf_lo(w.w), bf_hi(w.w));
}

__device__ __forceinline__ float f16_to_f32(uint32_t h) { return __half2float(__ushort_as_half((unsigned short)h)); }

// f32 -> the nearest bf16 (ties to even) as an f32. A NaN is canonicalised first (the integer trick would turn some into inf).
__device__ __forceinline__ float round_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return __uint_as_float(0x7fc00000u);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(u);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Gather and splice: bf16 rows widened to f32.

// One bf16 row of 4096 widened into one f32 row, 16 bytes per load; the block's 256 threads each take two loads.
__device__ __forceinline__ void widen_row(const uint4* __restrict__ src, float4* __restrict__ dst) {
    for (int q = threadIdx.x; q < HIDDEN / 8; q += blockDim.x) widen8_bf16(__ldg(src + q), dst + 2 * q);
}

__global__ void __launch_bounds__(256) embed_gather_kernel(const uint4* __restrict__ table, const int* __restrict__ ids, int vocab,
                                                           float4* __restrict__ out) {
    const int r = blockIdx.x;
    const int id = ids[r];
    float4* dst = out + (size_t)r * (HIDDEN / 4);
    if (id < 0 || id >= vocab) {
        for (int q = threadIdx.x; q < HIDDEN / 4; q += blockDim.x) dst[q] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }
    widen_row(table + (size_t)id * (HIDDEN / 8), dst);
}

__global__ void __launch_bounds__(256) splice_rows_kernel(const uint4* __restrict__ feats, float4* __restrict__ dst) {
    const int r = blockIdx.x;
    widen_row(feats + (size_t)r * (HIDDEN / 8), dst + (size_t)r * (HIDDEN / 4));
}

// ---------------------------------------------------------------------------------------------------------------------------------
// RMSNorm: see the tree in the file header.

constexpr int NORM_THREADS = 256;

__global__ void __launch_bounds__(NORM_THREADS) rmsnorm_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                                                float* __restrict__ out, int d, long long x_stride,
                                                                long long out_stride, float eps) {
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * (size_t)x_stride;
    float acc = 0.f;
#pragma unroll 4
    for (int c = threadIdx.x; c < d; c += NORM_THREADS) {
        const float v = __ldg(xr + c);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(FULL, acc, off));
    __shared__ float part[NORM_THREADS / 32];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    const float total = __fadd_rn(__fadd_rn(__fadd_rn(part[0], part[1]), __fadd_rn(part[2], part[3])),
                                  __fadd_rn(__fadd_rn(part[4], part[5]), __fadd_rn(part[6], part[7])));
    const float mean = __fdiv_rn(total, (float)d);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    float* o = out + (size_t)row * (size_t)out_stride;
#pragma unroll 4
    for (int c = threadIdx.x; c < d; c += NORM_THREADS) {
        const float wv = __uint_as_float((uint32_t)__ldg(w + c) << 16);
        o[c] = __fmul_rn(__fmul_rn(__ldg(xr + c), inv), wv);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// in_proj split.

constexpr int INPROJ_V4 = INPROJ_COLS / 4;  // float4 per row
constexpr int Z_V4 = Z_COLS / 4;
constexpr int XBC_V4 = XBC_COLS / 4;
constexpr int DT_V4 = DT_COLS / 4;

__global__ void __launch_bounds__(256) split_inproj_kernel(const float4* __restrict__ y, float4* __restrict__ z, float4* __restrict__ xbc,
                                                           float4* __restrict__ dt, size_t total) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / INPROJ_V4;
    const int q = (int)(i - r * INPROJ_V4);
    float4 v = __ldg(y + i);
    if (q < Z_V4) {
        v.x = round_bf16(v.x);
        v.y = round_bf16(v.y);
        v.z = round_bf16(v.z);
        v.w = round_bf16(v.w);
        z[r * Z_V4 + q] = v;
    } else if (q < Z_V4 + XBC_V4) {
        xbc[r * XBC_V4 + (q - Z_V4)] = v;
    } else {
        dt[r * DT_V4 + (q - Z_V4 - XBC_V4)] = v;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Widens.

template <bool F16>
__global__ void __launch_bounds__(256) widen_kernel(const uint16_t* __restrict__ src, long long n, float* __restrict__ dst) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t t0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t groups = (size_t)n / 8;
    for (size_t g = t0; g < groups; g += stride) {
        const uint4 w = __ldg(reinterpret_cast<const uint4*>(src) + g);
        float4* d4 = reinterpret_cast<float4*>(dst) + 2 * g;
        if (F16) {
            d4[0] = make_float4(f16_to_f32(w.x & 0xffffu), f16_to_f32(w.x >> 16), f16_to_f32(w.y & 0xffffu), f16_to_f32(w.y >> 16));
            d4[1] = make_float4(f16_to_f32(w.z & 0xffffu), f16_to_f32(w.z >> 16), f16_to_f32(w.w & 0xffffu), f16_to_f32(w.w >> 16));
        } else {
            widen8_bf16(w, d4);
        }
    }
    for (size_t i = groups * 8 + t0; i < (size_t)n; i += stride) {
        const uint32_t h = __ldg(src + i);
        dst[i] = F16 ? f16_to_f32(h) : __uint_as_float(h << 16);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Top-k.

typedef unsigned long long u64;

constexpr int TK_THREADS = 256;
constexpr int TK_WARPS = TK_THREADS / 32;  // candidate lists per block (the Rust side's TOPK_WARPS)
constexpr int TK_U = 4;                    // elements per lane per window
constexpr int TK_WIN = 32 * TK_U;          // elements a warp takes in at once (and the alignment of every slice)
constexpr int TK_DENSE = 3;                // a window with more candidates than this is sorted and merged, not inserted one by one

// The sortable key of element `id` with value `f`: larger is better. See the file header.
__device__ __forceinline__ u64 topk_key(float f, uint32_t id) {
    const uint32_t u = __float_as_uint(f);
    uint32_t k;
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        k = 0;  // NaN: below every number, including -inf (0x007fffff)
    } else if ((u << 1) == 0u) {
        k = 0x80000000u;  // -0.0 and +0.0 compare equal
    } else {
        k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    return ((u64)k << 32) | (u64)(0xffffffffu - id);
}

__device__ __forceinline__ u64 shfl_u64(u64 v, int src) {
    const uint32_t lo = __shfl_sync(FULL, (uint32_t)v, src);
    const uint32_t hi = __shfl_sync(FULL, (uint32_t)(v >> 32), src);
    return ((u64)hi << 32) | lo;
}

__device__ __forceinline__ u64 shfl_up_u64(u64 v, int delta) {
    const uint32_t lo = __shfl_up_sync(FULL, (uint32_t)v, delta);
    const uint32_t hi = __shfl_up_sync(FULL, (uint32_t)(v >> 32), delta);
    return ((u64)hi << 32) | lo;
}

// A warp's best-64 list: entry i is `a` of lane i (i < 32) or `b` of lane i - 32, sorted descending; 0 is "empty". `thr` is entry
// count - 1 (warp-uniform): a key at or below it cannot be among the best `count`. Entries at or past `count` are keys that passed
// the filter earlier (valid, but not necessarily the next best): only entries below `count` are exact.
struct TopList {
    u64 a, b, thr;
};

// Per-warp shared scratch: the list being merged (x), the sorted run it is merged with (y), the merge's output (z) and an unsorted
// window of candidates (w).
struct WarpScratch {
    u64 x[64];
    u64 y[64];
    u64 z[64];
    u64 w[TK_WIN];
};

// Inserts `v` (> thr, warp-uniform): the entries above it stay, the rest move down one place and the last one drops out.
__device__ __forceinline__ void list_insert(TopList& t, u64 v, int lane, int count) {
    const unsigned ba = __ballot_sync(FULL, t.a > v);
    const unsigned bb = __ballot_sync(FULL, t.b > v);
    const int p = __popc(ba) + __popc(bb);  // entries greater than v: a prefix of the sorted list
    const u64 a_up = shfl_up_u64(t.a, 1);   // entry lane - 1
    const u64 b_up = shfl_up_u64(t.b, 1);   // entry 32 + lane - 1 (lane 0 needs entry 31, the `a` of lane 31)
    const u64 a31 = shfl_u64(t.a, 31);
    const u64 b_prev = lane == 0 ? a31 : b_up;
    t.a = lane < p ? t.a : (lane == p ? v : a_up);
    const int ib = 32 + lane;
    t.b = ib < p ? t.b : (ib == p ? v : b_prev);
    const int j = count - 1;  // warp-uniform
    t.thr = shfl_u64(j < 32 ? t.a : t.b, j & 31);
}

// Offers one key per lane (0 = none). Warp-collective.
__device__ __forceinline__ void list_offer(TopList& t, u64 c, int lane, int count) {
    unsigned m = __ballot_sync(FULL, c > t.thr);
    while (m) {
        const int src = __ffs(m) - 1;
        m &= m - 1;
        const u64 v = shfl_u64(c, src);
        if (v > t.thr) list_insert(t, v, lane, count);  // an earlier insert may have raised thr past v
    }
}

// The number of entries of the 64-entry descending array `s` (shared memory, zero padded) that are greater than `k`.
__device__ __forceinline__ int count_greater(const u64* s, u64 k) {
    int pos = 0;
#pragma unroll
    for (int step = 32; step > 0; step >>= 1) pos += (s[pos + step - 1] > k) ? step : 0;
    return pos + ((pos == 63 && s[63] > k) ? 1 : 0);
}

// Merges the sorted run S.y (64 entries, descending, zero padded) into the list: the new list is the best 64 of both. Every element's
// place is its own index plus the number of the other side's elements that are greater (the keys are distinct, and 0 is skipped), so
// the whole merge is a handful of dependent shared-memory reads instead of one insertion chain per element. Warp-collective.
__device__ __forceinline__ void list_merge(TopList& t, WarpScratch& S, int lane, int count) {
    S.x[lane] = t.a;
    S.x[32 + lane] = t.b;
    S.z[lane] = 0;
    S.z[32 + lane] = 0;
    __syncwarp();
    const u64 y0 = S.y[lane], y1 = S.y[32 + lane];
    const int rx0 = lane + count_greater(S.y, t.a), rx1 = 32 + lane + count_greater(S.y, t.b);
    const int ry0 = lane + count_greater(S.x, y0), ry1 = 32 + lane + count_greater(S.x, y1);
    if (t.a != 0 && rx0 < 64) S.z[rx0] = t.a;
    if (t.b != 0 && rx1 < 64) S.z[rx1] = t.b;
    if (y0 != 0 && ry0 < 64) S.z[ry0] = y0;
    if (y1 != 0 && ry1 < 64) S.z[ry1] = y1;
    __syncwarp();
    t.a = S.z[lane];
    t.b = S.z[32 + lane];
    t.thr = S.z[count - 1];
    __syncwarp();  // the next merge may overwrite x, y and z
}

// A window of TK_U keys per lane in which more than TK_DENSE keys pass the filter: the passing keys are gathered, ranked against
// each other (a rank is the number of greater keys: one broadcast read per key) and scattered sorted into S.y, then merged.
__device__ __forceinline__ void list_dense_window(TopList& t, WarpScratch& S, const u64 (&c)[TK_U], int lane, int count) {
    int n = 0;
#pragma unroll
    for (int u = 0; u < TK_U; ++u) {
        const bool pass = c[u] > t.thr;
        const unsigned m = __ballot_sync(FULL, pass);
        if (pass) S.w[n + __popc(m & ((1u << lane) - 1u))] = c[u];
        n += __popc(m);
    }
    S.y[lane] = 0;
    S.y[32 + lane] = 0;
    __syncwarp();
    u64 v[TK_U];
    int r[TK_U];
#pragma unroll
    for (int h = 0; h < TK_U; ++h) {
        const int e = h * 32 + lane;
        v[h] = e < n ? S.w[e] : 0ull;
        r[h] = 0;
    }
#pragma unroll 4
    for (int j = 0; j < n; ++j) {
        const u64 wj = S.w[j];
#pragma unroll
        for (int h = 0; h < TK_U; ++h) r[h] += (wj > v[h]) ? 1 : 0;
    }
#pragma unroll
    for (int h = 0; h < TK_U; ++h)
        if (v[h] != 0 && r[h] < 64) S.y[r[h]] = v[h];
    __syncwarp();
    list_merge(t, S, lane, count);
}

__device__ __forceinline__ long long round_up(long long v, long long m) { return (v + m - 1) / m * m; }

// Stage 1: grid (blocks per row, rows). Warp w of block s scans its contiguous part of the row's slice (slice and part boundaries
// are multiples of TK_WIN, so a window is a run of 512 aligned bytes when the row is) and writes its best `count` keys, sorted, to
// ws[((row * B + s) * 8 + w) * count + i] (0 for a missing entry).
__global__ void __launch_bounds__(TK_THREADS) topk_stage1_kernel(const float* __restrict__ logits, int vocab, int count, int blocks,
                                                                  u64* __restrict__ ws) {
    __shared__ WarpScratch scratch[TK_WARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.y, split = blockIdx.x;
    WarpScratch& S = scratch[warp];
    const float* x = logits + (size_t)row * (size_t)vocab;
    const long long slice = round_up(((long long)vocab + blocks - 1) / blocks, TK_WIN);
    const long long lo = min((long long)split * slice, (long long)vocab);
    const long long hi = min(lo + slice, (long long)vocab);
    const long long per_warp = round_up((slice + TK_WARPS - 1) / TK_WARPS, TK_WIN);
    const long long wlo = min(lo + (long long)warp * per_warp, hi);
    const long long whi = min(wlo + per_warp, hi);

    TopList t;
    t.a = 0;
    t.b = 0;
    t.thr = 0;
    float cur[TK_U], nxt[TK_U];
#pragma unroll
    for (int u = 0; u < TK_U; ++u) {
        const long long e = wlo + u * 32 + lane;
        cur[u] = e < whi ? __ldg(x + e) : 0.f;
    }
    for (long long base = wlo; base < whi; base += TK_WIN) {
#pragma unroll
        for (int u = 0; u < TK_U; ++u) {
            const long long e = base + TK_WIN + u * 32 + lane;
            nxt[u] = e < whi ? __ldg(x + e) : 0.f;
        }
        u64 c[TK_U];
        int pass = 0;
#pragma unroll
        for (int u = 0; u < TK_U; ++u) {
            const long long e = base + u * 32 + lane;
            c[u] = e < whi ? topk_key(cur[u], (uint32_t)e) : 0ull;
            pass += __popc(__ballot_sync(FULL, c[u] > t.thr));
        }
        if (pass > TK_DENSE) {
            list_dense_window(t, S, c, lane, count);
        } else if (pass > 0) {
#pragma unroll
            for (int u = 0; u < TK_U; ++u) list_offer(t, c[u], lane, count);
        }
#pragma unroll
        for (int u = 0; u < TK_U; ++u) cur[u] = nxt[u];
    }
    u64* o = ws + ((size_t)row * blocks + split) * TK_WARPS * count + (size_t)warp * count;
    if (lane < count) o[lane] = t.a;
    if (32 + lane < count) o[32 + lane] = t.b;
}

// Stage 2: one block per row. `lists` = blocks * 8 sorted candidate lists of `count` keys each: warp w merges lists w, w + 8, ...
// into one, and warp 0 merges those 8.
__global__ void __launch_bounds__(TK_THREADS) topk_stage2_kernel(const float* __restrict__ logits, int vocab, int count, int lists,
                                                                  const u64* __restrict__ ws, int* __restrict__ ids,
                                                                  float* __restrict__ vals) {
    __shared__ WarpScratch scratch[TK_WARPS];
    __shared__ u64 mid[TK_WARPS * 64];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x;
    WarpScratch& S = scratch[warp];
    const u64* cand = ws + (size_t)row * lists * count;

    TopList t;
    t.a = 0;
    t.b = 0;
    t.thr = 0;
    // the next list's loads are issued before the merge of this one
    u64 n0 = 0, n1 = 0;
    if (warp < lists) {
        const u64* src = cand + (size_t)warp * count;
        n0 = lane < count ? src[lane] : 0ull;
        n1 = 32 + lane < count ? src[32 + lane] : 0ull;
    }
    for (int l = warp; l < lists; l += TK_WARPS) {
        S.y[lane] = n0;
        S.y[32 + lane] = n1;
        if (l + TK_WARPS < lists) {
            const u64* src = cand + (size_t)(l + TK_WARPS) * count;
            n0 = lane < count ? src[lane] : 0ull;
            n1 = 32 + lane < count ? src[32 + lane] : 0ull;
        }
        list_merge(t, S, lane, count);
    }
    mid[warp * 64 + lane] = t.a;
    mid[warp * 64 + 32 + lane] = t.b;
    __syncthreads();
    if (warp != 0) return;
    for (int w = 1; w < TK_WARPS; ++w) {
        S.y[lane] = mid[w * 64 + lane];
        S.y[32 + lane] = mid[w * 64 + 32 + lane];
        list_merge(t, S, lane, count);
    }
    const float* x = logits + (size_t)row * (size_t)vocab;
    const u64 mine[2] = {t.a, t.b};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = h * 32 + lane;
        if (r < count) {
            const u64 key = mine[h];
            const int id = key != 0 ? (int)(0xffffffffu - (uint32_t)key) : -1;  // key 0 cannot occur: count <= vocab
            ids[(size_t)row * count + r] = id;
            vals[(size_t)row * count + r] = id >= 0 ? x[id] : __uint_as_float(0x7fc00000u);
        }
    }
}

bool aligned(const void* p, size_t a) { return ((uintptr_t)p & (a - 1)) == 0; }

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// Launchers. Every one returns 0, cudaErrorInvalidValue for an argument outside its contract (nothing launched), or the launch
// status; a zero-sized call is a successful no-op.

// table bf16 [vocab, 4096], ids i32 [rows], out f32 [rows, 4096]; table and out 16-byte aligned.
GB_EXPORT int gb_nemo_glue_embed_gather(const void* table, const void* ids, int vocab, int rows, void* out, void* stream) {
    if (rows < 0 || vocab < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    if (!aligned(table, 16) || !aligned(out, 16) || !aligned(ids, 4)) return (int)cudaErrorInvalidValue;
    embed_gather_kernel<<<rows, 256, 0, gb_stream(stream)>>>((const uint4*)table, (const int*)ids, vocab, (float4*)out);
    return GB_CHECK_LAUNCH();
}

// x f32 with row stride x_stride, w bf16 [d], out f32 with row stride out_stride (elements). Rows of `out` must not overlap.
GB_EXPORT int gb_nemo_glue_rmsnorm(const void* x, const void* w, int rows, int d, long long x_stride, float eps, void* out,
                                   long long out_stride, void* stream) {
    if (rows < 0 || d < 0 || x_stride < 0 || out_stride < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    if (d == 0 || (rows > 1 && (x_stride < d || out_stride < d))) return (int)cudaErrorInvalidValue;
    if (!aligned(x, 4) || !aligned(out, 4) || !aligned(w, 2)) return (int)cudaErrorInvalidValue;
    rmsnorm_kernel<<<rows, NORM_THREADS, 0, gb_stream(stream)>>>((const float*)x, (const uint16_t*)w, (float*)out, d, x_stride,
                                                                  out_stride, eps);
    return GB_CHECK_LAUNCH();
}

// y f32 [rows, 18560] -> z f32 [rows, 8192] (bf16 grid), xbc f32 [rows, 10240], dt f32 [rows, 128]; all 16-byte aligned.
GB_EXPORT int gb_nemo_glue_split_inproj(const void* y, int rows, void* z, void* xbc, void* dt, void* stream) {
    if (rows < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    if (!aligned(y, 16) || !aligned(z, 16) || !aligned(xbc, 16) || !aligned(dt, 16)) return (int)cudaErrorInvalidValue;
    const size_t total = (size_t)rows * INPROJ_V4;
    const size_t blocks = (total + 255) / 256;
    if (blocks > 0x7fffffffull) return (int)cudaErrorInvalidValue;
    split_inproj_kernel<<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>((const float4*)y, (float4*)z, (float4*)xbc, (float4*)dt,
                                                                          total);
    return GB_CHECK_LAUNCH();
}

static int launch_widen(bool f16, const void* src, long long n, void* dst, void* stream) {
    if (n < 0) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;
    if (!aligned(src, 16) || !aligned(dst, 16)) return (int)cudaErrorInvalidValue;
    long long groups = n / 8;
    long long blocks = (groups + 255) / 256;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    if (f16) {
        widen_kernel<true><<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>((const uint16_t*)src, n, (float*)dst);
    } else {
        widen_kernel<false><<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>((const uint16_t*)src, n, (float*)dst);
    }
    return GB_CHECK_LAUNCH();
}

// src bf16 [n] -> dst f32 [n]; both 16-byte aligned.
GB_EXPORT int gb_nemo_glue_widen_bf16(const void* src, long long n, void* dst, void* stream) {
    return launch_widen(false, src, n, dst, stream);
}

// src fp16 [n] -> dst f32 [n]; both 16-byte aligned.
GB_EXPORT int gb_nemo_glue_widen_f16(const void* src, long long n, void* dst, void* stream) {
    return launch_widen(true, src, n, dst, stream);
}

// logits f32 [rows, vocab]; 1 <= count <= min(64, vocab); `blocks` stage-1 blocks per row (the Rust side's plan); ws holds
// rows * blocks * 8 * count u64 (8-byte aligned); ids i32 and vals f32 [rows, count].
GB_EXPORT int gb_nemo_glue_topk(const void* logits, int rows, int vocab, int count, int blocks, void* ws, size_t ws_bytes, void* ids,
                                void* vals, void* stream) {
    if (rows < 0 || vocab < 0 || count < 0 || blocks < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    if (count < 1 || count > 64 || count > vocab || blocks < 1 || rows > 65535) return (int)cudaErrorInvalidValue;
    const size_t need = (size_t)rows * (size_t)blocks * TK_WARPS * (size_t)count * sizeof(u64);
    if (ws_bytes < need || !aligned(ws, 8) || !aligned(logits, 4) || !aligned(ids, 4) || !aligned(vals, 4))
        return (int)cudaErrorInvalidValue;
    topk_stage1_kernel<<<dim3((unsigned)blocks, (unsigned)rows), TK_THREADS, 0, gb_stream(stream)>>>((const float*)logits, vocab, count,
                                                                                                     blocks, (u64*)ws);
    int rc = GB_CHECK_LAUNCH();
    if (rc != 0) return rc;
    topk_stage2_kernel<<<rows, TK_THREADS, 0, gb_stream(stream)>>>((const float*)logits, vocab, count, blocks * TK_WARPS,
                                                                    (const u64*)ws, (int*)ids, (float*)vals);
    return GB_CHECK_LAUNCH();
}

// feats bf16 rows [*, 4096] from src_row, dst f32 rows [*, 4096] from dst_row, n rows; both 16-byte aligned.
GB_EXPORT int gb_nemo_glue_splice_rows(const void* feats, long long src_row, void* dst, long long dst_row, long long n, void* stream) {
    if (src_row < 0 || dst_row < 0 || n < 0 || n > 0x7fffffffll) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;
    if (!aligned(feats, 16) || !aligned(dst, 16)) return (int)cudaErrorInvalidValue;
    const uint4* src = (const uint4*)feats + (size_t)src_row * (HIDDEN / 8);
    float4* d = (float4*)dst + (size_t)dst_row * (HIDDEN / 4);
    splice_rows_kernel<<<(unsigned)n, 256, 0, gb_stream(stream)>>>(src, d);
    return GB_CHECK_LAUNCH();
}
