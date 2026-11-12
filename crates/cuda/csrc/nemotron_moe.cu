// gb-nvcc: --fmad=false
// Latent-MoE block of the Nemotron 3.5 Super VL engine (the E layers): an fp32 router with top-22 selection, the routed experts
// (EXL3 4-bit trellis or bf16 HF weights, chosen per expert), the latent projections fc1 and fc2, and the shared expert. The oracle is
// crates/model/src/families/nemotron_h_omni/moe_ref.rs; EXL3 weights are decoded exactly as gb_exl3::unpack does.
//
// Arithmetic. Every reduction has a fixed order that depends on the shape only. A projection's k sum is a chain of fmaf in increasing k
// inside S slices of the EXL3 k tiles; the S slice sums are added in slice order. A bf16 projection's chain is 8-element chunks in
// increasing order, one lane per chunk residue, then an xor tree. Nothing adds in an order that depends on scheduling (the atomics are
// tickets and bit masks), so a row's bits never depend on how many rows share a call or on which block or chunk computed it (row
// invariance). Pairs are (row, slot) with index row * 22 + slot.
//
// Two sets of kernels give those bits. The reference kernels (nm_*) are the arithmetic written out, one launch per step, and run any
// projection. The fast kernels (nm2_*) run 4-bit Mul1 EXL3 projections and the router with the same operations in the same order,
// scheduled for bandwidth: crates/cuda/tests/nemotron_moe_perf.rs gates them bit for bit against the reference ones, and
// docs/NEMOTRON_MOE_PERF.md has the numbers.
//
// Reference kernels, in launch order (crates/cuda/src/nemotron_moe.rs holds the sequence):
//   nm_router_logits<RB, G>  fp32 logits [rows, 512] = h . gate^T, gate bf16 [512, 4096] (G = uint16_t) or f32 (G = float: an F16
//                         gate widened exactly, decision D1 of v0.1.2; the same chains): warp per (expert, row chunk)
//   nm_router_select      warp per row: sigmoid, + bias, top-22 with ties to the lower id, weights = (sig / sum) * scale
//   nm_bucket             block of 512: per expert the ascending pair ids that selected it
//   nm_t1                 per input row: x * suh (EXL3) then H_128 per 128-block; bf16 experts pass x through
//   nm_gemm_exl3          per (column block, K slice, group): fp32 partial sums over the group's rows
//   nm_gemv_bf16          per (column, group): a bf16 expert's full dot products, written to partial slot 0
//   nm_t2                 per output row: slice sum, then H_128, * svh, + bias, relu2 (optional), + add (optional)
//   nm_combine            per token: routed = sum over its 22 slots of w * y, in slot order
//   nm_decode             debug: the fp16 bits of W_q [K, N] of one EXL3 projection (must equal gb_exl3::unpack)
//
// Fast kernels:
//   nm2_router_logits<RB, EW, G>  the logits of nm_router_logits, a warp carrying EW experts for RB rows, the gate chunks in flight
//   nm2_select                 selection, weights and buckets (and the list of experts that have rows) in one launch
//   nm2_gemm4<DENSE>           nm_t1 + nm_gemm_exl3 + nm_t2 for 4-bit Mul1 projections: rotation in the prologue, the output side in
//                              the last slice block (ticket); routed groups over the active list, dense ones over row chunks
// Launches of the fast path carry the programmatic-dependent-launch attribute (pdl_wait / pdl_trigger).

#include <stdint.h>

#include "common.cuh"

namespace {

constexpr int kFmtExl3 = 0;    // trellis + su/sv scales (+ optional bias)
constexpr int kFmtBf16 = 1;    // bf16 weight [N, K] (HF layout)
constexpr int kFmtF16 = 3;     // fp16 weight [N, K] (HF layout; a MixedK pack's latent projections): the bf16 chain on fp16 values
// Any other fmt (2 = absent: an expert the caller did not load) gives zero outputs in nm_t1 and nm_t2 and writes no partials.
constexpr int kTopK = 22;
constexpr int kExperts = 512;
constexpr int kGemmRb = 16;    // rows a block carries through one EXL3 projection pass
constexpr float kInvSqrt128 = 0.08838834764831845f;
constexpr int kCbMul1 = 2;     // codebook id (gb_exl3::Codebook order: 3inst 0, mcg 1, mul1 2)
constexpr int kFastTwice = 8;  // the 4-bit width the fast kernels decode

// One projection, as the kernels read it (56 bytes; the Rust side mirrors it with #[repr(C)]).
struct NmProj {
    const void* w;       // EXL3: int16 trellis [K/16][N/16][8 * twice]; bf16: weight [N, K]
    const float* suh;    // EXL3: input scale [K]
    const float* svh;    // EXL3: output scale [N]
    const float* bias;   // EXL3: output bias [N], or null
    const uint16_t* cb;  // EXL3: fp16 bits of the 65536-entry codebook (the reference kernels read it; the fast ones compute it)
    int fmt;             // bits 0..3 the format (kFmt*, 2 = absent), bits 4..5 the codebook id of an EXL3 projection
    int twice;           // EXL3: bits * 2 (8 for 4-bit)
    int k, n;
};

// Programmatic dependent launch (the fast path's launches carry the attribute): a kernel may begin while its predecessor in the
// stream drains, and does only work that needs nothing the predecessor writes (weight prefetch) before pdl_wait(), which returns once
// the predecessor has completed and its writes are visible. pdl_trigger() lets the successor launch (it does so when every block of the
// grid has triggered or exited). Both are no-ops for a kernel that was launched without the attribute.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
// Invalidate the 128-byte line at p in L2 without writing it back (p 128-byte aligned): for scratch that is dead.
__device__ __forceinline__ void discard_l2(const void* p) { asm volatile("discard.global.L2 [%0], 128;" ::"l"(p) : "memory"); }

__device__ __forceinline__ int fmt_of(const NmProj& p) { return p.fmt & 15; }
// The projections the fast kernels (4-bit, Mul1) run; every other EXL3 projection takes the reference kernels.
__device__ __forceinline__ bool is_fast(const NmProj& p) {
    return (p.fmt & 15) == kFmtExl3 && p.twice == kFastTwice && (p.fmt >> 4) == kCbMul1;
}
// The projections the any-width fast kernel (nm2_gemmk: Mul1 at an integer width of 2 to 8 bits, 4 included) runs.
__device__ __forceinline__ bool is_fastk(const NmProj& p) {
    return (p.fmt & 15) == kFmtExl3 && (p.fmt >> 4) == kCbMul1 && (p.twice & 1) == 0 && p.twice >= 4 && p.twice <= 16;
}
// The reference kernels' skip levels: 0 runs every entry; 1 skips the entries nm2_gemm4 ran (4-bit Mul1, the original fast path); 2
// skips the entries nm2_gemmk ran (a table with another width among its fast entries).
__device__ __forceinline__ bool skipped(const NmProj& p, int skip) { return skip == 1 ? is_fast(p) : (skip == 2 && is_fastk(p)); }
// A dense weight [N, K] read by the bf16 chain: bf16, or fp16 (its values widened exactly in the chunk).
__device__ __forceinline__ bool is_plain(int fmt) { return fmt == kFmtBf16 || fmt == kFmtF16; }

// The 16-bit state of stream position p in one tile: gb_exl3::states for one value. `tile` holds the tile's 4 * twice u32 words,
// word i = int16 pair (2i, 2i + 1) little endian, and the stream is read MSB first across the words, wrapping at the tile end.
__device__ __forceinline__ uint32_t tile_state(const uint32_t* tile, int twice, int p) {
    const uint32_t p1 = (uint32_t)p + 1u;
    // E(p): where the value's window ends (Bits::stream_ends). Half-bit widths (odd twice) alternate.
    const uint32_t end = (twice & 1) ? (p1 * (uint32_t)twice - (p1 & 1u)) / 2u : p1 * (uint32_t)(twice / 2);
    const uint32_t nw = 4u * (uint32_t)twice;
    const uint32_t first = end + 32u * nw - 16u;
    const uint32_t i0 = (first / 32u) % nw;
    const uint32_t off = first % 32u;
    const uint32_t i1 = (i0 + 1u) % nw;
    const uint64_t pair = ((uint64_t)tile[i0] << 32) | (uint64_t)tile[i1];
    return (uint32_t)((pair >> (48u - off)) & 0xFFFFu);
}

// The stream position of element (kr, c) of a 16 x 16 tile: the inverse of gb_exl3::tile_positions.
__device__ __forceinline__ int tile_pos(int kr, int c) {
    const int lane = 4 * (c & 7) + ((kr & 7) >> 1);
    const int j = (kr & 1) + 2 * (kr >> 3) + 4 * (c >> 3);
    return 8 * lane + j;
}

// In-place unnormalised Walsh-Hadamard transform of 128 values, one per thread of a 128-thread block (`buf` in shared memory).
__device__ __forceinline__ void fwht128(float* buf) {
    const int t = threadIdx.x;
#pragma unroll
    for (int h = 1; h < 128; h <<= 1) {
        __syncthreads();
        if (t < 64) {
            const int i = ((t & ~(h - 1)) << 1) | (t & (h - 1));
            const float a = buf[i], b = buf[i + h];
            buf[i] = a + b;
            buf[i + h] = a - b;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// acc += x[0..8] . w[0..8] as eight fused multiply-adds in element order (the bf16 chunk recipe).
__device__ __forceinline__ float chunk_fma(float acc, const float4 a, const float4 b, const uint4 w) {
    acc = __fmaf_rn(a.x, bf_lo(w.x), acc);
    acc = __fmaf_rn(a.y, bf_hi(w.x), acc);
    acc = __fmaf_rn(a.z, bf_lo(w.y), acc);
    acc = __fmaf_rn(a.w, bf_hi(w.y), acc);
    acc = __fmaf_rn(b.x, bf_lo(w.z), acc);
    acc = __fmaf_rn(b.y, bf_hi(w.z), acc);
    acc = __fmaf_rn(b.z, bf_lo(w.w), acc);
    acc = __fmaf_rn(b.w, bf_hi(w.w), acc);
    return acc;
}

// The same eight fused multiply-adds in element order on fp16 weights (each widened exactly): the chain of a F16 projection.
__device__ __forceinline__ float chunk_fma_h(float acc, const float4 a, const float4 b, const uint4 w) {
    const float2 w0 = __half22float2(*reinterpret_cast<const half2*>(&w.x)), w1 = __half22float2(*reinterpret_cast<const half2*>(&w.y));
    const float2 w2 = __half22float2(*reinterpret_cast<const half2*>(&w.z)), w3 = __half22float2(*reinterpret_cast<const half2*>(&w.w));
    acc = __fmaf_rn(a.x, w0.x, acc);
    acc = __fmaf_rn(a.y, w0.y, acc);
    acc = __fmaf_rn(a.z, w1.x, acc);
    acc = __fmaf_rn(a.w, w1.y, acc);
    acc = __fmaf_rn(b.x, w2.x, acc);
    acc = __fmaf_rn(b.y, w2.y, acc);
    acc = __fmaf_rn(b.z, w3.x, acc);
    acc = __fmaf_rn(b.w, w3.y, acc);
    return acc;
}

// Order-preserving map of a float key to uint32 (larger float, larger code); NaN gets the smallest code.
__device__ __forceinline__ uint32_t key_code(float k) {
    if (k != k) return 0u;
    const uint32_t u = __float_as_uint(k);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- the router gate's element type -------------------------------------------------------------------------------------------------
// A gate is bf16 bits (uint16_t: the published pack) or f32 (float: an F16 gate widened exactly on the device, never rounded to bf16).
// The router kernels are templated on it; a chunk is the gate row's 8 values q * 8 .. + 7 either way, and chunk_fma takes them in the
// same element order, so an f32 gate holding bf16 values gives the bf16 kernel's bits exactly.

// Eight f32 gate values (one chunk).
struct G32 {
    float4 lo, hi;
};

// acc += x[0..8] . w[0..8] as eight fused multiply-adds in element order: the bf16 recipe above on f32 gate values.
__device__ __forceinline__ float chunk_fma(float acc, const float4 a, const float4 b, const G32 w) {
    acc = __fmaf_rn(a.x, w.lo.x, acc);
    acc = __fmaf_rn(a.y, w.lo.y, acc);
    acc = __fmaf_rn(a.z, w.lo.z, acc);
    acc = __fmaf_rn(a.w, w.lo.w, acc);
    acc = __fmaf_rn(b.x, w.hi.x, acc);
    acc = __fmaf_rn(b.y, w.hi.y, acc);
    acc = __fmaf_rn(b.z, w.hi.z, acc);
    acc = __fmaf_rn(b.w, w.hi.w, acc);
    return acc;
}

template <typename G>
struct GateChunk;
template <>
struct GateChunk<uint16_t> {
    using V = uint4;
    // the chunk that starts at element p of the gate (p a multiple of 8)
    static __device__ __forceinline__ V load(const uint16_t* g) { return __ldg(reinterpret_cast<const uint4*>(g)); }
};
template <>
struct GateChunk<float> {
    using V = G32;
    static __device__ __forceinline__ V load(const float* g) {
        return G32{__ldg(reinterpret_cast<const float4*>(g)), __ldg(reinterpret_cast<const float4*>(g + 4))};
    }
};

// An F16 gate as stored (v0.1.2: the device keeps the pack's fp16 bits): a chunk is its 8 fp16 values (16 bytes, the bf16 chunk's
// registers), widened exactly in the chunk's multiply-adds, which are the f32 gate's, in the same order: the bits of the f32 kernels
// on the widened gate.
struct H8 {
    uint4 w;
};
__device__ __forceinline__ float chunk_fma(float acc, const float4 a, const float4 b, const H8 w) { return chunk_fma_h(acc, a, b, w.w); }
template <>
struct GateChunk<__half> {
    using V = H8;
    static __device__ __forceinline__ V load(const __half* g) { return H8{__ldg(reinterpret_cast<const uint4*>(g))}; }
};

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// Router

// Block (e, rc): warp w computes expert e = blockIdx.x * 8 + w for rows rc * RB .. + RB. Lane l owns the 8-element chunks l + 32 i
// of the gate row; each row's chain is the lane's chunks in increasing i, then the xor tree 16 -> 1. G: the gate's element type.
template <int RB, typename G>
__global__ void __launch_bounds__(256) nm_router_logits(const float* __restrict__ h, const G* __restrict__ gate,
                                                        float* __restrict__ logits, int rows, int K, int E) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * 8 + warp;
    if (e >= E) return;  // a whole warp returns
    const int r0 = blockIdx.y * RB;
    const G* grow = gate + (size_t)e * K;
    float acc[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc[r] = 0.f;
    for (int q = lane; q < K / 8; q += 32) {
        const typename GateChunk<G>::V wv = GateChunk<G>::load(grow + 8 * q);
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            if (r0 + r < rows) {
                const float* xr = h + (size_t)(r0 + r) * K + 8 * q;
                acc[r] = chunk_fma(acc[r], __ldg(reinterpret_cast<const float4*>(xr)), __ldg(reinterpret_cast<const float4*>(xr + 4)), wv);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
        if (lane == 0 && r0 + r < rows) logits[(size_t)(r0 + r) * E + e] = acc[r];
    }
}

// One warp per row. Candidate j of lane l is expert l + 32 j; comp packs the key code (high word) and ~id (low word), so the largest comp
// is the largest key and, among equal keys, the lowest id. Rank k's expert is the k-th largest comp (a taken candidate is zeroed).
__global__ void __launch_bounds__(32) nm_router_select(const float* __restrict__ logits, const float* __restrict__ bias, float scale,
                                                       int* __restrict__ pick, float* __restrict__ wts) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const float* lg = logits + (size_t)row * kExperts;
    uint64_t comp[kExperts / 32];
#pragma unroll
    for (int j = 0; j < kExperts / 32; ++j) {
        const int id = lane + 32 * j;
        const float key = sigmoid_f(lg[id]) + bias[id];
        comp[j] = ((uint64_t)key_code(key) << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)id);
    }
    int my_id = 0;
    float my_sig = 0.f;
    for (int k = 0; k < kTopK; ++k) {
        uint64_t best = 0;
#pragma unroll
        for (int j = 0; j < kExperts / 32; ++j) best = comp[j] > best ? comp[j] : best;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const uint64_t o = ((uint64_t)__shfl_xor_sync(0xffffffffu, (uint32_t)(best >> 32), m) << 32) |
                               (uint64_t)__shfl_xor_sync(0xffffffffu, (uint32_t)best, m);
            best = o > best ? o : best;
        }
        const int id = (int)(0xFFFFFFFFu - (uint32_t)best);
#pragma unroll
        for (int j = 0; j < kExperts / 32; ++j)
            if (comp[j] == best) comp[j] = 0;
        if (lane == k) {
            my_id = id;
            my_sig = sigmoid_f(lg[id]);  // the same expression as the key: the same bits
        }
    }
    // The oracle's denominator: the selected sigmoids summed in rank order (sequential), plus 1e-20.
    float denom = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) denom = denom + __shfl_sync(0xffffffffu, my_sig, k);
    denom = denom + 1e-20f;
    if (lane < kTopK) {
        pick[(size_t)row * kTopK + lane] = my_id;
        wts[(size_t)row * kTopK + lane] = (my_sig / denom) * scale;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Expert buckets

// One thread per expert: the ascending pair ids (row * 22 + slot) whose pick is that expert. A token picks an expert at most once, so
// an expert's list holds at most `cap` (= rows) entries.
__global__ void __launch_bounds__(512) nm_bucket(const int* __restrict__ pick, int P, int cap, int* __restrict__ cnt,
                                                 int* __restrict__ lists) {
    const int e = threadIdx.x;
    int* out = lists + (size_t)e * cap;
    int n = 0;
    for (int p = 0; p < P; ++p)
        if (pick[p] == e) out[n++] = p;
    cnt[e] = n;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Projections

// dst[r, :] = the input side of projection tab[expert_of_row[r]] applied to src[r / div, :]: EXL3 gives H_128 (x * suh) per 128-block;
// bf16 gives x; absent gives zeros. expert_of_row null means entry 0 for every row.
__global__ void __launch_bounds__(128) nm_t1(const float* __restrict__ src, int div, const int* __restrict__ expert_of_row,
                                             const NmProj* __restrict__ tab, int rows, int K, float* __restrict__ dst, int skip_fast) {
    __shared__ float buf[128];
    const int r = blockIdx.x;
    if (r >= rows) return;  // uniform per block
    const NmProj p = tab[expert_of_row ? expert_of_row[r] : 0];
    if (skipped(p, skip_fast)) return;  // rotated in nm2_gemm4's (nm2_gemmk's) prologue; uniform per block
    const float* x = src + (size_t)(r / div) * K;
    float* y = dst + (size_t)r * K;
    const int t = threadIdx.x;
    if (fmt_of(p) != kFmtExl3) {
        for (int k = t; k < K; k += 128) y[k] = is_plain(fmt_of(p)) ? x[k] : 0.f;
        return;
    }
    for (int b = 0; b < K; b += 128) {
        buf[t] = x[b + t] * p.suh[b + t];
        fwht128(buf);
        y[b + t] = buf[t] * kInvSqrt128;
    }
}

// Rows of group g: lists[g * lcap + i] (or the identity when lists is null, with fixed_cnt rows) and cnt[g] (or fixed_cnt). The group
// of block z is z, or active[z] (blocks past *nact exit) when the compacted list of the groups with rows is given.
// Block (nb, s, g): output columns nb * 128 .. + 128 (thread j is column nb * 128 + j), k tiles of slice s. A block carries kGemmRb rows
// at a time; thread j's chain for each row is the k tiles in order and the 16 k rows of each tile in order.
// The tile of column block nb in k tile kt is 8 trellis tiles, staged in shared memory; thread j decodes its column's 16 elements of
// every tile (positions tile_pos(kr, c)).
__global__ void __launch_bounds__(128) nm_gemm_exl3(const float* __restrict__ xin, const int* __restrict__ lists,
                                                    const int* __restrict__ cnt, int fixed_cnt, int lcap,
                                                    const NmProj* __restrict__ tab, int K, int N, int S, float* __restrict__ partial,
                                                    int cap, int skip_fast, const int* __restrict__ active,
                                                    const int* __restrict__ nact) {
    if (active && (int)blockIdx.z >= *nact) return;  // uniform per block
    const int g = active ? active[blockIdx.z] : (int)blockIdx.z;
    const NmProj p = tab[g];
    if (fmt_of(p) != kFmtExl3 || skipped(p, skip_fast)) return;  // uniform per block
    const int count = cnt ? cnt[g] : fixed_cnt;
    if (count == 0) return;
    const int nb = blockIdx.x, s = blockIdx.y, j = threadIdx.x;
    const int kts = (K / 16) / S;
    const int kt0 = s * kts;
    const int nt_total = N / 16;
    const int nw = 4 * p.twice;
    const int nt_loc = j >> 4, c = j & 15;
    const uint32_t* trel = static_cast<const uint32_t*>(p.w);
    __shared__ uint32_t tw[8 * 64];
    __shared__ float xs[kGemmRb][16];
    __shared__ int rid[kGemmRb];
    for (int c0 = 0; c0 < count; c0 += kGemmRb) {
        __syncthreads();  // rid and xs of the previous chunk are no longer read
        const int here = min(kGemmRb, count - c0);
        if (j < kGemmRb) rid[j] = j < here ? (lists ? lists[(size_t)g * lcap + c0 + j] : c0 + j) : -1;
        float acc[kGemmRb];
#pragma unroll
        for (int r = 0; r < kGemmRb; ++r) acc[r] = 0.f;
        for (int kt = kt0; kt < kt0 + kts; ++kt) {
            __syncthreads();  // rid is visible; the previous tile's tw and xs are no longer read
            const uint32_t* src = trel + ((size_t)kt * nt_total + (size_t)nb * 8) * nw;
            for (int i = j; i < 8 * nw; i += 128) tw[i] = src[i];
            for (int i = j; i < kGemmRb * 16; i += 128) {
                const int r = i >> 4, kr = i & 15;
                xs[r][kr] = r < here ? xin[(size_t)rid[r] * K + kt * 16 + kr] : 0.f;
            }
            __syncthreads();
#pragma unroll 4
            for (int kr = 0; kr < 16; ++kr) {
                const uint32_t st = tile_state(tw + nt_loc * nw, p.twice, tile_pos(kr, c));
                const float w = __half2float(__ushort_as_half(p.cb[st]));
#pragma unroll
                for (int r = 0; r < kGemmRb; ++r)
                    if (r < here) acc[r] = fmaf(xs[r][kr], w, acc[r]);
            }
        }
        // A constant bound (not `here`) keeps acc[] in registers.
#pragma unroll
        for (int r = 0; r < kGemmRb; ++r)
            if (r < here) partial[((size_t)s * cap + rid[r]) * N + (size_t)nb * 128 + j] = acc[r];
    }
}

// The bf16 expert's dot products. Block of 8 warps, warp w is column blockIdx.x * 8 + w; lane l owns the 8-element chunks l + 32 q of the
// weight row, accumulated in increasing q, then the xor tree. Writes partial slot 0 (row idx of the group, cap rows per slot).
__global__ void __launch_bounds__(256) nm_gemv_bf16(const float* __restrict__ xin, const int* __restrict__ lists,
                                                    const int* __restrict__ cnt, int fixed_cnt, int lcap,
                                                    const NmProj* __restrict__ tab, int K, int N, float* __restrict__ partial,
                                                    int cap, const int* __restrict__ active, const int* __restrict__ nact) {
    if (active && (int)blockIdx.y >= *nact) return;  // uniform per block
    const int g = active ? active[blockIdx.y] : (int)blockIdx.y;
    const NmProj p = tab[g];
    if (!is_plain(fmt_of(p))) return;  // uniform per block
    const bool h16 = fmt_of(p) == kFmtF16;
    const int count = cnt ? cnt[g] : fixed_cnt;
    if (count == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;  // a whole warp returns
    const uint16_t* wrow = static_cast<const uint16_t*>(p.w) + (size_t)n * K;
    const int chunks = K / 8;
    for (int r = 0; r < count; ++r) {
        const int idx = lists ? lists[(size_t)g * lcap + r] : r;
        const float* xr = xin + (size_t)idx * K;
        float acc = 0.f;
        for (int q = lane; q < chunks; q += 32) {
            const uint4 wv = __ldg(reinterpret_cast<const uint4*>(wrow + 8 * q));
            const float4 xa = __ldg(reinterpret_cast<const float4*>(xr + 8 * q)), xb = __ldg(reinterpret_cast<const float4*>(xr + 8 * q + 4));
            acc = h16 ? chunk_fma_h(acc, xa, xb, wv) : chunk_fma(acc, xa, xb, wv);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) partial[(size_t)idx * N + n] = acc;
    }
}

// out[r, n] for output row r with the projection tab[expert_of_row[r]] (entry 0 when null): the sum of the partial slices in slice order
// (EXL3) or slot 0 (bf16), then for EXL3 H_128 per 128-block, * svh, + bias. Then relu2 when `relu2`, and + add[r, n] when add is given.
__global__ void __launch_bounds__(128) nm_t2(const float* __restrict__ partial, int S, int cap, const int* __restrict__ expert_of_row,
                                             const NmProj* __restrict__ tab, int rows, int N, int relu2, const float* __restrict__ add,
                                             float* __restrict__ out, int skip_fast) {
    __shared__ float buf[128];
    const int r = blockIdx.y;
    if (r >= rows) return;  // uniform per block
    const int t = threadIdx.x, n = blockIdx.x * 128 + t;
    const NmProj p = tab[expert_of_row ? expert_of_row[r] : 0];
    if (skipped(p, skip_fast)) return;  // its output side is done (nm2_gemm4's / nm2_gemmk's last block); uniform per block
    float v = 0.f;
    if (is_plain(fmt_of(p))) {
        v = partial[(size_t)r * N + n];
    } else if (fmt_of(p) == kFmtExl3) {
        v = partial[(size_t)r * N + n];
        for (int s = 1; s < S; ++s) v = v + partial[((size_t)s * cap + r) * N + n];
        buf[t] = v;
        fwht128(buf);
        v = (buf[t] * kInvSqrt128) * p.svh[n];
        if (p.bias) v = v + p.bias[n];
    }
    if (relu2) {
        const float q = fmaxf(v, 0.f);
        v = q * q;
    }
    if (add) v = v + add[(size_t)r * N + n];
    out[(size_t)r * N + n] = v;
}

// routed[t, n] = sum over the 22 slots of wts[t, s] * y[t * 22 + s, n], in slot order.
__global__ void __launch_bounds__(128) nm_combine(const float* __restrict__ y, const float* __restrict__ wts, int N,
                                                  float* __restrict__ routed) {
    pdl_wait();
    pdl_trigger();
    const int t = blockIdx.y, n = blockIdx.x * 128 + threadIdx.x;
    float acc = 0.f;
    for (int s = 0; s < kTopK; ++s) acc = __fadd_rn(acc, __fmul_rn(wts[(size_t)t * kTopK + s], y[((size_t)t * kTopK + s) * N + n]));
    routed[(size_t)t * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Fast projections: 4-bit EXL3 with the Mul1 codebook (every routed expert, the shared expert and the latent projections of the pack)
//
// The same arithmetic as nm_t1 + nm_gemm_exl3, scheduled for bandwidth (the reference kernels above stay as the oracle). Per output
// element nothing changes: the k chain of slice s is fmaf over the slice's k tiles in order and the 16 k rows of each tile in order,
// starting from 0; the input is H_128(x * suh) * (1 / sqrt(128)) per 128-block (same stage order, same operand roles); the weight is the
// fp16 value of the 16-bit window state under Mul1 (computed, as the EXL3 decode kernel does, instead of read from the 128 KB table).
//
// Block (nb, s, g): 128 output columns of slice s of group g, 64 threads. Thread (nt, a) owns columns a and a + 8 of the nt-th 16-column
// tile. Its 16 + 16 weights of a tile come from five words of the tile's 32-word circular bitstream: words 4a - 1 .. 4a + 3 (one 16-byte
// load, plus the word before it from the neighbouring thread by shuffle), so every window sits at a compile-time bit offset.
// The rotated input of the block's k range (H_128 per 128-block of each row of the pass) is computed in the block's prologue, so the
// separate rotate launch and its [rows, K] intermediate are gone.

namespace {

constexpr int kFastThreads = 64;  // 2 warps: 8 tiles of 16 columns, 8 threads each
constexpr int kRoutedRb = 8;      // rows of a routed group's pass through one slice
constexpr int kDenseRb = 16;      // rows of a dense projection's chunk (one block of its grid in z)

// H_128 (unnormalised) of 128 values, 4 to a lane (values 4 * lane .. + 3). Stage order h = 1, 2, 4, .., 64 with a + b to the lower index
// and a - b (lower minus upper) to the upper one: the bits of fwht128 above.
__device__ __forceinline__ void fwht128_warp(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// The 16-bit window that starts d bits below the top of w[0] in the big-endian stream w[0..4] (d is a constant once the callers unroll).
__device__ __forceinline__ uint32_t win16(const uint32_t (&w)[5], int d) {
    const int wi = d >> 5, off = d & 31;
    return off <= 16 ? ((w[wi] >> (16 - off)) & 0xffffu) : (__funnelshift_l(w[wi + 1], w[wi], off) >> 16);
}

// The fp16 codebook values of two 16-bit states under Mul1 (gbx::decode2<CB_MUL1>: 1024 + the byte sum of s * 0x83DCD12D, fused with the
// fp16 scale and bias), as floats.
__device__ __forceinline__ void mul1_pair(uint32_t s0, uint32_t s1, float& w0, float& w1) {
    const uint32_t h0 = __dp4a(s0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const uint32_t h1 = __dp4a(s1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const uint32_t hh = __byte_perm(h0, h1, 0x5410);  // (h0 & 0xffff) | (h1 << 16)
    const half2 r = __hfma2(*reinterpret_cast<const half2*>(&hh), __half2half2(__ushort_as_half(0x1eee)),
                            __half2half2(__ushort_as_half(0xc931)));
    const float2 f = __half22float2(r);
    w0 = f.x;
    w1 = f.y;
}

// The weights of one half tile (k rows 8 H .. 8 H + 7) of columns a (c0) and a + 8 (c1). A value at tile position p = 8 L + j, with
// L = 4 a + q (q = (kr & 7) >> 1) and j = (kr & 1) + 2 (kr >> 3) + 4 (column >> 3), has the window that starts at bit 32 q + 4 j + 20
// below the top of word 4 a - 1 (E(p) = 4 (p + 1), minus 16, relative to the base word).
template <int H>
__device__ __forceinline__ void decode_half(const uint32_t (&w)[5], float (&c0)[8], float (&c1)[8]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int d = 32 * q + 8 * H + 20;
        mul1_pair(win16(w, d), win16(w, d + 4), c0[2 * q], c0[2 * q + 1]);
        mul1_pair(win16(w, d + 16), win16(w, d + 20), c1[2 * q], c1[2 * q + 1]);
    }
}

// acc += x . w over eight k rows for RB rows, each row's chain in increasing k (xt: the half tile's x of row 0; rows are 128 apart).
template <int RB>
__device__ __forceinline__ void fma_half(float (&a0)[RB], float (&a1)[RB], const float* xt, const float (&c0)[8], const float (&c1)[8]) {
#pragma unroll
    for (int r = 0; r < RB; ++r) {
        const float4 xa = *reinterpret_cast<const float4*>(xt + r * 128);
        const float4 xb = *reinterpret_cast<const float4*>(xt + r * 128 + 4);
        const float x[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            a0[r] = fmaf(x[k], c0[k], a0[r]);
            a1[r] = fmaf(x[k], c1[k], a1[r]);
        }
    }
}

// One slice of one column block for the pass's `here` rows (RB >= here, padding rows are computed and dropped).
template <int RB>
__device__ __forceinline__ void fast_chunk(const float* __restrict__ src, int ksrc, const NmProj& p, int K, int N, int S, int nb, int s,
                                           int here, float* __restrict__ partial, int cap, float (*xs)[128], const int* rid_s,
                                           const int* srow_s) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, a = tid & 7, nt = tid >> 3;
    const int kts = (K / 16) / S, kt0 = s * kts, kt_end = kt0 + kts;
    const size_t kstride = (size_t)(N / 16) * 32;  // words between the k tiles of one column
    const uint32_t* base = static_cast<const uint32_t*>(p.w) + ((size_t)nb * 8 + nt) * 32 + 4 * a;
    float acc0[RB], acc1[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc0[r] = acc1[r] = 0.f;

    uint4 cur = __ldg(reinterpret_cast<const uint4*>(base + (size_t)kt0 * kstride));
    uint4 n1 = kts > 1 ? __ldg(reinterpret_cast<const uint4*>(base + (size_t)(kt0 + 1) * kstride)) : cur;
    for (int kt = kt0; kt < kt_end; ++kt) {
        const uint4 n2 = (kt + 2 < kt_end) ? __ldg(reinterpret_cast<const uint4*>(base + (size_t)(kt + 2) * kstride)) : cur;
        if (kt == kt0 || (kt & 7) == 0) {
            // the rotated input of the 128-block kt >> 3 for every row of the pass
            __syncthreads();  // the previous block's xs are no longer read; rid_s and srow_s are visible
            const int hb = kt >> 3;
            const float4 sv = __ldg(reinterpret_cast<const float4*>(p.suh + hb * 128 + 4 * lane));
            for (int r = warp; r < here; r += kFastThreads / 32) {
                const float4 xv = __ldg(reinterpret_cast<const float4*>(src + (size_t)srow_s[r] * ksrc + hb * 128 + 4 * lane));
                float v[4] = {xv.x * sv.x, xv.y * sv.y, xv.z * sv.z, xv.w * sv.w};
                fwht128_warp(v, lane);
                *reinterpret_cast<float4*>(&xs[r][4 * lane]) =
                    make_float4(v[0] * kInvSqrt128, v[1] * kInvSqrt128, v[2] * kInvSqrt128, v[3] * kInvSqrt128);
            }
            __syncthreads();
        }
        // words 4a - 1 .. 4a + 3 of the tile: the one before this thread's 16 bytes is its neighbour's last word (thread a = 0 takes
        // word 31 from thread a = 7, the stream being circular)
        const uint32_t prev = __shfl_sync(0xffffffffu, cur.w, (lane & ~7) | ((a + 7) & 7));
        const uint32_t w[5] = {prev, cur.x, cur.y, cur.z, cur.w};
        const float* xt = &xs[0][0] + (kt & 7) * 16;
        {
            float c0[8], c1[8];
            decode_half<0>(w, c0, c1);
            fma_half<RB>(acc0, acc1, xt, c0, c1);
        }
        {
            float c0[8], c1[8];
            decode_half<1>(w, c0, c1);
            fma_half<RB>(acc0, acc1, xt + 8, c0, c1);
        }
        cur = n1;
        n1 = n2;
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) {
        if (r < here) {
            float* dst = partial + ((size_t)s * cap + rid_s[r]) * N + (size_t)nb * 128 + nt * 16 + a;
            dst[0] = acc0[r];
            dst[8] = acc1[r];
        }
    }
}

// The output side of nm_t2 for the `here` rows of a pass and the 128 columns of block nb, as the last of the S slice blocks runs it:
// the S partial slices added in slice order, H_128, * svh, + bias, then relu2 and + add as asked. A warp takes a row at a time, a lane
// four columns. The partials were written by other blocks: read through L2.
__device__ __forceinline__ void fast_epilogue(const NmProj& p, int N, int S, int nb, int here, const float* partial, int cap,
                                              const int* rid_s, int relu2, const float* __restrict__ add, float* __restrict__ out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = nb * 128 + 4 * lane;
    const float4 sv = __ldg(reinterpret_cast<const float4*>(p.svh + n));
    const bool has_bias = p.bias != nullptr;
    const float4 bv = has_bias ? __ldg(reinterpret_cast<const float4*>(p.bias + n)) : make_float4(0.f, 0.f, 0.f, 0.f);
    for (int r = warp; r < here; r += kFastThreads / 32) {
        const int id = rid_s[r];
        const float* pr = partial + (size_t)id * N + n;
        float4 v = __ldcg(reinterpret_cast<const float4*>(pr));
        for (int s2 = 1; s2 < S; ++s2) {
            const float4 q = __ldcg(reinterpret_cast<const float4*>(pr + (size_t)s2 * cap * N));
            v.x = v.x + q.x;
            v.y = v.y + q.y;
            v.z = v.z + q.z;
            v.w = v.w + q.w;
        }
        // The slices are read and nothing reads them again: drop their lines from L2 so that they are not written back to DRAM (at 64
        // rows the partials are 60 MB a projection, a tenth of the bytes of the weights). Their contents are undefined from here on.
        __syncwarp();
        if ((lane & 7) == 0) {
            for (int s2 = 0; s2 < S; ++s2) discard_l2(pr + (size_t)s2 * cap * N);
        }
        float t[4] = {v.x, v.y, v.z, v.w};
        fwht128_warp(t, lane);
        const float sc[4] = {sv.x, sv.y, sv.z, sv.w}, bc[4] = {bv.x, bv.y, bv.z, bv.w};
        float4 ad = make_float4(0.f, 0.f, 0.f, 0.f);
        if (add) ad = *reinterpret_cast<const float4*>(add + (size_t)id * N + n);
        const float av[4] = {ad.x, ad.y, ad.z, ad.w};
        float o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float x = (t[j] * kInvSqrt128) * sc[j];
            if (has_bias) x = x + bc[j];
            if (relu2) {
                const float q = fmaxf(x, 0.f);
                x = q * q;
            }
            if (add) x = x + av[j];
            o[j] = x;
        }
        *reinterpret_cast<float4*>(out + (size_t)id * N + n) = make_float4(o[0], o[1], o[2], o[3]);
    }
}

// Block (nb, s, z): 128 columns of slice s. With `lists`, z is the group (or the z-th active group, active[z], when `active` is given;
// blocks past *nact exit) and its rows are lists[g * lcap + i], i < cnt[g], taken kRoutedRb at a time. Without, the table entry is 0 and z is a chunk of kDenseRb
// rows of 0 .. fixed_cnt - 1: the blocks of one dense projection spread over the rows (the arithmetic of a row does not depend on which
// block carries it). Row r of the pass reads src row (id / div) (K floats, stride ksrc) and is rotated here;
// partial[(s * cap + id) * N + n] as the reference kernel writes it.
// DENSE: the instantiation for a dense projection (no lists): chunks of kDenseRb rows, which takes more registers than the routed one.
template <bool DENSE>
__global__ void __launch_bounds__(kFastThreads, DENSE ? 6 : 8)
    nm2_gemm4(const float* __restrict__ src, int div, int ksrc, const int* __restrict__ lists, const int* __restrict__ cnt,
              const int* __restrict__ active, const int* __restrict__ nact, int fixed_cnt, int lcap, const NmProj* __restrict__ tab,
              int K, int N, int S, float* __restrict__ partial, int cap, float* __restrict__ out, int relu2,
              const float* __restrict__ add, unsigned* __restrict__ tickets, int npass) {
    constexpr int RBMAX = DENSE ? kDenseRb : kRoutedRb;
    __shared__ __align__(16) float xs[RBMAX][128];
    __shared__ int rid_s[RBMAX], srow_s[RBMAX];
    __shared__ int last_s;
    if (!lists) {
        // A dense projection's table entry is fixed: bring this block's first tiles toward L2 while the producer of src finishes.
        const NmProj p0 = tab[0];
        if (is_fast(p0)) {
            const int kts = (K / 16) / S;
            const uint32_t* base = static_cast<const uint32_t*>(p0.w) + ((size_t)blockIdx.x * 8 + (threadIdx.x >> 3)) * 32 + 4 * (threadIdx.x & 7);
            const size_t kstride = (size_t)(N / 16) * 32;
            for (int j = 0; j < 4 && j < kts; ++j) prefetch_l2(base + (size_t)(blockIdx.y * kts + j) * kstride);
        }
    }
    pdl_wait();
    pdl_trigger();
    int g = 0, c_lo = (int)blockIdx.z * RBMAX, c_hi = min(fixed_cnt, c_lo + RBMAX);
    if (lists) {
        if (active && (int)blockIdx.z >= *nact) return;  // uniform per block
        g = active ? active[blockIdx.z] : (int)blockIdx.z;
        c_lo = 0;
        c_hi = cnt[g];
    }
    if (c_hi <= c_lo) return;
    const NmProj p = tab[g];
    if (!is_fast(p)) return;
    const int nb = blockIdx.x, s = blockIdx.y;
    for (int c0 = c_lo; c0 < c_hi; c0 += RBMAX) {
        const int here = min(RBMAX, c_hi - c0);
        __syncthreads();  // rid_s and xs of the previous pass are no longer read
        if ((int)threadIdx.x < here) {
            const int id = lists ? lists[(size_t)g * lcap + c0 + threadIdx.x] : c0 + (int)threadIdx.x;
            rid_s[threadIdx.x] = id;
            srow_s[threadIdx.x] = id / div;
        }
        if (here == 1) {
            fast_chunk<1>(src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        } else if (here == 2) {
            fast_chunk<2>(src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        } else if (here <= 4) {
            fast_chunk<4>(src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        } else if (here <= 8) {
            fast_chunk<8>(src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        } else if constexpr (DENSE) {
            fast_chunk<16>(src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        }
        // The output side: the last of the S slice blocks of this (group or row chunk, column block, pass) to write its partials runs it
        // (a ticket that wraps to zero by itself; the order of the sums does not depend on which block that is).
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned* tk = tickets + ((size_t)blockIdx.z * gridDim.x + nb) * npass + (c0 - c_lo) / RBMAX;
            last_s = atomicInc(tk, (unsigned)S - 1u) == (unsigned)S - 1u;
        }
        __syncthreads();
        if (last_s) {
            __threadfence();
            fast_epilogue(p, N, S, nb, here, partial, cap, rid_s, relu2, add, out);
        }
    }
}

// ---- any integer width: nm2_gemmk -----------------------------------------------------------------------------------------------
//
// nm2_gemm4's schedule and arithmetic for Mul1 projections of KB = 2 .. 8 bits (4 included), the width read per table entry, so a layer
// whose experts are stored at different widths (a MixedK pack: K 2 to 6, up and down of one expert often apart) runs in one launch. A
// tile of KB bits is 8 KB words (256 values of KB bits); thread (nt, a) owns its stream positions 32 a .. 32 a + 31 (columns a and a + 8),
// which are its KB words KB a .. KB a + KB - 1 and the last 16 - KB bits of the word before (its neighbour's last word, by shuffle,
// circular at a = 0). Value 32 a + i has its window at bit KB (i + 1) + 16 below the top of that word (E(p) = KB (p + 1), minus 16):
// for KB = 4 these are nm2_gemm4's offsets, and per output element every operation and its order are nm2_gemm4's (and so the reference
// kernels'), whatever the width.

// The 16-bit window that starts d bits below the top of w[0] in the big-endian stream w[0 .. NW - 1] (d a constant once unrolled).
template <int NW>
__device__ __forceinline__ uint32_t win16k(const uint32_t (&w)[NW], int d) {
    const int wi = d >> 5, off = d & 31;
    return off <= 16 ? ((w[wi] >> (16 - off)) & 0xffffu) : (__funnelshift_l(w[wi + 1 < NW ? wi + 1 : wi], w[wi], off) >> 16);
}

// The weights of one half tile (k rows 8 H .. 8 H + 7) of columns a (c0) and a + 8 (c1) at KB bits: c0[2 q + b] is position
// 8 q + b + 2 H of the thread's 32, c1[2 q + b] position 8 q + b + 2 H + 4 (decode_half's map at any width).
template <int KB, int H>
__device__ __forceinline__ void decode_half_k(const uint32_t (&w)[KB + 1], float (&c0)[8], float (&c1)[8]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int d = KB * (8 * q + 2 * H + 1) + 16;
        mul1_pair(win16k(w, d), win16k(w, d + KB), c0[2 * q], c0[2 * q + 1]);
        mul1_pair(win16k(w, d + 4 * KB), win16k(w, d + 5 * KB), c1[2 * q], c1[2 * q + 1]);
    }
}

// The thread's KB words at p (word offset KB a of a tile: 16-byte aligned for KB = 4 or 8, 8-byte for 2 and 6, 4-byte otherwise).
template <int KB>
__device__ __forceinline__ void ldg_words(const uint32_t* p, uint32_t (&w)[KB]) {
    if constexpr (KB % 4 == 0) {
#pragma unroll
        for (int i = 0; i < KB / 4; ++i) {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(p) + i);
            w[4 * i] = v.x;
            w[4 * i + 1] = v.y;
            w[4 * i + 2] = v.z;
            w[4 * i + 3] = v.w;
        }
    } else if constexpr (KB % 2 == 0) {
#pragma unroll
        for (int i = 0; i < KB / 2; ++i) {
            const uint2 v = __ldg(reinterpret_cast<const uint2*>(p) + i);
            w[2 * i] = v.x;
            w[2 * i + 1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < KB; ++i) w[i] = __ldg(p + i);
    }
}

// fast_chunk at KB bits: one slice of column block nb (and, PAIR, of nb + 1 too) for the pass's `here` rows (RB >= here, padding rows
// computed and dropped). A pair shares the block's rotated input (its prologue) and its tickets: what a block costs beside its weights
// is then amortised over twice the bytes, which the narrow widths need (a 2-bit column block of a slice is 8 KB). Per output element
// the operations and their order do not depend on PAIR. (The second column block's registers are named, not a second row of arrays:
// references to rows of a 2-D register array go to local memory.)
template <int RB, int KB, bool PAIR>
__device__ __forceinline__ void fastk_chunk(const float* __restrict__ src, int ksrc, const NmProj& p, int K, int N, int S, int nb, int s,
                                            int here, float* __restrict__ partial, int cap, float (*xs)[128], const int* rid_s,
                                            const int* srow_s) {
    constexpr int TW = 8 * KB;  // words of a tile
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, a = tid & 7, nt = tid >> 3;
    const int kts = (K / 16) / S, kt0 = s * kts, kt_end = kt0 + kts;
    const size_t kstride = (size_t)(N / 16) * TW;  // words between the k tiles of one column
    const uint32_t* base = static_cast<const uint32_t*>(p.w) + ((size_t)nb * 8 + nt) * TW + KB * a;
    const uint32_t* base2 = base + 8 * TW;  // the next column block's tile (PAIR)
    float acc0[RB], acc1[RB], bcc0[RB], bcc1[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc0[r] = acc1[r] = bcc0[r] = bcc1[r] = 0.f;

    uint32_t cur[KB], n1[KB], dcur[KB], d1[KB];
    ldg_words<KB>(base + (size_t)kt0 * kstride, cur);
    if constexpr (PAIR) ldg_words<KB>(base2 + (size_t)kt0 * kstride, dcur);
    if (KB <= 4 && kts > 1) {
        ldg_words<KB>(base + (size_t)(kt0 + 1) * kstride, n1);
        if constexpr (PAIR) ldg_words<KB>(base2 + (size_t)(kt0 + 1) * kstride, d1);
    } else {
#pragma unroll
        for (int i = 0; i < KB; ++i) {
            n1[i] = cur[i];
            d1[i] = dcur[i];
        }
    }
    for (int kt = kt0; kt < kt_end; ++kt) {
        // two tiles ahead up to 4 bits (nm2_gemm4's depth), one above (the wider tiles carry as many bytes in flight in fewer
        // registers; two ahead spills there)
        uint32_t n2[KB], d2[KB];
        if constexpr (KB <= 4) {
            if (kt + 2 < kt_end) {
                ldg_words<KB>(base + (size_t)(kt + 2) * kstride, n2);
                if constexpr (PAIR) ldg_words<KB>(base2 + (size_t)(kt + 2) * kstride, d2);
            } else {
#pragma unroll
                for (int i = 0; i < KB; ++i) {
                    n2[i] = cur[i];
                    d2[i] = dcur[i];
                }
            }
        } else {
            if (kt + 1 < kt_end) ldg_words<KB>(base + (size_t)(kt + 1) * kstride, n1);
        }
        if (kt == kt0 || (kt & 7) == 0) {
            // the rotated input of the 128-block kt >> 3 for every row of the pass (fast_chunk's prologue)
            __syncthreads();
            const int hb = kt >> 3;
            const float4 sv = __ldg(reinterpret_cast<const float4*>(p.suh + hb * 128 + 4 * lane));
            for (int r = warp; r < here; r += kFastThreads / 32) {
                const float4 xv = __ldg(reinterpret_cast<const float4*>(src + (size_t)srow_s[r] * ksrc + hb * 128 + 4 * lane));
                float v[4] = {xv.x * sv.x, xv.y * sv.y, xv.z * sv.z, xv.w * sv.w};
                fwht128_warp(v, lane);
                *reinterpret_cast<float4*>(&xs[r][4 * lane]) =
                    make_float4(v[0] * kInvSqrt128, v[1] * kInvSqrt128, v[2] * kInvSqrt128, v[3] * kInvSqrt128);
            }
            __syncthreads();
        }
        const float* xt = &xs[0][0] + (kt & 7) * 16;
        // the word before this thread's KB words is its neighbour's last (thread a = 0 takes the tile's last word from a = 7)
        const int from = (lane & ~7) | ((a + 7) & 7);
        {
            uint32_t w[KB + 1];
            w[0] = __shfl_sync(0xffffffffu, cur[KB - 1], from);
#pragma unroll
            for (int i = 0; i < KB; ++i) w[i + 1] = cur[i];
            float c0[8], c1[8];
            decode_half_k<KB, 0>(w, c0, c1);
            fma_half<RB>(acc0, acc1, xt, c0, c1);
            decode_half_k<KB, 1>(w, c0, c1);
            fma_half<RB>(acc0, acc1, xt + 8, c0, c1);
        }
        if constexpr (PAIR) {
            uint32_t w[KB + 1];
            w[0] = __shfl_sync(0xffffffffu, dcur[KB - 1], from);
#pragma unroll
            for (int i = 0; i < KB; ++i) w[i + 1] = dcur[i];
            float c0[8], c1[8];
            decode_half_k<KB, 0>(w, c0, c1);
            fma_half<RB>(bcc0, bcc1, xt, c0, c1);
            decode_half_k<KB, 1>(w, c0, c1);
            fma_half<RB>(bcc0, bcc1, xt + 8, c0, c1);
        }
#pragma unroll
        for (int i = 0; i < KB; ++i) {
            cur[i] = n1[i];
            dcur[i] = d1[i];
            if constexpr (KB <= 4) {
                n1[i] = n2[i];
                d1[i] = d2[i];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) {
        if (r < here) {
            float* dst = partial + ((size_t)s * cap + rid_s[r]) * N + (size_t)nb * 128 + nt * 16 + a;
            dst[0] = acc0[r];
            dst[8] = acc1[r];
            if constexpr (PAIR) {
                dst[128] = bcc0[r];
                dst[136] = bcc1[r];
            }
        }
    }
}

// The widths whose blocks take two column blocks (2 and 3 bits).
__device__ __forceinline__ bool paired(int twice) { return twice <= 6; }

// fastk_chunk at the entry's width (twice = 2 KB), over `nc` column blocks from nb (2 only for a paired width).
template <int RB>
__device__ __forceinline__ void fastk_any(int twice, int nc, const float* __restrict__ src, int ksrc, const NmProj& p, int K, int N, int S,
                                          int nb, int s, int here, float* __restrict__ partial, int cap, float (*xs)[128],
                                          const int* rid_s, const int* srow_s) {
#define FASTK(KB_, PAIR_) fastk_chunk<RB, KB_, PAIR_>(src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s)
#define FASTK_AT(KB_, NB_) fastk_chunk<RB, KB_, false>(src, ksrc, p, K, N, S, NB_, s, here, partial, cap, xs, rid_s, srow_s)
    // a pair in one pass of the k tiles up to 2 rows a pass (the decode and the narrow verify windows, where the block's fixed cost is what
    // matters); a wider pass takes its two column blocks one after the other (the registers of two would spill)
    switch (twice) {
        case 4:
            if (nc == 1) FASTK(2, false);
            else if constexpr (RB <= 2) FASTK(2, true);
            else { FASTK_AT(2, nb); FASTK_AT(2, nb + 1); }
            break;
        case 6:
            if (nc == 1) FASTK(3, false);
            else if constexpr (RB <= 2) FASTK(3, true);
            else { FASTK_AT(3, nb); FASTK_AT(3, nb + 1); }
            break;
        case 8: FASTK(4, false); break;
        case 10: FASTK(5, false); break;
        case 12: FASTK(6, false); break;
        case 14: FASTK(7, false); break;
        default: FASTK(8, false); break;
    }
#undef FASTK_AT
#undef FASTK
}

// nm2_gemm4 for every entry is_fastk takes (the arguments, grid, passes, tickets and output side are nm2_gemm4's). The routed
// instantiation asks for 7 blocks an SM, not 8: at 8 (128 registers) the 8-bit paths spill, at 7 none does (ptxas -v).
template <bool DENSE>
__global__ void __launch_bounds__(kFastThreads, DENSE ? 6 : 7)
    nm2_gemmk(const float* __restrict__ src, int div, int ksrc, const int* __restrict__ lists, const int* __restrict__ cnt,
              const int* __restrict__ active, const int* __restrict__ nact, int fixed_cnt, int lcap, const NmProj* __restrict__ tab,
              int K, int N, int S, float* __restrict__ partial, int cap, float* __restrict__ out, int relu2,
              const float* __restrict__ add, unsigned* __restrict__ tickets, int npass) {
    constexpr int RBMAX = DENSE ? kDenseRb : kRoutedRb;
    __shared__ __align__(16) float xs[RBMAX][128];
    __shared__ int rid_s[RBMAX], srow_s[RBMAX];
    __shared__ int last_s;
    if (!lists) {
        const NmProj p0 = tab[0];
        if (is_fastk(p0)) {
            const int kts = (K / 16) / S, kb = p0.twice / 2;
            const uint32_t* base =
                static_cast<const uint32_t*>(p0.w) + ((size_t)blockIdx.x * 8 + (threadIdx.x >> 3)) * (8 * kb) + kb * (threadIdx.x & 7);
            const size_t kstride = (size_t)(N / 16) * (8 * kb);
            for (int j = 0; j < 4 && j < kts; ++j) prefetch_l2(base + (size_t)(blockIdx.y * kts + j) * kstride);
        }
    }
    pdl_wait();
    pdl_trigger();
    int g = 0, c_lo = (int)blockIdx.z * RBMAX, c_hi = min(fixed_cnt, c_lo + RBMAX);
    if (lists) {
        if (active && (int)blockIdx.z >= *nact) return;  // uniform per block
        g = active ? active[blockIdx.z] : (int)blockIdx.z;
        c_lo = 0;
        c_hi = cnt[g];
    }
    if (c_hi <= c_lo) return;
    const NmProj p = tab[g];
    if (!is_fastk(p)) return;
    // a paired width's block x takes column blocks 2x and 2x + 1 (the grid is the same for every width of the table; the blocks past
    // the pairs exit)
    int nb = blockIdx.x, nc = 1;
    if (paired(p.twice)) {
        nb = 2 * (int)blockIdx.x;
        if (nb >= N / 128) return;  // uniform per block
        nc = min(2, N / 128 - nb);
    }
    const int s = blockIdx.y;
    for (int c0 = c_lo; c0 < c_hi; c0 += RBMAX) {
        const int here = min(RBMAX, c_hi - c0);
        __syncthreads();
        if ((int)threadIdx.x < here) {
            const int id = lists ? lists[(size_t)g * lcap + c0 + threadIdx.x] : c0 + (int)threadIdx.x;
            rid_s[threadIdx.x] = id;
            srow_s[threadIdx.x] = id / div;
        }
        if (here == 1) {
            fastk_any<1>(p.twice, nc, src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        } else if (here == 2) {
            fastk_any<2>(p.twice, nc, src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        } else if (here <= 4) {
            fastk_any<4>(p.twice, nc, src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        } else if (here <= 8) {
            fastk_any<8>(p.twice, nc, src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        } else if constexpr (DENSE) {
            fastk_any<16>(p.twice, nc, src, ksrc, p, K, N, S, nb, s, here, partial, cap, xs, rid_s, srow_s);
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned* tk = tickets + ((size_t)blockIdx.z * gridDim.x + nb) * npass + (c0 - c_lo) / RBMAX;
            last_s = atomicInc(tk, (unsigned)S - 1u) == (unsigned)S - 1u;
        }
        __syncthreads();
        if (last_s) {
            __threadfence();
            for (int c = 0; c < nc; ++c) fast_epilogue(p, N, S, nb + c, here, partial, cap, rid_s, relu2, add, out);
        }
    }
}

// ---- a dense fp16 projection in one launch: nm2_dense16 ---------------------------------------------------------------------------
//
// out[r, n] = (x[r] . W[n]) for a F16 weight W [N, K] (HF layout), then relu2 and + add as nm_t2 applies them: the reference launches'
// bits (nm_t1 passes x through, nm_gemv_bf16's chain on the fp16 values, nm_t2 takes slot 0) without the copy, the partials or the two
// extra launches. Block of 8 warps, warp w is column blockIdx.x * 8 + w; lane l loads the weight chunks l + 32 q (q increasing) once for
// RB rows at a time, each (row, column) chain those chunks in increasing q, then the xor tree 16 -> 1.
template <int RB>
__global__ void __launch_bounds__(256) nm2_dense16(const float* __restrict__ x, const uint16_t* __restrict__ w, int rows, int K, int N,
                                                   float* __restrict__ out, int relu2, const float* __restrict__ add) {
    pdl_wait();
    pdl_trigger();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;  // a whole warp returns
    const uint16_t* wrow = w + (size_t)n * K;
    const int chunks = K / 8;
    for (int r0 = 0; r0 < rows; r0 += RB) {
        float acc[RB];
#pragma unroll
        for (int r = 0; r < RB; ++r) acc[r] = 0.f;
#pragma unroll 4
        for (int q = lane; q < chunks; q += 32) {
            const uint4 wv = __ldg(reinterpret_cast<const uint4*>(wrow + 8 * q));
#pragma unroll
            for (int r = 0; r < RB; ++r) {
                if (r0 + r < rows) {
                    const float* xr = x + (size_t)(r0 + r) * K + 8 * q;
                    acc[r] = chunk_fma_h(acc[r], __ldg(reinterpret_cast<const float4*>(xr)), __ldg(reinterpret_cast<const float4*>(xr + 4)), wv);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RB; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
            if (lane == 0 && r0 + r < rows) {
                float v = acc[r];
                if (relu2) {
                    const float q = fmaxf(v, 0.f);
                    v = q * q;
                }
                if (add) v = v + add[(size_t)(r0 + r) * N + n];
                out[(size_t)(r0 + r) * N + n] = v;
            }
        }
    }
}

// Router logits, as nm_router_logits computes them (lane l owns the 8-element chunks l + 32 i of the K = 4096 gate row; each row's chain is
// the lane's chunks in increasing i, then the xor tree 16 -> 1), but a warp carries EW experts for RB rows, so a chunk of x is loaded
// once for EW experts and the 16 chunks of a gate row are in flight together instead of one at a time.
// G: the gate's element type. An f32 gate carries twice the bytes a chunk, so the one-expert layout prefetches the first half of its
// row before the wait (16 x 8 floats would hold 128 registers) and loads the rest in the loop; the bits do not depend on when a chunk
// is loaded.
template <int RB, int EW, typename G>
__global__ void __launch_bounds__(128) nm2_router_logits(const float* __restrict__ h, const G* __restrict__ gate,
                                                         float* __restrict__ logits, int rows) {
    constexpr int K = 4096;
    constexpr int NCH = K / 8 / 32;  // chunks a lane owns
    constexpr int PRE = EW == 1 ? (sizeof(typename GateChunk<G>::V) == 16 ? NCH : NCH / 2) : 0;
    using V = typename GateChunk<G>::V;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e0 = (blockIdx.x * 4 + warp) * EW;
    if (e0 >= kExperts) return;  // a whole warp returns
    const int r0 = blockIdx.y * RB;
    // One expert a warp: the chunks of the gate row are loaded before the producer of h is waited for (weights, not its output).
    V wpre[PRE > 0 ? PRE : 1];
    if constexpr (PRE > 0) {
#pragma unroll
        for (int i = 0; i < PRE; ++i) wpre[i] = GateChunk<G>::load(gate + (size_t)e0 * K + 8 * (lane + 32 * i));
    }
    pdl_wait();
    pdl_trigger();
    float acc[RB][EW];
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
        for (int e = 0; e < EW; ++e) acc[r][e] = 0.f;
#pragma unroll
    for (int i = 0; i < NCH; ++i) {
        const int q = lane + 32 * i;
        V w[EW];
#pragma unroll
        for (int e = 0; e < EW; ++e) {
            if (i < PRE) w[e] = wpre[i < PRE ? i : 0];
            else w[e] = GateChunk<G>::load(gate + (size_t)(e0 + e) * K + 8 * q);
        }
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            if (r0 + r < rows) {
                const float* xr = h + (size_t)(r0 + r) * K + 8 * q;
                const float4 xa = __ldg(reinterpret_cast<const float4*>(xr)), xb = __ldg(reinterpret_cast<const float4*>(xr + 4));
#pragma unroll
                for (int e = 0; e < EW; ++e) acc[r][e] = chunk_fma(acc[r][e], xa, xb, w[e]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) {
#pragma unroll
        for (int e = 0; e < EW; ++e) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc[r][e] += __shfl_xor_sync(0xffffffffu, acc[r][e], off);
            if (lane == 0 && r0 + r < rows) logits[(size_t)(r0 + r) * kExperts + e0 + e] = acc[r][e];
        }
    }
}

// Selection, weights and the expert buckets in one launch: block `row` selects that row's 22 experts, and the last block to finish
// (a ticket on `ctr`, which it resets) builds the buckets of every row. One thread per expert.
//
// Selection: the key of expert t is sigmoid(logit) + bias, ordered (key descending, id ascending) through the packed comp of
// nm_router_select. Round k takes the largest remaining comp of the block (a warp reduction of the code word, then of the inverted id
// among the lanes that hold it, then the same across the 16 warps) and its owner leaves the race; the 22 rounds give the experts in rank
// order, the oracle's. Weights: the selected sigmoids summed in rank order, + 1e-20, then (sig / sum) * scale.
// Buckets: as nm_bucket (each expert's ascending pair ids), plus the compacted list of experts with rows. Up to 64 rows build them
// from a 64-bit token mask per expert (pairs in ascending token order, one pair per token and expert); more rows scan.
__global__ void __launch_bounds__(512) nm2_select(const float* __restrict__ logits, const float* __restrict__ bias, float scale,
                                                  int* __restrict__ pick, float* __restrict__ wts, int rows, int cap,
                                                  int* __restrict__ cnt, int* __restrict__ lists, int* __restrict__ active,
                                                  int* __restrict__ nact, unsigned* __restrict__ ctr) {
    __shared__ uint32_t wb_hi[2][16], wb_lo[2][16];
    __shared__ float sel_sig[kTopK];
    __shared__ unsigned long long mask[kExperts];
    __shared__ int warp_tot[16];
    __shared__ int last;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, row = blockIdx.x;

    pdl_wait();
    pdl_trigger();
    const float sig = sigmoid_f(logits[(size_t)row * kExperts + t]);
    const float key = sig + bias[t];
    uint32_t hi = key_code(key), lo = 0xFFFFFFFFu - (uint32_t)t;
    for (int k = 0; k < kTopK; ++k) {
        const uint32_t mh = __reduce_max_sync(0xffffffffu, hi);
        const uint32_t ml = __reduce_max_sync(0xffffffffu, hi == mh ? lo : 0u);
        if (lane == 0) {
            wb_hi[k & 1][w] = mh;
            wb_lo[k & 1][w] = ml;
        }
        __syncthreads();
        const uint32_t vh = lane < 16 ? wb_hi[k & 1][lane] : 0u, vl = lane < 16 ? wb_lo[k & 1][lane] : 0u;
        const uint32_t bh = __reduce_max_sync(0xffffffffu, vh);
        const uint32_t bl = __reduce_max_sync(0xffffffffu, vh == bh ? vl : 0u);
        const int id = (int)(0xFFFFFFFFu - bl);
        if (t == id) {
            sel_sig[k] = sig;
            pick[(size_t)row * kTopK + k] = id;
            hi = 0;  // out of the race (a taken comp is 0, as in nm_router_select)
            lo = 0;
        }
        (void)bh;
    }
    __syncthreads();
    if (t < kTopK) {
        float denom = 0.f;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) denom = denom + sel_sig[k];
        denom = denom + 1e-20f;
        wts[(size_t)row * kTopK + t] = (sel_sig[t] / denom) * scale;
    }
    __threadfence();  // this block's picks and weights are visible before its ticket is
    __syncthreads();
    if (t == 0) last = (atomicAdd(ctr, 1u) == (unsigned)rows - 1u);
    __syncthreads();
    if (!last) return;
    __threadfence();

    // the last block: buckets of all rows
    const int P = rows * kTopK;
    int n = 0;
    int* out = lists + (size_t)t * cap;
    if (rows <= 64) {
        mask[t] = 0ull;
        __syncthreads();
        for (int p = t; p < P; p += 512) atomicOr(&mask[__ldcg(pick + p)], 1ull << (p / kTopK));
        __syncthreads();
        unsigned long long m = mask[t];
        while (m) {
            const int tok = __ffsll((long long)m) - 1;
            m &= m - 1ull;
            int slot = 0;
            for (int sl = 0; sl < kTopK; ++sl)
                if (__ldcg(pick + tok * kTopK + sl) == t) {
                    slot = sl;
                    break;
                }
            out[n++] = tok * kTopK + slot;
        }
    } else {
        for (int p = 0; p < P; ++p)
            if (__ldcg(pick + p) == t) out[n++] = p;
    }
    cnt[t] = n;
    const unsigned ball = __ballot_sync(0xffffffffu, n > 0);
    if (lane == 0) warp_tot[w] = __popc(ball);
    __syncthreads();
    int base = 0;
    for (int i = 0; i < w; ++i) base += warp_tot[i];
    if (n > 0) active[base + __popc(ball & ((1u << lane) - 1u))] = t;
    if (t == kExperts - 1) {
        *nact = base + __popc(ball);
        *ctr = 0u;  // the next call's tickets start again
    }
}

}  // namespace

// Debug: W_q [K, N] fp16 bits of projection tab[idx] (EXL3), row-major, as gb_exl3::unpack writes it.
__global__ void __launch_bounds__(256) nm_decode(const NmProj* __restrict__ tab, int idx, uint16_t* __restrict__ out) {
    const NmProj p = tab[idx];
    const int N = p.n, K = p.k;
    const size_t total = (size_t)K * N;
    const uint32_t* trel = static_cast<const uint32_t*>(p.w);
    const int nt_total = N / 16;
    const int nw = 4 * p.twice;
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x) {
        const int k = (int)(e / N), n = (int)(e % N);
        const int kk = k >> 4, kr = k & 15, nn = n >> 4, c = n & 15;
        const uint32_t st = tile_state(trel + ((size_t)kk * nt_total + nn) * nw, p.twice, tile_pos(kr, c));
        out[e] = p.cb[st];
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Launchers

// The fast path's launches carry the programmatic-stream-serialization attribute (see pdl_wait). `gb_nm_set_pdl(0)` turns it off, for
// timing the difference; the bits do not depend on it.
static int g_pdl = 1;

GB_EXPORT int gb_nm_set_pdl(int on) {
    g_pdl = on;
    return 0;
}

template <typename... KArgs, typename... Args>
static int launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = g_pdl ? 1 : 0;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
    return e != cudaSuccess ? (int)e : GB_CHECK_LAUNCH();
}

template <int RB, typename G>
static int launch_router_logits(const float* h, const G* gate, float* logits, int rows, int K, int E, cudaStream_t st) {
    const dim3 grid((unsigned)((E + 7) / 8), (unsigned)((rows + RB - 1) / RB));
    nm_router_logits<RB, G><<<grid, 256, 0, st>>>(h, gate, logits, rows, K, E);
    return GB_CHECK_LAUNCH();
}

template <typename G>
static int router_logits_ref(const void* h, const void* gate, void* logits, int rows, int K, int E, int rb, void* stream) {
    if (rows < 1 || K % 8 != 0 || E != kExperts || !h || !gate || !logits) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const float* hp = static_cast<const float*>(h);
    const G* gp = static_cast<const G*>(gate);
    float* lp = static_cast<float*>(logits);
    if (rb == 1) return launch_router_logits<1, G>(hp, gp, lp, rows, K, E, st);
    if (rb == 4) return launch_router_logits<4, G>(hp, gp, lp, rows, K, E, st);
    if (rb == 16) return launch_router_logits<16, G>(hp, gp, lp, rows, K, E, st);
    return (int)cudaErrorInvalidValue;
}

// gate: bf16 bits [E, K].
GB_EXPORT int gb_nm_router_logits(const void* h, const void* gate, void* logits, int rows, int K, int E, int rb, void* stream) {
    return router_logits_ref<uint16_t>(h, gate, logits, rows, K, E, rb, stream);
}

// The same with an f32 gate [E, K] (an F16 gate widened exactly): the same chains, the gate's values read as f32.
GB_EXPORT int gb_nm_router_logits_g32(const void* h, const void* gate, void* logits, int rows, int K, int E, int rb, void* stream) {
    return router_logits_ref<float>(h, gate, logits, rows, K, E, rb, stream);
}

// The same with an F16 gate [E, K] as stored (fp16 bits): the bits of gb_nm_router_logits_g32 on the widened gate.
GB_EXPORT int gb_nm_router_logits_g16(const void* h, const void* gate, void* logits, int rows, int K, int E, int rb, void* stream) {
    return router_logits_ref<__half>(h, gate, logits, rows, K, E, rb, stream);
}

GB_EXPORT int gb_nm_router_select(const void* logits, const void* bias, float scale, void* pick, void* wts, int rows, void* stream) {
    if (rows < 1 || !logits || !bias || !pick || !wts) return (int)cudaErrorInvalidValue;
    nm_router_select<<<rows, 32, 0, gb_stream(stream)>>>(static_cast<const float*>(logits), static_cast<const float*>(bias), scale,
                                                         static_cast<int*>(pick), static_cast<float*>(wts));
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nm_bucket(const void* pick, int P, int cap, void* cnt, void* lists, void* stream) {
    if (P < 1 || cap < 1 || !pick || !cnt || !lists) return (int)cudaErrorInvalidValue;
    nm_bucket<<<1, kExperts, 0, gb_stream(stream)>>>(static_cast<const int*>(pick), P, cap, static_cast<int*>(cnt),
                                                    static_cast<int*>(lists));
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nm_t1(const void* src, int div, const void* expert_of_row, const void* tab, int rows, int K, void* dst, int skip_fast,
                       void* stream) {
    if (rows < 1 || div < 1 || K % 128 != 0 || !src || !tab || !dst) return (int)cudaErrorInvalidValue;
    nm_t1<<<rows, 128, 0, gb_stream(stream)>>>(static_cast<const float*>(src), div, static_cast<const int*>(expert_of_row),
                                              static_cast<const NmProj*>(tab), rows, K, static_cast<float*>(dst), skip_fast);
    return GB_CHECK_LAUNCH();
}

// groups: the table's entries that run (512 for the experts, 1 for a dense projection). N % 128 == 0, (K / 16) % S == 0.
GB_EXPORT int gb_nm_gemm_exl3(const void* xin, const void* lists, const void* cnt, int fixed_cnt, int lcap, const void* tab, int groups,
                              int K, int N, int S, void* partial, int cap, int skip_fast, const void* active, const void* nact,
                              void* stream) {
    if (groups < 1 || K % 16 != 0 || N % 128 != 0 || S < 1 || (K / 16) % S != 0 || !xin || !tab || !partial || (active && !nact))
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(N / 128), (unsigned)S, (unsigned)groups);
    nm_gemm_exl3<<<grid, 128, 0, gb_stream(stream)>>>(static_cast<const float*>(xin), static_cast<const int*>(lists),
                                                     static_cast<const int*>(cnt), fixed_cnt, lcap, static_cast<const NmProj*>(tab), K, N,
                                                     S, static_cast<float*>(partial), cap, skip_fast,
                                                     static_cast<const int*>(active), static_cast<const int*>(nact));
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nm_gemv_bf16(const void* xin, const void* lists, const void* cnt, int fixed_cnt, int lcap, const void* tab, int groups,
                              int K, int N, void* partial, int cap, const void* active, const void* nact, void* stream) {
    if (groups < 1 || K % 8 != 0 || N < 1 || !xin || !tab || !partial || (active && !nact)) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)((N + 7) / 8), (unsigned)groups);
    nm_gemv_bf16<<<grid, 256, 0, gb_stream(stream)>>>(static_cast<const float*>(xin), static_cast<const int*>(lists),
                                                     static_cast<const int*>(cnt), fixed_cnt, lcap, static_cast<const NmProj*>(tab), K, N,
                                                     static_cast<float*>(partial), cap, static_cast<const int*>(active),
                                                     static_cast<const int*>(nact));
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nm_t2(const void* partial, int S, int cap, const void* expert_of_row, const void* tab, int rows, int N, int relu2,
                       const void* add, void* out, int skip_fast, void* stream) {
    if (rows < 1 || N % 128 != 0 || S < 1 || !partial || !tab || !out) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(N / 128), (unsigned)rows);
    nm_t2<<<grid, 128, 0, gb_stream(stream)>>>(static_cast<const float*>(partial), S, cap, static_cast<const int*>(expert_of_row),
                                              static_cast<const NmProj*>(tab), rows, N, relu2, static_cast<const float*>(add),
                                              static_cast<float*>(out), skip_fast);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nm_combine(const void* y, const void* wts, int rows, int N, void* routed, void* stream) {
    if (rows < 1 || N % 128 != 0 || !y || !wts || !routed) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(N / 128), (unsigned)rows);
    return launch_pdl(nm_combine, grid, dim3(128), gb_stream(stream), static_cast<const float*>(y), static_cast<const float*>(wts), N,
                      static_cast<float*>(routed));
}

// The fast projection of `zgroups` groups, or of ceil(fixed_cnt / 16) row chunks of one group when `lists` is null (the entries of tab
// that are 4-bit Mul1 run; the others are left to the reference kernels, nm_t2 with skip_fast included). `active`/`nact`: the compacted
// list of groups with rows (nm2_select), or null for z = the group. The output side runs in the last block of each (z, column block,
// pass): `out` [rows, N], relu2, and `add` [rows, N] or null; `tickets` is zero on entry and on exit, with at least
// zgroups * (N / 128) * npass entries, npass the passes of 8 rows of the longest group list. N % 128 == 0, (K / 16) % S == 0.
GB_EXPORT int gb_nm2_gemm4(const void* src, int div, int ksrc, const void* lists, const void* cnt, const void* active, const void* nact,
                           int fixed_cnt, int lcap, const void* tab, int zgroups, int K, int N, int S, void* partial, int cap,
                           void* out, int relu2, const void* add, void* tickets, int npass, void* stream) {
    if (zgroups < 1 || div < 1 || K % 128 != 0 || N % 128 != 0 || S < 1 || (K / 16) % S != 0 || ksrc < K || npass < 1 || !src || !tab ||
        !partial || !out || !tickets || (active && !nact))
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(N / 128), (unsigned)S, (unsigned)zgroups);
    return lists ? launch_pdl(nm2_gemm4<false>, grid, dim3(kFastThreads), gb_stream(stream), static_cast<const float*>(src), div, ksrc,
                              static_cast<const int*>(lists), static_cast<const int*>(cnt), static_cast<const int*>(active),
                              static_cast<const int*>(nact), fixed_cnt, lcap, static_cast<const NmProj*>(tab), K, N, S,
                              static_cast<float*>(partial), cap, static_cast<float*>(out), relu2, static_cast<const float*>(add),
                              static_cast<unsigned*>(tickets), npass)
                 : launch_pdl(nm2_gemm4<true>, grid, dim3(kFastThreads), gb_stream(stream), static_cast<const float*>(src), div, ksrc,
                              static_cast<const int*>(lists), static_cast<const int*>(cnt), static_cast<const int*>(active),
                              static_cast<const int*>(nact), fixed_cnt, lcap, static_cast<const NmProj*>(tab), K, N, S,
                              static_cast<float*>(partial), cap, static_cast<float*>(out), relu2, static_cast<const float*>(add),
                              static_cast<unsigned*>(tickets), npass);
}

// gb_nm2_gemm4 for the entries of every integer width from 2 to 8 bits (Mul1), 4 included: the same arguments and contract.
GB_EXPORT int gb_nm2_gemmk(const void* src, int div, int ksrc, const void* lists, const void* cnt, const void* active, const void* nact,
                           int fixed_cnt, int lcap, const void* tab, int zgroups, int K, int N, int S, void* partial, int cap,
                           void* out, int relu2, const void* add, void* tickets, int npass, void* stream) {
    if (zgroups < 1 || div < 1 || K % 128 != 0 || N % 128 != 0 || S < 1 || (K / 16) % S != 0 || ksrc < K || npass < 1 || !src || !tab ||
        !partial || !out || !tickets || (active && !nact))
        return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(N / 128), (unsigned)S, (unsigned)zgroups);
    return lists ? launch_pdl(nm2_gemmk<false>, grid, dim3(kFastThreads), gb_stream(stream), static_cast<const float*>(src), div, ksrc,
                              static_cast<const int*>(lists), static_cast<const int*>(cnt), static_cast<const int*>(active),
                              static_cast<const int*>(nact), fixed_cnt, lcap, static_cast<const NmProj*>(tab), K, N, S,
                              static_cast<float*>(partial), cap, static_cast<float*>(out), relu2, static_cast<const float*>(add),
                              static_cast<unsigned*>(tickets), npass)
                 : launch_pdl(nm2_gemmk<true>, grid, dim3(kFastThreads), gb_stream(stream), static_cast<const float*>(src), div, ksrc,
                              static_cast<const int*>(lists), static_cast<const int*>(cnt), static_cast<const int*>(active),
                              static_cast<const int*>(nact), fixed_cnt, lcap, static_cast<const NmProj*>(tab), K, N, S,
                              static_cast<float*>(partial), cap, static_cast<float*>(out), relu2, static_cast<const float*>(add),
                              static_cast<unsigned*>(tickets), npass);
}

// A dense F16 projection (weight w [N, K] fp16 bits, HF layout) of x [rows, K] into out [rows, N], relu2 and add [rows, N] (or null) as
// nm_t2 applies them, in one launch (nm2_dense16). K % 8 == 0.
GB_EXPORT int gb_nm2_dense16(const void* x, const void* w, int rows, int K, int N, void* out, int relu2, const void* add, void* stream) {
    if (rows < 1 || K % 8 != 0 || N < 1 || !x || !w || !out) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)((N + 7) / 8));
    const float* xp = static_cast<const float*>(x);
    const uint16_t* wp = static_cast<const uint16_t*>(w);
    float* op = static_cast<float*>(out);
    const float* ap = static_cast<const float*>(add);
    if (rows == 1) return launch_pdl(nm2_dense16<1>, grid, dim3(256), gb_stream(stream), xp, wp, rows, K, N, op, relu2, ap);
    if (rows <= 2) return launch_pdl(nm2_dense16<2>, grid, dim3(256), gb_stream(stream), xp, wp, rows, K, N, op, relu2, ap);
    return launch_pdl(nm2_dense16<4>, grid, dim3(256), gb_stream(stream), xp, wp, rows, K, N, op, relu2, ap);
}

// Router logits for K = 4096 and E = 512 on the warp-per-EW-experts layout; rb is the rows a warp carries (1, 4, 8 or 16), ew the
// experts (1, 2 or 4). The sums are those of gb_nm_router_logits.
template <typename G>
static int router_logits_fast(const void* h, const void* gate, void* logits, int rows, int rb, int ew, void* stream) {
    if (rows < 1 || !h || !gate || !logits) return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const float* hp = static_cast<const float*>(h);
    const G* gp = static_cast<const G*>(gate);
    float* lp = static_cast<float*>(logits);
#define NM2_LOGITS(RB_, EW_)                                                                                                  \
    if (rb == RB_ && ew == EW_) {                                                                                             \
        const dim3 grid((unsigned)((kExperts / EW_ + 3) / 4), (unsigned)((rows + RB_ - 1) / RB_));                            \
        return launch_pdl(nm2_router_logits<RB_, EW_, G>, grid, dim3(128), st, hp, gp, lp, rows);                             \
    }
    NM2_LOGITS(1, 1)
    NM2_LOGITS(1, 2)
    NM2_LOGITS(1, 4)
    NM2_LOGITS(2, 2)
    NM2_LOGITS(4, 1)
    NM2_LOGITS(4, 2)
    NM2_LOGITS(4, 4)
    NM2_LOGITS(8, 2)
    NM2_LOGITS(8, 4)
    NM2_LOGITS(16, 2)
    NM2_LOGITS(16, 4)
#undef NM2_LOGITS
    return (int)cudaErrorInvalidValue;
}

// gate: bf16 bits [512, 4096].
GB_EXPORT int gb_nm2_router_logits(const void* h, const void* gate, void* logits, int rows, int rb, int ew, void* stream) {
    return router_logits_fast<uint16_t>(h, gate, logits, rows, rb, ew, stream);
}

// The same with an f32 gate [512, 4096] (an F16 gate widened exactly); the sums are those of gb_nm_router_logits_g32.
GB_EXPORT int gb_nm2_router_logits_g32(const void* h, const void* gate, void* logits, int rows, int rb, int ew, void* stream) {
    return router_logits_fast<float>(h, gate, logits, rows, rb, ew, stream);
}

// The same with an F16 gate [512, 4096] as stored: the bits of gb_nm2_router_logits_g32 on the widened gate.
GB_EXPORT int gb_nm2_router_logits_g16(const void* h, const void* gate, void* logits, int rows, int rb, int ew, void* stream) {
    return router_logits_fast<__half>(h, gate, logits, rows, rb, ew, stream);
}

// Selection, weights and buckets in one launch (see nm2_select). `ctr` is one zero-initialised u32 that the kernel leaves at zero.
GB_EXPORT int gb_nm2_select(const void* logits, const void* bias, float scale, void* pick, void* wts, int rows, int cap, void* cnt,
                            void* lists, void* active, void* nact, void* ctr, void* stream) {
    if (rows < 1 || cap < 1 || !logits || !bias || !pick || !wts || !cnt || !lists || !active || !nact || !ctr) return (int)cudaErrorInvalidValue;
    return launch_pdl(nm2_select, dim3((unsigned)rows), dim3(512), gb_stream(stream), static_cast<const float*>(logits),
                      static_cast<const float*>(bias), scale, static_cast<int*>(pick), static_cast<float*>(wts), rows, cap,
                      static_cast<int*>(cnt), static_cast<int*>(lists), static_cast<int*>(active), static_cast<int*>(nact),
                      static_cast<unsigned*>(ctr));
}

GB_EXPORT int gb_nm_decode(const void* tab, int idx, void* out, void* stream) {
    if (!tab || !out) return (int)cudaErrorInvalidValue;
    nm_decode<<<1024, 256, 0, gb_stream(stream)>>>(static_cast<const NmProj*>(tab), idx, static_cast<uint16_t*>(out));
    return GB_CHECK_LAUNCH();
}
