// gb-nvcc: --fmad=false
// Large-row prefill path of the Nemotron 3.5 Super VL latent-MoE block (the E layers), for slices of 65 to 512 rows (it is correct for
// any row count from 1). Package P1; docs/NEMOTRON_MOE_PREFILL.md has the design and the numbers.
//
// The per-row arithmetic is the decode kernels' (csrc/nemotron_moe.cu), operation for operation: a projection's k sum is S slice
// chains of fmaf over the slice's EXL3 k tiles (the 16 k rows of a tile in order, from zero), added in slice order; the input is
// H_128(x * suh) * (1 / sqrt 128) per 128-block; the output is H_128 of the slice sum, * svh, + bias, then relu2 and + add. Nothing
// here changes the order of any floating-point operation of a row, so a row's bits do not depend on the number of rows in the call, on
// the row's position in it, or on which block carried it, and they equal NemotronMoe::forward_in's.
//
// What differs is the schedule. The decode path's routed GEMM carries at most 8 rows per pass of an (expert, slice) block, with S partial
// sums per output in global memory and a ticket to find the block that adds them, and every block rotates its own input in a prologue.
// Here a block carries up to 32 rows of ONE expert through the whole K of 128 output columns: the expert's weights are decoded once for
// all of them, the S slice sums are added in slice order (running sums in registers, the decode path's epilogue), and no partial sum or
// ticket exists. The input rows are rotated once per (row, projection) by pf_rotate (and the next projection's input by the epilogue of
// this one), so the block's loop is weight tiles and FMAs: the tiles come through a ring of cp.async copies, the rotated rows through a
// double buffer. A block holds 14 to 22 KB of shared memory and an SM 100 KB: the blocks (the warps that hide each other's latency) an
// SM keeps are set by that and by registers, which is why the sums are in registers and share their bytes with the x buffers. The rows
// of each expert come from a counting sort of the picks (ascending pair ids, as the decode path's lists), and the blocks are work items
// (expert, first row, rows), the experts with many rows and the ones with few alternating.
//
// Kernels, in launch order (crates/cuda/src/nemotron_moe_prefill.rs holds the sequence):
//   (router logits: nemotron_moe.cu's gb_nm2_router_logits, unchanged)
//   pf_select        top-22 selection and weights (nm2_select's, without its serial bucket stage)
//   pf_bucket_lists  per expert, the ascending pair ids that picked it (one block an expert)
//   pf_bucket_work   the work items: experts ordered by row count, split in chunks of at most 32 rows (16, mode 2), fast and slow apart
//   pf_rotate        the input side of a projection (x * suh, H_128, 1/sqrt 128) for a set of rows
//   pf_gemm4         a 4-bit Mul1 projection: routed (work items) or dense (row chunks); fused rotation of the next projection's input
//   pf_slow, pf_slow4  a bf16 or absent expert's projection, or a dense bf16 projection (the bf16 chain of nm_gemv_bf16, rows blocked;
//                    pf_slow4 takes four columns and eight rows a warp)
//
// Measurement switches (not defined in a build): PF_KSTRIDE0 reads the same weight tile at every k (no DRAM traffic), PF_NOFMA drops the
// multiply-adds (decode, staging and the epilogue remain). Their outputs are wrong by construction; docs/NEMOTRON_MOE_PREFILL.md uses
// them to split a block's time into decode, FMA and memory.

#include <stdint.h>

#include "common.cuh"

namespace {

constexpr int kFmtExl3 = 0;
constexpr int kFmtBf16 = 1;
constexpr int kFmtF16 = 3;  // fp16 weight [N, K] (a MixedK pack's latent projections): the bf16 chain on the values widened exactly
constexpr int kTopK = 22;
constexpr int kExperts = 512;
constexpr float kInvSqrt128 = 0.08838834764831845f;
constexpr int kCbMul1 = 2;
constexpr int kFastTwice = 8;
constexpr int kThreads = 64;  // two warps: 8 column tiles of 16, 8 threads each
constexpr int kClassRb = 16;  // an expert with at most this many rows is one item of the small class
constexpr int kMaxRb = 32;    // rows of a work item (the large class)
constexpr int kDepth = 4;     // weight tiles in flight per block
constexpr int kXChunk = 4;    // k tiles (16 k each) of the rotated rows staged at a time (>= kDepth: the x chunk rides in a ring group)
constexpr int kMinBlocksSmall = 6;  // blocks of 64 threads an SM keeps resident (launch bounds): the 16-row kernel (168 registers)
constexpr int kMinBlocksLarge = 4;  // the 32-row kernel (254 registers, 22 KB of shared memory)

// One projection, as nemotron_moe.cu's NmProj (56 bytes; the Rust side builds the same table).
struct PfProj {
    const void* w;
    const float* suh;
    const float* svh;
    const float* bias;
    const uint16_t* cb;
    int fmt;
    int twice;
    int k, n;
};

__device__ __forceinline__ int fmt_of(const PfProj& p) { return p.fmt & 15; }
__device__ __forceinline__ bool is_fast(const PfProj& p) {
    return (p.fmt & 15) == kFmtExl3 && p.twice == kFastTwice && (p.fmt >> 4) == kCbMul1;
}
// Mul1 at an integer width of 2 to 8 bits, 4 included: what pf_gemmk runs (a MixedK layer's experts and dense projections).
__device__ __forceinline__ bool is_fastk(const PfProj& p) {
    return (p.fmt & 15) == kFmtExl3 && (p.fmt >> 4) == kCbMul1 && (p.twice & 1) == 0 && p.twice >= 4 && p.twice <= 16;
}
// The fast entries of a launch: 4-bit Mul1 (KM false: pf_gemm4's, the published pack's tables) or any width (KM true: pf_gemmk's).
template <bool KM>
__device__ __forceinline__ bool fast_of(const PfProj& p) {
    if constexpr (KM) return is_fastk(p);
    else return is_fast(p);
}
// A dense weight [N, K] the bf16 chain reads: bf16, or fp16 (widened exactly in the chunk).
__device__ __forceinline__ bool is_plain(int fmt) { return fmt == kFmtBf16 || fmt == kFmtF16; }

// ---- copies of nemotron_moe.cu's device helpers (that file keeps them in an anonymous namespace; the bits are gated against it) ----

// H_128 (unnormalised) of 128 values, 4 to a lane (values 4 * lane .. + 3). Stage order h = 1, 2, 4, .., 64, a + b to the lower index
// and a - b (lower minus upper) to the upper one.
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

// The 16-bit window that starts d bits below the top of w[0] in the big-endian stream w[0..4].
__device__ __forceinline__ uint32_t win16(const uint32_t (&w)[5], int d) {
    const int wi = d >> 5, off = d & 31;
    return off <= 16 ? ((w[wi] >> (16 - off)) & 0xffffu) : (__funnelshift_l(w[wi + 1], w[wi], off) >> 16);
}

// The fp16 codebook values of two 16-bit states under Mul1, as floats.
__device__ __forceinline__ void mul1_pair(uint32_t s0, uint32_t s1, float& w0, float& w1) {
    const uint32_t h0 = __dp4a(s0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const uint32_t h1 = __dp4a(s1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
    const uint32_t hh = __byte_perm(h0, h1, 0x5410);
    const half2 r = __hfma2(*reinterpret_cast<const half2*>(&hh), __half2half2(__ushort_as_half(0x1eee)),
                            __half2half2(__ushort_as_half(0xc931)));
    const float2 f = __half22float2(r);
    w0 = f.x;
    w1 = f.y;
}

// The weights of one half tile (k rows 8 H .. 8 H + 7) of columns a (c0) and a + 8 (c1).
template <int H>
__device__ __forceinline__ void decode_half(const uint32_t (&w)[5], float (&c0)[8], float (&c1)[8]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int d = 32 * q + 8 * H + 20;
        mul1_pair(win16(w, d), win16(w, d + 4), c0[2 * q], c0[2 * q + 1]);
        mul1_pair(win16(w, d + 16), win16(w, d + 20), c1[2 * q], c1[2 * q + 1]);
    }
}

// acc += x . w over eight k rows for RB rows, each row's chain in increasing k (xt: the half tile's x of row 0; rows are 64 apart).
template <int RB>
__device__ __forceinline__ void fma_half(float (&a0)[RB], float (&a1)[RB], const float* xt, const float (&c0)[8], const float (&c1)[8]) {
#pragma unroll
    for (int r = 0; r < RB; ++r) {
        const float4 xa = *reinterpret_cast<const float4*>(xt + r * (kXChunk * 16));
        const float4 xb = *reinterpret_cast<const float4*>(xt + r * (kXChunk * 16) + 4);
        const float x[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            a0[r] = fmaf(x[k], c0[k], a0[r]);
            a1[r] = fmaf(x[k], c1[k], a1[r]);
        }
    }
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

// The same eight fused multiply-adds on fp16 weights (each widened exactly): the chain of an F16 projection.
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

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

struct GemmArgs {
    const float* src;    // rotated input rows: row id is src + id * ksrc (K floats)
    int ksrc;
    const int* lists;    // lists[g * lcap + i]: the pair ids of expert g, ascending; null: a dense projection (ids are row numbers)
    int lcap;
    const int* work;     // work items (g, first, rows) x n; null: dense, z is a chunk of kMaxRb rows
    const int* nwork;    // the item count (device)
    int dense_rows;
    const PfProj* tab;   // entry g
    const PfProj* ntab;  // the next projection's table (entry g): its input is the rotated output; null: plain output
    int K, N, S;
    float* out;          // out[id * N + n]
    int relu2;
    const float* add;    // [rows, N] or null
};

// One work item's rows through the whole K of column block nb: the S slice chains (from zero) added in slice order (running sums in
// registers), then written to vs. xs holds the x chunks (2 buffers of RBMAX rows of kXChunk * 16 floats) and, once the loop is done,
// the sums vs[RBMAX][128]: the same bytes, so a block needs the larger of the two and not both (shared memory sets the blocks an SM
// holds, and the warps that hide each other's latency).
template <int RB, int RBMAX>
__device__ __forceinline__ void pf_run(const GemmArgs& a, const PfProj& p, int count, int nb, uint4 (*ring)[kThreads], float* xs,
                                       const int* rid_s) {
    const int tid = threadIdx.x, lane = tid & 31, aa = tid & 7, nt = tid >> 3;
    const int KT = a.K / 16, NCH = KT / kXChunk, kts = KT / a.S;
#ifdef PF_KSTRIDE0
    const size_t kstride = 0;
#else
    const size_t kstride = (size_t)(a.N / 16) * 32;  // words between the k tiles of one column
#endif
    const uint32_t* base = static_cast<const uint32_t*>(p.w) + ((size_t)nb * 8 + nt) * 32 + 4 * aa;
    float acc0[RB], acc1[RB], sum0[RB], sum1[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc0[r] = acc1[r] = sum0[r] = sum1[r] = 0.f;

    // The rows of chunk c (k = 16 kXChunk c ..) of the rotated input into x buffer `buf` (every thread takes a share).
    auto stage_x = [&](int c, int buf) {
        for (int i = tid; i < count * (kXChunk * 4); i += kThreads) {
            const int r = i / (kXChunk * 4), q = i % (kXChunk * 4);
            cp_async16(xs + ((size_t)buf * RBMAX + r) * (kXChunk * 16) + 4 * q,
                       a.src + (size_t)rid_s[r] * a.ksrc + c * (kXChunk * 16) + 4 * q);
        }
    };
    // prologue: tile j is group j; x chunk 0 rides with group 0
    stage_x(0, 0);
#pragma unroll
    for (int j = 0; j < kDepth; ++j) {
        if (j < KT) cp_async16(&ring[j][tid], base + (size_t)j * kstride);
        cp_commit();
    }
    int s = 0, kt_in = 0;
    for (int kt = 0; kt < KT; ++kt) {
        cp_wait<kDepth - 1>();  // tile kt, and at a chunk start the chunk's rows, have landed for this thread
        const int c = kt / kXChunk;
        if (kt % kXChunk == 0) {
            __syncthreads();  // everyone's rows of chunk c are visible, and chunk c - 1 is no longer read
            if (c + 1 < NCH) stage_x(c + 1, (c + 1) & 1);
        }
        const uint4 cur = ring[kt % (kDepth + 1)][tid];
        if (kt + kDepth < KT) cp_async16(&ring[(kt + kDepth) % (kDepth + 1)][tid], base + (size_t)(kt + kDepth) * kstride);
        cp_commit();
        // words 4a - 1 .. 4a + 3 of the tile: the one before this thread's 16 bytes is its neighbour's last word (the stream is circular)
        const uint32_t prev = __shfl_sync(0xffffffffu, cur.w, (lane & ~7) | ((aa + 7) & 7));
        const uint32_t w[5] = {prev, cur.x, cur.y, cur.z, cur.w};
        const float* xt = xs + (size_t)(c & 1) * RBMAX * (kXChunk * 16) + (kt % kXChunk) * 16;
        {
            float c0[8], c1[8];
            decode_half<0>(w, c0, c1);
#ifdef PF_NOFMA
            acc0[0] += c0[0] + c0[1] + c0[2] + c0[3] + c0[4] + c0[5] + c0[6] + c0[7] + xt[0];
            acc1[0] += c1[0] + c1[1] + c1[2] + c1[3] + c1[4] + c1[5] + c1[6] + c1[7];
#else
            fma_half<RB>(acc0, acc1, xt, c0, c1);
#endif
        }
        {
            float c0[8], c1[8];
            decode_half<1>(w, c0, c1);
#ifdef PF_NOFMA
            acc0[0] += c0[0] + c0[1] + c0[2] + c0[3] + c0[4] + c0[5] + c0[6] + c0[7] + xt[8];
            acc1[0] += c1[0] + c1[1] + c1[2] + c1[3] + c1[4] + c1[5] + c1[6] + c1[7];
#else
            fma_half<RB>(acc0, acc1, xt + 8, c0, c1);
#endif
        }
        if (++kt_in == kts) {
            // the end of slice s: its chain is complete. Slice 0 is the sum's first term; the others are added in slice order, as the
            // decode path's output side does (v = p0; v = v + p1; ...).
            kt_in = 0;
            if (s == 0) {
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    sum0[r] = acc0[r];
                    sum1[r] = acc1[r];
                }
            } else {
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    sum0[r] = sum0[r] + acc0[r];
                    sum1[r] = sum1[r] + acc1[r];
                }
            }
#pragma unroll
            for (int r = 0; r < RB; ++r) acc0[r] = acc1[r] = 0.f;
            ++s;
        }
    }
    // The sums to vs (the bytes of the x buffers): every thread is done with the last chunk first.
    __syncthreads();
    float* col = xs + nt * 16 + aa;
#pragma unroll
    for (int r = 0; r < RB; ++r) {
        col[r * 128] = sum0[r];
        col[r * 128 + 8] = sum1[r];
    }
}

// Block (nb, 0, z): column block nb (128 columns) of work item z (routed) or row chunk z (dense), 64 threads. RBMAX: the rows a block of
// this instantiation carries (16: the small class, 32: the large one and the dense projections).
template <int RBMAX>
__global__ void __launch_bounds__(kThreads, RBMAX == 16 ? kMinBlocksSmall : kMinBlocksLarge) pf_gemm4(GemmArgs a) {
    __shared__ __align__(16) uint4 ring[kDepth + 1][kThreads];
    // the x chunks (2 buffers of RBMAX rows of kXChunk * 16 floats), then the slice sums [RBMAX][128]: the larger of the two
    constexpr int kXsFloats = (2 * kXChunk * 16 > 128 ? 2 * kXChunk * 16 : 128) * RBMAX;
    __shared__ __align__(16) float xs[kXsFloats];
    float (*vs)[128] = reinterpret_cast<float (*)[128]>(xs);
    __shared__ int rid_s[RBMAX];
    int g = 0, first = 0, count = 0;
    if (a.work) {
        if ((int)blockIdx.z >= *a.nwork) return;  // uniform per block
        const int* w = a.work + 3 * blockIdx.z;
        g = w[0];
        first = w[1];
        count = w[2];
    } else {
        first = (int)blockIdx.z * RBMAX;
        count = min(RBMAX, a.dense_rows - first);
    }
    if (count <= 0) return;
    const PfProj p = a.tab[g];
    if (!is_fast(p)) return;  // a slow expert: pf_slow's
    const int nb = blockIdx.x, tid = threadIdx.x;
    if (tid < count) rid_s[tid] = a.lists ? a.lists[(size_t)g * a.lcap + first + tid] : first + tid;
    __syncthreads();
    if (count <= 4) {
        pf_run<4, RBMAX>(a, p, count, nb, ring, xs, rid_s);
    } else if (count <= 8) {
        pf_run<8, RBMAX>(a, p, count, nb, ring, xs, rid_s);
    } else if (count <= 12) {
        pf_run<12, RBMAX>(a, p, count, nb, ring, xs, rid_s);
    } else if (count <= 16 || RBMAX == 16) {
        pf_run<16, RBMAX>(a, p, count, nb, ring, xs, rid_s);
    } else if constexpr (RBMAX > 16) {
        if (count <= 20) pf_run<20, RBMAX>(a, p, count, nb, ring, xs, rid_s);
        else if (count <= 24) pf_run<24, RBMAX>(a, p, count, nb, ring, xs, rid_s);
        else if (count <= 28) pf_run<28, RBMAX>(a, p, count, nb, ring, xs, rid_s);
        else pf_run<RBMAX, RBMAX>(a, p, count, nb, ring, xs, rid_s);
    }
    __syncthreads();  // every thread's slice sums are in vs

    // The output side of nm_t2 for the item's rows and the 128 columns: H_128, * svh, + bias, then relu2 and + add; then, when the next
    // projection is fast, its input side (x * suh, H_128, * 1/sqrt 128) so that the next GEMM reads rotated rows.
    const int lane = tid & 31, warp = tid >> 5;
    const int n = nb * 128 + 4 * lane;
    const float4 sv = __ldg(reinterpret_cast<const float4*>(p.svh + n));
    const bool has_bias = p.bias != nullptr;
    const float4 bv = has_bias ? __ldg(reinterpret_cast<const float4*>(p.bias + n)) : make_float4(0.f, 0.f, 0.f, 0.f);
    bool rot_out = false;
    float4 nsv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (a.ntab) {
        const PfProj np = a.ntab[g];
        if (is_fast(np)) {
            rot_out = true;
            nsv = __ldg(reinterpret_cast<const float4*>(np.suh + n));
        }
    }
    // Two rows of the warp at a time: their butterflies are independent, so the shuffles of one fill the latency of the other.
    constexpr int kW = kThreads / 32;
    for (int r0 = warp; r0 < count; r0 += 2 * kW) {
        const int r1 = r0 + kW;
        const bool two = r1 < count;
        const int id0 = rid_s[r0], id1 = two ? rid_s[r1] : id0;
        const float4 v0 = *reinterpret_cast<const float4*>(&vs[r0][4 * lane]);
        const float4 v1 = *reinterpret_cast<const float4*>(&vs[two ? r1 : r0][4 * lane]);
        float t0[4] = {v0.x, v0.y, v0.z, v0.w};
        float t1[4] = {v1.x, v1.y, v1.z, v1.w};
        fwht128_warp(t0, lane);
        fwht128_warp(t1, lane);
        const float sc[4] = {sv.x, sv.y, sv.z, sv.w}, bc[4] = {bv.x, bv.y, bv.z, bv.w};
        float4 ad0 = make_float4(0.f, 0.f, 0.f, 0.f), ad1 = ad0;
        if (a.add) {
            ad0 = *reinterpret_cast<const float4*>(a.add + (size_t)id0 * a.N + n);
            ad1 = *reinterpret_cast<const float4*>(a.add + (size_t)id1 * a.N + n);
        }
        const float av0[4] = {ad0.x, ad0.y, ad0.z, ad0.w}, av1[4] = {ad1.x, ad1.y, ad1.z, ad1.w};
        float o0[4], o1[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float x = (t0[j] * kInvSqrt128) * sc[j];
            float y = (t1[j] * kInvSqrt128) * sc[j];
            if (has_bias) {
                x = x + bc[j];
                y = y + bc[j];
            }
            if (a.relu2) {
                const float qx = fmaxf(x, 0.f), qy = fmaxf(y, 0.f);
                x = qx * qx;
                y = qy * qy;
            }
            if (a.add) {
                x = x + av0[j];
                y = y + av1[j];
            }
            o0[j] = x;
            o1[j] = y;
        }
        if (rot_out) {
            const float ns[4] = {nsv.x, nsv.y, nsv.z, nsv.w};
            float u0[4] = {o0[0] * ns[0], o0[1] * ns[1], o0[2] * ns[2], o0[3] * ns[3]};
            float u1[4] = {o1[0] * ns[0], o1[1] * ns[1], o1[2] * ns[2], o1[3] * ns[3]};
            fwht128_warp(u0, lane);
            fwht128_warp(u1, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o0[j] = u0[j] * kInvSqrt128;
                o1[j] = u1[j] * kInvSqrt128;
            }
        }
        *reinterpret_cast<float4*>(a.out + (size_t)id0 * a.N + n) = make_float4(o0[0], o0[1], o0[2], o0[3]);
        if (two) *reinterpret_cast<float4*>(a.out + (size_t)id1 * a.N + n) = make_float4(o1[0], o1[1], o1[2], o1[3]);
    }
}

// ---- any integer width: pf_gemmk (a MixedK layer) -----------------------------------------------------------------------------
//
// pf_gemm4's schedule and arithmetic for Mul1 projections of KB = 2 .. 8 bits (4 included), the width read per item's entry, so a layer
// whose experts are stored at different widths (up and down of one expert often apart) runs in one launch a projection. A thread's
// stream positions are nemotron_moe.cu's nm2_gemmk's (positions 32 a .. 32 a + 31 of its column tile: its KB words and the 16 - KB bits
// before them, the neighbour's last word by shuffle); every chain, slice sum and output step is pf_gemm4's, so a row's bits are the
// decode path's at any width. Items of at most 16 rows (the 16-row class's registers; schedule 2's items), any count of them.

// The 16-bit window that starts d bits below the top of w[0] in the big-endian stream w[0 .. NW - 1] (nm2_gemmk's).
template <int NW>
__device__ __forceinline__ uint32_t win16k(const uint32_t (&w)[NW], int d) {
    const int wi = d >> 5, off = d & 31;
    return off <= 16 ? ((w[wi] >> (16 - off)) & 0xffffu) : (__funnelshift_l(w[wi + 1 < NW ? wi + 1 : wi], w[wi], off) >> 16);
}

// The weights of one half tile at KB bits (nm2_gemmk's decode_half_k).
template <int KB, int H>
__device__ __forceinline__ void decode_half_k(const uint32_t (&w)[KB + 1], float (&c0)[8], float (&c1)[8]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int d = KB * (8 * q + 2 * H + 1) + 16;
        mul1_pair(win16k(w, d), win16k(w, d + KB), c0[2 * q], c0[2 * q + 1]);
        mul1_pair(win16k(w, d + 4 * KB), win16k(w, d + 5 * KB), c1[2 * q], c1[2 * q + 1]);
    }
}

__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s), "l"(gmem) : "memory");
}
// A thread's KB words of a tile into shared memory (16-byte copies at 4 and 8 bits, 8-byte at 2 and 6, 4-byte otherwise: the words'
// alignment at word offset KB a).
template <int KB>
__device__ __forceinline__ void cp_words(uint32_t* smem, const uint32_t* g) {
    if constexpr (KB % 4 == 0) {
#pragma unroll
        for (int i = 0; i < KB / 4; ++i) cp_async16(smem + 4 * i, g + 4 * i);
    } else if constexpr (KB % 2 == 0) {
#pragma unroll
        for (int i = 0; i < KB / 2; ++i) cp_async8(smem + 2 * i, g + 2 * i);
    } else {
#pragma unroll
        for (int i = 0; i < KB; ++i) cp_async4(smem + i, g + i);
    }
}

// The same words back from shared memory, with the copies' vector widths (a thread's words are KB apart from its neighbour's: scalar
// reads at an even stride would conflict on the banks).
template <int KB>
__device__ __forceinline__ void lds_words(const uint32_t* smem, uint32_t (&w)[KB]) {
    if constexpr (KB % 4 == 0) {
#pragma unroll
        for (int i = 0; i < KB / 4; ++i) {
            const uint4 v = reinterpret_cast<const uint4*>(smem)[i];
            w[4 * i] = v.x;
            w[4 * i + 1] = v.y;
            w[4 * i + 2] = v.z;
            w[4 * i + 3] = v.w;
        }
    } else if constexpr (KB % 2 == 0) {
#pragma unroll
        for (int i = 0; i < KB / 2; ++i) {
            const uint2 v = reinterpret_cast<const uint2*>(smem)[i];
            w[2 * i] = v.x;
            w[2 * i + 1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < KB; ++i) w[i] = smem[i];
    }
}

constexpr int kRingWords = (kDepth + 1) * kThreads * 8;  // the ring at the widest width (8 bits: 8 words a thread a tile)

// pf_run at KB bits: one item's rows through the whole K of column block nb, the slice sums to vs (the bytes of xs).
template <int RB, int RBMAX, int KB>
__device__ __forceinline__ void pf_run_k(const GemmArgs& a, const PfProj& p, int count, int nb, uint32_t* ringw, float* xs,
                                         const int* rid_s) {
    constexpr int TW = 8 * KB;  // words of a tile
    const int tid = threadIdx.x, lane = tid & 31, aa = tid & 7, nt = tid >> 3;
    const int KT = a.K / 16, NCH = KT / kXChunk, kts = KT / a.S;
    const size_t kstride = (size_t)(a.N / 16) * TW;  // words between the k tiles of one column
    const uint32_t* base = static_cast<const uint32_t*>(p.w) + ((size_t)nb * 8 + nt) * TW + KB * aa;
    auto slot = [&](int j) { return ringw + ((size_t)(j % (kDepth + 1)) * kThreads + tid) * KB; };
    float acc0[RB], acc1[RB], sum0[RB], sum1[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc0[r] = acc1[r] = sum0[r] = sum1[r] = 0.f;

    auto stage_x = [&](int c, int buf) {
        for (int i = tid; i < count * (kXChunk * 4); i += kThreads) {
            const int r = i / (kXChunk * 4), q = i % (kXChunk * 4);
            cp_async16(xs + ((size_t)buf * RBMAX + r) * (kXChunk * 16) + 4 * q,
                       a.src + (size_t)rid_s[r] * a.ksrc + c * (kXChunk * 16) + 4 * q);
        }
    };
    // prologue: tile j is group j; x chunk 0 rides with group 0
    stage_x(0, 0);
#pragma unroll
    for (int j = 0; j < kDepth; ++j) {
        if (j < KT) cp_words<KB>(slot(j), base + (size_t)j * kstride);
        cp_commit();
    }
    int s = 0, kt_in = 0;
    for (int kt = 0; kt < KT; ++kt) {
        cp_wait<kDepth - 1>();  // tile kt, and at a chunk start the chunk's rows, have landed for this thread
        const int c = kt / kXChunk;
        if (kt % kXChunk == 0) {
            __syncthreads();  // everyone's rows of chunk c are visible, and chunk c - 1 is no longer read
            if (c + 1 < NCH) stage_x(c + 1, (c + 1) & 1);
        }
        uint32_t cur[KB];
        lds_words<KB>(slot(kt), cur);
        if (kt + kDepth < KT) cp_words<KB>(slot(kt + kDepth), base + (size_t)(kt + kDepth) * kstride);
        cp_commit();
        uint32_t w[KB + 1];
        w[0] = __shfl_sync(0xffffffffu, cur[KB - 1], (lane & ~7) | ((aa + 7) & 7));
#pragma unroll
        for (int i = 0; i < KB; ++i) w[i + 1] = cur[i];
        const float* xt = xs + (size_t)(c & 1) * RBMAX * (kXChunk * 16) + (kt % kXChunk) * 16;
        {
            float c0[8], c1[8];
            decode_half_k<KB, 0>(w, c0, c1);
            fma_half<RB>(acc0, acc1, xt, c0, c1);
        }
        {
            float c0[8], c1[8];
            decode_half_k<KB, 1>(w, c0, c1);
            fma_half<RB>(acc0, acc1, xt + 8, c0, c1);
        }
        if (++kt_in == kts) {
            // the end of slice s: pf_run's order (slice 0 the sum's first term, the others added in slice order)
            kt_in = 0;
            if (s == 0) {
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    sum0[r] = acc0[r];
                    sum1[r] = acc1[r];
                }
            } else {
#pragma unroll
                for (int r = 0; r < RB; ++r) {
                    sum0[r] = sum0[r] + acc0[r];
                    sum1[r] = sum1[r] + acc1[r];
                }
            }
#pragma unroll
            for (int r = 0; r < RB; ++r) acc0[r] = acc1[r] = 0.f;
            ++s;
        }
    }
    __syncthreads();
    float* col = xs + nt * 16 + aa;
#pragma unroll
    for (int r = 0; r < RB; ++r) {
        col[r * 128] = sum0[r];
        col[r * 128 + 8] = sum1[r];
    }
}

template <int RB, int RBMAX>
__device__ __forceinline__ void pf_run_any(int twice, const GemmArgs& a, const PfProj& p, int count, int nb, uint32_t* ringw, float* xs,
                                           const int* rid_s) {
    switch (twice) {
        case 4: pf_run_k<RB, RBMAX, 2>(a, p, count, nb, ringw, xs, rid_s); break;
        case 6: pf_run_k<RB, RBMAX, 3>(a, p, count, nb, ringw, xs, rid_s); break;
        case 8: pf_run_k<RB, RBMAX, 4>(a, p, count, nb, ringw, xs, rid_s); break;
        case 10: pf_run_k<RB, RBMAX, 5>(a, p, count, nb, ringw, xs, rid_s); break;
        case 12: pf_run_k<RB, RBMAX, 6>(a, p, count, nb, ringw, xs, rid_s); break;
        case 14: pf_run_k<RB, RBMAX, 7>(a, p, count, nb, ringw, xs, rid_s); break;
        default: pf_run_k<RB, RBMAX, 8>(a, p, count, nb, ringw, xs, rid_s); break;
    }
}

// Block (nb, 0, z): pf_gemm4<16> for the entries is_fastk takes (work items of at most 16 rows, or dense chunks of 16), the width per
// item's entry; the output side, and the next projection's input side when that one is any-width fast, as pf_gemm4's.
__global__ void __launch_bounds__(kThreads, kMinBlocksSmall) pf_gemmk(GemmArgs a) {
    constexpr int RBMAX = kClassRb;
    __shared__ __align__(16) uint32_t ringw[kRingWords];
    constexpr int kXsFloats = (2 * kXChunk * 16 > 128 ? 2 * kXChunk * 16 : 128) * RBMAX;
    __shared__ __align__(16) float xs[kXsFloats];
    float (*vs)[128] = reinterpret_cast<float (*)[128]>(xs);
    __shared__ int rid_s[RBMAX];
    int g = 0, first = 0, count = 0;
    if (a.work) {
        if ((int)blockIdx.z >= *a.nwork) return;  // uniform per block
        const int* w = a.work + 3 * blockIdx.z;
        g = w[0];
        first = w[1];
        count = w[2];
        if (count > RBMAX) __trap();  // the caller's items are of at most 16 rows (schedule 2); never drop rows silently
    } else {
        first = (int)blockIdx.z * RBMAX;
        count = min(RBMAX, a.dense_rows - first);
    }
    if (count <= 0) return;
    const PfProj p = a.tab[g];
    if (!is_fastk(p)) return;  // a slow expert: pf_slow's
    const int nb = blockIdx.x, tid = threadIdx.x;
    if (tid < count) rid_s[tid] = a.lists ? a.lists[(size_t)g * a.lcap + first + tid] : first + tid;
    __syncthreads();
    if (count <= 4) {
        pf_run_any<4, RBMAX>(p.twice, a, p, count, nb, ringw, xs, rid_s);
    } else if (count <= 8) {
        pf_run_any<8, RBMAX>(p.twice, a, p, count, nb, ringw, xs, rid_s);
    } else if (count <= 12) {
        pf_run_any<12, RBMAX>(p.twice, a, p, count, nb, ringw, xs, rid_s);
    } else {
        pf_run_any<16, RBMAX>(p.twice, a, p, count, nb, ringw, xs, rid_s);
    }
    __syncthreads();  // every thread's slice sums are in vs

    // pf_gemm4's output side, and the next projection's input side when it is any-width fast
    const int lane = tid & 31, warp = tid >> 5;
    const int n = nb * 128 + 4 * lane;
    const float4 sv = __ldg(reinterpret_cast<const float4*>(p.svh + n));
    const bool has_bias = p.bias != nullptr;
    const float4 bv = has_bias ? __ldg(reinterpret_cast<const float4*>(p.bias + n)) : make_float4(0.f, 0.f, 0.f, 0.f);
    bool rot_out = false;
    float4 nsv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (a.ntab) {
        const PfProj np = a.ntab[g];
        if (is_fastk(np)) {
            rot_out = true;
            nsv = __ldg(reinterpret_cast<const float4*>(np.suh + n));
        }
    }
    constexpr int kW = kThreads / 32;
    for (int r0 = warp; r0 < count; r0 += 2 * kW) {
        const int r1 = r0 + kW;
        const bool two = r1 < count;
        const int id0 = rid_s[r0], id1 = two ? rid_s[r1] : id0;
        const float4 v0 = *reinterpret_cast<const float4*>(&vs[r0][4 * lane]);
        const float4 v1 = *reinterpret_cast<const float4*>(&vs[two ? r1 : r0][4 * lane]);
        float t0[4] = {v0.x, v0.y, v0.z, v0.w};
        float t1[4] = {v1.x, v1.y, v1.z, v1.w};
        fwht128_warp(t0, lane);
        fwht128_warp(t1, lane);
        const float sc[4] = {sv.x, sv.y, sv.z, sv.w}, bc[4] = {bv.x, bv.y, bv.z, bv.w};
        float4 ad0 = make_float4(0.f, 0.f, 0.f, 0.f), ad1 = ad0;
        if (a.add) {
            ad0 = *reinterpret_cast<const float4*>(a.add + (size_t)id0 * a.N + n);
            ad1 = *reinterpret_cast<const float4*>(a.add + (size_t)id1 * a.N + n);
        }
        const float av0[4] = {ad0.x, ad0.y, ad0.z, ad0.w}, av1[4] = {ad1.x, ad1.y, ad1.z, ad1.w};
        float o0[4], o1[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float x = (t0[j] * kInvSqrt128) * sc[j];
            float y = (t1[j] * kInvSqrt128) * sc[j];
            if (has_bias) {
                x = x + bc[j];
                y = y + bc[j];
            }
            if (a.relu2) {
                const float qx = fmaxf(x, 0.f), qy = fmaxf(y, 0.f);
                x = qx * qx;
                y = qy * qy;
            }
            if (a.add) {
                x = x + av0[j];
                y = y + av1[j];
            }
            o0[j] = x;
            o1[j] = y;
        }
        if (rot_out) {
            const float ns[4] = {nsv.x, nsv.y, nsv.z, nsv.w};
            float u0[4] = {o0[0] * ns[0], o0[1] * ns[1], o0[2] * ns[2], o0[3] * ns[3]};
            float u1[4] = {o1[0] * ns[0], o1[1] * ns[1], o1[2] * ns[2], o1[3] * ns[3]};
            fwht128_warp(u0, lane);
            fwht128_warp(u1, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o0[j] = u0[j] * kInvSqrt128;
                o1[j] = u1[j] * kInvSqrt128;
            }
        }
        *reinterpret_cast<float4*>(a.out + (size_t)id0 * a.N + n) = make_float4(o0[0], o0[1], o0[2], o0[3]);
        if (two) *reinterpret_cast<float4*>(a.out + (size_t)id1 * a.N + n) = make_float4(o1[0], o1[1], o1[2], o1[3]);
    }
}

// The input side of a projection for `rows` rows: out[row, :] = H_128(src[row / div, :] * suh) * (1 / sqrt 128) per 128-block, with
// suh of table entry pick[row] (entry 0 when pick is null). Rows whose entry is not fast are left alone (pf_slow reads them plain).
// One warp a (row, 128-block); a block of 8 warps. `out` may be `src` (ksrc == K, div == 1): a warp reads its 128 values before it
// writes them, and no other warp touches them.
// KM: the fast entries are any-width (a MixedK table, pf_gemmk's) instead of 4-bit Mul1 (pf_gemm4's).
template <bool KM>
__global__ void __launch_bounds__(256) pf_rotate(const float* src, int ksrc, int div, const int* __restrict__ pick,
                                                 const PfProj* __restrict__ tab, int rows, int K, float* out) {
    const int lane = threadIdx.x & 31;
    const int nblk = K / 128;
    const long u = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (u >= (long)rows * nblk) return;  // a whole warp returns
    const int row = (int)(u / nblk), hb = (int)(u % nblk);
    const PfProj p = tab[pick ? pick[row] : 0];
    if (!fast_of<KM>(p)) return;
    const float4 xv = __ldg(reinterpret_cast<const float4*>(src + (size_t)(row / div) * ksrc + hb * 128 + 4 * lane));
    const float4 sv = __ldg(reinterpret_cast<const float4*>(p.suh + hb * 128 + 4 * lane));
    float v[4] = {xv.x * sv.x, xv.y * sv.y, xv.z * sv.z, xv.w * sv.w};
    fwht128_warp(v, lane);
    *reinterpret_cast<float4*>(out + (size_t)row * K + hb * 128 + 4 * lane) =
        make_float4(v[0] * kInvSqrt128, v[1] * kInvSqrt128, v[2] * kInvSqrt128, v[3] * kInvSqrt128);
}

// ---- bf16 projections: a bf16 expert's up or down, and a bf16 dense projection ----------------------------------------------------
//
// Two kernels with the same arithmetic. pf_slow is the first one (a warp a column, rows 4 at a time: gated bit for bit against the decode
// path); pf_slow4 is the four-column kernel below it (gated in the harness of the package, see docs/NEMOTRON_MOE_PREFILL.md).
//
// pf_slow: a bf16 expert's projection (the chain of nm_gemv_bf16: lane l owns the 8-element chunks l + 32 q of the weight row, in
// increasing q, then the xor tree), or an absent expert's zeros, for the rows of a work item. Block (nbw, z): 8 warps, warp w is
// column nbw * 8 + w. Rows are taken 4 at a time so that a weight chunk is loaded once for 4 rows. The output is the plain activation
// (no rotation: the next projection of a slow expert is slow as well). With `lists` null this is a dense bf16 projection (table entry
// 0): item z is the rows 32 z .. 32 z + 31 of `dense_rows`, row ids are row numbers, and `add` (when given) is added after relu2, as
// nm_t2 does.
__global__ void __launch_bounds__(256) pf_slow(const float* __restrict__ src, int ksrc, int div, const int* __restrict__ lists, int lcap,
                                               const int* __restrict__ work, const int* __restrict__ nwork, int dense_rows,
                                               const PfProj* __restrict__ tab, int K, int N, float* __restrict__ out, int relu2,
                                               const float* __restrict__ add) {
    int g = 0, first = 0, count = 0;
    if (lists) {
        if ((int)blockIdx.y >= *nwork) return;  // uniform per block
        const int* w = work + 3 * blockIdx.y;
        g = w[0];
        first = w[1];
        count = w[2];
    } else {
        first = (int)blockIdx.y * kMaxRb;
        if (first >= dense_rows) return;  // uniform per block
        count = min(kMaxRb, dense_rows - first);
    }
    const PfProj p = tab[g];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;  // a whole warp returns
    const int fmt = fmt_of(p);
    const bool h16 = fmt == kFmtF16;
    for (int r0 = 0; r0 < count; r0 += 4) {
        const int here = min(4, count - r0);
        int id[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) id[i] = lists ? lists[(size_t)g * lcap + first + r0 + (i < here ? i : 0)] : first + r0 + (i < here ? i : 0);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if (is_plain(fmt)) {
            const uint16_t* wrow = static_cast<const uint16_t*>(p.w) + (size_t)n * K;
            for (int q = lane; q < K / 8; q += 32) {
                const uint4 wv = __ldg(reinterpret_cast<const uint4*>(wrow + 8 * q));
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (i < here) {
                        const float* xr = src + (size_t)(id[i] / div) * ksrc + 8 * q;
                        const float4 xa = __ldg(reinterpret_cast<const float4*>(xr)), xb = __ldg(reinterpret_cast<const float4*>(xr + 4));
                        acc[i] = h16 ? chunk_fma_h(acc[i], xa, xb, wv) : chunk_fma(acc[i], xa, xb, wv);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], off);
        }
        if (lane == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < here) {
                    float v = is_plain(fmt) ? acc[i] : 0.f;  // absent: zeros
                    if (relu2) {
                        const float q = fmaxf(v, 0.f);
                        v = q * q;
                    }
                    if (add) v = v + add[(size_t)id[i] * N + n];
                    out[(size_t)id[i] * N + n] = v;
                }
            }
        }
    }
}

// pf_slow4. Per (row, column) the same chain as pf_slow's (lane l of a warp owns the 8-element chunks l, l + 32, l + 64, .. of the
// weight row, accumulates them in increasing order, eight fmaf in element order a chunk, from zero; the 32 lane sums are added by the
// xor tree, offsets 16, 8, 4, 2, 1, acc = acc + partner). Here a warp takes 4 columns and up to 8 rows a pass: the weights of the 4
// columns are loaded once for the 8 rows, the x chunk of a row once for the 4 columns, and the 32 xor trees of the pass are done together as a
// reduce-scatter (each step a lane keeps half of its values and adds the partner's copy of them: the same pairs added in the same
// order, so the same bits, in 31 shuffles instead of 160).

// One reduce-scatter step over the first M of the lane's values: the lane keeps the half its `o` bit selects and adds the partner's copy.
template <int NV, int M>
__device__ __forceinline__ void rs_step(float (&v)[NV], int o, int lane) {
    const bool up = (lane & o) != 0;
#pragma unroll
    for (int j = 0; j < M / 2; ++j) {
        const float lo = v[j], hi = v[j + M / 2];
        const float send = up ? lo : hi, keep = up ? hi : lo;
        v[j] = keep + __shfl_xor_sync(0xffffffffu, send, o);
    }
}

// One pass: columns n0 .. n0 + 3, rows id[0 .. here - 1] (here <= RB, RB = 4 or 8). `w` is the weight row of column n0 (null: absent
// expert, zeros). Value index c * RB + r of a lane ends, after the reduce-scatter, on lane index (RB = 8) or lane >> 1 (RB = 4).
// H16: the weights are fp16 (widened exactly), not bf16.
template <int RB, bool H16>
__device__ __forceinline__ void bf16_pass(const float* __restrict__ src, int ksrc, int div, const int (&id)[8], int here,
                                          const uint16_t* __restrict__ w, int K, int N, int n0, float* __restrict__ out, int relu2,
                                          const float* __restrict__ add, int lane) {
    constexpr int NV = 4 * RB;
    float v[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) v[i] = 0.f;
    if (w) {
        for (int q = lane; q < K / 8; q += 32) {
            float wf[4][8];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 wv = __ldg(reinterpret_cast<const uint4*>(w + (size_t)c * K + 8 * q));
                if constexpr (H16) {
                    const uint32_t u[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(*reinterpret_cast<const half2*>(&u[j]));
                        wf[c][2 * j] = f.x;
                        wf[c][2 * j + 1] = f.y;
                    }
                } else {
                    wf[c][0] = bf_lo(wv.x);
                    wf[c][1] = bf_hi(wv.x);
                    wf[c][2] = bf_lo(wv.y);
                    wf[c][3] = bf_hi(wv.y);
                    wf[c][4] = bf_lo(wv.z);
                    wf[c][5] = bf_hi(wv.z);
                    wf[c][6] = bf_lo(wv.w);
                    wf[c][7] = bf_hi(wv.w);
                }
            }
#pragma unroll
            for (int r = 0; r < RB; ++r) {
                if (r < here) {
                    const float* xr = src + (size_t)(id[r] / div) * ksrc + 8 * q;
                    const float4 xa = __ldg(reinterpret_cast<const float4*>(xr));
                    const float4 xb = __ldg(reinterpret_cast<const float4*>(xr + 4));
                    const float x[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                    for (int c = 0; c < 4; ++c)
#pragma unroll
                        for (int j = 0; j < 8; ++j) v[c * RB + r] = __fmaf_rn(x[j], wf[c][j], v[c * RB + r]);
                }
            }
        }
    }
    rs_step<NV, NV>(v, 16, lane);
    rs_step<NV, NV / 2>(v, 8, lane);
    rs_step<NV, NV / 4>(v, 4, lane);
    rs_step<NV, NV / 8>(v, 2, lane);
    float val;
    int idx;
    bool mine = true;
    if constexpr (RB == 8) {
        rs_step<NV, 2>(v, 1, lane);
        val = v[0];
        idx = lane;
    } else {
        val = v[0] + __shfl_xor_sync(0xffffffffu, v[0], 1);  // the last offset: both lanes of a pair hold the sum
        idx = lane >> 1;
        mine = (lane & 1) == 0;
    }
    const int c = idx / RB, r = idx % RB;
    int row = id[0];
#pragma unroll
    for (int i = 1; i < RB; ++i) row = r == i ? id[i] : row;
    if (mine && r < here) {
        if (relu2) {
            const float q = fmaxf(val, 0.f);
            val = q * q;
        }
        const size_t at = (size_t)row * N + n0 + c;
        if (add) val = val + add[at];
        out[at] = val;
    }
}

// The kernel: the arguments and the work items of pf_slow's. Block (nb, z): 8 warps, warp w the columns 4 (8 nb + w) .. + 3.
__global__ void __launch_bounds__(256) pf_slow4(const float* __restrict__ src, int ksrc, int div, const int* __restrict__ lists, int lcap,
                                               const int* __restrict__ work, const int* __restrict__ nwork, int dense_rows,
                                               const PfProj* __restrict__ tab, int K, int N, float* __restrict__ out, int relu2,
                                               const float* __restrict__ add) {
    int g = 0, first = 0, count = 0;
    if (lists) {
        if ((int)blockIdx.y >= *nwork) return;  // uniform per block
        const int* w = work + 3 * blockIdx.y;
        g = w[0];
        first = w[1];
        count = w[2];
    } else {
        first = (int)blockIdx.y * kMaxRb;
        if (first >= dense_rows) return;  // uniform per block
        count = min(kMaxRb, dense_rows - first);
    }
    const PfProj p = tab[g];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * 8 + warp) * 4;
    if (n0 >= N) return;  // a whole warp returns
    const uint16_t* wrow = is_plain(fmt_of(p)) ? static_cast<const uint16_t*>(p.w) + (size_t)n0 * K : nullptr;
    const bool h16 = fmt_of(p) == kFmtF16;
    for (int r0 = 0; r0 < count; r0 += 8) {
        const int here = min(8, count - r0);
        int id[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = r0 + (i < here ? i : 0);
            id[i] = lists ? lists[(size_t)g * lcap + first + k] : first + k;
        }
        if (h16) {
            if (here <= 4) bf16_pass<4, true>(src, ksrc, div, id, here, wrow, K, N, n0, out, relu2, add, lane);
            else bf16_pass<8, true>(src, ksrc, div, id, here, wrow, K, N, n0, out, relu2, add, lane);
        } else {
            if (here <= 4) bf16_pass<4, false>(src, ksrc, div, id, here, wrow, K, N, n0, out, relu2, add, lane);
            else bf16_pass<8, false>(src, ksrc, div, id, here, wrow, K, N, n0, out, relu2, add, lane);
        }
    }
}

// Selection and weights, one block of 512 a row: nm2_select without its bucket stage (the same rounds: the largest remaining comp of the
// block, key code then the inverted id; the weights (sig / ordered sum) * scale).
__global__ void __launch_bounds__(512) pf_select(const float* __restrict__ logits, const float* __restrict__ bias, float scale,
                                                 int* __restrict__ pick, float* __restrict__ wts) {
    __shared__ uint32_t wb_hi[2][16], wb_lo[2][16];
    __shared__ float sel_sig[kTopK];
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, row = blockIdx.x;
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
            hi = 0;
            lo = 0;
        }
    }
    __syncthreads();
    if (t < kTopK) {
        float denom = 0.f;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) denom = denom + sel_sig[k];
        denom = denom + 1e-20f;
        wts[(size_t)row * kTopK + t] = (sel_sig[t] / denom) * scale;
    }
}

// Block e: the ascending pair ids (row * 22 + slot) whose pick is expert e, into lists[e * cap ..], and their number into cnt[e].
__global__ void __launch_bounds__(256) pf_bucket_lists(const int* __restrict__ pick, int P, int cap, int* __restrict__ cnt,
                                                       int* __restrict__ lists) {
    __shared__ int wcnt[8];
    const int e = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
    int* out = lists + (size_t)e * cap;
    int base = 0;
    for (int p0 = 0; p0 < P; p0 += 256) {
        const int p = p0 + t;
        const bool hit = p < P && pick[p] == e;
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) wcnt[w] = __popc(m);
        __syncthreads();
        int woff = 0, tot = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            woff += i < w ? wcnt[i] : 0;
            tot += wcnt[i];
        }
        if (hit) out[base + woff + __popc(m & ((1u << lane) - 1u))] = p;
        base += tot;
        __syncthreads();
    }
    if (t == 0) cnt[e] = base;
}

// One block of 512 (thread e): the work items. An expert with c rows is ceil(c / 32) items of near-equal rows (item k of m takes list
// positions [k c / m, (k + 1) c / m)). Class 2 is the slow experts (anything but 4-bit Mul1 EXL3 up and down: bf16, absent). The fast
// experts are one class (mode 1: 0) ordered LARGEST AND SMALLEST ALTERNATELY (the block scheduler then hands each SM compute-bound
// blocks of the experts with many rows and memory-bound blocks of the ones with few, which overlap), or two classes (mode 0: 0 with at
// most 16 rows and 1 with more; each ordered by c, largest first, for two launches). An expert's items are consecutive. Class k's
// items go to work + k * stride (three ints each), its count to nwork[k].
// KM: the fast experts are those whose up and down are any-width fast (a MixedK table) instead of 4-bit Mul1.
template <bool KM>
__global__ void __launch_bounds__(kExperts) pf_bucket_work(const int* __restrict__ cnt, const PfProj* __restrict__ tab_up,
                                                           const PfProj* __restrict__ tab_down, int* __restrict__ work, int stride,
                                                           int* __restrict__ nwork, int mode) {
    __shared__ int s_c[kExperts], s_cls[kExperts], s_pos[kExperts], s_it[kExperts];
    const int e = threadIdx.x;
    const int c = cnt[e];
    const bool fast = fast_of<KM>(tab_up[e]) && fast_of<KM>(tab_down[e]);
    const int cls = !fast ? 2 : (mode >= 1 ? 0 : (c <= kClassRb ? 0 : 1));
    // rows of an item: 32, or 16 for the fast experts of mode 2 (a launch of the 16-row kernel, which holds more blocks an SM)
    const int cap = (mode == 2 && fast) ? kClassRb : kMaxRb;
    s_c[e] = c;
    s_cls[e] = cls;
    s_it[e] = (c + cap - 1) / cap;
    __syncthreads();
    int rank = 0, nact = 0;
    for (int j = 0; j < kExperts; ++j) {
        const int cj = s_c[j];
        const bool same = s_cls[j] == cls;
        rank += (same && (cj > c || (cj == c && j < e))) ? 1 : 0;
        nact += (same && cj > 0) ? 1 : 0;
    }
    // position in the class's item order: by rank, or (mode 1) alternately from the large end and from the small end
    int pos = rank;
    if (mode >= 1 && c > 0) pos = rank < (nact + 1) / 2 ? 2 * rank : 2 * (nact - 1 - rank) + 1;
    s_pos[e] = pos;
    __syncthreads();
    const int items = s_it[e];
    int ibase = 0, tot0 = 0, tot1 = 0, tot2 = 0;
    for (int j = 0; j < kExperts; ++j) {
        const int ij = s_it[j];
        if (s_cls[j] == cls) ibase += s_pos[j] < pos ? ij : 0;
        if (e == 0) {
            if (s_cls[j] == 0) tot0 += ij;
            else if (s_cls[j] == 1) tot1 += ij;
            else tot2 += ij;
        }
    }
    int* wk = work + (size_t)cls * stride * 3;
    for (int k = 0; k < items; ++k) {
        int* it = wk + 3 * (ibase + k);
        const int lo = (k * c) / items, hi = ((k + 1) * c) / items;
        it[0] = e;
        it[1] = lo;
        it[2] = hi - lo;
    }
    if (e == 0) {
        nwork[0] = tot0;
        nwork[1] = tot1;
        nwork[2] = tot2;
    }
}

// The output of an ABSENT expert's pairs (a TP rank's non-owned experts): exact +0.0 in out[id * N .. id * N + N) for every pair of every
// slow-class item whose expert is absent, which is what pf_slow / pf_slow4 write there (v = 0.f, no relu2, no add on the down
// projection). Grid-stride over the items the device counted (nwork), so the grid is a few waves and not n_slow * ceil(rows / 32)
// items of N / 8 blocks that mostly exit at once. An item of a bf16 expert is skipped (the caller runs pf_slow for those layers).
__global__ void __launch_bounds__(256) pf_zero_absent(const int* __restrict__ lists, int lcap, const int* __restrict__ work,
                                                      const int* __restrict__ nwork, const PfProj* __restrict__ tab, int N,
                                                      float* __restrict__ out) {
    const int items = *nwork;
    const int n4 = N / 4;
    for (int z = blockIdx.x; z < items; z += gridDim.x) {
        const int* w = work + 3 * z;
        const int g = w[0], first = w[1], count = w[2];
        if (is_plain(fmt_of(tab[g]))) continue;  // uniform per block
        for (int i = threadIdx.x; i < count * n4; i += blockDim.x) {
            const int r = i / n4, q = i % n4;
            const int id = lists[(size_t)g * lcap + first + r];
            reinterpret_cast<float4*>(out + (size_t)id * N)[q] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// Launchers

// The absent experts' pairs of the slow-class items (`work`, count at `nwork`, both device) to +0.0 in `out` ([pairs, N], N % 4 == 0).
GB_EXPORT int gb_nmp_zero_absent(const void* lists, int lcap, const void* work, const void* nwork, const void* tab, int N, void* out,
                                 int blocks, void* stream) {
    if (!lists || !work || !nwork || !tab || !out || N < 4 || N % 4 != 0 || blocks < 1) return (int)cudaErrorInvalidValue;
    pf_zero_absent<<<(unsigned)blocks, 256, 0, gb_stream(stream)>>>(static_cast<const int*>(lists), lcap, static_cast<const int*>(work),
                                                                    static_cast<const int*>(nwork), static_cast<const PfProj*>(tab), N,
                                                                    static_cast<float*>(out));
    return GB_CHECK_LAUNCH();
}

// Selection and weights of `rows` rows from logits [rows, 512].
GB_EXPORT int gb_nmp_select(const void* logits, const void* bias, float scale, void* pick, void* wts, int rows, void* stream) {
    if (rows < 1 || !logits || !bias || !pick || !wts) return (int)cudaErrorInvalidValue;
    pf_select<<<rows, 512, 0, gb_stream(stream)>>>(static_cast<const float*>(logits), static_cast<const float*>(bias), scale,
                                                  static_cast<int*>(pick), static_cast<float*>(wts));
    return GB_CHECK_LAUNCH();
}

// Per-expert pair lists (stride cap) and counts, then the work items of the three classes: `work` holds three regions of `stride` items
// of three ints (stride >= 512 + ceil(rows * 22 / 16)), `nwork` three ints. mode 1: the fast experts are one class, large and small
// alternating; mode 2: as mode 1 with items of at most 16 rows; mode 0: two classes by rows (see pf_bucket_work).
static int bucket_impl(const void* pick, int rows, int cap, const void* tab_up, const void* tab_down, void* cnt, void* lists, void* work,
                       int stride, void* nwork, int mode, int kmode, void* stream) {
    if (rows < 1 || cap < rows || mode < 0 || mode > 2 || stride < kExperts + (rows * kTopK + kClassRb - 1) / kClassRb || !pick ||
        !tab_up || !tab_down || !cnt || !lists || !work || !nwork)
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    pf_bucket_lists<<<kExperts, 256, 0, st>>>(static_cast<const int*>(pick), rows * kTopK, cap, static_cast<int*>(cnt),
                                              static_cast<int*>(lists));
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    // kmode: the fast experts are any-width (pf_gemmk's items, schedule 2's 16 rows at most)
    if (kmode)
        pf_bucket_work<true><<<1, kExperts, 0, st>>>(static_cast<const int*>(cnt), static_cast<const PfProj*>(tab_up),
                                                     static_cast<const PfProj*>(tab_down), static_cast<int*>(work), stride,
                                                     static_cast<int*>(nwork), mode);
    else
        pf_bucket_work<false><<<1, kExperts, 0, st>>>(static_cast<const int*>(cnt), static_cast<const PfProj*>(tab_up),
                                                      static_cast<const PfProj*>(tab_down), static_cast<int*>(work), stride,
                                                      static_cast<int*>(nwork), mode);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nmp_bucket(const void* pick, int rows, int cap, const void* tab_up, const void* tab_down, void* cnt, void* lists,
                            void* work, int stride, void* nwork, int mode, void* stream) {
    return bucket_impl(pick, rows, cap, tab_up, tab_down, cnt, lists, work, stride, nwork, mode, 0, stream);
}

// gb_nmp_bucket for a MixedK table: the fast experts are those whose up and down are any-width fast (Mul1, 2 to 8 bits).
GB_EXPORT int gb_nmp_bucket_k(const void* pick, int rows, int cap, const void* tab_up, const void* tab_down, void* cnt, void* lists,
                              void* work, int stride, void* nwork, int mode, void* stream) {
    return bucket_impl(pick, rows, cap, tab_up, tab_down, cnt, lists, work, stride, nwork, mode, 1, stream);
}

// The input side of `rows` rows of a projection (K % 128 == 0): `tab` entry pick[row] (or entry 0 when pick is null), source row row / div.
static int rotate_impl(const void* src, int ksrc, int div, const void* pick, const void* tab, int rows, int K, void* out, int kmode,
                       void* stream) {
    if (rows < 1 || div < 1 || K % 128 != 0 || ksrc < K || !src || !tab || !out) return (int)cudaErrorInvalidValue;
    const long units = (long)rows * (K / 128);
    // kmode: the rows of any-width fast entries (a MixedK table) are rotated, not only the 4-bit ones
    if (kmode)
        pf_rotate<true><<<(unsigned)((units + 7) / 8), 256, 0, gb_stream(stream)>>>(
            static_cast<const float*>(src), ksrc, div, static_cast<const int*>(pick), static_cast<const PfProj*>(tab), rows, K,
            static_cast<float*>(out));
    else
        pf_rotate<false><<<(unsigned)((units + 7) / 8), 256, 0, gb_stream(stream)>>>(
            static_cast<const float*>(src), ksrc, div, static_cast<const int*>(pick), static_cast<const PfProj*>(tab), rows, K,
            static_cast<float*>(out));
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nmp_rotate(const void* src, int ksrc, int div, const void* pick, const void* tab, int rows, int K, void* out,
                            void* stream) {
    return rotate_impl(src, ksrc, div, pick, tab, rows, K, out, 0, stream);
}

// gb_nmp_rotate for a MixedK table: the rows of any-width fast entries are rotated, not only the 4-bit ones.
GB_EXPORT int gb_nmp_rotate_k(const void* src, int ksrc, int div, const void* pick, const void* tab, int rows, int K, void* out,
                              void* stream) {
    return rotate_impl(src, ksrc, div, pick, tab, rows, K, out, 1, stream);
}

// A fast projection over rotated rows: `zitems` work items of a routed list (work != null) or ceil(dense_rows / rbmax) row chunks of
// table entry 0. rbmax is 16 (the small class) or 32. `ntab` (entry g): the next projection, whose rotated input is written instead of
// the plain output.
GB_EXPORT int gb_nmp_gemm(const void* src, int ksrc, const void* lists, int lcap, const void* work, const void* nwork, int dense_rows,
                          int zitems, int rbmax, const void* tab, const void* ntab, int K, int N, int S, void* out, int relu2,
                          const void* add, void* stream) {
    if (zitems < 1 || (rbmax != 16 && rbmax != 32) || K % 128 != 0 || N % 128 != 0 || S < 1 || (K / 16) % S != 0 || ksrc < K || !src ||
        !tab || !out || (work && !nwork) || (!work && dense_rows < 1))
        return (int)cudaErrorInvalidValue;
    GemmArgs a;
    a.src = static_cast<const float*>(src);
    a.ksrc = ksrc;
    a.lists = static_cast<const int*>(lists);
    a.lcap = lcap;
    a.work = static_cast<const int*>(work);
    a.nwork = static_cast<const int*>(nwork);
    a.dense_rows = dense_rows;
    a.tab = static_cast<const PfProj*>(tab);
    a.ntab = static_cast<const PfProj*>(ntab);
    a.K = K;
    a.N = N;
    a.S = S;
    a.out = static_cast<float*>(out);
    a.relu2 = relu2;
    a.add = static_cast<const float*>(add);
    const dim3 grid((unsigned)(N / 128), 1, (unsigned)zitems);
    if (rbmax == 16) pf_gemm4<16><<<grid, kThreads, 0, gb_stream(stream)>>>(a);
    else pf_gemm4<32><<<grid, kThreads, 0, gb_stream(stream)>>>(a);
    return GB_CHECK_LAUNCH();
}

// pf_gemm4's launch for the entries of every integer width (pf_gemmk): items of at most 16 rows (schedule 2, `work`), or
// ceil(dense_rows / 16) row chunks of table entry 0 (`zitems`); the arguments otherwise gb_nmp_gemm's.
GB_EXPORT int gb_nmp_gemmk(const void* src, int ksrc, const void* lists, int lcap, const void* work, const void* nwork, int dense_rows,
                           int zitems, const void* tab, const void* ntab, int K, int N, int S, void* out, int relu2, const void* add,
                           void* stream) {
    if (zitems < 1 || K % 128 != 0 || N % 128 != 0 || S < 1 || (K / 16) % S != 0 || ksrc < K || !src || !tab || !out ||
        (work && !nwork) || (!work && dense_rows < 1))
        return (int)cudaErrorInvalidValue;
    GemmArgs a;
    a.src = static_cast<const float*>(src);
    a.ksrc = ksrc;
    a.lists = static_cast<const int*>(lists);
    a.lcap = lcap;
    a.work = static_cast<const int*>(work);
    a.nwork = static_cast<const int*>(nwork);
    a.dense_rows = dense_rows;
    a.tab = static_cast<const PfProj*>(tab);
    a.ntab = static_cast<const PfProj*>(ntab);
    a.K = K;
    a.N = N;
    a.S = S;
    a.out = static_cast<float*>(out);
    a.relu2 = relu2;
    a.add = static_cast<const float*>(add);
    pf_gemmk<<<dim3((unsigned)(N / 128), 1, (unsigned)zitems), kThreads, 0, gb_stream(stream)>>>(a);
    return GB_CHECK_LAUNCH();
}

// The slow experts' projection over `zitems` work items (bf16 or absent), plain output from plain rows (source row id / div). With
// `lists` null: one dense bf16 projection (table entry `tab[0]`) over the rows 0 .. dense_rows - 1, ceil(dense_rows / 32) items, plus
// `add` (may be null) after relu2. `variant` 0: pf_slow (a warp a column), 1: pf_slow4 (four columns a warp; N % 32 == 0).
GB_EXPORT int gb_nmp_slow(const void* src, int ksrc, int div, const void* lists, int lcap, const void* work, const void* nwork,
                          int dense_rows, int zitems, const void* tab, int K, int N, void* out, int relu2, const void* add, int variant,
                          void* stream) {
    if (zitems < 1 || div < 1 || K % 8 != 0 || ksrc < K || N < 1 || !src || !tab || !out || variant < 0 || variant > 1)
        return (int)cudaErrorInvalidValue;
    if (lists ? (!work || !nwork) : (dense_rows < 1 || zitems * kMaxRb < dense_rows)) return (int)cudaErrorInvalidValue;
    if (variant == 1) {
        if (N % 32 != 0) return (int)cudaErrorInvalidValue;
        pf_slow4<<<dim3((unsigned)(N / 32), (unsigned)zitems), 256, 0, gb_stream(stream)>>>(
            static_cast<const float*>(src), ksrc, div, static_cast<const int*>(lists), lcap, static_cast<const int*>(work),
            static_cast<const int*>(nwork), dense_rows, static_cast<const PfProj*>(tab), K, N, static_cast<float*>(out), relu2,
            static_cast<const float*>(add));
    } else {
        pf_slow<<<dim3((unsigned)((N + 7) / 8), (unsigned)zitems), 256, 0, gb_stream(stream)>>>(
            static_cast<const float*>(src), ksrc, div, static_cast<const int*>(lists), lcap, static_cast<const int*>(work),
            static_cast<const int*>(nwork), dense_rows, static_cast<const PfProj*>(tab), K, N, static_cast<float*>(out), relu2,
            static_cast<const float*>(add));
    }
    return GB_CHECK_LAUNCH();
}
