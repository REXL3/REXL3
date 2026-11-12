// gb-nvcc: --fmad=false
// Tensor-core prefill path of the Nemotron 3.5 Super VL latent-MoE block under contract P2 (lane S2; docs/NEMOTRON_PREFILL_MMA.md has
// the contract, the design and the numbers). Opt-in: crates/cuda/src/nemotron_moe_mma.rs; the default prefill path
// (nemotron_moe_prefill.cu, fp32 FMA chains with the decode path's bits) and the decode path are untouched.
//
// The arithmetic of a projection, per row:
//   input side   x * suh, H_128, * 1/sqrt(128) per 128-block, in fp32 (pf_rotate's operations), then converted to ScaledRows
//                (exl3_mma.cuh: fp16 values and one power-of-two factor per 128-block);
//   k sum        per 128-k block b (b = 0, 1, .. in order): the m16n8k16 chain over its 8 k tiles from zero (fp16 x fp16, fp32
//                accumulate), times the row's factor of block b, added to the running sum (tot = fma(part, inv, tot)). The order is a
//                function of K alone; no split K, no atomics;
//   output side  H_128 of the sum, * 1/sqrt(128), * svh, + bias, relu2, + add (pf_gemm4's operations), then either the fp32 output or,
//                for a projection whose next projection reads it, that projection's input side straight to ScaledRows.
// A row meets other rows only inside an mma, whose rows are independent, so a row's bits do not depend on the other rows, the slice
// width, the item it lands in, the padding or the order of rows (gated in tests/nemotron_moe_mma.rs).
//
// Schedule. A work item is (expert, first list position, rows <= 16 MT) of one expert's rows (the ascending pair ids of the counting
// sort, pf_bucket_lists), or (dense) a chunk of 16 MT consecutive rows. Block (column block of 128, item): 8 warps, warp w owns columns
// 16 w .. 16 w + 15 of the block and every row of the item. The item's A rows (gathered by pair id) and the weights' raw trellis tiles
// stream through a 3-stage cp.async ring of 64 k; each warp decodes its own 16 x 16 tile once per k tile (the decode path's values,
// straight into B fragments) and applies it to all of the item's m16 tiles: a weight is decoded once per item, i.e. once per 16 MT rows
// of an expert (once per slice for an expert with at most 16 MT rows). Rows past the item's count are zero-filled and their outputs
// dropped. Epilogue: the sums go through shared memory in passes of 32 rows, one warp a row for the 128-wide output side.
//
// Measurement switch (never defined in a build; a standalone harness defines it, the outputs are then wrong): NMM_ABL bit 16 skips the
// output side of the epilogue. (The k loop's former switches, bits 1 2 4 8 32, went with lane S5's rewrite of the loop.)

#include <stdint.h>

#include <type_traits>

#include "common.cuh"
#include "exl3_mma.cuh"

using namespace gbm;

namespace {

constexpr int kFmtExl3 = 0;
constexpr int kCbMul1 = 2;
constexpr int kFastTwice = 8;
constexpr int kExperts = 512;
constexpr int kThreads = 256;  // 8 warps: 8 column tiles of 16
#ifndef NMM_STAGES
#define NMM_STAGES 3
#endif
#ifndef NMM_WSTAGES
#define NMM_WSTAGES 5
#endif
constexpr int kAStages = NMM_STAGES;   // A ring depth (prefetch distance kAStages - 1; a harness may override it)
constexpr int kWStages = NMM_WSTAGES;  // weight ring depth (prefetch distance kWStages - 1)
static_assert(kWStages >= kAStages, "the weights are prefetched at least as far ahead as the rows");
constexpr int kEStride = 136;  // floats a row of the epilogue staging (128 + 8: conflict-free float2 stores)
constexpr float kInvSqrt128 = 0.08838834764831845f;
constexpr int kK2 = 8;         // 4 bits: 2 x bits
constexpr int kTileWords = 32;

// nemotron_moe.cu's NmProj (56 bytes; the Rust side builds the same table).
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

__device__ __forceinline__ bool is_fast(const PfProj& p) {
    return (p.fmt & 15) == kFmtExl3 && p.twice == kFastTwice && (p.fmt >> 4) == kCbMul1;
}
// Mul1 at an integer width of 2 to 8 bits (a MixedK layer's experts).
__device__ __forceinline__ bool is_fastk(const PfProj& p) {
    return (p.fmt & 15) == kFmtExl3 && (p.fmt >> 4) == kCbMul1 && (p.twice & 1) == 0 && p.twice >= 4 && p.twice <= 16;
}
// The entries a K2 instantiation of nmm_gemm runs: Mul1 at K2 / 2 bits (K2 = 8: is_fast, the original kernel's).
template <int K2>
__device__ __forceinline__ bool is_k2(const PfProj& p) {
    return (p.fmt & 15) == kFmtExl3 && p.twice == K2 && (p.fmt >> 4) == kCbMul1;
}
template <bool KM>
__device__ __forceinline__ bool fast_of(const PfProj& p) {
    if constexpr (KM) return is_fastk(p);
    else return is_fast(p);
}

// H_128 (unnormalised), 4 values a lane: nemotron_moe_prefill.cu's fwht128_warp (the same butterflies in the same order).
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

// mma<BF>(d, a, b) on accumulators holding +0: d = a b + 0 with the zero as an operand, so no instruction sets the accumulators first.
template <bool BF>
__device__ __forceinline__ void mma_z(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    const float z = 0.f;
    if constexpr (BF) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(z));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(z));
    }
}

struct MmaArgs {
    const uint16_t* a;   // A rows (ScaledRows values): row id at a + id * K
    const float* ainv;   // their factors: row id at ainv + id * (K / 128)
    const int* lists;    // lists[g * lcap + i]: pair ids of expert g, ascending; null: dense (ids are row numbers)
    int lcap;
    const int* work;     // items (g, first, rows); null: dense, item z = rows 16 MT z ..
    const int* nwork;
    int dense_rows;
    const PfProj* tab;   // entry g (dense: entry 0)
    const PfProj* ntab;  // the next projection (entry g): its input side is applied and the output is ScaledRows; null: fp32 output
    int K, N, nb0;       // nb0: the first column block of this call (a TP rank's column range starts there)
    float* out;          // fp32 output out[id * N + n] (ntab null)
    uint16_t* out16;     // ScaledRows output (ntab set): out16[id * N + n], factors oinv[id * (N / 128) + n / 128]
    float* oinv;
    int relu2;
    const float* add;    // [rows, N] or null (dense only: added after relu2)
    const float* add2;   // [rows, N] or null (dense only: added after `add`; may be `out` itself: read, then written, by one thread)
};

template <int MT>
__host__ __device__ constexpr int a_stage_bytes() {
    return 16 * MT * kKStage * 2;  // A rows of 64 halfs
}
// 4 k tiles x 8 column tiles of raw words at K2 / 2 bits (4 K2 words a tile)
template <int K2>
__host__ __device__ constexpr int w_stage_bytes() {
    return 4 * 8 * tile_words<K2>() * 4;
}
constexpr int kWStageBytes = w_stage_bytes<kK2>();

template <int MT, int K2 = kK2>
__host__ __device__ constexpr int ring_bytes() {
    return kAStages * a_stage_bytes<MT>() + kWStages * w_stage_bytes<K2>();
}

// The rings' region: the rings, or the epilogue staging [32][kEStride] f32 that reuses them when it is larger (a narrow width at a small
// tile: 2 bits at MT 1). At 4 bits the rings are larger at every tile, so the layout is the original kernel's.
template <int MT, int K2 = kK2>
__host__ __device__ constexpr int region_bytes() {
    return ring_bytes<MT, K2>() > 32 * kEStride * 4 ? ring_bytes<MT, K2>() : 32 * kEStride * 4;
}

template <int MT, int K2 = kK2>
__host__ __device__ constexpr int smem_bytes(int K) {
    // the A ring, the weight ring (the epilogue staging [32][kEStride] f32 reuses both), the row ids [16 MT], the rows' factors as
    // exponents [16 MT][K / 128] int8
    return region_bytes<MT, K2>() + 16 * MT * 4 + ((16 * MT * (K / kScaleBlock) + 15) & ~15);
}

// The exponent of a power-of-two factor (a normal float, or 0: -127, which decodes to 0.0f).
__device__ __forceinline__ int8_t factor_exp(float f) {
    const int e = (int)((__float_as_uint(f) >> 23) & 0xffu) - 127;
    return (int8_t)(f == 0.f ? -127 : e);
}
__device__ __forceinline__ float exp_factor(int8_t e) { return __int_as_float((127 + (int)e) << 23); }

// Block (x, 0, z): column block nb0 + x of unit z (an item of at most 16 MT rows of one expert, or a chunk of 16 MT dense rows).
// Two cp.async rings: the weights (streamed from DRAM: prefetched kWStages - 1 stages ahead, and the first ones issued before the row
// ids are even read) and the A rows (gathered, mostly L2 hits: kAStages - 1 ahead). A column block's arithmetic does not depend on the
// block or the call that ran it (a TP rank's column range).
// K2: the width (2 x bits) of the entries this instantiation runs, others' items exit (a MixedK table takes one launch a width it
// holds); K2 = 8 is the original kernel, the published pack's.
template <int MT, bool BF, int K2 = kK2>
__global__ void __launch_bounds__(kThreads, MT >= 8 ? 1 : 2) nmm_gemm(MmaArgs a) {
    static_assert(region_bytes<MT, K2>() >= 32 * kEStride * 4, "the epilogue staging must fit in the rings' region");
    static_assert(K2 != kK2 || region_bytes<MT, K2>() == ring_bytes<MT, K2>(), "the 4-bit kernel's layout is the original one");
    constexpr int TW = tile_words<K2>();  // words of a tile
    constexpr int WSB = w_stage_bytes<K2>();
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* aring = smem;
    unsigned char* wring = smem + kAStages * a_stage_bytes<MT>();
    int* rid_s = reinterpret_cast<int*>(smem + region_bytes<MT, K2>());
    int8_t* exp_s = reinterpret_cast<int8_t*>(rid_s + 16 * MT);
    const int nkb = a.K / kScaleBlock;

    int g = 0, first = 0, count = 0;
    if (a.work) {
        if ((int)blockIdx.z >= *a.nwork) return;  // uniform per block
        const int* w = a.work + 3 * blockIdx.z;
        g = w[0];
        first = w[1];
        count = w[2];
    } else {
        first = (int)blockIdx.z * 16 * MT;
        count = min(16 * MT, a.dense_rows - first);
    }
    if (count <= 0) return;
    const PfProj p = a.tab[g];
    if (!is_k2<K2>(p)) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nb = a.nb0 + (int)blockIdx.x;
    const int mt = (count + 15) >> 4;
    const int rows_used = 16 * mt;
    const int NS = a.K / kKStage;
    const size_t kt_words = (size_t)(a.N / 16) * TW;  // words between k tile rows
    const uint32_t* wcol = static_cast<const uint32_t*>(p.w) + (size_t)nb * 8 * TW;

    // a thread's copies, addressed once (lane S5: the per-stage address arithmetic was a large share of the k loop's instructions). At 4
    // bits a stage is 256 copies of 16 bytes (4 k tiles x 8 tiles x 8), one a thread; at K2 a tile is K2 copies, a stage 32 K2, so a
    // thread takes up to K2 / 8 of them (rounded up; the last round partial)
    constexpr int kWCopies = (32 * K2 + kThreads - 1) / kThreads;
    const uint32_t* w_src[kWCopies];
    int w_dst[kWCopies];
    bool w_on[kWCopies];
    if constexpr (K2 == 8) {  // the original kernel's addressing, unchanged: one copy a thread
        const int w_kt = tid >> 6, w_ntile = (tid & 63) >> 3, w_q = tid & 7;
        w_src[0] = wcol + (size_t)w_kt * kt_words + w_ntile * TW + 4 * w_q;
        w_dst[0] = (w_kt * 8 + w_ntile) * TW + 4 * w_q;
        w_on[0] = true;
    } else {
#pragma unroll
        for (int j = 0; j < kWCopies; ++j) {
            const int c = tid + kThreads * j;
            const int kt = c / (8 * K2), rem = c % (8 * K2), tile = rem / K2, q = rem % K2;
            w_on[j] = c < 32 * K2;
            w_src[j] = wcol + (size_t)kt * kt_words + tile * TW + 4 * q;
            w_dst[j] = (kt * 8 + tile) * TW + 4 * q;
        }
    }
    auto load_w = [&](int s) {
        uint32_t* ts = reinterpret_cast<uint32_t*>(wring + (s % kWStages) * WSB);
        if constexpr (K2 == 8) {
            cp_async16(ts + w_dst[0], w_src[0] + (size_t)s * 4 * kt_words);
        } else {
#pragma unroll
            for (int j = 0; j < kWCopies; ++j)
                if (w_on[j]) cp_async16(ts + w_dst[j], w_src[j] + (size_t)s * 4 * kt_words);
        }
    };
    // the A copies of this thread: chunk i = tid + kThreads j (row i / 8, 16-byte chunk i % 8), set once the row ids are known
    constexpr int kACh = (16 * MT * 8 + kThreads - 1) / kThreads;
    const uint16_t* a_src[kACh];
    int a_dst[kACh];
    bool a_issue[kACh], a_valid[kACh];
    auto load_a = [&](int s) {
        uint16_t* as = reinterpret_cast<uint16_t*>(aring + (s % kAStages) * a_stage_bytes<MT>());
#pragma unroll
        for (int j = 0; j < kACh; ++j)
            if (a_issue[j]) cp_async16_z(as + a_dst[j], a_src[j] + (a_valid[j] ? s * kKStage : 0), a_valid[j]);
    };

    // the weights first (they do not depend on the rows), then the row ids, the factors and the first A stages
#pragma unroll
    for (int s = 0; s < kWStages - 1; ++s)
        if (s < NS) load_w(s);
    cp_commit();
    for (int r = tid; r < 16 * MT; r += kThreads) rid_s[r] = r < count ? (a.lists ? a.lists[(size_t)g * a.lcap + first + r] : first + r) : -1;
    __syncthreads();
    for (int i = tid; i < rows_used * nkb; i += kThreads) {
        const int r = i / nkb, b = i % nkb;
        exp_s[i] = r < count ? factor_exp(__ldg(a.ainv + (size_t)rid_s[r] * nkb + b)) : (int8_t)-127;
    }
#pragma unroll
    for (int j = 0; j < kACh; ++j) {
        const int i = tid + kThreads * j, r = i >> 3, c = i & 7;
        a_issue[j] = i < rows_used * 8;
        a_valid[j] = r < count;
        a_src[j] = a.a + (a_issue[j] && a_valid[j] ? (size_t)rid_s[r] * a.K + c * 8 : 0);
        a_dst[j] = a_chunk(r, c);
    }
#pragma unroll
    for (int s = 0; s < kAStages - 1; ++s) {
        if (s < NS) load_a(s);
        cp_commit();  // one group a stage: step s waits for A s with only the later ones pending
    }

    float acc[MT][2][4], tot[MT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][h][c] = tot[m][h][c] = 0.f;
    const int gq = lane >> 2, tq = lane & 3;

    // The k loop, specialised on the item's active m16 tiles (lane S5): with MA a compile-time constant no MMA is predicated (a
    // predicated HMMA is issued with a full-latency stall, MEASURED with ncu); the operations of every active tile are unchanged.
    auto kloop = [&](auto ma_c) {
        constexpr int MA = decltype(ma_c)::value;
        // Groups: {W 0 .. kWStages - 2}, {A 0}, .., {A kAStages - 2}, then one a step s: {A s + kAStages - 1, W s + kWStages - 1}. At step
        // s, A s is in group {A s} or step s - (kAStages - 1)'s, with kAStages - 2 groups after it; W s is in an older one.
        auto stage = [&](int s, auto first_c) {
            constexpr bool FIRST = decltype(first_c)::value;  // the first stage of a 128-k block: its first k tile starts the chains
            cp_wait<kAStages - 2>();
            __syncthreads();  // stage s has landed for every thread, and the buffers of step s - 1 are no longer read
            if (s + kAStages - 1 < NS) load_a(s + kAStages - 1);
            if (s + kWStages - 1 < NS) load_w(s + kWStages - 1);
            cp_commit();
            const uint16_t* as = reinterpret_cast<const uint16_t*>(aring + (s % kAStages) * a_stage_bytes<MT>());
            const uint32_t* ts = reinterpret_cast<const uint32_t*>(wring + (s % kWStages) * WSB);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b0[2], b1[2];
                tile_frags<K2, kCbMul1, BF>(ts + (kk * 8 + warp) * TW, lane, b0, b1);
#pragma unroll
                for (int m = 0; m < MA; ++m) {
                    uint32_t af[4];
                    ldmatrix_x4(af, a_frag_addr(as, m, kk, lane));
                    if (FIRST && kk == 0) {  // d = a b + 0: what mma() gives on accumulators set to +0
                        mma_z<BF>(acc[m][0], af, b0);
                        mma_z<BF>(acc[m][1], af, b1);
                    } else {
                        mma<BF>(acc[m][0], af, b0);
                        mma<BF>(acc[m][1], af, b1);
                    }
                }
            }
        };
#pragma unroll 1
        for (int s = 0; s < NS; s += 2) {  // NS is even: K is a multiple of 128
            stage(s, std::true_type{});
            stage(s + 1, std::false_type{});
            // the end of 128-k block s / 2: fold_block's operations (the chains restart from +0 at the next block's first k tile)
            const int bk = s >> 1;
#pragma unroll
            for (int m = 0; m < MA; ++m) {
                const float lo = exp_factor(exp_s[(16 * m + gq) * nkb + bk]);
                const float hi = exp_factor(exp_s[(16 * m + gq + 8) * nkb + bk]);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    tot[m][h][0] = __fmaf_rn(acc[m][h][0], lo, tot[m][h][0]);
                    tot[m][h][1] = __fmaf_rn(acc[m][h][1], lo, tot[m][h][1]);
                    tot[m][h][2] = __fmaf_rn(acc[m][h][2], hi, tot[m][h][2]);
                    tot[m][h][3] = __fmaf_rn(acc[m][h][3], hi, tot[m][h][3]);
                }
            }
        }
    };
    switch (mt) {
        case 1: kloop(std::integral_constant<int, 1>{}); break;
        case 2: if constexpr (MT >= 2) kloop(std::integral_constant<int, MT >= 2 ? 2 : 1>{}); break;
        case 3: if constexpr (MT >= 3) kloop(std::integral_constant<int, MT >= 3 ? 3 : 1>{}); break;
        case 4: if constexpr (MT >= 4) kloop(std::integral_constant<int, MT >= 4 ? 4 : 1>{}); break;
        default: kloop(std::integral_constant<int, MT>{}); break;
    }
    cp_wait<0>();
    __syncthreads();  // every warp is done with the rings: their bytes become the epilogue staging

    // ---- epilogue ----
    float* E = reinterpret_cast<float*>(smem);
    const int n = nb * 128 + 4 * lane;
    const float4 sv = __ldg(reinterpret_cast<const float4*>(p.svh + n));
    const bool has_bias = p.bias != nullptr;
    const float4 bv = has_bias ? __ldg(reinterpret_cast<const float4*>(p.bias + n)) : make_float4(0.f, 0.f, 0.f, 0.f);
    const bool rot_out = a.ntab != nullptr;
    float4 nsv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (rot_out) nsv = __ldg(reinterpret_cast<const float4*>(a.ntab[g].suh + n));
    const int nkb_out = a.N / kScaleBlock;
#pragma unroll
    for (int pass = 0; pass < (MT + 1) / 2; ++pass) {
        if (2 * pass >= mt) break;  // uniform
#pragma unroll
        for (int mm = 0; mm < 2; ++mm) {
            const int m = 2 * pass + mm;
            if (m < MT && m < mt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = 16 * warp + 8 * h + 2 * tq;
                    *reinterpret_cast<float2*>(E + (16 * mm + gq) * kEStride + col) = make_float2(tot[m][h][0], tot[m][h][1]);
                    *reinterpret_cast<float2*>(E + (16 * mm + gq + 8) * kEStride + col) = make_float2(tot[m][h][2], tot[m][h][3]);
                }
            }
        }
        __syncthreads();
        // one warp a row: the output side
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int rl = warp + 8 * j, r = 32 * pass + rl;
#if defined(NMM_ABL) && (NMM_ABL & 16)
            if (r < 0) {
#else
            if (r < count) {
#endif
                const int id = rid_s[r];
                const float4 v = *reinterpret_cast<const float4*>(E + rl * kEStride + 4 * lane);
                float t[4] = {v.x, v.y, v.z, v.w};
                fwht128_warp(t, lane);
                const float sc[4] = {sv.x, sv.y, sv.z, sv.w}, bc[4] = {bv.x, bv.y, bv.z, bv.w};
                float4 ad = make_float4(0.f, 0.f, 0.f, 0.f);
                if (a.add) ad = *reinterpret_cast<const float4*>(a.add + (size_t)id * a.N + n);
                const float av[4] = {ad.x, ad.y, ad.z, ad.w};
                float4 ad2 = make_float4(0.f, 0.f, 0.f, 0.f);
                if (a.add2) ad2 = *reinterpret_cast<const float4*>(a.add2 + (size_t)id * a.N + n);
                const float av2[4] = {ad2.x, ad2.y, ad2.z, ad2.w};
                float o[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float x = (t[q] * kInvSqrt128) * sc[q];
                    if (has_bias) x = x + bc[q];
                    if (a.relu2) {
                        const float qx = fmaxf(x, 0.f);
                        x = qx * qx;
                    }
                    if (a.add) x = x + av[q];
                    if (a.add2) x = x + av2[q];
                    o[q] = x;
                }
                if (rot_out) {
                    const float ns[4] = {nsv.x, nsv.y, nsv.z, nsv.w};
                    float u[4] = {o[0] * ns[0], o[1] * ns[1], o[2] * ns[2], o[3] * ns[3]};
                    fwht128_warp(u, lane);
#pragma unroll
                    for (int q = 0; q < 4; ++q) u[q] = u[q] * kInvSqrt128;
                    store_scaled_block<BF>(u, lane, a.out16 + (size_t)id * a.N + nb * 128, a.oinv + (size_t)id * nkb_out + nb);
                } else {
                    *reinterpret_cast<float4*>(a.out + (size_t)id * a.N + n) = make_float4(o[0], o[1], o[2], o[3]);
                }
            }
        }
        __syncthreads();  // E is rewritten by the next pass
    }
}

// The input side of a projection to ScaledRows: row `row` of out16 = H_128(src[row / div] * suh) * 1/sqrt(128) per 128-block (the fp32
// values are pf_rotate's), suh of table entry pick[row] (entry 0 when pick is null). Rows whose entry is not fast are skipped. One
// warp a (row, 128-block).
// KM: the rows of any-width fast entries (a MixedK table) are converted, not only the 4-bit ones.
template <bool BF, bool KM = false>
__global__ void __launch_bounds__(256) nmm_rotate(const float* __restrict__ src, int ksrc, int div, const int* __restrict__ pick,
                                                  const PfProj* __restrict__ tab, int rows, int K, uint16_t* __restrict__ out16,
                                                  float* __restrict__ oinv) {
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
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * kInvSqrt128;
    store_scaled_block<BF>(v, lane, out16 + (size_t)row * K + hb * 128, oinv + (size_t)row * nblk + hb);
}

// A MoE layer's RMSNorm with the block's two h rotations in one pass (lane S5): block = one row of 4,096 (256 threads). nemotron_glue.cu's
// rmsnorm_kernel, operation for operation (all _rn intrinsics there and here), writing xn (the router and the slow paths read it), then
// from the normed row in shared memory nmm_rotate's operations for fc1 (tab1) and the shared expert's up projection (tab2): the ScaledRows
// inputs gb_nmm_rotate would make from xn. Both entries must be fast.
template <bool BF>
__global__ void __launch_bounds__(256) nmm_norm_rot(const float* __restrict__ x, const uint16_t* __restrict__ w, float eps,
                                                   float* __restrict__ xn, const PfProj* __restrict__ tab1, const PfProj* __restrict__ tab2,
                                                   uint16_t* __restrict__ o1, float* __restrict__ i1, uint16_t* __restrict__ o2,
                                                   float* __restrict__ i2) {
    constexpr int D = 4096, T = 256, NB = D / 128;
    __shared__ float part[T / 32];
    __shared__ __align__(16) float xs[D];
    const int row = blockIdx.x, t = threadIdx.x, lane = t & 31;
    const float* xr = x + (size_t)row * D;
    float acc = 0.f;
#pragma unroll 4
    for (int c = t; c < D; c += T) {
        const float v = __ldg(xr + c);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, off));
    if (lane == 0) part[t >> 5] = acc;
    __syncthreads();
    const float total = __fadd_rn(__fadd_rn(__fadd_rn(part[0], part[1]), __fadd_rn(part[2], part[3])),
                                  __fadd_rn(__fadd_rn(part[4], part[5]), __fadd_rn(part[6], part[7])));
    const float mean = __fdiv_rn(total, (float)D);
    const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    float* xo = xn + (size_t)row * D;
#pragma unroll 4
    for (int c = t; c < D; c += T) {
        const float wv = __uint_as_float((uint32_t)__ldg(w + c) << 16);
        const float v = __fmul_rn(__fmul_rn(__ldg(xr + c), inv), wv);
        xo[c] = v;
        xs[c] = v;
    }
    __syncthreads();
    // nmm_rotate's lines, one warp a (projection, 128-block)
#pragma unroll 1
    for (int u = t >> 5; u < 2 * NB; u += T / 32) {
        const PfProj& p = u < NB ? *tab1 : *tab2;
        const int hb = u % NB;
        const float4 xv = *reinterpret_cast<const float4*>(xs + hb * 128 + 4 * lane);
        const float4 sv = __ldg(reinterpret_cast<const float4*>(p.suh + hb * 128 + 4 * lane));
        float v[4] = {xv.x * sv.x, xv.y * sv.y, xv.z * sv.z, xv.w * sv.w};
        fwht128_warp(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = v[j] * kInvSqrt128;
        if (u < NB) store_scaled_block<BF>(v, lane, o1 + (size_t)row * D + hb * 128, i1 + (size_t)row * NB + hb);
        else store_scaled_block<BF>(v, lane, o2 + (size_t)row * D + hb * 128, i2 + (size_t)row * NB + hb);
    }
}

// Router logits on tensor cores: logits[r, e] = h[r, :] . gate[e, :], gate bf16 [512][K] (exact in bf16), h fp32 split into two bf16
// terms hi = bf16(h), lo = bf16(h - hi) (about 16 significant bits of h; the products are exact, the sums fp32): per k tile the chain
// adds hi . gate then lo . gate, k tiles in order from zero. A row's logits depend on that row alone; the order is a function of K.
// Block (expert block of 128, row block of 64): 8 warps, warp w the experts 16 w .. 16 w + 15 of the block and every row; stages of 64 k:
// the gate rows by cp.async (two buffers), h converted to hi / lo bf16 in shared memory by the threads.
constexpr int kRRows = 64, kRK = 64;
constexpr int kRouterSmem = 2 * (128 + 2 * kRRows) * kRK * 2;  // 64 KB: dynamic, one block an SM
__global__ void __launch_bounds__(kThreads, 1) nmm_router(const float* __restrict__ h, const uint16_t* __restrict__ gate, int rows, int K,
                                                          float* __restrict__ logits) {
    extern __shared__ __align__(16) unsigned char rsm[];
    uint16_t (*ghs)[128 * kRK] = reinterpret_cast<uint16_t (*)[128 * kRK]>(rsm);                         // gate rows [expert][k]
    uint16_t (*ahs)[kRRows * kRK] = reinterpret_cast<uint16_t (*)[kRRows * kRK]>(rsm + 2 * 128 * kRK * 2);  // h hi
    uint16_t (*als)[kRRows * kRK] = reinterpret_cast<uint16_t (*)[kRRows * kRK]>(rsm + 2 * (128 + kRRows) * kRK * 2);  // h lo
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int e0 = blockIdx.x * 128, r0 = blockIdx.y * kRRows;
    const int nrows = min(kRRows, rows - r0);
    const int mt = (nrows + 15) >> 4;
    const int NS = K / kRK;
    auto load_g = [&](int s, int buf) {
        for (int i = tid; i < 128 * 8; i += kThreads) {
            const int e = i >> 3, c = i & 7;
            cp_async16(ghs[buf] + a_chunk(e, c), gate + (size_t)(e0 + e) * K + s * kRK + c * 8);
        }
        cp_commit();
    };
    auto load_h = [&](int s, int buf) {  // 64 rows x 64 k: 1024 float4, 4 a thread
        for (int i = tid; i < kRRows * 16; i += kThreads) {
            const int r = i >> 4, q = i & 15;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < nrows) v = __ldg(reinterpret_cast<const float4*>(h + (size_t)(r0 + r) * K + s * kRK + 4 * q));
            const __nv_bfloat162 h01 = __floats2bfloat162_rn(v.x, v.y), h23 = __floats2bfloat162_rn(v.z, v.w);
            const float2 f01 = __bfloat1622float2(h01), f23 = __bfloat1622float2(h23);
            const __nv_bfloat162 l01 = __floats2bfloat162_rn(v.x - f01.x, v.y - f01.y), l23 = __floats2bfloat162_rn(v.z - f23.x, v.w - f23.y);
            const int c = q >> 1, off = (q & 1) * 4;  // chunk of 8 values, half of it
            uint2 hu, lu;
            hu.x = *reinterpret_cast<const uint32_t*>(&h01);
            hu.y = *reinterpret_cast<const uint32_t*>(&h23);
            lu.x = *reinterpret_cast<const uint32_t*>(&l01);
            lu.y = *reinterpret_cast<const uint32_t*>(&l23);
            *reinterpret_cast<uint2*>(ahs[buf] + a_chunk(r, c) + off) = hu;
            *reinterpret_cast<uint2*>(als[buf] + a_chunk(r, c) + off) = lu;
        }
    };
    float acc[4][2][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][hh][c] = 0.f;
    load_g(0, 0);
    load_h(0, 0);
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
        const int buf = s & 1;
        cp_wait<0>();
        __syncthreads();  // stage s is in shared memory; stage s - 1's buffers are free
        if (s + 1 < NS) {
            load_g(s + 1, buf ^ 1);
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            // B fragments of the warp's 16 experts: matrices (experts 0-7, k 0-7), (0-7, k 8-15), (8-15, k 0-7), (8-15, k 8-15)
            uint32_t bq[4];
            {
                const int e = 16 * warp + (lane & 7) + ((lane >> 4) << 3), c = 2 * kk + ((lane >> 3) & 1);
                ldmatrix_x4(bq, ghs[buf] + a_chunk(e, c));
            }
            const uint32_t b0[2] = {bq[0], bq[1]}, b1[2] = {bq[2], bq[3]};
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < mt) {
                    uint32_t ah[4], al[4];
                    ldmatrix_x4(ah, a_frag_addr(ahs[buf], m, kk, lane));
                    ldmatrix_x4(al, a_frag_addr(als[buf], m, kk, lane));
                    mma<true>(acc[m][0], ah, b0);
                    mma<true>(acc[m][0], al, b0);
                    mma<true>(acc[m][1], ah, b1);
                    mma<true>(acc[m][1], al, b1);
                }
            }
        }
        if (s + 1 < NS) load_h(s + 1, buf ^ 1);  // after this stage's reads of the other buffer are long done (synced at the top)
    }
    const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int e = e0 + 16 * warp + 8 * hh + 2 * tq;
            const int ra = 16 * m + gq, rb = ra + 8;
            if (ra < nrows) *reinterpret_cast<float2*>(logits + (size_t)(r0 + ra) * kExperts + e) = make_float2(acc[m][hh][0], acc[m][hh][1]);
            if (rb < nrows) *reinterpret_cast<float2*>(logits + (size_t)(r0 + rb) * kExperts + e) = make_float2(acc[m][hh][2], acc[m][hh][3]);
        }
    }
}

// Router logits in fp32 with the decode path's bits (lane S5): logits[r, e] = h[r, :] . gate[e, :] computed exactly as
// nemotron_moe.cu's nm2_router_logits computes it: lane l of a warp owns the 8-element chunks q = l + 32 i (i = 0 .. 15) of the K = 4096
// row, each (row, expert) chain is that lane's chunks in increasing i (eight fused multiply-adds in element order a chunk, the gate's
// bf16 widened exactly), then the xor tree 16, 8, 4, 2, 1 across the lanes. Only where the operands come from differs: a block of 16
// warps (4 x 4, each 8 rows x 8 experts) covers 32 rows x 32 experts and stages each k slab of 256 (one i) of its rows and experts
// through shared memory by cp.async, double buffered, so a slab is read from L2 once per block instead of once per warp (ncu: the
// decode-path kernel at prefill widths is bound by L2 bandwidth). h is staged as the two float4 halves of each chunk in separate arrays
// (conflict-free 16-byte reads per lane).
constexpr int kRtRows = 32, kRtK = 4096;

__device__ __forceinline__ float rt_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float rt_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float rt_chunk(float acc, const float4 a, const float4 b, const uint4 w) {
    acc = __fmaf_rn(a.x, rt_lo(w.x), acc);
    acc = __fmaf_rn(a.y, rt_hi(w.x), acc);
    acc = __fmaf_rn(a.z, rt_lo(w.y), acc);
    acc = __fmaf_rn(a.w, rt_hi(w.y), acc);
    acc = __fmaf_rn(b.x, rt_lo(w.z), acc);
    acc = __fmaf_rn(b.y, rt_hi(w.z), acc);
    acc = __fmaf_rn(b.z, rt_lo(w.w), acc);
    acc = __fmaf_rn(b.w, rt_hi(w.w), acc);
    return acc;
}

// An f32 gate's chunk (eight values; an F16 gate widened exactly): the same eight fused multiply-adds in element order.
struct RtG32 {
    float4 lo, hi;
};
__device__ __forceinline__ float rt_chunk(float acc, const float4 a, const float4 b, const RtG32 w) {
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

// The block shape of nmm_router_f32 for a gate element type: a bf16 gate's block is 32 rows x 32 experts (16 warps); an f32 gate's
// chunks are twice the bytes, so its block is 32 rows x 16 experts (8 warps) and a stage stays at 48 KB (two of them under the 99 KB
// a block may have). Each warp is 8 rows x 8 experts either way.
template <typename G>
struct RtShape;
template <>
struct RtShape<uint16_t> {
    using V = uint4;
    static constexpr int kExp = 32, kPieces = 1;  // 16-byte pieces of a (expert, lane) chunk
};
template <>
struct RtShape<float> {
    using V = RtG32;
    static constexpr int kExp = 16, kPieces = 2;
};
// An F16 gate as stored: a chunk is 16 bytes of fp16 bits (the bf16 block shape), widened exactly in rt_chunk.
struct RtH8 {
    uint4 w;
};
__device__ __forceinline__ float rt_chunk(float acc, const float4 a, const float4 b, const RtH8 h) {
    const float2 w0 = __half22float2(*reinterpret_cast<const half2*>(&h.w.x)), w1 = __half22float2(*reinterpret_cast<const half2*>(&h.w.y));
    const float2 w2 = __half22float2(*reinterpret_cast<const half2*>(&h.w.z)), w3 = __half22float2(*reinterpret_cast<const half2*>(&h.w.w));
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
template <>
struct RtShape<__half> {
    using V = RtH8;
    static constexpr int kExp = 32, kPieces = 1;
};
template <typename G>
__host__ __device__ constexpr int rt_threads() {
    return (kRtRows / 8) * (RtShape<G>::kExp / 8) * 32;
}
template <typename G>
__host__ __device__ constexpr int rt_stage() {
    return kRtRows * 256 * 4 + RtShape<G>::kExp * 256 * (int)sizeof(G);  // 32 KB of h; 16 KB of gate
}

template <typename G>
__global__ void __launch_bounds__(rt_threads<G>(), 1) nmm_router_f32(const float* __restrict__ h, const G* __restrict__ gate,
                                                                    float* __restrict__ logits, int rows) {
    constexpr int kRtExp = RtShape<G>::kExp, kPc = RtShape<G>::kPieces, kRtThreads = rt_threads<G>(), kRtStage = rt_stage<G>();
    constexpr int kWe = kRtExp / 8;  // warps across the experts
    using V = typename RtShape<G>::V;
    extern __shared__ __align__(16) unsigned char rsm[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wr = warp / kWe, we = warp % kWe;
    const int e0 = blockIdx.x * kRtExp, r0 = blockIdx.y * kRtRows;
    // stage layout: xa [32 rows][32 lanes] float4, xb [32][32] float4, g [kRtExp experts][32 lanes] V
    auto xa = [&](int buf) { return reinterpret_cast<float4*>(rsm + buf * kRtStage); };
    auto xb = [&](int buf) { return reinterpret_cast<float4*>(rsm + buf * kRtStage + kRtRows * 32 * 16); };
    auto gs = [&](int buf) { return reinterpret_cast<V*>(rsm + buf * kRtStage + 2 * kRtRows * 32 * 16); };
    auto load = [&](int i, int buf) {
        // h: 32 rows x 64 pieces of 16 bytes; gate: kRtExp experts x 32 lanes x kPc pieces
        for (int p = tid; p < kRtRows * 64 + kRtExp * 32 * kPc; p += kRtThreads) {
            if (p < kRtRows * 64) {
                const int r = p >> 6, c = p & 63, l = c >> 1;
                const bool ok = r0 + r < rows;
                const float* src = h + (ok ? (size_t)(r0 + r) * kRtK + 256 * i + 4 * c : 0);
                cp_async16_z((c & 1 ? xb(buf) : xa(buf)) + r * 32 + l, src, ok);
            } else {
                const int q = p - kRtRows * 64, e = q / (32 * kPc), l = (q / kPc) & 31, pc = q % kPc;
                cp_async16(reinterpret_cast<uint4*>(gs(buf) + e * 32 + l) + pc,
                           gate + (size_t)(e0 + e) * kRtK + 256 * i + 8 * l + pc * (16 / (int)sizeof(G)));
            }
        }
        cp_commit();
    };
    float acc[8][8];
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[r][e] = 0.f;
    load(0, 0);
#pragma unroll 1
    for (int i = 0; i < kRtK / 256; ++i) {
        const int buf = i & 1;
        if (i + 1 < kRtK / 256) {
            load(i + 1, buf ^ 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        V w[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) w[e] = gs(buf)[(8 * we + e) * 32 + lane];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float4 a = xa(buf)[(8 * wr + r) * 32 + lane], b = xb(buf)[(8 * wr + r) * 32 + lane];
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[r][e] = rt_chunk(acc[r][e], a, b, w[e]);
        }
        __syncthreads();  // buffer buf is refilled by the next iteration's load
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc[r][e] += __shfl_xor_sync(0xffffffffu, acc[r][e], off);
        }
        const int row = r0 + 8 * wr + r;
        if (lane == 0 && row < rows) {
#pragma unroll
            for (int e = 0; e < 8; ++e) logits[(size_t)row * kExperts + e0 + 8 * we + e] = acc[r][e];
        }
    }
}

// The tensor-core router with an f32 gate (an F16 gate widened): the gate is split as h is, g = g_hi + g_lo in bf16 (exact for an F16
// value: its 11 significant bits are bf16's 8 and a remainder of at most 3, with exponents well inside bf16's range), and per k tile the
// chain adds hi . g_hi, lo . g_hi, hi . g_lo, then lo . g_lo (k tiles in order from zero). The same layout as nmm_router; the gate's two
// halves are converted by the threads into shared memory (64 KB), h's as before (32 KB). Like nmm_router, not the decode path's bits.
constexpr int kRouterSmemG32 = 2 * (2 * 128 + 2 * kRRows) * kRK * 2;  // 96 KB: dynamic, one block an SM
template <typename G>
__global__ void __launch_bounds__(kThreads, 1) nmm_router_g32(const float* __restrict__ h, const G* __restrict__ gate, int rows, int K,
                                                              float* __restrict__ logits) {
    extern __shared__ __align__(16) unsigned char rsm[];
    uint16_t (*ghi)[128 * kRK] = reinterpret_cast<uint16_t (*)[128 * kRK]>(rsm);                          // gate hi [expert][k]
    uint16_t (*glo)[128 * kRK] = reinterpret_cast<uint16_t (*)[128 * kRK]>(rsm + 2 * 128 * kRK * 2);      // gate lo
    uint16_t (*ahs)[kRRows * kRK] = reinterpret_cast<uint16_t (*)[kRRows * kRK]>(rsm + 4 * 128 * kRK * 2);  // h hi
    uint16_t (*als)[kRRows * kRK] = reinterpret_cast<uint16_t (*)[kRRows * kRK]>(rsm + (4 * 128 + 2 * kRRows) * kRK * 2);  // h lo
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int e0 = blockIdx.x * 128, r0 = blockIdx.y * kRRows;
    const int nrows = min(kRRows, rows - r0);
    const int mt = (nrows + 15) >> 4;
    const int NS = K / kRK;
    // 4 f32 values to (hi, lo) bf16 pairs at chunk (row r, chunk c), half `off` of it, in hs / ls
    auto split4 = [](const float4 v, uint16_t* hs, uint16_t* ls) {
        const __nv_bfloat162 h01 = __floats2bfloat162_rn(v.x, v.y), h23 = __floats2bfloat162_rn(v.z, v.w);
        const float2 f01 = __bfloat1622float2(h01), f23 = __bfloat1622float2(h23);
        const __nv_bfloat162 l01 = __floats2bfloat162_rn(v.x - f01.x, v.y - f01.y), l23 = __floats2bfloat162_rn(v.z - f23.x, v.w - f23.y);
        uint2 hu, lu;
        hu.x = *reinterpret_cast<const uint32_t*>(&h01);
        hu.y = *reinterpret_cast<const uint32_t*>(&h23);
        lu.x = *reinterpret_cast<const uint32_t*>(&l01);
        lu.y = *reinterpret_cast<const uint32_t*>(&l23);
        *reinterpret_cast<uint2*>(hs) = hu;
        *reinterpret_cast<uint2*>(ls) = lu;
    };
    auto load_g = [&](int s, int buf) {  // 128 experts x 64 k: 2048 groups of 4, 8 a thread
        for (int i = tid; i < 128 * 16; i += kThreads) {
            const int e = i >> 4, q = i & 15;
            float4 v;
            if constexpr (sizeof(G) == 4) {
                v = __ldg(reinterpret_cast<const float4*>(gate + (size_t)(e0 + e) * K + s * kRK + 4 * q));
            } else {  // fp16 bits, widened exactly: the f32 gate's values
                const uint2 u = __ldg(reinterpret_cast<const uint2*>(gate + (size_t)(e0 + e) * K + s * kRK + 4 * q));
                const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x)), b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
                v = make_float4(a.x, a.y, b.x, b.y);
            }
            const int c = q >> 1, off = (q & 1) * 4;
            split4(v, ghi[buf] + a_chunk(e, c) + off, glo[buf] + a_chunk(e, c) + off);
        }
    };
    auto load_h = [&](int s, int buf) {  // 64 rows x 64 k: 1024 float4, 4 a thread
        for (int i = tid; i < kRRows * 16; i += kThreads) {
            const int r = i >> 4, q = i & 15;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < nrows) v = __ldg(reinterpret_cast<const float4*>(h + (size_t)(r0 + r) * K + s * kRK + 4 * q));
            const int c = q >> 1, off = (q & 1) * 4;
            split4(v, ahs[buf] + a_chunk(r, c) + off, als[buf] + a_chunk(r, c) + off);
        }
    };
    float acc[4][2][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][hh][c] = 0.f;
    load_g(0, 0);
    load_h(0, 0);
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
        const int buf = s & 1;
        __syncthreads();  // stage s is in shared memory; stage s - 1's buffers are free
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t bq[4], bl[4];
            {
                const int e = 16 * warp + (lane & 7) + ((lane >> 4) << 3), c = 2 * kk + ((lane >> 3) & 1);
                ldmatrix_x4(bq, ghi[buf] + a_chunk(e, c));
                ldmatrix_x4(bl, glo[buf] + a_chunk(e, c));
            }
            const uint32_t b0[2] = {bq[0], bq[1]}, b1[2] = {bq[2], bq[3]};
            const uint32_t c0[2] = {bl[0], bl[1]}, c1[2] = {bl[2], bl[3]};
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < mt) {
                    uint32_t ah[4], al[4];
                    ldmatrix_x4(ah, a_frag_addr(ahs[buf], m, kk, lane));
                    ldmatrix_x4(al, a_frag_addr(als[buf], m, kk, lane));
                    mma<true>(acc[m][0], ah, b0);
                    mma<true>(acc[m][0], al, b0);
                    mma<true>(acc[m][0], ah, c0);
                    mma<true>(acc[m][0], al, c0);
                    mma<true>(acc[m][1], ah, b1);
                    mma<true>(acc[m][1], al, b1);
                    mma<true>(acc[m][1], ah, c1);
                    mma<true>(acc[m][1], al, c1);
                }
            }
        }
        if (s + 1 < NS) {  // the other buffer: its last reads were before this iteration's barrier
            load_g(s + 1, buf ^ 1);
            load_h(s + 1, buf ^ 1);
        }
    }
    const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int e = e0 + 16 * warp + 8 * hh + 2 * tq;
            const int ra = 16 * m + gq, rb = ra + 8;
            if (ra < nrows) *reinterpret_cast<float2*>(logits + (size_t)(r0 + ra) * kExperts + e) = make_float2(acc[m][hh][0], acc[m][hh][1]);
            if (rb < nrows) *reinterpret_cast<float2*>(logits + (size_t)(r0 + rb) * kExperts + e) = make_float2(acc[m][hh][2], acc[m][hh][3]);
        }
    }
}

// The work items of the fast experts (4-bit Mul1 up and down) from the per-expert row counts: an expert with c rows is ceil(c / cap)
// items of near-equal rows (item k of m takes list positions [k c / m, (k + 1) c / m)), the experts ordered by c, largest first (ties
// by id), so the longest blocks start first. One block of 512 threads (thread e: expert e). The order changes no bit of any row.
// KM: the fast experts are those whose up and down are any-width fast (a MixedK table).
template <bool KM = false>
__global__ void __launch_bounds__(kExperts) nmm_items(const int* __restrict__ cnt, const PfProj* __restrict__ tab_up,
                                                      const PfProj* __restrict__ tab_down, int cap, int* __restrict__ work,
                                                      int* __restrict__ nwork) {
    __shared__ int s_c[kExperts], s_it[kExperts];
    const int e = threadIdx.x;
    const bool fast = fast_of<KM>(tab_up[e]) && fast_of<KM>(tab_down[e]);
    const int c = fast ? cnt[e] : 0;
    s_c[e] = c;
    s_it[e] = (c + cap - 1) / cap;
    __syncthreads();
    int base = 0, tot = 0;
    for (int j = 0; j < kExperts; ++j) {
        const int cj = s_c[j];
        base += (cj > c || (cj == c && j < e)) ? s_it[j] : 0;
        tot += s_it[j];
    }
    const int items = s_it[e];
    for (int k = 0; k < items; ++k) {
        int* it = work + 3 * (base + k);
        const int lo = (k * c) / items, hi = ((k + 1) * c) / items;
        it[0] = e;
        it[1] = lo;
        it[2] = hi - lo;
    }
    if (e == 0) *nwork = tot;
}

template <int MT, bool BF, int K2 = kK2>
int launch_gemm(const MmaArgs& a, int ncb, int zitems, cudaStream_t st) {
    static bool attr_set = false;  // per instantiation
    const int bytes = smem_bytes<MT, K2>(a.K);
    if (!attr_set) {
        const cudaError_t e = cudaFuncSetAttribute(nmm_gemm<MT, BF, K2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
        if (e != cudaSuccess) return (int)e;
        attr_set = true;
    }
    nmm_gemm<MT, BF, K2><<<dim3((unsigned)ncb, 1, (unsigned)zitems), kThreads, bytes, st>>>(a);
    return GB_CHECK_LAUNCH();
}

// ---- fp16 dense projections (v0.1.2 lane K): a stored fp16 weight [N][K] on tensor cores under P2 --------------------------------
//
// The MixedK pack's fc1 / fc2 latents are fp16 dense, not EXL3: no rotation on either side and no scales. The P2 arithmetic of such a
// projection, per row:
//   input side   the plain fp32 values to ScaledRows (fp16 values, one power-of-two factor a 128-block: nmm_plain_rows; fp16 keeps an
//                11-bit significand for every value of a block, so the plain values need no rotation for their precision);
//   k sum        nmm_gemm's: per 128-k block b in order, the m16n8k16 chain over its 8 k tiles from zero (fp16 x fp16, fp32
//                accumulate), times the row's factor of block b, folded into the running sum (tot = fma(part, inv, tot));
//   output side  relu2, + add, + add2 (nmm_gemm's order; no Hadamard, scale or bias: the stored weight is the projection), fp32.
// The B operands are the stored fp16 values exactly. A row meets other rows only inside an mma: a row's bits do not depend on the other
// rows, the chunk, the tile or the call, as nmm_gemm's.
// Block (column block of 128, row chunk of 16 MT): 8 warps, warp w the columns 16 w .. 16 w + 15 and every row of the chunk. The A rows
// and the weight rows stream through one 3-stage cp.async ring of 64 k; a weight stage is 128 rows x 64 halves with the A stage's
// XOR-swizzled 16-byte chunks (ldmatrix of 8 consecutive rows at one chunk is conflict-free). The epilogue writes from the registers.

constexpr int kHStageHalves = 128 * kKStage;  // 16 KB a weight stage

template <int MT>
__host__ __device__ constexpr int h16_smem_bytes(int K) {
    // the A ring, the weight ring, the rows' factors as exponents [16 MT][K / 128] int8
    return kAStages * a_stage_bytes<MT>() + kAStages * kHStageHalves * 2 + ((16 * MT * (K / kScaleBlock) + 15) & ~15);
}

struct H16Args {
    const uint16_t* a;  // ScaledRows rows [rows][K]
    const float* ainv;  // their factors [rows][K / 128]
    int rows;
    const PfProj* tab;  // entry 0: the fp16 weight (fmt 3), w [N][K]
    int K, N;
    float* out;         // [rows][N]
    int relu2;
    const float* add;   // [rows][N] or null
    const float* add2;  // [rows][N] or null (may be out: read, then written, by one thread)
};

// This lane's ldmatrix.x4 address of the B fragments of k tile kk for warp `warp`: matrices (n 0-7, k 0-7), (n 0-7, k 8-15),
// (n 8-15, k 0-7), (n 8-15, k 8-15) of the warp's 16 columns are b0[0], b0[1], b1[0], b1[1].
__device__ __forceinline__ const uint16_t* h16_frag_addr(const uint16_t* stage, int warp, int kk, int lane) {
    const int j = lane >> 3, n = 16 * warp + ((j >> 1) << 3) + (lane & 7), c = 2 * kk + (j & 1);
    return stage + n * kKStage + ((c ^ (n & 7)) << 3);
}

template <int MT>
__global__ void __launch_bounds__(kThreads, 1) nmm_dense_h16(H16Args a) {
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* aring = smem;
    uint16_t* wring = reinterpret_cast<uint16_t*>(smem + kAStages * a_stage_bytes<MT>());
    int8_t* exp_s = reinterpret_cast<int8_t*>(smem + kAStages * a_stage_bytes<MT>() + kAStages * kHStageHalves * 2);
    const int nkb = a.K / kScaleBlock;
    const int first = (int)blockIdx.z * 16 * MT;
    const int count = min(16 * MT, a.rows - first);
    if (count <= 0) return;
    const PfProj p = a.tab[0];
    if ((p.fmt & 15) != 3) return;  // not an fp16 weight
    const uint16_t* w = static_cast<const uint16_t*>(p.w);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nb = (int)blockIdx.x;
    const int mt = (count + 15) >> 4;
    const int rows_used = 16 * mt;
    const int NS = a.K / kKStage;

    // the weight copies of a stage: 128 rows x 8 chunks of 16 bytes, 4 a thread
    const uint16_t* w_src[4];
    int w_dst[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c = tid + kThreads * j, r = c >> 3, ch = c & 7;
        w_src[j] = w + (size_t)(nb * 128 + r) * a.K + 8 * ch;
        w_dst[j] = r * kKStage + ((ch ^ (r & 7)) << 3);
    }
    // the A copies of this thread: chunk i = tid + kThreads j (row i / 8, 16-byte chunk i % 8); rows past the count are zero-filled
    constexpr int kACh = (16 * MT * 8 + kThreads - 1) / kThreads;
    const uint16_t* a_src[kACh];
    int a_dst[kACh];
    bool a_issue[kACh], a_valid[kACh];
#pragma unroll
    for (int j = 0; j < kACh; ++j) {
        const int i = tid + kThreads * j, r = i >> 3, c = i & 7;
        a_issue[j] = i < rows_used * 8;
        a_valid[j] = r < count;
        a_src[j] = a.a + (a_issue[j] && a_valid[j] ? (size_t)(first + r) * a.K + c * 8 : 0);
        a_dst[j] = a_chunk(r, c);
    }
    auto load = [&](int s) {
        uint16_t* as = reinterpret_cast<uint16_t*>(aring + (s % kAStages) * a_stage_bytes<MT>());
        uint16_t* ws = wring + (s % kAStages) * kHStageHalves;
#pragma unroll
        for (int j = 0; j < 4; ++j) cp_async16(ws + w_dst[j], w_src[j] + (size_t)s * kKStage);
#pragma unroll
        for (int j = 0; j < kACh; ++j)
            if (a_issue[j]) cp_async16_z(as + a_dst[j], a_src[j] + (a_valid[j] ? s * kKStage : 0), a_valid[j]);
    };
#pragma unroll
    for (int s = 0; s < kAStages - 1; ++s) {
        if (s < NS) load(s);
        cp_commit();  // one group a stage
    }
    for (int i = tid; i < rows_used * nkb; i += kThreads) {
        const int r = i / nkb, b = i % nkb;
        exp_s[i] = r < count ? factor_exp(__ldg(a.ainv + (size_t)(first + r) * nkb + b)) : (int8_t)-127;
    }

    float acc[MT][2][4], tot[MT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][h][c] = tot[m][h][c] = 0.f;
    const int gq = lane >> 2, tq = lane & 3;

    // the k loop, specialised on the chunk's active m16 tiles (no predicated MMA), as nmm_gemm's
    auto kloop = [&](auto ma_c) {
        constexpr int MA = decltype(ma_c)::value;
        auto stage = [&](int s, auto first_c) {
            constexpr bool FIRST = decltype(first_c)::value;  // the first stage of a 128-k block: its first k tile starts the chains
            cp_wait<kAStages - 2>();
            __syncthreads();  // stage s has landed for every thread (and the factors are in), and stage s - 1's buffers are free
            if (s + kAStages - 1 < NS) load(s + kAStages - 1);
            cp_commit();
            const uint16_t* as = reinterpret_cast<const uint16_t*>(aring + (s % kAStages) * a_stage_bytes<MT>());
            const uint16_t* ws = wring + (s % kAStages) * kHStageHalves;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t bfr[4];
                ldmatrix_x4(bfr, h16_frag_addr(ws, warp, kk, lane));
                const uint32_t b0[2] = {bfr[0], bfr[1]}, b1[2] = {bfr[2], bfr[3]};
#pragma unroll
                for (int m = 0; m < MA; ++m) {
                    uint32_t af[4];
                    ldmatrix_x4(af, a_frag_addr(as, m, kk, lane));
                    if (FIRST && kk == 0) {  // d = a b + 0: what mma() gives on accumulators set to +0
                        mma_z<false>(acc[m][0], af, b0);
                        mma_z<false>(acc[m][1], af, b1);
                    } else {
                        mma<false>(acc[m][0], af, b0);
                        mma<false>(acc[m][1], af, b1);
                    }
                }
            }
        };
#pragma unroll 1
        for (int s = 0; s < NS; s += 2) {  // NS is even: K is a multiple of 128
            stage(s, std::true_type{});
            stage(s + 1, std::false_type{});
            // the end of 128-k block s / 2: nmm_gemm's fold
            const int bk = s >> 1;
#pragma unroll
            for (int m = 0; m < MA; ++m) {
                const float lo = exp_factor(exp_s[(16 * m + gq) * nkb + bk]);
                const float hi = exp_factor(exp_s[(16 * m + gq + 8) * nkb + bk]);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    tot[m][h][0] = __fmaf_rn(acc[m][h][0], lo, tot[m][h][0]);
                    tot[m][h][1] = __fmaf_rn(acc[m][h][1], lo, tot[m][h][1]);
                    tot[m][h][2] = __fmaf_rn(acc[m][h][2], hi, tot[m][h][2]);
                    tot[m][h][3] = __fmaf_rn(acc[m][h][3], hi, tot[m][h][3]);
                }
            }
        }
    };
    switch (mt) {
        case 1: kloop(std::integral_constant<int, 1>{}); break;
        case 2: if constexpr (MT >= 2) kloop(std::integral_constant<int, MT >= 2 ? 2 : 1>{}); break;
        case 3: if constexpr (MT >= 3) kloop(std::integral_constant<int, MT >= 3 ? 3 : 1>{}); break;
        case 4: if constexpr (MT >= 4) kloop(std::integral_constant<int, MT >= 4 ? 4 : 1>{}); break;
        default: kloop(std::integral_constant<int, MT>{}); break;
    }
    cp_wait<0>();

    // ---- epilogue: straight from the registers (two adjacent columns a value pair) ----
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m >= mt) break;  // uniform
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int n = nb * 128 + 16 * warp + 8 * h + 2 * tq;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int r = 16 * m + gq + 8 * half;
                if (r >= count) continue;
                const size_t at = (size_t)(first + r) * a.N + n;
                float x[2] = {tot[m][h][2 * half], tot[m][h][2 * half + 1]};
                float2 ad = make_float2(0.f, 0.f), ad2 = make_float2(0.f, 0.f);
                if (a.add) ad = *reinterpret_cast<const float2*>(a.add + at);
                if (a.add2) ad2 = *reinterpret_cast<const float2*>(a.add2 + at);
                const float av[2] = {ad.x, ad.y}, av2[2] = {ad2.x, ad2.y};
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    if (a.relu2) {
                        const float qx = fmaxf(x[q], 0.f);
                        x[q] = qx * qx;
                    }
                    if (a.add) x[q] = x[q] + av[q];
                    if (a.add2) x[q] = x[q] + av2[q];
                }
                *reinterpret_cast<float2*>(a.out + at) = make_float2(x[0], x[1]);
            }
        }
    }
}

// Plain fp32 rows to ScaledRows (no rotation, no scale): row `row` of out16 = src[row] per 128-block. One warp a (row, 128-block).
__global__ void __launch_bounds__(256) nmm_plain_rows(const float* __restrict__ src, int ksrc, int rows, int K, uint16_t* __restrict__ out16,
                                                      float* __restrict__ oinv) {
    const int lane = threadIdx.x & 31;
    const int nblk = K / 128;
    const long u = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (u >= (long)rows * nblk) return;  // a whole warp returns
    const int row = (int)(u / nblk), hb = (int)(u % nblk);
    const float4 xv = __ldg(reinterpret_cast<const float4*>(src + (size_t)row * ksrc + hb * 128 + 4 * lane));
    const float v[4] = {xv.x, xv.y, xv.z, xv.w};
    store_scaled_block<false>(v, lane, out16 + (size_t)row * K + hb * 128, oinv + (size_t)row * nblk + hb);
}

template <int MT>
int launch_h16(const H16Args& a, cudaStream_t st) {
    static bool attr_set = false;  // per instantiation
    const int bytes = h16_smem_bytes<MT>(a.K);
    if (bytes > 101376) return (int)cudaErrorInvalidValue;
    if (!attr_set) {
        const cudaError_t e = cudaFuncSetAttribute(nmm_dense_h16<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
        if (e != cudaSuccess) return (int)e;
        attr_set = true;
    }
    nmm_dense_h16<MT><<<dim3((unsigned)(a.N / 128), 1, (unsigned)((a.rows + 16 * MT - 1) / (16 * MT))), kThreads, bytes, st>>>(a);
    return GB_CHECK_LAUNCH();
}

// launch_gemm at a runtime width k2 (4, 6, .., 16) and tile mt (1, 2, 4, 8), fp16 operands only (the bf16 operands are the quality
// comparison's, at 4 bits)
template <int K2>
int launch_gemm_k2(const MmaArgs& a, int mt, int ncb, int zitems, cudaStream_t st) {
    if (smem_bytes<8, K2>(a.K) > 101376 && mt == 8) return (int)cudaErrorInvalidValue;
    switch (mt) {
        case 1: return launch_gemm<1, false, K2>(a, ncb, zitems, st);
        case 2: return launch_gemm<2, false, K2>(a, ncb, zitems, st);
        case 4: return launch_gemm<4, false, K2>(a, ncb, zitems, st);
        case 8: return launch_gemm<8, false, K2>(a, ncb, zitems, st);
        default: return (int)cudaErrorInvalidValue;
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// Launchers

// Dynamic shared memory of one block of the GEMM for tile MT (1, 2, 4, 8) and K; 0 for another MT.
GB_EXPORT int gb_nmm_smem_bytes(int mt, int K) {
    switch (mt) {
        case 1: return smem_bytes<1>(K);
        case 2: return smem_bytes<2>(K);
        case 4: return smem_bytes<4>(K);
        case 8: return smem_bytes<8>(K);
        default: return 0;
    }
}

// Router logits [rows, 512] of h [rows, K] (fp32) against gate [512, K] (bf16) on tensor cores (hi / lo bf16 split of h; see nmm_router).
GB_EXPORT int gb_nmm_router(const void* h, const void* gate, void* logits, int rows, int K, void* stream) {
    if (rows < 1 || K % kRK != 0 || !h || !gate || !logits) return (int)cudaErrorInvalidValue;
    static bool attr_set = false;
    if (!attr_set) {
        const cudaError_t e = cudaFuncSetAttribute(nmm_router, cudaFuncAttributeMaxDynamicSharedMemorySize, kRouterSmem);
        if (e != cudaSuccess) return (int)e;
        attr_set = true;
    }
    nmm_router<<<dim3(kExperts / 128, (unsigned)((rows + kRRows - 1) / kRRows)), kThreads, kRouterSmem, gb_stream(stream)>>>(
        static_cast<const float*>(h), static_cast<const uint16_t*>(gate), rows, K, static_cast<float*>(logits));
    return GB_CHECK_LAUNCH();
}

template <typename G>
static int router_f32_tiled(const void* h, const void* gate, void* logits, int rows, void* stream) {
    if (rows < 1 || !h || !gate || !logits) return (int)cudaErrorInvalidValue;
    constexpr int smem = 2 * rt_stage<G>();
    static bool attr_set = false;  // per instantiation
    if (!attr_set) {
        const cudaError_t e = cudaFuncSetAttribute(nmm_router_f32<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return (int)e;
        attr_set = true;
    }
    nmm_router_f32<G><<<dim3(kExperts / RtShape<G>::kExp, (unsigned)((rows + kRtRows - 1) / kRtRows)), rt_threads<G>(), smem,
                        gb_stream(stream)>>>(static_cast<const float*>(h), static_cast<const G*>(gate), static_cast<float*>(logits),
                                             rows);
    return GB_CHECK_LAUNCH();
}

// Router logits [rows, 512] of h [rows, 4096] (fp32) against gate [512, 4096] (bf16), fp32, with the bits of the decode path's
// gb_nm2_router_logits (nmm_router_f32).
GB_EXPORT int gb_nmm_router_f32(const void* h, const void* gate, void* logits, int rows, void* stream) {
    return router_f32_tiled<uint16_t>(h, gate, logits, rows, stream);
}

// The same with an f32 gate [512, 4096] (an F16 gate widened exactly): the bits of gb_nm2_router_logits_g32.
GB_EXPORT int gb_nmm_router_f32_g32(const void* h, const void* gate, void* logits, int rows, void* stream) {
    return router_f32_tiled<float>(h, gate, logits, rows, stream);
}

// The same with an F16 gate as stored: the bits of gb_nmm_router_f32_g32 on the widened gate.
GB_EXPORT int gb_nmm_router_f32_g16(const void* h, const void* gate, void* logits, int rows, void* stream) {
    return router_f32_tiled<__half>(h, gate, logits, rows, stream);
}

template <typename G>
static int router_g32_tc(const void* h, const void* gate, void* logits, int rows, int K, void* stream) {
    if (rows < 1 || K % kRK != 0 || !h || !gate || !logits) return (int)cudaErrorInvalidValue;
    static bool attr_set = false;  // per instantiation
    if (!attr_set) {
        const cudaError_t e = cudaFuncSetAttribute(nmm_router_g32<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, kRouterSmemG32);
        if (e != cudaSuccess) return (int)e;
        attr_set = true;
    }
    nmm_router_g32<G><<<dim3(kExperts / 128, (unsigned)((rows + kRRows - 1) / kRRows)), kThreads, kRouterSmemG32, gb_stream(stream)>>>(
        static_cast<const float*>(h), static_cast<const G*>(gate), rows, K, static_cast<float*>(logits));
    return GB_CHECK_LAUNCH();
}

// Router logits [rows, 512] of h [rows, K] (fp32) against an f32 gate [512, K] on tensor cores (hi / lo bf16 splits of h and of the gate;
// see nmm_router_g32).
GB_EXPORT int gb_nmm_router_g32(const void* h, const void* gate, void* logits, int rows, int K, void* stream) {
    return router_g32_tc<float>(h, gate, logits, rows, K, stream);
}

// The same with an F16 gate as stored (its values widened exactly before the split): the bits of gb_nmm_router_g32 on the widened gate.
GB_EXPORT int gb_nmm_router_g16(const void* h, const void* gate, void* logits, int rows, int K, void* stream) {
    return router_g32_tc<__half>(h, gate, logits, rows, K, stream);
}

// A MoE layer's RMSNorm (x f32 [rows, 4096], bf16 weight w, eps) into xn f32 and, in the same pass, the ScaledRows inputs of the dense
// entries tab1 (fc1: o1, i1) and tab2 (shared up: o2, i2), both fast: the bits of gb_nemo_glue_rmsnorm then two gb_nmm_rotate (lane S5).
GB_EXPORT int gb_nmm_norm_rot(const void* x, const void* w, float eps, void* xn, const void* tab1, const void* tab2, void* o1, void* i1,
                              void* o2, void* i2, int rows, int bf, void* stream) {
    if (rows < 1 || !x || !w || !xn || !tab1 || !tab2 || !o1 || !i1 || !o2 || !i2) return (int)cudaErrorInvalidValue;
    if (bf)
        nmm_norm_rot<true><<<(unsigned)rows, 256, 0, gb_stream(stream)>>>(
            static_cast<const float*>(x), static_cast<const uint16_t*>(w), eps, static_cast<float*>(xn), static_cast<const PfProj*>(tab1),
            static_cast<const PfProj*>(tab2), static_cast<uint16_t*>(o1), static_cast<float*>(i1), static_cast<uint16_t*>(o2),
            static_cast<float*>(i2));
    else
        nmm_norm_rot<false><<<(unsigned)rows, 256, 0, gb_stream(stream)>>>(
            static_cast<const float*>(x), static_cast<const uint16_t*>(w), eps, static_cast<float*>(xn), static_cast<const PfProj*>(tab1),
            static_cast<const PfProj*>(tab2), static_cast<uint16_t*>(o1), static_cast<float*>(i1), static_cast<uint16_t*>(o2),
            static_cast<float*>(i2));
    return GB_CHECK_LAUNCH();
}

// Work items of the fast experts, at most `cap` rows each, into work (three ints an item, room for 512 + ceil(P / cap)), the count to
// nwork[0].
GB_EXPORT int gb_nmm_items(const void* cnt, const void* tab_up, const void* tab_down, int cap, void* work, void* nwork, void* stream) {
    if (cap < 1 || !cnt || !tab_up || !tab_down || !work || !nwork) return (int)cudaErrorInvalidValue;
    nmm_items<<<1, kExperts, 0, gb_stream(stream)>>>(static_cast<const int*>(cnt), static_cast<const PfProj*>(tab_up),
                                                     static_cast<const PfProj*>(tab_down), cap, static_cast<int*>(work),
                                                     static_cast<int*>(nwork));
    return GB_CHECK_LAUNCH();
}

// The input side of `rows` rows of a projection to ScaledRows (K % 128 == 0): table entry pick[row] (entry 0 when pick is null), source
// row row / div. bf: bf16 values instead of fp16.
GB_EXPORT int gb_nmm_rotate(const void* src, int ksrc, int div, const void* pick, const void* tab, int rows, int K, void* out16,
                            void* oinv, int bf, void* stream) {
    if (rows < 1 || div < 1 || K % 128 != 0 || ksrc < K || !src || !tab || !out16 || !oinv) return (int)cudaErrorInvalidValue;
    const long units = (long)rows * (K / 128);
    const unsigned blocks = (unsigned)((units + 7) / 8);
    if (bf)
        nmm_rotate<true><<<blocks, 256, 0, gb_stream(stream)>>>(static_cast<const float*>(src), ksrc, div, static_cast<const int*>(pick),
                                                                static_cast<const PfProj*>(tab), rows, K, static_cast<uint16_t*>(out16),
                                                                static_cast<float*>(oinv));
    else
        nmm_rotate<false><<<blocks, 256, 0, gb_stream(stream)>>>(static_cast<const float*>(src), ksrc, div, static_cast<const int*>(pick),
                                                                 static_cast<const PfProj*>(tab), rows, K, static_cast<uint16_t*>(out16),
                                                                 static_cast<float*>(oinv));
    return GB_CHECK_LAUNCH();
}

// A fast projection on tensor cores over ScaledRows input `a16`/`ainv` (row id = pair id, or row number for a dense projection):
// `zitems` work items (work != null, at most 16 mt rows each) or ceil(dense_rows / (16 mt)) row chunks of table entry 0; column blocks
// nb0 .. nb0 + ncb - 1 (of 128 columns), one block each. ntab != null: the output is the next projection's ScaledRows input (out16, oinv), else fp32
// (out). relu2 and add (dense, [rows, N]) as pf_gemm4. mt: 1, 2, 4 or 8 (16 to 128 rows an item); bf: bf16 operands.
GB_EXPORT int gb_nmm_gemm(const void* a16, const void* ainv, const void* lists, int lcap, const void* work, const void* nwork,
                          int dense_rows, int zitems, int mt, const void* tab, const void* ntab, int K, int N, int nb0, int ncb, void* out,
                          void* out16, void* oinv, int relu2, const void* add, int bf, void* stream) {
    if (zitems < 1 || K % 128 != 0 || N % 128 != 0 || nb0 < 0 || ncb < 1 || (nb0 + ncb) * 128 > N || !a16 || !ainv || !tab ||
        (work && !nwork) || (!work && dense_rows < 1) || (ntab ? (!out16 || !oinv) : !out))
        return (int)cudaErrorInvalidValue;
    MmaArgs a;
    a.a = static_cast<const uint16_t*>(a16);
    a.ainv = static_cast<const float*>(ainv);
    a.lists = static_cast<const int*>(lists);
    a.lcap = lcap;
    a.work = static_cast<const int*>(work);
    a.nwork = static_cast<const int*>(nwork);
    a.dense_rows = dense_rows;
    a.tab = static_cast<const PfProj*>(tab);
    a.ntab = static_cast<const PfProj*>(ntab);
    a.K = K;
    a.N = N;
    a.nb0 = nb0;
    a.out = static_cast<float*>(out);
    a.out16 = static_cast<uint16_t*>(out16);
    a.oinv = static_cast<float*>(oinv);
    a.relu2 = relu2;
    a.add = static_cast<const float*>(add);
    a.add2 = nullptr;
    const cudaStream_t st = gb_stream(stream);
    if (gb_nmm_smem_bytes(mt, K) > 101376) return (int)cudaErrorInvalidValue;
    switch (mt * 2 + (bf ? 1 : 0)) {
        case 2: return launch_gemm<1, false>(a, ncb, zitems, st);
        case 3: return launch_gemm<1, true>(a, ncb, zitems, st);
        case 4: return launch_gemm<2, false>(a, ncb, zitems, st);
        case 5: return launch_gemm<2, true>(a, ncb, zitems, st);
        case 8: return launch_gemm<4, false>(a, ncb, zitems, st);
        case 9: return launch_gemm<4, true>(a, ncb, zitems, st);
        case 16: return launch_gemm<8, false>(a, ncb, zitems, st);
        case 17: return launch_gemm<8, true>(a, ncb, zitems, st);
        default: return (int)cudaErrorInvalidValue;
    }
}

// gb_nmm_gemm for the items whose entry is Mul1 at k2 / 2 bits (k2 = 4, 6, .., 16; the others' items exit): a MixedK table takes one
// such launch a width it holds. fp16 operands. The arguments otherwise gb_nmm_gemm's.
GB_EXPORT int gb_nmm_gemm_k(const void* a16, const void* ainv, const void* lists, int lcap, const void* work, const void* nwork,
                            int dense_rows, int zitems, int mt, const void* tab, const void* ntab, int K, int N, int nb0, int ncb, void* out,
                            void* out16, void* oinv, int relu2, const void* add, int k2, void* stream) {
    if (zitems < 1 || K % 128 != 0 || N % 128 != 0 || nb0 < 0 || ncb < 1 || (nb0 + ncb) * 128 > N || !a16 || !ainv || !tab ||
        (work && !nwork) || (!work && dense_rows < 1) || (ntab ? (!out16 || !oinv) : !out))
        return (int)cudaErrorInvalidValue;
    MmaArgs a;
    a.a = static_cast<const uint16_t*>(a16);
    a.ainv = static_cast<const float*>(ainv);
    a.lists = static_cast<const int*>(lists);
    a.lcap = lcap;
    a.work = static_cast<const int*>(work);
    a.nwork = static_cast<const int*>(nwork);
    a.dense_rows = dense_rows;
    a.tab = static_cast<const PfProj*>(tab);
    a.ntab = static_cast<const PfProj*>(ntab);
    a.K = K;
    a.N = N;
    a.nb0 = nb0;
    a.out = static_cast<float*>(out);
    a.out16 = static_cast<uint16_t*>(out16);
    a.oinv = static_cast<float*>(oinv);
    a.relu2 = relu2;
    a.add = static_cast<const float*>(add);
    a.add2 = nullptr;
    const cudaStream_t st = gb_stream(stream);
    switch (k2) {
        case 4: return launch_gemm_k2<4>(a, mt, ncb, zitems, st);
        case 6: return launch_gemm_k2<6>(a, mt, ncb, zitems, st);
        case 8: return launch_gemm_k2<8>(a, mt, ncb, zitems, st);
        case 10: return launch_gemm_k2<10>(a, mt, ncb, zitems, st);
        case 12: return launch_gemm_k2<12>(a, mt, ncb, zitems, st);
        case 14: return launch_gemm_k2<14>(a, mt, ncb, zitems, st);
        case 16: return launch_gemm_k2<16>(a, mt, ncb, zitems, st);
        default: return (int)cudaErrorInvalidValue;
    }
}

// gb_nmm_items for a MixedK table: the fast experts are those whose up and down are any-width (Mul1, 2 to 8 bits).
GB_EXPORT int gb_nmm_items_k(const void* cnt, const void* tab_up, const void* tab_down, int cap, void* work, void* nwork, void* stream) {
    if (cap < 1 || !cnt || !tab_up || !tab_down || !work || !nwork) return (int)cudaErrorInvalidValue;
    nmm_items<true><<<1, kExperts, 0, gb_stream(stream)>>>(static_cast<const int*>(cnt), static_cast<const PfProj*>(tab_up),
                                                           static_cast<const PfProj*>(tab_down), cap, static_cast<int*>(work),
                                                           static_cast<int*>(nwork));
    return GB_CHECK_LAUNCH();
}

// gb_nmm_rotate for a MixedK table: the rows of any-width fast entries (fp16 ScaledRows; bf16 operands are not taken here).
GB_EXPORT int gb_nmm_rotate_k(const void* src, int ksrc, int div, const void* pick, const void* tab, int rows, int K, void* out16, void* oinv,
                              void* stream) {
    if (rows < 1 || div < 1 || K % 128 != 0 || ksrc < K || !src || !tab || !out16 || !oinv) return (int)cudaErrorInvalidValue;
    const long units = (long)rows * (K / 128);
    nmm_rotate<false, true><<<(unsigned)((units + 7) / 8), 256, 0, gb_stream(stream)>>>(
        static_cast<const float*>(src), ksrc, div, static_cast<const int*>(pick), static_cast<const PfProj*>(tab), rows, K,
        static_cast<uint16_t*>(out16), static_cast<float*>(oinv));
    return GB_CHECK_LAUNCH();
}

// gb_nmm_gemm for a dense projection with a second addend after `add` (lane S5): out = ((output side) + add) + add2, add2 may be out
// itself (the residual stream updated in place: (fc2 + shared) + x, bit for bit x + mix since IEEE addition commutes).
GB_EXPORT int gb_nmm_gemm_add2(const void* add2, const void* a16, const void* ainv, const void* lists, int lcap, const void* work, const void* nwork,
                          int dense_rows, int zitems, int mt, const void* tab, const void* ntab, int K, int N, int nb0, int ncb, void* out,
                          void* out16, void* oinv, int relu2, const void* add, int bf, void* stream) {
    if ((add2 && work) ||
        zitems < 1 || K % 128 != 0 || N % 128 != 0 || nb0 < 0 || ncb < 1 || (nb0 + ncb) * 128 > N || !a16 || !ainv || !tab ||
        (work && !nwork) || (!work && dense_rows < 1) || (ntab ? (!out16 || !oinv) : !out))
        return (int)cudaErrorInvalidValue;
    MmaArgs a;
    a.a = static_cast<const uint16_t*>(a16);
    a.ainv = static_cast<const float*>(ainv);
    a.lists = static_cast<const int*>(lists);
    a.lcap = lcap;
    a.work = static_cast<const int*>(work);
    a.nwork = static_cast<const int*>(nwork);
    a.dense_rows = dense_rows;
    a.tab = static_cast<const PfProj*>(tab);
    a.ntab = static_cast<const PfProj*>(ntab);
    a.K = K;
    a.N = N;
    a.nb0 = nb0;
    a.out = static_cast<float*>(out);
    a.out16 = static_cast<uint16_t*>(out16);
    a.oinv = static_cast<float*>(oinv);
    a.relu2 = relu2;
    a.add = static_cast<const float*>(add);
    a.add2 = static_cast<const float*>(add2);
    const cudaStream_t st = gb_stream(stream);
    if (gb_nmm_smem_bytes(mt, K) > 101376) return (int)cudaErrorInvalidValue;
    switch (mt * 2 + (bf ? 1 : 0)) {
        case 2: return launch_gemm<1, false>(a, ncb, zitems, st);
        case 3: return launch_gemm<1, true>(a, ncb, zitems, st);
        case 4: return launch_gemm<2, false>(a, ncb, zitems, st);
        case 5: return launch_gemm<2, true>(a, ncb, zitems, st);
        case 8: return launch_gemm<4, false>(a, ncb, zitems, st);
        case 9: return launch_gemm<4, true>(a, ncb, zitems, st);
        case 16: return launch_gemm<8, false>(a, ncb, zitems, st);
        case 17: return launch_gemm<8, true>(a, ncb, zitems, st);
        default: return (int)cudaErrorInvalidValue;
    }
}

// An fp16 dense projection (table entry tab, fmt 3: w [N][K] fp16) of `rows` ScaledRows rows a16 / ainv on tensor cores (nmm_dense_h16,
// the P2 arithmetic of a plain weight): out [rows][N] fp32 = relu2? then + add, then + add2 (add2 may be out). mt: 1, 2, 4 or 8 (8 only
// where its shared memory fits; the bits do not depend on mt). K % 128 == 0, N % 128 == 0.
GB_EXPORT int gb_nmm_dense_h16(const void* a16, const void* ainv, int rows, int mt, const void* tab, int K, int N, void* out, int relu2,
                               const void* add, const void* add2, void* stream) {
    if (rows < 1 || K % 128 != 0 || N % 128 != 0 || !a16 || !ainv || !tab || !out) return (int)cudaErrorInvalidValue;
    H16Args a;
    a.a = static_cast<const uint16_t*>(a16);
    a.ainv = static_cast<const float*>(ainv);
    a.rows = rows;
    a.tab = static_cast<const PfProj*>(tab);
    a.K = K;
    a.N = N;
    a.out = static_cast<float*>(out);
    a.relu2 = relu2;
    a.add = static_cast<const float*>(add);
    a.add2 = static_cast<const float*>(add2);
    const cudaStream_t st = gb_stream(stream);
    switch (mt) {
        case 1: return launch_h16<1>(a, st);
        case 2: return launch_h16<2>(a, st);
        case 4: return launch_h16<4>(a, st);
        case 8: return launch_h16<8>(a, st);
        default: return (int)cudaErrorInvalidValue;
    }
}

// Dynamic shared memory of gb_nmm_dense_h16 at tile mt and K (0 for another mt): above 101,376 bytes the launch is refused.
GB_EXPORT int gb_nmm_dense_h16_smem(int mt, int K) {
    switch (mt) {
        case 1: return h16_smem_bytes<1>(K);
        case 2: return h16_smem_bytes<2>(K);
        case 4: return h16_smem_bytes<4>(K);
        case 8: return h16_smem_bytes<8>(K);
        default: return 0;
    }
}

// Plain fp32 rows (src [rows][ksrc], the first K values of each) to ScaledRows out16 [rows][K] / oinv [rows][K / 128] (fp16 values,
// no rotation): the input side of an fp16 dense projection under P2.
GB_EXPORT int gb_nmm_plain_rows(const void* src, int ksrc, int rows, int K, void* out16, void* oinv, void* stream) {
    if (rows < 1 || K % 128 != 0 || ksrc < K || !src || !out16 || !oinv) return (int)cudaErrorInvalidValue;
    const long units = (long)rows * (K / 128);
    nmm_plain_rows<<<(unsigned)((units + 7) / 8), 256, 0, gb_stream(stream)>>>(static_cast<const float*>(src), ksrc, rows, K,
                                                                            static_cast<uint16_t*>(out16), static_cast<float*>(oinv));
    return GB_CHECK_LAUNCH();
}
