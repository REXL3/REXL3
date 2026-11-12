// gb-nvcc: --expt-relaxed-constexpr
// Nemotron 3.5 Super VL prefill: the dense EXL3 linears on tensor cores, any slice width (1 to 4,096+ rows; lane S3, contract "P2").
// docs/NEMOTRON_PREFILL_DENSE_MMA.md has the design, the gates and the numbers.
//
// y [M, N] = ((x * suh) H_K . W_q . H_N) * svh (+ bias) (relu2) (+ add), H = 128-block Sylvester Hadamard / sqrt(128).
// Used for the Mamba in_proj / out_proj, the attention q / k / v / o, and the MoE block's fc1, fc2 and shared expert up / down
// (every one a 4-bit Mul1 trellis in the pack). Default OFF: the caller opts in for prefill slices only, at every width
// (crates/cuda/src/nemotron_dense_mma.rs); decode and speculative-verify windows keep their kernels and their bits.
//
// THE ARITHMETIC CONTRACT (P2, accepted by the owner for prefill):
//  (a) row independence: a row's output bits depend on that row's input and the weights only, never on M, on the other rows, on the
//      row's index in the call, on the block it lands in or on padding. mma.sync keeps rows apart; every row of every M tile runs the
//      same instruction sequence; rows past M are zero-filled in shared memory and their outputs dropped.
//  (b) the K reduction of an output is ONE chain of m16n8k16 MMAs over the k tiles kt = 0, 1, .., K/16 - 1 in order, starting from
//      +0.0 in fp32. Nothing in it depends on M, N, the tile shape (BM), the grid or the rank: no split K, no stream K, no atomics. So a
//      column cut across TP ranks (128-column strips, the unit of the output Hadamard) gives each kept column its single-rank bits.
//  (c) activations enter the MMA as fp16 (the rotated input (x * suh) H / sqrt 128, computed in fp32 and rounded to fp16 once). fp16
//      and not bf16 because the EXL3 codebook values are fp16 values that bf16 cannot hold: decoding them exactly as the decode path
//      does (exl3_linear_decode.cuh, bit for bit) forces the f16.f16 MMA. fp32 accumulate. The epilogue (H_128 of the fp32 sums,
//      * svh, + bias, relu2, + add, the next projection's rotation) is fp32 in a fixed order.
//  (d) deterministic run to run: no atomics, no data-dependent schedule.
//  The bits differ from the decode path's (gb_exl3_linear sums K in warps / splits planned from (K, N); pf_gemm4 is fp32 FFMA chains).
//
// STRUCTURE (structural ideas named after their upstream origin, ExLlamaV3 `exl3_gemm` (MIT), no code taken): the trellis tiles of a
// k chunk and the activation tile are staged through shared memory by cp.async in a multi-stage ring (upstream: "SH_STAGES" pipe 0),
// the A tile is XOR-swizzled by 16-byte chunk so ldmatrix.x4 reads are bank-conflict free (upstream: "A_SWIZZLE"), and each warp decodes
// the trellis tiles of ITS OWN columns straight into mma B fragments from shared memory (upstream: dq_dispatch into frag_b). Unlike
// upstream's dense kernel (16-row tiles, stream-K over k slices with locks), a block here owns a whole BM x 128 output tile through the
// whole K: the 8 warps each own one 16-column tile and all BM rows, so a decoded B fragment feeds 2 * BM / 16 MMAs, and the block's
// epilogue holds every one of the 128 columns the output Hadamard mixes.

#include <math.h>
#include <stdint.h>

#include "common.cuh"
#include "exl3_linear_decode.cuh"

using namespace gbx;

#ifndef NDM_ABL
#define NDM_ABL 0  // measurement switch (never defined in a build): 1 no trellis decode, 2 no A ldmatrix, 3 no MMA; outputs wrong
#endif

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

constexpr int kThreads = 256;  // 8 warps, warp w owns columns 16 w .. 16 w + 15 of the block's 128
#ifndef NDM_BK
#define NDM_BK 64
#endif
#ifndef NDM_STAGES
#define NDM_STAGES 3
#endif
constexpr int kBK = NDM_BK;    // k of one pipeline stage (kBK / 16 k tiles)
constexpr int kStages = NDM_STAGES;

__device__ __forceinline__ float load_scale(const void* p, int f32, int i) {
    return f32 ? __ldg(static_cast<const float*>(p) + i) : __half2float(__ldg(static_cast<const half*>(p) + i));
}

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// 16 bytes global -> shared, zero-filled when !valid (src must still be a readable address).
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&a)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

// a [M, K] fp16 = (x * suh) H / sqrt(128), fp32 inside, rounded to fp16 once. One warp a 128-wide block of one row.
__global__ void __launch_bounds__(128) ndm_rotate_kernel(const void* __restrict__ x, int x_dtype, long long ldx,
                                                         const void* __restrict__ suh, int suh_f32, half* __restrict__ a, int K) {
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5), row = blockIdx.x, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4];
    load4(x, x_dtype, (size_t)row * ldx + k, v);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= load_scale(suh, suh_f32, k + j);
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(a, F16, (size_t)row * K + k, v);
}

// The Mamba gated RMSNorm with out_proj's input rotation in one pass (lane S5): block (row, group of 1024 channels). First
// nemotron_ssm.cu's gated_norm_kernel, operation for operation (that file is compiled --fmad=false; here every product and sum is an _rn
// intrinsic, so this file's contraction setting cannot fuse them), the result rounded to bf16 as gated_norm stores it; then the group's 8
// blocks of 128 rotated exactly as ndm_rotate_kernel rotates out_proj's bf16 input (the same source lines, in this file): `gated` is
// never written. z: f32 (rounded to bf16 on read) or bf16.
__device__ __forceinline__ float gr_z(float z) { return __bfloat162float(__float2bfloat16_rn(z)); }
__device__ __forceinline__ float gr_z(__nv_bfloat16 z) { return __bfloat162float(z); }

template <typename TZ>
__global__ void __launch_bounds__(256) ndm_gated_rot_kernel(const __nv_bfloat16* __restrict__ y, const TZ* __restrict__ z,
                                                             const float* __restrict__ w, float eps, const void* __restrict__ suh,
                                                             int suh_f32, half* __restrict__ a) {
    constexpr int INNER = 8192, GROUP = 1024;
    __shared__ float red[8];
    __shared__ __align__(16) float gs[GROUP];
    const int row = blockIdx.x, grp = blockIdx.y, t = threadIdx.x, lane = t & 31;
    const size_t base = (size_t)row * INNER + (size_t)grp * GROUP;
    float v[4];
    float sq = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = t + 256 * k;
        const float zb = gr_z(z[base + c]);
        v[k] = __fmul_rn(__bfloat162float(y[base + c]), __fdiv_rn(zb, __fadd_rn(1.0f, expf(-zb))));
        sq = __fadd_rn(sq, __fmul_rn(v[k], v[k]));
    }
#pragma unroll
    for (int m = 16; m; m >>= 1) sq = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, m));
    if (lane == 0) red[t >> 5] = sq;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) total = __fadd_rn(total, red[i]);
    const float inv = rsqrtf(__fadd_rn(__fdiv_rn(total, (float)GROUP), eps));
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = t + 256 * k;
        gs[c] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(v[k], w[grp * GROUP + c]), inv)));
    }
    __syncthreads();
    // ndm_rotate_kernel's lines for the 128-block blk = t / 32 of this group (k: the column in the 8192-wide row)
    {
        const int k = grp * GROUP + (t >> 5) * 128 + 4 * lane;
        const float4 u = *reinterpret_cast<const float4*>(gs + (t >> 5) * 128 + 4 * lane);
        float vv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) vv[j] *= load_scale(suh, suh_f32, k + j);
        fwht128(vv, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) vv[j] *= HAD_SCALE;
        store4(a, F16, (size_t)row * INNER + k, vv);
    }
}

// The layer's RMSNorm with the next linear's input rotation in one pass (lane S5): block = one row of 4,096, 256 threads. First
// nemotron_glue.cu's rmsnorm_kernel, operation for operation (every operation there is an _rn intrinsic: the same here), the normed row
// kept in shared memory; then its 32 blocks of 128 rotated exactly as ndm_rotate_kernel rotates an f32 input (the same source lines, in
// this file). The f32 normed row is not written.
__global__ void __launch_bounds__(256) ndm_norm_rot_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w, float eps,
                                                           const void* __restrict__ suh, int suh_f32, half* __restrict__ a) {
    constexpr int D = 4096, T = 256;
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
#pragma unroll 4
    for (int c = t; c < D; c += T) {
        const float wv = __uint_as_float((uint32_t)__ldg(w + c) << 16);
        xs[c] = __fmul_rn(__fmul_rn(__ldg(xr + c), inv), wv);
    }
    __syncthreads();
    // ndm_rotate_kernel's lines, 4 blocks of 128 a warp
#pragma unroll 1
    for (int blk = t >> 5; blk < D / 128; blk += T / 32) {
        const int k = blk * 128 + 4 * lane;
        const float4 u = *reinterpret_cast<const float4*>(xs + k);
        float v[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= load_scale(suh, suh_f32, k + j);
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
        store4(a, F16, (size_t)row * D + k, v);
    }
}

struct GemmArgs {
    const half* a;          // [M, K] fp16, rotated
    const uint32_t* T;      // trellis [K/16][N/16][TW words]
    const void* svh;        // [N]
    const void* bias;       // [N] or null
    int scale_f32;          // svh / bias / next_suh are f32 (1) or fp16 (0)
    void* y;                // [M, ldy] of y_dtype at column 0 (or [M, N] fp16 rotated for the next projection when next_suh)
    int y_dtype;
    long long ldy;
    const float* add;       // [M, ld_add] f32 or null, added after relu2
    long long ld_add;
    const void* next_suh;   // the next projection's suh [N]: y becomes its rotated fp16 input
    int relu2;
    int M, K, N;
    // the Mamba in_proj split (lane S5): split_z > 0 writes columns [0, split_z) to z (rounded to the bf16 grid, f32), [split_z,
    // split_z + split_x) to xbc and the rest to dt, f32 rows of their own widths: split_inproj's bytes, without the f32 [M, N] y
    float* z;
    float* xbc;
    float* dt;
    int split_z, split_x;
    int split_bf16;  // z and xbc stored as bf16 (round to nearest even: the value every consumer reads them as), dt f32
};

// nemotron_glue.cu's round_bf16 (split_inproj's rounding of z), the same operations
__device__ __forceinline__ float split_round_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return __uint_as_float(0x7fc00000u);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(u);
}

// The block's tile: blockIdx.x walks GROUP_M M tiles fastest inside a group, then the column blocks, then the groups (the
// "threadblock swizzle" of CUTLASS / upstream's slice order idea): the CTAs resident at once share a few column blocks' trellis and
// a group's A rows, so both stay in L2 (a plain M-fastest order re-read the whole A from DRAM for every column block at 4,096 rows).
constexpr int kGroupM = 8;
__device__ __forceinline__ void tile_of(int bid, int num_m, int num_n, int& mt, int& nt) {
    const int group = kGroupM * num_n;
    const int gid = bid / group, first = gid * kGroupM;
    const int gsz = min(num_m - first, kGroupM);
    const int in = bid % group;
    mt = first + in % gsz;
    nt = in / gsz;
}

// One BM x (128 SN) output tile through the whole K (see the file comment): SN 128-column strips (the output Hadamard's unit) a block;
// warp w owns the SN 16-column tiles w SN .. w SN + SN - 1 (strip w SN / 8) and all BM rows. A strip past N (an odd strip count) is
// skipped whole by the warps that own it.
#ifndef NDM_MINB64
#define NDM_MINB64 2
#endif
template <int K2, int CB, int BM, int SN>
__global__ void __launch_bounds__(kThreads, BM == 64 ? NDM_MINB64 : 1) ndm_gemm_kernel(const GemmArgs g) {
    constexpr int TW = tile_words<K2>();                   // 32-bit words of a 16 x 16 tile
    constexpr int LW = lane_words<K2>();
    constexpr int MT = BM / 16;                            // m16 tiles of the block
    constexpr int A_BYTES = BM * kBK * 2;                  // a stage's A tile
    constexpr int TN = 8 * SN;                             // 16-column tiles of the block
    constexpr int B_WORDS = (kBK / 16) * TN * TW;          // a stage's trellis: kBK / 16 k tiles x TN column tiles
    constexpr int STAGE = A_BYTES + B_WORDS * 4;
    extern __shared__ __align__(128) unsigned char smem[];

    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int g8 = lane >> 2, t4 = lane & 3;
    const int M = g.M, K = g.K, N = g.N;
    const int strips = N / 128;
    int mt, nt;
    tile_of(blockIdx.x, (M + BM - 1) / BM, (strips + SN - 1) / SN, mt, nt);
    const int m0 = mt * BM, s0 = nt * SN;                  // first row, first strip
    const int my_strips = min(SN, strips - s0);            // strips of this block inside N
    const bool active = (warp * SN) / 8 < my_strips;       // this warp's strip exists
    const int KC = K / kBK;
    const int tiles_n = N / 16;

    auto stage_a = [&](int s) { return reinterpret_cast<half*>(smem + s * STAGE); };
    auto stage_b = [&](int s) { return reinterpret_cast<uint32_t*>(smem + s * STAGE + A_BYTES); };

    auto load_stage = [&](int s, int kc) {
        half* sa = stage_a(s);
        // A: BM rows x kBK / 8 chunks of 16 B; chunk c of row r lands at chunk c ^ (r & 7)
        constexpr int CPR = kBK / 8;
#pragma unroll
        for (int i = 0; i < BM * CPR / kThreads; ++i) {
            const int idx = i * kThreads + t, r = idx / CPR, c = idx % CPR;
            const int gr = m0 + r;
            const bool ok = gr < M;
            const half* src = g.a + (size_t)(ok ? gr : 0) * K + kc * kBK + c * 8;
            cp_async16(sa + r * kBK + ((c ^ (r & 7)) * 8), src, ok);
        }
        // B: the 4 k tiles' runs of TN tiles (contiguous in the trellis); a strip past N is not loaded (nobody reads it)
        uint32_t* sb = stage_b(s);
        constexpr int CH = B_WORDS / 4;                    // 16-byte chunks
        constexpr int CH_ROW = TN * TW / 4;                // chunks of one k tile's run
        const int ch_valid = my_strips * 8 * TW / 4;
#pragma unroll
        for (int i = 0; i < (CH + kThreads - 1) / kThreads; ++i) {
            const int c = i * kThreads + t;
            if (c < CH) {
                const int kl = c / CH_ROW, w = c % CH_ROW;
                if (w < ch_valid) {
                    const uint32_t* src = g.T + ((size_t)(kc * (kBK / 16) + kl) * tiles_n + s0 * 8) * TW + w * 4;
                    cp_async16(sb + kl * TN * TW + w * 4, src, true);
                }
            }
        }
    };

    float acc[MT][SN][2][4];
#pragma unroll
    for (int p = 0; p < MT; ++p)
#pragma unroll
        for (int j = 0; j < SN; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[p][j][h][c] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < KC) load_stage(s, s);
        cp_async_commit();
    }

    // ldmatrix.x4 addressing: lane l gives the row (l % 8) + 8 ((l / 8) % 2) of the m16 tile and the 8-column half l / 16
    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1), lc = lane >> 4;

#pragma unroll 1
    for (int kc = 0; kc < KC; ++kc) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        {
            const int nk = kc + kStages - 1;
            if (nk < KC) load_stage(nk % kStages, nk);
            cp_async_commit();
        }
        if (!active) continue;
        const half* sa = stage_a(kc % kStages);
        const uint32_t* sb = stage_b(kc % kStages);
#pragma unroll
        for (int kk = 0; kk < kBK / 16; ++kk) {
            uint32_t b[SN][2][2];
#pragma unroll
            for (int j = 0; j < SN; ++j) {
                uint32_t w[LW];
                load_lane_words<K2>(sb + (kk * TN + warp * SN + j) * TW, lane, w);
#if NDM_ABL == 1
                b[j][0][0] = w[0]; b[j][0][1] = w[1]; b[j][1][0] = w[0] ^ w[1]; b[j][1][1] = w[1] + 1u;
#else
                decode_lane<K2, CB>(w, lane, b[j][0], b[j][1]);
#endif
            }
#pragma unroll
            for (int p = 0; p < MT; ++p) {
                uint32_t a[4];
#if NDM_ABL == 2
                a[0] = b[0][0][0] + p; a[1] = b[0][0][1]; a[2] = b[0][1][0]; a[3] = b[0][1][1] ^ p;
#else
                const int r = p * 16 + lr, c = kk * 2 + lc;
                ldsm_x4(a, sa + r * kBK + ((c ^ (r & 7)) * 8));
#endif
#pragma unroll
                for (int j = 0; j < SN; ++j) {
#if NDM_ABL == 3
                    acc[p][j][0][0] += __uint_as_float(a[0] ^ b[j][0][0]);
                    acc[p][j][1][1] += __uint_as_float(a[1] ^ b[j][1][1]);
#else
                    mma16816(acc[p][j][0], a, b[j][0]);
                    mma16816(acc[p][j][1], a, b[j][1]);
#endif
                }
            }
        }
    }
    cp_async_wait<0>();

    // Epilogue, one strip at a time: the fp32 sums of the BM x 128 strip through shared memory, then per row (one warp, 4 columns a
    // lane): H_128, * 1/sqrt 128, * svh, + bias, relu2, + add; or the next projection's rotation.
    constexpr int LDS = 128 + 4;
    float* stg = reinterpret_cast<float*>(smem);
    const int rows = min(BM, M - m0);
#pragma unroll 1
    for (int s = 0; s < my_strips; ++s) {
        __syncthreads();                                   // the pipeline's or the previous strip's use of smem is over
        if ((warp * SN) / 8 == s) {
#pragma unroll
            for (int p = 0; p < MT; ++p)
#pragma unroll
                for (int j = 0; j < SN; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = ((warp * SN + j) % 8) * 16 + h * 8 + 2 * t4;
                        const int r = p * 16 + g8;
                        *reinterpret_cast<float2*>(stg + r * LDS + col) = make_float2(acc[p][j][h][0], acc[p][j][h][1]);
                        *reinterpret_cast<float2*>(stg + (r + 8) * LDS + col) = make_float2(acc[p][j][h][2], acc[p][j][h][3]);
                    }
        }
        __syncthreads();

        const int n = (s0 + s) * 128 + 4 * lane;
        float sv[4], bv[4] = {0.f, 0.f, 0.f, 0.f}, ns[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sv[j] = load_scale(g.svh, g.scale_f32, n + j);
            if (g.bias) bv[j] = load_scale(g.bias, g.scale_f32, n + j);
            if (g.next_suh) ns[j] = load_scale(g.next_suh, g.scale_f32, n + j);
        }
        for (int r = warp; r < rows; r += kThreads / 32) {
            const float4 u = *reinterpret_cast<const float4*>(stg + r * LDS + 4 * lane);
            float v[4] = {u.x, u.y, u.z, u.w};
            fwht128(v, lane);
            float ad[4] = {0.f, 0.f, 0.f, 0.f};
            if (g.add) {
                const float4 q = *reinterpret_cast<const float4*>(g.add + (size_t)(m0 + r) * g.ld_add + n);
                ad[0] = q.x; ad[1] = q.y; ad[2] = q.z; ad[3] = q.w;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float x = (v[j] * HAD_SCALE) * sv[j];
                if (g.bias) x = x + bv[j];
                if (g.relu2) {
                    const float q = fmaxf(x, 0.f);
                    x = q * q;
                }
                if (g.add) x = x + ad[j];
                v[j] = x;
            }
            if (g.next_suh) {
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] *= ns[j];
                fwht128(v, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
                store4(g.y, F16, (size_t)(m0 + r) * N + n, v);
            } else if (g.split_z > 0) {
                const size_t row = (size_t)(m0 + r);
                if (n < g.split_z) {
                    if (g.split_bf16) {
                        store4(g.z, BF16, row * g.split_z + n, v);
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; ++j) v[j] = split_round_bf16(v[j]);
                        store4(g.z, F32, row * g.split_z + n, v);
                    }
                } else if (n < g.split_z + g.split_x) {
                    store4(g.xbc, g.split_bf16 ? BF16 : F32, row * g.split_x + (n - g.split_z), v);
                } else {
                    const int wd = N - g.split_z - g.split_x;
                    store4(g.dt, F32, row * wd + (n - g.split_z - g.split_x), v);
                }
            } else {
                store4(g.y, g.y_dtype, (size_t)(m0 + r) * g.ldy + n, v);
            }
        }
    }
}

template <int K2, int BM, int SN>
constexpr size_t smem_bytes() {
    constexpr size_t stage = (size_t)BM * kBK * 2 + (size_t)(kBK / 16) * 8 * SN * tile_words<K2>() * 4;
    constexpr size_t pipe = kStages * stage;
    constexpr size_t epi = (size_t)BM * (128 + 4) * 4;
    return pipe > epi ? pipe : epi;
}

template <int K2, int CB, int BM, int SN>
int launch_gemm(const GemmArgs& g, cudaStream_t st) {
    constexpr size_t sm = smem_bytes<K2, BM, SN>();
    static_assert(sm <= 101376, "a block's shared memory must fit the 99 KB an SM gives one");
    static bool attr = false;  // idempotent; a race between host threads sets the same value twice
    if (!attr) {
        const cudaError_t e = cudaFuncSetAttribute(ndm_gemm_kernel<K2, CB, BM, SN>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        if (e != cudaSuccess) return (int)e;
        attr = true;
    }
    const int strips = g.N / 128;
    const unsigned blocks = (unsigned)(((g.M + BM - 1) / BM) * ((strips + SN - 1) / SN));
    ndm_gemm_kernel<K2, CB, BM, SN><<<blocks, kThreads, sm, st>>>(g);
    return GB_CHECK_LAUNCH();
}

// The tile shape (never changes a bit: see (b)). 0: pick; else BM * 10 + SN (641, 642, 1281, 1282).
// The pick (MEASURED on GB10 over the ten Nemotron shapes at 256 to 4,096 rows, docs/NEMOTRON_PREFILL_DENSE_MMA.md): 64 rows by 2
// strips (two blocks resident a SM) whenever that gives at least two waves of resident blocks, else 64 rows by 1 strip (twice the
// blocks: a narrow layer or a short slice). 128-row tiles were never more than 2 % faster and up to 15 % slower.
int pick_tile(int m, int n, int force, int& bm, int& sn) {
    if (force) {
        bm = force / 10;
        sn = force % 10;
        return (bm == 64 || bm == 128) && (sn == 1 || sn == 2) ? 0 : (int)cudaErrorInvalidValue;
    }
    const int strips = n / 128, sms = 48;
    const long long blocks2 = (long long)((m + 63) / 64) * ((strips + 1) / 2);
    bm = 64;
    sn = blocks2 >= 2 * (2 * sms) ? 2 : 1;
    return 0;
}

int dispatch(int bits, int cb, int bm, int sn, const GemmArgs& g, cudaStream_t st) {
#define NDM_T(K2_, CB_)                                                                                                    \
    if (bm == 64) return sn == 1 ? launch_gemm<K2_, CB_, 64, 1>(g, st) : launch_gemm<K2_, CB_, 64, 2>(g, st); \
    return sn == 1 ? launch_gemm<K2_, CB_, 128, 1>(g, st) : launch_gemm<K2_, CB_, 128, 2>(g, st);
    if (bits == 4) {
        if (cb == CB_MUL1) { NDM_T(8, CB_MUL1) }
        if (cb == CB_MCG) { NDM_T(8, CB_MCG) }
        if (cb == CB_3INST) { NDM_T(8, CB_3INST) }
        return (int)cudaErrorInvalidValue;
    }
    // the other whole widths (v0.1.2 lane K, P1-6: a MixedK pack's Mamba and attention linears at 5 to 8 bits), the Mul1 codebook
    if (cb != CB_MUL1) return (int)cudaErrorInvalidValue;
    switch (bits) {
        case 2: { NDM_T(4, CB_MUL1) }
        case 3: { NDM_T(6, CB_MUL1) }
        case 5: { NDM_T(10, CB_MUL1) }
        case 6: { NDM_T(12, CB_MUL1) }
        case 7: { NDM_T(14, CB_MUL1) }
        case 8: { NDM_T(16, CB_MUL1) }
        default: return (int)cudaErrorInvalidValue;
    }
#undef NDM_T
}

bool dtype_ok(int d) { return d >= 0 && d <= 2; }

}  // namespace

// Whether the tensor-core prefill GEMM takes a layer of this shape and format: K and N multiples of 128; 4 bits with any codebook, or
// 2, 3 and 5 to 8 bits with the Mul1 codebook (v0.1.2 P1-6; every tile shape's shared memory fits at 8 bits: 96 KB at 128 x 2).
GB_EXPORT int gb_ndm_supported(int k, int n, int bits, int codebook) {
    const bool shape = k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0;
    return shape && ((bits == 4 && codebook >= 0 && codebook <= 2) || (bits >= 2 && bits <= 8 && codebook == CB_MUL1));
}

// a [m, k] fp16 = (x * suh) H / sqrt(128); x [m, ldx] of x_dtype (F16 0, BF16 1, F32 2), suh [k] f32 (suh_f32) or fp16.
GB_EXPORT int gb_ndm_rotate(const void* x, int x_dtype, long long ldx, const void* suh, int suh_f32, void* a, int m, int k,
                            void* stream) {
    if (m < 1 || k <= 0 || k % 128 || ldx < k || !dtype_ok(x_dtype) || !x || !suh || !a) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)m, (unsigned)((k / 128 + 3) / 4));
    ndm_rotate_kernel<<<grid, 128, 0, gb_stream(stream)>>>(x, x_dtype, ldx, suh, suh_f32, static_cast<half*>(a), k);
    return GB_CHECK_LAUNCH();
}

// y = the EXL3 layer applied to the rotated rows a [m, k] fp16 (gb_ndm_rotate's output, or a previous call's next_suh output).
// y [m, ldy] of y_dtype from column 0; with next_suh y is instead [m, n] fp16, the next projection's rotated input. svh, bias,
// next_suh f32 (scale_f32) or fp16. add [m, ld_add] f32 or null (added after relu2). tile_force 0 (pick) or BM * 10 + SN (641, 642,
// 1281, 1282: 64 or 128 rows by 1 or 2 strips of 128 columns), the tile shape, which never changes a bit. Returns 0 or a cudaError_t; never allocates or synchronizes.
GB_EXPORT int gb_ndm_gemm(const void* a, const void* trellis, const void* svh, const void* bias, int scale_f32, void* y, int y_dtype,
                          long long ldy, const void* add, long long ld_add, const void* next_suh, int relu2, int m, int k, int n,
                          int bits, int codebook, int tile_force, void* stream) {
    if (m < 1 || !gb_ndm_supported(k, n, bits, codebook) || !dtype_ok(y_dtype) || !a || !trellis || !svh || !y)
        return (int)cudaErrorInvalidValue;
    if (!next_suh && ldy < n) return (int)cudaErrorInvalidValue;
    if (add && ld_add < n) return (int)cudaErrorInvalidValue;
    GemmArgs g{static_cast<const half*>(a), static_cast<const uint32_t*>(trellis), svh, bias, scale_f32, y, y_dtype, ldy,
               static_cast<const float*>(add), ld_add, next_suh, relu2, m, k, n, nullptr, nullptr, nullptr, 0, 0, 0};
    int bm = 0, sn = 0;
    const int rc = pick_tile(m, n, tile_force, bm, sn);
    if (rc) return rc;
    return dispatch(bits, codebook, bm, sn, g, gb_stream(stream));
}

// gb_ndm_gemm with the Mamba in_proj split as the output (lane S5; the bytes split_inproj writes from gb_ndm_gemm's f32 y): columns
// [0, split_z) to z [m, split_z] rounded to the bf16 grid (f32), [split_z, split_z + split_x) to xbc [m, split_x], the rest to dt
// [m, n - split_z - split_x], all f32 (zx_bf16: z and xbc as bf16 instead, the bf16 values their consumers read); no relu2, no add.
// Both cuts on 128-column strips.
GB_EXPORT int gb_ndm_gemm_split(const void* a, const void* trellis, const void* svh, const void* bias, int scale_f32, void* z, void* xbc,
                                void* dt, int split_z, int split_x, int zx_bf16, int m, int k, int n, int bits, int codebook,
                                int tile_force, void* stream) {
    if (m < 1 || !gb_ndm_supported(k, n, bits, codebook) || !a || !trellis || !svh || !z || !xbc || !dt || split_z < 128 ||
        split_x < 128 || split_z % 128 || split_x % 128 || split_z + split_x >= n)
        return (int)cudaErrorInvalidValue;
    GemmArgs g{static_cast<const half*>(a), static_cast<const uint32_t*>(trellis), svh, bias, scale_f32, z, F32, (long long)n,
               nullptr, 0, nullptr, 0, m, k, n, static_cast<float*>(z), static_cast<float*>(xbc), static_cast<float*>(dt), split_z,
               split_x, zx_bf16 ? 1 : 0};
    int bm = 0, sn = 0;
    const int rc = pick_tile(m, n, tile_force, bm, sn);
    if (rc) return rc;
    return dispatch(bits, codebook, bm, sn, g, gb_stream(stream));
}

// The Mamba gated RMSNorm of `rows` rows (y bf16 [rows, 8192], z f32 or bf16 (z_bf16) [rows, 8192], w f32 [8192]) straight into
// out_proj's rotated fp16 input a [rows, 8192] (suh: out_proj's, f32 or fp16): the bits of gated_norm then gb_ndm_rotate (lane S5).
GB_EXPORT int gb_ndm_gated_rot(const void* y, const void* z, int z_bf16, const float* w, float eps, const void* suh, int suh_f32, void* a,
                               int rows, void* stream) {
    if (rows < 1 || !y || !z || !w || !suh || !a) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)rows, 8);
    if (z_bf16)
        ndm_gated_rot_kernel<__nv_bfloat16><<<grid, 256, 0, gb_stream(stream)>>>(
            static_cast<const __nv_bfloat16*>(y), static_cast<const __nv_bfloat16*>(z), w, eps, suh, suh_f32, static_cast<half*>(a));
    else
        ndm_gated_rot_kernel<float><<<grid, 256, 0, gb_stream(stream)>>>(static_cast<const __nv_bfloat16*>(y), static_cast<const float*>(z),
                                                                         w, eps, suh, suh_f32, static_cast<half*>(a));
    return GB_CHECK_LAUNCH();
}

// RMSNorm (bf16 weight w [4096], eps) of `rows` rows of x f32 [rows, 4096] straight into the next linear's rotated fp16 input
// a [rows, 4096] (its suh, f32 or fp16): the bits of gb_nemo_glue_rmsnorm then gb_ndm_rotate (lane S5).
GB_EXPORT int gb_ndm_norm_rot(const void* x, const void* w, float eps, const void* suh, int suh_f32, void* a, int rows, void* stream) {
    if (rows < 1 || !x || !w || !suh || !a) return (int)cudaErrorInvalidValue;
    ndm_norm_rot_kernel<<<(unsigned)rows, 256, 0, gb_stream(stream)>>>(static_cast<const float*>(x), static_cast<const uint16_t*>(w), eps,
                                                                       suh, suh_f32, static_cast<half*>(a));
    return GB_CHECK_LAUNCH();
}

// The tile gb_ndm_gemm picks for (m, n) when not forced, as BM * 10 + SN (for the tables).
GB_EXPORT int gb_ndm_pick_tile(int m, int n) {
    int bm = 0, sn = 0;
    pick_tile(m, n, 0, bm, sn);
    return bm * 10 + sn;
}
