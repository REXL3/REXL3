// gb-nvcc: --fmad=false
// Ling-3.1-flash-1019 Gated-MLA under the LP2 arithmetic contract (the `ling-mla-ae-...` / `ling-mla-attn-...`
// ids of `src/ling_mla_attn.rs`): the absorb, the dense causal absorbed attention and the expand of the
// `BailingMoeV3MultiLatentAttention` on tensor cores, in place of `csrc/mla_gated.cu`'s scalar `attend_kernel`
// and skinny absorb / expand. The kernels this file replaces stay compiled (contract P1); this file is selected
// per call, never by environment.
//
// Exact Ling-3.1 geometry (see `mla_gated.rs` / `testdata/ling31-flash-1019/config.json`):
//   NH=64 heads, DN=128 (qk_nope_head_dim == v_head_dim), DR=64 (qk_rope_head_dim), DC=512 (kv_lora_rank),
//   cache row 576 T (512 normalised latent + 64 rotated rope), softmax scale 192^-0.5, full causal
//   (key j visible at query row r iff j <= row_pos[r]; row_pos < 0 is a padding row -> zeros).
//
//   absorb   q_lat[r][h][c] = sum_{d<128} q[r][h*192 + d]       * W_UK[h][d][c]      (c < 512)
//   score    s[h][j]        = (q_lat[r][h] . c_kv[j] + q_rot[r][h] . k_rot[j]) * 192^-0.5
//   attend   o_lat[r][h][c] = sum_{j<=pos} softmax_j(s[h][j]) * c_kv[j][c]
//   expand   out[r][h*128+n] = sum_{c<512} o_lat[r][h][c] * W_UV[h][c][n]            (n < 128)
//   gate     gated[r][h*128+n] = out[...] * sigmoid(x[r] . g_proj[h])   (g_proj bf16 [64][4096])
//
// Layouts are `mla_gated.cu`'s, T = bf16 only (Ling-3.1 binds every MLA activation, cache and kv_b as bf16;
// the fp16 instantiation of the scalar path stays in `mla_gated.cu`):
//   q      T [rows][q_row_stride]   head h at h*192: [0,128) nope, [128,192) rope
//   q_lat  T [rows][64][512]        (absorb output)
//   q_rot  T [rows][64][64]         (half-split rotated rope, from the shared q_rope_kernel in mla_gated.cu)
//   cache  T [n_blocks][256][576]   latent 0..511, rotated rope 512..575
//   w_uk   T [64][128][512]         kv_b_proj rows h*256 + d, c contiguous
//   w_uvt  T [64][512][128]         kv_b_proj rows h*256 + 128 + n, transposed to [c][n]
//   o_lat  T [rows][64][512]
//   out / gated  T [rows][64*128]
//
// ROW-INVARIANCE BY CONSTRUCTION, the LP2 (a)-(d) contract: a row's bits are a function of that row's
// operands, its slot's keys 0..=pos, its own position and the packed weights only. Absorb / expand tile rows
// in 16-row MMA groups but never mix two rows in a reduction (the mma M dimension is independent rows);
// the k order inside one output is a fixed chain of mma k16 steps in ascending k (absorb: d = 0..127;
// expand: c = 0..511, folded per 128-c block as glm_p2_mla.cu's fold128). The attention CTA walks one row's
// key tiles in ascending order, 32 keys a tile, with a single online-softmax state per (head, latent lane);
// there is no split-K, no atomics, no chunk-count-dependent merge tree. Run to run the same launch replays
// the same bits; a row chunked as 1+1+... or as one wide prefill row sees identical bits.
// (attn_kernel and attn_pair_kernel, the prefill kernels, stay exactly that. The DECODE kernels, attn_split_kernel + attn_combine_kernel
// below, split a row's keys into absolute 1024-key chunks and merge them in a fixed ascending order: still no atomics, and a row's bits are
// still a function of its own operands, its slot's keys 0..=pos and pos alone; see the header of that section.)
#include "common.cuh"
#include "exl3_linear_decode.cuh"

namespace {

constexpr int NH = 64;            // heads
constexpr int DC = 512;           // latent width
constexpr int DN = 128;           // qk_nope_head_dim == v_head_dim
constexpr int DR = 64;            // qk_rope_head_dim
constexpr int QK = DN + DR;       // 192
constexpr int CACHE_W = DC + DR;  // 576 elements per cache token
constexpr int PAGE = 256;         // tokens per cache block
constexpr int HIDDEN = 4096;
constexpr float SM_SCALE = 0.07216878364870322f;  // 192^-0.5
constexpr float LOG2E = 1.4426950408889634f;
constexpr float SM_SCALE2 = SM_SCALE * LOG2E;     // log2-domain softmax scale

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 16 bytes global -> shared, zero-filled without reading the source's bytes when !valid (src stays a mapped
// address, so the instruction never faults).
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
// d = a b + d (bf16 operands, fp32 accumulate), the m16n8k16 row.col form of every tensor-core op here.
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// ---------------------------------------------------------------- tensor-core absorb / expand (and fused expand+gate)
//
// `mma_kernel<K, N>`: y[r][h][j] = sum_{k<K} x[r][h*K + k] * w[h][k][j] (j < N), x bf16 with row stride xs,
// y bf16 with row stride NH*N. The 16-row / 128-column tiling is glm_p2_mla.cu's structure generalised to
// Ling's bf16 operands, K in {128 (absorb), 512 (expand)}, N in {512 (absorb), 128 (expand)}. One CTA =
// 16 rows x 128 columns of one head, 4 warps of (16 rows x 32 cols); a K pass is BK=32 in a 3-stage cp.async
// ring, padded shared rows (A 40 halves, B 136 halves: every ldmatrix's eight 16-byte rows hit distinct banks).
// The contraction is one fixed ascending k16 chain per 128-k fold block; a fold block's chain starts from +0.0
// and folds into the running total with one IEEE fp32 add (glm_p2_mla.cu's fold128, kept for the arithmetic id).
// For K = 128 (absorb) there is exactly one fold block; the code path is the same.
constexpr int BK = 32;
constexpr int STAGES = 3;
constexpr int ATHR = 128;         // absorb/expand CTA threads
constexpr int BM = 16;            // rows a CTA
constexpr int BN = 128;           // columns a CTA
constexpr int APAD = BK + 8;
constexpr int BPAD = BN + 8;
constexpr int FOLD_K = 128;

template <int K, int N, bool GATED>
__global__ void __launch_bounds__(ATHR) mma_kernel(const bf16* __restrict__ x, int64_t xs, int head_stride,
                                                   const bf16* __restrict__ w, bf16* __restrict__ y,
                                                   int rows, const bf16* __restrict__ xgate,
                                                   const bf16* __restrict__ gproj,
                                                   const half* __restrict__ dense_suh,
                                                   half* __restrict__ dense_xh) {
    static_assert(K % FOLD_K == 0 && FOLD_K % BK == 0 && N % BN == 0, "tiling");
    __shared__ __align__(16) bf16 sa[STAGES][BM * APAD];
    __shared__ __align__(16) bf16 sb[STAGES][BK * BPAD];
    // The CTA's rows staged for the fused dense rot_in (rows x 128 bf16 of head h, the same post-gate bytes
    // the global `y` store got); free after the K loop, used only when dense_suh/dense_xh are given.
    __shared__ __align__(16) bf16 sg[BM * BN];
    // gate dots, one per (row, head) of this CTA, only when GATED (expand): filled before the main loop.
    __shared__ float sgate[BM];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wn = warp;  // columns 32 wn .. 32 wn + 31 (4 warps x 32 = BN = 128)
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, h = blockIdx.z;
    const bf16* xh = x + (int64_t)h * head_stride;
    const bf16* wh = w + (int64_t)h * K * N;

    if constexpr (GATED) {
        // gate[r] = sigmoid(xgate[r] . gproj[h]) for the BM rows of this tile; 8 threads a row read the
        // 4096-wide dot as 64 coalesced 128-byte segments (one uint4 = 8 bf16 per thread per segment, stride
        // 64 halves across threads), ascending c per thread, then a fixed 8-partial tree per row (one
        // sigmoid per (row, head)). The per-thread FMA chain covers the same 512 elements in ascending order of
        // 16-byte groups (a different grouping than the old scalar stride-8 loop, same multiset and tree).
        __shared__ float red[ATHR];
        const int r_local = tid >> 3;         // 0..15 (the row of the tile)
        const int sub = tid & 7;              // 8 threads a row
        const int rr = m0 + r_local;
        float acc = 0.0f;
        {
            const bf16* xr = xgate + (int64_t)(rr < rows ? rr : 0) * HIDDEN;
            const bf16* gr2 = gproj + (int64_t)h * HIDDEN;
#pragma unroll 4
            for (int c8 = sub * 8; c8 < HIDDEN; c8 += 64) {
                const uint4 vx = __ldg(reinterpret_cast<const uint4*>(xr + c8));
                const uint4 vg = __ldg(reinterpret_cast<const uint4*>(gr2 + c8));
                const __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(&vx.x);
                const __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(&vx.y);
                const __nv_bfloat162 x45 = *reinterpret_cast<const __nv_bfloat162*>(&vx.z);
                const __nv_bfloat162 x67 = *reinterpret_cast<const __nv_bfloat162*>(&vx.w);
                const __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(&vg.x);
                const __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(&vg.y);
                const __nv_bfloat162 g45 = *reinterpret_cast<const __nv_bfloat162*>(&vg.z);
                const __nv_bfloat162 g67 = *reinterpret_cast<const __nv_bfloat162*>(&vg.w);
                acc = __fmaf_rn(__bfloat162float(x01.x), __bfloat162float(g01.x), acc);
                acc = __fmaf_rn(__bfloat162float(x01.y), __bfloat162float(g01.y), acc);
                acc = __fmaf_rn(__bfloat162float(x23.x), __bfloat162float(g23.x), acc);
                acc = __fmaf_rn(__bfloat162float(x23.y), __bfloat162float(g23.y), acc);
                acc = __fmaf_rn(__bfloat162float(x45.x), __bfloat162float(g45.x), acc);
                acc = __fmaf_rn(__bfloat162float(x45.y), __bfloat162float(g45.y), acc);
                acc = __fmaf_rn(__bfloat162float(x67.x), __bfloat162float(g67.x), acc);
                acc = __fmaf_rn(__bfloat162float(x67.y), __bfloat162float(g67.y), acc);
            }
        }
        red[tid] = acc;
        __syncthreads();
        if (sub == 0) {
            const float v0 = red[tid] + red[tid + 4];
            const float v1 = red[tid + 2] + red[tid + 6];
            const float v2 = red[tid + 1] + red[tid + 5];
            const float v3 = red[tid + 3] + red[tid + 7];
            const float v = (v0 + v1) + (v2 + v3);
            sgate[r_local] = 1.0f / (1.0f + expf(-v));
        }
        __syncthreads();
    }

    auto load = [&](int stage, int kt) {
        const int k0 = kt * BK;
        {
            // A: BM rows x BK halves = 64 pieces of 16 bytes; two threads a row (BM * 2 = 32 pieces a pass,
            // 4 passes of 128 threads... exact: BM*BK/8 = 64 pieces, so the first 64 threads each copy one).
            const int r = tid >> 2, c = tid & 3;
            if (tid < BM * (BK / 8)) {
                const int gr = m0 + r;
                const bool v = gr < rows;
                cp16(&sa[stage][r * APAD + c * 8], xh + (int64_t)(v ? gr : 0) * xs + k0 + c * 8, v);
            }
        }
#pragma unroll
        for (int i = tid; i < BK * (BN / 8); i += ATHR) {
            const int kr = i / (BN / 8), c = i % (BN / 8);
            cp16(&sb[stage][kr * BPAD + c * 8], wh + (int64_t)(k0 + kr) * N + n0 + c * 8, true);
        }
    };

    float part[4][4], tot[4][4];
#pragma unroll
    for (int nj = 0; nj < 4; ++nj)
#pragma unroll
        for (int e = 0; e < 4; ++e) part[nj][e] = 0.0f, tot[nj][e] = 0.0f;

    constexpr int KT = K / BK;
    constexpr int FOLD = FOLD_K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        wait_group<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < KT) load(nk % STAGES, nk);
            commit();
        }
        const bf16* A = sa[kt % STAGES];
        const bf16* B = sb[kt % STAGES];
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[4];
            ldmatrix4(af, A + (lane & 15) * APAD + kk + (lane >> 4) * 8);
            uint32_t bf[4][2];
#pragma unroll
            for (int nj = 0; nj < 4; nj += 2) {
                const int q = lane >> 3, j = lane & 7;
                uint32_t r[4];
                ldmatrix4t(r, B + (kk + (q & 1) * 8 + j) * BPAD + wn * 32 + nj * 8 + (q >> 1) * 8);
                bf[nj][0] = r[0];
                bf[nj][1] = r[1];
                bf[nj + 1][0] = r[2];
                bf[nj + 1][1] = r[3];
            }
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) mma(part[nj], af, bf[nj][0], bf[nj][1]);
        }
        if ((kt + 1) % FOLD == 0) {
#pragma unroll
            for (int nj = 0; nj < 4; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    tot[nj][e] = __fadd_rn(tot[nj][e], part[nj][e]);
                    part[nj][e] = 0.0f;
                }
        }
    }
    wait_group<0>();

    const int g = lane >> 2, t = lane & 3;
    const int r0 = m0 + g;
    const int col0 = wn * 32;                       // this warp's columns of the 128-block
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
        const int col = h * N + n0 + col0 + nj * 8 + 2 * t;
        float v0 = tot[nj][0], v1 = tot[nj][1], v2 = tot[nj][2], v3 = tot[nj][3];
        if constexpr (GATED) {
            // the two mma row groups are DIFFERENT rows: r0 = m0+g (rows 0..7 of the tile) and r0+8 =
            // m0+g+8 (rows 8..15). Each needs its own head gate — sgate[g] and sgate[g+8] respectively.
            const float gate_lo = sgate[g];
            const float gate_hi = sgate[g + 8];
            v0 = __fmul_rn(v0, gate_lo);
            v1 = __fmul_rn(v1, gate_lo);
            v2 = __fmul_rn(v2, gate_hi);
            v3 = __fmul_rn(v3, gate_hi);
        }
        if (r0 < rows) *reinterpret_cast<uint32_t*>(y + (int64_t)r0 * (NH * N) + col) = pack2(v0, v1);
        if (r0 + 8 < rows) *reinterpret_cast<uint32_t*>(y + (int64_t)(r0 + 8) * (NH * N) + col) = pack2(v2, v3);
        if constexpr (GATED) {
            if (dense_suh && dense_xh) {
                // stage the same rounded bytes for the fused dense rot_in below (only rows the rotation reads);
                // tile-relative row index rr = r - m0 (0..15), so any tile of any chunk width fits the BM x BN
                // staging buffer (prefill chunks stage rows past 16 this way too)
                const int cb = col & (BN - 1);
                if (r0 < rows) *reinterpret_cast<uint32_t*>(sg + (r0 - m0) * BN + cb) = pack2(v0, v1);
                if (r0 + 8 < rows) *reinterpret_cast<uint32_t*>(sg + (r0 + 8 - m0) * BN + cb) = pack2(v2, v3);
            }
        }
    }
    if constexpr (GATED) {
        // The dense projection's input rotation fused into the expand epilogue: warp w rotates the tile's rows
        // rr = w, w+4, ... (row r = m0 + rr) with the exact ling_rot_in_kernel math (suh fp16 scaling, the
        // 128-point Walsh-Hadamard butterfly in the same fixed order, HAD_SCALE, fp16 stores) into dense_xh
        // [(r * NH + h) * 128 + ...], reading the staged bf16 (the same values `y` got). Works at any chunk
        // width: the four warps cover the BM = 16 tile rows (stride 4) and rows past the call are skipped.
        if (dense_suh && dense_xh) {
            __syncthreads();
            for (int rr = warp; rr < BM; rr += 4) {
                const int r = m0 + rr;
                if (r >= rows) break;
                const int c0 = 4 * lane;
                const uint2 u = *reinterpret_cast<const uint2*>(sg + rr * BN + c0);
                const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                float v[4] = {a.x, a.y, b.x, b.y};
                const half* su = dense_suh + h * 128 + c0;
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] *= __half2float(__ldg(su + j));
                gbx::fwht128(v, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] *= gbx::HAD_SCALE;
                half* dst = dense_xh + ((size_t)r * NH + h) * 128 + c0;
                half2 p0 = __floats2half2_rn(v[0], v[1]), p1 = __floats2half2_rn(v[2], v[3]);
                *reinterpret_cast<uint2*>(dst) = make_uint2(*reinterpret_cast<const uint32_t*>(&p0),
                                                            *reinterpret_cast<const uint32_t*>(&p1));
            }
        }
    }
}

// ---------------------------------------------------------------- fused kv_prep + absorb + q_rope
//
// One launch for the three independent pre-attention steps of a Gated-MLA layer: `kv_prep` (RMSNorm + rope of
// the latent into the cache, the shared scalar kernel of mla_gated.cu), the tensor-core `absorb`
// (q_nope @ W_UK, the mma_kernel<DN, DC, false> body) and `q_rope_only` (the shared scalar rope kernel).
// Grid (gm, 4, NH + 1): z < NH runs the absorb for head h (blockIdx.y = the 128-column band; only y == 0 also
// rotates this tile's rows' rope queries, the 4 warps walking the BM = 16 rows), z == NH runs kv_prep for the
// 16 rows of the x-tile (one CTA per row, 128 of the CTA's threads each, reproducing the 256-thread reduction
// exactly: thread tid computes both red[tid] and red[tid + 128] in ascending column order). The three steps
// read disjoint inputs and write disjoint outputs and every output's arithmetic is the unfused kernel's, so
// every bit matches the three launches.
__global__ void __launch_bounds__(ATHR)
prep_absorb_rope_kernel(const bf16* __restrict__ q, int64_t qs, int head_stride,
                        const bf16* __restrict__ w, bf16* __restrict__ q_lat, int rows,
                        const bf16* __restrict__ kv, int kv_stride,
                        const bf16* __restrict__ norm_w,
                        const float* __restrict__ cos, const float* __restrict__ sin, int table_stride,
                        const int* __restrict__ table, int max_blocks,
                        const int* __restrict__ row_slot, const int* __restrict__ row_pos,
                        bf16* __restrict__ cache, bf16* __restrict__ q_rot) {
    __shared__ __align__(16) bf16 sa[STAGES][BM * APAD];
    __shared__ __align__(16) bf16 sb[STAGES][BK * BPAD];
    __shared__ float red[256];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.x * BM;
    const int h = blockIdx.z;

    if (h < NH) {
        // ---------------- tensor-core absorb: the exact mma_kernel<DN, DC, false> body ----------------
        const int wn = warp;
        const int n0 = blockIdx.y * BN;
        const bf16* xh = q + (int64_t)h * head_stride;
        const bf16* wh = w + (int64_t)h * DN * DC;

        auto load = [&](int stage, int kt) {
            const int k0 = kt * BK;
            {
                const int r = tid >> 2, c = tid & 3;
                if (tid < BM * (BK / 8)) {
                    const int gr = m0 + r;
                    const bool v = gr < rows;
                    cp16(&sa[stage][r * APAD + c * 8], xh + (int64_t)(v ? gr : 0) * qs + k0 + c * 8, v);
                }
            }
#pragma unroll
            for (int i = tid; i < BK * (BN / 8); i += ATHR) {
                const int kr = i / (BN / 8), c = i % (BN / 8);
                cp16(&sb[stage][kr * BPAD + c * 8], wh + (int64_t)(k0 + kr) * DC + n0 + c * 8, true);
            }
        };

        float part[4][4], tot[4][4];
#pragma unroll
        for (int nj = 0; nj < 4; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) part[nj][e] = 0.0f, tot[nj][e] = 0.0f;

        constexpr int KT = DN / BK;
        constexpr int FOLD = FOLD_K / BK;
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < KT) load(s, s);
            commit();
        }
        for (int kt = 0; kt < KT; ++kt) {
            wait_group<STAGES - 2>();
            __syncthreads();
            {
                const int nk = kt + STAGES - 1;
                if (nk < KT) load(nk % STAGES, nk);
                commit();
            }
            const bf16* A = sa[kt % STAGES];
            const bf16* B = sb[kt % STAGES];
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                uint32_t af[4];
                ldmatrix4(af, A + (lane & 15) * APAD + kk + (lane >> 4) * 8);
                uint32_t bf[4][2];
#pragma unroll
                for (int nj = 0; nj < 4; nj += 2) {
                    const int qq = lane >> 3, j = lane & 7;
                    uint32_t r[4];
                    ldmatrix4t(r, B + (kk + (qq & 1) * 8 + j) * BPAD + wn * 32 + nj * 8 + (qq >> 1) * 8);
                    bf[nj][0] = r[0];
                    bf[nj][1] = r[1];
                    bf[nj + 1][0] = r[2];
                    bf[nj + 1][1] = r[3];
                }
#pragma unroll
                for (int nj = 0; nj < 4; ++nj) mma(part[nj], af, bf[nj][0], bf[nj][1]);
            }
            if ((kt + 1) % FOLD == 0) {
#pragma unroll
                for (int nj = 0; nj < 4; ++nj)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[nj][e] = __fadd_rn(tot[nj][e], part[nj][e]);
                        part[nj][e] = 0.0f;
                    }
            }
        }
        wait_group<0>();

        const int g = lane >> 2, t = lane & 3;
        const int r0 = m0 + g;
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
            const int col = h * DC + n0 + wn * 32 + nj * 8 + 2 * t;
            const float v0 = tot[nj][0], v1 = tot[nj][1], v2 = tot[nj][2], v3 = tot[nj][3];
            if (r0 < rows) *reinterpret_cast<uint32_t*>(q_lat + (int64_t)r0 * (NH * DC) + col) = pack2(v0, v1);
            if (r0 + 8 < rows) *reinterpret_cast<uint32_t*>(q_lat + (int64_t)(r0 + 8) * (NH * DC) + col) = pack2(v2, v3);
        }

        // ---------------- q_rope: mla_gated.cu's q_rope_kernel<bf16>, for this tile's rows ----------------
        // (DR / 2 = 32 pairs; the 32 lanes are i = 0..31, the half-split writes cover all 64 columns.)
        if (blockIdx.y == 0) {
            for (int rr = warp; rr < BM; rr += 4) {
                const int r = m0 + rr;
                if (r >= rows) continue;
                const bf16* qh = q + (int64_t)r * qs + (int64_t)h * head_stride + DN;
                bf16* dst = q_rot + ((int64_t)r * NH + h) * DR;
                const int pos = row_pos[r];
                if (pos < 0) {
                    dst[lane] = f2bf(0.0f);
                    dst[DR / 2 + lane] = f2bf(0.0f);
                    continue;
                }
                const float* cs = cos + (size_t)pos * table_stride;
                const float* sn = sin + (size_t)pos * table_stride;
                const float a = bf2f(qh[2 * lane]);
                const float b = bf2f(qh[2 * lane + 1]);
                const float ci = cs[lane], si = sn[lane];
                dst[lane] = f2bf(__fsub_rn(__fmul_rn(a, ci), __fmul_rn(b, si)));
                dst[DR / 2 + lane] = f2bf(__fadd_rn(__fmul_rn(b, ci), __fmul_rn(a, si)));
            }
        }
        return;
    }

    // ---------------- kv_prep: mla_gated.cu's kv_prep_kernel<bf16> with 128 threads ----------------
    // 4 CTAs (blockIdx.y in 0..3) each walk every 4th of this tile's BM = 16 rows with their 128 threads.
    for (int rr = blockIdx.y; rr < BM; rr += 4) {
        const int r = m0 + rr;
        if (r >= rows) break;
        const int pos = row_pos[r];
        if (pos < 0) continue;
        const int blk = table[(int64_t)row_slot[r] * max_blocks + pos / PAGE];
        const bf16* src = kv + (int64_t)r * kv_stride;
        bf16* dst = cache + ((int64_t)blk * PAGE + (pos % PAGE)) * CACHE_W;
        if (tid < ATHR) {
            // the 256-thread block's red[t] sums columns t, t+256 in ascending order; thread tid computes both
            // t = tid and t = tid + 128, so red[] is bit-identical and so is the warp-0 tree below.
            float ss0 = 0.0f, ss1 = 0.0f;
            for (int c = tid; c < DC; c += 256) {
                const float v = bf2f(src[c]);
                ss0 = __fmaf_rn(v, v, ss0);
            }
            for (int c = tid + 128; c < DC; c += 256) {
                const float v = bf2f(src[c]);
                ss1 = __fmaf_rn(v, v, ss1);
            }
            red[tid] = ss0;
            red[tid + 128] = ss1;
        }
        __syncthreads();
        if (tid < 32) {
            const float r0 = red[tid] + red[tid + 128];
            const float r2 = red[tid + 64] + red[tid + 192];
            const float r1 = red[tid + 32] + red[tid + 160];
            const float r3 = red[tid + 96] + red[tid + 224];
            float v = (r0 + r2) + (r1 + r3);
            v += __shfl_down_sync(0xffffffffu, v, 16);
            v += __shfl_down_sync(0xffffffffu, v, 8);
            v += __shfl_down_sync(0xffffffffu, v, 4);
            v += __shfl_down_sync(0xffffffffu, v, 2);
            v += __shfl_down_sync(0xffffffffu, v, 1);
            if (tid == 0) red[0] = v;
        }
        __syncthreads();
        if (tid < ATHR) {
            const float inv = rsqrtf(red[0] / (float)DC + 1e-6f);
            for (int c = tid; c < DC; c += 256) {
                dst[c] = f2bf(__fmul_rn(__fmul_rn(bf2f(src[c]), inv), bf2f(norm_w[c])));
            }
            for (int c = tid + 128; c < DC; c += 256) {
                dst[c] = f2bf(__fmul_rn(__fmul_rn(bf2f(src[c]), inv), bf2f(norm_w[c])));
            }
            const float* cs = cos + (size_t)pos * table_stride;
            const float* sn = sin + (size_t)pos * table_stride;
            for (int i = tid; i < DR / 2; i += 256) {
                const float a = bf2f(src[DC + 2 * i]);
                const float b = bf2f(src[DC + 2 * i + 1]);
                const float ci = cs[i], si = sn[i];
                dst[DC + i] = f2bf(__fsub_rn(__fmul_rn(a, ci), __fmul_rn(b, si)));
                dst[DC + DR / 2 + i] = f2bf(__fadd_rn(__fmul_rn(b, ci), __fmul_rn(a, si)));
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- tensor-core absorbed attention
//
// One CTA = one query row x 32 heads (two 16-head MMA tiles, h_base = blockIdx.y * 32), 256 threads.
// Warp w in {0..7}: head tile mi = w >> 2 (0 or 1), d-quarter dq = w & 3 (latent columns dq*128 .. dq*128+127).
// Every 32-key tile of the row is cp.async-staged ONCE in shared memory (latent 512 + rope 64 = 576 halves a
// key, 36 KB a tile, two stages) and read by all 8 warps / 32 heads.
//
// Per warp and tile:
//   QK: this warp's 16 heads x 32 keys over its d-quarter (128 of the 512 latent), 8 k16 mma steps;
//       the four quarters' fp32 partial scores are added in the fixed order dq = 0, 1, 2, 3 through a shared
//       float4 exchange; the rope 64 part (4 k16 steps, all heads x 32 keys) is added once, by quarter 0.
//   softmax: log2-domain online softmax per head row (m, l), the causal mask j <= row_pos[r], keys in order.
//   PV: exp2(s - m) repacked bf16 as the A operand; this warp's 16 heads x its 128 latent output columns
//       accumulated over the tile's 32 keys as two k16 steps (mma with the staged latent tile transposed).
// After the row's last tile, o / l rounded once to bf16, written to o_lat[r][h][0..512).
//
// Score chain order is fixed per (head, key): latent quarters 0..3 then rope; each quarter's own chain is
// ascending k16 from +0.0. A row's result does not depend on rows, grid or the other head tile of the CTA
// (the two head tiles only share the key tiles and a barrier).
constexpr int KT = 32;            // keys a tile
constexpr int HT = 16;            // heads an MMA tile
constexpr int MT = 2;             // MMA head tiles a CTA (32 heads)
constexpr int NWQ = 4;            // d-quarters = warps a head tile
constexpr int DQ = DC / NWQ;      // 128 latent columns a warp
constexpr int TNTHR = 256;
constexpr int KPAD = CACHE_W + 8; // halves a staged key row (576 + 8: ldmatrix bank spread)
constexpr int SWAPS = 2;          // key tile stages

// score exchange: [head tile][d-quarter][KT/8 n8 tiles][32 lanes] float4 — single-buffered (the bar.sync of
// each head tile serializes produce/consume within one tile iteration, so unlike the key tiles it needs no
// double buffering; this also keeps the total dynamic shared under sm_121's 99 KB opt-in).
constexpr int SX_F4 = (KT / 8) * 32;

__global__ void __launch_bounds__(TNTHR, 1) attn_kernel(
    const bf16* __restrict__ q_lat, const bf16* __restrict__ q_rot, const bf16* __restrict__ cache,
    const int* __restrict__ table, int max_blocks, const int* __restrict__ row_slot,
    const int* __restrict__ row_pos, bf16* __restrict__ out) {
    extern __shared__ __align__(16) bf16 smem[];
    bf16* sk = smem;                                        // SWAPS * KT * KPAD halves
    float4* sx = reinterpret_cast<float4*>(sk + SWAPS * KT * KPAD);  // MT * NWQ * SX_F4 float4 (latent)
    // the rope exchange is a SEPARATE region from the latent partials: quarter 0 writes the rope QK here
    // while the other quarters may still be summing the latent partials out of sx, so sharing a slot would
    // race (and corrupt the scores). MT * SX_F4 float4.
    float4* sr = sx + MT * NWQ * SX_F4;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int mi = wid >> 2, dq = wid & 3;                  // head tile of the pair, d-quarter
    const int row = blockIdx.x, h0 = blockIdx.y * (MT * HT) + mi * HT;
    const int g = lane >> 2, qd = lane & 3, mt = lane >> 3; // mma row group, column pair, 8-key half
    const int pos = row_pos[row];
    // this warp's two mma rows of heads (g and g+8) x its 128 latent columns
    bf16* dst0 = out + ((int64_t)row * NH + h0 + g) * DC + dq * DQ + 2 * qd;
    bf16* dst1 = out + ((int64_t)row * NH + h0 + g + 8) * DC + dq * DQ + 2 * qd;
    if (pos < 0) {
#pragma unroll
        for (int i = 0; i < DQ / 8; ++i) {
            *reinterpret_cast<uint32_t*>(dst0 + 8 * i) = 0u;
            *reinterpret_cast<uint32_t*>(dst1 + 8 * i) = 0u;
        }
        return;
    }
    const int limit = pos + 1;
    const int total = (limit + KT - 1) / KT;
    const int* tbl = table + (int64_t)row_slot[row] * max_blocks;

    // stage the tile t (keys t*KT .. t*KT+31) into stage st; every thread stages a share of the whole tile
    // (all 32 heads of the CTA share it). Keys past the row's end are zero-filled, masked in the softmax.
    auto stage = [&](int t, int st) {
        const int key0 = t * KT;
        const int blk = tbl[key0 / PAGE];
        const bf16* base = cache + ((int64_t)blk * PAGE + key0 % PAGE) * CACHE_W;
        // KT * CACHE_W / 8 = 32 * 72 = 2304 pieces of 16 bytes; 9 passes of 256 threads.
        constexpr int PIECES = KT * (CACHE_W / 8);
#pragma unroll
        for (int it = 0; it < (PIECES + TNTHR - 1) / TNTHR; ++it) {
            const int p = tid + TNTHR * it;
            if (p < PIECES) {
                const int j = p / (CACHE_W / 8), c8 = p % (CACHE_W / 8);
                const bool in = key0 + j < limit;
                cp16(sk + (st * KT + j) * KPAD + c8 * 8, base + (int64_t)j * CACHE_W + c8 * 8, in);
            }
        }
    };
    stage(0, 0);
    commit();

    // this warp's quarter of its 16 heads' q_lat as A fragments (8 k16 steps of the 128-wide quarter)
    uint32_t qa[DQ / 16][4];
    {
        const bf16* qrow = q_lat + ((int64_t)row * NH + h0) * DC + dq * DQ + 2 * qd;
#pragma unroll
        for (int k = 0; k < DQ / 16; ++k) {
            qa[k][0] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k);
            qa[k][1] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k);
            qa[k][2] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k + 8);
            qa[k][3] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k + 8);
        }
    }
    // rope fragments (all of the 64-wide rope, 4 k16 steps), only quarter 0 uses them
    uint32_t qr[DR / 16][4];
    if (dq == 0) {
        const bf16* qrow = q_rot + ((int64_t)row * NH + h0) * DR + 2 * qd;
#pragma unroll
        for (int k = 0; k < DR / 16; ++k) {
            qr[k][0] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DR + 16 * k);
            qr[k][1] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DR + 16 * k);
            qr[k][2] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DR + 16 * k + 8);
            qr[k][3] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DR + 16 * k + 8);
        }
    }

    float o[DQ / 8][4];
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int t = 0; t < total; ++t) {
        wait_group<0>();
        __syncthreads();  // tile t visible to every warp; all warps past the last read of stage (t+1)&1
        if (t + 1 < total) stage(t + 1, (t + 1) & 1);
        commit();
        const bf16* tile = sk + (t & 1) * KT * KPAD;

        // ---- QK of this d-quarter: 16 heads x 32 keys over 128 latent columns, 8 k16 steps from +0.0
        float s[KT / 8][4];
        {
            // B fragments straight from the key rows: key j's latent is tile[j*KPAD + dq*DQ + ...], so a
            // ldmatrix4 (non-trans) over eight key rows of 16 halves gives the (k, n) = (latent, key) pairs
            // the row.col mma wants for B = K^T.
#pragma unroll
            for (int k = 0; k < DQ / 16; ++k) {
#pragma unroll
                for (int n = 0; n < KT / 8; n += 2) {
                    uint32_t b[4];
                    ldmatrix4(b, tile + (8 * (n + (mt >> 1)) + (lane & 7)) * KPAD + dq * DQ + 16 * k + (mt & 1) * 8);
                    if (k == 0) {
                        mma0(s[n], qa[k], b[0], b[1]);
                        mma0(s[n + 1], qa[k], b[2], b[3]);
                    } else {
                        mma(s[n], qa[k], b[0], b[1]);
                        mma(s[n + 1], qa[k], b[2], b[3]);
                    }
                }
            }
        }
        // ---- the four quarters' partial scores added in the fixed order dq = 0, 1, 2, 3
        float4* mine = sx + (mi * NWQ + dq) * SX_F4;
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) mine[n * 32 + lane] = make_float4(s[n][0], s[n][1], s[n][2], s[n][3]);
        asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");  // the four warps of this head tile
        const float4* all = sx + mi * NWQ * SX_F4;
        float sc[KT / 8][4];
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            const float4 a = all[0 * SX_F4 + n * 32 + lane], b = all[1 * SX_F4 + n * 32 + lane];
            const float4 c = all[2 * SX_F4 + n * 32 + lane], d = all[3 * SX_F4 + n * 32 + lane];
            sc[n][0] = __fadd_rn(__fadd_rn(__fadd_rn(a.x, b.x), c.x), d.x);
            sc[n][1] = __fadd_rn(__fadd_rn(__fadd_rn(a.y, b.y), c.y), d.y);
            sc[n][2] = __fadd_rn(__fadd_rn(__fadd_rn(a.z, b.z), c.z), d.z);
            sc[n][3] = __fadd_rn(__fadd_rn(__fadd_rn(a.w, b.w), c.w), d.w);
        }
        // ---- rope: quarter 0 computes the 64-wide rope QK for all 16 heads x 32 keys into the SEPARATE
        // rope region sr, then every quarter adds it after the latent sum (rope last in the score chain).
        if (dq == 0) {
            float srq[KT / 8][4];
#pragma unroll
            for (int k = 0; k < DR / 16; ++k) {
#pragma unroll
                for (int n = 0; n < KT / 8; n += 2) {
                    uint32_t b[4];
                    ldmatrix4(b, tile + (8 * (n + (mt >> 1)) + (lane & 7)) * KPAD + DC + 16 * k + (mt & 1) * 8);
                    if (k == 0) {
                        mma0(srq[n], qr[k], b[0], b[1]);
                        mma0(srq[n + 1], qr[k], b[2], b[3]);
                    } else {
                        mma(srq[n], qr[k], b[0], b[1]);
                        mma(srq[n + 1], qr[k], b[2], b[3]);
                    }
                }
            }
            float4* dst = sr + mi * SX_F4;
#pragma unroll
            for (int n = 0; n < KT / 8; ++n) {
                dst[n * 32 + lane] = make_float4(srq[n][0], srq[n][1], srq[n][2], srq[n][3]);
            }
        }
        asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");
        const float4* rp = sr + mi * SX_F4;
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            const float4 v = rp[n * 32 + lane];
            sc[n][0] = __fadd_rn(sc[n][0], v.x);
            sc[n][1] = __fadd_rn(sc[n][1], v.y);
            sc[n][2] = __fadd_rn(sc[n][2], v.z);
            sc[n][3] = __fadd_rn(sc[n][3], v.w);
        }

        // ---- online softmax fold (log2 domain): mask j > pos, per-row m/l update, keys ascending
        const int kb = t * KT + 2 * qd;
        bool v0[KT / 8], v1[KT / 8];
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            v0[n] = kb + 8 * n <= pos;
            v1[n] = kb + 8 * n + 1 <= pos;
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            sc[n][0] = v0[n] ? __fmul_rn(sc[n][0], SM_SCALE2) : -INFINITY;
            sc[n][1] = v1[n] ? __fmul_rn(sc[n][1], SM_SCALE2) : -INFINITY;
            sc[n][2] = v0[n] ? __fmul_rn(sc[n][2], SM_SCALE2) : -INFINITY;
            sc[n][3] = v1[n] ? __fmul_rn(sc[n][3], SM_SCALE2) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
            tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
        }
#pragma unroll
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
        }
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float a0 = m0 == -INFINITY ? 0.0f : ex2(__fsub_rn(m0, n0));
        const float a1 = m1 == -INFINITY ? 0.0f : ex2(__fsub_rn(m1, n1));
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            sc[n][0] = v0[n] ? ex2(__fsub_rn(sc[n][0], n0)) : 0.0f;
            sc[n][1] = v1[n] ? ex2(__fsub_rn(sc[n][1], n0)) : 0.0f;
            sc[n][2] = v0[n] ? ex2(__fsub_rn(sc[n][2], n1)) : 0.0f;
            sc[n][3] = v1[n] ? ex2(__fsub_rn(sc[n][3], n1)) : 0.0f;
        }
        uint32_t pa[KT / 16][4];
#pragma unroll
        for (int j = 0; j < KT / 16; ++j) {
            pa[j][0] = pack2(sc[2 * j][0], sc[2 * j][1]);
            pa[j][1] = pack2(sc[2 * j][2], sc[2 * j][3]);
            pa[j][2] = pack2(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pa[j][3] = pack2(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            const uint32_t wlo = pa[n >> 1][(n & 1) * 2], whi = pa[n >> 1][(n & 1) * 2 + 1];
            const float x0 = __uint_as_float(wlo << 16), x1 = __uint_as_float(wlo & 0xffff0000u);
            const float y0 = __uint_as_float(whi << 16), y1 = __uint_as_float(whi & 0xffff0000u);
            s0 = __fadd_rn(s0, __fadd_rn(x0, x1));
            s1 = __fadd_rn(s1, __fadd_rn(y0, y1));
        }
        l0 = __fmaf_rn(l0, a0, s0);
        l1 = __fmaf_rn(l1, a1, s1);
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int i = 0; i < DQ / 8; ++i) {
            o[i][0] = __fmul_rn(o[i][0], a0);
            o[i][1] = __fmul_rn(o[i][1], a0);
            o[i][2] = __fmul_rn(o[i][2], a1);
            o[i][3] = __fmul_rn(o[i][3], a1);
        }
        // ---- PV over this quarter's output columns, keys in ascending k16 order
#pragma unroll
        for (int ks = 0; ks < KT / 16; ++ks) {
#pragma unroll
            for (int i = 0; i < DQ / 8; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, tile + (16 * ks + 8 * (mt & 1) + (lane & 7)) * KPAD + dq * DQ + i * 8 + (mt >> 1) * 8);
                mma(o[i], pa[ks], b[0], b[1]);
                mma(o[i + 1], pa[ks], b[2], b[3]);
            }
        }
    }
    wait_group<0>();
    // The softmax denominator is a per-lane partial: each of the 4 qd lanes of a head accumulated only its
    // own keys' exp2 weights. The head's full normaliser is the sum across those 4 lanes (the same
    // quad_sum mla_prefill.cu applies at chunk merge), shared by every column of the head. Without this a
    // lane whose keys were all masked would divide by zero and zero out its columns.
    float lf0 = l0, lf1 = l1;
#pragma unroll
    for (int x = 1; x <= 2; x *= 2) {
        lf0 = __fadd_rn(lf0, __shfl_xor_sync(0xffffffffu, lf0, x));
        lf1 = __fadd_rn(lf1, __shfl_xor_sync(0xffffffffu, lf1, x));
    }
    const float il0 = lf0 > 0.0f ? __fdiv_rn(1.0f, lf0) : 0.0f;
    const float il1 = lf1 > 0.0f ? __fdiv_rn(1.0f, lf1) : 0.0f;
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) {
        *reinterpret_cast<uint32_t*>(dst0 + 8 * i) = pack2(__fmul_rn(o[i][0], il0), __fmul_rn(o[i][1], il0));
        *reinterpret_cast<uint32_t*>(dst1 + 8 * i) = pack2(__fmul_rn(o[i][2], il1), __fmul_rn(o[i][3], il1));
    }
}

// Two query rows per CTA (grid = (ceil(rows/2), NH/HT), HT = 16 heads), the LP2 row-invariant contract
// unchanged: warp w in {0..7} owns row r0 + (w >> 2) (r0 = blockIdx.x * 2) x the 16 heads
// h0 = blockIdx.y * 16, d-quarter dq = w & 3. The 32-key tile is staged ONCE per CTA (cp.async) and shared
// by both rows, so each staged tile serves two query rows (the per-warp arithmetic — one row x 16 heads x
// 128 latent columns — is bit-identical to attn_kernel's). Rows of a pair may have different causal positions
// (pos) and one may be padding (pos < 0) or out of range (odd rows): the per-warp loop skips the
// QK/softmax/PV body for tiles past its own total, still hitting the two named barriers of its 4-warp group
// (the barrier group is per-row, so the skip is uniform within it); padding rows fall out with l = 0 -> zero
// output. A row past `rows` writes nothing.
__global__ void __launch_bounds__(TNTHR, 1) attn_pair_kernel(
    const bf16* __restrict__ q_lat, const bf16* __restrict__ q_rot, const bf16* __restrict__ cache,
    const int* __restrict__ table, int max_blocks, const int* __restrict__ row_slot,
    const int* __restrict__ row_pos, bf16* __restrict__ out, int rows) {
    extern __shared__ __align__(16) bf16 smem[];
    bf16* sk = smem;                                        // SWAPS * KT * KPAD halves
    float4* sx = reinterpret_cast<float4*>(sk + SWAPS * KT * KPAD);
    float4* sr = sx + MT * NWQ * SX_F4;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int mi = wid >> 2, dq = wid & 3;                  // row of the pair, d-quarter
    const int r0 = blockIdx.x * 2;
    const int row = r0 + mi;
    const bool valid = row < rows;
    const int r_safe = valid ? row : r0;  // odd-row padding warps read row0, never q_lat/q_rot past the allocation
    const int h0 = blockIdx.y * HT;
    const int g = lane >> 2, qd = lane & 3, mt = lane >> 3; // mma row group, column pair, 8-key half
    const int p0 = row_pos[r0];                             // r0 is always < rows
    const int p1 = (r0 + 1 < rows) ? row_pos[r0 + 1] : -1;
    const int pos = valid ? (mi == 0 ? p0 : p1) : -1;
    // CTA staging covers the longer of the two rows (lim_max); the shorter row masks in its softmax.
    const int lim_max = p0 > p1 ? p0 : p1;
    const int total_max = (lim_max + 1 + KT - 1) / KT;
    const int limit = pos + 1;
    const int total = limit > 0 ? (limit + KT - 1) / KT : 0;
    // this warp's two mma rows of heads (g and g+8) x its 128 latent columns
    bf16* dst0 = out + ((int64_t)row * NH + h0 + g) * DC + dq * DQ + 2 * qd;
    bf16* dst1 = out + ((int64_t)row * NH + h0 + g + 8) * DC + dq * DQ + 2 * qd;
    const int* tbl = table + (int64_t)row_slot[r0] * max_blocks;

    auto stage = [&](int t, int st) {
        const int key0 = t * KT;
        const int blk = tbl[key0 / PAGE];
        const bf16* base = cache + ((int64_t)blk * PAGE + key0 % PAGE) * CACHE_W;
        constexpr int PIECES = KT * (CACHE_W / 8);
#pragma unroll
        for (int it = 0; it < (PIECES + TNTHR - 1) / TNTHR; ++it) {
            const int p = tid + TNTHR * it;
            if (p < PIECES) {
                const int j = p / (CACHE_W / 8), c8 = p % (CACHE_W / 8);
                const bool in = key0 + j <= lim_max;
                cp16(sk + (st * KT + j) * KPAD + c8 * 8, base + (int64_t)j * CACHE_W + c8 * 8, in);
            }
        }
    };
    stage(0, 0);
    commit();

    uint32_t qa[DQ / 16][4];
    {
        const bf16* qrow = q_lat + ((int64_t)r_safe * NH + h0) * DC + dq * DQ + 2 * qd;
#pragma unroll
        for (int k = 0; k < DQ / 16; ++k) {
            qa[k][0] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k);
            qa[k][1] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k);
            qa[k][2] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k + 8);
            qa[k][3] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k + 8);
        }
    }
    uint32_t qr[DR / 16][4];
    if (dq == 0) {
        const bf16* qrow = q_rot + ((int64_t)r_safe * NH + h0) * DR + 2 * qd;
#pragma unroll
        for (int k = 0; k < DR / 16; ++k) {
            qr[k][0] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DR + 16 * k);
            qr[k][1] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DR + 16 * k);
            qr[k][2] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DR + 16 * k + 8);
            qr[k][3] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DR + 16 * k + 8);
        }
    }

    float o[DQ / 8][4];
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int t = 0; t < total_max; ++t) {
        wait_group<0>();
        __syncthreads();  // tile t visible to every warp
        if (t + 1 < total_max) stage(t + 1, (t + 1) & 1);
        commit();
        const bf16* tile = sk + (t & 1) * KT * KPAD;
        const bool active = t < total && pos >= 0;

        if (active) {
            // ---- QK of this d-quarter: 16 heads x 32 keys over 128 latent columns, 8 k16 steps from +0.0
            float s[KT / 8][4];
#pragma unroll
                for (int k = 0; k < DQ / 16; ++k) {
#pragma unroll
                    for (int n = 0; n < KT / 8; n += 2) {
                        uint32_t b[4];
                        ldmatrix4(b, tile + (8 * (n + (mt >> 1)) + (lane & 7)) * KPAD + dq * DQ + 16 * k + (mt & 1) * 8);
                        if (k == 0) {
                            mma0(s[n], qa[k], b[0], b[1]);
                            mma0(s[n + 1], qa[k], b[2], b[3]);
                        } else {
                            mma(s[n], qa[k], b[0], b[1]);
                            mma(s[n + 1], qa[k], b[2], b[3]);
                        }
                    }
                }
            float4* mine = sx + (mi * NWQ + dq) * SX_F4;
#pragma unroll
            for (int n = 0; n < KT / 8; ++n) mine[n * 32 + lane] = make_float4(s[n][0], s[n][1], s[n][2], s[n][3]);
            if (dq == 0) {
#pragma unroll
                for (int k = 0; k < DR / 16; ++k) {
#pragma unroll
                    for (int n = 0; n < KT / 8; n += 2) {
                        uint32_t b[4];
                        ldmatrix4(b, tile + (8 * (n + (mt >> 1)) + (lane & 7)) * KPAD + DC + 16 * k + (mt & 1) * 8);
                        if (k == 0) {
                            mma0(s[n], qr[k], b[0], b[1]);
                            mma0(s[n + 1], qr[k], b[2], b[3]);
                        } else {
                            mma(s[n], qr[k], b[0], b[1]);
                            mma(s[n + 1], qr[k], b[2], b[3]);
                        }
                    }
                }
                float4* dst = sr + mi * SX_F4;
#pragma unroll
                for (int n = 0; n < KT / 8; ++n) {
                    dst[n * 32 + lane] = make_float4(s[n][0], s[n][1], s[n][2], s[n][3]);
                }
            }
            asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");
            const float4* all = sx + mi * NWQ * SX_F4;
#pragma unroll
            for (int n = 0; n < KT / 8; ++n) {
                const float4 a = all[0 * SX_F4 + n * 32 + lane], b = all[1 * SX_F4 + n * 32 + lane];
                const float4 c = all[2 * SX_F4 + n * 32 + lane], d = all[3 * SX_F4 + n * 32 + lane];
                s[n][0] = __fadd_rn(__fadd_rn(__fadd_rn(a.x, b.x), c.x), d.x);
                s[n][1] = __fadd_rn(__fadd_rn(__fadd_rn(a.y, b.y), c.y), d.y);
                s[n][2] = __fadd_rn(__fadd_rn(__fadd_rn(a.z, b.z), c.z), d.z);
                s[n][3] = __fadd_rn(__fadd_rn(__fadd_rn(a.w, b.w), c.w), d.w);
            }
            asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");
            const float4* rp = sr + mi * SX_F4;
#pragma unroll
            for (int n = 0; n < KT / 8; ++n) {
                const float4 v = rp[n * 32 + lane];
                s[n][0] = __fadd_rn(s[n][0], v.x);
                s[n][1] = __fadd_rn(s[n][1], v.y);
                s[n][2] = __fadd_rn(s[n][2], v.z);
                s[n][3] = __fadd_rn(s[n][3], v.w);
            }

            const int kb = t * KT + 2 * qd;
            bool v0[KT / 8], v1[KT / 8];
#pragma unroll
            for (int n = 0; n < KT / 8; ++n) {
                v0[n] = kb + 8 * n <= pos;
                v1[n] = kb + 8 * n + 1 <= pos;
            }
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < KT / 8; ++n) {
                s[n][0] = v0[n] ? __fmul_rn(s[n][0], SM_SCALE2) : -INFINITY;
                s[n][1] = v1[n] ? __fmul_rn(s[n][1], SM_SCALE2) : -INFINITY;
                s[n][2] = v0[n] ? __fmul_rn(s[n][2], SM_SCALE2) : -INFINITY;
                s[n][3] = v1[n] ? __fmul_rn(s[n][3], SM_SCALE2) : -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(s[n][0], s[n][1]));
                tm1 = fmaxf(tm1, fmaxf(s[n][2], s[n][3]));
            }
#pragma unroll
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
            const float a0 = m0 == -INFINITY ? 0.0f : ex2(__fsub_rn(m0, n0));
            const float a1 = m1 == -INFINITY ? 0.0f : ex2(__fsub_rn(m1, n1));
#pragma unroll
            for (int n = 0; n < KT / 8; ++n) {
                s[n][0] = v0[n] ? ex2(__fsub_rn(s[n][0], n0)) : 0.0f;
                s[n][1] = v1[n] ? ex2(__fsub_rn(s[n][1], n0)) : 0.0f;
                s[n][2] = v0[n] ? ex2(__fsub_rn(s[n][2], n1)) : 0.0f;
                s[n][3] = v1[n] ? ex2(__fsub_rn(s[n][3], n1)) : 0.0f;
            }
            uint32_t pa[KT / 16][4];
#pragma unroll
            for (int j = 0; j < KT / 16; ++j) {
                pa[j][0] = pack2(s[2 * j][0], s[2 * j][1]);
                pa[j][1] = pack2(s[2 * j][2], s[2 * j][3]);
                pa[j][2] = pack2(s[2 * j + 1][0], s[2 * j + 1][1]);
                pa[j][3] = pack2(s[2 * j + 1][2], s[2 * j + 1][3]);
            }
            float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
            for (int n = 0; n < KT / 8; ++n) {
                const uint32_t wlo = pa[n >> 1][(n & 1) * 2], whi = pa[n >> 1][(n & 1) * 2 + 1];
                const float x0 = __uint_as_float(wlo << 16), x1 = __uint_as_float(wlo & 0xffff0000u);
                const float y0 = __uint_as_float(whi << 16), y1 = __uint_as_float(whi & 0xffff0000u);
                s0 = __fadd_rn(s0, __fadd_rn(x0, x1));
                s1 = __fadd_rn(s1, __fadd_rn(y0, y1));
            }
            l0 = __fmaf_rn(l0, a0, s0);
            l1 = __fmaf_rn(l1, a1, s1);
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < DQ / 8; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
#pragma unroll
            for (int ks = 0; ks < KT / 16; ++ks) {
#pragma unroll
                for (int i = 0; i < DQ / 8; i += 2) {
                    uint32_t b[4];
                    ldmatrix4t(b, tile + (16 * ks + 8 * (mt & 1) + (lane & 7)) * KPAD + dq * DQ + i * 8 + (mt >> 1) * 8);
                    mma(o[i], pa[ks], b[0], b[1]);
                    mma(o[i + 1], pa[ks], b[2], b[3]);
                }
            }
        } else {
            // tiles past this row's total (or a padding row): no QK, still hit this group's two barriers
            asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");
            asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");
        }
    }
    wait_group<0>();
    if (!valid) return;
    float lf0 = l0, lf1 = l1;
#pragma unroll
    for (int x = 1; x <= 2; x *= 2) {
        lf0 = __fadd_rn(lf0, __shfl_xor_sync(0xffffffffu, lf0, x));
        lf1 = __fadd_rn(lf1, __shfl_xor_sync(0xffffffffu, lf1, x));
    }
    const float il0 = lf0 > 0.0f ? __fdiv_rn(1.0f, lf0) : 0.0f;
    const float il1 = lf1 > 0.0f ? __fdiv_rn(1.0f, lf1) : 0.0f;
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) {
        *reinterpret_cast<uint32_t*>(dst0 + 8 * i) = pack2(__fmul_rn(o[i][0], il0), __fmul_rn(o[i][1], il0));
        *reinterpret_cast<uint32_t*>(dst1 + 8 * i) = pack2(__fmul_rn(o[i][2], il1), __fmul_rn(o[i][3], il1));
    }
}

// sk (two 36 KB key stages) + sx (latent partials) + sr (rope partials, separate to avoid the produce/consume
// race on the latent slots). 74752 + 16384 + 8192 = 99328 bytes <= sm_121's 101376 opt-in.
constexpr int kAttnSmemBytes = SWAPS * KT * KPAD * 2 + MT * NWQ * SX_F4 * 16 + MT * SX_F4 * 16;

// ---------------------------------------------------------------- split-key decode attention
//
// The decode / verify / draft attention (one to a few rows, many keys). `attn_kernel` walks a row's keys in one CTA per 32-head group, so a
// 30K-key row keeps 2 of the 48 SMs busy; this path splits the key range into ABSOLUTE chunks of SPLIT_KEYS keys ([c * SPLIT_KEYS,
// (c + 1) * SPLIT_KEYS), counted from the row's key 0) and runs one CTA per (row, 32-head group, chunk).
//
//   chunk partial  (m, l, o[512]) per head: the SAME per-warp tile code as `attn_kernel` (32-key tiles, d-quarter score order, rope last,
//                  log2-domain online softmax, PV over the tile), started from the zero state (m = -inf, l = 0, o = 0) at the chunk's first
//                  tile and run over the chunk's tiles in key order. o is left UNNORMALISED (fp32), l is the quad-summed denominator.
//   combine        one thread block per (row, head): M = max_c m_c over the row's chunks c = 0 .. pos / SPLIT_KEYS (the chunk count is a
//                  function of the row's own position), w_c = 2^(m_c - M), then in ASCENDING chunk order
//                      L = l_0 w_0, then L = fma(l_c, w_c, L);   O = o_0 w_0, then O = fma(o_c, w_c, O)   (c = 1 ..)
//                  and out = bf16(O * (1 / L)) with the same reciprocal and rounding as `attn_kernel`'s epilogue.
//
// So a row's output is a function of its q, its slot's keys 0..=pos and pos only. It does not depend on the number of rows of the call, on
// which rows share it, on the grid (the chunk grid spans the block table's pages and a chunk past the row's last key exits), on the number of
// SMs or CTAs resident, on how many row batches the launcher splits a large call into, or on anything in the cache past key pos. A row with
// pos < SPLIT_KEYS has ONE chunk, and one chunk combines to exactly the bits of `attn_kernel` (w_0 = 2^0 = 1, O = o_0 * 1, L = l_0 * 1).
// The partials live in a caller-owned fp32 workspace (SPLIT_SLOT_BYTES a (row, chunk) slot); there are no atomics.
constexpr int SPLIT_KEYS = 1024;                       // keys a chunk (the arithmetic id names it)
constexpr int SPLIT_TILES = SPLIT_KEYS / KT;           // 32-key tiles a chunk
static_assert(SPLIT_KEYS % KT == 0 && SPLIT_KEYS % PAGE == 0, "chunks are whole tiles and whole pages");
constexpr int64_t SPLIT_SLOT_O = (int64_t)NH * DC;     // floats of o a (row, chunk) slot
constexpr int64_t SPLIT_SLOT_BYTES = SPLIT_SLOT_O * 4 + (int64_t)NH * 8;  // o, then (m, l) per head

__global__ void __launch_bounds__(TNTHR, 1) attn_split_kernel(
    const bf16* __restrict__ q_lat, const bf16* __restrict__ q_rot, const bf16* __restrict__ cache,
    const int* __restrict__ table, int max_blocks, const int* __restrict__ row_slot,
    const int* __restrict__ row_pos, float* __restrict__ part_o, float2* __restrict__ part_ml, int max_chunks,
    bf16* __restrict__ out, int* __restrict__ split_cnt) {
    extern __shared__ __align__(16) bf16 smem[];
    bf16* sk = smem;                                        // SWAPS * KT * KPAD halves
    float4* sx = reinterpret_cast<float4*>(sk + SWAPS * KT * KPAD);
    float4* sr = sx + MT * NWQ * SX_F4;
    const int row = blockIdx.x, chunk = blockIdx.z;
    const int pos = row_pos[row];
    // a padding row: the chunk-0 CTA of each 32-head group zeroes its heads' output (the fused combine's padding
    // branch, exactly the bytes `attn_combine_kernel` wrote for a negative position) and exits
    if (pos < 0) {
        if (chunk == 0 && out) {
            const int h0 = blockIdx.y * (MT * HT);
            for (int idx = threadIdx.x; idx < (MT * HT) * DC; idx += TNTHR) {
                const int h = idx >> 9, c = idx & 511;
                out[(int64_t)row * NH * DC + (h0 + h) * DC + c] = 0;
            }
        }
        return;
    }
    // a chunk wholly past the row's last key: nothing to compute (CTA-uniform exit, before any barrier)
    if (chunk * SPLIT_KEYS > pos) return;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int mi = wid >> 2, dq = wid & 3;                  // head tile of the pair, d-quarter
    const int h0 = blockIdx.y * (MT * HT) + mi * HT;
    const int g = lane >> 2, qd = lane & 3, mt = lane >> 3; // mma row group, column pair, 8-key half
    const int limit = pos + 1;
    const int total = (limit + KT - 1) / KT;
    const int t0 = chunk * SPLIT_TILES;
    const int t1 = min(total, t0 + SPLIT_TILES);            // this chunk's tiles are t0 .. t1 - 1, t0 < t1
    const int* tbl = table + (int64_t)row_slot[row] * max_blocks;

    auto stage = [&](int t, int st) {
        const int key0 = t * KT;
        const int blk = tbl[key0 / PAGE];
        const bf16* base = cache + ((int64_t)blk * PAGE + key0 % PAGE) * CACHE_W;
        constexpr int PIECES = KT * (CACHE_W / 8);
#pragma unroll
        for (int it = 0; it < (PIECES + TNTHR - 1) / TNTHR; ++it) {
            const int p = tid + TNTHR * it;
            if (p < PIECES) {
                const int j = p / (CACHE_W / 8), c8 = p % (CACHE_W / 8);
                const bool in = key0 + j < limit;
                cp16(sk + (st * KT + j) * KPAD + c8 * 8, base + (int64_t)j * CACHE_W + c8 * 8, in);
            }
        }
    };
    stage(t0, 0);
    commit();

    uint32_t qa[DQ / 16][4];
    {
        const bf16* qrow = q_lat + ((int64_t)row * NH + h0) * DC + dq * DQ + 2 * qd;
#pragma unroll
        for (int k = 0; k < DQ / 16; ++k) {
            qa[k][0] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k);
            qa[k][1] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k);
            qa[k][2] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DC + 16 * k + 8);
            qa[k][3] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DC + 16 * k + 8);
        }
    }
    uint32_t qr[DR / 16][4];
    if (dq == 0) {
        const bf16* qrow = q_rot + ((int64_t)row * NH + h0) * DR + 2 * qd;
#pragma unroll
        for (int k = 0; k < DR / 16; ++k) {
            qr[k][0] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DR + 16 * k);
            qr[k][1] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DR + 16 * k);
            qr[k][2] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)g * DR + 16 * k + 8);
            qr[k][3] = *reinterpret_cast<const uint32_t*>(qrow + (int64_t)(g + 8) * DR + 16 * k + 8);
        }
    }

    float o[DQ / 8][4];
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int t = t0; t < t1; ++t) {
        const int tl = t - t0;  // tile index inside the chunk: the stage parity (t0 is even, so it equals attn_kernel's t & 1)
        wait_group<0>();
        __syncthreads();  // tile t visible to every warp; all warps past the last read of stage (tl+1)&1
        if (t + 1 < t1) stage(t + 1, (tl + 1) & 1);
        commit();
        const bf16* tile = sk + (tl & 1) * KT * KPAD;

        // ---- QK of this d-quarter: 16 heads x 32 keys over 128 latent columns, 8 k16 steps from +0.0
        float s[KT / 8][4];
        {
#pragma unroll
            for (int k = 0; k < DQ / 16; ++k) {
#pragma unroll
                for (int n = 0; n < KT / 8; n += 2) {
                    uint32_t b[4];
                    ldmatrix4(b, tile + (8 * (n + (mt >> 1)) + (lane & 7)) * KPAD + dq * DQ + 16 * k + (mt & 1) * 8);
                    if (k == 0) {
                        mma0(s[n], qa[k], b[0], b[1]);
                        mma0(s[n + 1], qa[k], b[2], b[3]);
                    } else {
                        mma(s[n], qa[k], b[0], b[1]);
                        mma(s[n + 1], qa[k], b[2], b[3]);
                    }
                }
            }
        }
        // ---- the four quarters' partial scores added in the fixed order dq = 0, 1, 2, 3
        float4* mine = sx + (mi * NWQ + dq) * SX_F4;
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) mine[n * 32 + lane] = make_float4(s[n][0], s[n][1], s[n][2], s[n][3]);
        asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");  // the four warps of this head tile
        const float4* all = sx + mi * NWQ * SX_F4;
        float sc[KT / 8][4];
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            const float4 a = all[0 * SX_F4 + n * 32 + lane], b = all[1 * SX_F4 + n * 32 + lane];
            const float4 c = all[2 * SX_F4 + n * 32 + lane], d = all[3 * SX_F4 + n * 32 + lane];
            sc[n][0] = __fadd_rn(__fadd_rn(__fadd_rn(a.x, b.x), c.x), d.x);
            sc[n][1] = __fadd_rn(__fadd_rn(__fadd_rn(a.y, b.y), c.y), d.y);
            sc[n][2] = __fadd_rn(__fadd_rn(__fadd_rn(a.z, b.z), c.z), d.z);
            sc[n][3] = __fadd_rn(__fadd_rn(__fadd_rn(a.w, b.w), c.w), d.w);
        }
        // ---- rope: quarter 0 computes the 64-wide rope QK for all 16 heads x 32 keys into the separate rope region
        if (dq == 0) {
            float srq[KT / 8][4];
#pragma unroll
            for (int k = 0; k < DR / 16; ++k) {
#pragma unroll
                for (int n = 0; n < KT / 8; n += 2) {
                    uint32_t b[4];
                    ldmatrix4(b, tile + (8 * (n + (mt >> 1)) + (lane & 7)) * KPAD + DC + 16 * k + (mt & 1) * 8);
                    if (k == 0) {
                        mma0(srq[n], qr[k], b[0], b[1]);
                        mma0(srq[n + 1], qr[k], b[2], b[3]);
                    } else {
                        mma(srq[n], qr[k], b[0], b[1]);
                        mma(srq[n + 1], qr[k], b[2], b[3]);
                    }
                }
            }
            float4* dst = sr + mi * SX_F4;
#pragma unroll
            for (int n = 0; n < KT / 8; ++n) {
                dst[n * 32 + lane] = make_float4(srq[n][0], srq[n][1], srq[n][2], srq[n][3]);
            }
        }
        asm volatile("bar.sync %0, 128;\n" ::"r"(1 + mi) : "memory");
        const float4* rp = sr + mi * SX_F4;
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            const float4 v = rp[n * 32 + lane];
            sc[n][0] = __fadd_rn(sc[n][0], v.x);
            sc[n][1] = __fadd_rn(sc[n][1], v.y);
            sc[n][2] = __fadd_rn(sc[n][2], v.z);
            sc[n][3] = __fadd_rn(sc[n][3], v.w);
        }

        // ---- online softmax fold (log2 domain): mask j > pos, per-row m/l update, keys ascending
        const int kb = t * KT + 2 * qd;
        bool v0[KT / 8], v1[KT / 8];
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            v0[n] = kb + 8 * n <= pos;
            v1[n] = kb + 8 * n + 1 <= pos;
        }
        float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            sc[n][0] = v0[n] ? __fmul_rn(sc[n][0], SM_SCALE2) : -INFINITY;
            sc[n][1] = v1[n] ? __fmul_rn(sc[n][1], SM_SCALE2) : -INFINITY;
            sc[n][2] = v0[n] ? __fmul_rn(sc[n][2], SM_SCALE2) : -INFINITY;
            sc[n][3] = v1[n] ? __fmul_rn(sc[n][3], SM_SCALE2) : -INFINITY;
            tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
            tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
        }
#pragma unroll
        for (int x = 1; x <= 2; x *= 2) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
        }
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float a0 = m0 == -INFINITY ? 0.0f : ex2(__fsub_rn(m0, n0));
        const float a1 = m1 == -INFINITY ? 0.0f : ex2(__fsub_rn(m1, n1));
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            sc[n][0] = v0[n] ? ex2(__fsub_rn(sc[n][0], n0)) : 0.0f;
            sc[n][1] = v1[n] ? ex2(__fsub_rn(sc[n][1], n0)) : 0.0f;
            sc[n][2] = v0[n] ? ex2(__fsub_rn(sc[n][2], n1)) : 0.0f;
            sc[n][3] = v1[n] ? ex2(__fsub_rn(sc[n][3], n1)) : 0.0f;
        }
        uint32_t pa[KT / 16][4];
#pragma unroll
        for (int j = 0; j < KT / 16; ++j) {
            pa[j][0] = pack2(sc[2 * j][0], sc[2 * j][1]);
            pa[j][1] = pack2(sc[2 * j][2], sc[2 * j][3]);
            pa[j][2] = pack2(sc[2 * j + 1][0], sc[2 * j + 1][1]);
            pa[j][3] = pack2(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        }
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int n = 0; n < KT / 8; ++n) {
            const uint32_t wlo = pa[n >> 1][(n & 1) * 2], whi = pa[n >> 1][(n & 1) * 2 + 1];
            const float x0 = __uint_as_float(wlo << 16), x1 = __uint_as_float(wlo & 0xffff0000u);
            const float y0 = __uint_as_float(whi << 16), y1 = __uint_as_float(whi & 0xffff0000u);
            s0 = __fadd_rn(s0, __fadd_rn(x0, x1));
            s1 = __fadd_rn(s1, __fadd_rn(y0, y1));
        }
        l0 = __fmaf_rn(l0, a0, s0);
        l1 = __fmaf_rn(l1, a1, s1);
        m0 = n0;
        m1 = n1;
#pragma unroll
        for (int i = 0; i < DQ / 8; ++i) {
            o[i][0] = __fmul_rn(o[i][0], a0);
            o[i][1] = __fmul_rn(o[i][1], a0);
            o[i][2] = __fmul_rn(o[i][2], a1);
            o[i][3] = __fmul_rn(o[i][3], a1);
        }
        // ---- PV over this quarter's output columns, keys in ascending k16 order
#pragma unroll
        for (int ks = 0; ks < KT / 16; ++ks) {
#pragma unroll
            for (int i = 0; i < DQ / 8; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, tile + (16 * ks + 8 * (mt & 1) + (lane & 7)) * KPAD + dq * DQ + i * 8 + (mt >> 1) * 8);
                mma(o[i], pa[ks], b[0], b[1]);
                mma(o[i + 1], pa[ks], b[2], b[3]);
            }
        }
    }
    wait_group<0>();
    // The denominator is a per-lane partial: each of a head's 4 qd lanes summed only its own keys' weights; the head's chunk denominator is
    // their sum (the same quad sum as attn_kernel's epilogue).
    float lf0 = l0, lf1 = l1;
#pragma unroll
    for (int x = 1; x <= 2; x *= 2) {
        lf0 = __fadd_rn(lf0, __shfl_xor_sync(0xffffffffu, lf0, x));
        lf1 = __fadd_rn(lf1, __shfl_xor_sync(0xffffffffu, lf1, x));
    }
    // the unnormalised o of this warp's 16 heads x 128 latent columns, to the (row, chunk) slot; (m, l) once per head (every d-quarter holds
    // the same pair: the scores they fold are the same four-quarter sums)
    float* po = part_o + (int64_t)(row * max_chunks + chunk) * SPLIT_SLOT_O;
    float2* pm = part_ml + (int64_t)(row * max_chunks + chunk) * NH;
#pragma unroll
    for (int i = 0; i < DQ / 8; ++i) {
        const int col = dq * DQ + 8 * i + 2 * qd;
        *reinterpret_cast<float2*>(po + (int64_t)(h0 + g) * DC + col) = make_float2(o[i][0], o[i][1]);
        *reinterpret_cast<float2*>(po + (int64_t)(h0 + g + 8) * DC + col) = make_float2(o[i][2], o[i][3]);
    }
    if (dq == 0 && qd == 0) {
        pm[h0 + g] = make_float2(m0, lf0);
        pm[h0 + g + 8] = make_float2(m1, lf1);
    }
    // ---- the fused combine (the header above): a single-chunk row folds inline right here (bit-identical to
    // `attn_combine_kernel` at nch == 1: mx = m0, w = ex2(0) = 1, L = lf0, O = o, il = 1 / L), and a
    // multi-chunk row's parts are folded by the row's LAST CTA (a self-resetting per-(row, 32-head group)
    // atomic counter in the workspace's spare last slot; the call is row-batched so the counters are reused
    // across passes, always left at 0 by the pass's last block). The `out == nullptr` / `split_cnt == nullptr`
    // form is the test reference path (parts-only; the separate combine launch folds them).
    if (!out || !split_cnt) return;
    const int nch = pos / SPLIT_KEYS + 1;
    if (nch == 1) {
        const float il0 = lf0 > 0.0f ? __fdiv_rn(1.0f, lf0) : 0.0f;
        const float il1 = lf1 > 0.0f ? __fdiv_rn(1.0f, lf1) : 0.0f;
#pragma unroll
        for (int i = 0; i < DQ / 8; ++i) {
            const int col = dq * DQ + 8 * i + 2 * qd;
            // col is only 4-byte aligned (2 * qd), so store each half pair as a uint32 (the exact bytes
            // `attn_combine_kernel` wrote for the row's heads and columns)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)row * NH + h0 + g) * DC + col) =
                pack2(__fmul_rn(o[i][0], il0), __fmul_rn(o[i][1], il0));
            *reinterpret_cast<uint32_t*>(out + ((int64_t)row * NH + h0 + g + 8) * DC + col) =
                pack2(__fmul_rn(o[i][2], il1), __fmul_rn(o[i][3], il1));
        }
        return;
    }
    __syncthreads();
    __threadfence();
    __shared__ int s_last;
    if (tid == 0) {
        const int idx = row * 2 + blockIdx.y;
        const int prev = atomicAdd(split_cnt + idx, 1);
        if (prev == nch - 1) {
            split_cnt[idx] = 0;
            s_last = 1;
        } else {
            s_last = 0;
        }
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    {
        // the exact `attn_combine_kernel` body over this group's 32 heads, two heads per iteration across the two
        // 128-thread halves (half_id = tid >> 7, t128 = tid & 127)
        const int half_id = tid >> 7, t128 = tid & 127;
        for (int hh = 0; hh < MT * HT; hh += 2) {
            const int head = blockIdx.y * (MT * HT) + hh + half_id;
            bf16* dst = out + ((int64_t)row * NH + head) * DC + t128 * 4;
            const float2* pm = part_ml + (int64_t)row * max_chunks * NH + head;
            const float* po = part_o + (int64_t)row * max_chunks * SPLIT_SLOT_O + (int64_t)head * DC + t128 * 4;
            float mx = -INFINITY;
            for (int c = 0; c < nch; ++c) mx = fmaxf(mx, pm[(int64_t)c * NH].x);
            float L = 0.0f;
            float4 O = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int c = 0; c < nch; ++c) {
                const float2 ml = pm[(int64_t)c * NH];
                const float w = ex2(__fsub_rn(ml.x, mx));
                const float4 v = *reinterpret_cast<const float4*>(po + (int64_t)c * SPLIT_SLOT_O);
                if (c == 0) {
                    L = __fmul_rn(ml.y, w);
                    O = make_float4(__fmul_rn(v.x, w), __fmul_rn(v.y, w), __fmul_rn(v.z, w), __fmul_rn(v.w, w));
                } else {
                    L = __fmaf_rn(ml.y, w, L);
                    O.x = __fmaf_rn(v.x, w, O.x);
                    O.y = __fmaf_rn(v.y, w, O.y);
                    O.z = __fmaf_rn(v.z, w, O.z);
                    O.w = __fmaf_rn(v.w, w, O.w);
                }
            }
            const float il = L > 0.0f ? __fdiv_rn(1.0f, L) : 0.0f;
            *reinterpret_cast<uint2*>(dst) =
                make_uint2(pack2(__fmul_rn(O.x, il), __fmul_rn(O.y, il)), pack2(__fmul_rn(O.z, il), __fmul_rn(O.w, il)));
        }
    }
}

// One block per (row, head), 128 threads x 4 latent columns. Folds the row's chunk partials in ascending chunk order (see the header above).
__global__ void __launch_bounds__(128) attn_combine_kernel(
    const float* __restrict__ part_o, const float2* __restrict__ part_ml, const int* __restrict__ row_pos,
    bf16* __restrict__ out, int max_chunks) {
    const int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    const int pos = row_pos[row];
    bf16* dst = out + ((int64_t)row * NH + head) * DC + tid * 4;
    if (pos < 0) {
        *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
        return;
    }
    const int nch = pos / SPLIT_KEYS + 1;
    const float2* pm = part_ml + (int64_t)row * max_chunks * NH + head;
    const float* po = part_o + (int64_t)row * max_chunks * SPLIT_SLOT_O + (int64_t)head * DC + tid * 4;
    float mx = -INFINITY;
    for (int c = 0; c < nch; ++c) mx = fmaxf(mx, pm[(int64_t)c * NH].x);
    float L = 0.0f;
    float4 O = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int c = 0; c < nch; ++c) {
        const float2 ml = pm[(int64_t)c * NH];
        const float w = ex2(__fsub_rn(ml.x, mx));
        const float4 v = *reinterpret_cast<const float4*>(po + (int64_t)c * SPLIT_SLOT_O);
        if (c == 0) {
            L = __fmul_rn(ml.y, w);
            O = make_float4(__fmul_rn(v.x, w), __fmul_rn(v.y, w), __fmul_rn(v.z, w), __fmul_rn(v.w, w));
        } else {
            L = __fmaf_rn(ml.y, w, L);
            O.x = __fmaf_rn(v.x, w, O.x);
            O.y = __fmaf_rn(v.y, w, O.y);
            O.z = __fmaf_rn(v.z, w, O.z);
            O.w = __fmaf_rn(v.w, w, O.w);
        }
    }
    const float il = L > 0.0f ? __fdiv_rn(1.0f, L) : 0.0f;
    *reinterpret_cast<uint2*>(dst) =
        make_uint2(pack2(__fmul_rn(O.x, il), __fmul_rn(O.y, il)), pack2(__fmul_rn(O.z, il), __fmul_rn(O.w, il)));
}

// ---------------------------------------------------------------- launchers

int absorb_launch(const void* q, int64_t q_row_stride, const void* w_uk, void* q_lat, int rows, cudaStream_t s) {
    const dim3 grid((rows + BM - 1) / BM, DC / BN, NH);
    // head h's nope operands start at h*QK (192) of the q row: [0,128) nope, [128,192) rope (only nope feeds W_UK).
    mma_kernel<DN, DC, false><<<grid, ATHR, 0, s>>>((const bf16*)q, q_row_stride, QK, (const bf16*)w_uk,
                                                     (bf16*)q_lat, rows, nullptr, nullptr, nullptr, nullptr);
    return GB_CHECK_LAUNCH();
}

int attend_launch(const void* q_lat, const void* q_rot, const void* cache, const int* table, int max_blocks,
                  const int* row_slot, const int* row_pos, void* out, int rows, bool pair, cudaStream_t s) {
    // 105 KB of dynamic shared memory (two 36 KB key-tile stages + the 32 KB score exchange) exceeds the
    // 48 KB default; opt in to the device maximum. Idempotent, so do it on every launch (a failed set is
    // returned, not silently launched past).
    const cudaError_t cfg = cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmemBytes);
    if (cfg != cudaSuccess) return (int)cfg;
    if (pair) {
        // two consecutive query rows per CTA, 16 heads: grid = (ceil(rows/2), NH/HT). The 32-key tile staged
        // once serves both rows (halved per-row KV traffic); the per-warp arithmetic is the same as the
        // unpaired path.
        const cudaError_t cfgp = cudaFuncSetAttribute(attn_pair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmemBytes);
        if (cfgp != cudaSuccess) return (int)cfgp;
        const dim3 grid((rows + 1) / 2, NH / HT);
        attn_pair_kernel<<<grid, TNTHR, kAttnSmemBytes, s>>>((const bf16*)q_lat, (const bf16*)q_rot, (const bf16*)cache,
                                                              table, max_blocks, row_slot, row_pos, (bf16*)out, rows);
        return GB_CHECK_LAUNCH();
    }
    const dim3 grid(rows, NH / (MT * HT));
    attn_kernel<<<grid, TNTHR, kAttnSmemBytes, s>>>((const bf16*)q_lat, (const bf16*)q_rot, (const bf16*)cache,
                                                    table, max_blocks, row_slot, row_pos, (bf16*)out);
    return GB_CHECK_LAUNCH();
}

// Rows the split path's workspace holds per pass: SPLIT_WS_SLOTS (row, chunk) slots, whatever the table width. A call whose rows x chunks
// exceed it is launched in row batches (each row is combined on its own, so the batching moves no bit).
constexpr int SPLIT_WS_SLOTS = 320;  // 320 x 1024 keys = tables up to 327,680 tokens
constexpr int64_t SPLIT_WS_BYTES = SPLIT_WS_SLOTS * SPLIT_SLOT_BYTES;
// the fused combine's self-resetting per-(row, 32-head group) counters live AFTER the parts (2 ints per slot row:
// row x 2 head groups; the per-pass row count never exceeds SPLIT_WS_SLOTS / chunks <= 320)
constexpr int64_t SPLIT_CNT_BYTES = 2 * SPLIT_WS_SLOTS * sizeof(int);
constexpr int64_t SPLIT_WS_TOTAL_BYTES = SPLIT_WS_BYTES + SPLIT_CNT_BYTES;
int attend_split_launch(const void* q_lat, const void* q_rot, const void* cache, const int* table, int max_blocks,
                        const int* row_slot, const int* row_pos, void* out, int rows, void* ws, size_t ws_bytes,
                        cudaStream_t s) {
    // the chunk grid spans every page of the table (the launch is graph-captured once and replayed as positions grow; a chunk past a row's
    // last key exits at once)
    const int max_chunks = (max_blocks * PAGE + SPLIT_KEYS - 1) / SPLIT_KEYS;
    if (max_chunks > SPLIT_WS_SLOTS || ws == nullptr || ws_bytes < (size_t)SPLIT_WS_TOTAL_BYTES) return (int)cudaErrorInvalidValue;
    const cudaError_t cfg = cudaFuncSetAttribute(attn_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmemBytes);
    if (cfg != cudaSuccess) return (int)cfg;
    float* part_o = (float*)ws;
    float2* part_ml = (float2*)((char*)ws + (int64_t)SPLIT_WS_SLOTS * SPLIT_SLOT_O * 4);
    // the fused combine's self-resetting per-(row, 32-head group) counters. The workspace is allocated zeroed
    // (`DevBuf::zeroed`), and each pass's last block leaves its counters at 0, so re-entrant calls need no
    // per-call memset.
    int* split_cnt = (int*)((char*)ws + (int64_t)SPLIT_WS_BYTES);
    const int per_pass = SPLIT_WS_SLOTS / max_chunks;  // >= 1
    for (int base = 0; base < rows; base += per_pass) {
        const int n = rows - base < per_pass ? rows - base : per_pass;
        const dim3 grid(n, NH / (MT * HT), max_chunks);
        attn_split_kernel<<<grid, TNTHR, kAttnSmemBytes, s>>>(
            (const bf16*)q_lat + (int64_t)base * NH * DC, (const bf16*)q_rot + (int64_t)base * NH * DR, (const bf16*)cache, table,
            max_blocks, row_slot + base, row_pos + base, part_o, part_ml, max_chunks,
            (bf16*)out + (int64_t)base * NH * DC, split_cnt);
        const int rc = GB_CHECK_LAUNCH();
        if (rc) return rc;
    }
    return 0;
}

// The pre-fusion two-launch reference (split writes the parts, then `attn_combine_kernel` folds them), kept compiled
// so the A/B bit test pins the fused kernel against it. Same operands and semantics as `attend_split_launch`.
int attend_split_ref_launch(const void* q_lat, const void* q_rot, const void* cache, const int* table, int max_blocks,
                            const int* row_slot, const int* row_pos, void* out, int rows, void* ws, size_t ws_bytes,
                            cudaStream_t s) {
    const int max_chunks = (max_blocks * PAGE + SPLIT_KEYS - 1) / SPLIT_KEYS;
    if (max_chunks > SPLIT_WS_SLOTS || ws == nullptr || ws_bytes < (size_t)SPLIT_WS_TOTAL_BYTES) return (int)cudaErrorInvalidValue;
    const cudaError_t cfg = cudaFuncSetAttribute(attn_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmemBytes);
    if (cfg != cudaSuccess) return (int)cfg;
    float* part_o = (float*)ws;
    float2* part_ml = (float2*)((char*)ws + (int64_t)SPLIT_WS_SLOTS * SPLIT_SLOT_O * 4);
    const int per_pass = SPLIT_WS_SLOTS / max_chunks;  // >= 1
    for (int base = 0; base < rows; base += per_pass) {
        const int n = rows - base < per_pass ? rows - base : per_pass;
        const dim3 grid(n, NH / (MT * HT), max_chunks);
        attn_split_kernel<<<grid, TNTHR, kAttnSmemBytes, s>>>(
            (const bf16*)q_lat + (int64_t)base * NH * DC, (const bf16*)q_rot + (int64_t)base * NH * DR, (const bf16*)cache, table,
            max_blocks, row_slot + base, row_pos + base, part_o, part_ml, max_chunks, nullptr, nullptr);
        const int rc = GB_CHECK_LAUNCH();
        if (rc) return rc;
        attn_combine_kernel<<<dim3(n, NH), 128, 0, s>>>(part_o, part_ml, row_pos + base, (bf16*)out + (int64_t)base * NH * DC,
                                                       max_chunks);
        const int rc2 = GB_CHECK_LAUNCH();
        if (rc2) return rc2;
    }
    return 0;
}

int expand_launch(const void* o_lat, const void* w_uvt, void* out, int rows, cudaStream_t s) {
    const dim3 grid((rows + BM - 1) / BM, DN / BN, NH);
    mma_kernel<DC, DN, false><<<grid, ATHR, 0, s>>>((const bf16*)o_lat, (int64_t)NH * DC, DC, (const bf16*)w_uvt,
                                                     (bf16*)out, rows, nullptr, nullptr, nullptr, nullptr);
    return GB_CHECK_LAUNCH();
}

int expand_gate_launch(const void* o_lat, const void* w_uvt, const void* x, const void* gproj, void* out,
                       int rows, const void* dense_suh, void* dense_xh, cudaStream_t s) {
    const dim3 grid((rows + BM - 1) / BM, DN / BN, NH);
    mma_kernel<DC, DN, true><<<grid, ATHR, 0, s>>>((const bf16*)o_lat, (int64_t)NH * DC, DC, (const bf16*)w_uvt,
                                                   (bf16*)out, rows, (const bf16*)x, (const bf16*)gproj,
                                                   (const half*)dense_suh, (half*)dense_xh);
    return GB_CHECK_LAUNCH();
}

}  // namespace

// ---------------------------------------------------------------- exported launchers (bf16 only)
//
// Every entry validates like the scalar path it replaces. rows == 0 is a no-op; a negative rows or a
// malformed stride is cudaErrorInvalidValue. All pointers are device pointers on `stream`.

// q: bf16 [rows][q_row_stride >= 64*192] (head h at h*192, nope 0..127); w_uk: bf16 [64][128][512];
// q_lat: bf16 [rows][64][512].
GB_EXPORT int gb_ling_mla_absorb_tc(const void* q, int64_t q_row_stride, const void* w_uk, void* q_lat, int rows,
                                    void* stream) {
    if (rows < 0 || q_row_stride < (int64_t)NH * QK || (q_row_stride & 7)) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    return absorb_launch(q, q_row_stride, w_uk, q_lat, rows, gb_stream(stream));
}

// The three pre-attention steps of a Gated-MLA layer in ONE launch (bf16 only): `kv_prep` (kv: bf16
// [rows][kv_stride >= 640], cols 0..511 latent, 512..575 rope; norm_w: bf16 [512]; cos/sin: f32 tables with
// table_stride >= 32; cache: bf16 [n_blocks][256][576], written at (slot, pos) of each row) + the tensor-core
// absorb (`q_lat [rows][64][512] = q_nope @ w_uk`) + `q_rope_only` (q_rot: bf16 [rows][64][64]).
// Bit-identical per output to gb_ling_mla_absorb_tc + gb_mlag_kv_prep + gb_mlag_q_rope with the same
// operands: the absorb runs the exact mma_kernel<128, 512, false> body on grid (gm, 4, 64), the rope the
// exact q_rope_kernel<bf16> body (y == 0 CTAs), kv_prep the exact kv_prep_kernel<bf16> body (z == 64 CTAs,
// 128 threads each reproducing the 256-thread reduction).
GB_EXPORT int gb_ling_mla_prep_absorb_rope_tc(
    const void* q, int64_t q_row_stride,
    const void* kv, int kv_stride,
    const void* norm_w,
    const void* cos, const void* sin, int table_stride,
    const int* block_table, int max_blocks,
    const int* row_slot, const int* row_pos,
    const void* w_uk,
    void* q_lat, void* cache, void* q_rot,
    int rows, void* stream) {
    if (rows < 0 || q_row_stride < (int64_t)NH * QK || (q_row_stride & 7) || kv_stride < 640 ||
        table_stride < 32 || max_blocks <= 0 || !q || !kv || !norm_w || !cos || !sin || !w_uk ||
        !block_table || !row_slot || !row_pos || !q_lat || !cache || !q_rot)
        return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid((rows + BM - 1) / BM, DC / BN, NH + 1);
    prep_absorb_rope_kernel<<<grid, ATHR, 0, gb_stream(stream)>>>(
        (const bf16*)q, q_row_stride, QK, (const bf16*)w_uk, (bf16*)q_lat, rows,
        (const bf16*)kv, kv_stride, (const bf16*)norm_w,
        (const float*)cos, (const float*)sin, table_stride,
        block_table, max_blocks, row_slot, row_pos,
        (bf16*)cache, (bf16*)q_rot);
    return GB_CHECK_LAUNCH();
}

// q_lat, out: bf16 [rows][64][512]; q_rot: bf16 [rows][64][64]; cache: bf16 [n_blocks][256][576];
// row_slot, row_pos: i32 [rows]. Row r attends keys 0..=row_pos[r] of slot row_slot[r]; row_pos < 0 -> zeros.
// pair != 0: run two consecutive query rows per CTA (16 heads), sharing each staged 32-key tile (the prefill
// path: all rows share one sequence slot); 0 keeps one row x 32 heads per CTA.
GB_EXPORT int gb_ling_mla_attend_tc(const void* q_lat, const void* q_rot, const void* cache,
                                    const int* block_table, int max_blocks, const int* row_slot,
                                    const int* row_pos, void* out, int rows, int pair, void* stream) {
    if (rows < 1 || max_blocks <= 0) return (int)cudaErrorInvalidValue;
    return attend_launch(q_lat, q_rot, cache, block_table, max_blocks, row_slot, row_pos, out, rows, pair != 0,
                         gb_stream(stream));
}

// The split-key decode attention (ids: ling-mla-attn-tc32-split1024-bf16-v1). Same operands and layouts as gb_ling_mla_attend_tc and the same
// row semantics (row r attends keys 0..=row_pos[r] of slot row_slot[r]; row_pos < 0 -> zeros), but a row's key range is cut into absolute
// 1024-key chunks run on separate CTAs and merged in ascending chunk order. `ws` is gb_ling_mla_split_ws_bytes() bytes of device memory the
// caller owns (reused by every call on the stream). The chunk grid spans the block table's pages (max_blocks), so the launch is the same for
// every position and may be captured in a CUDA graph; a table wider than gb_ling_mla_split_ws_slots() chunks is cudaErrorInvalidValue.
GB_EXPORT int gb_ling_mla_attend_split_tc(const void* q_lat, const void* q_rot, const void* cache,
                                          const int* block_table, int max_blocks, const int* row_slot,
                                          const int* row_pos, void* out, int rows, void* ws, size_t ws_bytes,
                                          void* stream) {
    if (rows < 1 || max_blocks <= 0) return (int)cudaErrorInvalidValue;
    return attend_split_launch(q_lat, q_rot, cache, block_table, max_blocks, row_slot, row_pos, out, rows, ws, ws_bytes,
                               gb_stream(stream));
}

// Test-only reference: the pre-fusion two-launch split+combine sequence (the A/B bit test pins the fused
// `gb_ling_mla_attend_split_tc` against it). Same operands, same workspace, same row semantics.
GB_EXPORT int gb_ling_mla_attend_split_ref_tc(const void* q_lat, const void* q_rot, const void* cache,
                                              const int* block_table, int max_blocks, const int* row_slot,
                                              const int* row_pos, void* out, int rows, void* ws, size_t ws_bytes,
                                              void* stream) {
    if (rows < 1 || max_blocks <= 0) return (int)cudaErrorInvalidValue;
    return attend_split_ref_launch(q_lat, q_rot, cache, block_table, max_blocks, row_slot, row_pos, out, rows, ws, ws_bytes,
                                   gb_stream(stream));
}
GB_EXPORT size_t gb_ling_mla_split_ws_bytes() { return (size_t)SPLIT_WS_TOTAL_BYTES; }
GB_EXPORT int gb_ling_mla_split_keys() { return SPLIT_KEYS; }
GB_EXPORT int gb_ling_mla_split_ws_slots() { return SPLIT_WS_SLOTS; }

// o_lat: bf16 [rows][64][512]; w_uvt: bf16 [64][512][128]; out: bf16 [rows][64*128].
GB_EXPORT int gb_ling_mla_expand_tc(const void* o_lat, const void* w_uvt, void* out, int rows, void* stream) {
    if (rows < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    return expand_launch(o_lat, w_uvt, out, rows, gb_stream(stream));
}

// The expand fused with the head-wise sigmoid output gate: out[r][h*128+n] =
// (sum_c o_lat[r][h][c] * W_UV[h][c][n]) * sigmoid(x[r] . g_proj[h]). x: bf16 [rows][4096] (the layer input);
// g_proj: bf16 [64][4096]. Replaces expand + gate_apply of the scalar path in one launch. `dense_suh` (fp16
// [8192]) and `dense_xh` (fp16 [rows][8192]), when both given (decode rows <= 8 and the P2 prefill chunks),
// additionally fuse the dense projection's input rotation into the epilogue: each CTA rotates its own head's rows
// with the exact `ling_rot_in_kernel` math into `dense_xh[(r * 64 + h) * 128 + ...]`, so the dense linear can skip
// its `rot_in` launch (bit-identical per element to the separate `gb_exl3_rotate_in` on the gated output).
GB_EXPORT int gb_ling_mla_expand_gate_tc(const void* o_lat, const void* w_uvt, const void* x, const void* gproj,
                                         void* out, int rows, const void* dense_suh, void* dense_xh,
                                         void* stream) {
    if (rows < 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    if ((dense_suh == nullptr) != (dense_xh == nullptr)) return (int)cudaErrorInvalidValue;
    return expand_gate_launch(o_lat, w_uvt, x, gproj, out, rows, dense_suh, dense_xh, gb_stream(stream));
}

// ================================================================ MTP draft projection (eh_proj) over many rows
// Ling-3.1-flash-1019 MTP draft projection (`eh_proj`) over many rows: the row-batched twin of
// `mla_gated.cu`'s `gb_mlag_eh_proj_cat_bf16_kernel`, selected only by the Ling model path
// (`ling_mla_attn::eh_proj_cat_bf16`). The shared kernel stays compiled and untouched (GLM, DeepSeek and the sealed Ling
// builds keep linking the same bits); this section is new.
//
//   y[r][j] = bf16_rn( sum_{c<h} e_norm[r][c] * w[j][c]  +  sum_{c<h} h_norm[r][c] * w[j][h+c] ),
//   e_norm, h_norm bf16 [rows, h], w bf16 [h, 2h] (the 64 MiB L72 eh_proj at h = 4096), y bf16 [rows, h], h % 256 == 0.
//
// Why a twin. The shared kernel launches a warp per (row, column) and every warp streams its 16 KiB weight row, so the
// weight (64 MiB, far beyond the 24 MiB L2) is re-read from DRAM once per row: 0.277 ms per row at 242 GB/s. The first
// MTP round after a cold prompt absorbs up to the last 8,191 prompt rows through L72 (`mtp_absorb_range`) and the
// snapshot write absorbs the cut's rows again, so 1.1 s (4K) to 2.3 s (8K and up) of the first round, and two 567 ms
// calls inside a snapshot-writing request, were this one kernel re-reading one weight.
//
// Here a warp owns J consecutive output columns and walks the call's rows R at a time: one weight tile (J columns x
// the lane's uint4 of the row) is loaded once and applied to R rows, and a CTA walks `rows_per_cta` rows so the DRAM
// weight stream is paid once per CTA slab of rows (the re-reads inside a slab hit L1/L2). The CTA grid is therefore
// (h / (8 J), ceil(rows / rows_per_cta)).
//
// ROW-COUNT INVARIANCE IS THE CONTRACT, and it is the shared kernel's own arithmetic, not a new one. For every output
// (r, j) the kernel runs the shared kernel's exact chain, in the same order, with the same roundings:
//   acc = 0; for c8 = lane, lane+32, ... < h/8:  acc = fma8(acc, e_norm[r][c8*8..+8), w[j][c8*8..+8))   (e half first)
//             for c8 = lane, lane+32, ... < h/8:  acc = fma8(acc, h_norm[r][c8*8..+8), w[j][h + c8*8..+8))  (then h half)
//   fma8 = 8 ascending __fmaf_rn on bf16 -> fp32 widened operands (element order x.lo, x.hi of uint4 words .x .y .z .w);
//   acc = warp_reduce_sum5(acc) (shfl_down 16, 8, 4, 2, 1, plain fp32 adds); lane 0 stores __float2bfloat16_rn(acc).
// The accumulator of a (row, column) lives in one lane's register for the whole chain and is never combined with any
// other row's, so J, R, rows_per_cta, the grid, the row's index inside the call and the number of rows in the call
// cannot change a bit: a row computed alone (a 1-row chain step) equals the same row inside a 4,096-row absorb. The
// file is compiled --fmad=false and every product-add is an explicit __fmaf_rn, as in mla_gated.cu. Tile tails (rows
// not a multiple of R) clamp the row index to the last valid row and discard the result; they add no arithmetic to any
// valid row. The `eh_proj_rows` tests in `tests/ling_mla_attn.rs` pin this bit for bit against the shared kernel.
//
// No tensor cores: an MMA k order would change the bits. (The L72 output only steers the drafts, never the committed
// tokens, but a bit-identical draft keeps every acceptance count and every MTP round unchanged, which is what this change promises.)
namespace {

// Identical to mla_gated.cu's fma8_bf16 (copied, not shared: that file is not edited).
__device__ __forceinline__ float eh_fma8_bf16(float acc, const uint4 xu, const uint4 wu) {
    const float2 xa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu.x));
    const float2 xb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu.y));
    const float2 xc = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu.z));
    const float2 xd = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu.w));
    const float2 wa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wu.x));
    const float2 wb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wu.y));
    const float2 wc = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wu.z));
    const float2 wd = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wu.w));
    acc = __fmaf_rn(xa.x, wa.x, acc);
    acc = __fmaf_rn(xa.y, wa.y, acc);
    acc = __fmaf_rn(xb.x, wb.x, acc);
    acc = __fmaf_rn(xb.y, wb.y, acc);
    acc = __fmaf_rn(xc.x, wc.x, acc);
    acc = __fmaf_rn(xc.y, wc.y, acc);
    acc = __fmaf_rn(xd.x, wd.x, acc);
    acc = __fmaf_rn(xd.y, wd.y, acc);
    return acc;
}

// Identical to mla_gated.cu's warp_reduce_sum5.
__device__ __forceinline__ float eh_warp_reduce_sum5(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

// One input half of the chain: acc[jj][rr] += over c8 = lane, lane+32, ... of fma8(x[rr][c8], w[jj][c8]).
template <int J, int R>
__device__ __forceinline__ void eh_half_chain(float (&acc)[J][R], const bf16* const (&xr)[R],
                                           const bf16* const (&wj)[J], int h8, int lane) {
#pragma unroll 2
    for (int c8 = lane; c8 < h8; c8 += 32) {
        uint4 wu[J];
#pragma unroll
        for (int jj = 0; jj < J; ++jj) wu[jj] = *reinterpret_cast<const uint4*>(wj[jj] + c8 * 8);
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            const uint4 xu = *reinterpret_cast<const uint4*>(xr[rr] + c8 * 8);
#pragma unroll
            for (int jj = 0; jj < J; ++jj) acc[jj][rr] = eh_fma8_bf16(acc[jj][rr], xu, wu[jj]);
        }
    }
}

template <int J, int R>
__global__ void __launch_bounds__(256)
eh_proj_rows_kernel(const bf16* __restrict__ e_norm, const bf16* __restrict__ h_norm, const bf16* __restrict__ w,
                    bf16* __restrict__ y, int rows, int h, int rows_per_cta) {
    const int lane = threadIdx.x & 31;
    const int j0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * J;  // first of this warp's J columns (h % (8 J) == 0)
    const int row_lo = blockIdx.y * rows_per_cta;
    const int row_hi = min(rows, row_lo + rows_per_cta);
    const int h8 = h >> 3;
    const bf16* wj[J];
#pragma unroll
    for (int jj = 0; jj < J; ++jj) wj[jj] = w + (int64_t)(j0 + jj) * (2 * h);
    for (int r0 = row_lo; r0 < row_hi; r0 += R) {
        const bf16* er[R];
        const bf16* hr[R];
        int rv[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            rv[rr] = min(r0 + rr, row_hi - 1);  // a tail tile recomputes the last valid row; its copy is never stored
            er[rr] = e_norm + (int64_t)rv[rr] * h;
            hr[rr] = h_norm + (int64_t)rv[rr] * h;
        }
        float acc[J][R];
#pragma unroll
        for (int jj = 0; jj < J; ++jj)
#pragma unroll
            for (int rr = 0; rr < R; ++rr) acc[jj][rr] = 0.0f;
        eh_half_chain<J, R>(acc, er, wj, h8, lane);
        const bf16* wh[J];
#pragma unroll
        for (int jj = 0; jj < J; ++jj) wh[jj] = wj[jj] + h;
        eh_half_chain<J, R>(acc, hr, wh, h8, lane);
#pragma unroll
        for (int jj = 0; jj < J; ++jj)
#pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                const float s = eh_warp_reduce_sum5(acc[jj][rr]);
                if (lane == 0 && r0 + rr < row_hi) y[(int64_t)(r0 + rr) * h + j0 + jj] = __float2bfloat16_rn(s);
            }
    }
}

template <int J, int R>
int eh_launch_rows(const void* e_norm, const void* h_norm, const void* w, void* y, int rows, int h, int rows_per_cta,
                cudaStream_t s) {
    if (h % (8 * J) != 0) return (int)cudaErrorInvalidValue;
    const dim3 grid(h / (8 * J), (rows + rows_per_cta - 1) / rows_per_cta);
    eh_proj_rows_kernel<J, R><<<grid, 256, 0, s>>>((const bf16*)e_norm, (const bf16*)h_norm, (const bf16*)w,
                                                    (bf16*)y, rows, h, rows_per_cta);
    return GB_CHECK_LAUNCH();
}

}  // namespace

// The tile the Ling model path runs (J = 2, R = 16, 1,024-row slabs: the fastest spill-free tile of a bench on N2, see the PR body).
#ifndef LING_EH_J
#define LING_EH_J 2
#endif
#ifndef LING_EH_R
#define LING_EH_R 16
#endif
#ifndef LING_EH_SLAB
#define LING_EH_SLAB 1024
#endif

// Fused eh_proj cat over `rows` rows, bit-identical per row to gb_mlag_eh_proj_cat_bf16 (see the header).
GB_EXPORT int gb_ling_eh_proj_cat_bf16(const void* e_norm, const void* h_norm, const void* w, void* y, int rows, int h,
                                       void* stream) {
    if (rows < 0 || h <= 0 || h % 256 != 0) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    return eh_launch_rows<LING_EH_J, LING_EH_R>(e_norm, h_norm, w, y, rows, h, LING_EH_SLAB, gb_stream(stream));
}

// The tile and slab as arguments, for the bit tests and the tile-size bench only (the model path never calls this).
// Returns cudaErrorInvalidValue for a (j, r) pair that is not instantiated.
GB_EXPORT int gb_ling_eh_proj_cat_bf16_tile(const void* e_norm, const void* h_norm, const void* w, void* y, int rows,
                                            int h, int j, int r, int rows_per_cta, void* stream) {
    if (rows < 0 || h <= 0 || h % 256 != 0 || rows_per_cta < 1) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    cudaStream_t s = gb_stream(stream);
#define LING_EH_CASE(JJ, RR) \
    if (j == JJ && r == RR) return eh_launch_rows<JJ, RR>(e_norm, h_norm, w, y, rows, h, rows_per_cta, s);
    LING_EH_CASE(1, 1) LING_EH_CASE(1, 3) LING_EH_CASE(1, 8) LING_EH_CASE(2, 1) LING_EH_CASE(2, 4) LING_EH_CASE(2, 8)
    LING_EH_CASE(2, 16) LING_EH_CASE(8, 4)
#undef LING_EH_CASE
    return (int)cudaErrorInvalidValue;
}
