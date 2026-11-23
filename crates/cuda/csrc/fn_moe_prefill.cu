// gb-nvcc: --expt-relaxed-constexpr --fmad=false
// Qwen3.8-Flash-Next prefill: the routed experts' two grouped GEMMs (gate/up with the SwiGLU epilogue, down) for a prompt chunk, with
// the ARITHMETIC of csrc/moe_prefill.cu (so of the decode path, csrc/moe_experts.cu) and the SCHEDULE of csrc/glm_p2_moe.cu: warps
// that own output columns, fed by a cp.async ring. The Rust side is crates/cuda/src/fn_moe_prefill.rs.
//
// The arithmetic of a (row, expert) pair, every rounding point and every sum order csrc/moe_prefill.cu's:
//   input      A_g, A_u [sorted pair][K] fp16 = fp16((x[row] * suh[e]) H_128 / sqrt 128): rot_in's values, stored row-major;
//   gate / up  per output, K in 4 consecutive ranges of PW = K / 16 / 4 k tiles (the decode plan's four warps at one K split; a
//              plan with more splits is refused): range c's sum is one fresh fp32 m16n8k16 chain from +0.0 over its k tiles in
//              order, and the output is ((c0 + c1) + c2) + c3, fp32 round to nearest: `reduce_sweep`'s `s = red[0]; s += red[w]`;
//              then H_128 / sqrt 128 * svh, silu(gate) clamped from above and up clamped to +-limit (limit > 0), the product,
//              * suh_down, H_128 / sqrt 128, fp16: the down projection's A row (gateup_kernel's epilogue, expression for expression);
//   down       the K sum in the same order, H_128 / sqrt 128 * svh_down: the pair's fp32 y (down_kernel's epilogue).
// What differs is who holds an accumulator and when: in csrc/moe_prefill.cu a warp owns a K range and every column, and the four
// ranges meet in shared memory; here a warp owns columns and walks all of K, closing a chain at each range's last k tile and adding
// it to its running sum in the same order. mma keeps rows apart, padding rows are zero-filled and never stored, and nothing is
// atomic, so a pair's bits depend on its row and its expert's weights only (tests/fn_moe_prefill.rs: bit for bit against
// csrc/moe_prefill.cu at every row count and routing style).
//
// Schedule. The plan is csrc/moe_prefill.cu's (gb_moe_pf_plan_parallel: the pairs sorted by expert, each expert's pairs padded to
// 16-row tiles; its per-expert tile offsets and pair counts), cut into SLICES of at most 16 tiles of one expert, in expert order
// (gb_fn_pf_slices). One block per (128 output columns, slice) walks the slice in groups of at most four tiles. Gate and up run in one block (warps 0-3 gate, 4-7 up, 32 columns a
// warp); down has 16 columns a warp. A 3-stage cp.async ring of 64 k streams the A rows (16-byte chunks XOR-swizzled by row for
// conflict-free ldmatrix) and the raw trellis tiles; each warp decodes its own tiles once per k tile and group into B fragments
// (exl3_linear_decode.cuh) and multiplies the group's m16 tiles by them. An expert's stripe of packed weights comes from memory for
// its first group and from the cache for the rest.
//
// Derived from csrc/glm_p2_moe.cu (the ring, the swizzle, the column-owning warps: REXL3's own) and csrc/moe_prefill.cu (the
// arithmetic); their headers carry the upstream notices (TensorFold and ExLlamaV3, MIT; see THIRD_PARTY_NOTICES.md).

#include <math.h>
#include <stdint.h>

#include <type_traits>

#include "common.cuh"
#include "exl3_linear_decode.cuh"

using namespace gbx;

namespace {

constexpr int kNW = 8;       // warps a block
constexpr int kBK = 64;      // k of one ring stage (4 k tiles)
constexpr int kStages = 3;
constexpr int kMT = 4;       // 16-row tiles a group holds at most (they share every tile decode)
constexpr int kGroups = 4;   // groups a slice holds at most: a hotter expert is several slices, so its work spreads over the machine
constexpr int kSliceThreads = 1024;  // one expert a thread: E <= 1024
constexpr int kLds = 128 + 4;  // floats a row of the epilogue staging

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

__device__ __forceinline__ void load4h(const half* p, float (&v)[4]) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

__device__ __forceinline__ void store4h(half* p, const float (&v)[4]) {
    const half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&a);
    u.y = *reinterpret_cast<const uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// The half offset of 16-byte chunk c (0..7) of row r in an A stage (rows of 64 values): chunk c of row r lands at c ^ (r & 7), so the 8
// rows an ldmatrix phase reads hit 8 distinct chunks.
__device__ __forceinline__ int a_off(int r, int c) { return r * kBK + ((c ^ (r & 7)) << 3); }

// A stage of the ring: per matrix the group's A rows (16 kMT rows of 64 fp16) then 4 k tiles x 8 column tiles of raw trellis words.
template <int K2>
constexpr int mstage_bytes() {
    return 16 * kMT * kBK * 2 + (kBK / 16) * 8 * tile_words<K2>() * 4;
}
template <int K2, int MATS>
constexpr int smem_bytes() {
    constexpr int ring = kStages * MATS * mstage_bytes<K2>();
    constexpr int epi = MATS * 16 * kMT * kLds * 4;
    return ring > epi ? ring : epi;
}

struct Args {
    const half* a0;          // A rows of matrix 0 [S][K] fp16 (sorted rows)
    const half* a1;          // matrix 1 (MATS = 2)
    const uint64_t* t0;      // per-expert trellis device addresses of matrix 0 [E]
    const uint64_t* t1;      // matrix 1
    const int4* slices;      // [0].x = the number of slices; then {expert, first sorted row, tiles, rows of pairs} a slice
    const half* svh0;        // [E][N] output scales of matrix 0
    const half* svh1;        // matrix 1
    const half* suh_next;    // [E][N] the down projection's input scales (MATS = 2)
    half* out16;             // MATS = 2: the down projection's A rows [S][N] fp16
    float* y;                // MATS = 1: y [P][N] fp32 at the pair of each sorted row
    const int* spair;
    int K, N;
    int PW;                  // k tiles of one K range (a chain)
    float limit;
};

// One group: `tiles` (1..kMT) 16-row tiles of expert e from sorted row `row0`, `vrows` of them pairs. Leaves the group's outputs in
// global memory. MATS = 2: gate and up with the SwiGLU epilogue; MATS = 1: down.
template <int K2, int MATS>
__device__ __forceinline__ void run_group(const Args& g, unsigned char* smem, int e, int row0, int tiles, int vrows) {
    constexpr int WPM = kNW / MATS;                     // warps a matrix
    constexpr int TPW = 8 / WPM;                        // 16-column tiles a warp
    constexpr int kThreads = kNW * 32;
    constexpr int TW = tile_words<K2>();
    constexpr int A_BYTES = 16 * kMT * kBK * 2;         // one matrix's A stage
    constexpr int B_WORDS = (kBK / 16) * 8 * TW;        // one matrix's B stage: 4 k tiles x 8 column tiles
    constexpr int MSTAGE = mstage_bytes<K2>();
    constexpr int STAGE = MATS * MSTAGE;

    const int nb = blockIdx.x;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int mat = warp / WPM, wm = warp % WPM;        // this warp's matrix and its index among the matrix's warps
    const int K = g.K, N = g.N;
    const int KC = K / kBK;
    const int tiles_n = N / 16;
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(g.t0[e]) + (size_t)nb * 8 * TW;
    const uint32_t* T1 = MATS == 2 ? reinterpret_cast<const uint32_t*>(g.t1[e]) + (size_t)nb * 8 * TW : T0;

    auto stage_a = [&](int s, int m) { return reinterpret_cast<half*>(smem + s * STAGE + m * MSTAGE); };
    auto stage_b = [&](int s, int m) { return reinterpret_cast<uint32_t*>(smem + s * STAGE + m * MSTAGE + A_BYTES); };

    // a stage's copies: the A rows (16 kMT rows x 8 chunks a matrix; rows past `vrows` zero-filled) and the trellis tiles (the 8
    // column tiles of a k tile are contiguous in memory)
    auto load_stage = [&](int s, int kc) {
        constexpr int ACH = 16 * kMT * 8;                 // A chunks a matrix
#pragma unroll
        for (int i = 0; i < (MATS * ACH + kThreads - 1) / kThreads; ++i) {
            const int idx = i * kThreads + t;
            if (idx < MATS * ACH) {
                const int m = idx / ACH, rem = idx % ACH, r = rem >> 3, c = rem & 7;
                const bool ok = r < vrows;
                const half* src = (m ? g.a1 : g.a0) + (size_t)(row0 + (ok ? r : 0)) * K + kc * kBK + c * 8;
                cp_async16(stage_a(s, m) + a_off(r, c), src, ok);
            }
        }
        constexpr int BCH = B_WORDS / 4;                  // 16-byte chunks a matrix
#pragma unroll
        for (int i = 0; i < (MATS * BCH + kThreads - 1) / kThreads; ++i) {
            const int idx = i * kThreads + t;
            if (idx < MATS * BCH) {
                const int m = idx / BCH, rem = idx % BCH;
                const int kl = rem / (8 * TW / 4), w = rem % (8 * TW / 4);
                const uint32_t* src = (m ? T1 : T0) + (size_t)(kc * (kBK / 16) + kl) * tiles_n * TW + w * 4;
                cp_async16(stage_b(s, m) + kl * 8 * TW + w * 4, src, true);
            }
        }
    };

    // The open chain, and the closed chains' sum in range order. The sum starts at -0.0: under IEEE round to nearest -0.0 + c is c
    // itself for every c (either zero keeps its sign, anything else is exact), so adding range 0's chain to it IS the assignment
    // `s = red[0]` of csrc/moe_prefill.cu, with no first-range case in the loop (a select there spills registers).
    float acc[kMT][TPW][2][4];
    float tot[kMT][TPW][2][4];
#pragma unroll
    for (int p = 0; p < kMT; ++p)
#pragma unroll
        for (int j = 0; j < TPW; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    acc[p][j][h][c] = 0.f;
                    tot[p][j][h][c] = -0.f;
                }

    __syncthreads();  // nobody still reads the previous group's epilogue staging, which the ring overwrites
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < KC) load_stage(s, s);
        cp_async_commit();
    }

    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1), lc = lane >> 4;
    const int PW = g.PW;

    auto kloop = [&](auto ma_c) {
        constexpr int MA = decltype(ma_c)::value;
        int left = PW;       // k tiles until the open chain closes
#pragma unroll 1
        for (int kc = 0; kc < KC; ++kc) {
            cp_async_wait<kStages - 2>();
            __syncthreads();  // stage kc has landed for every thread; nobody reads the slot the next load overwrites
            {
                const int nk = kc + kStages - 1;
                if (nk < KC) load_stage(nk % kStages, nk);
                cp_async_commit();
            }
            const half* sa = stage_a(kc % kStages, mat);
            const uint32_t* sb = stage_b(kc % kStages, mat);
#pragma unroll
            for (int kk = 0; kk < kBK / 16; ++kk) {
                uint32_t b[TPW][2][2];
#pragma unroll
                for (int j = 0; j < TPW; ++j) {
                    uint32_t w[lane_words<K2>()];
                    load_lane_words<K2>(sb + (kk * 8 + wm * TPW + j) * TW, lane, w);
                    decode_lane<K2, CB_MUL1>(w, lane, b[j][0], b[j][1]);
                }
#pragma unroll
                for (int p = 0; p < MA; ++p) {
                    uint32_t a[4];
                    ldsm_x4(a, sa + a_off(p * 16 + lr, kk * 2 + lc));
#pragma unroll
                    for (int j = 0; j < TPW; ++j) {
                        mma16816(acc[p][j][0], a, b[j][0]);
                        mma16816(acc[p][j][1], a, b[j][1]);
                    }
                }
                if (--left == 0) {  // the range's last k tile: its chain joins the running sum, in range order
#pragma unroll
                    for (int p = 0; p < MA; ++p)
#pragma unroll
                        for (int j = 0; j < TPW; ++j)
#pragma unroll
                            for (int h = 0; h < 2; ++h)
#pragma unroll
                                for (int c = 0; c < 4; ++c) {
                                    tot[p][j][h][c] = __fadd_rn(tot[p][j][h][c], acc[p][j][h][c]);
                                    acc[p][j][h][c] = 0.f;
                                }
                    left = PW;
                }
            }
        }
    };
    switch (tiles) {
        case 1: kloop(std::integral_constant<int, 1>{}); break;
        case 2: kloop(std::integral_constant<int, 2>{}); break;
        case 3: kloop(std::integral_constant<int, 3>{}); break;
        default: kloop(std::integral_constant<int, kMT>{}); break;
    }
    cp_async_wait<0>();
    __syncthreads();  // the ring's bytes become the epilogue staging

    // the sums of the group's rows through shared memory: matrix m's [16 kMT][kLds] f32 at E + m * 16 kMT * kLds
    float* E = reinterpret_cast<float*>(smem);
    {
        const int g8 = lane >> 2, t4 = lane & 3;
        float* Em = E + mat * 16 * kMT * kLds;
#pragma unroll
        for (int p = 0; p < kMT; ++p) {
            if (p < tiles) {
#pragma unroll
                for (int j = 0; j < TPW; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (wm * TPW + j) * 16 + h * 8 + 2 * t4;
                        const int r = p * 16 + g8;
                        *reinterpret_cast<float2*>(Em + r * kLds + col) = make_float2(tot[p][j][h][0], tot[p][j][h][1]);
                        *reinterpret_cast<float2*>(Em + (r + 8) * kLds + col) = make_float2(tot[p][j][h][2], tot[p][j][h][3]);
                    }
            }
        }
    }
    __syncthreads();

    const int n = nb * 128 + 4 * lane;
    if constexpr (MATS == 2) {
        // csrc/moe_prefill.cu's gateup epilogue, expression for expression, on these sums
        float sg4[4], su4[4], sd4[4];
        load4h(g.svh0 + (size_t)e * N + n, sg4);
        load4h(g.svh1 + (size_t)e * N + n, su4);
        load4h(g.suh_next + (size_t)e * N + n, sd4);
        const float* Eg = E;
        const float* Eu = E + 16 * kMT * kLds;
        const float limit = g.limit;
        for (int r = warp; r < vrows; r += kNW) {
            const float4 g4 = *reinterpret_cast<const float4*>(Eg + r * kLds + 4 * lane);
            const float4 u4 = *reinterpret_cast<const float4*>(Eu + r * kLds + 4 * lane);
            float gv[4] = {g4.x, g4.y, g4.z, g4.w}, uv[4] = {u4.x, u4.y, u4.z, u4.w}, v[4];
            fwht128(gv, lane);
            fwht128(uv, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gg = gv[j] * HAD_SCALE * sg4[j];
                float u = uv[j] * HAD_SCALE * su4[j];
                float xx = silu_f(gg);
                if (limit > 0.0f) {
                    xx = fminf(xx, limit);
                    u = fminf(fmaxf(u, -limit), limit);
                }
                v[j] = (xx * u) * sd4[j];
            }
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
            store4h(g.out16 + (size_t)(row0 + r) * N + n, v);
        }
    } else {
        // csrc/moe_prefill.cu's down epilogue: H, * svh_down, the pair's fp32 y
        float sv[4];
        load4h(g.svh0 + (size_t)e * N + n, sv);
        for (int r = warp; r < vrows; r += kNW) {
            const float4 u = *reinterpret_cast<const float4*>(E + r * kLds + 4 * lane);
            float v[4] = {u.x, u.y, u.z, u.w};
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * sv[j];
            *reinterpret_cast<float4*>(g.y + (size_t)g.spair[row0 + r] * N + n) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
}

// Block (x = 128-column block, y = slice): the slice's groups one after another. The slice is uniform over the block's threads, so
// every barrier inside a group is reached by all of them.
template <int K2, int MATS>
__global__ void __launch_bounds__(kNW * 32, MATS == 1 ? 2 : 1) gemm_kernel(const Args g) {
    extern __shared__ __align__(128) unsigned char smem[];
    if ((int)blockIdx.y >= g.slices[0].x) return;
    const int4 sl = g.slices[1 + blockIdx.y];
#pragma unroll 1
    for (int g0 = 0; g0 < sl.z; g0 += kMT)
        run_group<K2, MATS>(g, smem, sl.x, sl.y + 16 * g0, min(kMT, sl.z - g0), min(16 * kMT, sl.w - 16 * g0));
}

// One block, one expert a thread: the expert's tiles cut into slices of at most kGroups * kMT, the slices of all experts listed in
// expert order behind their count. A function of the per-expert counts alone.
__global__ void __launch_bounds__(kSliceThreads) slices_kernel(const int* __restrict__ toff, const int* __restrict__ cnt,
                                                               int4* __restrict__ slices, int E) {
    __shared__ int ws[33];
    constexpr int ST = kGroups * kMT;
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const int c = e < E ? cnt[e] : 0;
    const int tiles = (c + 15) >> 4, ns = (tiles + ST - 1) / ST;
    // exclusive prefix sum of ns over the block (csrc/moe_prefill.cu's block_excl_scan)
    int inc = ns;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int x = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += x;
    }
    if (lane == 31) ws[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int w = ws[lane];
        int sum = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, sum, o);
            if (lane >= o) sum += x;
        }
        ws[lane] = sum - w;
        if (lane == 31) ws[32] = sum;
    }
    __syncthreads();
    const int off = ws[warp] + inc - ns;
    if (e == 0) slices[0] = make_int4(ws[32], 0, 0, 0);
    for (int z = 0; z < ns; ++z)
        slices[1 + off + z] = make_int4(e, (toff[e] + z * ST) * 16, min(ST, tiles - z * ST), min(16 * ST, c - 16 * ST * z));
}

// Slices at most for `sorted_rows` padded sorted rows over E experts (one partial slice an expert).
long long slice_cap(long long sorted_rows, int E) { return sorted_rows / (16 * kGroups * kMT) + E; }

template <int K2, int MATS>
int launch(const Args& a, long long cap, cudaStream_t st) {
    constexpr int sm = smem_bytes<K2, MATS>();
    static_assert(sm <= 101376, "a block's shared memory must fit what an SM gives one");
    static bool attr = false;  // idempotent; a race between host threads sets the same value twice
    if (!attr) {
        const cudaError_t e = cudaFuncSetAttribute(gemm_kernel<K2, MATS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        if (e != cudaSuccess) return (int)e;
        attr = true;
    }
    gemm_kernel<K2, MATS><<<dim3((unsigned)(a.N / 128), (unsigned)cap), kNW * 32, sm, st>>>(a);
    return GB_CHECK_LAUNCH();
}

template <int MATS>
int dispatch(int bits, const Args& a, long long cap, cudaStream_t st) {
    switch (bits) {
        case 2: return launch<4, MATS>(a, cap, st);
        case 3: return launch<6, MATS>(a, cap, st);
        case 4: return launch<8, MATS>(a, cap, st);
        default: return (int)cudaErrorInvalidValue;
    }
}

// K in four ranges of whole k tiles at one K split (gb_moe_default_plan's sk = 1, wk = 4).
bool gemm_args_ok(int K, int N, int sk, int wk, int E, long long sorted_rows) {
    return sorted_rows >= 16 && K > 0 && N > 0 && K % 128 == 0 && N % 128 == 0 && sk == 1 && wk == 4 && (K / 16) % 4 == 0 && E >= 1 &&
           E <= kSliceThreads;
}

}  // namespace

// Whether these GEMMs take a bank: the Mul1 codebook at 2 to 4 bits, widths multiples of 128, and the decode plan's K order one
// they reproduce (one K split, four warps, for gate/up and for down). Any other bank keeps csrc/moe_prefill.cu's launches.
GB_EXPORT int gb_fn_pf_supported(int hidden, int inter, int bits_gu, int bits_d, int codebook, int sk_gu, int wk_gu, int sk_d, int wk_d) {
    return hidden > 0 && inter > 0 && hidden % 128 == 0 && inter % 128 == 0 && bits_gu >= 2 && bits_gu <= 4 && bits_d >= 2 && bits_d <= 4 &&
           codebook == CB_MUL1 && sk_gu == 1 && wk_gu == 4 && sk_d == 1 && wk_d == 4;
}

// int4 entries the slice list needs for `sorted_rows` padded sorted rows (the plan's capacity) over E experts: its count, then the
// slices.
GB_EXPORT long long gb_fn_pf_slice_entries(long long sorted_rows, int E) { return 1 + slice_cap(sorted_rows, E); }

// The slice list of a plan: toff / cnt = gb_moe_pf_plan_parallel's per-expert tile offsets and pair counts [E]; slices
// [gb_fn_pf_slice_entries] int4.
GB_EXPORT int gb_fn_pf_slices(const void* toff, const void* cnt, void* slices, int E, void* stream) {
    if (E < 1 || E > kSliceThreads || !toff || !cnt || !slices) return (int)cudaErrorInvalidValue;
    slices_kernel<<<1, kSliceThreads, 0, gb_stream(stream)>>>(static_cast<const int*>(toff), static_cast<const int*>(cnt),
                                                              static_cast<int4*>(slices), E);
    return GB_CHECK_LAUNCH();
}

// Gate and up with the SwiGLU epilogue (K = the model width, N = the expert width): ad [padded rows][N] fp16 row-major, the down
// projection's A rows. a0 / a1: the rotated inputs [padded rows][K] fp16 row-major (gb_glm_p2m_rotate: rot_in's values). tg / tu:
// per-expert trellis device addresses (uint64 [E]). slices: gb_fn_pf_slices's list. sk / wk: the layer's decode plan for two
// matrices (refused unless 1 and 4). sorted_rows: the most padded sorted rows the chunk can have (it sizes the grid; blocks past the
// list's count return). limit <= 0: no clamp.
GB_EXPORT int gb_fn_pf_gateup(const void* a0, const void* a1, const void* tg, const void* tu, const void* slices, const void* svh_g,
                              const void* svh_u, const void* suh_d, void* ad, int K, int N, int bits, int codebook, int sk, int wk, int E,
                              long long sorted_rows, float limit, void* stream) {
    if (!gemm_args_ok(K, N, sk, wk, E, sorted_rows) || codebook != CB_MUL1 || !a0 || !a1 || !tg || !tu || !slices || !svh_g || !svh_u ||
        !suh_d || !ad)
        return (int)cudaErrorInvalidValue;
    const Args a{static_cast<const half*>(a0), static_cast<const half*>(a1), static_cast<const uint64_t*>(tg),
                 static_cast<const uint64_t*>(tu), static_cast<const int4*>(slices), static_cast<const half*>(svh_g),
                 static_cast<const half*>(svh_u), static_cast<const half*>(suh_d), static_cast<half*>(ad), nullptr, nullptr, K, N,
                 K / 16 / 4, limit};
    return dispatch<2>(bits, a, slice_cap(sorted_rows, E), gb_stream(stream));
}

// Down with its epilogue (K = the expert width, N = the model width): y [P][N] fp32 at the pair of each sorted row (`spair`).
// sk / wk: the layer's decode plan for one matrix (refused unless 1 and 4).
GB_EXPORT int gb_fn_pf_down(const void* ad, const void* td, const void* slices, const void* svh_d, const void* spair, void* y, int K, int N,
                            int bits, int codebook, int sk, int wk, int E, long long sorted_rows, void* stream) {
    if (!gemm_args_ok(K, N, sk, wk, E, sorted_rows) || codebook != CB_MUL1 || !ad || !td || !slices || !svh_d || !spair || !y)
        return (int)cudaErrorInvalidValue;
    const Args a{static_cast<const half*>(ad), nullptr, static_cast<const uint64_t*>(td), nullptr, static_cast<const int4*>(slices),
                 static_cast<const half*>(svh_d), nullptr, nullptr, nullptr, static_cast<float*>(y), static_cast<const int*>(spair), K, N,
                 K / 16 / 4, 0.f};
    return dispatch<1>(bits, a, slice_cap(sorted_rows, E), gb_stream(stream));
}
