// ling_fmoe.cu — Ling-owned FrostMoE decode (bailing_moe_v3 / Ling-3.1), bit-identical to fmoe.cu.
//
// This file is the decode machinery of `fmoe.cu` (route, gate/up, down) with two Ling-owned changes, and is
// registered as a Ling-owned file (bailing_moe_v3 scope), so `fmoe.cu` stays untouched for the other 7 families:
//
//  * ling_route_group_kernel: the group-limited noaux_tc router with each lane's 16 experts' sigmoid score su[i]
//    and biased score sb[i] cached ONCE per token in read-only registers (the same math as fmoe::p_route_group_kernel
//    lines 1289..1345, fed through fmoe::route_group_kernel's single-CTA structure): 1x expf per expert instead of
//    the legacy 9x reload, STACK 0 / LOCAL 0, and 0 bit diffs.
//
//  * ling_router_and_route_group_kernel: the decode router GEMV (logits [T][512] fp32) and the group-limited route
//    kernel fused into ONE 32-CTA launch; the last CTA to finish (a self-resetting atomic counter in the workspace)
//    runs the routing body. The GEMV per column is bit-identical to ling_bf16_linear_f32_kernel, so the route
//    buffer keeps its bits.
//  * ling_gate_up_kernel (the 4-CTA/SM global-partial kernel) and ling_down_kernel self-reset their atomic counters
//    when the last warp finishes, so no zero_i32 launches are needed (workspace contract: zeroed once at allocation).
//
// Everything else (route_kernel, gate_up_kernel, down_kernel, the workspace layout) is a verbatim copy of the
// fmoe.cu kernels, renamed ling_* so the translation unit links beside fmoe.cu.

#include "common.cuh"
#include "exl3_linear_decode.cuh"
#include <cstdlib>
#include <cstring>

namespace lingfmoe {

constexpr int GT = 8;             // tokens a group (mma n8)
constexpr int WARPS = 4;          // warps a block
constexpr float RS128 = 0.088388347648318440f;   // 1 / sqrt(128)
constexpr float MUL1_SCALE = 1774.0f / 262144.0f; // fp16 0x1EEE
constexpr float MUL1_BIAS = -10.3828125f;         // fp16 0xC931
constexpr uint32_t MUL1_MUL = 0x83DCD12Du;
// Route buffer: int32 words. hdr[0] = groups U, hdr[1] = entries E, hdr[2] = tokens T, hdr[3] = entries a token.
// ---------------------------------------------------------------------------------------------------------------
struct Route {
    int* hdr;
    int* g_exp;   // [emax] expert of group
    int* g_ntok;  // [emax] tokens of group (1..8)
    int* g_off;   // [emax] first entry slot of group
    int* s_tok;   // [emax] token of entry slot
    float* s_w;   // [emax] routing weight of entry slot
    int* e_slot;  // [emax] (token * per_tok + k) -> entry slot
};

__host__ __device__ inline Route route_view(void* base, int emax) {
    int* p = static_cast<int*>(base);
    Route r;
    r.hdr = p;
    r.g_exp = p + 4;
    r.g_ntok = r.g_exp + emax;
    r.g_off = r.g_ntok + emax;
    r.s_tok = r.g_off + emax;
    r.s_w = reinterpret_cast<float*>(r.s_tok + emax);
    r.e_slot = reinterpret_cast<int*>(r.s_w + emax);
    return r;
}

inline size_t route_bytes(int emax) { return (size_t)(4 + 6 * emax) * 4; }
// Device helpers
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t h2u(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

// D = A (16x16 f16, row) * B (16x8 f16, col) + D, fp32 accumulate.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Unnormalized Sylvester Hadamard of 128 values spread 4 a lane (element 4 * lane + i).
__device__ __forceinline__ void had128(float (&v)[4], int lane) {
    const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2;
    v[1] = a1 + a3;
    v[2] = a0 - a2;
    v[3] = a1 - a3;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool hi = (lane & m) != 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float p = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = hi ? (p - v[i]) : (v[i] + p);
        }
    }
}

__device__ __forceinline__ void ld_bf16x4(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

__device__ __forceinline__ void ld_bf16x4_g(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

__device__ __forceinline__ void ld_f16x4(const half* p, float (&v)[4]) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

__device__ __forceinline__ void st_f16x4(half* p, const float (&v)[4]) {
    uint2 u;
    u.x = h2u(v[0], v[1]);
    u.y = h2u(v[2], v[3]);
    *reinterpret_cast<uint2*>(p) = u;
}

// The rmsnorm bit helpers of ling_kda_proj_fused.cu (the exact `ling_norm_rot_in_{triple,pair,single}_kernel`
// math runs explicit RN intrinsics; every expression is a lone multiply/add, so no FMA can form under this TU's
// default fmad=true either).
__device__ __forceinline__ float lpf_bf2f(uint16_t u) { return __uint_as_float((uint32_t)u << 16); }
__device__ __forceinline__ uint16_t lpf_f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ void lpf_unpack8(const uint4& u, float (&f)[8]) {
    f[0] = __uint_as_float(u.x << 16); f[1] = __uint_as_float(u.x & 0xFFFF0000u);
    f[2] = __uint_as_float(u.y << 16); f[3] = __uint_as_float(u.y & 0xFFFF0000u);
    f[4] = __uint_as_float(u.z << 16); f[5] = __uint_as_float(u.z & 0xFFFF0000u);
    f[6] = __uint_as_float(u.w << 16); f[7] = __uint_as_float(u.w & 0xFFFF0000u);
}
__device__ __forceinline__ uint32_t lpf_pack2(float a, float b) {
    return (uint32_t)lpf_f2bf(a) | ((uint32_t)lpf_f2bf(b) << 16);
}
__device__ __forceinline__ uint4 lpf_pack8(const float (&f)[8]) {
    uint4 u;
    u.x = lpf_pack2(f[0], f[1]); u.y = lpf_pack2(f[2], f[3]);
    u.z = lpf_pack2(f[4], f[5]); u.w = lpf_pack2(f[6], f[7]);
    return u;
}
__device__ __forceinline__ float lpf_warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ void st_bf16x4(__nv_bfloat16* p, const float (&v)[4]) {
    __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
}

__device__ __forceinline__ float4 ldcg4(const float* p) { return __ldcg(reinterpret_cast<const float4*>(p)); }
template <int K2>
struct GeoK {
    static constexpr int NW = 4 * K2;                 // 32-bit words a tile
    static constexpr int NL = gbx::lane_words<K2>();  // words a lane's 8 windows span
};

// First word and bit offset (from that word's MSB) of lane's first window.
template <int K2>
__device__ __forceinline__ void lane_origin(int lane, int& i0, int& off0) {
    constexpr int RING = 32 * GeoK<K2>::NW;
    const int s0 = (gbx::stream_end<K2>(8 * lane) - 16 + RING) % RING;
    i0 = s0 >> 5;
    off0 = s0 & 31;
}

template <int K2>
__device__ __forceinline__ void tile_load(const uint32_t* __restrict__ tile, int i0, uint32_t (&w)[GeoK<K2>::NL]) {
#pragma unroll
    for (int q = 0; q < GeoK<K2>::NL; ++q) {
        int idx = i0 + q;
        if (idx >= GeoK<K2>::NW) idx -= GeoK<K2>::NW;
        w[q] = __ldg(tile + idx);
    }
}

__device__ __forceinline__ float mul1(uint32_t s) {
    return fmaf((float)__dp4a(s * MUL1_MUL, 0x01010101u, 1024u), MUL1_SCALE, MUL1_BIAS);
}

// The 16-bit state of value j (0..7) of a lane, from its words and first-window bit offset (gbx::window16 over the
// 96-bit stream shifted so the lane's first window starts at bit 0). `w` points at the lane's words (at least
// GeoK<K2>::NL of them; the caller's arrays hold 3).
template <int K2>
__device__ __forceinline__ uint32_t lane_window(const uint32_t* w, int off0, int j) {
    constexpr int NL = GeoK<K2>::NL;
    const uint32_t c = NL > 2 ? w[2] : 0u;
    const uint32_t hi = __funnelshift_l(w[1], w[0], off0);
    const uint32_t mid = __funnelshift_l(c, w[1], off0);
    const uint32_t lo = c << off0;
    return gbx::window16(hi, mid, lo, gbx::stream_end<K2>(j) - gbx::stream_end<K2>(0));
}

template <int K2>
__device__ __forceinline__ void tile_decode(const uint32_t* w, int off0, uint32_t (&a)[4]) {
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = mul1(lane_window<K2>(w, off0, j));
    a[0] = h2u(v[0], v[1]);
    a[1] = h2u(v[4], v[5]);
    a[2] = h2u(v[2], v[3]);
    a[3] = h2u(v[6], v[7]);
}

// acc[j] += W^T tile (kt, nb * 8 + j) * B(kt) for kt in [0, nkt), tiles of row kt at base + kt * stride_k + j * NW.
// getb(kt, b0, b1) supplies the B fragment (the group's tokens) of k tile kt.
template <int K2, class GetB>
__device__ __forceinline__ void run_tiles(const uint32_t* __restrict__ base, size_t stride_k, int nkt, int lane,
                                          float (&acc)[8][4], GetB getb) {
    constexpr int NW = GeoK<K2>::NW, NL = GeoK<K2>::NL;
    int i0, off0;
    lane_origin<K2>(lane, i0, off0);
    uint32_t cur[8][NL], nxt[8][NL];
#pragma unroll
    for (int j = 0; j < 8; ++j) tile_load<K2>(base + j * NW, i0, cur[j]);
#pragma unroll 1
    for (int kt = 0; kt < nkt; ++kt) {
        if (kt + 1 < nkt) {
            const uint32_t* row = base + (size_t)(kt + 1) * stride_k;
#pragma unroll
            for (int j = 0; j < 8; ++j) tile_load<K2>(row + j * NW, i0, nxt[j]);
        }
        uint32_t b0, b1;
        getb(kt, b0, b1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t a[4];
            tile_decode<K2>(cur[j], off0, a);
            mma16816(acc[j], a, b0, b1);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < NL; ++q) cur[j][q] = nxt[j][q];
    }
}

template <class GetB>
__device__ __forceinline__ bool run_tiles_bits(int k2, const uint32_t* base, size_t stride_k, int nkt, int lane,
                                               float (&acc)[8][4], GetB getb) {
    switch (k2) {
        case 2: run_tiles<2>(base, stride_k, nkt, lane, acc, getb); return true;
        case 3: run_tiles<3>(base, stride_k, nkt, lane, acc, getb); return true;
        case 4: run_tiles<4>(base, stride_k, nkt, lane, acc, getb); return true;
        case 5: run_tiles<5>(base, stride_k, nkt, lane, acc, getb); return true;
        case 6: run_tiles<6>(base, stride_k, nkt, lane, acc, getb); return true;
        case 7: run_tiles<7>(base, stride_k, nkt, lane, acc, getb); return true;
        case 8: run_tiles<8>(base, stride_k, nkt, lane, acc, getb); return true;
        case 10: run_tiles<10>(base, stride_k, nkt, lane, acc, getb); return true;
        case 12: run_tiles<12>(base, stride_k, nkt, lane, acc, getb); return true;
        case 14: run_tiles<14>(base, stride_k, nkt, lane, acc, getb); return true;
        case 16: run_tiles<16>(base, stride_k, nkt, lane, acc, getb); return true;
        default: return false;
    }
}

// Partial tile outputs of the warp: acc[j][c] belongs to column nb * 128 + j * 16 + g (+8 for c = 2, 3) and token
// column 2t (c = 0, 2) or 2t + 1 (c = 1, 3) of the group. dst(col_token) -> float* row of 128 columns.
template <class Dst>
__device__ __forceinline__ void store_partials(const float (&acc)[8][4], int lane, int ntok, Dst dst) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int c = 2 * t + h;
        if (c < ntok) {
            float* row = dst(c);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                row[j * 16 + g] = acc[j][h];
                row[j * 16 + g + 8] = acc[j][2 + h];
            }
        }
    }
}
constexpr int ROUTE_THREADS = 512;
constexpr int ROUTE_EMAX = 64 * 11 + 64;   // smem capacity (64 tokens, top-k <= 11 with the shared expert)
// Route helpers. A token's logits come in as bf16 (every other family's MoE) or as fp32 (Ling-3.1's
// router_dtype = "fp32"): the launchers take the dtype code (0 bf16, 1 fp32) and the kernels load through
// these, so the routing arithmetic below is byte-for-byte the same in both cases apart from the load itself.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float route_logit(const void* logits, int f32, size_t i) {
    return f32 ? __ldg(static_cast<const float*>(logits) + i)
               : __bfloat162float(__ldg(static_cast<const __nv_bfloat16*>(logits) + i));
}
// The Ling-3.1 MoE router GEMV helpers: a VERBATIM copy of ling_kda_proj_fused.cu's ling_fma8_bf16 and
// ling_warp_reduce_sum5 (same FMA chain, same lane order, same warp reduction), so the fused router kernel below
// reproduces gb_ling_bf16_linear_f32's per-(row, column) bits exactly in the same translation unit.
__device__ __forceinline__ float ling_fma8_bf16(float acc, const uint4 xu, const uint4 wu) {
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

__device__ __forceinline__ float ling_warp_reduce_sum5(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

__global__ void __launch_bounds__(ROUTE_THREADS) ling_route_kernel(const void* __restrict__ logits, int logits_f32,
                                                              const __nv_bfloat16* __restrict__ shared_gate,
                                                              void* route_base, int emax, int T, int NE, int K) {
    __shared__ int s_exp[ROUTE_EMAX];
    __shared__ float s_wt[ROUTE_EMAX];
    __shared__ int s_rank[ROUTE_EMAX];
    __shared__ int s_gid[ROUTE_EMAX];
    __shared__ int s_ntok[ROUTE_EMAX];
    __shared__ int s_goff[ROUTE_EMAX];
    // per-warp top-k scratch (the winners every lane of the warp agree on): in shared memory so the dynamic `k`
    // indexing of the winners never touches the local/stack memory (STACK: 0)
    __shared__ float s_vk[ROUTE_THREADS / 32][16];
    __shared__ int s_ik[ROUTE_THREADS / 32][16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarp = blockDim.x >> 5;
    const int KT = K + (shared_gate ? 1 : 0);
    const int E = T * KT;
    Route r = route_view(route_base, emax);

    // 1. top-k a token, one warp a token. NE <= 512: lane holds experts lane + 32 i.
    for (int tok = warp; tok < T; tok += nwarp) {
        float l[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int e = lane + 32 * i;
            l[i] = e < NE ? route_logit(logits, logits_f32, (size_t)tok * NE + e) : -INFINITY;
        }
        float* vk = s_vk[warp];
        int* ik = s_ik[warp];
        float top_v = 0.f;
        for (int k = 0; k < K; ++k) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int i = 0; i < 16; ++i)
                if (l[i] > bv) { bv = l[i]; bi = lane + 32 * i; }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
#pragma unroll
            for (int i = 0; i < 16; ++i)
                if (lane + 32 * i == bi) l[i] = -INFINITY;
            if (k == 0) top_v = bv;
            if (lane == 0) { vk[k] = bv; ik[k] = bi; }
        }
        if (lane == 0) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) sum += expf(vk[k] - top_v);
            const float inv = 1.f / sum;
            for (int k = 0; k < K; ++k) {
                s_exp[tok * KT + k] = ik[k];
                s_wt[tok * KT + k] = expf(vk[k] - top_v) * inv;
            }
            if (shared_gate) {
                s_exp[tok * KT + K] = NE;
                s_wt[tok * KT + K] = 1.f / (1.f + expf(-__bfloat162float(shared_gate[tok])));
            }
        }
    }
    __syncthreads();

    // 2. rank of every entry among the entries of its expert (entries are token-major, so tokens ascend in a group)
    for (int i = tid; i < E; i += blockDim.x) {
        const int e = s_exp[i];
        int rank = 0;
        for (int j = 0; j < i; ++j) rank += s_exp[j] == e;
        s_rank[i] = rank;
    }
    __syncthreads();
    // 3. group leaders (rank % 8 == 0) get ids in entry order; their size; then offsets in group order
    for (int i = tid; i < E; i += blockDim.x) {
        if (s_rank[i] % GT == 0) {
            int gid = 0;
            for (int j = 0; j < i; ++j) gid += (s_rank[j] % GT) == 0;
            const int e = s_exp[i];
            int n = 0;
            for (int j = i; j < E; ++j) n += (s_exp[j] == e && s_rank[j] >= s_rank[i] && s_rank[j] < s_rank[i] + GT);
            s_gid[i] = gid;
            s_ntok[gid] = n;
        }
    }
    __syncthreads();
    for (int i = tid; i < E; i += blockDim.x) {
        if (s_rank[i] % GT == 0) {
            const int gid = s_gid[i];
            int off = 0;
            for (int q = 0; q < gid; ++q) off += s_ntok[q];
            s_goff[gid] = off;
        }
    }
    __syncthreads();
    // 4. every entry: its group (the leader with the same expert and chunk), slot = group offset + rank % 8
    for (int i = tid; i < E; i += blockDim.x) {
        const int e = s_exp[i], rk = s_rank[i];
        int lead = i;
        if (rk % GT) {
            const int want = rk - rk % GT;
            for (int j = 0; j < i; ++j)
                if (s_exp[j] == e && s_rank[j] == want) { lead = j; break; }
        }
        const int gid = s_gid[lead];
        const int slot = s_goff[gid] + rk % GT;
        r.s_tok[slot] = i / KT;
        r.s_w[slot] = s_wt[i];
        r.e_slot[i] = slot;
        if (lead == i) {
            r.g_exp[gid] = e;
            r.g_ntok[gid] = s_ntok[gid];
            r.g_off[gid] = s_goff[gid];
        }
    }
    if (tid == 0) {
        int U = 0;
        for (int i = 0; i < E; ++i) U += (s_rank[i] % GT) == 0;
        r.hdr[0] = U;
        r.hdr[1] = E;
        r.hdr[2] = T;
        r.hdr[3] = KT;
    }
}
// Kernel 1b: group-limited noaux_tc route (BailingMoeV3Gate): scores = sigmoid(logits) + expert_bias, the top-2
// scores of every group of n_group consecutive experts summed, the topk_group groups selected, the rest masked to -inf,
// top-K over the masked scores (tie: lowest index). The weights are the UNBIASED sigmoid scores of the selected experts,
// normalized by their sum (+ 1e-20) and scaled by `scale`; the shared expert is entry K with sigmoid(shared_gate).
// Group size must be 64 (NE % n_group == 0, NE / n_group == 64): each lane holds exactly two experts of every group.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void g2(float a, float b, float& m1, float& m2) {
    if (a >= b) {
        m1 = a;
        m2 = b;
    } else {
        m1 = b;
        m2 = a;
    }
}

__device__ __forceinline__ void g4(float a0, float a1, float a2, float a3, float& m1, float& m2) {
    float h1, l1, h2, l2;
    g2(a0, a1, h1, l1);
    g2(a2, a3, h2, l2);
    if (h1 >= h2) {
        m1 = h1;
        m2 = fmaxf(l1, h2);
    } else {
        m1 = h2;
        m2 = fmaxf(l2, h1);
    }
}

// The top-2 of the warp's union of each lane's (v1, v2) pair, XOR butterfly (every lane ends with the same pair).
__device__ __forceinline__ void g_butterfly(float& v1, float& v2) {
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        const float o1 = __shfl_xor_sync(0xffffffffu, v1, m);
        const float o2 = __shfl_xor_sync(0xffffffffu, v2, m);
        g4(v1, v2, o1, o2, v1, v2);
    }
}

// The group (0..7) of expert e under the 64-expert groups of the group-limited router.
__device__ __forceinline__ int group_of(int e) { return e >> 6; }
// Kernel 1b: group-limited noaux_tc route (BailingMoeV3Gate): scores = sigmoid(logits) + expert_bias, the top-2
// scores of every group of n_group consecutive experts summed, the topk_group groups selected, the rest masked to -inf,
// top-K over the masked scores (tie: lowest index). The weights are the UNBIASED sigmoid scores of the selected experts,
// normalized by their sum (+ 1e-20) and scaled by `scale`; the shared expert is entry K with sigmoid(shared_gate).
// Group size must be 64 (NE % n_group == 0, NE / n_group == 64): each lane holds exactly two experts of every group.
struct RouteGroupArgs {
    const void* logits;
    int logits_f32;
    const __nv_bfloat16* shared_gate;
    const float* expert_bias;
    float scale;
    void* route_base;
    int emax, T, NE, K, n_group, topk_group;
};

// The routing body shared by the standalone ling_route_group_kernel (ROUTE_THREADS = 512 threads) and the fused
// router's last CTA (256 threads). The warp-per-token stage runs `warp` to `T` step `nwarp`, and the later stages
// stride over blockDim.x, so for T <= 8 both call shapes assign warp w to token w and touch every entry exactly once:
// the route buffer is bit-identical from either.
__device__ void ling_route_group_body(const RouteGroupArgs& a,
                                      int* s_exp, float* s_wt, int* s_rank, int* s_gid,
                                      int* s_ntok, int* s_goff,
                                      float (*s_suk)[16], int (*s_ik)[16], float (*s_gs)[8]) {
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarp = blockDim.x >> 5;
    const int KT = a.K + (a.shared_gate ? 1 : 0);
    const int E = a.T * KT;
    constexpr int GS = 64;  // experts a group (must equal a.NE / a.n_group)
    Route r = route_view(a.route_base, a.emax);

    // top-2 of every group across the warp: each lane holds experts lane + 32i, so of group g it holds exactly
    // experts 64g + lane and 64g + lane + 32 (the lane's sc[2g] and sc[2g+1])
    for (int tok = warp; tok < a.T; tok += nwarp) {
        float* suk = s_suk[warp];
        int* ik = s_ik[warp];
        // each lane's 16 experts: sigmoid score su[i] and biased score sb[i], loaded ONCE per token as read-only
        // registers (the k loop indexes them by compile-time constants only and never writes them, so ptxas
        // keeps them in registers: STACK 0, LOCAL 0, 1x expf instead of the legacy 9x reload)
        float sb[16], su[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int e = lane + 32 * i;
            if (e < a.NE) {
                const float lg = route_logit(a.logits, a.logits_f32, (size_t)tok * a.NE + e);
                su[i] = 1.0f / (1.0f + expf(-lg));
                sb[i] = su[i] + a.expert_bias[e];
            } else {
                su[i] = -INFINITY;
                sb[i] = -INFINITY;
            }
        }
        // top-2 of every group across the warp: each lane holds experts lane + 32i, so of group g it holds exactly
        // experts 64g + lane and 64g + lane + 32 (the lane's sb[2g] and sb[2g+1])
        float gs[8];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            float a, b;
            g2(sb[2 * g], sb[2 * g + 1], a, b);
            g_butterfly(a, b);
            gs[g] = a + b;
        }
        // the topk_group groups by (score DESC, index ASC)
        uint32_t sel = 0u;  // bit g: group g selected (register state)
        for (int q = 0; q < a.topk_group; ++q) {
            float bv = -INFINITY;
            int bg = 0x7fffffff;
#pragma unroll
            for (int g = 0; g < 8; ++g)
                if (!(sel & (1u << g)) && (gs[g] > bv || (gs[g] == bv && g < bg))) {
                    bv = gs[g];
                    bg = g;
                }
            if (bg >= 8) bg = q;  // NaN-safe: never select a bogus group
            sel |= 1u << bg;
        }
        // top-k across the full expert set, only the selected groups' experts
        uint32_t excl = 0u;  // bit i: expert lane + 32i already picked (per-lane register state)
        for (int k = 0; k < a.K; ++k) {
            float bv = -INFINITY;
            float bsu = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (excl & (1u << i)) continue;
                const int e = lane + 32 * i;
                if (e >= a.NE || !(sel & (1u << group_of(e)))) continue;
                if (sb[i] > bv) { bv = sb[i]; bsu = su[i]; bi = e; }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const float osu = __shfl_xor_sync(0xffffffffu, bsu, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bsu = osu;
                    bi = oi;
                }
            }
            // exclude the picked expert on the lane that owns it (bi is warp-uniform after the reduction)
            if ((bi & 31) == lane && bi < a.NE) excl |= 1u << (bi >> 5);
            if (bi >= a.NE) bi = k;  // NaN-safe: keep a valid expert id
            if (lane == 0) { suk[k] = bsu; ik[k] = bi; }
        }
        if (lane == 0) {
            float sum = 0.f;
            for (int k = 0; k < a.K; ++k) sum += suk[k];  // the selected entries' unbiased sigmoid scores
            const float inv = 1.f / (sum + 1e-20f);
            for (int k = 0; k < a.K; ++k) {
                s_exp[tok * KT + k] = ik[k];
                s_wt[tok * KT + k] = suk[k] * inv * a.scale;
            }
            if (a.shared_gate) {
                s_exp[tok * KT + a.K] = a.NE;
                s_wt[tok * KT + a.K] = 1.f / (1.f + expf(-__bfloat162float(a.shared_gate[tok])));
            }
        }
    }
    __syncthreads();

    // 2. rank of every entry among the entries of its expert (entries are token-major, so tokens ascend in a group)
    for (int i = tid; i < E; i += blockDim.x) {
        const int e = s_exp[i];
        int rank = 0;
        for (int j = 0; j < i; ++j) rank += s_exp[j] == e;
        s_rank[i] = rank;
    }
    __syncthreads();
    // 3. group leaders (rank % 8 == 0) get ids in entry order; their size; then offsets in group order
    for (int i = tid; i < E; i += blockDim.x) {
        if (s_rank[i] % GT == 0) {
            int gid = 0;
            for (int j = 0; j < i; ++j) gid += (s_rank[j] % GT) == 0;
            const int e = s_exp[i];
            int n = 0;
            for (int j = i; j < E; ++j) n += (s_exp[j] == e && s_rank[j] >= s_rank[i] && s_rank[j] < s_rank[i] + GT);
            s_gid[i] = gid;
            s_ntok[gid] = n;
        }
    }
    __syncthreads();
    for (int i = tid; i < E; i += blockDim.x) {
        if (s_rank[i] % GT == 0) {
            const int gid = s_gid[i];
            int off = 0;
            for (int q = 0; q < gid; ++q) off += s_ntok[q];
            s_goff[gid] = off;
        }
    }
    __syncthreads();
    // 4. every entry: its group (the leader with the same expert and chunk), slot = group offset + rank % 8
    for (int i = tid; i < E; i += blockDim.x) {
        const int e = s_exp[i], rk = s_rank[i];
        int lead = i;
        if (rk % GT) {
            const int want = rk - rk % GT;
            for (int j = 0; j < i; ++j)
                if (s_exp[j] == e && s_rank[j] == want) {
                    lead = j;
                    break;
                }
        }
        const int gid = s_gid[lead];
        const int slot = s_goff[gid] + rk % GT;
        r.s_tok[slot] = i / KT;
        r.s_w[slot] = s_wt[i];
        r.e_slot[i] = slot;
        if (lead == i) {
            r.g_exp[gid] = e;
            r.g_ntok[gid] = s_ntok[gid];
            r.g_off[gid] = s_goff[gid];
        }
    }
    if (tid == 0) {
        int U = 0;
        for (int i = 0; i < E; ++i) U += (s_rank[i] % GT) == 0;
        r.hdr[0] = U;
        r.hdr[1] = E;
        r.hdr[2] = a.T;
        r.hdr[3] = KT;
    }

}

__global__ void __launch_bounds__(ROUTE_THREADS) ling_route_group_kernel(const void* __restrict__ logits, int logits_f32,
                                                                    const __nv_bfloat16* __restrict__ shared_gate,
                                                                    const float* __restrict__ expert_bias, float scale,
                                                                    void* route_base, int emax, int T, int NE, int K,
                                                                    int n_group, int topk_group) {
    __shared__ int s_exp[ROUTE_EMAX];
    __shared__ float s_wt[ROUTE_EMAX];
    __shared__ int s_rank[ROUTE_EMAX];
    __shared__ int s_gid[ROUTE_EMAX];
    __shared__ int s_ntok[ROUTE_EMAX];
    __shared__ int s_goff[ROUTE_EMAX];
    __shared__ float s_suk[ROUTE_THREADS / 32][16];
    __shared__ int s_ik[ROUTE_THREADS / 32][16];
    __shared__ float s_gs[ROUTE_THREADS / 32][8];
    ling_route_group_body(RouteGroupArgs{logits, logits_f32, shared_gate, expert_bias, scale, route_base, emax,
                                         T, NE, K, n_group, topk_group},
                          s_exp, s_wt, s_rank, s_gid, s_ntok, s_goff, s_suk, s_ik, s_gs);
}

// The Ling-3.1 MoE router GEMV fused with the group-limited route kernel: logits [T][512] fp32 = x [T][4096]
// bf16 @ router [512][4096]^T bf16, one warp computing TWO output columns j0/j1 per CTA (grid 32 x 256 covers the
// 512 columns), then the LAST CTA to finish runs ling_route_group_body on the grid (a self-resetting atomic counter
// at `route_cnt`). Per (row, column) the GEMV is bit-identical to ling_bf16_linear_f32_kernel<ROWS>: same
// ascending c8 order, same fma8 chain, same warp_reduce_sum5, same fp32 store (the xu load is shared by the two
// columns). w.cntc[H / 128] holds the counter: one int past the down kernel's H/128 slots, inside the 256-aligned
// cntc part of the workspace (it must start 0; the last CTA rewrites it to 0). Launched only when group routing,
// T <= 8, H == 4096, NE == 512: any other shape keeps the two-launch path.
__global__ void __launch_bounds__(256) ling_router_and_route_group_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    float* __restrict__ logits, const __nv_bfloat16* __restrict__ shared_gate,
    const float* __restrict__ expert_bias, float scale, void* route_base, int emax,
    int T, int NE, int H, int K, int n_group, int topk_group, int* __restrict__ route_cnt) {
    __shared__ int s_exp[ROUTE_EMAX];
    __shared__ float s_wt[ROUTE_EMAX];
    __shared__ int s_rank[ROUTE_EMAX];
    __shared__ int s_gid[ROUTE_EMAX];
    __shared__ int s_ntok[ROUTE_EMAX];
    __shared__ int s_goff[ROUTE_EMAX];
    __shared__ float s_suk[ROUTE_THREADS / 32][16];
    __shared__ int s_ik[ROUTE_THREADS / 32][16];
    __shared__ float s_gs[ROUTE_THREADS / 32][8];
    __shared__ int s_last;

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int j0 = blockIdx.x * 16 + 2 * warp;
    const int j1 = j0 + 1;
    const int k8 = H >> 3;
    for (int r = 0; r < T; ++r) {
        const __nv_bfloat16* xr = x + (size_t)r * H;
        float a0 = 0.f, a1 = 0.f;
        for (int c8 = lane; c8 < k8; c8 += 32) {
            const uint4 xu = *reinterpret_cast<const uint4*>(xr + c8 * 8);
            const uint4 wu0 = *reinterpret_cast<const uint4*>(w + (size_t)j0 * H + c8 * 8);
            const uint4 wu1 = *reinterpret_cast<const uint4*>(w + (size_t)j1 * H + c8 * 8);
            a0 = ling_fma8_bf16(a0, xu, wu0);
            a1 = ling_fma8_bf16(a1, xu, wu1);
        }
        const float v0 = ling_warp_reduce_sum5(a0);
        const float v1 = ling_warp_reduce_sum5(a1);
        if (lane == 0) {
            logits[(size_t)r * NE + j0] = v0;
            logits[(size_t)r * NE + j1] = v1;
        }
    }
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) s_last = (atomicAdd(route_cnt, 1) == (int)gridDim.x - 1);
    __syncthreads();
    if (!s_last) return;
    if (threadIdx.x == 0) *route_cnt = 0;

    RouteGroupArgs args{logits, 1, shared_gate, expert_bias, scale, route_base, emax,
                        T, NE, K, n_group, topk_group};
    ling_route_group_body(args, s_exp, s_wt, s_rank, s_gid, s_ntok, s_goff, s_suk, s_ik, s_gs);
}

struct GateUpArgs {
    const __nv_bfloat16* x;  // [T][H]
    const void* route;
    int emax;
    const uint64_t *gT, *gSu, *gSv, *uT, *uSu, *uSv, *dSu;
    const int* ktab;  // [experts + 1][3]
    float* part;
    int* cnt;
    half* xd;
    int H, I, KC;     // KC: k chunks (H / 128 must be a multiple)
    int ne;           // routed expert count (the shared expert is entry ne)
    float routed_limit, shared_limit;  // SwiGLU clamps (<= 0: no clamp)
};
__global__ void __launch_bounds__(WARPS * 32, 4) ling_gate_up_kernel(GateUpArgs a) {
    extern __shared__ __align__(16) half smem_x[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const Route r = route_view(const_cast<void*>(a.route), a.emax);
    const int NB = a.I >> 7;
    const int jobs = 2 * NB * a.KC;
    const int job = blockIdx.x * WARPS + warp;
    const int grp = job / jobs;
    if (grp >= r.hdr[0]) return;
    int q = job - grp * jobs;
    const int kc = q % a.KC;
    q /= a.KC;
    const int nb = q % NB;
    const int mat = q / NB;  // 0 gate, 1 up

    const int e = r.g_exp[grp], ntok = r.g_ntok[grp], off = r.g_off[grp];
    const float limit = (e == a.ne) ? a.shared_limit : a.routed_limit;
    const int bits = a.ktab[e * 3 + mat];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? a.uT[e] : a.gT[e]);
    const half* su = reinterpret_cast<const half*>(mat ? a.uSu[e] : a.gSu[e]);

    const int kblocks = (a.H >> 7) / a.KC;  // 128-wide k blocks this warp covers
    const int kb0 = kc * kblocks;
    const int xstride = kblocks * 128 + 8;  // +8 halves: conflict-free B fragment reads
    half* xs = smem_x + warp * GT * xstride;

    // stage the group's tokens rotated by this matrix's suh: xs[c][k] = (H (x * suh) / sqrt(128))[kb0 * 128 + k]
    for (int c = 0; c < GT; ++c) {
        for (int b = 0; b < kblocks; ++b) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (c < ntok) {
                const int tok = r.s_tok[off + c];
                const int k = (kb0 + b) * 128 + 4 * lane;
                float s[4];
                ld_bf16x4(a.x + (size_t)tok * a.H + k, v);
                ld_f16x4(su + k, s);
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] *= s[i];
                had128(v, lane);
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] *= RS128;
            }
            st_f16x4(xs + c * xstride + b * 128 + 4 * lane, v);
        }
    }
    __syncwarp();

    const int g = lane >> 2, t = lane & 3;
    const size_t tw = 4 * (size_t)bits;  // K2 words a tile (4 * K2)
    const size_t stride_k = (size_t)(a.I >> 4) * tw;
    const uint32_t* base = T + (size_t)(kb0 * 8) * stride_k + (size_t)nb * 8 * tw;
    const half* xrow = xs + g * xstride + 2 * t;
    auto getb = [&](int kt, uint32_t& b0, uint32_t& b1) {
        b0 = *reinterpret_cast<const uint32_t*>(xrow + kt * 16);
        b1 = *reinterpret_cast<const uint32_t*>(xrow + kt * 16 + 8);
    };
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[j][c] = 0.f;
    run_tiles_bits(bits, base, stride_k, kblocks * 8, lane, acc, getb);

    const size_t pstride_mat = (size_t)a.KC * a.I;
    store_partials(acc, lane, ntok, [&](int c) {
        return a.part + ((size_t)(off + c) * 2 + mat) * pstride_mat + (size_t)kc * a.I + nb * 128;
    });
    __threadfence();
    int last = 0;
    if (lane == 0) last = atomicAdd(a.cnt + grp * NB + nb, 1) == 2 * a.KC - 1;
    last = __shfl_sync(0xffffffffu, last, 0);
    if (!last) return;
    __threadfence();

    // finish column block nb of every token of the group
    const half* gsv = reinterpret_cast<const half*>(a.gSv[e]);
    const half* usv = reinterpret_cast<const half*>(a.uSv[e]);
    const half* dsu = reinterpret_cast<const half*>(a.dSu[e]);
    const int col = nb * 128 + 4 * lane;
    float sg[4], sv[4], sd[4];
    ld_f16x4(gsv + col, sg);
    ld_f16x4(usv + col, sv);
    ld_f16x4(dsu + col, sd);
    for (int c = 0; c < ntok; ++c) {
        const int slot = off + c;
        const float* pg = a.part + ((size_t)slot * 2 + 0) * pstride_mat + col;
        const float* pu = a.part + ((size_t)slot * 2 + 1) * pstride_mat + col;
        float vg[4] = {0.f, 0.f, 0.f, 0.f}, vu[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k = 0; k < a.KC; ++k) {
            const float4 x0 = ldcg4(pg + (size_t)k * a.I);
            const float4 x1 = ldcg4(pu + (size_t)k * a.I);
            vg[0] += x0.x; vg[1] += x0.y; vg[2] += x0.z; vg[3] += x0.w;
            vu[0] += x1.x; vu[1] += x1.y; vu[2] += x1.z; vu[3] += x1.w;
        }
        had128(vg, lane);
        had128(vu, lane);
        float act[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float gv = vg[i] * RS128 * sg[i];
            float uv = vu[i] * RS128 * sv[i];
            if (limit > 0.0f) {
                gv = fminf(gv, limit);
                uv = fmaxf(-limit, fminf(uv, limit));
            }
            act[i] = gv / (1.f + __expf(-gv)) * uv * sd[i];
        }
        had128(act, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) act[i] *= RS128;
        st_f16x4(a.xd + (size_t)slot * a.I + col, act);
    }
    if (lane == 0) a.cnt[grp * NB + nb] = 0;
}
struct DownArgs {
    const half* xd;  // [emax][I] rotated activations
    const void* route;
    int emax;
    const uint64_t *dT, *dSv;
    const int* ktab;
    float* partd;
    float* y;
    int* cntd;
    int* cntc;
    __nv_bfloat16* out;  // [T][H]
    int H, I;
    const __nv_bfloat16* hidden;  // [T][H] residual stream (post previous layer)
    const float* norm_w;          // next layer input norm (or final norm)
    __nv_bfloat16* norm_out;      // next_dst (norm / final_normed)
    float eps;
    const half* suh_p0; half* xh_p0;
    const half* suh_p1; half* xh_p1;
    const half* suh_p2; half* xh_p2;
    int* cnt_epi;                 // cntc + NBD + 1 (self-resetting all-column-blocks counter)
};

__device__ __forceinline__ void ling_down_fused_epilogue_row_cta(const DownArgs& a, int row, float* sm) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t base = (size_t)row * a.H;
    const int nvec = a.H / 8;
    const int NBD = a.H >> 7;

    #pragma unroll 1
    for (int hb = 0; hb < 2; ++hb) {
        const int vt = tid + 128 * hb;
        float ss = 0.f;
        #pragma unroll 1
        for (int v = vt; v < nvec; v += 256) {
            const uint2* xp = reinterpret_cast<const uint2*>(a.hidden + base + (size_t)v * 8);
            const uint2* rp = reinterpret_cast<const uint2*>(a.out + base + (size_t)v * 8);
            uint2* hp = (uint2*)const_cast<__nv_bfloat16*>(a.hidden + base + (size_t)v * 8);
            #pragma unroll 1
            for (int u = 0; u < 2; ++u) {
                const uint2 xu = xp[u], ru = rp[u];
                const uint16_t b0 = lpf_f2bf(__fadd_rn(lpf_bf2f(xu.x & 0xFFFFu), lpf_bf2f(ru.x & 0xFFFFu)));
                const uint16_t b1 = lpf_f2bf(__fadd_rn(lpf_bf2f(xu.x >> 16), lpf_bf2f(ru.x >> 16)));
                const uint16_t b2 = lpf_f2bf(__fadd_rn(lpf_bf2f(xu.y & 0xFFFFu), lpf_bf2f(ru.y & 0xFFFFu)));
                const uint16_t b3 = lpf_f2bf(__fadd_rn(lpf_bf2f(xu.y >> 16), lpf_bf2f(ru.y >> 16)));
                const float f0 = lpf_bf2f(b0), f1 = lpf_bf2f(b1), f2 = lpf_bf2f(b2), f3 = lpf_bf2f(b3);
                ss = __fadd_rn(ss, __fmul_rn(f0, f0));
                ss = __fadd_rn(ss, __fmul_rn(f1, f1));
                ss = __fadd_rn(ss, __fmul_rn(f2, f2));
                ss = __fadd_rn(ss, __fmul_rn(f3, f3));
                hp[u] = make_uint2((uint32_t)b0 | ((uint32_t)b1 << 16), (uint32_t)b2 | ((uint32_t)b3 << 16));
            }
        }
        ss = lpf_warp_sum(ss);
        if (lane == 0) sm[warp + 4 * hb] = ss;
    }
    __syncthreads();
    float s = sm[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) s = s + sm[w];
    const float inv = 1.0f / sqrtf(s / (float)a.H + a.eps);
    __syncthreads();

    #pragma unroll 1
    for (int v = tid; v < nvec; v += 128) {
        const uint2* xp = reinterpret_cast<const uint2*>(a.hidden + base + (size_t)v * 8);
        const float4* wp = reinterpret_cast<const float4*>(a.norm_w + (size_t)v * 8);
        uint2* yp = reinterpret_cast<uint2*>(a.norm_out + base + (size_t)v * 8);
        #pragma unroll 1
        for (int u = 0; u < 2; ++u) {
            const uint2 xu = xp[u];
            const float4 wv = wp[u];
            const uint16_t o0 = lpf_f2bf(__fmul_rn(__fmul_rn(lpf_bf2f(xu.x & 0xFFFFu), inv), wv.x));
            const uint16_t o1 = lpf_f2bf(__fmul_rn(__fmul_rn(lpf_bf2f(xu.x >> 16), inv), wv.y));
            const uint16_t o2 = lpf_f2bf(__fmul_rn(__fmul_rn(lpf_bf2f(xu.y & 0xFFFFu), inv), wv.z));
            const uint16_t o3 = lpf_f2bf(__fmul_rn(__fmul_rn(lpf_bf2f(xu.y >> 16), inv), wv.w));
            yp[u] = make_uint2((uint32_t)o0 | ((uint32_t)o1 << 16), (uint32_t)o2 | ((uint32_t)o3 << 16));
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int blk = warp; blk < NBD; blk += WARPS) {
        const int col = blk * 128 + 4 * lane;
        const uint2 nu = *(const uint2*)(a.norm_out + base + col);
        const float n0 = lpf_bf2f(nu.x & 0xFFFFu), n1 = lpf_bf2f(nu.x >> 16);
        const float n2 = lpf_bf2f(nu.y & 0xFFFFu), n3 = lpf_bf2f(nu.y >> 16);
        #pragma unroll 1
        for (int p = 0; p < 3; ++p) {
            const half* suh = (p == 0) ? a.suh_p0 : (p == 1) ? a.suh_p1 : a.suh_p2;
            half* xh = (p == 0) ? a.xh_p0 : (p == 1) ? a.xh_p1 : a.xh_p2;
            if (!suh || !xh) continue;
            float v[4], su[4];
            ld_f16x4(suh + col, su);
            v[0] = n0 * su[0]; v[1] = n1 * su[1]; v[2] = n2 * su[2]; v[3] = n3 * su[3];
            gbx::fwht128(v, lane);
            #pragma unroll
            for (int j = 0; j < 4; ++j) v[j] *= gbx::HAD_SCALE;
            st_f16x4(xh + base + col, v);
        }
    }
}

__global__ void __launch_bounds__(WARPS * 32, 4) ling_down_kernel(DownArgs a) {
    do {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const Route r = route_view(const_cast<void*>(a.route), a.emax);
        const int NBD = a.H >> 7, KCD = a.I >> 7;
        const int jobs = NBD * KCD;
        const int job = blockIdx.x * WARPS + warp;
        const int grp = job / jobs;
        const int U = r.hdr[0];
        if (grp >= U) break;
        const int q = job - grp * jobs;
        const int kc = q % KCD, nb = q / KCD;

        const int e = r.g_exp[grp], ntok = r.g_ntok[grp], off = r.g_off[grp];
        const int bits = a.ktab[e * 3 + 2];
        const uint32_t* T = reinterpret_cast<const uint32_t*>(a.dT[e]);
        const int g = lane >> 2, t = lane & 3;

        uint32_t bx[8][2];
        {
            const int c = g < ntok ? g : 0;
            const half* src = a.xd + (size_t)(off + c) * a.I + kc * 128 + 2 * t;
#pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                bx[kt][0] = g < ntok ? __ldcg(reinterpret_cast<const unsigned int*>(src + kt * 16)) : 0u;
                bx[kt][1] = g < ntok ? __ldcg(reinterpret_cast<const unsigned int*>(src + kt * 16 + 8)) : 0u;
            }
        }
        const size_t tw = 4 * (size_t)bits;
        const size_t stride_k = (size_t)(a.H >> 4) * tw;
        const uint32_t* base = T + (size_t)(kc * 8) * stride_k + (size_t)nb * 8 * tw;
        auto getb = [&](int kt, uint32_t& b0, uint32_t& b1) {
            uint32_t x0 = bx[0][0], x1 = bx[0][1];
#pragma unroll
            for (int i = 1; i < 8; ++i)
                if (kt == i) { x0 = bx[i][0]; x1 = bx[i][1]; }
            b0 = x0;
            b1 = x1;
        };
        float acc[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[j][c] = 0.f;
        run_tiles_bits(bits, base, stride_k, 8, lane, acc, getb);

        store_partials(acc, lane, ntok, [&](int c) {
            return a.partd + ((size_t)(off + c) * KCD + kc) * a.H + nb * 128;
        });
        __threadfence();
        int last = 0;
        if (lane == 0) last = atomicAdd(a.cntd + grp * NBD + nb, 1) == KCD - 1;
        last = __shfl_sync(0xffffffffu, last, 0);
        if (!last) break;
        __threadfence();

        const half* dsv = reinterpret_cast<const half*>(a.dSv[e]);
        const int col = nb * 128 + 4 * lane;
        float sv[4];
        ld_f16x4(dsv + col, sv);
        for (int c = 0; c < ntok; ++c) {
            const int slot = off + c;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            for (int k = 0; k < KCD; ++k) {
                const float4 x = ldcg4(a.partd + ((size_t)slot * KCD + k) * a.H + col);
                v[0] += x.x; v[1] += x.y; v[2] += x.z; v[3] += x.w;
            }
            had128(v, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= RS128 * sv[i];
            *reinterpret_cast<float4*>(a.y + (size_t)slot * a.H + col) = make_float4(v[0], v[1], v[2], v[3]);
        }
        if (lane == 0) a.cntd[grp * NBD + nb] = 0;
        __threadfence();
        int lastc = 0;
        if (lane == 0) lastc = atomicAdd(a.cntc + nb, 1) == U - 1;
        lastc = __shfl_sync(0xffffffffu, lastc, 0);
        if (!lastc) break;
        __threadfence();

        const int Tn = r.hdr[2], KT = r.hdr[3];
        for (int tok = 0; tok < Tn; ++tok) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            for (int k = 0; k < KT; ++k) {
                const int slot = r.e_slot[tok * KT + k];
                const float w = r.s_w[slot];
                const float4 x = ldcg4(a.y + (size_t)slot * a.H + col);
                v[0] += w * x.x; v[1] += w * x.y; v[2] += w * x.z; v[3] += w * x.w;
            }
            st_bf16x4(a.out + (size_t)tok * a.H + col, v);
        }
        if (lane == 0) a.cntc[nb] = 0;
    } while (0);

    if (a.norm_w && a.norm_out && a.hidden && a.cnt_epi) {
        __syncthreads();
        __threadfence();
        __shared__ int s_last_cta;
        __shared__ float sm[8];
        if (threadIdx.x == 0) {
            const int prev = atomicAdd(a.cnt_epi, 1);
            if (prev == (int)gridDim.x - 1) {
                *a.cnt_epi = 0;
                s_last_cta = 1;
            } else {
                s_last_cta = 0;
            }
        }
        __syncthreads();
        if (!s_last_cta) return;
        __threadfence();
        const int Tn = route_view(const_cast<void*>(a.route), a.emax).hdr[2];
        #pragma unroll 1
        for (int row = 0; row < Tn; ++row) {
            ling_down_fused_epilogue_row_cta(a, row, sm);
            __syncthreads();
        }
    }
}
}  // namespace lingfmoe

using namespace lingfmoe;
// Workspace layout (bytes, each part 256-aligned): route | part | xd | partd | y | cnt | cntd | cntc
struct FmoeWs {
    char *route, *part, *xd, *partd, *y, *cnt, *cntd, *cntc;
    size_t total;
};

static size_t al256(size_t v) { return (v + 255) & ~(size_t)255; }

static FmoeWs fmoe_layout(char* base, int emax, int H, int I, int KC) {
    FmoeWs w;
    size_t o = 0;
    auto take = [&](size_t bytes) { char* p = base ? base + o : nullptr; o += al256(bytes); return p; };
    w.route = take(route_bytes(emax));
    w.part = take((size_t)emax * 2 * KC * I * 4);
    w.xd = take((size_t)emax * I * 2);
    w.partd = take((size_t)emax * (I / 128) * H * 4);
    w.y = take((size_t)emax * H * 4);
    w.cnt = take((size_t)emax * (I / 128) * 4);
    w.cntd = take((size_t)emax * (H / 128) * 4);
    w.cntc = take((size_t)(H / 128) * 4);
    w.total = o;
    return w;
}

// Bytes of workspace for up to max_tokens rows (top_k routed + the shared expert). Must be zeroed once at allocation.
GB_EXPORT size_t gb_ling_fmoe_workspace_bytes(int max_tokens, int top_k, int H, int I, int KC) {
    const int emax = max_tokens * (top_k + 1);
    return fmoe_layout(nullptr, emax, H, I, KC).total;
}

// out [T][H] bf16 = MoE(x [T][H] bf16) with routing from logits [T][NE] (bf16 when logits_f32 == 0, fp32
// otherwise — Ling-3.1's router_dtype = fp32) and the shared expert gated by
// sigmoid(shared_gate [T] bf16). Pointer tables hold NE + 1 entries (the shared expert last); ktab [NE + 1][3] holds
// the EXL3 K2 (twice the bits, 2..16) of gate, up, down. T <= max_tokens of the workspace, T <= 64, NE <= 512,
// top_k <= 16, H % 128 == 0, I % 128 == 0, (H/128) % KC == 0.
// Routing: expert_bias == nullptr (or n_group < 2) selects the legacy top-k softmax (no bias, weight = softmax of the
// logits, no scale); otherwise the group-limited noaux_tc router runs (register-cached ling_route_group_kernel:
// sigmoid scores + expert_bias, top-2 per group, topk_group groups of NE / n_group = 64 experts, weights = normalized
// unbiased sigmoid scores * scale). SwiGLU clamps: routed_limit applies to the routed experts, shared_limit to the
// shared expert (entry NE); <= 0: none.
// Bit-identical to fmoe::forward on every input: same routing outputs, same ascending-kc float32 gate/up sum order
// (the legacy global-partial ling_gate_up_kernel), same down_kernel. The route kernel writes the route buffer; the
// gate/up and down kernels self-reset their atomic counters, so the workspace only needs its initial zeroing at
// allocation (see gb_ling_fmoe_workspace_bytes).
GB_EXPORT int gb_ling_fmoe_forward(const void* x, const void* logits, int logits_f32, const void* shared_gate, const uint64_t* gT,
                                   const uint64_t* gSu, const uint64_t* gSv, const uint64_t* uT, const uint64_t* uSu,
                                   const uint64_t* uSv, const uint64_t* dT, const uint64_t* dSu, const uint64_t* dSv,
                                   const int* ktab, void* out, int T, int NE, int top_k, int H, int I, int KC,
                                   int max_tokens, void* ws, void* stream, const float* expert_bias, int n_group,
                                   int topk_group, float routed_limit, float shared_limit, float routed_scaling) {
    if (T < 1 || T > 64 || T > max_tokens || NE > 512 || top_k > 16 || H % 128 || I % 128 || KC < 1 ||
        (H / 128) % KC)
        return (int)cudaErrorInvalidValue;
    const bool group = expert_bias != nullptr && n_group >= 2 && topk_group >= 1;
    if (group ? (NE % n_group != 0 || NE / n_group != 64 || topk_group > n_group)
              : (expert_bias != nullptr || n_group != 0 || topk_group != 0))
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const int emax = max_tokens * (top_k + 1);
    FmoeWs w = fmoe_layout(static_cast<char*>(ws), emax, H, I, KC);
    const int E = T * (top_k + (shared_gate ? 1 : 0));

    if (group)
        ling_route_group_kernel<<<1, ROUTE_THREADS, 0, st>>>(logits, logits_f32, (const __nv_bfloat16*)shared_gate,
                                                             expert_bias, routed_scaling, w.route, emax, T, NE, top_k, n_group,
                                                             topk_group);
    else
        ling_route_kernel<<<1, ROUTE_THREADS, 0, st>>>(logits, logits_f32, (const __nv_bfloat16*)shared_gate,
                                                       w.route, emax, T, NE, top_k);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;

    GateUpArgs ga{(const __nv_bfloat16*)x, w.route, emax, gT, gSu, gSv, uT, uSu, uSv, dSu, ktab,
                  (float*)w.part, (int*)w.cnt, (half*)w.xd, H, I, KC, NE, routed_limit, shared_limit};
    const int jobs_a = E * 2 * (I / 128) * KC;
    const size_t smem_a = (size_t)WARPS * GT * ((H / 128 / KC) * 128 + 8) * sizeof(half);
    // the staged tokens live in the default 48 KiB of dynamic shared memory (the legacy kernel never raises its limit): at most 5 blocks of
    // 128 a chunk, so H = 2048 (16 blocks) takes KC 4, 8 and 16 and Flash-Next's H = 2560 (20 blocks) KC 4, 5, 10 and 20
    if (smem_a > 48 * 1024) return (int)cudaErrorInvalidValue;

    // The 4-CTA/SM global-partial gate/up kernel (always; the overlapped-smem ling_gate_up_kernel_smem variants are
    // gone). `__launch_bounds__(WARPS * 32, 4)` keeps 4 CTAs/SM (192 concurrent CTAs across GB10's 48 SMs) with the
    // gate and up column blocks of independent groups in flight: 104.4 ms across the decode battery vs 128.2 ms for
    // the single-CTA/SM 70.6-KiB smem variant, which serializes gate then up inside each CTA. Its atomic counters
    // (w.cnt) are self-resetting: the warp that finishes the last (group, block) job rewrites its slot to 0, so no
    // zero fills are needed between calls — the workspace contract stays "zeroed once at allocation" (bailing_moe_v3
    // does via DevBuf::zeroed).
    ling_gate_up_kernel<<<(jobs_a + WARPS - 1) / WARPS, WARPS * 32, smem_a, st>>>(ga);
    rc = GB_CHECK_LAUNCH();
    if (rc) return rc;

    // Same story for the down kernel: its count slots (w.cntd per (group, block), w.cntc per column block) self-reset
    // when the last group finishes a column block, so the two zero_i32 launches are removed too.
    DownArgs da{(const half*)w.xd, w.route, emax, dT, dSv, ktab, (float*)w.partd, (float*)w.y, (int*)w.cntd,
                (int*)w.cntc, (__nv_bfloat16*)out, H, I};
    const int jobs_b = E * (H / 128) * (I / 128);
    ling_down_kernel<<<(jobs_b + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(da);
    return GB_CHECK_LAUNCH();
}

// The fused-router variant of gb_ling_fmoe_forward for the Ling-3.1 decode shape: computes the router logits
// [T][512] fp32 = x [T][4096] bf16 @ router [512][4096]^T bf16 AND routes in ONE launch (see
// ling_router_and_route_group_kernel), then gate/up and down — 3 launches instead of 4. Requires group routing,
// T <= 8, H == 4096, NE == 512; callers fall back to gb_ling_fmoe_forward (after computing logits) otherwise.
// `logits` is an OUTPUT buffer [T][NE] fp32 written with the same bits as gb_ling_bf16_linear_f32. Workspace and
// counter contracts as gb_ling_fmoe_forward (ws zeroed once at allocation; the fused router's self-resetting counter
// lives at w.cntc + H / 128, one int past the down kernel's H/128 slots, inside the 256-aligned cntc part).
GB_EXPORT int gb_ling_fmoe_forward_with_router(const void* x, const void* router_w, float* logits,
                                               const void* shared_gate, const uint64_t* gT,
                                               const uint64_t* gSu, const uint64_t* gSv, const uint64_t* uT,
                                               const uint64_t* uSu, const uint64_t* uSv, const uint64_t* dT,
                                               const uint64_t* dSu, const uint64_t* dSv, const int* ktab,
                                               void* out, int T, int NE, int top_k, int H, int I, int KC,
                                               int max_tokens, void* ws, void* stream, const float* expert_bias,
                                               int n_group, int topk_group, float routed_limit, float shared_limit,
                                               float routed_scaling,
                                               const void* hidden, const float* norm_w, void* norm_out, float eps,
                                               const void* suh_p0, void* xh_p0, const void* suh_p1, void* xh_p1,
                                               const void* suh_p2, void* xh_p2) {
    if (T < 1 || T > 8 || T > max_tokens || NE != 512 || top_k > 16 || H != 4096 || I % 128 || KC < 1 ||
        (H / 128) % KC)
        return (int)cudaErrorInvalidValue;
    const bool group = expert_bias != nullptr && n_group >= 2 && topk_group >= 1;
    if (!group || NE % n_group != 0 || NE / n_group != 64 || topk_group > n_group)
        return (int)cudaErrorInvalidValue;
    // The fused post-MoE epilogue (norm + prerot) is all-or-nothing and needs the counter slot cntc[NBD + 1]
    // inside the 256-aligned cntc part (right after the router's counter at cntc[NBD]).
    const bool fused_epi = norm_w != nullptr;
    if (fused_epi &&
        (hidden == nullptr || norm_out == nullptr || eps <= 0.0f ||
         (size_t)(H / 128) * 4 + 4 > 256))
        return (int)cudaErrorInvalidValue;
    if ((suh_p0 == nullptr) != (xh_p0 == nullptr) || (suh_p1 == nullptr) != (xh_p1 == nullptr) ||
        (suh_p2 == nullptr) != (xh_p2 == nullptr))
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const int emax = max_tokens * (top_k + 1);
    FmoeWs w = fmoe_layout(static_cast<char*>(ws), emax, H, I, KC);
    const int E = T * (top_k + (shared_gate ? 1 : 0));

    ling_router_and_route_group_kernel<<<32, 256, 0, st>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)router_w, logits, (const __nv_bfloat16*)shared_gate,
        expert_bias, routed_scaling, w.route, emax, T, NE, H, top_k, n_group, topk_group,
        (int*)w.cntc + H / 128);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;

    GateUpArgs ga{(const __nv_bfloat16*)x, w.route, emax, gT, gSu, gSv, uT, uSu, uSv, dSu, ktab,
                  (float*)w.part, (int*)w.cnt, (half*)w.xd, H, I, KC, NE, routed_limit, shared_limit};
    const int jobs_a = E * 2 * (I / 128) * KC;
    const size_t smem_a = (size_t)WARPS * GT * ((H / 128 / KC) * 128 + 8) * sizeof(half);
    if (smem_a > 48 * 1024) return (int)cudaErrorInvalidValue;
    ling_gate_up_kernel<<<(jobs_a + WARPS - 1) / WARPS, WARPS * 32, smem_a, st>>>(ga);
    rc = GB_CHECK_LAUNCH();
    if (rc) return rc;

    DownArgs da{(const half*)w.xd, w.route, emax, dT, dSv, ktab, (float*)w.partd, (float*)w.y, (int*)w.cntd,
                (int*)w.cntc, (__nv_bfloat16*)out, H, I,
                fused_epi ? (const __nv_bfloat16*)hidden : nullptr,
                norm_w, fused_epi ? (__nv_bfloat16*)norm_out : nullptr, eps,
                (const half*)suh_p0, (half*)xh_p0,
                (const half*)suh_p1, (half*)xh_p1,
                (const half*)suh_p2, (half*)xh_p2,
                fused_epi ? (int*)w.cntc + (H >> 7) + 1 : nullptr};
    const int jobs_b = E * (H / 128) * (I / 128);
    ling_down_kernel<<<(jobs_b + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(da);
    return GB_CHECK_LAUNCH();
}


// ===========================================================================
// Ling-owned prefill MoE fusions
// ===========================================================================
// ling_fmoe_prefill.cu — Ling-owned FrostMoE prefill (bailing_moe_v3 / Ling-3.1), bit-identical to fmoe.cu's
// gb_fmoep_forward + glue::add_rmsnorm with two fusions (scan+scatter, combine+add_rmsnorm). Split into its own
// translation unit (the combined TU tripped ptxas under -lineinfo). The helper block is a verbatim copy of
// ling_fmoe.cu's (same device functions, same math), so the kernels here see the exact same helper set.

namespace lingfmoe_prefill {

constexpr int GT = 8;             // tokens a group (mma n8)
constexpr int WARPS = 4;          // warps a block
constexpr float RS128 = 0.088388347648318440f;   // 1 / sqrt(128)
constexpr float MUL1_SCALE = 1774.0f / 262144.0f; // fp16 0x1EEE
constexpr float MUL1_BIAS = -10.3828125f;         // fp16 0xC931
constexpr uint32_t MUL1_MUL = 0x83DCD12Du;
__device__ __forceinline__ uint32_t h2u(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void had128(float (&v)[4], int lane) {
    const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2;
    v[1] = a1 + a3;
    v[2] = a0 - a2;
    v[3] = a1 - a3;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool hi = (lane & m) != 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float p = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = hi ? (p - v[i]) : (v[i] + p);
        }
    }
}

__device__ __forceinline__ void ld_bf16x4(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}
__device__ __forceinline__ void ld_bf16x4_g(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}
__device__ __forceinline__ void ld_f16x4(const half* p, float (&v)[4]) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}
__device__ __forceinline__ void st_f16x4(half* p, const float (&v)[4]) {
    uint2 u;
    u.x = h2u(v[0], v[1]);
    u.y = h2u(v[2], v[3]);
    *reinterpret_cast<uint2*>(p) = u;
}

__device__ __forceinline__ float lpf_bf2f(uint16_t u) { return __uint_as_float((uint32_t)u << 16); }
__device__ __forceinline__ uint16_t lpf_f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ void lpf_unpack8(const uint4& u, float (&f)[8]) {
    f[0] = __uint_as_float(u.x << 16); f[1] = __uint_as_float(u.x & 0xFFFF0000u);
    f[2] = __uint_as_float(u.y << 16); f[3] = __uint_as_float(u.y & 0xFFFF0000u);
    f[4] = __uint_as_float(u.z << 16); f[5] = __uint_as_float(u.z & 0xFFFF0000u);
    f[6] = __uint_as_float(u.w << 16); f[7] = __uint_as_float(u.w & 0xFFFF0000u);
}
__device__ __forceinline__ uint32_t lpf_pack2(float a, float b) {
    return (uint32_t)lpf_f2bf(a) | ((uint32_t)lpf_f2bf(b) << 16);
}
__device__ __forceinline__ uint4 lpf_pack8(const float (&f)[8]) {
    uint4 u;
    u.x = lpf_pack2(f[0], f[1]); u.y = lpf_pack2(f[2], f[3]);
    u.z = lpf_pack2(f[4], f[5]); u.w = lpf_pack2(f[6], f[7]);
    return u;
}
__device__ __forceinline__ float lpf_warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ void st_bf16x4(__nv_bfloat16* p, const float (&v)[4]) {
    __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
}
__device__ __forceinline__ float4 ldcg4(const float* p) { return __ldcg(reinterpret_cast<const float4*>(p)); }
template <int K2>
struct GeoK {
    static constexpr int NW = 4 * K2;                 // 32-bit words a tile
    static constexpr int NL = gbx::lane_words<K2>();  // words a lane's 8 windows span
};

template <int K2>
__device__ __forceinline__ void lane_origin(int lane, int& i0, int& off0) {
    constexpr int RING = 32 * GeoK<K2>::NW;
    const int s0 = (gbx::stream_end<K2>(8 * lane) - 16 + RING) % RING;
    i0 = s0 >> 5;
    off0 = s0 & 31;
}

template <int K2>
__device__ __forceinline__ void tile_load(const uint32_t* __restrict__ tile, int i0, uint32_t (&w)[GeoK<K2>::NL]) {
#pragma unroll
    for (int q = 0; q < GeoK<K2>::NL; ++q) {
        int idx = i0 + q;
        if (idx >= GeoK<K2>::NW) idx -= GeoK<K2>::NW;
        w[q] = __ldg(tile + idx);
    }
}
template <int K2>
__device__ __forceinline__ uint32_t lane_window(const uint32_t* w, int off0, int j) {
    constexpr int NL = GeoK<K2>::NL;
    const uint32_t c = NL > 2 ? w[2] : 0u;
    const uint32_t hi = __funnelshift_l(w[1], w[0], off0);
    const uint32_t mid = __funnelshift_l(c, w[1], off0);
    const uint32_t lo = c << off0;
    return gbx::window16(hi, mid, lo, gbx::stream_end<K2>(j) - gbx::stream_end<K2>(0));
}
__device__ __forceinline__ float route_logit(const void* logits, int f32, size_t i) {
    return f32 ? __ldg(static_cast<const float*>(logits) + i)
               : __bfloat162float(__ldg(static_cast<const __nv_bfloat16*>(logits) + i));
}
// The Ling-3.1 MoE router GEMV helpers: a VERBATIM copy of ling_kda_proj_fused.cu's ling_fma8_bf16 and
__device__ __forceinline__ void g2(float a, float b, float& m1, float& m2) {
    if (a >= b) {
        m1 = a;
        m2 = b;
    } else {
        m1 = b;
        m2 = a;
    }
}

__device__ __forceinline__ void g4(float a0, float a1, float a2, float a3, float& m1, float& m2) {
    float h1, l1, h2, l2;
    g2(a0, a1, h1, l1);
    g2(a2, a3, h2, l2);
    if (h1 >= h2) {
        m1 = h1;
        m2 = fmaxf(l1, h2);
    } else {
        m1 = h2;
        m2 = fmaxf(l2, h1);
    }
}
__device__ __forceinline__ void g_butterfly(float& v1, float& v2) {
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        const float o1 = __shfl_xor_sync(0xffffffffu, v1, m);
        const float o2 = __shfl_xor_sync(0xffffffffu, v2, m);
        g4(v1, v2, o1, o2, v1, v2);
    }
}

// The group (0..7) of expert e under the 64-expert groups of the group-limited router.
__device__ __forceinline__ int group_of(int e) { return e >> 6; }

// ---------------------------------------------------------------------------------------------------------------
// Prefill path (many tokens a call): a verbatim copy of fmoe.cu's prefill kernels (route, rotate, gate/up, down,
// combine) renamed ling_p_*, with two clean fusions:
//   * ling_p_scan_and_scatter_kernel merges p_scan_kernel and p_scatter_kernel into one launch (the scatter body
//     runs inside the scan's single 1024-thread CTA after a barrier; every sorted slot's values depend only on
//     s_tok[p]/s_exp[p], never on which slot p it was assigned, so the downstream kernels' outputs are
//     bit-identical);
//   * ling_p_combine_add_rmsnorm_kernel merges p_combine_kernel with glue's rmsnorm_k<true> (the post-MoE residual
//     add + next input norm) into one 256-thread CTA per token, staging the rounded bf16 combine codes in shared
//     memory so the normalization pass reads them once instead of re-reading global L2.
// Bits: every expression below is the source kernel's verbatim (same operands, same order, same roundings).
// ---------------------------------------------------------------------------------------------------------------
constexpr int PT = 80;           // tokens a tile (REG: 128 STACK: 0)
constexpr int PNJ = PT / 8;      // the n tiles a tile fills (acc[2][PNJ][4] and the zero-init loop MUST match this)
constexpr int PW = 8;            // warps a prefill block
constexpr int PXS = 128 + 8;     // gate/up staging row (halves): one 128-wide k block, +8 for conflict-free B reads
constexpr int POS = 128 + 4;     // gate/up output row (floats)
constexpr int PKC = 64;          // k a staged chunk (4 k tiles)
constexpr int PKS = PKC + 8;     // staged row stride (halves): conflict-free B reads; gate/up 46,080 B, 2 blocks 90 KiB

__device__ __forceinline__ uint32_t p_fma_h2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// a 16-bit window -> 0x6400 + bytesum(window * MUL1_MUL): the bits of half(1024 + bytesum)
__device__ __forceinline__ uint32_t p_bytesum_bits(uint32_t window16) {
    return __dp4a(window16 * MUL1_MUL, 0x01010101u, 0x6400u);
}

__device__ __forceinline__ uint32_t p_pair(uint32_t d0, uint32_t d1) {
    return p_fma_h2(__byte_perm(d0, d1, 0x5410), 0x1EEE1EEEu, 0xC931C931u);
}

template <int K2>
__device__ __forceinline__ void tile_decode_h(const uint32_t* w, int off0, uint32_t (&a)[4]) {
    uint32_t d[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) d[j] = p_bytesum_bits(lane_window<K2>(w, off0, j));
    a[0] = p_pair(d[0], d[1]);
    a[1] = p_pair(d[4], d[5]);
    a[2] = p_pair(d[2], d[3]);
    a[3] = p_pair(d[6], d[7]);
}

// the first k tile of a tile row pair (both weight tiles of the warp)
template <int K2>
__device__ __forceinline__ void p_prime(const uint32_t* __restrict__ t0, int lane, uint32_t (&cur)[2][3]) {
    constexpr int NW = GeoK<K2>::NW, NL = GeoK<K2>::NL;
    int i0, off0;
    lane_origin<K2>(lane, i0, off0);
    uint32_t w[NL];
    tile_load<K2>(t0, i0, w);
#pragma unroll
    for (int q = 0; q < NL; ++q) cur[0][q] = w[q];
    tile_load<K2>(t0 + NW, i0, w);
#pragma unroll
    for (int q = 0; q < NL; ++q) cur[1][q] = w[q];
}

__device__ __forceinline__ void p_ldsm_x4(uint32_t (&r)[4], uint32_t saddr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(saddr)
                 : "memory");
}

template <int K2, int NJ>
__device__ __forceinline__ void p_mm2(const uint32_t* __restrict__ t0, size_t stride_k, int nkt, bool more, uint32_t xs_lane,
                                      int lane, uint32_t (&cur)[2][3], float (&acc)[2][10][4]) {
    constexpr int NW = GeoK<K2>::NW, NL = GeoK<K2>::NL;
    int i0, off0;
    lane_origin<K2>(lane, i0, off0);
#pragma unroll 1
    for (int kt = 0; kt < nkt; ++kt) {
        uint32_t nxt[2][NL];
        if (kt + 1 < nkt || more) {
            const uint32_t* row = t0 + (size_t)(kt + 1) * stride_k;
            tile_load<K2>(row, i0, nxt[0]);
            tile_load<K2>(row + NW, i0, nxt[1]);
        } else {
#pragma unroll
            for (int m = 0; m < 2; ++m)
#pragma unroll
                for (int q = 0; q < NL; ++q) nxt[m][q] = cur[m][q];
        }
        uint32_t a0[4], a1[4];
        tile_decode_h<K2>(cur[0], off0, a0);
        tile_decode_h<K2>(cur[1], off0, a1);
        const uint32_t base = xs_lane + (uint32_t)kt * 32u;
#pragma unroll
        for (int jp = 0; jp < NJ / 2; ++jp) {
            uint32_t r[4];
            p_ldsm_x4(r, base + (uint32_t)jp * (16u * PKS * 2u));
            mma16816(acc[0][2 * jp], a0, r[0], r[1]);
            mma16816(acc[1][2 * jp], a1, r[0], r[1]);
            mma16816(acc[0][2 * jp + 1], a0, r[2], r[3]);
            mma16816(acc[1][2 * jp + 1], a1, r[2], r[3]);
        }
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int q = 0; q < NL; ++q) cur[m][q] = nxt[m][q];
    }
}

template <int NJ>
__device__ __forceinline__ bool p_mm_bits2(int k2, const uint32_t* t0, size_t stride_k, int nkt, bool more, uint32_t xs_lane,
                                           int lane, uint32_t (&cur)[2][3], float (&acc)[2][10][4]) {
    switch (k2) {
        case 2: p_mm2<2, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 3: p_mm2<3, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 4: p_mm2<4, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 5: p_mm2<5, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 6: p_mm2<6, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 7: p_mm2<7, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 8: p_mm2<8, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 10: p_mm2<10, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 12: p_mm2<12, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 14: p_mm2<14, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        case 16: p_mm2<16, NJ>(t0, stride_k, nkt, more, xs_lane, lane, cur, acc); return true;
        default: return false;
    }
}

__device__ __forceinline__ bool p_prime_bits(int k2, const uint32_t* t0, int lane, uint32_t (&cur)[2][3]) {
    switch (k2) {
        case 2: p_prime<2>(t0, lane, cur); return true;
        case 3: p_prime<3>(t0, lane, cur); return true;
        case 4: p_prime<4>(t0, lane, cur); return true;
        case 5: p_prime<5>(t0, lane, cur); return true;
        case 6: p_prime<6>(t0, lane, cur); return true;
        case 7: p_prime<7>(t0, lane, cur); return true;
        case 8: p_prime<8>(t0, lane, cur); return true;
        case 10: p_prime<10>(t0, lane, cur); return true;
        case 12: p_prime<12>(t0, lane, cur); return true;
        case 14: p_prime<14>(t0, lane, cur); return true;
        case 16: p_prime<16>(t0, lane, cur); return true;
        default: return false;
    }
}

__device__ __forceinline__ bool p_mm_run(int ntn, int k2, const uint32_t* t0, size_t stride_k, int nkt, bool more,
                                         uint32_t xs_lane, int lane, uint32_t (&cur)[2][3], float (&acc)[2][10][4]) {
    if (ntn <= 2) return p_mm_bits2<2>(k2, t0, stride_k, nkt, more, xs_lane, lane, cur, acc);
    if (ntn <= 4) return p_mm_bits2<4>(k2, t0, stride_k, nkt, more, xs_lane, lane, cur, acc);
    if (ntn <= 8) return p_mm_bits2<8>(k2, t0, stride_k, nkt, more, xs_lane, lane, cur, acc);
    return p_mm_bits2<10>(k2, t0, stride_k, nkt, more, xs_lane, lane, cur, acc);
}

// 16-byte async copy to shared memory, or zeros without reading when `read` is false
__device__ __forceinline__ void p_cp16(void* dst, const void* src, bool read) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(read ? 16 : 0));
}
__device__ __forceinline__ void p_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void p_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

struct PGateUp {
    const half* xr;  // [sorted slot][2][H] rotated inputs
    const int *hdr, *t_exp, *t_p0, *t_n;
    const uint64_t *gT, *gSv, *uT, *uSv, *dSu;
    const int* ktab;
    half* xd;  // [sorted slot][I], rotated for down
    int H, I;
    int ne;           // routed expert count (the shared expert is entry ne)
    float routed_limit, shared_limit;  // SwiGLU clamps (<= 0: no clamp)
};

// Block = (tile, 128-column block). Warps 0-3: gate, 4-7: up, 32 columns each. The tile's pre-rotated rows are staged in
// 64-wide k chunks, double buffered with async copies, while every warp runs its columns over the previous chunk.
__global__ void __launch_bounds__(PW * 32, 2) ling_p_gate_up_kernel(PGateUp a) {
    extern __shared__ __align__(16) unsigned char p_smem[];
    const int cb = blockIdx.x, tile = blockIdx.y;
    if (tile >= a.hdr[0]) return;
    const int e = a.t_exp[tile], p0 = a.t_p0[tile], n = a.t_n[tile];
    const float limit = (e == a.ne) ? a.shared_limit : a.routed_limit;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int mat = warp >> 2, wq = warp & 3;
    half* xs = reinterpret_cast<half*>(p_smem);  // [2 buffers][2 mats][PT][PKS]
    constexpr int BUF = 2 * PT * PKS;
    const int bits = a.ktab[e * 3 + mat];
    const uint32_t* Tm = reinterpret_cast<const uint32_t*>(mat ? a.uT[e] : a.gT[e]);
    const size_t nwb = 4 * (size_t)bits;  // K2 words a tile (4 * K2)
    const size_t stride_k = (size_t)(a.I >> 4) * nwb;
    const int nt0 = cb * 8 + wq * 2;
    const int ntn = (n + 7) >> 3;
    const int NC = a.H / PKC;
    const int q8 = (threadIdx.x & 7) * 8;
    auto stage = [&](int c, int buf) {
        for (int r = threadIdx.x >> 3; r < 2 * PT; r += (PW * 32) / (PKC / 8)) {
            const int um = (r >= PT);
            const int tk = um ? (r - PT) : r;
            const bool in = tk < n;
            const half* src = a.xr + ((size_t)(p0 + (in ? tk : 0)) * 2 + um) * a.H + c * PKC + q8;
            p_cp16(xs + (size_t)buf * BUF + (size_t)r * PKS + q8, src, in);
        }
        p_commit();
    };
    float acc[2][PNJ][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int j = 0; j < PNJ; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][j][c] = 0.f;

    stage(0, 0);
    uint32_t cur[2][3];
    p_prime_bits(bits, Tm + (size_t)nt0 * nwb, lane, cur);
    // this lane's ldmatrix row: matrix i = lane / 8 of an x4 load is (n tile i >> 1, k half i & 1), row lane % 8 of it
    const uint32_t xs_lane0 = (uint32_t)__cvta_generic_to_shared(xs) +
                              (uint32_t)(((lane >> 4) * 8 + (lane & 7)) * PKS * 2 + ((lane >> 3) & 1) * 16);
    for (int c = 0; c < NC; ++c) {
        p_wait_all();
        __syncthreads();  // chunk c landed for every thread; chunk c - 1's buffer is free
        if (c + 1 < NC) stage(c + 1, (c + 1) & 1);
        const uint32_t* t0 = Tm + (size_t)(c * (PKC / 16)) * stride_k + (size_t)nt0 * nwb;
        p_mm_run(ntn, bits, t0, stride_k, PKC / 16, c + 1 < NC, xs_lane0 + (uint32_t)(((c & 1) * BUF + mat * PT * PKS) * 2), lane,
                 cur, acc);
    }
    __syncthreads();

    // epilogue in two halves of PT / 2 = 40 tokens so the fp32 outputs fit in the staging area: [2][40][POS] (42,240 B)
    float* outs = reinterpret_cast<float*>(p_smem);
    constexpr int HT = PT / 2;
    constexpr int HNJ = HT / 8;
    const int g = lane >> 2, t = lane & 3;
    const int col = cb * 128 + 4 * lane;
    float sg[4], sv[4], sd[4];
    ld_f16x4(reinterpret_cast<const half*>(a.gSv[e]) + col, sg);
    ld_f16x4(reinterpret_cast<const half*>(a.uSv[e]) + col, sv);
    ld_f16x4(reinterpret_cast<const half*>(a.dSu[e]) + col, sd);
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int jj = 0; jj < HNJ; ++jj) {
                const int j = hh * HNJ + jj;
                if (j < ntn) {
                    const int c = wq * 32 + m * 16 + g, tk = jj * 8 + 2 * t;
                    float* o = outs + ((size_t)mat * HT + tk) * POS + c;
                    o[0] = acc[m][j][0];
                    o[POS] = acc[m][j][1];
                    o[8] = acc[m][j][2];
                    o[POS + 8] = acc[m][j][3];
                }
            }
        __syncthreads();
        const int nh = min(HT, n - hh * HT);
        for (int tk = warp; tk < nh; tk += PW) {
            float vg[4], vu[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                vg[i] = outs[(size_t)tk * POS + 4 * lane + i];
                vu[i] = outs[((size_t)HT + tk) * POS + 4 * lane + i];
            }
            had128(vg, lane);
            had128(vu, lane);
            float act[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float gv = vg[i] * RS128 * sg[i];
                float uv = vu[i] * RS128 * sv[i];
                if (limit > 0.0f) {
                    gv = fminf(gv, limit);
                    uv = fmaxf(-limit, fminf(uv, limit));
                }
                act[i] = gv / (1.f + __expf(-gv)) * uv * sd[i];
            }
            had128(act, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) act[i] *= RS128;
            st_f16x4(a.xd + (size_t)(p0 + hh * HT + tk) * a.I + col, act);
        }
        __syncthreads();
    }
}

struct PDown {
    const half* xd;
    const int *hdr, *t_exp, *t_p0, *t_n;
    const uint64_t *dT, *dSv;
    const int* ktab;
    float* y;  // [sorted slot][H]
    int H, I;
};

constexpr int DOS = 256 + 4;  // down output row (floats)

// Block = (tile, 256-column block), 8 warps of 32 columns. The tile's activation rows are staged one 128-wide k block
// at a time and the epilogue runs in two halves of PT / 2 = 40 tokens; the launcher sizes shared memory to the larger of
// double-buffered staging (23,040 B) and the epilogue (41,600 B), so two blocks (83.2 KiB) fit an SM.
__global__ void __launch_bounds__(PW * 32, 2) ling_p_down_kernel(PDown a) {
    extern __shared__ __align__(16) unsigned char p_smem[];
    const int cb = blockIdx.x, tile = blockIdx.y;
    if (tile >= a.hdr[0]) return;
    const int e = a.t_exp[tile], p0 = a.t_p0[tile], n = a.t_n[tile];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    half* xs = reinterpret_cast<half*>(p_smem);  // [2 buffers][PT][PKS]
    constexpr int BUF = PT * PKS;
    const int bits = a.ktab[e * 3 + 2];
    const uint32_t* Tm = reinterpret_cast<const uint32_t*>(a.dT[e]);
    const size_t nwb = 4 * (size_t)bits;  // K2 words a tile (4 * K2)
    const size_t stride_k = (size_t)(a.H >> 4) * nwb;
    const int nt0 = cb * 16 + warp * 2;
    const int ntn = (n + 7) >> 3;
    const int NC = a.I / PKC;
    const int q8 = (threadIdx.x & 7) * 8;
    auto stage = [&](int c, int buf) {
        #pragma unroll
        for (int tk = threadIdx.x >> 3; tk < PT; tk += (PW * 32) / (PKC / 8)) {
            const bool in = tk < n;
            p_cp16(xs + (size_t)buf * BUF + (size_t)tk * PKS + q8, a.xd + (size_t)(p0 + (in ? tk : 0)) * a.I + c * PKC + q8, in);
        }
        p_commit();
    };
    float acc[2][PNJ][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int j = 0; j < PNJ; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][j][c] = 0.f;
    stage(0, 0);
    uint32_t cur[2][3];
    p_prime_bits(bits, Tm + (size_t)nt0 * nwb, lane, cur);
    const uint32_t xs_lane0 = (uint32_t)__cvta_generic_to_shared(xs) +
                              (uint32_t)(((lane >> 4) * 8 + (lane & 7)) * PKS * 2 + ((lane >> 3) & 1) * 16);
    for (int c = 0; c < NC; ++c) {
        p_wait_all();
        __syncthreads();
        if (c + 1 < NC) stage(c + 1, (c + 1) & 1);
        p_mm_run(ntn, bits, Tm + (size_t)(c * (PKC / 16)) * stride_k + (size_t)nt0 * nwb, stride_k, PKC / 16, c + 1 < NC,
                 xs_lane0 + (uint32_t)((c & 1) * BUF * 2), lane, cur, acc);
    }
    __syncthreads();

    constexpr int HT = PT / 2;
    constexpr int HNJ = HT / 8;
    float* outs = reinterpret_cast<float*>(p_smem);  // [HT][DOS] (41,600 B at HT = 40)
    const int g = lane >> 2, t = lane & 3;
    const half* dsv = reinterpret_cast<const half*>(a.dSv[e]);
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int jj = 0; jj < HNJ; ++jj) {
                const int j = hh * HNJ + jj;
                if (j < ntn) {
                    const int c = warp * 32 + m * 16 + g, tk = jj * 8 + 2 * t;
                    float* o = outs + (size_t)tk * DOS + c;
                    o[0] = acc[m][j][0];
                    o[DOS] = acc[m][j][1];
                    o[8] = acc[m][j][2];
                    o[DOS + 8] = acc[m][j][3];
                }
            }
        __syncthreads();
        const int nh = min(HT, n - hh * HT);
        for (int u = warp; u < 2 * nh; u += PW) {
            const int tk = u >> 1, hb = (u & 1) * 128;
            float v[4], sv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = outs[(size_t)tk * DOS + hb + 4 * lane + i];
            const int col = cb * 256 + hb + 4 * lane;
            ld_f16x4(dsv + col, sv);
            had128(v, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= RS128 * sv[i];
            *reinterpret_cast<float4*>(a.y + (size_t)(p0 + hh * HT + tk) * a.H + col) = make_float4(v[0], v[1], v[2], v[3]);
        }
        __syncthreads();
    }
}

// Route, one warp a token: top-k softmax (lowest index wins ties), the shared expert as entry K with sigmoid(gate).
__global__ void __launch_bounds__(256) ling_p_route_kernel(const void* __restrict__ logits, int logits_f32,
                                                           const __nv_bfloat16* __restrict__ shared_gate,
                                                           int* __restrict__ ent_exp, float* __restrict__ ent_w,
                                                           int* __restrict__ cnt, int T, int NE, int K) {
    __shared__ float s_vk[8][16];
    __shared__ int s_ik[8][16];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tok = blockIdx.x * 8 + warp;
    if (tok >= T) return;
    const int KT = K + (shared_gate ? 1 : 0);
    float l[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int e = lane + 32 * i;
        l[i] = e < NE ? route_logit(logits, logits_f32, (size_t)tok * NE + e) : -INFINITY;
    }
    float* vk = s_vk[warp];
    int* ik = s_ik[warp];
    float top_v = 0.f;
    for (int k = 0; k < K; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            if (l[i] > bv) { bv = l[i]; bi = lane + 32 * i; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
#pragma unroll
        for (int i = 0; i < 16; ++i)
            if (lane + 32 * i == bi) l[i] = -INFINITY;
        if (k == 0) top_v = bv;
        if (bi >= NE) bi = k;  // NaN-safe: keep a valid expert id
        if (lane == 0) { vk[k] = bv; ik[k] = bi; }
    }
    if (lane == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += expf(vk[k] - top_v);
        const float inv = 1.f / sum;
        for (int k = 0; k < K; ++k) {
            ent_exp[tok * KT + k] = ik[k];
            ent_w[tok * KT + k] = expf(vk[k] - top_v) * inv;
            atomicAdd(cnt + ik[k], 1);
        }
        if (shared_gate) {
            ent_exp[tok * KT + K] = NE;
            ent_w[tok * KT + K] = 1.f / (1.f + expf(-__bfloat162float(shared_gate[tok])));
            atomicAdd(cnt + NE, 1);
        }
    }
}

// Fusion A2: the p_scan_and_scatter_kernel body run by the LAST route-group CTA (a self-resetting arrival counter
// in hdr[1]) on this block's 256 threads: each thread covers experts e = tid, tid + 256, tid + 512, tid + 768
// (the same Hillis-Steele inclusive scan math over 1024 shared slots as the standalone kernel), then the scatter
// body over E entries. Bit-identical to the separate ling_p_scan_and_scatter_kernel launch (same prefix sums, same
// hdr[0] total, same per-entry atomicAdd slot order). `s_c`/`s_t` are the kernel's shared scan slots (the device
// function cannot allocate shared memory of its own).
__device__ void p_scan_scatter_fused(int* __restrict__ s_c, int* __restrict__ s_t,
                                     int* __restrict__ cnt, int* __restrict__ cur, int* __restrict__ eoff,
                                     int* __restrict__ t_exp, int* __restrict__ t_p0, int* __restrict__ t_n,
                                     int* __restrict__ hdr, const int* __restrict__ ent_exp,
                                     int* __restrict__ s_tok, int* __restrict__ s_exp, int* __restrict__ e_pos,
                                     int NEX, int E, int KT) {
    const int tid = threadIdx.x;
    for (int e = tid; e < 1024; e += 256) {
        const int c = e < NEX ? cnt[e] : 0;
        const int nt = (c + PT - 1) / PT;
        s_c[e] = c;
        s_t[e] = nt;
    }
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {
        int ac[4], bc[4], ec[4];
        int n = 0;
        for (int e = tid; e < 1024; e += 256) {
            ec[n] = e;
            ac[n] = e >= o ? s_c[e - o] : 0;
            bc[n] = e >= o ? s_t[e - o] : 0;
            ++n;
        }
        __syncthreads();
        for (int i = 0; i < n; ++i) {
            s_c[ec[i]] += ac[i];
            s_t[ec[i]] += bc[i];
        }
        __syncthreads();
    }
    for (int e = tid; e < NEX; e += 256) {
        const int c = cnt[e];
        const int nt = (c + PT - 1) / PT;
        const int eo = s_c[e] - c, to = s_t[e] - nt;
        eoff[e] = eo;
        cur[e] = 0;
        cnt[e] = 0;
        for (int j = 0; j < nt; ++j) {
            t_exp[to + j] = e;
            t_p0[to + j] = eo + j * PT;
            t_n[to + j] = min(PT, c - j * PT);
        }
    }
    if (tid == 0) {
        hdr[0] = s_t[1023];
        hdr[2] = 1;   // cnt is clean after this scan (zeroed for e < NEX)
    }
    __syncthreads();
    for (int i = tid; i < E; i += 256) {
        const int ex = ent_exp[i];
        const int p = eoff[ex] + atomicAdd(cur + ex, 1);
        s_tok[p] = i / KT;
        s_exp[p] = ex;
        e_pos[i] = p;
    }
}

// Route, one warp a token: the group-limited noaux_tc route (see ling_route_group_body), the shared expert as entry K.
__global__ void __launch_bounds__(256) ling_p_route_group_kernel(const void* __restrict__ logits, int logits_f32,
                                                                 const __nv_bfloat16* __restrict__ shared_gate,
                                                                 const float* __restrict__ expert_bias, float scale,
                                                                 int* __restrict__ ent_exp, float* __restrict__ ent_w,
                                                                 int* __restrict__ cnt, int T, int NE, int K, int n_group,
                                                                 int topk_group, int* __restrict__ cur,
                                                                 int* __restrict__ eoff, int* __restrict__ t_exp,
                                                                 int* __restrict__ t_p0, int* __restrict__ t_n,
                                                                 int* __restrict__ hdr, int* __restrict__ s_tok,
                                                                 int* __restrict__ s_exp, int* __restrict__ e_pos,
                                                                 int E, int KT, int NEX) {
    __shared__ float s_suk[8][16];
    __shared__ int s_ik[8][16];
    __shared__ bool s_last_block;
    __shared__ int s_c[1024], s_t[1024];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tok = blockIdx.x * 8 + warp;
    // Invalid tail warps (tok >= T) skip the route body but still arrive at the grid-wide barrier below, so
    // the last CTA's fused scan can use all 256 threads.
    if (tok < T) {
    float sb[16], su[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int e = lane + 32 * i;
        if (e < NE) {
            const float lg = route_logit(logits, logits_f32, (size_t)tok * NE + e);
            su[i] = 1.0f / (1.0f + expf(-lg));
            sb[i] = su[i] + expert_bias[e];
        } else {
            su[i] = -INFINITY;
            sb[i] = -INFINITY;
        }
    }
    float* suk = s_suk[warp];
    int* ik = s_ik[warp];
    float gs[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        float a, b;
        g2(sb[2 * g], sb[2 * g + 1], a, b);
        g_butterfly(a, b);
        gs[g] = a + b;
    }
    uint32_t sel = 0u;  // bit g: group g selected (register state)
    for (int q = 0; q < topk_group; ++q) {
        float bv = -INFINITY;
        int bg = 0x7fffffff;
#pragma unroll
        for (int g = 0; g < 8; ++g)
            if (!(sel & (1u << g)) && (gs[g] > bv || (gs[g] == bv && g < bg))) {
                bv = gs[g];
                bg = g;
            }
        if (bg >= 8) bg = q;  // NaN-safe: never select a bogus group
        sel |= 1u << bg;
    }
    uint32_t excl = 0u;  // bit i: expert lane + 32i already picked (per-lane register state)
    for (int k = 0; k < K; ++k) {
        float bv = -INFINITY;
        float bsu = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (excl & (1u << i)) continue;
            const int e = lane + 32 * i;
            if (e >= NE || !(sel & (1u << group_of(e)))) continue;
            const float lg = route_logit(logits, logits_f32, (size_t)tok * NE + e);
            const float su = 1.0f / (1.0f + expf(-lg));
            const float sb = su + expert_bias[e];
            if (sb > bv) { bv = sb; bsu = su; bi = e; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const float osu = __shfl_xor_sync(0xffffffffu, bsu, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bsu = osu;
                bi = oi;
            }
        }
        if ((bi & 31) == lane && bi < NE) excl |= 1u << (bi >> 5);
        if (bi >= NE) bi = k;  // NaN-safe: keep a valid expert id
        if (lane == 0) { suk[k] = bsu; ik[k] = bi; }
    }
    if (lane == 0) {
        float sum = 0.f;
        for (int k = 0; k < K; ++k) sum += suk[k];
        const float inv = 1.f / (sum + 1e-20f);
        for (int k = 0; k < K; ++k) {
            ent_exp[tok * KT + k] = ik[k];
            ent_w[tok * KT + k] = suk[k] * inv * scale;
            atomicAdd(cnt + ik[k], 1);
        }
        if (shared_gate) {
            ent_exp[tok * KT + K] = NE;
            ent_w[tok * KT + K] = 1.f / (1.f + expf(-__bfloat162float(shared_gate[tok])));
            atomicAdd(cnt + NE, 1);
        }
    }
    }  // if (tok < T)
    // Fused scan+scatter: the LAST CTA to finish routing (a grid-wide arrival counter in hdr[1]) runs the exact
    // p_scan_and_scatter_kernel body on this block's 256 threads, so the standalone scan launch disappears. The
    // counter is self-resetting; the host re-zeroes it every call as a guard against a failed launch leaving it
    // dirty.
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        const int done = atomicAdd(hdr + 1, 1);
        s_last_block = done == (int)gridDim.x - 1;
        if (s_last_block) {
            hdr[1] = 0;
        }
    }
    __syncthreads();
    if (s_last_block) {
        __threadfence();
        p_scan_scatter_fused(s_c, s_t, cnt, cur, eoff, t_exp, t_p0, t_n, hdr, ent_exp, s_tok, s_exp, e_pos, NEX, E, KT);
    }
}

// Fusion A: the scan (one block of 1024: per expert the first sorted slot and its tiles; the tile table; counters back
// to zero) with the scatter body (every entry's sorted slot assignment) in the SAME CTA. The scan's eoff/cur/cnt/t_*
// writes are visible block-wide after the barrier; the atomicAdd assigns slots in the same per-entry order as the
// separate launch, and each sorted slot's values depend only on s_tok[p]/s_exp[p] (never on which slot p it was
// assigned), so downstream rows are bit-identical.
__global__ void __launch_bounds__(1024) ling_p_scan_and_scatter_kernel(int* __restrict__ cnt, int* __restrict__ cur,
                                                                       int* __restrict__ eoff, int* __restrict__ t_exp,
                                                                       int* __restrict__ t_p0, int* __restrict__ t_n,
                                                                       int* __restrict__ hdr,
                                                                       const int* __restrict__ ent_exp,
                                                                       int* __restrict__ s_tok, int* __restrict__ s_exp,
                                                                       int* __restrict__ e_pos,
                                                                       int NEX, int E, int KT) {
    __shared__ int s_c[1024], s_t[1024];
    const int e = threadIdx.x;
    const int c = e < NEX ? cnt[e] : 0;
    const int nt = (c + PT - 1) / PT;
    s_c[e] = c;
    s_t[e] = nt;
    __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) {
        const int a = e >= o ? s_c[e - o] : 0, b = e >= o ? s_t[e - o] : 0;
        __syncthreads();
        s_c[e] += a;
        s_t[e] += b;
        __syncthreads();
    }
    if (e < NEX) {
        const int eo = s_c[e] - c, to = s_t[e] - nt;
        eoff[e] = eo;
        cur[e] = 0;
        cnt[e] = 0;
        for (int j = 0; j < nt; ++j) {
            t_exp[to + j] = e;
            t_p0[to + j] = eo + j * PT;
            t_n[to + j] = min(PT, c - j * PT);
        }
    }
    if (e == 1023) hdr[0] = s_t[1023];
    __syncthreads();
    for (int i = threadIdx.x; i < E; i += 1024) {
        const int ex = ent_exp[i];
        const int p = eoff[ex] + atomicAdd(cur + ex, 1);
        s_tok[p] = i / KT;
        s_exp[p] = ex;
        e_pos[i] = p;
    }
}

// Every sorted slot's token rotated ONCE per matrix: xr[p][m][k] = fp16(H(x[tok] * suh_m[e]) / sqrt(128))[k], so the
// gate/up blocks (five column blocks a tile) only copy their rows. One block per (slot, matrix): its 8 warps split the
// 32 independent 128-wide k blocks (warp w takes kb = w, w+8, w+16, w+24), so all warps stream the SAME token row and
// expert scale rows (an 8x smaller L1 working set). Each kb is an independent 128-wide Sylvester Hadamard run wholly by
// one warp, so the output is bit-for-bit identical to one warp per entry.
__global__ void __launch_bounds__(256) ling_p_rotate_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ s_tok,
                                                            const int* __restrict__ s_exp, const uint64_t* __restrict__ gSu,
                                                            const uint64_t* __restrict__ uSu, half* __restrict__ xr, int E,
                                                            int H) {
    const int lane = threadIdx.x & 31;
    const int KB = H >> 7;
    const int p = blockIdx.x;
    if (p >= E) return;
    const int e = s_exp[p], tok = s_tok[p];
    const half* sg = reinterpret_cast<const half*>(gSu[e]);
    const half* su = reinterpret_cast<const half*>(uSu[e]);
    const __nv_bfloat16* xrow = x + (size_t)tok * H;
    half* dg = xr + ((size_t)p * 2) * H;
    half* du = dg + H;
    const int w = threadIdx.x >> 5;
#pragma unroll 4
    for (int kb = w; kb < KB; kb += 8) {
        const int k = kb * 128 + 4 * lane;
        float v[4], a[4], b[4], g4[4], u4[4];
        ld_bf16x4_g(xrow + k, v);
        ld_f16x4(sg + k, g4);
        ld_f16x4(su + k, u4);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] = v[i] * g4[i];
            b[i] = v[i] * u4[i];
        }
        had128(a, lane);
        had128(b, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] *= RS128;
            b[i] *= RS128;
        }
        st_f16x4(dg + k, a);
        st_f16x4(du + k, b);
    }
}

// out[tok] = sum over its entries in top-k order of weight * y[slot of the entry]
__global__ void __launch_bounds__(256) ling_p_combine_kernel(const float* __restrict__ y, const int* __restrict__ e_pos,
                                                             const float* __restrict__ ent_w, __nv_bfloat16* __restrict__ out,
                                                             int H, int KT) {
    const int tok = blockIdx.x;
    const int c = (blockIdx.y * 256 + threadIdx.x) * 4;
    if (c >= H) return;
    const int base = tok * KT;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (KT == 9) {
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int p = __ldg(e_pos + base + k);
            const float w = __ldg(ent_w + base + k);
            const float4 x = ldcg4(y + (size_t)p * H + c);
            v[0] += w * x.x; v[1] += w * x.y; v[2] += w * x.z; v[3] += w * x.w;
        }
    } else {
        for (int k = 0; k < KT; ++k) {
            const int p = __ldg(e_pos + base + k);
            const float w = __ldg(ent_w + base + k);
            const float4 x = ldcg4(y + (size_t)p * H + c);
            v[0] += w * x.x; v[1] += w * x.y; v[2] += w * x.z; v[3] += w * x.w;
        }
    }
    st_bf16x4(out + (size_t)tok * H + c, v);
}

struct PCombEP {
    const float* y;
    const int* e_pos;
    const float* ent_w;
    __nv_bfloat16* out;        // moe_out
    const __nv_bfloat16* hidden;      // input hidden (bf16)
    __nv_bfloat16* hidden_out;        // output hidden (in place, aliases hidden)
    const float* norm_w;
    __nv_bfloat16* norm_out;
    int H, KT;
    float eps;
    const half* next_suh;     // next layer's first projection suh (fp16 [H]) or null
    half* next_xh;            // next layer's rotated input (fp16 [T][H]) or null
};

// Fusion B: p_combine_kernel's weight sum + glue's rmsnorm_k<true> (post-MoE residual add + next input norm) in ONE
// 256-thread CTA per token. Each thread owns 8 columns per iteration (two adjacent 4-column combine groups: c0 =
// v*8, c1 = v*8+4, exactly the separate launch's (blockIdx.y, threadIdx.x) owner pair), writes the rounded bf16
// combine codes to `out` AND stages them in shared `s_h` (same bits as the global write, so the norm pass reads them
// once instead of re-reading L2), then runs the exact rmsnorm add/accumulate in the source kernel's order and the
// exact block_sum reduction on 8 warps.
__global__ void __launch_bounds__(256) ling_p_combine_add_rmsnorm_kernel(PCombEP a) {
    extern __shared__ __align__(16) uint16_t s_h[];
    __shared__ float sm[8];
    const int tok = blockIdx.x;
    const int base = tok * a.KT;
    const int nvec = a.H / 8;
    float ss = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        const int c0 = v * 8, c1 = c0 + 4;
        const size_t y0 = (size_t)tok * a.H + c0;
        float v0[4] = {0.f, 0.f, 0.f, 0.f}, v1[4] = {0.f, 0.f, 0.f, 0.f};
        if (a.KT == 9) {
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                const int p = __ldg(a.e_pos + base + k);
                const float w = __ldg(a.ent_w + base + k);
                const float4 x0 = ldcg4(a.y + (size_t)p * a.H + c0);
                const float4 x1 = ldcg4(a.y + (size_t)p * a.H + c1);
                v0[0] += w * x0.x; v0[1] += w * x0.y; v0[2] += w * x0.z; v0[3] += w * x0.w;
                v1[0] += w * x1.x; v1[1] += w * x1.y; v1[2] += w * x1.z; v1[3] += w * x1.w;
            }
        } else {
            for (int k = 0; k < a.KT; ++k) {
                const int p = __ldg(a.e_pos + base + k);
                const float w = __ldg(a.ent_w + base + k);
                const float4 x0 = ldcg4(a.y + (size_t)p * a.H + c0);
                const float4 x1 = ldcg4(a.y + (size_t)p * a.H + c1);
                v0[0] += w * x0.x; v0[1] += w * x0.y; v0[2] += w * x0.z; v0[3] += w * x0.w;
                v1[0] += w * x1.x; v1[1] += w * x1.y; v1[2] += w * x1.z; v1[3] += w * x1.w;
            }
        }
        __nv_bfloat162 a0 = __floats2bfloat162_rn(v0[0], v0[1]), b0 = __floats2bfloat162_rn(v0[2], v0[3]);
        __nv_bfloat162 a1 = __floats2bfloat162_rn(v1[0], v1[1]), b1 = __floats2bfloat162_rn(v1[2], v1[3]);
        uint2 u0, u1;
        u0.x = *reinterpret_cast<uint32_t*>(&a0); u0.y = *reinterpret_cast<uint32_t*>(&b0);
        u1.x = *reinterpret_cast<uint32_t*>(&a1); u1.y = *reinterpret_cast<uint32_t*>(&b1);
        *reinterpret_cast<uint2*>(a.out + y0) = u0;
        *reinterpret_cast<uint2*>(a.out + y0 + 4) = u1;
        *reinterpret_cast<uint2*>(s_h + v * 8) = u0;
        *reinterpret_cast<uint2*>(s_h + v * 8 + 4) = u1;
        float f[8];
        lpf_unpack8(*(const uint4*)(a.hidden + y0), f);
        const uint16_t* sh = s_h + v * 8;
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float rf = lpf_bf2f(lpf_f2bf(f[i] + lpf_bf2f(sh[i])));
            f[i] = rf;
            o[i] = rf;
            ss = __fadd_rn(ss, __fmul_rn(rf, rf));
        }
        const uint4 pk = lpf_pack8(f);
        *(uint4*)(a.hidden_out + y0) = pk;
        *(uint4*)(s_h + v * 8) = pk;
    }
    ss = lpf_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = ss;
    __syncthreads();
    float s = sm[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) s = s + sm[w];
    const float inv = 1.0f / sqrtf(s / (float)a.H + a.eps);
    __syncthreads();
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8], o[8];
        lpf_unpack8(*(const uint4*)(s_h + v * 8), f);
        const float4 w0 = *(const float4*)(a.norm_w + (size_t)v * 8), w1 = *(const float4*)(a.norm_w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = (f[i] * inv) * wv[i];
        const uint4 pk = lpf_pack8(o);
        *(uint4*)(a.norm_out + (size_t)tok * a.H + v * 8) = pk;
        *(uint4*)(s_h + v * 8) = pk;   // the normed row for the fused next-projection rotation below
    }
    __syncthreads();
    // Fusion B2: the next layer's first projection input rotation fused into the combine epilogue: the 8 warps
    // rotate the normed row (staged in s_h) by next_suh into next_xh with the exact gb_glm_p2_rotate math
    // (bf16 -> fp32, the suh fp16 scale, the 128-point Walsh-Hadamard butterfly, HAD_SCALE, fp16 stores), so the
    // next layer's P2 GEMM skips its rotate launch. Bit-identical to a separate rotate on the norm_out row.
    if (a.next_suh && a.next_xh) {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const size_t base = (size_t)tok * a.H;
        for (int blk = warp; blk < a.H / 128; blk += 8) {
            const int col = blk * 128 + 4 * lane;
            float s[4], v[4];
            const uint2 su = *reinterpret_cast<const uint2*>(a.next_suh + col);
            const float2 sa = __half22float2(*reinterpret_cast<const half2*>(&su.x));
            const float2 sb = __half22float2(*reinterpret_cast<const half2*>(&su.y));
            s[0] = sa.x; s[1] = sa.y; s[2] = sb.x; s[3] = sb.y;
            v[0] = lpf_bf2f(s_h[col]);
            v[1] = lpf_bf2f(s_h[col + 1]);
            v[2] = lpf_bf2f(s_h[col + 2]);
            v[3] = lpf_bf2f(s_h[col + 3]);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] *= s[j];
            gbx::fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] *= gbx::HAD_SCALE;
            half2 p0 = __floats2half2_rn(v[0], v[1]), p1 = __floats2half2_rn(v[2], v[3]);
            *reinterpret_cast<uint2*>(a.next_xh + base + col) =
                make_uint2(*reinterpret_cast<const uint32_t*>(&p0), *reinterpret_cast<const uint32_t*>(&p1));
        }
    }
}

struct PWs {
    int *hdr, *cnt, *cur, *eoff, *t_exp, *t_p0, *t_n, *ent_exp, *s_tok, *s_exp, *e_pos;
    float *ent_w, *y;
    half *xd, *xr;
    size_t total;
};

static int p_max_tiles(int entries, int nex) {
    const int upper = (entries + PT - 1) / PT + nex;
    return entries < upper ? entries : upper;
}

static size_t p_al256(size_t v) { return (v + 255) & ~(size_t)255; }

static PWs p_layout(char* base, int max_tokens, int top_k, int NE, int H, int I) {
    const int nex = NE + 1, emax = max_tokens * (top_k + 1), mt = p_max_tiles(emax, nex);
    PWs w;
    size_t o = 0;
    auto take = [&](size_t bytes) { char* p = base ? base + o : nullptr; o += p_al256(bytes); return p; };
    w.hdr = (int*)take(16);
    w.cnt = (int*)take((size_t)nex * 4);
    w.cur = (int*)take((size_t)nex * 4);
    w.eoff = (int*)take((size_t)nex * 4);
    w.t_exp = (int*)take((size_t)mt * 4);
    w.t_p0 = (int*)take((size_t)mt * 4);
    w.t_n = (int*)take((size_t)mt * 4);
    w.ent_exp = (int*)take((size_t)emax * 4);
    w.ent_w = (float*)take((size_t)emax * 4);
    w.s_tok = (int*)take((size_t)emax * 4);
    w.s_exp = (int*)take((size_t)emax * 4);
    w.e_pos = (int*)take((size_t)emax * 4);
    w.xd = (half*)take((size_t)emax * I * 2);
    w.xr = (half*)take((size_t)emax * 2 * H * 2);
    w.y = (float*)take((size_t)emax * H * 4);
    w.total = o;
    return w;
}
}  // namespace lingfmoe_prefill

using namespace lingfmoe_prefill;

// Bytes of prefill workspace for up to max_tokens rows a call (the same layout as fmoe::prefill_workspace_bytes).
// Its contents on entry never matter (the expert counters are zeroed by the call itself), so one workspace serves
// every layer of a stream.
GB_EXPORT size_t gb_ling_fmoep_workspace_bytes(int max_tokens, int top_k, int NE, int H, int I) {
    if (max_tokens < 1 || top_k < 1 || NE < 1) return 0;
    return p_layout(nullptr, max_tokens, top_k, NE, H, I).total;
}

// Prefill MoE: out [T][H] bf16 from x [T][H] bf16, router logits [T][NE] (bf16 or fp32, `logits_f32`) and the
// shared expert gated by sigmoid(shared_gate [T] bf16). Same tables as gb_fmoe_forward (ktab holds K2, 2..16).
// T <= max_tokens, NE + 1 <= 1024, top_k <= 16, H % 256 == 0, I % 128 == 0. Routing and SwiGLU clamps as
// gb_fmoe_forward. Bit-identical to gb_fmoep_forward + glue::add_rmsnorm with the same arguments (the combine launch
// fuses the scan+scatter and the combine+add_rmsnorm kernels: -2 launches per call). When `norm_w` is null the
// combine launch runs alone (the plain gb_fmoep_forward equivalent, still with the scan+scatter fusion).
GB_EXPORT int gb_ling_fmoep_forward_with_epilogue(
    const void* x, const void* logits, int logits_f32, const void* shared_gate, const uint64_t* gT,
    const uint64_t* gSu, const uint64_t* gSv, const uint64_t* uT, const uint64_t* uSu,
    const uint64_t* uSv, const uint64_t* dT, const uint64_t* dSu, const uint64_t* dSv,
    const int* ktab, void* out, int T, int NE, int top_k, int H, int I, int max_tokens,
    void* ws, void* stream, const float* expert_bias, int n_group, int topk_group,
    float routed_limit, float shared_limit, float routed_scaling,
    const void* hidden, const float* norm_w, void* norm_out, float eps,
    const void* next_suh, void* next_xh) {
    if (T < 1 || T > max_tokens || NE < 1 || NE + 1 > 1024 || top_k < 1 || top_k > 16 || H % 256 || I % 128 || !shared_gate)
        return (int)cudaErrorInvalidValue;
    const bool group = expert_bias != nullptr && n_group >= 2 && topk_group >= 1;
    if (group ? (NE % n_group != 0 || NE / n_group != 64 || topk_group > n_group)
              : (expert_bias != nullptr || n_group != 0 || topk_group != 0))
        return (int)cudaErrorInvalidValue;
    const bool fused_epi = norm_w != nullptr;
    if (fused_epi && (hidden == nullptr || norm_out == nullptr || eps <= 0.0f))
        return (int)cudaErrorInvalidValue;
    if ((next_suh == nullptr) != (next_xh == nullptr) || (next_suh && !fused_epi))
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const int nex = NE + 1, KT = top_k + 1, E = T * KT;
    PWs w = p_layout(static_cast<char*>(ws), max_tokens, top_k, NE, H, I);
    const int mt = p_max_tiles(E, nex);
    const size_t smem_gu_s = (size_t)2 * 2 * PT * PKS * 2, smem_gu_e = (size_t)2 * (PT / 2) * POS * 4;
    const size_t smem_gu = smem_gu_s > smem_gu_e ? smem_gu_s : smem_gu_e;  // double-buffered staging, epilogue half
    const size_t smem_d = (size_t)2 * PT * PKS * 2, smem_d2 = (size_t)(PT / 2) * DOS * 4;  // staging, epilogue half
    const size_t smem_dn = smem_d > smem_d2 ? smem_d : smem_d2;
    static const cudaError_t c1 = cudaFuncSetAttribute(ling_p_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_gu);
    if (c1 != cudaSuccess) return (int)c1;
    static const cudaError_t c2 = cudaFuncSetAttribute(ling_p_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dn);
    if (c2 != cudaSuccess) return (int)c2;

    // The group router fuses the scan+scatter into its last CTA (a self-resetting arrival counter in hdr[1]),
    // and the fused scan resets cnt for every expert, so cnt starts every group call clean: the defensive memset
    // is skipped (the workspace is allocated zeroed; a successful scan re-zeroes cnt). The arrival counter itself
    // is re-zeroed every call (4 bytes) so a failed launch can never wedge it.
    const bool fused_scan = group;
    int rc = 0;
    if (!fused_scan) {
        rc = (int)cudaMemsetAsync(w.cnt, 0, (size_t)nex * sizeof(int), st);
        if (rc) return rc;
    } else {
        rc = (int)cudaMemsetAsync(w.hdr + 1, 0, sizeof(int), st);
        if (rc) return rc;
    }
    if (group)
        ling_p_route_group_kernel<<<(T + 7) / 8, 256, 0, st>>>(logits, logits_f32, (const __nv_bfloat16*)shared_gate,
                                                                expert_bias, routed_scaling, w.ent_exp, w.ent_w, w.cnt, T, NE,
                                                                top_k, n_group, topk_group, w.cur, w.eoff, w.t_exp,
                                                                w.t_p0, w.t_n, w.hdr, w.s_tok, w.s_exp, w.e_pos, E, KT, nex);
    else
        ling_p_route_kernel<<<(T + 7) / 8, 256, 0, st>>>(logits, logits_f32, (const __nv_bfloat16*)shared_gate,
                                                          w.ent_exp, w.ent_w, w.cnt, T, NE, top_k);
    rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    if (!fused_scan) {
        ling_p_scan_and_scatter_kernel<<<1, 1024, 0, st>>>(w.cnt, w.cur, w.eoff, w.t_exp, w.t_p0, w.t_n, w.hdr,
                                                            w.ent_exp, w.s_tok, w.s_exp, w.e_pos, nex, E, KT);
        if ((rc = GB_CHECK_LAUNCH())) return rc;
    }
    ling_p_rotate_kernel<<<(unsigned)E, 256, 0, st>>>((const __nv_bfloat16*)x, w.s_tok, w.s_exp, gSu, uSu, w.xr, E, H);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    PGateUp ga{w.xr, w.hdr, w.t_exp, w.t_p0, w.t_n, gT, gSv, uT, uSv, dSu, ktab, w.xd, H, I, NE, routed_limit, shared_limit};
    ling_p_gate_up_kernel<<<dim3((unsigned)(I / 128), (unsigned)mt), PW * 32, smem_gu, st>>>(ga);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    PDown da{w.xd, w.hdr, w.t_exp, w.t_p0, w.t_n, dT, dSv, ktab, w.y, H, I};
    ling_p_down_kernel<<<dim3((unsigned)(H / 256), (unsigned)mt), PW * 32, smem_dn, st>>>(da);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    if (fused_epi) {
        PCombEP ca{w.y, w.e_pos, w.ent_w, (__nv_bfloat16*)out, (const __nv_bfloat16*)hidden,
                   (__nv_bfloat16*)hidden, norm_w, (__nv_bfloat16*)norm_out, H, KT, eps,
                   (const half*)next_suh, (half*)next_xh};
        ling_p_combine_add_rmsnorm_kernel<<<T, 256, (size_t)H * 2, st>>>(ca);
    } else {
        ling_p_combine_kernel<<<dim3(T, (unsigned)((H + 1023) / 1024)), 256, 0, st>>>(w.y, w.e_pos, w.ent_w, (__nv_bfloat16*)out, H, KT);
    }
    return GB_CHECK_LAUNCH();
}
