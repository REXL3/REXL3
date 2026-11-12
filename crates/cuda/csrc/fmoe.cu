// FrostMoE: grouped multi-token MoE decode for packed EXL3 (mul1 codebook) experts.
// Author: Victor Cruz (vcruz305@gmail.com). Original REXL3 code.
//
// Written from the EXL3 format definition only (a tile is 256 values; value p is the 16-bit window that ends at
// bit (p + 1) * bits of the tile's circular MSB-first bitstream; mul1 maps a 16-bit state s to
// fp16((1024 + bytesum(s * 0x83DCD12D)) * 0x1EEE + 0xC931)). Validated black-box: the tile decoder against
// gb_exl3_reconstruct (bit for bit) and the layer against the CoopMK path and a CPU reference.
//
// Design:
//   * route: one block. Top-k softmax per token (lowest index wins ties), the shared expert as entry k = top_k with
//     weight sigmoid(shared gate), then entries grouped by expert, at most 8 tokens a group (more are split), so one
//     expert's weights are read ONCE per step however many window rows chose it. Fully on device: graph-capturable.
//   * gate_up / down: one warp per (group, matrix, 128-column block, k chunk). The decoded tile is the mma A operand
//     (W^T, 16 n x 16 k) and the group's tokens are the 8 columns of B, so a tile costs one m16n8k16 for up to 8
//     tokens. K chunks write fp32 partials; the last warp of a column block (arrival counter) adds them in chunk order
//     (deterministic), applies the output Hadamard and svh, and in gate_up also SwiGLU and the down layer's input
//     rotation. The last group of a down column block combines all experts per token in top-k order.
#include "common.cuh"
#include "exl3_linear_decode.cuh"
#include <cstdlib>
#include <cstring>

namespace fmoe {

constexpr int GT = 8;             // tokens a group (mma n8)
constexpr int WARPS = 4;          // warps a block
constexpr float RS128 = 0.088388347648318440f;   // 1 / sqrt(128)
constexpr float MUL1_SCALE = 1774.0f / 262144.0f; // fp16 0x1EEE
constexpr float MUL1_BIAS = -10.3828125f;         // fp16 0xC931
constexpr uint32_t MUL1_MUL = 0x83DCD12Du;

// ---------------------------------------------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------------------------------------------
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

__device__ __forceinline__ void st_bf16x4(__nv_bfloat16* p, const float (&v)[4]) {
    __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = u;
}

__device__ __forceinline__ float4 ldcg4(const float* p) { return __ldcg(reinterpret_cast<const float4*>(p)); }

// ---------------------------------------------------------------------------------------------------------------
// Tile decoder (integer bits B = 1..8). A lane decodes the 8 values 8 * lane .. 8 * lane + 7 of the tile; they are
// the W^T A fragment of mma.m16n8k16: reg0 (n = g, k = 2t, 2t+1) = values 0, 1; reg1 (n = g + 8) = values 4, 5;
// reg2 (n = g, k = 2t + 8, 2t + 9) = values 2, 3; reg3 (n = g + 8, k + 8) = values 6, 7  (g = lane / 4, t = lane % 4).
// ---------------------------------------------------------------------------------------------------------------
// Tile geometry for an EXL3 width of K2 = 2 * K bits a value (the trellis' half-integer K2: 2..16). A tile holds 256
// values; K2 / 2 bits a value; 4 * K2 32-bit words. The 16-bit window of value p ends at stream_end(p) (the odd K2
// widths pack two values into three bits, so the windows land on the non-uniform offsets of gbx::stream_end).
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

// ---------------------------------------------------------------------------------------------------------------
// Kernel 1: route
// ---------------------------------------------------------------------------------------------------------------
constexpr int ROUTE_THREADS = 512;
constexpr int ROUTE_EMAX = 64 * 11 + 64;   // smem capacity (64 tokens, top-k <= 11 with the shared expert)

// ---------------------------------------------------------------------------------------------------------------
// Route helpers. A token's logits come in as bf16 (every other family's MoE) or as fp32 (Ling-3.1's
// router_dtype = "fp32"): the launchers take the dtype code (0 bf16, 1 fp32) and the kernels load through
// these, so the routing arithmetic below is byte-for-byte the same in both cases apart from the load itself.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float route_logit(const void* logits, int f32, size_t i) {
    return f32 ? __ldg(static_cast<const float*>(logits) + i)
               : __bfloat162float(__ldg(static_cast<const __nv_bfloat16*>(logits) + i));
}

__global__ void __launch_bounds__(ROUTE_THREADS) route_kernel(const void* __restrict__ logits, int logits_f32,
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

// ---------------------------------------------------------------------------------------------------------------
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

__global__ void __launch_bounds__(ROUTE_THREADS) route_group_kernel(const void* __restrict__ logits, int logits_f32,
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
    // per-warp top-k scratch (the winners every lane of the warp agree on): in shared memory so the dynamic `k`
    // indexing of the winners never touches the local/stack memory (STACK: 0)
    __shared__ float s_suk[ROUTE_THREADS / 32][16];
    __shared__ int s_ik[ROUTE_THREADS / 32][16];
    __shared__ float s_gs[ROUTE_THREADS / 32][8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarp = blockDim.x >> 5;
    const int KT = K + (shared_gate ? 1 : 0);
    const int E = T * KT;
    constexpr int GS = 64;  // experts a group (must equal NE / n_group)
    Route r = route_view(route_base, emax);

    // top-2 of every group across the warp: each lane holds experts lane + 32i, so of group g it holds exactly
    // experts 64g + lane and 64g + lane + 32 (the lane's sc[2g] and sc[2g+1])
    for (int tok = warp; tok < T; tok += nwarp) {
        float* suk = s_suk[warp];
        int* ik = s_ik[warp];
        float* gs = s_gs[warp];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int e0 = 64 * g + lane, e1 = e0 + 32;
            const float lg0 = e0 < NE ? route_logit(logits, logits_f32, (size_t)tok * NE + e0) : -INFINITY;
            const float lg1 = e1 < NE ? route_logit(logits, logits_f32, (size_t)tok * NE + e1) : -INFINITY;
            const float su0 = e0 < NE ? 1.0f / (1.0f + expf(-lg0)) : -INFINITY;
            const float su1 = e1 < NE ? 1.0f / (1.0f + expf(-lg1)) : -INFINITY;
            const float sb0 = e0 < NE ? su0 + expert_bias[e0] : -INFINITY;
            const float sb1 = e1 < NE ? su1 + expert_bias[e1] : -INFINITY;
            float a, b;
            g2(sb0, sb1, a, b);
            g_butterfly(a, b);
            gs[g] = a + b;
        }
        // the topk_group groups by (score DESC, index ASC)
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
        // top-k across the full expert set, only the selected groups' experts
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
            // exclude the picked expert on the lane that owns it (bi is warp-uniform after the reduction)
            if ((bi & 31) == lane && bi < NE) excl |= 1u << (bi >> 5);
            if (bi >= NE) bi = k;  // NaN-safe: keep a valid expert id
            if (lane == 0) { suk[k] = bsu; ik[k] = bi; }
        }
        if (lane == 0) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) sum += suk[k];  // the selected entries' unbiased sigmoid scores
            const float inv = 1.f / (sum + 1e-20f);
            for (int k = 0; k < K; ++k) {
                s_exp[tok * KT + k] = ik[k];
                s_wt[tok * KT + k] = suk[k] * inv * scale;
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
        r.hdr[2] = T;
        r.hdr[3] = KT;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Kernel 2: gate + up (+ SwiGLU, + rotation into the down layer's input space)
// part:  [emax][2][KC][I] fp32,   cnt: [emax][I / 128] int (zero at rest),   xd: [emax][I] fp16 (rotated for down)
// ---------------------------------------------------------------------------------------------------------------
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

__global__ void __launch_bounds__(WARPS * 32, 4) gate_up_kernel(GateUpArgs a) {
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

// ---------------------------------------------------------------------------------------------------------------
// Kernel 3: down (+ per-expert finish, + top-k combine per token)
// partd: [emax][I / 128][H] fp32, y: [emax][H] fp32, cntd: [emax][H / 128], cntc: [H / 128]
// ---------------------------------------------------------------------------------------------------------------
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
};

__global__ void __launch_bounds__(WARPS * 32, 4) down_kernel(DownArgs a) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const Route r = route_view(const_cast<void*>(a.route), a.emax);
    const int NBD = a.H >> 7, KCD = a.I >> 7;
    const int jobs = NBD * KCD;
    const int job = blockIdx.x * WARPS + warp;
    const int grp = job / jobs;
    const int U = r.hdr[0];
    if (grp >= U) return;
    const int q = job - grp * jobs;
    const int kc = q % KCD, nb = q / KCD;

    const int e = r.g_exp[grp], ntok = r.g_ntok[grp], off = r.g_off[grp];
    const int bits = a.ktab[e * 3 + 2];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(a.dT[e]);
    const int g = lane >> 2, t = lane & 3;

    // the group's token column g: 8 k tiles of the rotated activations
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
    const size_t tw = 4 * (size_t)bits;  // K2 words a tile (4 * K2)
    const size_t stride_k = (size_t)(a.H >> 4) * tw;
    const uint32_t* base = T + (size_t)(kc * 8) * stride_k + (size_t)nb * 8 * tw;
    auto getb = [&](int kt, uint32_t& b0, uint32_t& b1) {
        // kt is a loop counter: select without dynamic register indexing
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
    if (!last) return;
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
    if (!lastc) return;
    __threadfence();

    // every group finished column block nb: combine per token in top-k order
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
}

// ---------------------------------------------------------------------------------------------------------------
// Kernel 2b: gate + up (+ SwiGLU, + rotation), in-CTA K reduction (the wave-6b prototype).
//
// One CTA owns every KC K-chunk of one (group, 128-column block) and computes BOTH matrices for each chunk (every
// warp decodes the gate and the up tiles of its kc — the SwiGLU activation needs both reduced values together).
// Warps 1..KC-1 stage their fp32 partials in shared memory (one row per kc, so the reduction keeps the exact
// deterministic ascending kc = 0..KC-1 float32 add order of the global-partial path); warp 0 keeps its own kc = 0
// partial in registers (it is the first term of the ascending chain), reduces the rest from shared, stores the
// merged gate vector in the spare shared row, and finishes the column block as the legacy kernel's last warp did.
// No global `part` traffic, no `__threadfence()` + `atomicAdd` barrier per column block.
//
// smem: KC * GT * (kblocks * 128 + 8) halves of staged x + KC * GT * 128 floats of partials (row KC-1 holds the
// merged gate vector between the two phases). Fits 100 KiB for the supported counts at H <= 2560.
// Grid: `E * NB` (legacy: `E * 2 * NB * KC` warps worth of work, one CTA per (group, nb)).
// ---------------------------------------------------------------------------------------------------------------
template <int NKC>
__global__ void __launch_bounds__(NKC * 32) gate_up_kernel_smem(GateUpArgs a) {
    extern __shared__ __align__(16) half smem_x[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (warp >= NKC) return;
    const Route r = route_view(const_cast<void*>(a.route), a.emax);
    const int NB = a.I >> 7;
    const int grp = blockIdx.x / NB;
    const int nb = blockIdx.x % NB;
    if (grp >= r.hdr[0]) return;
    const int kc = warp;

    const int e = r.g_exp[grp], ntok = r.g_ntok[grp], off = r.g_off[grp];
    const float limit = (e == a.ne) ? a.shared_limit : a.routed_limit;
    const int kblocks = (a.H >> 7) / NKC;
    const int kb0 = kc * kblocks;
    const int xstride = kblocks * 128 + 8;  // +8 halves: conflict-free B fragment reads
    half* xs = smem_x + warp * GT * xstride;
    float* s_part = reinterpret_cast<float*>(smem_x + NKC * GT * xstride);

    const int g = lane >> 2, t = lane & 3;
    // stage the group's tokens rotated by this matrix's suh (same as the legacy kernel's staging loop)
    auto stage = [&](const half* su) {
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
    };
    auto getb = [&](int kt, uint32_t& b0, uint32_t& b1) {
        const half* xrow = xs + g * xstride + 2 * t;
        b0 = *reinterpret_cast<const uint32_t*>(xrow + kt * 16);
        b1 = *reinterpret_cast<const uint32_t*>(xrow + kt * 16 + 8);
    };

    // ---- gate phase: mma, stage partials, reduce in warp 0 ----
    // `acc` is reused by the up phase below (the gate partials are consumed before the up mma overwrites it).
    float acc[8][4];
    {
        const int bits = a.ktab[e * 3 + 0];
        stage(reinterpret_cast<const half*>(a.gSu[e]));
        const uint32_t* T = reinterpret_cast<const uint32_t*>(a.gT[e]);
        const size_t tw = 4 * (size_t)bits;
        const size_t stride_k = (size_t)(a.I >> 4) * tw;
        const uint32_t* base = T + (size_t)(kb0 * 8) * stride_k + (size_t)nb * 8 * tw;
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[j][c] = 0.f;
        run_tiles_bits(bits, base, stride_k, kblocks * 8, lane, acc, getb);
        // every warp (including warp 0) stages its gate partials; row warp = kc
        store_partials(acc, lane, ntok, [&](int c) { return s_part + warp * GT * 128 + c * 128; });
    }
    __syncthreads();
    if (warp == 0) {
        const int col = nb * 128 + 4 * lane;
        for (int c = 0; c < ntok; ++c) {
            // gate sum in the exact legacy order: 0 + p0 + p1 + ... + p_{KC-1}
            float vg[4];
            vg[0] = vg[1] = vg[2] = vg[3] = 0.f;
            for (int k = 0; k < NKC; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(s_part + k * GT * 128 + c * 128 + 4 * lane);
                vg[0] += x.x; vg[1] += x.y; vg[2] += x.z; vg[3] += x.w;
            }
            had128(vg, lane);
            // keep the merged gate vector in the spare shared row until the up phase's activation
            *reinterpret_cast<float4*>(s_part + NKC * GT * 128 + c * 128 + 4 * lane) =
                make_float4(vg[0], vg[1], vg[2], vg[3]);
        }
    }
    __syncthreads();

    // ---- up phase: mma, stage partials, reduce in warp 0, SwiGLU against the stored gate vector ----
    {
        const int bits = a.ktab[e * 3 + 1];
        stage(reinterpret_cast<const half*>(a.uSu[e]));
        const uint32_t* T = reinterpret_cast<const uint32_t*>(a.uT[e]);
        const size_t tw = 4 * (size_t)bits;
        const size_t stride_k = (size_t)(a.I >> 4) * tw;
        const uint32_t* base = T + (size_t)(kb0 * 8) * stride_k + (size_t)nb * 8 * tw;
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[j][c] = 0.f;
        run_tiles_bits(bits, base, stride_k, kblocks * 8, lane, acc, getb);
        // every warp stages its up partials into the same rows (the gate rows are consumed above)
        store_partials(acc, lane, ntok, [&](int c) { return s_part + warp * GT * 128 + c * 128; });
        __syncthreads();
        if (warp == 0) {
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
                float vu[4];
                vu[0] = vu[1] = vu[2] = vu[3] = 0.f;
                for (int k = 0; k < NKC; ++k) {
                    const float4 x = *reinterpret_cast<const float4*>(s_part + k * GT * 128 + c * 128 + 4 * lane);
                    vu[0] += x.x; vu[1] += x.y; vu[2] += x.z; vu[3] += x.w;
                }
                had128(vu, lane);
                const float4 vg4 = *reinterpret_cast<const float4*>(s_part + NKC * GT * 128 + c * 128 + 4 * lane);
                const float vg[4] = {vg4.x, vg4.y, vg4.z, vg4.w};
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
        }
    }
}

// Test: W [K][N] fp16 from a trellis through this file's decoder (layout check against gb_exl3_reconstruct).
template <int K2>
__global__ void unpack_kernel(const uint32_t* __restrict__ T, half* __restrict__ W, int N) {
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x;
    constexpr int NW = GeoK<K2>::NW;
    const uint32_t* tile = T + ((size_t)kt * (N / 16) + nt) * NW;
    int i0, off0;
    lane_origin<K2>(lane, i0, off0);
    uint32_t w[GeoK<K2>::NL];
    tile_load<K2>(tile, i0, w);
    uint32_t a[4];
    tile_decode<K2>(w, off0, a);
    const int g = lane >> 2, t = lane & 3;
    const int k0 = kt * 16 + 2 * t, n0 = nt * 16 + g;
    auto put = [&](int k, int n, uint32_t reg, int hi) {
        const unsigned short h = hi ? (unsigned short)(reg >> 16) : (unsigned short)(reg & 0xffffu);
        W[(size_t)k * N + n] = __ushort_as_half(h);
    };
    put(k0, n0, a[0], 0);         put(k0 + 1, n0, a[0], 1);
    put(k0, n0 + 8, a[1], 0);     put(k0 + 1, n0 + 8, a[1], 1);
    put(k0 + 8, n0, a[2], 0);     put(k0 + 9, n0, a[2], 1);
    put(k0 + 8, n0 + 8, a[3], 0); put(k0 + 9, n0 + 8, a[3], 1);
}

// ---------------------------------------------------------------------------------------------------------------
// Prefill path (many tokens a call). The entries (token, expert) are counting-sorted by expert and cut into tiles of
// up to PT tokens of one expert. A block owns (tile, column block): every weight tile of its columns is decoded ONCE
// and multiplied into all of the tile's tokens (8 mma n-tiles of 8 tokens), over the full K in one warp, so a token's
// output does not depend on which other tokens share its tile. A per-token combine adds the experts in top-k order.
// ---------------------------------------------------------------------------------------------------------------
// PT = 80 (NJ = 10) for Ling-3.1-flash-1019: at 4,096 tokens the mean routed load is exactly 64 tokens per
// expert (4096 * 8 / 512), so PT = 64 tiles the mean and ~47 % of the 512 experts spill into a second tile of 1..21
// tokens (+47 % tiles). PT = 80 = 64 + 2 * sigma removes ~95 % of those splits (P(load > 80) ~ 2.3 % at Poisson(64),
// sigma = 8) while keeping the conflict-free +8 pad and 2 blocks an SM (no swizzle, STACK: 0).
constexpr int PT = 80;           // tokens a tile (REG: 128 STACK: 0)
constexpr int PNJ = PT / 8;      // the n tiles a tile fills (acc[2][PNJ][4] and the zero-init loop MUST match this)
                                 // True A/B on GB10 (bench_ling31_moe_4096, HNJ fix + acc[2][NJ][4] matched to PT):
                                 // PT = 64 (acc[2][8][4]) ~25.2 ms vs PT = 80 (acc[2][10][4]) ~27.9 ms.
constexpr int PW = 8;            // warps a prefill block
constexpr int PXS = 128 + 8;     // gate/up staging row (halves): one 128-wide k block, +8 for conflict-free B reads
constexpr int POS = 128 + 4;     // gate/up output row (floats)
constexpr int PKC = 64;          // k a staged chunk (4 k tiles)
constexpr int PKS = PKC + 8;     // staged row stride (halves): conflict-free B reads; gate/up 46,080 B, 2 blocks 90 KiB

// ---------------------------------------------------------------------------------------------------------------
// The prefill MMA loop. A tile's every output element is ONE fp32 chain of m16n8k16 steps over the k tiles in order from zero, on
// fp16 weight values decoded exactly (tile_decode_h gives the very bits the fp32 decoder of the decode kernels rounds to) and fp16
// token fragments, so what a token gets does not depend on the other tokens of its tile, on its column in the tile, or on how the
// prompt was cut into calls (the committed golden pins the bits; the invariance tests of tests/minicpm_moe_prefill.rs run the
// chunk grids). The loop's instruction stream is lean on purpose (E6a):
//   * the n tiles are a compile-time count NJ (2, 4, 8 or 10, the tile's `ntn` rounded up; the extra tiles read zero rows, which add
//     exactly 0), so the loop has no per-n-tile branch and no per-n-tile address arithmetic;
//   * the token fragments of two n tiles come from one ldmatrix.x4 (4 shared loads instead of 16 at NJ = 8);
//   * the weight words of the NEXT k tile are fetched across the chunk boundary (the caller carries `cur`), so a chunk's first tile is
//     never a cold load;
//   * the values of a tile are produced by one integer byte-sum and one packed half2 FMA per pair instead of an int-to-float
//     conversion and an fp32 FMA per value: (1024 + bytesum) is the exact half 0x6400 | bytesum, and hfma2(h, 0x1EEE, 0xC931) rounds
//     the exact value once, as the fp32 FMA followed by the round to half does (the product has 22 bits, the sum is exact in fp32).
// ---------------------------------------------------------------------------------------------------------------
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

// acc[m][j][.] += W^T tiles (nt0 + m) x tokens (n tile j < NJ), k tiles [0, nkt) of the chunk at `t0`. `xs_lane` is this lane's
// ldmatrix row address of token 0, k tile 0 of the staged chunk; `more` says the chunk after this one exists (its first tile is
// then prefetched here). `cur` enters holding this chunk's first tile pair and leaves holding the next chunk's.
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

// the n tiles a tile of `ntn` fills, as the loop's compile-time width
__device__ __forceinline__ bool p_mm_run(int ntn, int k2, const uint32_t* t0, size_t stride_k, int nkt, bool more,
                                         uint32_t xs_lane, int lane, uint32_t (&cur)[2][3], float (&acc)[2][10][4]) {
    if (ntn <= 2) return p_mm_bits2<2>(k2, t0, stride_k, nkt, more, xs_lane, lane, cur, acc);
    if (ntn <= 4) return p_mm_bits2<4>(k2, t0, stride_k, nkt, more, xs_lane, lane, cur, acc);
    if (ntn <= 8) return p_mm_bits2<8>(k2, t0, stride_k, nkt, more, xs_lane, lane, cur, acc);
    return p_mm_bits2<10>(k2, t0, stride_k, nkt, more, xs_lane, lane, cur, acc);
}

// Route, one warp a token: top-k softmax (lowest index wins ties), the shared expert as entry K with sigmoid(gate).
__global__ void __launch_bounds__(256) p_route_kernel(const void* __restrict__ logits, int logits_f32,
                                                      const __nv_bfloat16* __restrict__ shared_gate,
                                                      int* __restrict__ ent_exp, float* __restrict__ ent_w,
                                                      int* __restrict__ cnt, int T, int NE, int K) {
    // per-warp top-k scratch (the winners every lane of the warp agree on): in shared memory so the dynamic `k`
    // indexing of the winners never touches the local/stack memory (STACK: 0)
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

// Route, one warp a token: the group-limited noaux_tc route (see route_group_kernel), the shared expert as entry K.
__global__ void __launch_bounds__(256) p_route_group_kernel(const void* __restrict__ logits, int logits_f32,
                                                            const __nv_bfloat16* __restrict__ shared_gate,
                                                            const float* __restrict__ expert_bias, float scale,
                                                            int* __restrict__ ent_exp, float* __restrict__ ent_w,
                                                            int* __restrict__ cnt, int T, int NE, int K, int n_group,
                                                            int topk_group) {
    // per-warp top-k scratch (the winners every lane of the warp agree on): in shared memory so the dynamic `k`
    // indexing of the winners never touches the local/stack memory (STACK: 0)
    __shared__ float s_suk[8][16];
    __shared__ int s_ik[8][16];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tok = blockIdx.x * 8 + warp;
    if (tok >= T) return;
    const int KT = K + (shared_gate ? 1 : 0);
    // each lane's 16 experts: sigmoid score su[i] and biased score sb[i], loaded ONCE per token as read-only
    // registers (the `k` loop indexes them by compile-time constants only and never writes them, so ptxas
    // keeps them in registers: STACK 0, LOCAL 0)
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
        // exclude the picked expert on the lane that owns it (bi is warp-uniform after the reduction)
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
}

// One block of 1024: per expert the first sorted slot and its tiles; the tile table; counters back to zero.
__global__ void __launch_bounds__(1024) p_scan_kernel(int* __restrict__ cnt, int* __restrict__ cur,
                                                      int* __restrict__ eoff, int* __restrict__ t_exp,
                                                      int* __restrict__ t_p0, int* __restrict__ t_n,
                                                      int* __restrict__ hdr, int NEX) {
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
}

__global__ void p_scatter_kernel(const int* __restrict__ ent_exp, const int* __restrict__ eoff, int* __restrict__ cur,
                                 int* __restrict__ s_tok, int* __restrict__ s_exp, int* __restrict__ e_pos, int E, int KT) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const int e = ent_exp[i];
    const int p = eoff[e] + atomicAdd(cur + e, 1);
    s_tok[p] = i / KT;
    s_exp[p] = e;
    e_pos[i] = p;
}

// Every sorted slot's token rotated ONCE per matrix: xr[p][m][k] = fp16(H(x[tok] * suh_m[e]) / sqrt(128))[k], so the
// gate/up blocks (five column blocks a tile) only copy their rows. One block per (slot, matrix): its 8 warps split the
// 32 independent 128-wide k blocks (warp w takes kb = w, w+8, w+16, w+24), so all warps stream the SAME token row and
// expert scale rows (an 8x smaller L1 working set). Each kb is an independent 128-wide Sylvester Hadamard run wholly by
// one warp, so the output is bit-for-bit identical to one warp per entry.
__global__ void __launch_bounds__(256) p_rotate_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ s_tok,
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
__global__ void __launch_bounds__(PW * 32, 2) p_gate_up_kernel(PGateUp a) {
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

// Block = (tile, 256-column block), 8 warps of 32 columns. The tile's activation rows are staged one 128-wide k block
// at a time and the epilogue runs in two halves of PT / 2 = 40 tokens; the launcher sizes shared memory to the larger of
// double-buffered staging (23,040 B) and the epilogue (41,600 B), so two blocks (83.2 KiB) fit an SM.
constexpr int DOS = 256 + 4;  // down output row (floats)

__global__ void __launch_bounds__(PW * 32, 2) p_down_kernel(PDown a) {
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

// out[tok] = sum over its entries in top-k order of weight * y[slot of the entry]
__global__ void __launch_bounds__(256) p_combine_kernel(const float* __restrict__ y, const int* __restrict__ e_pos,
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

}  // namespace fmoe

using namespace fmoe;

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
GB_EXPORT size_t gb_fmoe_workspace_bytes(int max_tokens, int top_k, int H, int I, int KC) {
    const int emax = max_tokens * (top_k + 1);
    return fmoe_layout(nullptr, emax, H, I, KC).total;
}

// out [T][H] bf16 = MoE(x [T][H] bf16) with routing from logits [T][NE] (bf16 when logits_f32 == 0, fp32
// otherwise — Ling-3.1's router_dtype = fp32) and the shared expert gated by
// sigmoid(shared_gate [T] bf16). Pointer tables hold NE + 1 entries (the shared expert last); ktab [NE + 1][3] holds
// the EXL3 K2 (twice the bits, 2..16) of gate, up, down. T <= max_tokens of the workspace, T <= 64, NE <= 512,
// top_k <= 16, H % 128 == 0, I % 128 == 0, (H/128) % KC == 0.
// Routing: expert_bias == nullptr (or n_group < 2) selects the legacy top-k softmax (no bias, weight = softmax of the
// logits, no scale); otherwise the group-limited noaux_tc router runs (sigmoid scores + expert_bias, top-2 per group,
// topk_group groups of NE / n_group = 64 experts, weights = normalized unbiased sigmoid scores * scale).
// SwiGLU clamps: routed_limit applies to the routed experts, shared_limit to the shared expert (entry NE); <= 0: none.
// Test/debug hook: when 1, gb_fmoe_forward uses the legacy global-partial kernels (A/B bit-exactness tests).
static int g_fmoe_legacy = 0;
GB_EXPORT int gb_fmoe_forward(const void* x, const void* logits, int logits_f32, const void* shared_gate, const uint64_t* gT,
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
        route_group_kernel<<<1, ROUTE_THREADS, 0, st>>>(logits, logits_f32, (const __nv_bfloat16*)shared_gate,
                                                        expert_bias, routed_scaling, w.route, emax, T, NE, top_k, n_group,
                                                        topk_group);
    else
        route_kernel<<<1, ROUTE_THREADS, 0, st>>>(logits, logits_f32, (const __nv_bfloat16*)shared_gate,
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

    // Wave-6b in-CTA reductions (one CTA owns every K chunk of a (group, block); partials cross warps in shared
    // memory, keeping the legacy ascending kc sum order exactly). The gate/up fused path is the default (it is
    // bit-exact with the legacy path); gb_fmoe_set_legacy forces the global-partial kernels (A/B tests). The down
    // path stays the legacy down_kernel (bit-exact, the wave-6 67.07 tok/s winner).
    const bool use_gu_smem = !g_fmoe_legacy;
    if (use_gu_smem && (H / 128) % KC == 0) {
        const int kblocks = (H / 128) / KC;
        const size_t smem_gu = (size_t)KC * GT * (kblocks * 128 + 8) * sizeof(half) + (size_t)(KC + 1) * GT * 128 * sizeof(float);
        if (smem_gu <= 100 * 1024) {
            const int grid_gu = E * (I / 128);
            switch (KC) {
                case 2:
                    if (smem_gu > 48 * 1024) cudaFuncSetAttribute(gate_up_kernel_smem<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_gu);
                    gate_up_kernel_smem<2><<<grid_gu, 64, smem_gu, st>>>(ga);
                    break;
                case 4:
                    if (smem_gu > 48 * 1024) cudaFuncSetAttribute(gate_up_kernel_smem<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_gu);
                    gate_up_kernel_smem<4><<<grid_gu, 128, smem_gu, st>>>(ga);
                    break;
                case 5:
                    if (smem_gu > 48 * 1024) cudaFuncSetAttribute(gate_up_kernel_smem<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_gu);
                    gate_up_kernel_smem<5><<<grid_gu, 160, smem_gu, st>>>(ga);
                    break;
                case 8:
                    if (smem_gu > 48 * 1024) cudaFuncSetAttribute(gate_up_kernel_smem<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_gu);
                    gate_up_kernel_smem<8><<<grid_gu, 256, smem_gu, st>>>(ga);
                    break;
                case 10:
                    if (smem_gu > 48 * 1024) cudaFuncSetAttribute(gate_up_kernel_smem<10>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_gu);
                    gate_up_kernel_smem<10><<<grid_gu, 320, smem_gu, st>>>(ga);
                    break;
                case 16:
                    if (smem_gu > 48 * 1024) cudaFuncSetAttribute(gate_up_kernel_smem<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_gu);
                    gate_up_kernel_smem<16><<<grid_gu, 512, smem_gu, st>>>(ga);
                    break;
                case 20:
                    if (smem_gu > 48 * 1024) cudaFuncSetAttribute(gate_up_kernel_smem<20>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_gu);
                    gate_up_kernel_smem<20><<<grid_gu, 640, smem_gu, st>>>(ga);
                    break;
                default:
                    gate_up_kernel<<<(jobs_a + WARPS - 1) / WARPS, WARPS * 32, smem_a, st>>>(ga);
                    break;
            }
            rc = GB_CHECK_LAUNCH();
            if (rc) return rc;
            goto down_launch;
        }
    }
    gate_up_kernel<<<(jobs_a + WARPS - 1) / WARPS, WARPS * 32, smem_a, st>>>(ga);
    rc = GB_CHECK_LAUNCH();
    if (rc) return rc;

down_launch:
    DownArgs da{(const half*)w.xd, w.route, emax, dT, dSv, ktab, (float*)w.partd, (float*)w.y, (int*)w.cntd,
                (int*)w.cntc, (__nv_bfloat16*)out, H, I};
    const int jobs_b = E * (H / 128) * (I / 128);
    down_kernel<<<(jobs_b + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(da);
    return GB_CHECK_LAUNCH();
}

// Test/debug hook: force the legacy global-partial kernels (0 or 1); returns the previous setting.
GB_EXPORT int gb_fmoe_set_legacy(int legacy) {
    const int prev = g_fmoe_legacy;
    g_fmoe_legacy = legacy != 0;
    return prev;
}

// The group-limited noaux_tc router alone (test/debug hook): sel [T][K+1] i32 and wts [T][K+1] f32 with the shared
// expert as entry K, as gb_fmoe_forward routes. ws holds T*(K+1)*8 + (NE+1)*4 bytes (contents on entry never matter).
GB_EXPORT int gb_fmoe_group_route(const void* logits, int logits_f32, const void* shared_gate, const float* expert_bias, float scale,
                                  int* sel, float* wts, void* ws, int T, int NE, int K, int n_group, int topk_group,
                                  void* stream) {
    if (T < 1 || NE < 1 || NE > 512 || K < 1 || K > 16 || T * (K + 1) > ROUTE_EMAX || !expert_bias || n_group < 2 ||
        topk_group < 1 || NE % n_group != 0 || NE / n_group != 64 || topk_group > n_group)
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    int* ent_exp = static_cast<int*>(ws);
    float* ent_w = reinterpret_cast<float*>(static_cast<char*>(ws) + (size_t)T * (K + 1) * 4);
    int* cnt = reinterpret_cast<int*>(static_cast<char*>(ws) + (size_t)T * (K + 1) * 8);
    int rc = (int)cudaMemsetAsync(cnt, 0, (size_t)(NE + 1) * sizeof(int), st);
    if (rc) return rc;
    p_route_group_kernel<<<(T + 7) / 8, 256, 0, st>>>(logits, logits_f32, (const __nv_bfloat16*)shared_gate,
                                                      expert_bias, scale, ent_exp, ent_w, cnt, T, NE, K, n_group, topk_group);
    rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    cudaMemcpyAsync(sel, ent_exp, (size_t)T * (K + 1) * 4, cudaMemcpyDeviceToDevice, st);
    rc = (int)cudaMemcpyAsync(wts, ent_w, (size_t)T * (K + 1) * 4, cudaMemcpyDeviceToDevice, st);
    return rc;
}

GB_EXPORT int gb_fmoe_unpack(const void* trellis, void* w, int k, int n, int k2, void* stream) {
    if (k % 16 || n % 16 || k2 < 2 || k2 > 16 || k2 == 9 || k2 == 11 || k2 == 13 || k2 == 15) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(n / 16), (unsigned)(k / 16));
    cudaStream_t st = gb_stream(stream);
    const uint32_t* T = static_cast<const uint32_t*>(trellis);
    half* W = static_cast<half*>(w);
    switch (k2) {
        case 2: unpack_kernel<2><<<grid, 32, 0, st>>>(T, W, n); break;
        case 3: unpack_kernel<3><<<grid, 32, 0, st>>>(T, W, n); break;
        case 4: unpack_kernel<4><<<grid, 32, 0, st>>>(T, W, n); break;
        case 5: unpack_kernel<5><<<grid, 32, 0, st>>>(T, W, n); break;
        case 6: unpack_kernel<6><<<grid, 32, 0, st>>>(T, W, n); break;
        case 7: unpack_kernel<7><<<grid, 32, 0, st>>>(T, W, n); break;
        case 8: unpack_kernel<8><<<grid, 32, 0, st>>>(T, W, n); break;
        case 10: unpack_kernel<10><<<grid, 32, 0, st>>>(T, W, n); break;
        case 12: unpack_kernel<12><<<grid, 32, 0, st>>>(T, W, n); break;
        case 14: unpack_kernel<14><<<grid, 32, 0, st>>>(T, W, n); break;
        case 16: unpack_kernel<16><<<grid, 32, 0, st>>>(T, W, n); break;
    }
    return GB_CHECK_LAUNCH();
}

// ------------------------------------------------------------------------------------------------ prefill launcher
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

static PWs p_layout(char* base, int max_tokens, int top_k, int NE, int H, int I) {
    const int nex = NE + 1, emax = max_tokens * (top_k + 1), mt = p_max_tiles(emax, nex);
    PWs w;
    size_t o = 0;
    auto take = [&](size_t bytes) { char* p = base ? base + o : nullptr; o += al256(bytes); return p; };
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

// Bytes of prefill workspace for up to max_tokens rows a call. Its contents on entry never matter (the expert counters are zeroed
// by the call itself), so one workspace serves every layer of a stream.
GB_EXPORT size_t gb_fmoep_workspace_bytes(int max_tokens, int top_k, int NE, int H, int I) {
    if (max_tokens < 1 || top_k < 1 || NE < 1) return 0;
    return p_layout(nullptr, max_tokens, top_k, NE, H, I).total;
}

// Prefill MoE: out [T][H] bf16 from x [T][H] bf16, router logits [T][NE] (bf16 or fp32, `logits_f32`) and the
// shared expert gated by
// sigmoid(shared_gate [T] bf16). Same tables as gb_fmoe_forward (ktab holds K2, 2..16). T <= max_tokens,
// NE + 1 <= 1024, top_k <= 16, H % 256 == 0, I % 128 == 0. Routing and SwiGLU clamps as gb_fmoe_forward.
GB_EXPORT int gb_fmoep_forward(const void* x, const void* logits, int logits_f32, const void* shared_gate, const uint64_t* gT,
                               const uint64_t* gSu, const uint64_t* gSv, const uint64_t* uT, const uint64_t* uSu,
                               const uint64_t* uSv, const uint64_t* dT, const uint64_t* dSu, const uint64_t* dSv,
                               const int* ktab, void* out, int T, int NE, int top_k, int H, int I, int max_tokens,
                               void* ws, void* stream, const float* expert_bias, int n_group, int topk_group,
                               float routed_limit, float shared_limit, float routed_scaling) {
    if (T < 1 || T > max_tokens || NE < 1 || NE + 1 > 1024 || top_k < 1 || top_k > 16 || H % 256 || I % 128 || !shared_gate)
        return (int)cudaErrorInvalidValue;
    const bool group = expert_bias != nullptr && n_group >= 2 && topk_group >= 1;
    if (group ? (NE % n_group != 0 || NE / n_group != 64 || topk_group > n_group)
              : (expert_bias != nullptr || n_group != 0 || topk_group != 0))
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const int nex = NE + 1, KT = top_k + 1, E = T * KT;
    PWs w = p_layout(static_cast<char*>(ws), max_tokens, top_k, NE, H, I);
    const int mt = p_max_tiles(E, nex);
    const size_t smem_gu_s = (size_t)2 * 2 * PT * PKS * 2, smem_gu_e = (size_t)2 * (PT / 2) * POS * 4;
    const size_t smem_gu = smem_gu_s > smem_gu_e ? smem_gu_s : smem_gu_e;  // double-buffered staging, epilogue half
    const size_t smem_d = (size_t)2 * PT * PKS * 2, smem_d2 = (size_t)(PT / 2) * DOS * 4;  // staging, epilogue half
    const size_t smem_dn = smem_d > smem_d2 ? smem_d : smem_d2;
    static const cudaError_t c1 = cudaFuncSetAttribute(p_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_gu);
    if (c1 != cudaSuccess) return (int)c1;
    static const cudaError_t c2 = cudaFuncSetAttribute(p_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dn);
    if (c2 != cudaSuccess) return (int)c2;

    // the expert counters start every call at zero: the scan zeroes them again, but a call that failed between the route and the scan
    // (or a workspace handed over dirty) must not leak into the next one
    int rc = (int)cudaMemsetAsync(w.cnt, 0, (size_t)nex * sizeof(int), st);
    if (rc) return rc;
    if (group)
        p_route_group_kernel<<<(T + 7) / 8, 256, 0, st>>>(logits, logits_f32, (const __nv_bfloat16*)shared_gate,
                                                          expert_bias, routed_scaling, w.ent_exp, w.ent_w, w.cnt, T, NE,
                                                          top_k, n_group, topk_group);
    else
        p_route_kernel<<<(T + 7) / 8, 256, 0, st>>>(logits, logits_f32, (const __nv_bfloat16*)shared_gate,
                                                    w.ent_exp, w.ent_w, w.cnt, T, NE, top_k);
    rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    p_scan_kernel<<<1, 1024, 0, st>>>(w.cnt, w.cur, w.eoff, w.t_exp, w.t_p0, w.t_n, w.hdr, nex);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    p_scatter_kernel<<<(E + 255) / 256, 256, 0, st>>>(w.ent_exp, w.eoff, w.cur, w.s_tok, w.s_exp, w.e_pos, E, KT);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    p_rotate_kernel<<<(unsigned)E, 256, 0, st>>>((const __nv_bfloat16*)x, w.s_tok, w.s_exp, gSu, uSu, w.xr, E, H);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    PGateUp ga{w.xr, w.hdr, w.t_exp, w.t_p0, w.t_n, gT, gSv, uT, uSv, dSu, ktab, w.xd, H, I, NE, routed_limit, shared_limit};
    p_gate_up_kernel<<<dim3((unsigned)(I / 128), (unsigned)mt), PW * 32, smem_gu, st>>>(ga);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    PDown da{w.xd, w.hdr, w.t_exp, w.t_p0, w.t_n, dT, dSv, ktab, w.y, H, I};
    p_down_kernel<<<dim3((unsigned)(H / 256), (unsigned)mt), PW * 32, smem_dn, st>>>(da);
    if ((rc = GB_CHECK_LAUNCH())) return rc;
    p_combine_kernel<<<dim3(T, (unsigned)((H + 1023) / 1024)), 256, 0, st>>>(w.y, w.e_pos, w.ent_w, (__nv_bfloat16*)out, H, KT);
    return GB_CHECK_LAUNCH();
}
