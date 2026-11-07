// gb-nvcc: --fmad=false
// Causal GQA attention of the Nemotron 3.5 Super VL NoPE layers: 32 query heads of 128 over 2 kv heads, no RoPE, scale 128^-0.5,
// a fp16 KV cache in 256-token pages behind a block table. The reduction tree is the one of tools/attn_gpu_design.md (decisions 1-5)
// and of crates/model/src/families/nemotron_h_omni/attn_ref.rs (the CPU oracle, which these kernels are checked against).
//
//   gb_nemo_kv_write     round the fp32 K and V projections to fp16 once and scatter them into their pages.
//   gb_nemo_kv_write_i8  (`--kv-dtype int8`) round them once to int8 at a power-of-two scale per (token, kv head) into int8 pages
//                        (nemotron_kv8.cuh); gb_nemo_attn_partial_i8 reads those pages, dequantising exactly into the same fp16 tiles.
//   gb_nemo_attn_partial one CTA per (row, kv head, absolute 512-key chunk) with 16 warps, one per query head of the group. A warp
//                        runs the chunk's 64-key tiles in ascending order: fp32 scores, fp32 running max and denominator, P rounded
//                        to fp16 once per tile for the weighted sum of values. Writes (m, l, o[128]) per (row, chunk, query head).
//   gb_nemo_attn_merge   one CTA per (row, query head). The chunk states merge in the fixed balanced tree over the chunk index
//                        (a binary-counter stack, folded from the right, which is the same tree with the identity padding skipped),
//                        and the output is o / l.
//
// Both kernels take `limit`: a row whose position is at or above it belongs to another arithmetic (the flash path of
// nemotron_attn_prefill.cu serves rows at position >= 8192) and is skipped whole; no other row's computation depends on it, so every
// row below the limit has the bits it has without one. A launch with limit INT_MAX is the launch of before.
//
// Bits: a (row, query head, chunk) partial is computed by one warp with a fixed operation order (lane k owns keys k and k+32 for the
// scores, and dimensions 4k..4k+3 for the value sum), so it does not depend on the CTA, the grid, the other rows or their number.
// Decode and prefill launch these same kernels and differ only in the number of rows. Compiled with --fmad=false, so no multiply-add
// is contracted differently in two kernels. Masked keys are never read: a tile loads only keys below the row's end and zero-fills the
// rest of its smem rows, and a masked key contributes exact zeros.
#include "common.cuh"
#include "nemotron_kv8.cuh"

namespace {

constexpr int NQ = 32, NKV = 2, GRP = NQ / NKV, D = 128, PAGE = 256, CHUNK = 512, TILE = 64;
constexpr int THREADS = 32 * GRP;               // one warp per query head of a kv group
constexpr int KV_ROW = NKV * D;                 // halves per token, all kv heads
constexpr int KS_STRIDE = D / 2 + 1;            // 32-bit words per smem key row: 65 keeps the per-lane key reads conflict-free
constexpr int MAX_STACK = 16;                   // binary-counter depth: log2(chunks) + 1 entries
constexpr float SCALE = 0.08838834764831845f;   // 128^-0.5

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Every lane ends with the same value: a pair's sum is commutative, so each butterfly step leaves both partners equal.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// K and V of one token row (all kv heads): the cache is [block][PAGE tokens][NKV][D] fp16.
__global__ void kv_write_kernel(const float* __restrict__ k_in, const float* __restrict__ v_in, const int* __restrict__ slot,
                                const int* __restrict__ pos, const int* __restrict__ table, int max_blocks, __half* __restrict__ kc,
                                __half* __restrict__ vc) {
    const int r = blockIdx.x, e = threadIdx.x;  // e in [0, KV_ROW)
    const int p = pos[r];
    const int blk = table[slot[r] * max_blocks + p / PAGE];
    const size_t dst = ((size_t)blk * PAGE + p % PAGE) * KV_ROW + e;
    kc[dst] = __float2half_rn(k_in[(size_t)r * KV_ROW + e]);
    vc[dst] = __float2half_rn(v_in[(size_t)r * KV_ROW + e]);
}

// The int8 cache write (`--kv-dtype int8`, nemotron_kv8.cuh): one CTA a row, one thread a value. The exponent of each (row, kv head) of
// K and of V comes from the head's absolute maximum (4 warps a head), then every value is rounded once to int8 at that scale.
__global__ void kv_write8_kernel(const float* __restrict__ k_in, const float* __restrict__ v_in, const int* __restrict__ slot,
                                 const int* __restrict__ pos, const int* __restrict__ table, int max_blocks,
                                 uint8_t* __restrict__ kc, uint8_t* __restrict__ vc) {
    __shared__ float red[2][KV_ROW / 32];
    const int r = blockIdx.x, e = threadIdx.x, warp = e >> 5, lane = e & 31, g = e / D;
    const float kx = k_in[(size_t)r * KV_ROW + e], vx = v_in[(size_t)r * KV_ROW + e];
    const float ka = warp_max(fabsf(kx)), va = warp_max(fabsf(vx));
    if (lane == 0) {
        red[0][warp] = ka;
        red[1][warp] = va;
    }
    __syncthreads();
    const int w0 = g * (D / 32);
    const float km = fmaxf(fmaxf(red[0][w0], red[0][w0 + 1]), fmaxf(red[0][w0 + 2], red[0][w0 + 3]));
    const float vm = fmaxf(fmaxf(red[1][w0], red[1][w0 + 1]), fmaxf(red[1][w0 + 2], red[1][w0 + 3]));
    const int ke = kv8::exponent_i8(km), ve = kv8::exponent_i8(vm);
    const int p = pos[r];
    const size_t page = (size_t)table[slot[r] * max_blocks + p / PAGE] * kv8::PAGE_BYTES;
    const int t = p % PAGE;
    kc[page + (size_t)t * KV_ROW + e] = kv8::quant_i8(kx, ke);
    vc[page + (size_t)t * KV_ROW + e] = kv8::quant_i8(vx, ve);
    if ((e % D) == 0) {
        kc[page + kv8::EXP_OFF + t * NKV + g] = (uint8_t)(int8_t)ke;
        vc[page + kv8::EXP_OFF + t * NKV + g] = (uint8_t)(int8_t)ve;
    }
}

// Q8: the cache is the int8 page of nemotron_kv8.cuh; each value is dequantised (exactly) into the same fp16 shared tiles, so the rest of
// the kernel is the fp16 arithmetic and a row's bits equal the fp16 kernel's on a cache holding the dequantised values.
template <bool Q8>
__global__ void __launch_bounds__(THREADS)
partial_kernel(const float* __restrict__ q, const void* __restrict__ kc_, const void* __restrict__ vc_,
               const int* __restrict__ table, int max_blocks, const int* __restrict__ slot, const int* __restrict__ pos,
               int limit, int stride, float2* __restrict__ part_ml, float* __restrict__ part_o) {
    __shared__ __align__(16) float qs[GRP][D];
    __shared__ __align__(16) uint32_t ks[TILE * KS_STRIDE];
    __shared__ __align__(16) __half vs[TILE * D];
    __shared__ float ps[GRP][TILE];

    const int r = blockIdx.x, g = blockIdx.y, c = blockIdx.z;
    if (pos[r] >= limit) return;  // CTA-uniform: the row is another arithmetic's
    const int n = pos[r] + 1;
    const int c_start = c * CHUNK;
    if (c_start >= n) return;  // CTA-uniform: the chunk lies past this row's last key
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = g * GRP + warp;
    const int* tab = table + slot[r] * max_blocks;
    const __half* __restrict__ kc = (const __half*)kc_;
    const __half* __restrict__ vc = (const __half*)vc_;
    const uint8_t* __restrict__ kc8 = (const uint8_t*)kc_;
    const uint8_t* __restrict__ vc8 = (const uint8_t*)vc_;

    reinterpret_cast<float4*>(&qs[warp][0])[lane] = reinterpret_cast<const float4*>(q + ((size_t)r * NQ + h) * D)[lane];

    float m = neg_inf(), l = 0.f;
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    for (int u = 0; u < CHUNK / TILE; ++u) {
        const int ts = c_start + u * TILE;
        if (ts >= n) break;  // CTA-uniform: a tile wholly past the row's end is an exact no-op, so it is not visited
        const int w = min(TILE, n - ts);
        __syncthreads();  // the last tile's readers are done, and qs is complete before the first read

        for (int idx = threadIdx.x; idx < TILE * (D / 2); idx += THREADS) {
            const int t = idx >> 6, wd = idx & 63;
            uint32_t val = 0u;
            if (t < w) {
                const int key = ts + t;
                if constexpr (Q8) {
                    const uint8_t* pg = kc8 + (size_t)tab[key / PAGE] * kv8::PAGE_BYTES;
                    const int kt = key % PAGE;
                    const uint32_t two = *reinterpret_cast<const uint16_t*>(pg + kt * KV_ROW + g * D + 2 * wd);
                    val = kv8::dq2_i8(two, kv8::pow2((int)(int8_t)pg[kv8::EXP_OFF + kt * NKV + g]));
                } else {
                    const size_t tok = (size_t)tab[key / PAGE] * PAGE + key % PAGE;
                    val = *reinterpret_cast<const uint32_t*>(kc + tok * KV_ROW + g * D + 2 * wd);
                }
            }
            ks[t * KS_STRIDE + wd] = val;
        }
        for (int idx = threadIdx.x; idx < TILE * (D / 8); idx += THREADS) {
            const int t = idx >> 4, pc = idx & 15;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (t < w) {
                const int key = ts + t;
                if constexpr (Q8) {
                    const uint8_t* pg = vc8 + (size_t)tab[key / PAGE] * kv8::PAGE_BYTES;
                    const int kt = key % PAGE;
                    const uint2 raw = *reinterpret_cast<const uint2*>(pg + kt * KV_ROW + g * D + 8 * pc);
                    val = kv8::dq8_i8(raw, (int)(int8_t)pg[kv8::EXP_OFF + kt * NKV + g]);
                } else {
                    const size_t tok = (size_t)tab[key / PAGE] * PAGE + key % PAGE;
                    val = *reinterpret_cast<const uint4*>(vc + tok * KV_ROW + g * D + 8 * pc);
                }
            }
            *reinterpret_cast<uint4*>(&vs[t * D + 8 * pc]) = val;
        }
        __syncthreads();

        // scores: lane k owns keys k and k + 32, each a sequential sum over the 128 dimensions
        const bool v0 = lane < w, v1 = lane + 32 < w;
        float a0 = 0.f, a1 = 0.f;
        {
            const uint32_t* k0 = &ks[lane * KS_STRIDE];
            const uint32_t* k1 = &ks[(lane + 32) * KS_STRIDE];
#pragma unroll 8
            for (int wd = 0; wd < D / 2; ++wd) {
                const float2 qq = *reinterpret_cast<const float2*>(&qs[warp][2 * wd]);
                const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(&k0[wd]));
                const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&k1[wd]));
                a0 += qq.x * x0.x;
                a0 += qq.y * x0.y;
                a1 += qq.x * x1.x;
                a1 += qq.y * x1.y;
            }
        }
        const float s0 = v0 ? a0 * SCALE : neg_inf();
        const float s1 = v1 ? a1 * SCALE : neg_inf();

        const float tile_max = warp_max(fmaxf(s0, s1));
        const float m_new = fmaxf(m, tile_max);
        const float alpha = (m == neg_inf()) ? 0.f : expf(m - m_new);
        const float p0 = v0 ? expf(s0 - m_new) : 0.f;  // the denominator takes these in fp32
        const float p1 = v1 ? expf(s1 - m_new) : 0.f;
        l = l * alpha + warp_sum(p0 + p1);
        ps[warp][lane] = __half2float(__float2half_rn(p0));  // the weighted sum takes them rounded to fp16, once per tile
        ps[warp][lane + 32] = __half2float(__float2half_rn(p1));
        __syncwarp();

        float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
        for (int t = 0; t < w; ++t) {
            const float p = ps[warp][t];
            const uint2 vv = *reinterpret_cast<const uint2*>(&vs[t * D + lane * 4]);
            const float2 x01 = __half22float2(*reinterpret_cast<const __half2*>(&vv.x));
            const float2 x23 = __half22float2(*reinterpret_cast<const __half2*>(&vv.y));
            b0 += p * x01.x;
            b1 += p * x01.y;
            b2 += p * x23.x;
            b3 += p * x23.y;
        }
        o0 = o0 * alpha + b0;
        o1 = o1 * alpha + b1;
        o2 = o2 * alpha + b2;
        o3 = o3 * alpha + b3;
        m = m_new;
        __syncwarp();  // ps is rewritten next tile, after the block barrier; this keeps the warp's reads ordered
    }

    const size_t base = ((size_t)r * stride + c) * NQ + h;
    if (lane == 0) part_ml[base] = make_float2(m, l);
    reinterpret_cast<float4*>(part_o + base * D)[lane] = make_float4(o0, o1, o2, o3);
}

// grid (row, query head), one thread per dimension. The stack holds (m, l, o) of the tree's pending left subtrees, with their levels.
__global__ void __launch_bounds__(D)
merge_kernel(const int* __restrict__ pos, const float2* __restrict__ part_ml, const float* __restrict__ part_o, int max_chunks,
             int limit, float* __restrict__ out) {
    __shared__ float sm[MAX_STACK], sl[MAX_STACK];
    __shared__ int slev[MAX_STACK];
    __shared__ float so[MAX_STACK][D];
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (pos[r] >= limit) return;  // CTA-uniform, before any barrier: the row is another arithmetic's
    const int chunks = (pos[r] + 1 + CHUNK - 1) / CHUNK;
    int top = 0;
    for (int c = 0; c < chunks; ++c) {
        const size_t base = ((size_t)r * max_chunks + c) * NQ + h;
        if (d == 0) {
            const float2 ml = part_ml[base];
            sm[top] = ml.x;
            sl[top] = ml.y;
            slev[top] = 0;
        }
        so[top][d] = part_o[base * D + d];
        ++top;
        __syncthreads();
        while (top >= 2 && slev[top - 2] == slev[top - 1]) {
            const float ma = sm[top - 2], la = sl[top - 2], mb = sm[top - 1], lb = sl[top - 1];
            const int lev = slev[top - 1] + 1;
            const float m = fmaxf(ma, mb);
            const float sa = expf(ma - m), sb = expf(mb - m);
            const float l = la * sa + lb * sb;
            so[top - 2][d] = so[top - 2][d] * sa + so[top - 1][d] * sb;
            __syncthreads();
            if (d == 0) {
                sm[top - 2] = m;
                sl[top - 2] = l;
                slev[top - 2] = lev;
            }
            --top;
            __syncthreads();
        }
    }
    // the stack is the left spine of the tree: fold it from the right
    int cur = top - 1;
    for (int i = top - 2; i >= 0; --i) {
        const float ma = sm[i], la = sl[i], mb = sm[cur], lb = sl[cur];
        const float m = fmaxf(ma, mb);
        const float sa = expf(ma - m), sb = expf(mb - m);
        const float l = la * sa + lb * sb;
        so[i][d] = so[i][d] * sa + so[cur][d] * sb;
        __syncthreads();
        if (d == 0) {
            sm[i] = m;
            sl[i] = l;
        }
        __syncthreads();
        cur = i;
    }
    out[((size_t)r * NQ + h) * D + d] = so[0][d] / sl[0];
}

}  // namespace

GB_EXPORT int gb_nemo_kv_write(const float* k_in, const float* v_in, const int* slot, const int* pos, const int* table, int max_blocks,
                               void* kc, void* vc, int rows, void* stream) {
    if (rows <= 0) return 0;
    kv_write_kernel<<<rows, KV_ROW, 0, gb_stream(stream)>>>(k_in, v_in, slot, pos, table, max_blocks, (__half*)kc, (__half*)vc);
    return GB_CHECK_LAUNCH();
}

// `--kv-dtype int8`: the same page with int8 values (nemotron_kv8.cuh).
GB_EXPORT int gb_nemo_kv_write_i8(const float* k_in, const float* v_in, const int* slot, const int* pos, const int* table,
                                  int max_blocks, void* kc, void* vc, int rows, void* stream) {
    if (rows <= 0) return 0;
    kv_write8_kernel<<<rows, KV_ROW, 0, gb_stream(stream)>>>(k_in, v_in, slot, pos, table, max_blocks, (uint8_t*)kc,
                                                                   (uint8_t*)vc);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nemo_attn_partial(const float* q, const void* kc, const void* vc, const int* table, int max_blocks, const int* slot,
                                   const int* pos, int rows, int chunks, int limit, int stride, void* part_ml, void* part_o,
                                   void* stream) {
    if (rows <= 0 || chunks <= 0) return 0;
    dim3 grid(rows, NKV, chunks);
    partial_kernel<false><<<grid, THREADS, 0, gb_stream(stream)>>>(q, kc, vc, table, max_blocks, slot, pos, limit, stride,
                                                                   (float2*)part_ml, (float*)part_o);
    return GB_CHECK_LAUNCH();
}

// The tree's partials over an int8 page (`--kv-dtype int8`).
GB_EXPORT int gb_nemo_attn_partial_i8(const float* q, const void* kc, const void* vc, const int* table, int max_blocks, const int* slot,
                                      const int* pos, int rows, int chunks, int limit, int stride, void* part_ml, void* part_o,
                                      void* stream) {
    if (rows <= 0 || chunks <= 0) return 0;
    dim3 grid(rows, NKV, chunks);
    partial_kernel<true><<<grid, THREADS, 0, gb_stream(stream)>>>(q, kc, vc, table, max_blocks, slot, pos, limit, stride,
                                                                           (float2*)part_ml, (float*)part_o);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nemo_attn_merge(const int* pos, const void* part_ml, const void* part_o, int rows, int max_chunks, int limit,
                                 float* out, void* stream) {
    if (rows <= 0) return 0;
    dim3 grid(rows, NQ);
    merge_kernel<<<grid, D, 0, gb_stream(stream)>>>(pos, (const float2*)part_ml, (const float*)part_o, max_chunks, limit, out);
    return GB_CHECK_LAUNCH();
}

