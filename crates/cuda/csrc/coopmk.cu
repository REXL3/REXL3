// CoopMK: Runtime-K Cooperative MoE decode kernel for packed EXL3 weights.
// Author: Victor Cruz (vcruz305@gmail.com)
// Supports mixed-K per-expert bitwidths (K=1..8, mul1 codebook) with zero requantization.

#include "common.cuh"
#include "exl3_linear_decode.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using namespace gbx;

namespace {

constexpr int MOE_THREADS = 512;
constexpr int MOE_WARPS = MOE_THREADS / 32; // 16 warps
constexpr int MAX_EXPERTS = 512;
constexpr int MAX_TOPK = 32;

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

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

__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        if (bias) v[j] += __half2float(__ldg(bias + col + j));
    }
}

// Softmax + TopK selection on gate logits
extern "C" __global__ void moe_topk_softmax_kernel(
    const __nv_bfloat16* __restrict__ gate_logits, // [T, num_experts]
    const __nv_bfloat16* __restrict__ shared_gate_scalar, // [T] optional
    unsigned int* __restrict__ expert_indices,      // [T, stride_k]
    float* __restrict__ expert_weights,             // [T, stride_k]
    unsigned int num_experts,
    unsigned int top_k
) {
    const unsigned int t = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int stride_k = shared_gate_scalar ? (top_k + 1) : top_k;
    const __nv_bfloat16* logits = gate_logits + (size_t)t * num_experts;
    unsigned int* out_idx = expert_indices + (size_t)t * stride_k;
    float* out_w = expert_weights + (size_t)t * stride_k;

    __shared__ float s_logits[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOPK];
    __shared__ unsigned int s_top_idxs[MAX_TOPK];

    // Load logits into shared memory (up to 512 experts with 256 threads)
    for (unsigned int i = tid; i < num_experts; i += blockDim.x) {
        s_logits[i] = __bfloat162float(logits[i]);
    }
    __syncthreads();

    // Iteratively find top-k (block-cooperative)
    for (unsigned int k = 0; k < top_k; ++k) {
        float max_val = -1e30f;
        unsigned int max_idx = 0;

        for (unsigned int i = tid; i < num_experts; i += blockDim.x) {
            float v = s_logits[i];
            if (v > max_val) {
                max_val = v;
                max_idx = i;
            }
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_v = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
            unsigned int other_i = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
            if (other_v > max_val || (other_v == max_val && other_i < max_idx)) {
                max_val = other_v;
                max_idx = other_i;
            }
        }

        __shared__ float s_warp_max[8];
        __shared__ unsigned int s_warp_idx[8];
        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid & 31;
        if (lane == 0) {
            s_warp_max[warp_id] = max_val;
            s_warp_idx[warp_id] = max_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float b_max = s_warp_max[0];
            unsigned int b_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < (blockDim.x / 32); ++w) {
                if (s_warp_max[w] > b_max || (s_warp_max[w] == b_max && s_warp_idx[w] < b_idx)) {
                    b_max = s_warp_max[w];
                    b_idx = s_warp_idx[w];
                }
            }
            s_top_vals[k] = b_max;
            s_top_idxs[k] = b_idx;
            s_logits[b_idx] = -1e30f; // Mask out selected
        }
        __syncthreads();
    }

    // Softmax over selected top-k
    if (tid == 0) {
        float max_v = s_top_vals[0];
        float sum_exp = 0.0f;
        for (unsigned int k = 0; k < top_k; ++k) {
            sum_exp += __expf(s_top_vals[k] - max_v);
        }
        float inv_sum = 1.0f / sum_exp;
        for (unsigned int k = 0; k < top_k; ++k) {
            out_idx[k] = s_top_idxs[k];
            out_w[k] = __expf(s_top_vals[k] - max_v) * inv_sum;
        }
        if (shared_gate_scalar) {
            out_idx[top_k] = num_experts;
            float g_scalar = __bfloat162float(shared_gate_scalar[t]);
            out_w[top_k] = 1.0f / (1.0f + expf(-g_scalar));
        }
    }
}

// Runtime-K decode and MMA for mul1 codebook. `k2` is the EXL3 K2 (twice the bits, 2..16), as ktab now stores it
// (the same convention as FrostMoE).
__device__ __forceinline__ void decode_tile_mma(
    int k2,
    const uint32_t* tile,
    int lane,
    const uint32_t (&a)[4],
    float (&acc0)[4],
    float (&acc1)[4]
) {
    #define RUNTIME_K_CASE(K2_) \
    case K2_: { \
        constexpr int K2 = K2_; \
        uint32_t w[lane_words<K2>()]; \
        ldg_lane_words<K2>(tile, lane, w); \
        uint32_t b0[2], b1[2]; \
        decode_lane<K2, CB_MUL1>(w, lane, b0, b1); \
        mma16816(acc0, a, b0); \
        mma16816(acc1, a, b1); \
        break; \
    }
    switch (k2) {
        RUNTIME_K_CASE(2)
        RUNTIME_K_CASE(3)
        RUNTIME_K_CASE(4)
        RUNTIME_K_CASE(5)
        RUNTIME_K_CASE(6)
        RUNTIME_K_CASE(7)
        RUNTIME_K_CASE(8)
        RUNTIME_K_CASE(10)
        RUNTIME_K_CASE(12)
        RUNTIME_K_CASE(14)
        RUNTIME_K_CASE(16)
        default: break;
    }
    #undef RUNTIME_K_CASE
}

// The two kernels below are templates over the hidden width HB and the expert intermediate width IB (the literals the Flash-Next kernel
// had: H = 2560 = 20 x 128, I = 640 = 5 x 128, and the shared-memory offsets and block counts that follow from them). One body,
// instantiated for (2560, 640), which is the very code Flash-Next ran (the SASS of that instantiation is identical to the one of the
// literals' build; tests/fn_moe.rs and its golden pin it), and for (2048, 512) (the Qwen3.5/3.6-35B-A3B backbone of MiniCPM-V-4.7:
// 256 experts, top-8, one shared expert). The runtime H and I arguments stay (the launcher checks them against the instantiation).
//
// Stage A: Gate and Up projections + SwiGLU activation
// Grid: (num_slots, num_col_blocks_I) where num_col_blocks_I = I / 128 (640 / 128 = 5; 512 / 128 = 4)
// Block: 512 threads (16 warps)
#define COOPMK_A_PARAMS                                                                                                          \
    const __nv_bfloat16* __restrict__ x,  /* [T, H] */                                                                          \
    const unsigned int* __restrict__ sel, /* [T, top_k] */                                                                      \
    const uint64_t* __restrict__ g_trellis_ptrs, const uint64_t* __restrict__ g_suh_ptrs,                                      \
    const uint64_t* __restrict__ g_svh_ptrs, const uint64_t* __restrict__ u_trellis_ptrs,                                      \
    const uint64_t* __restrict__ u_suh_ptrs, const uint64_t* __restrict__ u_svh_ptrs,                                          \
    const int* __restrict__ k_tab,        /* [3 * num_experts]: gate, up, down */                                              \
    __nv_bfloat16* __restrict__ act_out,  /* [T, top_k, I] */                                                                  \
    unsigned int H, unsigned int I, unsigned int top_k

template <int HB, int IB>
__global__ __launch_bounds__(MOE_THREADS) void coopmk_stage_a_k(COOPMK_A_PARAMS) {
    const unsigned int slot_idx = blockIdx.x; // token_idx * top_k + k_idx
    const unsigned int col_blk = blockIdx.y;  // 0 .. 4 (each 128 columns)
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;

    const unsigned int expert_id = sel[slot_idx];
    const int k_gate = k_tab[expert_id * 3 + 0];
    const int k_up   = k_tab[expert_id * 3 + 1];

    const uint32_t* g_trellis = (const uint32_t*)g_trellis_ptrs[expert_id];
    const half* g_suh = (const half*)g_suh_ptrs[expert_id];
    const half* g_svh = (const half*)g_svh_ptrs[expert_id];

    const uint32_t* u_trellis = (const uint32_t*)u_trellis_ptrs[expert_id];
    const half* u_suh = (const half*)u_suh_ptrs[expert_id];
    const half* u_svh = (const half*)u_svh_ptrs[expert_id];

    extern __shared__ char smem_a_raw[];
    half* s_xg = reinterpret_cast<half*>(smem_a_raw);
    half* s_xu = reinterpret_cast<half*>(smem_a_raw + HB * 2);
    float* s_red_g = reinterpret_cast<float*>(smem_a_raw + HB * 4);
    float* s_red_u = s_red_g + 16 * 128;

    const unsigned int token_id = slot_idx / top_k;
    const __nv_bfloat16* x_tok = x + (size_t)token_id * H;

    // 1. Cooperative pre-rotation of x into shared memory
    for (int b = warp; b < HB / 128; b += MOE_WARPS) {
        int k_col = b * 128 + 4 * lane;
        float vg[4], vu[4], sg[4], su[4];
        load4(x_tok, BF16, k_col, vg);
        load4(x_tok, BF16, k_col, vu);
        load4(g_suh, F16, k_col, sg);
        load4(u_suh, F16, k_col, su);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            vg[j] *= sg[j];
            vu[j] *= su[j];
        }
        fwht128(vg, lane);
        fwht128(vu, lane);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            vg[j] *= HAD_SCALE;
            vu[j] *= HAD_SCALE;
        }
        store4(s_xg, F16, k_col, vg);
        store4(s_xu, F16, k_col, vu);
    }
    __syncthreads();

    const int K_tiles = H >> 4; // 160 tiles (128 at H = 2048)
    const int tiles_per_warp = K_tiles / MOE_WARPS; // 10 tiles (8)
    const int kt0 = warp * tiles_per_warp;

    float acc_g[8][2][4] = {0};
    float acc_u[8][2][4] = {0};

    const size_t g_words_per_tile = (size_t)k_gate * 4;  // K2 words a tile (4 * K2)
    const size_t u_words_per_tile = (size_t)k_up * 4;
    const size_t g_k_stride = (I >> 4) * g_words_per_tile;
    const size_t u_k_stride = (I >> 4) * u_words_per_tile;
    const size_t g_nb_stride = 8 * g_words_per_tile;
    const size_t u_nb_stride = 8 * u_words_per_tile;

    const uint32_t* g_tiles = g_trellis + col_blk * g_nb_stride;
    const uint32_t* u_tiles = u_trellis + col_blk * u_nb_stride;

    // 2. K-tiles MMA loop
    for (int i = 0; i < tiles_per_warp; ++i) {
        int kt = kt0 + i;
        if (kt >= K_tiles) break;

        const uint32_t* xg = reinterpret_cast<const uint32_t*>(s_xg + kt * 16);
        const uint32_t* xu = reinterpret_cast<const uint32_t*>(s_xu + kt * 16);

        uint32_t ag[4], au[4];
        ag[0] = xg[t];
        ag[1] = ag[0];
        ag[2] = xg[4 + t];
        ag[3] = ag[2];

        au[0] = xu[t];
        au[1] = au[0];
        au[2] = xu[4 + t];
        au[3] = au[2];

        const uint32_t* tile_g = g_tiles + (size_t)kt * g_k_stride;
        const uint32_t* tile_u = u_tiles + (size_t)kt * u_k_stride;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            decode_tile_mma(k_gate, tile_g + j * g_words_per_tile, lane, ag, acc_g[j][0], acc_g[j][1]);
            decode_tile_mma(k_up,   tile_u + j * u_words_per_tile, lane, au, acc_u[j][0], acc_u[j][1]);
        }
    }

    // Warp-level reduction across warps
    __syncthreads();
    if (g < 1) { // 1 row per token
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                int col = i * 16 + h * 8 + 2 * t;
                *reinterpret_cast<float2*>(s_red_g + warp * 128 + col) = make_float2(acc_g[i][h][0], acc_g[i][h][1]);
                *reinterpret_cast<float2*>(s_red_u + warp * 128 + col) = make_float2(acc_u[i][h][0], acc_u[i][h][1]);
            }
        }
    }
    __syncthreads();

    // Warp 0 reduces over warps, applies Had_N * svh, computes SwiGLU
    if (warp == 0) {
        float vg[4] = {0}, vu[4] = {0};
        int col = col_blk * 128 + 4 * lane;

        #pragma unroll
        for (int w = 0; w < MOE_WARPS; ++w) {
            float4 qg = *reinterpret_cast<const float4*>(s_red_g + w * 128 + 4 * lane);
            float4 qu = *reinterpret_cast<const float4*>(s_red_u + w * 128 + 4 * lane);
            vg[0] += qg.x; vg[1] += qg.y; vg[2] += qg.z; vg[3] += qg.w;
            vu[0] += qu.x; vu[1] += qu.y; vu[2] += qu.z; vu[3] += qu.w;
        }

        // Finish Gate: H_N / sqrt(128) * svh
        finish(vg, lane, g_svh, nullptr, col);
        // Finish Up: H_N / sqrt(128) * svh
        finish(vu, lane, u_svh, nullptr, col);

        // SwiGLU: silu(gate) * up
        float act[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float g_val = vg[j];
            float silu_g = g_val / (1.0f + __expf(-g_val));
            act[j] = silu_g * vu[j];
        }

        // Store to act_out [slot_idx, I]
        __nv_bfloat16* dst = act_out + (size_t)slot_idx * I + col;
        store4(dst, BF16, 0, act);
    }
}

// Stage B: Down projection + reduction into out
// Grid: (num_tokens, num_col_blocks_H) where num_col_blocks_H = H / 128 (2560 / 128 = 20; 2048 / 128 = 16)
// Block: 512 threads (16 warps)
#define COOPMK_B_PARAMS                                                                                                          \
    const __nv_bfloat16* __restrict__ act_out, /* [T, top_k, I] */                                                              \
    const unsigned int* __restrict__ sel,      /* [T, top_k] */                                                                 \
    const float* __restrict__ rw,              /* [T, top_k] */                                                                 \
    const uint64_t* __restrict__ d_trellis_ptrs, const uint64_t* __restrict__ d_suh_ptrs,                                      \
    const uint64_t* __restrict__ d_svh_ptrs,                                                                                    \
    const int* __restrict__ k_tab,             /* [3 * num_experts] */                                                          \
    __nv_bfloat16* __restrict__ out,           /* [T, H] */                                                                     \
    unsigned int H, unsigned int I, unsigned int top_k

template <int HB, int IB>
__global__ __launch_bounds__(MOE_THREADS) void coopmk_stage_b_k(COOPMK_B_PARAMS) {
    const unsigned int token_id = blockIdx.x;
    const unsigned int col_blk = blockIdx.y; // 0 .. 19 (each 128 columns of H)
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;

    extern __shared__ char smem_b_raw[];
    half* s_xd = reinterpret_cast<half*>(smem_b_raw);
    float* s_red_d = reinterpret_cast<float*>(smem_b_raw + IB * 2);

    float token_acc[4] = {0.0f};

    // Iterate over top-k experts for this token
    for (unsigned int k_idx = 0; k_idx < top_k; ++k_idx) {
        unsigned int slot_idx = token_id * top_k + k_idx;
        unsigned int expert_id = sel[slot_idx];
        float weight = rw[slot_idx];
        if (weight <= 0.0f) continue;

        const int k_down = k_tab[expert_id * 3 + 2];
        const uint32_t* d_trellis = (const uint32_t*)d_trellis_ptrs[expert_id];
        const half* d_suh = (const half*)d_suh_ptrs[expert_id];
        const half* d_svh = (const half*)d_svh_ptrs[expert_id];

        const size_t d_words_per_tile = (size_t)k_down * 4;  // K2 words a tile (4 * K2)
        const size_t d_k_stride = (H >> 4) * d_words_per_tile;
        const size_t d_nb_stride = 8 * d_words_per_tile;

        const uint32_t* d_tiles = d_trellis + col_blk * d_nb_stride;
        const __nv_bfloat16* act_tok = act_out + (size_t)slot_idx * I;

        // 1. Cooperative pre-rotation of act_tok into shared memory
        if (warp < IB / 128) {
            int k_col = warp * 128 + 4 * lane;
            float vd[4], sd[4];
            load4(act_tok, BF16, k_col, vd);
            load4(d_suh, F16, k_col, sd);
            #pragma unroll
            for (int j = 0; j < 4; ++j) vd[j] *= sd[j];
            fwht128(vd, lane);
            #pragma unroll
            for (int j = 0; j < 4; ++j) vd[j] *= HAD_SCALE;
            store4(s_xd, F16, k_col, vd);
        }
        __syncthreads();

        // 2. K-tiles MMA loop
        const int K_tiles = I >> 4; // 40 tiles (32 at I = 512)
        const int tiles_per_warp = (K_tiles + MOE_WARPS - 1) / MOE_WARPS;
        const int kt0 = warp * tiles_per_warp;

        float acc[8][2][4] = {0};

        for (int i = 0; i < tiles_per_warp; ++i) {
            int kt = kt0 + i;
            if (kt >= K_tiles) break;

            const uint32_t* xd = reinterpret_cast<const uint32_t*>(s_xd + kt * 16);
            uint32_t ad[4];
            ad[0] = xd[t];
            ad[1] = ad[0];
            ad[2] = xd[4 + t];
            ad[3] = ad[2];

            const uint32_t* tile_d = d_tiles + (size_t)kt * d_k_stride;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                decode_tile_mma(k_down, tile_d + j * d_words_per_tile, lane, ad, acc[j][0], acc[j][1]);
            }
        }

        // Reduction across warps
        __syncthreads();
        if (g < 1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int col = i * 16 + h * 8 + 2 * t;
                    *reinterpret_cast<float2*>(s_red_d + warp * 128 + col) = make_float2(acc[i][h][0], acc[i][h][1]);
                }
            }
        }
        __syncthreads();

        if (warp == 0) {
            float vd[4] = {0};
            int col = col_blk * 128 + 4 * lane;

            #pragma unroll
            for (int w = 0; w < MOE_WARPS; ++w) {
                float4 qd = *reinterpret_cast<const float4*>(s_red_d + w * 128 + 4 * lane);
                vd[0] += qd.x; vd[1] += qd.y; vd[2] += qd.z; vd[3] += qd.w;
            }

            // Finish Down: H_N / sqrt(128) * svh
            finish(vd, lane, d_svh, nullptr, col);

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                token_acc[j] += weight * vd[j];
            }
        }
    }

    if (warp == 0) {
        int col = col_blk * 128 + 4 * lane;
        __nv_bfloat16* dst = out + (size_t)token_id * H + col;
        store4(dst, BF16, 0, token_acc);
    }
}

} // namespace

// C ABI Exported Entry Points
GB_EXPORT int gb_moe_topk_softmax(
    const void* gate_logits,
    const void* shared_gate_scalar,
    unsigned int* expert_indices,
    float* expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int num_tokens,
    void* stream
) {
    if (num_tokens == 0) return 0;
    dim3 grid(num_tokens);
    dim3 block(256);
    moe_topk_softmax_kernel<<<grid, block, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)gate_logits,
        (const __nv_bfloat16*)shared_gate_scalar,
        expert_indices,
        expert_weights,
        num_experts,
        top_k
    );
    return GB_CHECK_LAUNCH();
}

GB_EXPORT size_t gb_coopmk_workspace_bytes(
    unsigned int num_tokens,
    unsigned int top_k,
    unsigned int I
) {
    // Intermediate activations: [num_tokens, top_k, I] in BF16
    return (size_t)num_tokens * top_k * I * sizeof(__nv_bfloat16);
}

GB_EXPORT int gb_coopmk_decode(
    const void* x,                     // [T, H] BF16
    const unsigned int* sel,           // [T, top_k]
    const float* rw,                   // [T, top_k]
    const uint64_t* g_trellis_ptrs,    // [512]
    const uint64_t* g_suh_ptrs,
    const uint64_t* g_svh_ptrs,
    const uint64_t* u_trellis_ptrs,
    const uint64_t* u_suh_ptrs,
    const uint64_t* u_svh_ptrs,
    const uint64_t* d_trellis_ptrs,
    const uint64_t* d_suh_ptrs,
    const uint64_t* d_svh_ptrs,
    const int* k_tab,                  // [512 * 3]
    void* act_workspace,               // [T * top_k * I] BF16
    void* out,                         // [T, H] BF16
    unsigned int H,                    // 2560 (Flash-Next) or 2048 (MiniCPM-V-4.7)
    unsigned int I,                    // 640 or 512
    unsigned int top_k,                // 10 or 9 (8 routed + the shared expert)
    unsigned int num_tokens,
    void* stream
) {
    // the two (H, I) the kernels are instantiated for; any other pair was never right (literals of 2560 and 640 inside) and is refused
    const bool flash_next = H == 2560 && I == 640;
    const bool minicpm = H == 2048 && I == 512;
    const bool qwen35_122b = H == 3072 && I == 1024;
    if (!flash_next && !minicpm && !qwen35_122b) return (int)cudaErrorInvalidValue;
    if (num_tokens == 0) return 0;
    cudaStream_t st = gb_stream(stream);

    // Stage A: Gate and Up projections -> SwiGLU -> act_workspace
    unsigned int slots = num_tokens * top_k;
    unsigned int col_blks_I = I / 128; // 5 (4)
    dim3 grid_a(slots, col_blks_I);
    size_t smem_a = (size_t)H * 4 + (size_t)2 * 16 * 128 * sizeof(float); // 26 KB (24 KB)

    if (flash_next) {
        coopmk_stage_a_k<2560, 640><<<grid_a, MOE_THREADS, smem_a, st>>>(
            (const __nv_bfloat16*)x, sel, g_trellis_ptrs, g_suh_ptrs, g_svh_ptrs, u_trellis_ptrs, u_suh_ptrs, u_svh_ptrs, k_tab,
            (__nv_bfloat16*)act_workspace, H, I, top_k);
    } else if (minicpm) {
        coopmk_stage_a_k<2048, 512><<<grid_a, MOE_THREADS, smem_a, st>>>(
            (const __nv_bfloat16*)x, sel, g_trellis_ptrs, g_suh_ptrs, g_svh_ptrs, u_trellis_ptrs, u_suh_ptrs, u_svh_ptrs, k_tab,
            (__nv_bfloat16*)act_workspace, H, I, top_k);
    } else {
        coopmk_stage_a_k<3072, 1024><<<grid_a, MOE_THREADS, smem_a, st>>>(
            (const __nv_bfloat16*)x, sel, g_trellis_ptrs, g_suh_ptrs, g_svh_ptrs, u_trellis_ptrs, u_suh_ptrs, u_svh_ptrs, k_tab,
            (__nv_bfloat16*)act_workspace, H, I, top_k);
    }
    int rc = GB_CHECK_LAUNCH();
    if (rc != 0) return rc;

    // Stage B: Down projection -> reduction -> out
    unsigned int col_blks_H = H / 128; // 20 (16)
    dim3 grid_b(num_tokens, col_blks_H);
    size_t smem_b = (size_t)I * 2 + (size_t)16 * 128 * sizeof(float); // 9.5 KB (9 KB)

    if (flash_next) {
        coopmk_stage_b_k<2560, 640><<<grid_b, MOE_THREADS, smem_b, st>>>(
            (const __nv_bfloat16*)act_workspace, sel, rw, d_trellis_ptrs, d_suh_ptrs, d_svh_ptrs, k_tab, (__nv_bfloat16*)out, H, I,
            top_k);
    } else if (minicpm) {
        coopmk_stage_b_k<2048, 512><<<grid_b, MOE_THREADS, smem_b, st>>>(
            (const __nv_bfloat16*)act_workspace, sel, rw, d_trellis_ptrs, d_suh_ptrs, d_svh_ptrs, k_tab, (__nv_bfloat16*)out, H, I,
            top_k);
    } else {
        coopmk_stage_b_k<3072, 1024><<<grid_b, MOE_THREADS, smem_b, st>>>(
            (const __nv_bfloat16*)act_workspace, sel, rw, d_trellis_ptrs, d_suh_ptrs, d_svh_ptrs, k_tab, (__nv_bfloat16*)out, H, I,
            top_k);
    }
    return GB_CHECK_LAUNCH();
}
