// Integer-rate circular Viterbi tile quantization: quantize_tiles_kernel.cuh from ExLlamaV3 (MIT), ported
// verbatim with no ATen headers. K = 1..8 bits per weight, cb = 0..2, L = 256 (16x16 EXL3 tile) or 160
// (n-gram embedding rows, mul1 only). Needs the codebook helpers of exl3_encode_codebook.cuh.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace gbx {

#ifndef H_INF
#define H_INF __ushort_as_half(0x7c00)
#endif

// Smallest K whose cost array ((65536 >> K) halves) is kept in shared memory; below it it lives in
// the caller's global scratch. A single cost buffer plus register staging keeps K = 1 (64 KiB) within the
// 99 KiB per-block ceiling of GB10 (sm_121), RTX 4090 (sm_89) and RTX 5090 (sm_120).
#ifndef QT_SHARED_COSTS_MIN_K
#define QT_SHARED_COSTS_MIN_K 1
#endif

#if defined(__CUDA_ARCH__)
#define QT_ARCH __CUDA_ARCH__
#else
#define QT_ARCH 0
#endif

__host__ __device__ constexpr int qt_num_threads(int K, int arch) {
    return K == 1 ? 1024 : ((arch < 1000 && K >= 7) ? (32768 >> K) : 512);
}
__host__ __device__ constexpr int qt_min_blocks(int K, int arch) {
    return (K == 6 || K == 1) ? 1 : 1024 / qt_num_threads(K, arch);
}

// Unsigned warp minimum (packed cost / rank keys).
__device__ __forceinline__ uint32_t qt_warp_min(uint32_t value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = min(value, __shfl_xor_sync(0xffffffff, value, offset));
    return value;
}

template <int K, int cb, int L = 256>
__global__ __launch_bounds__(qt_num_threads(K, QT_ARCH), qt_min_blocks(K, QT_ARCH))
void quantize_tiles_kernel
(
    const float* __restrict__ input_tiles_ptr,
    float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr,
    half* __restrict__ temp_costs_ptr,
    uint16_t* __restrict__ temp_edges_ptr,
    const half2* __restrict__ lut
)
{
    extern __shared__ uint8_t shbuf[];
    uint8_t* sh = shbuf;

    constexpr int NT = qt_num_threads(K, QT_ARCH);
    constexpr int NW = NT / 32;
    constexpr int Kr = 16 - K;
    constexpr int max_q = 1 << K;
    constexpr int edges = 65536 >> K;
    constexpr int PAIRS_PER_THREAD = (edges / 2) > NT ? ((edges / 2) / NT) : 1;

    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + L * tile_idx;
    float* output_tile = output_tiles_ptr + L * tile_idx;
    uint16_t* output_indices = output_indices_ptr + L * tile_idx;
    constexpr int hist_u16_stride = K <= 4 ? (edges / 4) : edges;
    uint16_t* temp_edges = temp_edges_ptr + (size_t) L * hist_u16_stride * tile_idx;

    half* sh_input_tile = (half*) sh; sh += L * sizeof(half);
    half* sh_min = (half*) sh; sh += 32 * sizeof(half);
    int* sh_idx = (int*) sh; sh += 32 * sizeof(int);

    // Single shared cost buffer (edges halves); the previous 2-buffer ping-pong exceeded the 99 KiB
    // shared-memory ceiling on GB10 / RTX 4090 / RTX 5090 at K = 1, so writes are staged in registers.
    half* sh_costs = (half*) sh;
    half* temp_costs = K >= QT_SHARED_COSTS_MIN_K ? sh_costs : temp_costs_ptr + (size_t) 2 * edges * tile_idx;
    half* temp_costs_inc = temp_costs + edges;

    for (int i = thread; i < L; i += NT) sh_input_tile[i] = __float2half_rn(input_tile[i]);
    __syncthreads();

    // ri = (i + roll) mod L, with i < L and roll in {0, L / 2}
    auto ring = [&](int i, int roll)
    {
        int ri = i + roll;
        if (ri >= L) ri -= L;
        return ri;
    };

    auto forward = [&](int roll, int pre_state)
    {
        constexpr bool shared_costs = K >= QT_SHARED_COSTS_MIN_K;
        int ri = ring(0, roll);

        // Computes the Viterbi minimum cost for one outgoing edge pair (two states), writes the
        // history row, and either stages the packed minimum into reg_out or stores it straight into
        // the cost buffer. costs_in == nullptr selects the first step (no incoming costs to read).
        auto process_pair = [&](int out_edge_idx, const half* costs_in, half2* reg_out)
        {
            const half2 w2 = __half2half2(sh_input_tile[ri]);
            int in_edge_idx = out_edge_idx >> K;
            uint32_t product0 = 0;
            uint32_t product1 = 0;
            half2 decoded2;
            if constexpr (cb == 1)
            {
                product0 = out_edge_idx * 0xCBAC1FEDu;
                product1 = product0 + 0xCBAC1FEDu;
                decoded2 = decode_mcg_product_2(product0, product1);
            }
            else if constexpr (cb == 2)
            {
                product0 = out_edge_idx * 0x83DCD12Du;
                product1 = product0 + 0x83DCD12Du;
                decoded2 = decode_mul1_product_2(product0, product1);
            }
            else
            {
                decoded2 = decode_3inst_2<cb>(out_edge_idx, out_edge_idx + 1);
            }
            half2 dh2 = __hsub2(decoded2, w2);
            half2 min_err2 = costs_in
                ? __hfma2(dh2, dh2, __half2half2(costs_in[in_edge_idx]))
                : __hmul2(dh2, dh2);
            if (!costs_in && pre_state >= 0 && in_edge_idx != pre_state) min_err2 = __half2half2(H_INF);
            int min_in_edge0 = in_edge_idx;
            int min_in_edge1 = in_edge_idx;
            uint8_t best_k0 = 0;
            uint8_t best_k1 = 0;

            #pragma unroll
            for (int k = 1; k < max_q; ++k)
            {
                const int state0 = (k << Kr) | out_edge_idx;
                in_edge_idx = state0 >> K;
                if constexpr (cb == 1)
                {
                    // MCG multiplication is linear modulo 2^32 across successive branch states.
                    constexpr uint32_t product_step = 0xCBAC1FEDu << Kr;
                    product0 += product_step;
                    product1 += product_step;
                    decoded2 = decode_mcg_product_2(product0, product1);
                }
                else if constexpr (cb == 2)
                {
                    // The mul1 multiplication is equally linear modulo 2^32.
                    constexpr uint32_t product_step = 0x83DCD12Du << Kr;
                    product0 += product_step;
                    product1 += product_step;
                    decoded2 = decode_mul1_product_2(product0, product1);
                }
                else
                {
                    decoded2 = decode_3inst_2<cb>(state0, state0 + 1);
                }
                dh2 = __hsub2(decoded2, w2);
                half2 err2 = costs_in
                    ? __hfma2(dh2, dh2, __half2half2(costs_in[in_edge_idx]))
                    : __hmul2(dh2, dh2);
                if (!costs_in && pre_state >= 0 && in_edge_idx != pre_state) err2 = __half2half2(H_INF);
                if (__hlt(__low2half(err2), __low2half(min_err2)))
                {
                    min_err2 = __halves2half2(__low2half(err2), __high2half(min_err2));
                    min_in_edge0 = in_edge_idx;
                    best_k0 = (uint8_t) k;
                }
                if (__hlt(__high2half(err2), __high2half(min_err2)))
                {
                    min_err2 = __halves2half2(__low2half(min_err2), __high2half(err2));
                    min_in_edge1 = in_edge_idx;
                    best_k1 = (uint8_t) k;
                }
            }

            if (reg_out)
                *reg_out = min_err2;
            else
                reinterpret_cast<half2*>(temp_costs)[out_edge_idx >> 1] = min_err2;
            // The first pass only traces back to position 0, so its history for the rolled second half
            // (ri >= L / 2) is never read
            if (pre_state >= 0 || ri < L / 2)
            {
                if constexpr (K <= 4)
                {
                    reinterpret_cast<uint8_t*>(temp_edges)[(size_t)(edges >> 1) * ri + (out_edge_idx >> 1)] =
                        best_k0 | (best_k1 << 4);
                }
                else
                {
                    temp_edges[edges * ri + out_edge_idx] = (uint16_t) min_in_edge0;
                    temp_edges[edges * ri + out_edge_idx + 1] = (uint16_t) min_in_edge1;
                }
            }
        };

        if constexpr (shared_costs)
        {
            // Single shared cost buffer, staged through registers: the first step writes straight into
            // it (nothing to read), later steps gather each thread's outputs into registers, sync,
            // flush the registers into the buffer, then sync again before the next step reads.
            for (int out_edge_idx = 2 * thread; out_edge_idx < edges; out_edge_idx += 2 * NT)
                process_pair(out_edge_idx, nullptr, nullptr);
            __syncthreads();

            for (int i = 1; i < L; ++i)
            {
                ri = ring(i, roll);
                half2 reg_costs[PAIRS_PER_THREAD];
                #pragma unroll
                for (int idx = 0; idx < PAIRS_PER_THREAD; ++idx)
                {
                    const int out_edge_idx = (thread + idx * NT) * 2;
                    if (out_edge_idx < edges)
                        process_pair(out_edge_idx, sh_costs, &reg_costs[idx]);
                }
                __syncthreads();
                #pragma unroll
                for (int idx = 0; idx < PAIRS_PER_THREAD; ++idx)
                {
                    const int out_edge_idx = (thread + idx * NT) * 2;
                    if (out_edge_idx < edges)
                        reinterpret_cast<half2*>(sh_costs)[out_edge_idx >> 1] = reg_costs[idx];
                }
                __syncthreads();
            }
        }
        else
        {
            // Ping-pong cost buffers in caller-provided global scratch (unchanged reference logic).
            half* t = temp_costs;
            temp_costs = temp_costs_inc;
            temp_costs_inc = t;

            for (int out_edge_idx = 2 * thread; out_edge_idx < edges; out_edge_idx += 2 * NT)
                process_pair(out_edge_idx, nullptr, nullptr);
            __syncthreads();

            for (int i = 1; i < L; ++i)
            {
                ri = ring(i, roll);
                t = temp_costs;
                temp_costs = temp_costs_inc;
                temp_costs_inc = t;

                for (int out_edge_idx = 2 * thread; out_edge_idx < edges; out_edge_idx += 2 * NT)
                    process_pair(out_edge_idx, temp_costs_inc, nullptr);
                __syncthreads();
            }
        }
    };

    auto argmin_cost = [&]()
    {
        constexpr bool shared_costs = K >= QT_SHARED_COSTS_MIN_K;
        const half* costs = shared_costs ? sh_costs : temp_costs;
        // forward leaves the final step's costs in the cost buffer (backward starts at that step). The
        // rank makes the tie order independent of the block size: bit-reversed warp, then
        // bit-reversed lane, then increasing 1024-state stripe, i.e. the historical 512-thread
        // reduction order. Every cost is a nonnegative half, so packed unsigned minima order them
        uint32_t best = 0x7c00ffffu;
        for (int e = thread; e < edges; e += NT)
        {
            unsigned v = e & 1023;
            unsigned rank = ((__brev(v >> 5) >> 27) << 10) | ((__brev(v & 31) >> 27) << 5) | (e >> 10);
            unsigned key = ((uint32_t) __half_as_ushort(costs[e]) << 16) | rank;
            best = min(best, key);
        }
        best = qt_warp_min(best);
        if ((thread & 31) == 0)
            ((uint32_t*) sh_idx)[thread >> 5] = best;
        __syncthreads();
        if (thread < 32)
        {
            best = thread < NW ? ((uint32_t*) sh_idx)[thread] : 0x7c00ffffu;
            best = qt_warp_min(best);
        }
        unsigned rank = best & 65535;
        unsigned v = ((__brev(rank >> 10) >> 27) << 5) | (__brev((rank >> 5) & 31) >> 27);
        // Every cost inf/NaN (degenerate input): return a valid edge so backward stays in bounds
        return best >= 0x7c000000u ? 0 : (int) (((rank & 31) << 10) | v);
    };

    auto backward = [&](int roll, bool write, int edge)
    {
        if (thread == 0)
        {
            for (int i = L - 1; i >= 0; --i)
            {
                const int ri = ring(i, roll);
                int prev_edge;
                if constexpr (K <= 4)
                {
                    const uint8_t pk = reinterpret_cast<const uint8_t*>(temp_edges)[(size_t)(edges >> 1) * ri + (edge >> 1)];
                    const int kb = (edge & 1) ? (pk >> 4) : (pk & 0xF);
                    prev_edge = ((kb << Kr) | edge) >> K;
                }
                else
                {
                    prev_edge = (int) temp_edges[edges * ri + edge];
                }
                const int encoded = (prev_edge << K) | edge;
                edge = prev_edge;
                if (write)
                {
                    output_indices[ri] = (uint16_t) encoded;
                    output_tile[ri] = __half2float(decode_3inst<cb>(encoded));
                }
                else if (ri == 0) break;
            }
        }
        if (thread == 0) sh_idx[0] = edge;
        __syncthreads();
        return sh_idx[0];
    };

    forward(L / 2, -1);
    int end_state = backward(L / 2, false, argmin_cost());
    forward(0, end_state);
    backward(0, true, end_state);
}

#undef QT_ARCH

}  // namespace gbx
