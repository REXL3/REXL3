// gb-nvcc: --expt-relaxed-constexpr
// EXL3 encode launchers: circular-Viterbi tile quantization (integer and fractional rates) and trellis
// bit-packing/unpacking. Pure C-ABI (GB_EXPORT): raw device pointers, 0 or a cudaError_t on return, no
// allocation, no synchronization, no ATen/PyTorch/c10 headers. The kernels are in exl3_encode_kernels.cuh
// (ported from ExLlamaV3; see THIRD_PARTY_NOTICES.md), launched with the reference geometry: one block per
// tile, the block size the compiled kernel's attributes report (the launch_bounds of the small K on older
// arches), dynamic shared memory exactly the reference sizing (the K = 1 cost buffer alone is 64 KB, so
// cudaFuncSetAttribute is required).
//
// Scratch layout (matches ExLlamaV3): temp_costs [batch, 2, edges] half, temp_edges [batch, L, edges]
// uint16_t, edges = 65536 >> K (integer) or 65536 >> KA (fractional). Both quantization paths keep their
// cost buffer in shared memory (single buffer, register-staged) and only use temp_costs when a K below
// QT_SHARED_COSTS_MIN_K is selected. Every chunk of max_batch tiles reads its own tile's scratch row, so
// the caller must size the batch dimension for at least max_batch.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

#include "common.cuh"
#include "exl3_linear_decode.cuh"
#include "exl3_encode_kernels.cuh"

using namespace gbx;

namespace {

template <int K, int CB, int L>
int launch_quantize_tiles(const float* input_tiles, float* output_tiles, uint16_t* output_indices, half* temp_costs,
                          uint16_t* temp_edges, int num_tiles, int max_batch, cudaStream_t st) {
    constexpr int edges = 65536 >> K;
    constexpr int cost_arrays = K >= QT_SHARED_COSTS_MIN_K ? 1 : 0;
    const int shmem = cost_arrays * edges * (int)sizeof(half) + L * (int)sizeof(half) + 64 + 128;
    cudaFuncAttributes attrs;
    cudaError_t e = cudaFuncGetAttributes(&attrs, quantize_tiles_kernel<K, CB, L>);
    if (e != cudaSuccess) return (int)e;
    const int nt = attrs.maxThreadsPerBlock;
    e = cudaFuncSetAttribute(quantize_tiles_kernel<K, CB, L>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    if (e != cudaSuccess) return (int)e;
    for (int b = 0; b < num_tiles; b += max_batch) {
        const int bsz = std::min(max_batch, num_tiles - b);
        quantize_tiles_kernel<K, CB, L><<<bsz, nt, shmem, st>>>(
            input_tiles + (size_t)L * b, output_tiles + (size_t)L * b, output_indices + (size_t)L * b,
            temp_costs, temp_edges, nullptr);
        e = cudaGetLastError();
        if (e != cudaSuccess) return (int)e;
    }
    return 0;
}

template <int KA, uint32_t MASK>
int launch_quantize_tiles_frac(const float* input_tiles, float* output_tiles, uint16_t* output_indices, half* temp_costs,
                               uint16_t* temp_edges, int num_tiles, int max_batch, cudaStream_t st) {
    constexpr int L = 256;
    constexpr int edges_max = 65536 >> KA;
    constexpr int nt = KA == 1 ? 1024 : 512;
    const int shmem = edges_max * (int)sizeof(half) + L * (int)sizeof(half) + 32 * (int)sizeof(int) + 128;
    cudaError_t e = cudaFuncSetAttribute(quantize_tiles_frac_kernel<KA, MASK>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    if (e != cudaSuccess) return (int)e;
    for (int b = 0; b < num_tiles; b += max_batch) {
        const int bsz = std::min(max_batch, num_tiles - b);
        quantize_tiles_frac_kernel<KA, MASK><<<bsz, nt, shmem, st>>>(
            input_tiles + (size_t)L * b, output_tiles + (size_t)L * b, output_indices + (size_t)L * b,
            temp_costs, temp_edges, nullptr);
        e = cudaGetLastError();
        if (e != cudaSuccess) return (int)e;
    }
    return 0;
}

// (K, codebook, L) -> the templated kernel. L = 160 is mul1-only, validated by the caller.
int dispatch_quantize_tiles(int K, int cb, int L, const float* input_tiles, float* output_tiles,
                            uint16_t* output_indices, half* temp_costs, uint16_t* temp_edges, int num_tiles,
                            int max_batch, cudaStream_t st) {
#define GB_QT(K_, CB_, L_) if (K == K_ && cb == CB_ && L == L_) \
        return launch_quantize_tiles<K_, CB_, L_>(input_tiles, output_tiles, output_indices, temp_costs, temp_edges, \
                                                  num_tiles, max_batch, st);
    GB_QT(1, 0, 256) GB_QT(1, 1, 256) GB_QT(1, 2, 256)
    GB_QT(2, 0, 256) GB_QT(2, 1, 256) GB_QT(2, 2, 256)
    GB_QT(3, 0, 256) GB_QT(3, 1, 256) GB_QT(3, 2, 256)
    GB_QT(4, 0, 256) GB_QT(4, 1, 256) GB_QT(4, 2, 256)
    GB_QT(5, 0, 256) GB_QT(5, 1, 256) GB_QT(5, 2, 256)
    GB_QT(6, 0, 256) GB_QT(6, 1, 256) GB_QT(6, 2, 256)
    GB_QT(7, 0, 256) GB_QT(7, 1, 256) GB_QT(7, 2, 256)
    GB_QT(8, 0, 256) GB_QT(8, 1, 256) GB_QT(8, 2, 256)
    GB_QT(1, 2, 160) GB_QT(2, 2, 160) GB_QT(3, 2, 160)
    GB_QT(4, 2, 160) GB_QT(5, 2, 160) GB_QT(6, 2, 160)
    GB_QT(7, 2, 160) GB_QT(8, 2, 160)
#undef GB_QT
    return (int)cudaErrorInvalidValue;
}

template <int K>
int launch_pack_trellis(uint16_t* packed, const uint16_t* unpacked, int kt, int nt, cudaStream_t st) {
    // Reference grid: (rows = kt, cols = nt), tile = kt * blockIdx.y + blockIdx.x
    const dim3 grid((unsigned)kt, (unsigned)nt);
    pack_trellis_kernel<K><<<grid, 128, 0, st>>>(packed, unpacked);
    return GB_CHECK_LAUNCH();
}

template <int K>
int launch_unpack_trellis(uint16_t* unpacked, const uint16_t* packed, int kt, int nt, cudaStream_t st) {
    // Reference grid: (cols = nt, rows = kt), tile = nt * blockIdx.y + blockIdx.x
    const dim3 grid((unsigned)nt, (unsigned)kt);
    unpack_trellis_kernel<K><<<grid, 128, 0, st>>>(unpacked, packed);
    return GB_CHECK_LAUNCH();
}

// k2 = 2 * KA + 1, mask = 0xAAAA (alternating KA / KA+1 bit steps), bpb = 8 * k2 uint16_t per tile
template <int KA>
int launch_pack_trellis_frac(uint16_t* packed, const uint16_t* unpacked, int kt, int nt, cudaStream_t st) {
    const int num_tiles = kt * nt;
    constexpr int bpb = 8 * (2 * KA + 1);
    pack_trellis_frac_kernel<<<(num_tiles + 127) / 128, 128, 0, st>>>(packed, unpacked, num_tiles, KA, 0xaaaau, bpb);
    return GB_CHECK_LAUNCH();
}

template <int KA>
int launch_unpack_trellis_frac(uint16_t* unpacked, const uint16_t* packed, int kt, int nt, cudaStream_t st) {
    const int num_tiles = kt * nt;
    constexpr int bpb = 8 * (2 * KA + 1);
    unpack_trellis_frac_kernel<<<num_tiles, 32, 0, st>>>(unpacked, packed, KA, 0xaaaau, bpb);
    return GB_CHECK_LAUNCH();
}

int dispatch_pack_trellis(int k2, uint16_t* packed, const uint16_t* unpacked, int kt, int nt, cudaStream_t st) {
    if ((k2 & 1) == 0) {
        switch (k2 / 2) {
            case 1: return launch_pack_trellis<1>(packed, unpacked, kt, nt, st);
            case 2: return launch_pack_trellis<2>(packed, unpacked, kt, nt, st);
            case 3: return launch_pack_trellis<3>(packed, unpacked, kt, nt, st);
            case 4: return launch_pack_trellis<4>(packed, unpacked, kt, nt, st);
            case 5: return launch_pack_trellis<5>(packed, unpacked, kt, nt, st);
            case 6: return launch_pack_trellis<6>(packed, unpacked, kt, nt, st);
            case 7: return launch_pack_trellis<7>(packed, unpacked, kt, nt, st);
            case 8: return launch_pack_trellis<8>(packed, unpacked, kt, nt, st);
            default: break;
        }
    } else {
        switch (k2) {
            case 3: return launch_pack_trellis_frac<1>(packed, unpacked, kt, nt, st);
            case 5: return launch_pack_trellis_frac<2>(packed, unpacked, kt, nt, st);
            case 7: return launch_pack_trellis_frac<3>(packed, unpacked, kt, nt, st);
            default: break;
        }
    }
    return (int)cudaErrorInvalidValue;
}

int dispatch_unpack_trellis(int k2, uint16_t* unpacked, const uint16_t* packed, int kt, int nt, cudaStream_t st) {
    if ((k2 & 1) == 0) {
        switch (k2 / 2) {
            case 1: return launch_unpack_trellis<1>(unpacked, packed, kt, nt, st);
            case 2: return launch_unpack_trellis<2>(unpacked, packed, kt, nt, st);
            case 3: return launch_unpack_trellis<3>(unpacked, packed, kt, nt, st);
            case 4: return launch_unpack_trellis<4>(unpacked, packed, kt, nt, st);
            case 5: return launch_unpack_trellis<5>(unpacked, packed, kt, nt, st);
            case 6: return launch_unpack_trellis<6>(unpacked, packed, kt, nt, st);
            case 7: return launch_unpack_trellis<7>(unpacked, packed, kt, nt, st);
            case 8: return launch_unpack_trellis<8>(unpacked, packed, kt, nt, st);
            default: break;
        }
    } else {
        switch (k2) {
            case 3: return launch_unpack_trellis_frac<1>(unpacked, packed, kt, nt, st);
            case 5: return launch_unpack_trellis_frac<2>(unpacked, packed, kt, nt, st);
            case 7: return launch_unpack_trellis_frac<3>(unpacked, packed, kt, nt, st);
            default: break;
        }
    }
    return (int)cudaErrorInvalidValue;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Exports.
// ---------------------------------------------------------------------------------------------------------------

// Circular Viterbi quantization of num_tiles floats tiles of tile_len weights into output_tiles (float) and
// output_indices (uint16_t), k_bits = 1..8 bits per weight, codebook 0..2 (3INST / MCG / MUL1), tile_len 256
// (any codebook) or 160 (mul1 only). max_batch: tiles of scratch the caller's temp_costs/temp_edges hold; the
// work is chunked into launches of at most max_batch tiles.
GB_EXPORT int gb_exl3_quantize_tiles(const float* input_tiles, float* output_tiles, uint16_t* output_indices,
                                     void* temp_costs, uint16_t* temp_edges, int num_tiles, int tile_len,
                                     int k_bits, int codebook, int max_batch, void* stream) {
    if (!input_tiles || !output_tiles || !output_indices || !temp_costs || !temp_edges || num_tiles <= 0 ||
        max_batch <= 0 || k_bits < 1 || k_bits > 8 || codebook < 0 || codebook > 2)
        return (int)cudaErrorInvalidValue;
    if (tile_len != 256 && !(tile_len == 160 && codebook == 2)) return (int)cudaErrorInvalidValue;
    return dispatch_quantize_tiles(k_bits, codebook, tile_len, input_tiles, output_tiles, output_indices,
                                   (half*)temp_costs, temp_edges, num_tiles, max_batch, gb_stream(stream));
}

// Same, at a fractional rate: ka = 1 / 2 / 3 with mask 0xaaaa (alternating ka / ka+1 bit steps = 1.5 / 2.5 /
// 3.5 bits per weight, mul1 codebook only), tile_len 256.
GB_EXPORT int gb_exl3_quantize_tiles_frac(const float* input_tiles, float* output_tiles, uint16_t* output_indices,
                                          void* temp_costs, uint16_t* temp_edges, int num_tiles, int ka,
                                          uint32_t mask, int max_batch, void* stream) {
    if (!input_tiles || !output_tiles || !output_indices || !temp_costs || !temp_edges || num_tiles <= 0 ||
        max_batch <= 0)
        return (int)cudaErrorInvalidValue;
    if (ka < 1 || ka > 3 || mask != 0xaaaau) return (int)cudaErrorInvalidValue;
    if (ka == 1)
        return launch_quantize_tiles_frac<1, 0xaaaau>(input_tiles, output_tiles, output_indices, (half*)temp_costs,
                                                       temp_edges, num_tiles, max_batch, gb_stream(stream));
    if (ka == 2)
        return launch_quantize_tiles_frac<2, 0xaaaau>(input_tiles, output_tiles, output_indices, (half*)temp_costs,
                                                       temp_edges, num_tiles, max_batch, gb_stream(stream));
    return launch_quantize_tiles_frac<3, 0xaaaau>(input_tiles, output_tiles, output_indices, (half*)temp_costs,
                                                   temp_edges, num_tiles, max_batch, gb_stream(stream));
}

// Pack a [kt, nt, 256] uint16_t trellis into [kt, nt, 8 * k2] uint16_t (integer k2 = 2..16: 16 * k2 / 2
// words; fractional k2 = 3 / 5 / 7: 8 * k2 words).
GB_EXPORT int gb_exl3_pack_trellis(uint16_t* packed, const uint16_t* unpacked, int kt, int nt, int k2, void* stream) {
    if (!packed || !unpacked || kt <= 0 || nt <= 0) return (int)cudaErrorInvalidValue;
    const bool ok_even = (k2 & 1) == 0 && k2 >= 2 && k2 <= 16;
    const bool ok_odd = k2 == 3 || k2 == 5 || k2 == 7;
    if (!ok_even && !ok_odd) return (int)cudaErrorInvalidValue;
    return dispatch_pack_trellis(k2, packed, unpacked, kt, nt, gb_stream(stream));
}

// Inverse of gb_exl3_pack_trellis.
GB_EXPORT int gb_exl3_unpack_trellis(uint16_t* unpacked, const uint16_t* packed, int kt, int nt, int k2, void* stream) {
    if (!packed || !unpacked || kt <= 0 || nt <= 0) return (int)cudaErrorInvalidValue;
    const bool ok_even = (k2 & 1) == 0 && k2 >= 2 && k2 <= 16;
    const bool ok_odd = k2 == 3 || k2 == 5 || k2 == 7;
    if (!ok_even && !ok_odd) return (int)cudaErrorInvalidValue;
    return dispatch_unpack_trellis(k2, unpacked, packed, kt, nt, gb_stream(stream));
}
