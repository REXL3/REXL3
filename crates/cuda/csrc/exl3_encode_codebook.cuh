// EXL3 codebook decode helpers (ExLlamaV3 codebook.cuh, MIT), self-contained: a 16-bit codebook index's value
// as fp16. The CUDA lop3 (x & 0x8fff8fff) ^ 0x3b603b60 (table 0x6a) is written as plain C, bit-identical.
// No ATen/PyTorch headers: this file only needs <cuda_fp16.h> and <cstdint>.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace gbx {

__device__ __forceinline__ uint32_t cb_lop3_6a(uint32_t x) { return (x & 0x8fff8fffu) ^ 0x3b603b60u; }

union half2_uint32 {
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16 {
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

// Decode two mul1 (cb 2) codebook entries from precomputed products x0 = idx0 * 0x83DCD12D, x1 = idx1 * 0x83DCD12D.
__device__ inline half2 decode_mul1_product_2(uint32_t x0, uint32_t x1) {
    const uint32_t acc = 0x6400u;  // 0x6400 -> 1024.0 .. 0x67FF -> 2047.0
    uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));  //  0.00677 = 1/147.7
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));  // -10.39 = (-1024.0 - 510.0) * k_inv_h
    half_uint16 h0((uint16_t) sum0);
    half_uint16 h1((uint16_t) sum1);
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

// Ditto mcg (cb 1).
__device__ inline half2 decode_mcg_product_2(uint32_t x0, uint32_t x1) {
    x0 = cb_lop3_6a(x0);
    x1 = cb_lop3_6a(x1);
    half2_uint32 xu0(x0);
    half2_uint32 xu1(x1);
    half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
    half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
    return __hadd2(d0, d1);
}

// One codebook value from one 16-bit index.
template <int cb>
__device__ inline half decode_3inst(uint32_t x) {
    if constexpr (cb == 0) {
        x *= 89226354u;
        x += 64248484u;
        x = cb_lop3_6a(x);
        half2_uint32 xu(x);
        return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
    }
    if constexpr (cb == 1) {
        x *= 0xCBAC1FEDu;
        x = cb_lop3_6a(x);
        half2_uint32 xu(x);
        return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
    }
    if constexpr (cb == 2) {
        x *= 0x83DCD12Du;
        const uint32_t acc = 0x6400u;  // 0x6400 -> 1024.0 .. 0x67FF -> 2047.0
        uint32_t sum = __dp4a(x, 0x01010101u, acc);
        const __half k_inv_h = __ushort_as_half(0x1eee);    //  0.00677 = 1/147.7
        const __half k_bias_h = __ushort_as_half(0xc931);   // -10.39 = (-1024.0 - 510.0) * k_inv_h
        half_uint16 h((uint16_t) sum);
        return __hfma(h.as_half, k_inv_h, k_bias_h);
    }
}

// Two codebook values from two 16-bit indices, as a half2 (the first index in the low half).
template <int cb>
__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1) {
    if constexpr (cb == 0) {
        x0 *= 89226354u;
        x1 *= 89226354u;
        x0 += 64248484u;
        x1 += 64248484u;
        x0 = cb_lop3_6a(x0);
        x1 = cb_lop3_6a(x1);
        half2_uint32 xu0(x0);
        half2_uint32 xu1(x1);
        half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
        half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
        return __hadd2(d0, d1);
    }
    if constexpr (cb == 1) {
        x0 *= 0xCBAC1FEDu;
        x1 *= 0xCBAC1FEDu;
        return decode_mcg_product_2(x0, x1);
    }
    if constexpr (cb == 2) {
        x0 *= 0x83DCD12Du;
        x1 *= 0x83DCD12Du;
        return decode_mul1_product_2(x0, x1);
    }
}

}  // namespace gbx
