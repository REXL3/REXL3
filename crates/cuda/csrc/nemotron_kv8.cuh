// The int8 KV page of the Nemotron 3.5 Super VL attention layers (`--kv-dtype int8`, docs/NEMOTRON_KV_INT8.md), shared by the cache
// write (nemotron_attn.cu), the tree and the flash path (nemotron_attn_prefill.cu).
//
// Page layout (one layer's K, or its V; 256 tokens): int8 values `[256 tokens][2 kv heads][128]` (65,536 bytes), then one signed exponent
// byte per (token, kv head) `[256][2]` (512 bytes): 66,048 bytes a page against 131,072 for fp16. A value is `q * 2^e`, q in [-127, 127].
// The scale is per (token, kv head) and a power of two:
//   * per token: a row's bytes depend on its own K (or V) only, so a token written by a decode step, a verify window or any prefill slice
//     is the same bytes (decode == prefill, chunked == whole, cached == uncached are kept); a scale per block of tokens is not (a block
//     filled one token at a time would requantise or depend on the slicing);
//   * per kv head: a TP rank that holds one kv head needs no other rank's maximum;
//   * a power of two: `q * 2^e` is EXACT in fp16 for every e in [I8_EMIN, I8_EMAX] (7 significant bits; 1 x 2^-24 is fp16's smallest
//     subnormal, 127 x 2^8 = 32,512 < 65,504), so the kernels dequantise into the very fp16 tiles they always used and everything
//     after the load is the fp16 arithmetic, bit for bit.
// Quantiser: e is the smallest exponent with amax(row of the head) / 2^e <= 127 (127 = 1.984375 x 2^6), clamped; q = rint(x * 2^-e)
// (nearest-even; x * 2^-e is exact), saturated to +-127. Deterministic: the maximum does not depend on the reduction order. A row of
// zeros gets e = I8_EMIN and zero bytes. (An E4M3 fp8 format on the same page was evaluated and rejected: docs/NEMOTRON_KV_INT8.md.)
#pragma once
#include <stdint.h>

namespace kv8 {

constexpr int PAGE_TOKENS = 256, NKV = 2, D = 128, ROW = NKV * D;
constexpr int DATA_BYTES = PAGE_TOKENS * ROW;                 // 65,536
constexpr int EXP_OFF = DATA_BYTES;                           // the exponents follow the values
constexpr int PAGE_BYTES = DATA_BYTES + PAGE_TOKENS * NKV;    // 66,048 (a multiple of 16)
constexpr int I8_EMIN = -24, I8_EMAX = 8;
static_assert(PAGE_BYTES % 16 == 0, "pages stay 16-byte aligned for cp.async");

__device__ __forceinline__ float pow2(int e) { return __int_as_float((127 + e) << 23); }

__device__ __forceinline__ int exponent_i8(float amax) {
    const uint32_t b = __float_as_uint(amax);
    const int E = (int)((b >> 23) & 0xffu) - 127;
    const int e = ((b & 0x7fffffu) <= 0x7e0000u) ? E - 6 : E - 5;
    return min(max(e, I8_EMIN), I8_EMAX);
}

__device__ __forceinline__ uint8_t quant_i8(float x, int e) {
    const float q = fminf(fmaxf(rintf(x * pow2(-e)), -127.f), 127.f);
    return (uint8_t)(int8_t)(int)q;
}

// Two int8 bytes (low byte first) times 2^e as an fp16 pair (low half first): exact.
__device__ __forceinline__ uint32_t dq2_i8(uint32_t two, float scale) {
    const float a = (float)(int8_t)(two & 0xffu), b = (float)(int8_t)((two >> 8) & 0xffu);
    const __half2 h = __floats2half2_rn(a * scale, b * scale);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Eight int8 bytes as eight fp16 values (16 bytes), times 2^e.
__device__ __forceinline__ uint4 dq8_i8(uint2 raw, int e) {
    const float s = pow2(e);
    return make_uint4(dq2_i8(raw.x, s), dq2_i8(raw.x >> 16, s), dq2_i8(raw.y, s), dq2_i8(raw.y >> 16, s));
}

}  // namespace kv8
