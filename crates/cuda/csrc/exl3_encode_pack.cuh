// Trellis bit-packing and unpacking: pack.cu and frac.cu from ExLlamaV3 (MIT), ported verbatim with no ATen
// headers. Integer rates: 256 K-bit indices -> 256 * K / 16 uint16_t (16 * K) per tile, MSB-first 32-bit
// words, the last word byte-swapped (SWAP16) so the stream reads back big-endian. Fractional rates: a tile is
// 16 * K ring bits = bpb = 8 * k2 uint16_t (whole 32-bit words), weight i's D(i) = KA + bit(i mod 16) of MASK
// new bits at ring positions; the same stream convention.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace gbx {

// ((x >> 16) | (x << 16)): __byte_perm(x, 0, 0x1032), the low and high 16-bit halves swapped.
#define GBSWAP16(x) (((x) >> 16) | ((x) << 16))

// Largest fractional tile in 32-bit words: KA = 7 with every extra bit set, 16 * 8 bits per 16 weights
#define FRAC_MAX_WORDS 64

__device__ __forceinline__ int frac_d(int i, int ka, uint32_t mask) { return ka + ((mask >> (i & 15)) & 1); }

// Ring position where window i ends, S(i)
__device__ __forceinline__ int frac_s(int i, int ka, uint32_t mask, int bpb)
{
    int s = (i >> 4) * bpb;
    for (int j = 0; j <= (i & 15); ++j) s += frac_d(j, ka, mask);
    return s;
}

// 16-bit window starting at ring bit `start` (mod R), from MSB-first 32-bit words
__device__ __forceinline__ uint32_t frac_window(const uint32_t* words, int nw, int start)
{
    const int R = nw * 32;
    start %= R; if (start < 0) start += R;
    const int wi = start >> 5, o = start & 31;
    const uint64_t v = ((uint64_t) words[wi] << 32) | (uint64_t) words[(wi + 1) % nw];
    return (uint32_t) ((v >> (48 - o)) & 0xFFFFu);
}

template <int K>
__global__ __launch_bounds__(128)
void pack_trellis_kernel
(
    uint16_t* __restrict__ g_packed,
    const uint16_t* __restrict__ g_unpacked
)
{
    constexpr int packed_size = 256 * K / 16;
    __shared__ uint16_t s_unpacked[256];
    __shared__ uint16_t s_packed[packed_size];

    int t = threadIdx.x;
    g_packed += (gridDim.x * blockIdx.y + blockIdx.x) * packed_size;
    g_unpacked += (gridDim.x * blockIdx.y + blockIdx.x) * 256;

    ((uint32_t*) s_unpacked)[t] = ((uint32_t*) g_unpacked)[t];
    __syncthreads();

    // 16 spans of 16 weights to guarantee alignment for any K
    const int spans = 16;
    const int len = 256 / spans;
    if (t < spans)
    {
        int i = len * t;
        int j = K * t;
        int k = 32;
        uint32_t buf = 0;
        for (int n = 0; n < len; ++n)
        {
            uint32_t v = (uint32_t) s_unpacked[i];
            v &= ((1 << K) - 1);
            k -= K;
            buf |= (v << k);
            if (k <= 16)
            {
                s_packed[j] = (uint16_t) (buf >> 16);
                buf <<= 16;
                k += 16;
                j++;
            }
            i++;
        }
    }
    __syncthreads();

    if (t < packed_size / 2)
        ((uint32_t*) g_packed)[t] = GBSWAP16(((uint32_t*) s_packed)[t]);
}

template <int K>
__global__ __launch_bounds__(128)
void unpack_trellis_kernel
(
    uint16_t* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed
)
{
    constexpr int packed_size = 256 * K / 16;
    __shared__ uint16_t s_packed[packed_size];

    int t = threadIdx.x;
    g_packed += (gridDim.x * blockIdx.y + blockIdx.x) * packed_size;
    g_unpacked += (gridDim.x * blockIdx.y + blockIdx.x) * 256;

    // Read packed tile
    if (t < packed_size / 2)
        ((uint32_t*) s_packed)[t] = ((uint32_t*) g_packed)[t];
    __syncthreads();

    // Index two words
    int b0 = t * 2 * K + K - 16 + 256 * K;          // start of word0
    int b1 = b0 + K;                                // start of word1
    int b2 = b1 + 16;                               // end of word1
    int i0 = b0 / 32;                               // uint32 containing first bit of word0
    int i1 = (b2 - 1) / 32;                         // uint32 containing last bit of word1, may be == i0
    int s1 = (i1 + 1) * 32 - b2;                    // shift to align word1 to 32-bit boundary

    // Load 32-64 bits containing word0 and word1, overlapping by 16-K bits, correct for endianness
    uint32_t a = ((uint32_t*) s_packed)[i0 % (K * 256 / 32)];
    uint32_t b = ((uint32_t*) s_packed)[i1 % (K * 256 / 32)];

    // Shift into place
    uint32_t w1 = __funnelshift_r(b, a, s1);
    uint32_t w0 = w1 >> K;
    w0 &= 0xffff;
    w1 &= 0xffff;

    // Store
    uint32_t word01 = (w1 << 16) | w0;
    ((uint32_t*) g_unpacked)[t] = word01;
}

__global__ __launch_bounds__(128)
void pack_trellis_frac_kernel(uint16_t* __restrict__ g_packed, const uint16_t* __restrict__ g_unpacked,
                              int num_tiles, int ka, uint32_t mask, int bpb)
{
    const int tile = blockIdx.x * 128 + threadIdx.x;
    if (tile >= num_tiles) return;
    const int nw = bpb / 2;
    uint32_t words[FRAC_MAX_WORDS];
    for (int w = 0; w < nw; ++w) words[w] = 0;
    const uint16_t* idx = g_unpacked + (size_t) tile * 256;
    int pos = 0;
    for (int i = 0; i < 256; ++i)
    {
        const int d = frac_d(i, ka, mask);
        const uint32_t v = (uint32_t) idx[i] & ((1u << d) - 1);
        for (int b = d - 1; b >= 0; --b, ++pos)
            if ((v >> b) & 1) words[pos >> 5] |= 1u << (31 - (pos & 31));
    }
    uint32_t* out = (uint32_t*) (g_packed + (size_t) tile * bpb);
    for (int w = 0; w < nw; ++w) out[w] = words[w];
}

__global__ __launch_bounds__(32)
void unpack_trellis_frac_kernel(uint16_t* __restrict__ g_unpacked, const uint16_t* __restrict__ g_packed,
                                int ka, uint32_t mask, int bpb)
{
    __shared__ uint32_t words[FRAC_MAX_WORDS];
    const int tile = blockIdx.x;
    const int t = threadIdx.x;
    const int nw = bpb / 2;
    for (int w = t; w < nw; w += 32) words[w] = ((const uint32_t*) (g_packed + (size_t) tile * bpb))[w];
    __syncwarp();
    for (int j = 0; j < 8; ++j)
    {
        const int i = t * 8 + j;
        g_unpacked[(size_t) tile * 256 + i] = (uint16_t) frac_window(words, nw, frac_s(i, ka, mask, bpb) - 16);
    }
}

#undef GBSWAP16
#undef FRAC_MAX_WORDS

}  // namespace gbx
