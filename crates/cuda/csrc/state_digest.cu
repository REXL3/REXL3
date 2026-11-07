// A content digest of device memory: what a rank's prefix-cache state was when it was snapshotted, compared with what it holds after
// a restore (`src/state_digest.rs` has the contract and the CPU twin the tests compare with).
//
// The digest of one range is a position-keyed SUM over its 4-byte words: word `i` of range `k` contributes
// mix64(mix64(seed_k + i) ^ word), added modulo 2^64. Integer addition is associative and commutative, so the result does not depend on
// how the words are spread over blocks or in which order the atomics land: the digest is deterministic. `mix64` is a bijection, so two
// different words at the same position always contribute differently, and the position key makes a word moved to another position
// contribute differently too. One pass, memory bound.
#include "common.cuh"
#include <stdint.h>

__device__ __forceinline__ unsigned long long sd_mix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// `w` holds `words` 4-byte words; the `tail_len` (< 4) bytes after them are word `words`, little endian, tagged with their count.
__global__ void state_digest_kernel(const uint32_t* __restrict__ w, unsigned long long words, const uint8_t* __restrict__ tail,
                                    int tail_len, unsigned long long seed, unsigned long long* out) {
    unsigned long long acc = 0;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    // four words a thread an iteration (16-byte loads where the range allows), the rest one by one
    const unsigned long long quads = words / 4;
    const bool vec = (((uintptr_t)w) & 15) == 0;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (vec) {
        const uint4* q = (const uint4*)w;
        for (unsigned long long j = i; j < quads; j += stride) {
            const uint4 v = q[j];
            const unsigned long long b = seed + 4 * j;
            acc += sd_mix64(sd_mix64(b) ^ (unsigned long long)v.x);
            acc += sd_mix64(sd_mix64(b + 1) ^ (unsigned long long)v.y);
            acc += sd_mix64(sd_mix64(b + 2) ^ (unsigned long long)v.z);
            acc += sd_mix64(sd_mix64(b + 3) ^ (unsigned long long)v.w);
        }
        for (unsigned long long j = 4 * quads + i; j < words; j += stride) {
            acc += sd_mix64(sd_mix64(seed + j) ^ (unsigned long long)w[j]);
        }
    } else {
        for (unsigned long long j = i; j < words; j += stride) {
            acc += sd_mix64(sd_mix64(seed + j) ^ (unsigned long long)w[j]);
        }
    }
    if (tail_len > 0 && i == 0) {
        unsigned long long t = 0;
        for (int k = 0; k < tail_len; ++k) t |= (unsigned long long)tail[k] << (8 * k);
        acc += sd_mix64(sd_mix64(seed + words) ^ t ^ ((unsigned long long)tail_len << 32));
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0 && acc != 0) atomicAdd(out, acc);
}

// Adds the digest of `bytes` bytes at `p` (4-byte aligned) under `seed` to the 64-bit sum at `out`, on `stream`.
GB_EXPORT int gb_state_digest_accumulate(const void* p, unsigned long long bytes, unsigned long long seed, void* out, void* stream) {
    if (bytes == 0) return 0;
    if (((uintptr_t)p) & 3) return (int)cudaErrorMisalignedAddress;
    const unsigned long long words = bytes / 4;
    const int tail_len = (int)(bytes % 4);
    const uint8_t* tail = (const uint8_t*)p + words * 4;
    const int threads = 256;
    const unsigned long long want = (words / 4 + threads - 1) / threads;
    const int blocks = (int)(want < 1 ? 1 : (want > 1024 ? 1024 : want));
    state_digest_kernel<<<blocks, threads, 0, gb_stream(stream)>>>((const uint32_t*)p, words, tail, tail_len, seed,
                                                                    (unsigned long long*)out);
    return GB_CHECK_LAUNCH();
}
