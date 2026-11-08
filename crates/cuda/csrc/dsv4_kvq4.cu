// DeepSeek V4.1 q4 KV cache: the store kernels (compress the latent into a 368-byte q4 row), the read kernels (window /
// sparse attention and the DSA indexer's scores, dequantising in place), and the 64-point Walsh-Hadamard rotation the
// layout needs. The design and its measurements are in src/dsv4_kvq4.rs.
//
// Layout of one q4 row of `dims` values whose last `rope_dims` stay fp16:
//
//     [ dims/2 nibble bytes ][ one E8M0 byte a group of 32 ][ pad to 16 ] [ rope_dims * 2 B ]
//
// `dims = 512, rope_dims = 64` gives 224 + 14 + 2 pad + 128 = 368 = 23 x 16, so every row is 16-byte aligned and the
// whole row is 23 vector loads. The nibble of dim `i` is byte `i / 2`, low half first; the sign is bit 3 and the
// magnitude is an E2M1 code, so `value = (-1)^s * E2M1[m] * 2^(scale - 127)`.
//
// Arithmetic contract (what src/dsv4_kvq4.rs computes on the host): the group amax is an f32 max, the scale is
// `2^ceil(log2(max(amax, 1e-4) / 6))`, rounding is nearest over the eight E2M1 magnitudes with ties to the lower code,
// and every dot product accumulates in f32 in index order.
#include "common.cuh"

namespace {

constexpr int THREADS = 128;

/// 2^(b - 127); 0xFF is NaN in E8M0.
__device__ __forceinline__ float e8m0v(uint8_t b) {
    return (b == 0xFFu) ? __int_as_float(0x7fc00000) : __int_as_float((uint32_t)b << 23);
}

/// The eight E2M1 magnitudes, `code * scale`.
__device__ __constant__ float E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

/// The magnitude of code `c` as arithmetic, not a table lookup.
///
/// `E2M1[c]` is a constant-memory load with a per-lane index, which the hardware serialises over the distinct indices
/// in a warp — eight of them here, sixteen times a key. E2M1 is `2^(exp-1) * (1 + man/2)` for the six normals and
/// `0, 0.5` for the two subnormals, so the whole table is one shift and two multiplies.
__device__ __forceinline__ float e2m1_mag(int c) {
    const float e = __int_as_float(((c >> 1) + 126) << 23);  // 2^(exp - 1)
    const float v = (1.0f + 0.5f * (float)(c & 1)) * e;
    return (c < 2) ? (0.5f * (float)c) : v;
}

/// Nearest E2M1 magnitude code to `q` (0..=7), ties to the lower code.
__device__ __forceinline__ int e2m1_code(float q) {
    const float a = fminf(fabsf(q), 6.0f);
    int best = 0;
    float bd = 1e30f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float d = fabsf(a - E2M1[i]);
        if (d < bd) {
            bd = d;
            best = i;
        }
    }
    return best;
}

/// The nibble for `q`: the magnitude code with the sign in bit 3.
__device__ __forceinline__ int e2m1_nibble(float q) {
    return e2m1_code(q) | (q < 0.0f ? 8 : 0);
}

/// The value a nibble holds at `scale` (exact IEEE-754 bit construction: (c + 252) << 22 for normals, 252 << 22 for 0.5).
__device__ __forceinline__ float nibble_value(int n, float scale) {
    const int c = n & 7;
    const uint32_t mag_bits = (c == 0) ? 0u : ((uint32_t)(c == 1 ? 252 : (c + 252)) << 22);
    const uint32_t signed_bits = mag_bits | ((uint32_t)(n & 8) << 28);
    return __uint_as_float(signed_bits) * scale;
}

/// The E8M0 byte for `amax`: `ceil(log2(max(amax, 1e-4) / 6)) + 127`, clamped to `[1, 254]` (byte 0 is a subnormal).
__device__ __forceinline__ int e8m0_byte(float amax) {
    const float a = fmaxf(amax, 1e-4f);
    const float e = ceilf(log2f(a / 6.0f));
    return (int)fminf(fmaxf(e + 127.0f, 1.0f), 254.0f);
}

__device__ __forceinline__ float ld16(const void* p, int dtype, size_t i) {
    if (dtype == 0) {
        return __half2float(static_cast<const __half*>(p)[i]);
    }
    return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

__device__ __forceinline__ void st16(void* p, int dtype, size_t i, float v) {
    if (dtype == 0) {
        static_cast<__half*>(p)[i] = __float2half_rn(v);
    } else {
        static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
    }
}

/// Bytes of one q4 row.
__device__ __host__ __forceinline__ int q4_row_bytes(int dims, int rope_dims) {
    const int nope = dims - rope_dims;
    const int packed = nope / 2 + (nope + 31) / 32;
    return ((packed + 15) / 16) * 16 + rope_dims * 2;
}

// ---------------------------------------------------------------------------------------------------- store

/// One block per row: amax a group of 32, E8M0, nearest E2M1, the RoPE tail straight through as fp16.
///
/// `ring_slots > 0` writes row `r` at `(row0 + r) % ring_slots` (the sliding-window ring); 0 writes `row0 + r` (the
/// compressed plane). `rope_dims == dims` is the indexer's key (all-quantised, no fp16 tail).
__global__ void q4_store_kernel(const void* src, int src_stride, uint8_t* dst, int row0, int dims, int rope_dims,
                                int ring_slots, int dtype) {
    extern __shared__ float scale[];  // one per group of the quantised part
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const int nope = dims - rope_dims;
    const int ng = (nope + 31) / 32;
    const int row_bytes = q4_row_bytes(dims, rope_dims);
    const int tail = row_bytes - rope_dims * 2;
    const int dst_row = (ring_slots > 0) ? ((row0 + r) % ring_slots) : (row0 + r);
    const void* s = static_cast<const uint8_t*>(src) + (size_t)r * src_stride;
    uint8_t* row = dst + (size_t)dst_row * row_bytes;

    for (int g = tid; g < ng; g += blockDim.x) {
        float amax = 0.0f;
        const int hi = min(nope, (g + 1) * 32);
        for (int i = g * 32; i < hi; ++i) {
            amax = fmaxf(amax, fabsf(ld16(s, dtype, i)));
        }
        const int b = e8m0_byte(amax);
        row[nope / 2 + g] = (uint8_t)b;
        scale[g] = e8m0v((uint8_t)b);
    }
    __syncthreads();
    // a whole byte a thread, so no two threads touch the same nibble
    for (int b = tid; b < nope / 2; b += blockDim.x) {
        const float sc = scale[b / 16];
        const int lo = e2m1_nibble(ld16(s, dtype, 2 * b) / sc);
        const int hi = e2m1_nibble(ld16(s, dtype, 2 * b + 1) / sc);
        row[b] = (uint8_t)(lo | (hi << 4));
    }
    for (int b = nope / 2 + ng + tid; b < tail; b += blockDim.x) {
        row[b] = 0;
    }
    for (int i = tid; i < rope_dims / 2; i += blockDim.x) {
        const __half2 h = __floats2half2_rn(ld16(s, dtype, nope + 2 * i), ld16(s, dtype, nope + 2 * i + 1));
        *reinterpret_cast<__half2*>(row + tail + 4 * i) = h;
    }
}

// ---------------------------------------------------------------------------------------------------- rotation

/// The orthonormal 64-point Walsh-Hadamard transform in place: one warp a 64-wide block, two values a lane.
/// `x[r][blk * block .. blk * block + dim]` is rotated for every `blk < sub`; `dim` must be a multiple of 64.
__global__ void walsh_kernel(void* x, int dtype, int stride, int dim, int block, int sub) {
    const int r = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk >= sub) {
        return;
    }
    const size_t base = (size_t)r * stride + (size_t)blk * block;
    // 64 values a pass, two a lane: 448 dims is seven passes
    if (dtype == 0) {
        __half2* x2 = reinterpret_cast<__half2*>(static_cast<__half*>(x) + base);
        if (dim == 448) {
            __half2 h[7];
#pragma unroll
            for (int t = 0; t < 7; ++t) {
                h[t] = x2[32 * t + lane];
            }
#pragma unroll
            for (int t = 0; t < 7; ++t) {
                const float2 ab = __half22float2(h[t]);
                float a = ab.x;
                float b = ab.y;
                {
                    const float p = a + b;
                    b = a - b;
                    a = p;
                }
#pragma unroll
                for (int k = 0; k < 5; ++k) {
                    const int d = 1 << k;
                    const float pa = __shfl_xor_sync(0xFFFFFFFFu, a, d);
                    const float pb = __shfl_xor_sync(0xFFFFFFFFu, b, d);
                    if ((lane >> k) & 1) {
                        a = pa - a;
                        b = pb - b;
                    } else {
                        a += pa;
                        b += pb;
                    }
                }
                h[t] = __floats2half2_rn(a * 0.125f, b * 0.125f);
            }
#pragma unroll
            for (int t = 0; t < 7; ++t) {
                x2[32 * t + lane] = h[t];
            }
            return;
        }
        for (int t = 0; t < dim / 64; ++t) {
            const float2 ab = __half22float2(x2[32 * t + lane]);
            float a = ab.x;
            float b = ab.y;
            {
                const float p = a + b;
                b = a - b;
                a = p;
            }
#pragma unroll
            for (int k = 0; k < 5; ++k) {
                const int d = 1 << k;
                const float pa = __shfl_xor_sync(0xFFFFFFFFu, a, d);
                const float pb = __shfl_xor_sync(0xFFFFFFFFu, b, d);
                if ((lane >> k) & 1) {
                    a = pa - a;
                    b = pb - b;
                } else {
                    a += pa;
                    b += pb;
                }
            }
            x2[32 * t + lane] = __floats2half2_rn(a * 0.125f, b * 0.125f);
        }
        return;
    }
    for (int t = 0; t < dim / 64; ++t) {
        const size_t o = base + (size_t)64 * t + 2 * lane;
        float a = ld16(x, dtype, o);
        float b = ld16(x, dtype, o + 1);
        // stride 1: the pair a lane holds
        {
            const float p = a + b;
            b = a - b;
            a = p;
        }
#pragma unroll
        for (int k = 0; k < 5; ++k) {
            const int d = 1 << k;
            const float pa = __shfl_xor_sync(0xFFFFFFFFu, a, d);
            const float pb = __shfl_xor_sync(0xFFFFFFFFu, b, d);
            if ((lane >> k) & 1) {
                // this lane holds the higher dimension of the pair: H[j + h] = a_j - a_{j+h}, partner - self
                a = pa - a;
                b = pb - b;
            } else {
                a += pa;
                b += pb;
            }
        }
        st16(x, dtype, o, a * 0.125f);
        st16(x, dtype, o + 1, b * 0.125f);
    }
}

// ---------------------------------------------------------------------------------------------------- attention

/// Sixteen values of a q4 row starting at dim `i0` (a multiple of 16, so one 8-byte load and one scale byte), and the
/// group's E8M0 scale.
__device__ __forceinline__ void q4_load16(const uint8_t* row, int i0, int nope, float* v, float tab_val) {
    const uint2 w = __ldg(reinterpret_cast<const uint2*>(row + (i0 >> 1)));
    const float s = e8m0v(__ldg(row + nope / 2 + (i0 >> 5)));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t word = (i < 8) ? w.x : w.y;
        const int nib = (word >> (4 * (i & 7))) & 0xF;
        v[i] = __shfl_sync(0x0FFFFFFFu, tab_val, nib) * s;
    }
}

/// Sixteen values of a q4 row's fp16 tail: chunk `k` of 16 dims at `row + tail + 32 * k`.
__device__ __forceinline__ void q4_rope16(const uint8_t* row, int tail, int k, float* v) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __half2 h = *reinterpret_cast<const __half2*>(row + tail + 32 * k + 4 * i);
        v[2 * i] = __low2float(h);
        v[2 * i + 1] = __high2float(h);
    }
}

__device__ __forceinline__ void f16_load16(const void* row, int i0, float* v) {
    const uint4* p4 = reinterpret_cast<const uint4*>(static_cast<const __half*>(row) + i0);
    const uint4 u0 = __ldg(p4 + 0);
    const uint4 u1 = __ldg(p4 + 1);
    const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

/// Online-softmax sink attention over the gathered rows, dequantising q4 planes on read.
///
/// **One warp per (row, head)**, 16 dims a lane: lane `l < nope/16` owns dims `16l .. 16l+15` of the nope part and the
/// lanes past it own the 16-dim chunks of the RoPE tail. Sixteen dims is exactly one 8-byte q4 load and one scale byte,
/// and it makes the whole key's dot product a five-shuffle warp reduction: the kernel takes **no `__syncthreads` in
/// the key loop at all**, which is what the block-per-head form this replaces spent its time on. Every lane keeps its
/// own running max, running sum and 16 accumulators, so the softmax needs no communication either.
///
/// `fmt_win` / `fmt_cmp` are 0 (fp16) or 1 (q4); a q4 plane's row stride is `win_row` / `cmp_row` bytes.
template <int FMT_WIN, int FMT_CMP, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    attn_q4_kernel(const void* __restrict__ q, const void* __restrict__ kv_win,
                   const void* __restrict__ kv_cmp, int split, const float* __restrict__ sink,
                   const int* __restrict__ idx, int idx_stride, int idx_len, void* __restrict__ out,
                   int n_heads, int head_dim, int nope, int win_row, int cmp_row, float scale, int dtype) {
    __shared__ float sm_m[WARPS];
    __shared__ float sm_l[WARPS];
    __shared__ float sm_acc[WARPS][256];

    const int r = blockIdx.x;
    const int h = blockIdx.y;
    if (h >= n_heads) {
        return;
    }
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nope_lanes = nope / 16;
    const bool is_nope = lane < nope_lanes;
    const int i0 = is_nope ? 16 * lane : nope + 16 * (lane - nope_lanes);

    const float tab_mag = e2m1_mag(lane & 7);
    const float tab_val = (lane & 8) ? -tab_mag : tab_mag;

    const size_t qbase = ((size_t)r * n_heads + h) * head_dim;
    float qv[16];
    f16_load16(q, (int)qbase + i0, qv);

    const int* row = idx + (size_t)r * idx_stride;
    const int len = min(idx_len, idx_stride);

    float m = (warp == 0) ? sink[h] : -INFINITY;
    float l = (warp == 0) ? 1.0f : 0.0f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        acc[i] = 0.0f;
    }

    const int tail = head_dim - nope;
    const int k = lane - nope_lanes;

    if constexpr (FMT_WIN == 0 && FMT_CMP == 0) {
#pragma unroll 2
        for (int j = warp; j < len; j += WARPS) {
            const int t = __ldg(row + j);
            if (t < 0) {
                continue;
            }
            float kv[16];
            if (t < split) {
                const uint8_t* p = static_cast<const uint8_t*>(kv_win) + (size_t)t * (head_dim * 2);
                f16_load16(p, i0, kv);
            } else {
                const uint8_t* p = static_cast<const uint8_t*>(kv_cmp) + (size_t)(t - split) * (head_dim * 2);
                f16_load16(p, i0, kv);
            }
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                d += qv[i] * kv[i];
            }
#pragma unroll
            for (int s = 16; s > 0; s >>= 1) {
                d += __shfl_xor_sync(0xFFFFFFFFu, d, s);
            }
            const float sc = d * scale;
            if (sc > m) {
                const float c = (l > 0.0f) ? __expf(m - sc) : 0.0f;
                m = sc;
                l = l * c + 1.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    acc[i] = acc[i] * c + kv[i];
                }
            } else {
                const float p = __expf(sc - m);
                l += p;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    acc[i] += p * kv[i];
                }
            }
        }
    } else {
#pragma unroll 2
        for (int j = warp; j < len; j += WARPS) {
            const int t = __ldg(row + j);
            if (t < 0) {
                continue;
            }
            float kv[16];
            if (t < split) {
                const uint8_t* p = static_cast<const uint8_t*>(kv_win) + (size_t)t * (FMT_WIN ? win_row : head_dim * 2);
                if constexpr (FMT_WIN == 1) {
                    if (is_nope) {
                        q4_load16(p, i0, nope, kv, tab_val);
                    } else {
                        q4_rope16(p, win_row - tail * 2, k, kv);
                    }
                } else {
                    f16_load16(p, i0, kv);
                }
            } else {
                const uint8_t* p =
                    static_cast<const uint8_t*>(kv_cmp) + (size_t)(t - split) * (FMT_CMP ? cmp_row : head_dim * 2);
                if constexpr (FMT_CMP == 1) {
                    if (is_nope) {
                        q4_load16(p, i0, nope, kv, tab_val);
                    } else {
                        q4_rope16(p, cmp_row - tail * 2, k, kv);
                    }
                } else {
                    f16_load16(p, i0, kv);
                }
            }
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                d += qv[i] * kv[i];
            }
#pragma unroll
            for (int s = 16; s > 0; s >>= 1) {
                d += __shfl_xor_sync(0xFFFFFFFFu, d, s);
            }
            const float sc = d * scale;
            if (sc > m) {
                const float c = (l > 0.0f) ? __expf(m - sc) : 0.0f;
                m = sc;
                l = l * c + 1.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    acc[i] = acc[i] * c + kv[i];
                }
            } else {
                const float p = __expf(sc - m);
                l += p;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    acc[i] += p * kv[i];
                }
            }
        }
    }

    if (lane == 0) {
        sm_m[warp] = m;
        sm_l[warp] = l;
    }
    __syncthreads();

    float m_max = sm_m[0];
#pragma unroll
    for (int w = 1; w < WARPS; ++w) {
        m_max = fmaxf(m_max, sm_m[w]);
    }
    float l_sum = 0.0f;
    float w_scale[WARPS];
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        const float sw = (sm_l[w] > 0.0f) ? __expf(sm_m[w] - m_max) : 0.0f;
        w_scale[w] = sw;
        l_sum += sm_l[w] * sw;
    }
    const float factor = (l_sum > 0.0f) ? (w_scale[warp] / l_sum) : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm_acc[warp][lane * 8 + i] = acc[i] * factor;
    }
    __syncthreads();
    for (int idx_out = tid; idx_out < 256; idx_out += WARPS * 32) {
        const int c_lane = idx_out >> 3;
        const int c_sub = idx_out & 7;
        const int out_idx = (c_lane < nope_lanes ? 16 * c_lane : nope + 16 * (c_lane - nope_lanes)) + c_sub;
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            s += sm_acc[w][idx_out];
        }
        static_cast<__half*>(out)[qbase + out_idx] = __float2half_rn(s);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm_acc[warp][lane * 8 + i] = acc[8 + i] * factor;
    }
    __syncthreads();
    for (int idx_out = tid; idx_out < 256; idx_out += WARPS * 32) {
        const int c_lane = idx_out >> 3;
        const int c_sub = idx_out & 7;
        const int out_idx = (c_lane < nope_lanes ? 16 * c_lane : nope + 16 * (c_lane - nope_lanes)) + 8 + c_sub;
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            s += sm_acc[w][idx_out];
        }
        static_cast<__half*>(out)[qbase + out_idx] = __float2half_rn(s);
    }
}


template <int FMT_WIN, int FMT_CMP, int WARPS, int H_TILE>
__global__ void __launch_bounds__(WARPS * 32)
    attn_q4_htile_kernel(const void* __restrict__ q, const void* __restrict__ kv_win,
                         const void* __restrict__ kv_cmp, int split, const float* __restrict__ sink,
                         const int* __restrict__ idx, int idx_stride, int idx_len, void* __restrict__ out,
                         int n_heads, int head_dim, int nope, int win_row, int cmp_row, float scale, int dtype) {
    (void)dtype;
    __shared__ float sm_q[H_TILE][16][32];
    __shared__ float sm_m[H_TILE][WARPS];
    __shared__ float sm_l[H_TILE][WARPS];

    const int r = blockIdx.x;
    const int h_base = blockIdx.y * H_TILE;
    if (h_base >= n_heads) {
        return;
    }
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nope_lanes = nope / 16;
    const bool is_nope = lane < nope_lanes;
    const int i0 = is_nope ? 16 * lane : nope + 16 * (lane - nope_lanes);

    const float tab_mag = e2m1_mag(lane & 7);
    const float tab_val = (lane & 8) ? -tab_mag : tab_mag;

    for (int hh = warp; hh < H_TILE; hh += WARPS) {
        if (h_base + hh < n_heads) {
            const size_t qbase = ((size_t)r * n_heads + (h_base + hh)) * head_dim;
            float qv[16];
            f16_load16(q, (int)qbase + i0, qv);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                sm_q[hh][i][lane] = qv[i];
            }
        }
    }
    __syncthreads();

    const int* row = idx + (size_t)r * idx_stride;
    const int len = min(idx_len, idx_stride);

    float m[H_TILE];
    float l[H_TILE];
    float acc[H_TILE][16];
#pragma unroll
    for (int hh = 0; hh < H_TILE; ++hh) {
        m[hh] = (warp == 0 && h_base + hh < n_heads) ? sink[h_base + hh] : -INFINITY;
        l[hh] = (warp == 0) ? 1.0f : 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            acc[hh][i] = 0.0f;
        }
    }

    const int tail = head_dim - nope;
    const int k = lane - nope_lanes;

#pragma unroll 1
    for (int j = warp; j < len; j += WARPS) {
        const int t = __ldg(row + j);
        if (t < 0) {
            continue;
        }
        float kv[16];
        if (t < split) {
            const uint8_t* p = static_cast<const uint8_t*>(kv_win) + (size_t)t * (FMT_WIN ? win_row : head_dim * 2);
            if constexpr (FMT_WIN == 1) {
                if (is_nope) {
                    q4_load16(p, i0, nope, kv, tab_val);
                } else {
                    q4_rope16(p, win_row - tail * 2, k, kv);
                }
            } else {
                f16_load16(p, i0, kv);
            }
        } else {
            const uint8_t* p =
                static_cast<const uint8_t*>(kv_cmp) + (size_t)(t - split) * (FMT_CMP ? cmp_row : head_dim * 2);
            if constexpr (FMT_CMP == 1) {
                if (is_nope) {
                    q4_load16(p, i0, nope, kv, tab_val);
                } else {
                    q4_rope16(p, cmp_row - tail * 2, k, kv);
                }
            } else {
                f16_load16(p, i0, kv);
            }
        }
        float d[H_TILE];
#pragma unroll
        for (int hh = 0; hh < H_TILE; ++hh) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                s += sm_q[hh][i][lane] * kv[i];
            }
            d[hh] = s;
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
#pragma unroll
            for (int hh = 0; hh < H_TILE; ++hh) {
                d[hh] += __shfl_xor_sync(0xFFFFFFFFu, d[hh], s);
            }
        }
#pragma unroll
        for (int hh = 0; hh < H_TILE; ++hh) {
            const float sc = d[hh] * scale;
            if (sc > m[hh]) {
                const float c = (l[hh] > 0.0f) ? __expf(m[hh] - sc) : 0.0f;
                m[hh] = sc;
                l[hh] = l[hh] * c + 1.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    acc[hh][i] = acc[hh][i] * c + kv[i];
                }
            } else {
                const float p = __expf(sc - m[hh]);
                l[hh] += p;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    acc[hh][i] += p * kv[i];
                }
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int hh = 0; hh < H_TILE; ++hh) {
            sm_m[hh][warp] = m[hh];
            sm_l[hh][warp] = l[hh];
        }
    }
    __syncthreads();

    float (*sm_acc)[256] = reinterpret_cast<float (*)[256]>(&sm_q[0][0][0]);

    for (int hh = 0; hh < H_TILE; ++hh) {
        if (h_base + hh >= n_heads) break;
        const size_t qbase = ((size_t)r * n_heads + (h_base + hh)) * head_dim;
        float m_max = sm_m[hh][0];
#pragma unroll
        for (int w = 1; w < WARPS; ++w) {
            m_max = fmaxf(m_max, sm_m[hh][w]);
        }
        float l_sum = 0.0f;
        float w_scale[WARPS];
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float sw = (sm_l[hh][w] > 0.0f) ? __expf(sm_m[hh][w] - m_max) : 0.0f;
            w_scale[w] = sw;
            l_sum += sm_l[hh][w] * sw;
        }
        const float factor = (l_sum > 0.0f) ? (w_scale[warp] / l_sum) : 0.0f;
        if (hh > 0) {
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sm_acc[warp][lane * 8 + i] = acc[hh][i] * factor;
        }
        __syncthreads();
        for (int idx_out = tid; idx_out < 256; idx_out += WARPS * 32) {
            const int c_lane = idx_out >> 3;
            const int c_sub = idx_out & 7;
            const int out_idx = (c_lane < nope_lanes ? 16 * c_lane : nope + 16 * (c_lane - nope_lanes)) + c_sub;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                s += sm_acc[w][idx_out];
            }
            static_cast<__half*>(out)[qbase + out_idx] = __float2half_rn(s);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sm_acc[warp][lane * 8 + i] = acc[hh][8 + i] * factor;
        }
        __syncthreads();
        for (int idx_out = tid; idx_out < 256; idx_out += WARPS * 32) {
            const int c_lane = idx_out >> 3;
            const int c_sub = idx_out & 7;
            const int out_idx = (c_lane < nope_lanes ? 16 * c_lane : nope + 16 * (c_lane - nope_lanes)) + 8 + c_sub;
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                s += sm_acc[w][idx_out];
            }
            static_cast<__half*>(out)[qbase + out_idx] = __float2half_rn(s);
        }
    }
}


__device__ __forceinline__ void mma16816_attn(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int IMM_BYTES>
__device__ __forceinline__ void ldsm4_attn_imm(uint32_t (&r)[4], uint32_t s_base) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4+%5];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s_base), "n"(IMM_BYTES));
}
template <int IMM_BYTES>
__device__ __forceinline__ void ldsm4t_attn_imm(uint32_t (&r)[4], uint32_t s_base) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4+%5];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s_base), "n"(IMM_BYTES));
}
__device__ __forceinline__ void ldsm4_attn_step(uint32_t (&r)[4], uint32_t s_base, int kt) {
    switch (kt) {
        case 0: ldsm4_attn_imm<0>(r, s_base); break;
        case 1: ldsm4_attn_imm<32>(r, s_base); break;
        case 2: ldsm4_attn_imm<64>(r, s_base); break;
        case 3: ldsm4_attn_imm<96>(r, s_base); break;
        case 4: ldsm4_attn_imm<128>(r, s_base); break;
        case 5: ldsm4_attn_imm<160>(r, s_base); break;
        case 6: ldsm4_attn_imm<192>(r, s_base); break;
        default: ldsm4_attn_imm<224>(r, s_base); break;
    }
}
__device__ __forceinline__ void ldsm4t_attn_step(uint32_t (&r)[4], uint32_t s_base, int kt) {
    switch (kt) {
        case 0: ldsm4t_attn_imm<0>(r, s_base); break;
        case 1: ldsm4t_attn_imm<32>(r, s_base); break;
        case 2: ldsm4t_attn_imm<64>(r, s_base); break;
        case 3: ldsm4t_attn_imm<96>(r, s_base); break;
        case 4: ldsm4t_attn_imm<128>(r, s_base); break;
        case 5: ldsm4t_attn_imm<160>(r, s_base); break;
        case 6: ldsm4t_attn_imm<192>(r, s_base); break;
        default: ldsm4t_attn_imm<224>(r, s_base); break;
    }
}
__device__ __forceinline__ void unpack_q4_lane16_to_u4(uint2 w, float s, float tab_val, uint4* dst4) {
    uint32_t h_words[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t word = (i < 4) ? w.x : w.y;
        const int nib0 = (word >> (8 * (i & 3))) & 0xF;
        const int nib1 = (word >> (8 * (i & 3) + 4)) & 0xF;
        const float v0 = __shfl_sync(0x0FFFFFFFu, tab_val, nib0) * s;
        const float v1 = __shfl_sync(0x0FFFFFFFu, tab_val, nib1) * s;
        const __half2 h2 = __floats2half2_rn(v0, v1);
        h_words[i] = *reinterpret_cast<const uint32_t*>(&h2);
    }
    dst4[0] = make_uint4(h_words[0], h_words[1], h_words[2], h_words[3]);
    dst4[1] = make_uint4(h_words[4], h_words[5], h_words[6], h_words[7]);
}

template <int FMT_WIN, int FMT_CMP, int H_GROUPS = 2>
__global__ void __launch_bounds__(H_GROUPS * 128, (H_GROUPS == 2) ? 2 : 1)
    attn_q4_tc_kernel(const void* __restrict__ q, const void* __restrict__ kv_win,
                      const void* __restrict__ kv_cmp, int split, const float* __restrict__ sink,
                      const int* __restrict__ idx, int idx_stride, int idx_len, void* __restrict__ out,
                      int n_heads, int head_dim, int nope, int win_row, int cmp_row, float scale, int dtype) {
    (void)head_dim;
    (void)nope;
    (void)dtype;
    constexpr int WARPS = H_GROUPS * 4;
    constexpr int THREADS = WARPS * 32;
    __shared__ __align__(16) __half sm_kv[16][520];
    __shared__ __align__(16) float sm_s[H_GROUPS][4][8][32];
    __shared__ __align__(16) uint4 sm_qa7[THREADS];

    const int r = blockIdx.x;
    const int h_cta = blockIdx.y * (H_GROUPS * 16);
    if (h_cta >= n_heads) {
        return;
    }
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int hg = warp >> 2;
    const int d_slice = warp & 3;
    const int h_base = h_cta + hg * 16;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int d_base = d_slice * 128;

    const bool is_nope = lane < 28;
    const int i0 = 16 * lane;
    const int k_rope = lane - 28;

    const float tab_mag = e2m1_mag(lane & 7);
    const float tab_val = (lane & 8) ? -tab_mag : tab_mag;

    // 1. Load Q [16 heads, 128 dims] into registers qa[7][4] + shared sm_qa7[tid]
    uint32_t qa[7][4];
    {
        const __half* q_lo = static_cast<const __half*>(q) + ((size_t)r * n_heads + (h_base + g)) * 512;
        const __half* q_hi = static_cast<const __half*>(q) + ((size_t)r * n_heads + (h_base + g + 8)) * 512;
#pragma unroll
        for (int kt = 0; kt < 7; ++kt) {
            const int col0 = d_base + kt * 16 + 2 * t;
            const int col2 = col0 + 8;
            qa[kt][0] = __ldg(reinterpret_cast<const uint32_t*>(q_lo + col0));
            qa[kt][1] = __ldg(reinterpret_cast<const uint32_t*>(q_hi + col0));
            qa[kt][2] = __ldg(reinterpret_cast<const uint32_t*>(q_lo + col2));
            qa[kt][3] = __ldg(reinterpret_cast<const uint32_t*>(q_hi + col2));
        }
        const int col0 = d_base + 7 * 16 + 2 * t;
        const int col2 = col0 + 8;
        sm_qa7[tid] = make_uint4(
            __ldg(reinterpret_cast<const uint32_t*>(q_lo + col0)),
            __ldg(reinterpret_cast<const uint32_t*>(q_hi + col0)),
            __ldg(reinterpret_cast<const uint32_t*>(q_lo + col2)),
            __ldg(reinterpret_cast<const uint32_t*>(q_hi + col2)));
    }

    float m_lo = sink[h_base + g];
    float l_lo = 1.0f;
    float m_hi = sink[h_base + g + 8];
    float l_hi = 1.0f;

    float o_acc[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        o_acc[nt][0] = 0.0f;
        o_acc[nt][1] = 0.0f;
        o_acc[nt][2] = 0.0f;
        o_acc[nt][3] = 0.0f;
    }

    const int* row = idx + (size_t)r * idx_stride;
    const int len = min(idx_len, idx_stride);
    const uint32_t group_mask = 0xFu << (g * 4);
    const uint32_t sm_kv_lane = static_cast<uint32_t>(__cvta_generic_to_shared(&sm_kv[lane & 15][d_base + ((lane >> 4) << 3)]));

    for (int j_base = 0; j_base < len; j_base += 16) {
        const int j_lane = j_base + (lane & 15);
        const int t_lane = (j_lane < len) ? __ldg(row + j_lane) : -1;
        const uint32_t valid_mask = __ballot_sync(0xFFFFFFFFu, t_lane >= 0) & 0xFFFFu;
        if (valid_mask == 0u) {
            continue;
        }
        // Step 1: Cooperatively load & dequantize up to 16 keys into sm_kv[16][520]
#pragma unroll
        for (int kk = warp; kk < 16; kk += WARPS) {
            const int t_idx = __shfl_sync(0xFFFFFFFFu, t_lane, kk);
            uint4* dst4 = reinterpret_cast<uint4*>(&sm_kv[kk][i0]);
            if (t_idx >= 0) {
                if (t_idx < split) {
                    const uint8_t* p = static_cast<const uint8_t*>(kv_win) + (size_t)t_idx * (FMT_WIN ? win_row : 1024);
                    if constexpr (FMT_WIN == 1) {
                        if (is_nope) {
                            const uint2 w = __ldg(reinterpret_cast<const uint2*>(p + (i0 >> 1)));
                            const float s = e8m0v(__ldg(p + 224 + (i0 >> 5)));
                            unpack_q4_lane16_to_u4(w, s, tab_val, dst4);
                        } else {
                            const uint4* src4 = reinterpret_cast<const uint4*>(p + (win_row - 128) + 32 * k_rope);
                            dst4[0] = __ldg(src4 + 0);
                            dst4[1] = __ldg(src4 + 1);
                        }
                    } else {
                        const uint4* src4 = reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(p) + i0);
                        dst4[0] = __ldg(src4 + 0);
                        dst4[1] = __ldg(src4 + 1);
                    }
                } else {
                    const uint8_t* p = static_cast<const uint8_t*>(kv_cmp) + (size_t)(t_idx - split) * (FMT_CMP ? cmp_row : 1024);
                    if constexpr (FMT_CMP == 1) {
                        if (is_nope) {
                            const uint2 w = __ldg(reinterpret_cast<const uint2*>(p + (i0 >> 1)));
                            const float s = e8m0v(__ldg(p + 224 + (i0 >> 5)));
                            unpack_q4_lane16_to_u4(w, s, tab_val, dst4);
                        } else {
                            const uint4* src4 = reinterpret_cast<const uint4*>(p + (cmp_row - 128) + 32 * k_rope);
                            dst4[0] = __ldg(src4 + 0);
                            dst4[1] = __ldg(src4 + 1);
                        }
                    } else {
                        const uint4* src4 = reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(p) + i0);
                        dst4[0] = __ldg(src4 + 0);
                        dst4[1] = __ldg(src4 + 1);
                    }
                }
            } else {
                dst4[0] = make_uint4(0u, 0u, 0u, 0u);
                dst4[1] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();

        // Step 2: Score GEMM S = Q * K^T ([16 heads, 16 keys])
        float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int kt = 0; kt < 7; ++kt) {
            uint32_t kb[4];
            ldsm4_attn_step(kb, sm_kv_lane, kt);
            const uint32_t b0[2] = {kb[0], kb[2]};
            const uint32_t b1[2] = {kb[1], kb[3]};
            mma16816_attn(s0, qa[kt], b0);
            mma16816_attn(s1, qa[kt], b1);
        }
        {
            uint32_t kb[4];
            ldsm4_attn_step(kb, sm_kv_lane, 7);
            const uint4 q7 = sm_qa7[tid];
            const uint32_t qa7[4] = {q7.x, q7.y, q7.z, q7.w};
            const uint32_t b0[2] = {kb[0], kb[2]};
            const uint32_t b1[2] = {kb[1], kb[3]};
            mma16816_attn(s0, qa7, b0);
            mma16816_attn(s1, qa7, b1);
        }
        sm_s[hg][d_slice][0][lane] = s0[0];
        sm_s[hg][d_slice][1][lane] = s0[1];
        sm_s[hg][d_slice][2][lane] = s0[2];
        sm_s[hg][d_slice][3][lane] = s0[3];
        sm_s[hg][d_slice][4][lane] = s1[0];
        sm_s[hg][d_slice][5][lane] = s1[1];
        sm_s[hg][d_slice][6][lane] = s1[2];
        sm_s[hg][d_slice][7][lane] = s1[3];
        __syncthreads();

        float sc_lo[4], sc_hi[4];
        const int k0 = 2 * t;
        const int k1 = 2 * t + 1;
        const int k2 = 8 + 2 * t;
        const int k3 = 8 + 2 * t + 1;
        const bool v0 = (valid_mask & (1u << k0)) != 0u;
        const bool v1 = (valid_mask & (1u << k1)) != 0u;
        const bool v2 = (valid_mask & (1u << k2)) != 0u;
        const bool v3 = (valid_mask & (1u << k3)) != 0u;
        sc_lo[0] = v0 ? ((sm_s[hg][0][0][lane] + sm_s[hg][1][0][lane] + sm_s[hg][2][0][lane] + sm_s[hg][3][0][lane]) * scale) : -INFINITY;
        sc_lo[1] = v1 ? ((sm_s[hg][0][1][lane] + sm_s[hg][1][1][lane] + sm_s[hg][2][1][lane] + sm_s[hg][3][1][lane]) * scale) : -INFINITY;
        sc_lo[2] = v2 ? ((sm_s[hg][0][4][lane] + sm_s[hg][1][4][lane] + sm_s[hg][2][4][lane] + sm_s[hg][3][4][lane]) * scale) : -INFINITY;
        sc_lo[3] = v3 ? ((sm_s[hg][0][5][lane] + sm_s[hg][1][5][lane] + sm_s[hg][2][5][lane] + sm_s[hg][3][5][lane]) * scale) : -INFINITY;

        sc_hi[0] = v0 ? ((sm_s[hg][0][2][lane] + sm_s[hg][1][2][lane] + sm_s[hg][2][2][lane] + sm_s[hg][3][2][lane]) * scale) : -INFINITY;
        sc_hi[1] = v1 ? ((sm_s[hg][0][3][lane] + sm_s[hg][1][3][lane] + sm_s[hg][2][3][lane] + sm_s[hg][3][3][lane]) * scale) : -INFINITY;
        sc_hi[2] = v2 ? ((sm_s[hg][0][6][lane] + sm_s[hg][1][6][lane] + sm_s[hg][2][6][lane] + sm_s[hg][3][6][lane]) * scale) : -INFINITY;
        sc_hi[3] = v3 ? ((sm_s[hg][0][7][lane] + sm_s[hg][1][7][lane] + sm_s[hg][2][7][lane] + sm_s[hg][3][7][lane]) * scale) : -INFINITY;

        // Step 3: Online softmax across the 16 keys (4 threads t=0..3 per head g)
        float bmax_lo = fmaxf(fmaxf(sc_lo[0], sc_lo[1]), fmaxf(sc_lo[2], sc_lo[3]));
        float bmax_hi = fmaxf(fmaxf(sc_hi[0], sc_hi[1]), fmaxf(sc_hi[2], sc_hi[3]));
        bmax_lo = fmaxf(bmax_lo, __shfl_xor_sync(group_mask, bmax_lo, 1));
        bmax_lo = fmaxf(bmax_lo, __shfl_xor_sync(group_mask, bmax_lo, 2));
        bmax_hi = fmaxf(bmax_hi, __shfl_xor_sync(group_mask, bmax_hi, 1));
        bmax_hi = fmaxf(bmax_hi, __shfl_xor_sync(group_mask, bmax_hi, 2));

        const float m_next_lo = fmaxf(m_lo, bmax_lo);
        const float m_next_hi = fmaxf(m_hi, bmax_hi);
        const float c_lo = __expf(m_lo - m_next_lo);
        const float c_hi = __expf(m_hi - m_next_hi);
        m_lo = m_next_lo;
        m_hi = m_next_hi;

        float p_lo[4], p_hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            p_lo[i] = (sc_lo[i] > -INFINITY) ? __expf(sc_lo[i] - m_lo) : 0.0f;
            p_hi[i] = (sc_hi[i] > -INFINITY) ? __expf(sc_hi[i] - m_hi) : 0.0f;
        }
        float psum_lo = (p_lo[0] + p_lo[1]) + (p_lo[2] + p_lo[3]);
        float psum_hi = (p_hi[0] + p_hi[1]) + (p_hi[2] + p_hi[3]);
        psum_lo += __shfl_xor_sync(group_mask, psum_lo, 1);
        psum_lo += __shfl_xor_sync(group_mask, psum_lo, 2);
        psum_hi += __shfl_xor_sync(group_mask, psum_hi, 1);
        psum_hi += __shfl_xor_sync(group_mask, psum_hi, 2);

        l_lo = l_lo * c_lo + psum_lo;
        l_hi = l_hi * c_hi + psum_hi;

#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            o_acc[nt][0] *= c_lo;
            o_acc[nt][1] *= c_lo;
            o_acc[nt][2] *= c_hi;
            o_acc[nt][3] *= c_hi;
        }

        const __half2 pa0 = __floats2half2_rn(p_lo[0], p_lo[1]);
        const __half2 pa1 = __floats2half2_rn(p_hi[0], p_hi[1]);
        const __half2 pa2 = __floats2half2_rn(p_lo[2], p_lo[3]);
        const __half2 pa3 = __floats2half2_rn(p_hi[2], p_hi[3]);
        const uint32_t pa[4] = {
            *reinterpret_cast<const uint32_t*>(&pa0),
            *reinterpret_cast<const uint32_t*>(&pa1),
            *reinterpret_cast<const uint32_t*>(&pa2),
            *reinterpret_cast<const uint32_t*>(&pa3),
        };

        // Step 4: Value GEMM O += P * K ([16 heads, 128 dims])
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            uint32_t v4[4];
            ldsm4t_attn_step(v4, sm_kv_lane, kt);
            const uint32_t vb0[2] = {v4[0], v4[1]};
            const uint32_t vb1[2] = {v4[2], v4[3]};
            mma16816_attn(o_acc[kt * 2 + 0], pa, vb0);
            mma16816_attn(o_acc[kt * 2 + 1], pa, vb1);
        }
        __syncthreads();
    }

    // Step 5: Normalize and write output
    const float inv_lo = (l_lo > 0.0f) ? (1.0f / l_lo) : 0.0f;
    const float inv_hi = (l_hi > 0.0f) ? (1.0f / l_hi) : 0.0f;
    __half* out_lo = static_cast<__half*>(out) + ((size_t)r * n_heads + (h_base + g)) * 512;
    __half* out_hi = static_cast<__half*>(out) + ((size_t)r * n_heads + (h_base + g + 8)) * 512;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int col = d_base + nt * 8 + 2 * t;
        *reinterpret_cast<__half2*>(out_lo + col) = __floats2half2_rn(o_acc[nt][0] * inv_lo, o_acc[nt][1] * inv_lo);
        *reinterpret_cast<__half2*>(out_hi + col) = __floats2half2_rn(o_acc[nt][2] * inv_hi, o_acc[nt][3] * inv_hi);
    }
}

// ---------------------------------------------------------------------------------------------------- indexer

/// `score[r][p] = sum_h relu(q[r][h] . k[p]) * w[r][h]` over q4 or fp16 index keys: the q4 form of
/// [`gb_dsv4_index_score`], unpacking a position's 128-wide key once per head out of L1.
template <int FMT>
__global__ void index_score_q4_kernel(const void* __restrict__ q, const void* __restrict__ index_k,
                                      const float* __restrict__ w, const uint8_t* __restrict__ mask,
                                      int mask_stride, float* __restrict__ score, int rows, int n_pos, int n_heads,
                                      int dim, float scale, int start_pos, int ratio, int key_row, int dtype) {
    if constexpr (FMT == 1) {
        if (dim == 128) {
            const int r = blockIdx.x;
            const int lane = threadIdx.x & 31;
            const int sub = lane & 3;
            const int p = blockIdx.y * (blockDim.x >> 2) + (threadIdx.x >> 2);
            if (p >= n_pos) {
                return;
            }
            const int len = (start_pos + r + 1) / ratio;
            if (p >= len || (mask != nullptr && mask[(size_t)r * mask_stride + p] == 0)) {
                if (sub == 0) {
                    score[(size_t)r * n_pos + p] = -INFINITY;
                }
                return;
            }
            // Note: all 4 threads (sub = 0..3) share the exact same `p`, `len`, and `mask[r * mask_stride + p]`,
            // so all 4 threads in the aligned 4-lane group (base_lane .. base_lane + 3) never diverge from each other!
            const uint8_t* kb = static_cast<const uint8_t*>(index_k) + (size_t)p * key_row;
            const int g = sub;
            const float s = e8m0v(__ldg(kb + 64 + g));
            const uint4 kb4 = __ldg(reinterpret_cast<const uint4*>(kb + g * 16));
            const uint32_t kw[4] = {kb4.x, kb4.y, kb4.z, kb4.w};
            float kv[32];
#pragma unroll
            for (int w_idx = 0; w_idx < 4; ++w_idx) {
                const uint32_t word = kw[w_idx];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t byte = (word >> (8 * b)) & 0xFFu;
                    kv[w_idx * 8 + 2 * b + 0] = nibble_value(byte & 0xFu, s);
                    kv[w_idx * 8 + 2 * b + 1] = nibble_value(byte >> 4, s);
                }
            }

            const uint32_t group_mask = 0xFu << (lane & ~3);
            float acc = 0.0f;
            for (int h = 0; h < n_heads; ++h) {
                const size_t qb = ((size_t)r * n_heads + h) * 128 + g * 32;
                float gd = 0.0f;
                if (dtype == 0) {
                    const uint4* qp4 = reinterpret_cast<const uint4*>(static_cast<const __half*>(q) + qb);
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const uint4 u = __ldg(qp4 + c);
                        const uint32_t qw[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&qw[j]));
                            gd += f2.x * kv[c * 8 + 2 * j + 0];
                            gd += f2.y * kv[c * 8 + 2 * j + 1];
                        }
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        gd += ld16(q, dtype, qb + i) * kv[i];
                    }
                }
                const int base_lane = lane & ~3;
                const float gd0 = __shfl_sync(group_mask, gd, base_lane + 0);
                const float gd1 = __shfl_sync(group_mask, gd, base_lane + 1);
                const float gd2 = __shfl_sync(group_mask, gd, base_lane + 2);
                const float gd3 = __shfl_sync(group_mask, gd, base_lane + 3);
                if (sub == 0) {
                    const float d = ((gd0 + gd1) + gd2) + gd3;
                    acc += fmaxf(d, 0.0f) * w[(size_t)r * n_heads + h];
                }
            }
            if (sub == 0) {
                score[(size_t)r * n_pos + p] = acc * scale;
            }
            return;
        }
    }

    const int r = blockIdx.x;
    const int p = blockIdx.y * blockDim.x + threadIdx.x;
    if (p >= n_pos) {
        return;
    }
    const int len = (start_pos + r + 1) / ratio;
    if (p >= len || (mask != nullptr && mask[(size_t)r * mask_stride + p] == 0)) {
        score[(size_t)r * n_pos + p] = -INFINITY;
        return;
    }
    const uint8_t* kb = static_cast<const uint8_t*>(index_k) + (size_t)p * (FMT ? key_row : dim * 2);
    float acc = 0.0f;
    for (int h = 0; h < n_heads; ++h) {
        const size_t qb = ((size_t)r * n_heads + h) * dim;
        float d = 0.0f;
        if (FMT) {
            for (int g = 0; g < dim / 32; ++g) {
                const float s = e8m0v(kb[dim / 2 + g]);
                float gd = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    const uint8_t byte = kb[g * 16 + i];
                    gd += ld16(q, dtype, qb + g * 16 * 2 + 2 * i) * nibble_value(byte & 0xF, s);
                    gd += ld16(q, dtype, qb + g * 16 * 2 + 2 * i + 1) * nibble_value(byte >> 4, s);
                }
                d += gd;
            }
        } else {
            for (int i = 0; i < dim; ++i) {
                d += ld16(q, dtype, qb + i) * ld16(kb, dtype, i);
            }
        }
        acc += fmaxf(d, 0.0f) * w[(size_t)r * n_heads + h];
    }
    score[(size_t)r * n_pos + p] = acc * scale;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- launchers

GB_EXPORT int gb_dsv4_kvq4_store(const void* src, int src_row0, int src_stride, void* dst, int rows, int row0,
                                 int dims, int rope_dims, int ring_slots, int dtype, void* stream) {
    if (rows <= 0 || dims <= 0 || rope_dims < 0 || rope_dims >= dims || (dims - rope_dims) % 32 != 0 ||
        rope_dims % 2 != 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int ng = (dims - rope_dims + 31) / 32;
    const int row_bytes = q4_row_bytes(dims, rope_dims);
    const uint8_t* s = static_cast<const uint8_t*>(src) + (size_t)src_row0 * src_stride;
    q4_store_kernel<<<rows, THREADS, (size_t)ng * sizeof(float), gb_stream(stream)>>>(
        s, src_stride, static_cast<uint8_t*>(dst), row0, dims, rope_dims, ring_slots, dtype);
    (void)row_bytes;
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_kvq4_rot(void* x, int dtype, int rows, int stride, int dim, int block, int sub, void* stream) {
    if (rows <= 0 || dim <= 0 || dim % 64 != 0 || sub <= 0 || block < dim) {
        return (int)cudaErrorInvalidValue;
    }
    const int warps = (sub >= 8) ? 8 : 4;
    const int gx = (sub + warps - 1) / warps;
    walsh_kernel<<<dim3(gx, rows), warps * 32, 0, gb_stream(stream)>>>(x, dtype, stride, dim, block, sub);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_kvq4_attn(const void* q, const void* kv_win, const void* kv_cmp, int split, const float* sink,
                                const int* idx, int idx_stride, int idx_len, void* out, int rows, int n_heads,
                                int head_dim, int fmt_win, int fmt_cmp, int win_row, int cmp_row, float scale,
                                int dtype, void* stream) {
    if (rows <= 0 || n_heads <= 0 || head_dim <= 0 || head_dim < 128) {
        return (int)cudaErrorInvalidValue;
    }
    const int nope = head_dim - 64;
    if (nope % 16 != 0 || nope / 16 + 4 != 32) {
        return (int)cudaErrorInvalidValue;  // 16 dims a lane, 32 lanes: 448 nope dims and a 64-dim tail
    }
    const int threads = 256;
    if ((n_heads % 64) == 0 && head_dim == 512 && nope == 448 && dtype == 0 && rows >= 16) {
        dim3 grid(rows, n_heads / 64);
        if (fmt_win == 0 && fmt_cmp == 0) {
            attn_q4_tc_kernel<0, 0, 4><<<grid, 512, 0, gb_stream(stream)>>>(
                q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
                scale, dtype);
        } else if (fmt_win == 0 && fmt_cmp == 1) {
            attn_q4_tc_kernel<0, 1, 4><<<grid, 512, 0, gb_stream(stream)>>>(
                q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
                scale, dtype);
        } else if (fmt_win == 1 && fmt_cmp == 1) {
            attn_q4_tc_kernel<1, 1, 4><<<grid, 512, 0, gb_stream(stream)>>>(
                q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
                scale, dtype);
        } else {
            attn_q4_tc_kernel<1, 0, 4><<<grid, 512, 0, gb_stream(stream)>>>(
                q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
                scale, dtype);
        }
        return GB_CHECK_LAUNCH();
    }
    if ((n_heads % 32) == 0 && head_dim == 512 && nope == 448 && dtype == 0) {
        dim3 grid(rows, n_heads / 32);
        if (fmt_win == 0 && fmt_cmp == 0) {
            attn_q4_tc_kernel<0, 0, 2><<<grid, 256, 0, gb_stream(stream)>>>(
                q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
                scale, dtype);
        } else if (fmt_win == 0 && fmt_cmp == 1) {
            attn_q4_tc_kernel<0, 1, 2><<<grid, 256, 0, gb_stream(stream)>>>(
                q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
                scale, dtype);
        } else if (fmt_win == 1 && fmt_cmp == 1) {
            attn_q4_tc_kernel<1, 1, 2><<<grid, 256, 0, gb_stream(stream)>>>(
                q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
                scale, dtype);
        } else {
            attn_q4_tc_kernel<1, 0, 2><<<grid, 256, 0, gb_stream(stream)>>>(
                q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
                scale, dtype);
        }
        return GB_CHECK_LAUNCH();
    }
    dim3 grid(rows, n_heads);
    if (fmt_win == 0 && fmt_cmp == 0) {
        attn_q4_kernel<0, 0, 8><<<grid, threads, 0, gb_stream(stream)>>>(
            q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
            scale, dtype);
    } else if (fmt_win == 0 && fmt_cmp == 1) {
        attn_q4_kernel<0, 1, 8><<<grid, threads, 0, gb_stream(stream)>>>(
            q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
            scale, dtype);
    } else if (fmt_win == 1 && fmt_cmp == 1) {
        attn_q4_kernel<1, 1, 8><<<grid, threads, 0, gb_stream(stream)>>>(
            q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
            scale, dtype);
    } else {
        attn_q4_kernel<1, 0, 8><<<grid, threads, 0, gb_stream(stream)>>>(
            q, kv_win, kv_cmp, split, sink, idx, idx_stride, idx_len, out, n_heads, head_dim, nope, win_row, cmp_row,
            scale, dtype);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_kvq4_index_score(const void* q, const void* index_k, const float* w, const uint8_t* mask,
                                       int mask_stride, float* score, int rows, int n_pos, int n_heads, int dim,
                                       float scale, int start_pos, int ratio, int fmt, int key_row, int dtype,
                                       void* stream) {
    if (rows <= 0 || n_pos <= 0 || n_heads <= 0 || dim <= 0 || dim % 32 != 0 || ratio <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const dim3 grid(rows, (fmt == 1 && dim == 128) ? ((n_pos + 31) / 32) : ((n_pos + 127) / 128));
    if (fmt == 1) {
        index_score_q4_kernel<1><<<grid, 128, 0, gb_stream(stream)>>>(
            q, index_k, w, mask, mask_stride, score, rows, n_pos, n_heads, dim, scale, start_pos, ratio, key_row, dtype);
    } else {
        index_score_q4_kernel<0><<<grid, 128, 0, gb_stream(stream)>>>(
            q, index_k, w, mask, mask_stride, score, rows, n_pos, n_heads, dim, scale, start_pos, ratio, key_row, dtype);
    }
    return GB_CHECK_LAUNCH();
}