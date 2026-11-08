#include "common.cuh"
#include "dsv4_round_params.cuh"
#include <cstdlib>  // REXL3 field patch P1: getenv
#include <cuda_fp8.h>

namespace {

constexpr int THREADS = 256;

/// 0 = f16, 1 = bf16 (the ABI codes of `exl3_linear::Dtype`). This file reads and writes f16; bf16 goes through the
/// same code path with the other conversion.
__device__ __forceinline__ float ld(const void* p, int dtype, size_t i) {
    if (dtype == 0) {
        return __half2float(static_cast<const __half*>(p)[i]);
    }
    return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

__device__ __forceinline__ void st(void* p, int dtype, size_t i, float v) {
    if (dtype == 0) {
        static_cast<__half*>(p)[i] = __float2half_rn(v);
    } else {
        static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
    }
}

/// One E4M3 byte as f32 (the hardware conversion; ±0, subnormals and 448 all exact).
__device__ __forceinline__ float e4m3(uint8_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

/// 2^(b - 127); 0xFF is NaN in E8M0.
__device__ __forceinline__ float e8m0(uint8_t b) {
    return (b == 0xFFu) ? __int_as_float(0x7fc00000) : ldexpf(1.0f, (int)b - 127);
}

/// Nearest E2M1 value: {0, .5, 1, 1.5, 2, 3, 4, 6} with the sign kept.
__device__ __forceinline__ float round_e2m1(float q) {
    const float a = fminf(fabsf(q), 6.0f);
    float best = 0.0f, bd = 1e30f;
    const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float d = fabsf(a - lut[i]);
        if (d < bd) {
            bd = d;
            best = lut[i];
        }
    }
    return q < 0.0f ? -best : best;
}

/// Nearest E4M3 value (3 mantissa bits, 4-bit exponent; no subnormal rounding below 2^-9).
__device__ __forceinline__ float round_e4m3(float q) {
    const float a = fabsf(q);
    const int e = (__float_as_int(a) >> 23) - 127;
    const int p = (e < -6) ? -9 : ((e > 8) ? 5 : (e - 3));
    const float step = __int_as_float((p + 127) << 23);
    const float inv_step = __int_as_float((127 - p) << 23);
    float r = rintf(a * inv_step) * step;
    r = (a < 1e-30f) ? 0.0f : ((a >= 448.0f) ? 448.0f : r);
    return (q < 0.0f) ? -r : r;
}

// ---------------------------------------------------------------------------------------------------- RoPE

/// `x[r][stride - dim .. stride]` rotated at position `pos0 + r`, adjacent pairs as complex numbers.
__global__ void rope_kernel(void* x, int dtype, int stride, int dim, int block, int sub, const float* cos,
                            const float* sin, int table_stride, int pos0, int pos_step, int inverse) {
    const int r = blockIdx.x;
    const int blk = blockIdx.y;
    if (blk >= sub) return;
    const int pos = pos0 + r * pos_step;
    const float* c = cos + (size_t)pos * table_stride;
    const float* sn = sin + (size_t)pos * table_stride;
    const float sign = inverse ? -1.0f : 1.0f;
    const int half = dim / 2;
    const size_t base = (size_t)r * stride + (size_t)blk * block + (size_t)(block - dim);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = ld(x, dtype, base + 2 * i);
        const float b = ld(x, dtype, base + 2 * i + 1);
        const float ci = c[i];
        const float si = sign * sn[i];
        st(x, dtype, base + 2 * i, a * ci - b * si);
        st(x, dtype, base + 2 * i + 1, b * ci + a * si);
    }
}

// Device-params variant (DSV4.1 speculative round graphs): `pos0` comes from the round's `start_pos` plus a baked
// per-call-site offset, so a captured graph replays as `start_pos` advances.
__global__ void rope_params_kernel(void* x, int dtype, int stride, int dim, int block, int sub, const float* cos,
                                   const float* sin, int table_stride, int pos_off, int pos_step, int inverse,
                                   const Dsv4MtpRoundParams* p) {
    const int r = blockIdx.x;
    const int blk = blockIdx.y;
    if (blk >= sub) return;
    const int pos = p->start_pos + pos_off + r * pos_step;
    const float* c = cos + (size_t)pos * table_stride;
    const float* sn = sin + (size_t)pos * table_stride;
    const float sign = inverse ? -1.0f : 1.0f;
    const int half = dim / 2;
    const size_t base = (size_t)r * stride + (size_t)blk * block + (size_t)(block - dim);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = ld(x, dtype, base + 2 * i);
        const float b = ld(x, dtype, base + 2 * i + 1);
        const float ci = c[i];
        const float si = sign * sn[i];
        st(x, dtype, base + 2 * i, a * ci - b * si);
        st(x, dtype, base + 2 * i + 1, b * ci + a * si);
    }
}

__global__ void rmsnorm_f32_to_f16_kernel(const float* x, const float* w, void* y, int width, float eps) {
    __shared__ float warp_sum[THREADS / 32];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float* row = x + (size_t)r * width;
    if ((width & 3) == 0 && (((uintptr_t)row | (uintptr_t)w | (uintptr_t)y) & 15u) == 0) {
        const int vec = width >> 2;
        const float4* row4 = reinterpret_cast<const float4*>(row);
        const float4* w4 = reinterpret_cast<const float4*>(w);
        uint2* y2 = reinterpret_cast<uint2*>(static_cast<__half*>(y) + (size_t)r * width);
        float s = 0.0f;
        for (int i = tid; i < vec; i += blockDim.x) {
            const float4 v = row4[i];
            s = fmaf(v.x, v.x, s);
            s = fmaf(v.y, v.y, s);
            s = fmaf(v.z, v.z, s);
            s = fmaf(v.w, v.w, s);
        }
        for (int m = 16; m > 0; m >>= 1) {
            s += __shfl_xor_sync(0xFFFFFFFFu, s, m);
        }
        if (lane == 0) {
            warp_sum[warp] = s;
        }
        __syncthreads();
        if (warp == 0) {
            s = (lane < (THREADS / 32)) ? warp_sum[lane] : 0.0f;
            for (int m = (THREADS / 32) / 2; m > 0; m >>= 1) {
                s += __shfl_xor_sync(0xFFFFFFFFu, s, m);
            }
            if (lane == 0) {
                warp_sum[0] = rsqrtf(s / (float)width + eps);
            }
        }
        __syncthreads();
        const float inv = warp_sum[0];
        for (int i = tid; i < vec; i += blockDim.x) {
            const float4 v = row4[i];
            const float4 g = __ldg(w4 + i);
            const __half2 h0 = __floats2half2_rn(v.x * inv * g.x, v.y * inv * g.y);
            const __half2 h1 = __floats2half2_rn(v.z * inv * g.z, v.w * inv * g.w);
            y2[i] = make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
        }
        return;
    }
    double acc = 0.0;
    for (int i = tid; i < width; i += blockDim.x) {
        const double v = (double)row[i];
        acc += v * v;
    }
    float s = (float)acc;
    for (int m = 16; m > 0; m >>= 1) {
        s += __shfl_xor_sync(0xFFFFFFFFu, s, m);
    }
    if (lane == 0) {
        warp_sum[warp] = s;
    }
    __syncthreads();
    if (warp == 0) {
        s = (lane < (THREADS / 32)) ? warp_sum[lane] : 0.0f;
        for (int m = (THREADS / 32) / 2; m > 0; m >>= 1) {
            s += __shfl_xor_sync(0xFFFFFFFFu, s, m);
        }
        if (lane == 0) {
            warp_sum[0] = rsqrtf(s / (float)width + eps);
        }
    }
    __syncthreads();
    const float inv = warp_sum[0];
    for (int i = tid; i < width; i += blockDim.x) {
        st(y, 0, (size_t)r * width + i, row[i] * inv * w[i]);
    }
}

__global__ void rmsnorm_kernel(void* x, const float* w, int dtype, int width, float eps) {
    __shared__ float red[THREADS];
    const int r = blockIdx.x;
    const size_t base = (size_t)r * width;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = ld(x, dtype, base + i);
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(red[0] / (float)width + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        st(x, dtype, base + i, ld(x, dtype, base + i) * scale * w[i]);
    }
}

// ---------------------------------------------------------------------------------------------------- quantise

/// Quantise-dequantise one `block`-wide group: amax over the group, then round to the nearest E4M3 (bits 8) or E2M1
/// (bits 4) value times the scale. `pow2_scale` is the E8M0 path (scale rounded up to a power of two).
__global__ void quant_warp_f16_b32_kernel(void* x, int width, int n_groups, int bits, int pow2_scale) {
    const int r = blockIdx.y;
    const int g_per_cta = (blockDim.x / 16) * 4;
    const int g_base = blockIdx.x * g_per_cta + (threadIdx.x >> 4) * 4;
    const int i = threadIdx.x & 15;
    if (g_base >= n_groups) return;
    __half* row_ptr = static_cast<__half*>(x) + (size_t)r * width;
    const float mx = (bits == 8) ? 448.0f : 6.0f;
    const float inv_mx = 1.0f / mx;
    if (g_base + 3 < n_groups && bits == 8 && pow2_scale) {
        __half2* x0 = reinterpret_cast<__half2*>(row_ptr + (size_t)(g_base + 0) * 32);
        __half2* x1 = reinterpret_cast<__half2*>(row_ptr + (size_t)(g_base + 1) * 32);
        __half2* x2 = reinterpret_cast<__half2*>(row_ptr + (size_t)(g_base + 2) * 32);
        __half2* x3 = reinterpret_cast<__half2*>(row_ptr + (size_t)(g_base + 3) * 32);
        const __half2 h0 = x0[i];
        const __half2 h1 = x1[i];
        const __half2 h2 = x2[i];
        const __half2 h3 = x3[i];
        const float2 v[4] = {__half22float2(h0), __half22float2(h1), __half22float2(h2), __half22float2(h3)};
        __half2 out_h[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float amax = fmaxf(fabsf(v[k].x), fabsf(v[k].y));
#pragma unroll
            for (int s = 8; s > 0; s >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, s));
            }
            const float a = fmaxf(amax, 1e-4f);
            const float scale = exp2f(ceilf(log2f(a * inv_mx)));
            const float inv = 1.0f / scale;
            out_h[k] = __floats2half2_rn(round_e4m3(v[k].x * inv) * scale, round_e4m3(v[k].y * inv) * scale);
        }
        x0[i] = out_h[0];
        x1[i] = out_h[1];
        x2[i] = out_h[2];
        x3[i] = out_h[3];
        return;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int g = g_base + k;
        if (g >= n_groups) break;
        __half2* x2 = reinterpret_cast<__half2*>(row_ptr + (size_t)g * 32);
        const float2 v = __half22float2(x2[i]);
        float amax = fmaxf(fabsf(v.x), fabsf(v.y));
#pragma unroll
        for (int s = 8; s > 0; s >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, s));
        }
        const float a = fmaxf(amax, 1e-4f);
        const float scale = pow2_scale ? exp2f(ceilf(log2f(a * inv_mx))) : a * inv_mx;
        const float inv = 1.0f / scale;
        const float q0 = v.x * inv;
        const float q1 = v.y * inv;
        const float r0 = (bits == 8 ? round_e4m3(q0) : round_e2m1(q0)) * scale;
        const float r1 = (bits == 8 ? round_e4m3(q1) : round_e2m1(q1)) * scale;
        x2[i] = __floats2half2_rn(r0, r1);
    }
}

__global__ void quant_warp_kernel(void* x, int dtype, int width, int block, int n_groups, int bits, int pow2_scale) {
    const int r = blockIdx.y;
    const int g_per_cta = blockDim.x / block;
    const int g = blockIdx.x * g_per_cta + (threadIdx.x / block);
    const int i = threadIdx.x % block;
    const size_t idx = (size_t)r * width + (size_t)g * block + i;
    const float mx = (bits == 8) ? 448.0f : 6.0f;
    const float val = (g < n_groups) ? ld(x, dtype, idx) : 0.0f;
    float amax = fabsf(val);
    for (int s = block / 2; s > 0; s >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, s));
    }
    if (g < n_groups) {
        const float a = fmaxf(amax, 1e-4f);
        const float scale = pow2_scale ? exp2f(ceilf(log2f(a / mx))) : a / mx;
        const float q = val / scale;
        st(x, dtype, idx, (bits == 8 ? round_e4m3(q) : round_e2m1(q)) * scale);
    }
}

__global__ void quant_kernel(void* x, int dtype, int width, int block, int bits, int pow2_scale) {
    const int r = blockIdx.y;
    const int g = blockIdx.x;
    const size_t base = (size_t)r * width + (size_t)g * block;
    const float mx = (bits == 8) ? 448.0f : 6.0f;
    __shared__ float red[THREADS];
    float amax = 0.0f;
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(ld(x, dtype, base + i)));
    }
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float a = fmaxf(red[0], 1e-4f);
    const float scale = pow2_scale ? exp2f(ceilf(log2f(a / mx))) : a / mx;
    const float inv = 1.0f / scale;
    __syncthreads();
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        const float q = ld(x, dtype, base + i) * inv;
        st(x, dtype, base + i, (bits == 8 ? round_e4m3(q) : round_e2m1(q)) * scale);
    }
}


__global__ void q_rope_rot_f16_kernel(void* x, int stride, int sub, const float* cos, const float* sin,
                                      int table_stride, int pos0, int pos_step, int do_rot) {
    const int r = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk >= sub) return;
    __half2* x2 = reinterpret_cast<__half2*>(static_cast<__half*>(x) + (size_t)r * stride + (size_t)blk * 512);
    const int pos = pos0 + r * pos_step;
    const float ci = __ldg(cos + (size_t)pos * table_stride + lane);
    const float si = __ldg(sin + (size_t)pos * table_stride + lane);
    if (do_rot) {
        __half2 h[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            h[t] = x2[32 * t + lane];
        }
#pragma unroll
        for (int t = 0; t < 7; ++t) {
            const float2 ab = __half22float2(h[t]);
            float a = ab.x, b = ab.y;
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
        const float2 ab = __half22float2(h[7]);
        h[7] = __floats2half2_rn(ab.x * ci - ab.y * si, ab.y * ci + ab.x * si);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            x2[32 * t + lane] = h[t];
        }
    } else {
        const float2 ab = __half22float2(x2[224 + lane]);
        x2[224 + lane] = __floats2half2_rn(ab.x * ci - ab.y * si, ab.y * ci + ab.x * si);
    }
}

__global__ void o_rot_rope_quant_f16_kernel(void* x, int stride, int sub, const float* cos, const float* sin,
                                            int table_stride, int pos0, int pos_step, int do_rot) {
    const int r = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk >= sub) return;
    __half2* x2 = reinterpret_cast<__half2*>(static_cast<__half*>(x) + (size_t)r * stride + (size_t)blk * 512);
    const int pos = pos0 + r * pos_step;
    const float ci = __ldg(cos + (size_t)pos * table_stride + lane);
    const float si = -__ldg(sin + (size_t)pos * table_stride + lane);
    __half2 h_in[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        h_in[t] = x2[32 * t + lane];
    }
    __half2 h_out[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float2 ab = __half22float2(h_in[t]);
        float a = ab.x, b = ab.y;
        if (t < 7) {
            if (do_rot) {
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
                const __half2 h = __floats2half2_rn(a * 0.125f, b * 0.125f);
                const float2 f = __half22float2(h);
                a = f.x;
                b = f.y;
            }
        } else {
            const __half2 h = __floats2half2_rn(a * ci - b * si, b * ci + a * si);
            const float2 f = __half22float2(h);
            a = f.x;
            b = f.y;
        }
        float amax = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
        for (int s = 8; s > 0; s >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, s));
        }
        const float scale = exp2f(ceilf(log2f(fmaxf(amax, 1e-4f) * (1.0f / 448.0f))));
        const float inv = 1.0f / scale;
        h_out[t] = __floats2half2_rn(round_e4m3(a * inv) * scale, round_e4m3(b * inv) * scale);
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        x2[32 * t + lane] = h_out[t];
    }
}

// ---------------------------------------------------------------------------------------------------- fused KV prepare
// One launch replaces the per-layer chain rmsnorm(kv) -> rope(kv) -> quant(kv) -> hadamard(kv) (four kernels
// that each pay a full launch + grid latency on tiny tensors). Each phase below is the verbatim body of the
// kernel it replaces, run sequentially inside one per-row block with barriers between phases, so every
// arithmetic step, reduction order and f16 rounding is bit-identical to the unfused chain.

__global__ void kv_prepare_kernel(void* x, const float* w, int width, float eps, const float* cos,
                                  const float* sin, int table_stride, int pos0, int pos_step, int rope_dim,
                                  int rope_block, int do_rot) {
    __shared__ float red[THREADS];
    const int r = blockIdx.x;
    const size_t base = (size_t)r * width;

    // phase 1: rmsnorm_kernel (in place, f16)
    float ss = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = ld(x, 0, base + i);
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(red[0] / (float)width + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        st(x, 0, base + i, ld(x, 0, base + i) * scale * w[i]);
    }
    __syncthreads();

    // phase 2: rope_kernel (last rope_dim of each rope_block)
    {
        const int pos = pos0 + r * pos_step;
        const float* c = cos + (size_t)pos * table_stride;
        const float* sn = sin + (size_t)pos * table_stride;
        const int half = rope_dim / 2;
        const size_t rb = base + (size_t)(rope_block - rope_dim);
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            const float a = ld(x, 0, rb + 2 * i);
            const float b = ld(x, 0, rb + 2 * i + 1);
            st(x, 0, rb + 2 * i, a * c[i] - b * sn[i]);
            st(x, 0, rb + 2 * i + 1, b * c[i] + a * sn[i]);
        }
    }
    __syncthreads();

    // phase 3: quant_warp_f16_b32_kernel (block 32, bits 8, pow2 scale; grid x == 1, so blockIdx.x == 0)
    {
        const int n_groups = width / 32;
        const int g_per_cta = (blockDim.x / 16) * 4;
        const int g_base = (threadIdx.x >> 4) * 4;
        const int i = threadIdx.x & 15;
        __half* row_ptr = static_cast<__half*>(x) + base;
        const float mx = 448.0f;
        const float inv_mx = 1.0f / mx;
        if (g_base + 3 < n_groups) {
            __half2* x0 = reinterpret_cast<__half2*>(row_ptr + (size_t)(g_base + 0) * 32);
            __half2* x1 = reinterpret_cast<__half2*>(row_ptr + (size_t)(g_base + 1) * 32);
            __half2* x2 = reinterpret_cast<__half2*>(row_ptr + (size_t)(g_base + 2) * 32);
            __half2* x3 = reinterpret_cast<__half2*>(row_ptr + (size_t)(g_base + 3) * 32);
            const __half2 h0 = x0[i];
            const __half2 h1 = x1[i];
            const __half2 h2 = x2[i];
            const __half2 h3 = x3[i];
            const float2 v[4] = {__half22float2(h0), __half22float2(h1), __half22float2(h2), __half22float2(h3)};
            __half2 out_h[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float amax = fmaxf(fabsf(v[k].x), fabsf(v[k].y));
#pragma unroll
                for (int s = 8; s > 0; s >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, s));
                }
                const float a = fmaxf(amax, 1e-4f);
                const float sc = exp2f(ceilf(log2f(a * inv_mx)));
                const float inv = 1.0f / sc;
                out_h[k] = __floats2half2_rn(round_e4m3(v[k].x * inv) * sc, round_e4m3(v[k].y * inv) * sc);
            }
            x0[i] = out_h[0];
            x1[i] = out_h[1];
            x2[i] = out_h[2];
            x3[i] = out_h[3];
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int g = g_base + k;
                if (g >= n_groups) break;
                __half2* xg = reinterpret_cast<__half2*>(row_ptr + (size_t)g * 32);
                const float2 v = __half22float2(xg[i]);
                float amax = fmaxf(fabsf(v.x), fabsf(v.y));
#pragma unroll
                for (int s = 8; s > 0; s >>= 1) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, s));
                }
                const float a = fmaxf(amax, 1e-4f);
                const float sc = exp2f(ceilf(log2f(a * inv_mx)));
                const float inv = 1.0f / sc;
                const float q0 = v.x * inv;
                const float q1 = v.y * inv;
                xg[i] = __floats2half2_rn(round_e4m3(q0) * sc, round_e4m3(q1) * sc);
            }
        }
    }
    __syncthreads();

    // phase 4: walsh_kernel's f16 path (do_rot), sub == 1 so one warp block
    if (do_rot) {
        const int lane = threadIdx.x & 31;
        const int dim = width - rope_dim;
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
        } else {
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
        }
    }
}

// ---------------------------------------------------------------------------------------------------- sink attention

/// One block per (row, head). `idx[0 .. idx_len)` names the attended rows of the concatenated plane: `< split` is
/// `kv_win`, the rest `kv_cmp` (offset by `split`), and `-1` is a hole.
///
/// Two passes over the index list: the max, then `exp` and the value sum. The sink enters only the denominator.
#define DSV4_ATTN_VPT 8
__global__ void attn_kernel(const void* q, const void* kv_win, const void* kv_cmp, int split, const float* sink,
                            const int* idx, int idx_stride, int idx_len, void* out, int n_heads, int head_dim,
                            float scale, int dtype) {
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int npt = (head_dim + blockDim.x - 1) / blockDim.x;  // values per thread
    __shared__ float red[THREADS];

    const size_t qbase = ((size_t)r * n_heads + h) * head_dim;

    float qv[DSV4_ATTN_VPT];
    for (int i = 0; i < npt; ++i) {
        const int c = tid + i * blockDim.x;
        qv[i] = (c < head_dim) ? ld(q, dtype, qbase + c) : 0.0f;
    }

    const int* row = idx + (size_t)r * idx_stride;
    const int len = min(idx_len, idx_stride);

    // pass 1: the max logit
    float mx = -1e30f;
    for (int j = 0; j < len; ++j) {
        const int t = row[j];
        if (t < 0) {
            continue;
        }
        const void* kp = (t < split) ? kv_win : kv_cmp;
        const size_t kbase = (size_t)(t < split ? t : t - split) * head_dim;
        float acc = 0.0f;
        for (int i = 0; i < npt; ++i) {
            const int c = tid + i * blockDim.x;
            if (c < head_dim) {
                acc += qv[i] * ld(kp, dtype, kbase + c);
            }
        }
        red[tid] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                red[tid] += red[tid + s];
            }
            __syncthreads();
        }
        mx = fmaxf(mx, red[0] * scale);
        __syncthreads();
    }

    // pass 2: exp and the value sum
    float acc_v[DSV4_ATTN_VPT];
    for (int i = 0; i < npt; ++i) {
        acc_v[i] = 0.0f;
    }
    float sum = 0.0f;
    for (int j = 0; j < len; ++j) {
        const int t = row[j];
        if (t < 0) {
            continue;
        }
        const void* kp = (t < split) ? kv_win : kv_cmp;
        const size_t kbase = (size_t)(t < split ? t : t - split) * head_dim;
        float acc = 0.0f;
        for (int i = 0; i < npt; ++i) {
            const int c = tid + i * blockDim.x;
            if (c < head_dim) {
                acc += qv[i] * ld(kp, dtype, kbase + c);
            }
        }
        red[tid] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                red[tid] += red[tid + s];
            }
            __syncthreads();
        }
        const float e = expf(red[0] * scale - mx);
        sum += e;
        for (int i = 0; i < npt; ++i) {
            const int c = tid + i * blockDim.x;
            if (c < head_dim) {
                acc_v[i] += e * ld(kp, dtype, kbase + c);
            }
        }
        __syncthreads();
    }
    sum += expf(sink[h] - mx);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = 0; i < npt; ++i) {
        const int c = tid + i * blockDim.x;
        if (c < head_dim) {
            st(out, dtype, qbase + c, acc_v[i] * inv);
        }
    }
}

// ---------------------------------------------------------------------------------------------------- grouped wo_a

__device__ __forceinline__ float4 woa_cvt4_e4m3(uint32_t w4) {
    const __half2 h01 = __nv_cvt_fp8x2_to_halfraw2(w4 & 0xFFFFu, __NV_E4M3);
    const __half2 h23 = __nv_cvt_fp8x2_to_halfraw2(w4 >> 16, __NV_E4M3);
    const float2 f01 = __half22float2(h01);
    const float2 f23 = __half22float2(h23);
    return make_float4(f01.x, f01.y, f23.x, f23.y);
}

__device__ __forceinline__ void ld16_o(const void* row_ptr, int dtype, int k_off, float (&xv)[16]) {
    const uint4* p4 = reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(row_ptr) + k_off);
    const uint4 u0 = __ldg(p4 + 0);
    const uint4 u1 = __ldg(p4 + 1);
    const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    if (dtype == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
            xv[2 * i + 0] = f.x;
            xv[2 * i + 1] = f.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t lo = (w[i] & 0xFFFFu) << 16;
            const uint32_t hi = w[i] & 0xFFFF0000u;
            xv[2 * i + 0] = __uint_as_float(lo);
            xv[2 * i + 1] = __uint_as_float(hi);
        }
    }
}

constexpr int WOA_WARPS = 8;
constexpr int WOA_COLS_PER_WARP = 4;
constexpr int WOA_TILE_L = WOA_WARPS * WOA_COLS_PER_WARP; // 32
constexpr int WOA_CHUNK_K = 512;

template <int ROWS>
__global__ void __launch_bounds__(WOA_WARPS * 32)
    woa_kernel(const void* __restrict__ o, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
               void* __restrict__ out, int rows, int groups, int o_lora, int o_in, int dtype) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int l_base = (int)blockIdx.x * WOA_TILE_L + warp * WOA_COLS_PER_WARP;
    const int g = (int)blockIdx.y;
    const int r_base = (int)blockIdx.z * ROWS;

    const int kt = o_in / 32;
    const size_t h_stride = (size_t)groups * (size_t)o_in;
    const size_t out_stride = (size_t)groups * (size_t)o_lora;
    const uint8_t* wg = w + (size_t)g * (size_t)o_lora * (size_t)o_in;
    const uint8_t* sg = s + (size_t)g * (size_t)(o_lora / 32) * (size_t)kt;

    const uint8_t* w_ptr[WOA_COLS_PER_WARP];
    const uint8_t* s_ptr[WOA_COLS_PER_WARP];
#pragma unroll
    for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
        const int lc = l_base + c;
        const int lc_clamped = (lc < o_lora) ? lc : 0;
        w_ptr[c] = wg + (size_t)lc_clamped * (size_t)o_in;
        s_ptr[c] = sg + (size_t)(lc_clamped / 32) * (size_t)kt;
    }

    float acc[ROWS][WOA_COLS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    if constexpr (ROWS <= 2) {
        const void* o_row_ptr[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = r_base + r;
            const int row_clamped = (row < rows) ? row : 0;
            if (dtype == 0) {
                o_row_ptr[r] = static_cast<const __half*>(o) + (size_t)row_clamped * h_stride + (size_t)g * (size_t)o_in;
            } else {
                o_row_ptr[r] = static_cast<const __nv_bfloat16*>(o) + (size_t)row_clamped * h_stride + (size_t)g * (size_t)o_in;
            }
        }
        for (int k_base = 0; k_base < o_in; k_base += WOA_CHUNK_K) {
            const int rem_k = o_in - k_base;
            const int s_off = lane * 16;
            const bool valid_k = (s_off < rem_k);
            const int k_off = k_base + s_off;
            const int kt_idx = k_off >> 5;
            float x_val[ROWS][16];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (valid_k && (r_base + r < rows)) {
                    ld16_o(o_row_ptr[r], dtype, k_off, x_val[r]);
                } else {
#pragma unroll
                    for (int j = 0; j < 16; ++j) x_val[r][j] = 0.0f;
                }
            }
#pragma unroll
            for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
                if (l_base + c < o_lora) {
                    const uint4 qw = valid_k ? __ldg(reinterpret_cast<const uint4*>(w_ptr[c] + k_off)) : make_uint4(0, 0, 0, 0);
                    const uint8_t sb = valid_k ? __ldg(s_ptr[c] + kt_idx) : 0;
                    const float sc = (sb == 0xFFu) ? __int_as_float(0x7fc00000) : __int_as_float((uint32_t)sb << 23);
                    const float4 wf0 = woa_cvt4_e4m3(qw.x);
                    const float4 wf1 = woa_cvt4_e4m3(qw.y);
                    const float4 wf2 = woa_cvt4_e4m3(qw.z);
                    const float4 wf3 = woa_cvt4_e4m3(qw.w);
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        float sub0 = 0.0f;
                        sub0 = fmaf(wf0.x, x_val[r][0], sub0);
                        sub0 = fmaf(wf0.y, x_val[r][1], sub0);
                        sub0 = fmaf(wf0.z, x_val[r][2], sub0);
                        sub0 = fmaf(wf0.w, x_val[r][3], sub0);
                        sub0 = fmaf(wf1.x, x_val[r][4], sub0);
                        sub0 = fmaf(wf1.y, x_val[r][5], sub0);
                        sub0 = fmaf(wf1.z, x_val[r][6], sub0);
                        sub0 = fmaf(wf1.w, x_val[r][7], sub0);
                        float sub1 = 0.0f;
                        sub1 = fmaf(wf2.x, x_val[r][8], sub1);
                        sub1 = fmaf(wf2.y, x_val[r][9], sub1);
                        sub1 = fmaf(wf2.z, x_val[r][10], sub1);
                        sub1 = fmaf(wf2.w, x_val[r][11], sub1);
                        sub1 = fmaf(wf3.x, x_val[r][12], sub1);
                        sub1 = fmaf(wf3.y, x_val[r][13], sub1);
                        sub1 = fmaf(wf3.z, x_val[r][14], sub1);
                        sub1 = fmaf(wf3.w, x_val[r][15], sub1);
                        const float sub_pair = __shfl_xor_sync(0xFFFFFFFFu, sub0 + sub1, 1);
                        const float tile_sum = (lane & 1) ? (sub_pair + (sub0 + sub1)) : ((sub0 + sub1) + sub_pair);
                        if ((lane & 1) == 0) {
                            acc[r][c] = fmaf(tile_sum, sc, acc[r][c]);
                        }
                    }
                }
            }
        }
    } else {
        __shared__ uint2 sx[ROWS][128];
        for (int k_base = 0; k_base < o_in; k_base += WOA_CHUNK_K) {
            for (int idx = tid; idx < ROWS * 128; idx += WOA_WARPS * 32) {
                const int r = idx >> 7;
                const int u = idx & 127;
                const int row = r_base + r;
                const int kk_u = k_base + u * 4;
                if (row < rows && kk_u < o_in) {
                    const uint16_t* orow = static_cast<const uint16_t*>(o) + (size_t)row * h_stride + (size_t)g * (size_t)o_in + kk_u;
                    sx[r][u] = __ldg(reinterpret_cast<const uint2*>(orow));
                } else {
                    sx[r][u] = make_uint2(0u, 0u);
                }
            }
            __syncthreads();

            const int s_off = lane * 16;
            const int k_off = k_base + s_off;
            const int kt_idx = k_off >> 5;
            const int k_safe = (k_off < o_in) ? k_off : 0;
            const int kt_safe = (k_off < o_in) ? kt_idx : 0;
            {
#pragma unroll
                for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
                    if (l_base + c < o_lora) {
                        const uint4 qw = (k_off < o_in) ? __ldg(reinterpret_cast<const uint4*>(w_ptr[c] + k_safe)) : make_uint4(0u, 0u, 0u, 0u);
                        const uint8_t sb = (k_off < o_in) ? __ldg(s_ptr[c] + kt_safe) : 0;
                        const float sc = (sb == 0xFFu) ? __int_as_float(0x7fc00000) : __int_as_float((uint32_t)sb << 23);
                        const float4 wf0 = woa_cvt4_e4m3(qw.x);
                        const float4 wf1 = woa_cvt4_e4m3(qw.y);
                        const float4 wf2 = woa_cvt4_e4m3(qw.z);
                        const float4 wf3 = woa_cvt4_e4m3(qw.w);
#pragma unroll
                        for (int r = 0; r < ROWS; ++r) {
                            const uint2 u0 = sx[r][lane * 4 + 0];
                            const uint2 u1 = sx[r][lane * 4 + 1];
                            const uint2 u2 = sx[r][lane * 4 + 2];
                            const uint2 u3 = sx[r][lane * 4 + 3];
                            float4 x0, x1, x2, x3;
                            if (dtype == 0) {
                                const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&u0.x));
                                const float2 b0 = __half22float2(*reinterpret_cast<const __half2*>(&u0.y));
                                x0 = make_float4(a0.x, a0.y, b0.x, b0.y);
                                const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&u1.x));
                                const float2 b1 = __half22float2(*reinterpret_cast<const __half2*>(&u1.y));
                                x1 = make_float4(a1.x, a1.y, b1.x, b1.y);
                                const float2 a2 = __half22float2(*reinterpret_cast<const __half2*>(&u2.x));
                                const float2 b2 = __half22float2(*reinterpret_cast<const __half2*>(&u2.y));
                                x2 = make_float4(a2.x, a2.y, b2.x, b2.y);
                                const float2 a3 = __half22float2(*reinterpret_cast<const __half2*>(&u3.x));
                                const float2 b3 = __half22float2(*reinterpret_cast<const __half2*>(&u3.y));
                                x3 = make_float4(a3.x, a3.y, b3.x, b3.y);
                            } else {
                                const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u0.x));
                                const float2 b0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u0.y));
                                x0 = make_float4(a0.x, a0.y, b0.x, b0.y);
                                const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u1.x));
                                const float2 b1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u1.y));
                                x1 = make_float4(a1.x, a1.y, b1.x, b1.y);
                                const float2 a2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.x));
                                const float2 b2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.y));
                                x2 = make_float4(a2.x, a2.y, b2.x, b2.y);
                                const float2 a3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u3.x));
                                const float2 b3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u3.y));
                                x3 = make_float4(a3.x, a3.y, b3.x, b3.y);
                            }
                            float sub0 = 0.0f;
                            sub0 = fmaf(wf0.x, x0.x, sub0);
                            sub0 = fmaf(wf0.y, x0.y, sub0);
                            sub0 = fmaf(wf0.z, x0.z, sub0);
                            sub0 = fmaf(wf0.w, x0.w, sub0);
                            sub0 = fmaf(wf1.x, x1.x, sub0);
                            sub0 = fmaf(wf1.y, x1.y, sub0);
                            sub0 = fmaf(wf1.z, x1.z, sub0);
                            sub0 = fmaf(wf1.w, x1.w, sub0);
                            float sub1 = 0.0f;
                            sub1 = fmaf(wf2.x, x2.x, sub1);
                            sub1 = fmaf(wf2.y, x2.y, sub1);
                            sub1 = fmaf(wf2.z, x2.z, sub1);
                            sub1 = fmaf(wf2.w, x2.w, sub1);
                            sub1 = fmaf(wf3.x, x3.x, sub1);
                            sub1 = fmaf(wf3.y, x3.y, sub1);
                            sub1 = fmaf(wf3.z, x3.z, sub1);
                            sub1 = fmaf(wf3.w, x3.w, sub1);
                            const float sub_pair = __shfl_xor_sync(0xFFFFFFFFu, sub0 + sub1, 1);
                            const float tile_sum = (lane & 1) ? (sub_pair + (sub0 + sub1)) : ((sub0 + sub1) + sub_pair);
                            if ((lane & 1) == 0) {
                                acc[r][c] = fmaf(tile_sum, sc, acc[r][c]);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
        const int lc = l_base + c;
        if (lc < o_lora) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = r_base + r;
                if (row < rows) {
                    float sum = acc[r][c];
#pragma unroll
                    for (int offset = 16; offset >= 2; offset >>= 1) {
                        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, offset);
                    }
                    if (lane == 0) {
                        st(out, dtype, (size_t)row * out_stride + (size_t)g * (size_t)o_lora + (size_t)lc, sum);
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void woa_sk_cp_async16(void* smem, const void* gmem, bool valid) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void woa_sk_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void woa_sk_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ------------------------------------------------------------------ woa small-M fast path
//
// The grouped output projection `wo_a` runs at decode rows 1-8 against a large FP8 weight (8 groups x
// [1024, 4096] on the shipped pack). The classic `woa_kernel` above stages each 512-wide K chunk of x
// through shared memory with a full barrier before and after, so the chunk's global-load latency is fully
// exposed 8 times per CTA. This variant keeps the EXACT per-output accumulation order (same k-ascending
// chunk loop, same fmaf chains, same pair-shuffle reduction tree and store) but
//   - for rows <= 2 double-buffers the chunk's x in registers and prefetches the next chunk's weights and
//     scales while the current chunk accumulates, and
//   - for rows >= 3 stages x through a 2-stage cp.async shared ring, so the next chunk's loads overlap the
//     current chunk's math.
// Only the load timing moves; the bits are identical to the classic kernel's by construction.
template <int ROWS>
__global__ void __launch_bounds__(WOA_WARPS * 32)
    woa_kernel_sk(const void* __restrict__ o, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                  void* __restrict__ out, int rows, int groups, int o_lora, int o_in, int dtype) {
    constexpr int THREADS = WOA_WARPS * 32;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int l_base = (int)blockIdx.x * WOA_TILE_L + warp * WOA_COLS_PER_WARP;
    const int g = (int)blockIdx.y;
    const int r_base = (int)blockIdx.z * ROWS;

    const int kt = o_in / 32;
    const size_t h_stride = (size_t)groups * (size_t)o_in;
    const size_t out_stride = (size_t)groups * (size_t)o_lora;
    const uint8_t* wg = w + (size_t)g * (size_t)o_lora * (size_t)o_in;
    const uint8_t* sg = s + (size_t)g * (size_t)(o_lora / 32) * (size_t)kt;

    const uint8_t* w_ptr[WOA_COLS_PER_WARP];
    const uint8_t* s_ptr[WOA_COLS_PER_WARP];
#pragma unroll
    for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
        const int lc = l_base + c;
        const int lc_clamped = (lc < o_lora) ? lc : 0;
        w_ptr[c] = wg + (size_t)lc_clamped * (size_t)o_in;
        s_ptr[c] = sg + (size_t)(lc_clamped / 32) * (size_t)kt;
    }

    float acc[ROWS][WOA_COLS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    const int nchunks = (o_in + WOA_CHUNK_K - 1) / WOA_CHUNK_K;
    const int s_off = lane * 16;

    if constexpr (ROWS <= 2) {
        // Register double-buffered direct path. x for the next chunk is prefetched while the current chunk
        // accumulates; weights and scales are prefetched the same way. The math, the pair-shuffle reduction
        // and the gating are identical to the classic woa_kernel<ROWS>.
        const void* o_row_ptr[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = r_base + r;
            const int row_clamped = (row < rows) ? row : 0;
            if (dtype == 0) {
                o_row_ptr[r] = static_cast<const __half*>(o) + (size_t)row_clamped * h_stride + (size_t)g * (size_t)o_in;
            } else {
                o_row_ptr[r] = static_cast<const __nv_bfloat16*>(o) + (size_t)row_clamped * h_stride + (size_t)g * (size_t)o_in;
            }
        }

        auto load_chunk = [&](int ci, float (&xv)[ROWS][16], uint4 (&wv)[WOA_COLS_PER_WARP], uint8_t (&sbv)[WOA_COLS_PER_WARP]) {
            const int k_base = ci * WOA_CHUNK_K;
            const int rem_k = o_in - k_base;
            const bool valid_k = (s_off < rem_k);
            const int k_off = k_base + s_off;
            const int kt_idx = k_off >> 5;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (valid_k && (r_base + r < rows)) {
                    ld16_o(o_row_ptr[r], dtype, k_off, xv[r]);
                } else {
#pragma unroll
                    for (int j = 0; j < 16; ++j) xv[r][j] = 0.0f;
                }
            }
#pragma unroll
            for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
                if (l_base + c < o_lora) {
                    wv[c] = valid_k ? __ldg(reinterpret_cast<const uint4*>(w_ptr[c] + k_off)) : make_uint4(0, 0, 0, 0);
                    sbv[c] = valid_k ? __ldg(s_ptr[c] + kt_idx) : 0;
                } else {
                    wv[c] = make_uint4(0, 0, 0, 0);
                    sbv[c] = 0;
                }
            }
        };

        float x_cur[ROWS][16], x_next[ROWS][16];
        uint4 w_cur[WOA_COLS_PER_WARP], w_next[WOA_COLS_PER_WARP];
        uint8_t sb_cur[WOA_COLS_PER_WARP], sb_next[WOA_COLS_PER_WARP];
        load_chunk(0, x_cur, w_cur, sb_cur);

        for (int ci = 0; ci < nchunks; ++ci) {
            if (ci + 1 < nchunks) {
                load_chunk(ci + 1, x_next, w_next, sb_next);
            }
#pragma unroll
            for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
                if (l_base + c < o_lora) {
                    const uint4 qw = w_cur[c];
                    const uint8_t sb = sb_cur[c];
                    const float sc = (sb == 0xFFu) ? __int_as_float(0x7fc00000) : __int_as_float((uint32_t)sb << 23);
                    const float4 wf0 = woa_cvt4_e4m3(qw.x);
                    const float4 wf1 = woa_cvt4_e4m3(qw.y);
                    const float4 wf2 = woa_cvt4_e4m3(qw.z);
                    const float4 wf3 = woa_cvt4_e4m3(qw.w);
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        float sub0 = 0.0f;
                        sub0 = fmaf(wf0.x, x_cur[r][0], sub0);
                        sub0 = fmaf(wf0.y, x_cur[r][1], sub0);
                        sub0 = fmaf(wf0.z, x_cur[r][2], sub0);
                        sub0 = fmaf(wf0.w, x_cur[r][3], sub0);
                        sub0 = fmaf(wf1.x, x_cur[r][4], sub0);
                        sub0 = fmaf(wf1.y, x_cur[r][5], sub0);
                        sub0 = fmaf(wf1.z, x_cur[r][6], sub0);
                        sub0 = fmaf(wf1.w, x_cur[r][7], sub0);
                        float sub1 = 0.0f;
                        sub1 = fmaf(wf2.x, x_cur[r][8], sub1);
                        sub1 = fmaf(wf2.y, x_cur[r][9], sub1);
                        sub1 = fmaf(wf2.z, x_cur[r][10], sub1);
                        sub1 = fmaf(wf2.w, x_cur[r][11], sub1);
                        sub1 = fmaf(wf3.x, x_cur[r][12], sub1);
                        sub1 = fmaf(wf3.y, x_cur[r][13], sub1);
                        sub1 = fmaf(wf3.z, x_cur[r][14], sub1);
                        sub1 = fmaf(wf3.w, x_cur[r][15], sub1);
                        const float sub_pair = __shfl_xor_sync(0xFFFFFFFFu, sub0 + sub1, 1);
                        const float tile_sum = (lane & 1) ? (sub_pair + (sub0 + sub1)) : ((sub0 + sub1) + sub_pair);
                        if ((lane & 1) == 0) {
                            acc[r][c] = fmaf(tile_sum, sc, acc[r][c]);
                        }
                    }
                }
            }
            if (ci + 1 < nchunks) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    for (int j = 0; j < 16; ++j) x_cur[r][j] = x_next[r][j];
                }
#pragma unroll
                for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
                    w_cur[c] = w_next[c];
                    sb_cur[c] = sb_next[c];
                }
            }
        }
    } else {
        // 2-stage cp.async shared ring for x; weights/scales are loaded per chunk in the compute phase
        // exactly like the classic path. The ring is zero-initialised so out-of-range slots (rows >= rows,
        // k beyond o_in) stay +0.0 and contribute the same bits as the classic zero-fill.
        __shared__ __align__(128) uint4 sx_ring[2][ROWS][WOA_CHUNK_K / 8];
        for (int idx = tid; idx < ROWS * (WOA_CHUNK_K / 8); idx += THREADS) {
            sx_ring[0][idx >> 6][idx & 63] = make_uint4(0u, 0u, 0u, 0u);
            sx_ring[1][idx >> 6][idx & 63] = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();

        auto stage_chunk = [&](int ci, int buf) {
            const int k_base = ci * WOA_CHUNK_K;
            for (int idx = tid; idx < ROWS * (WOA_CHUNK_K / 8); idx += THREADS) {
                const int r = idx >> 6;
                const int u = idx & 63;
                const int row = r_base + r;
                const int kk_u = k_base + u * 8;
                const bool ok = (row < rows) && (kk_u < o_in);
                const int rowc = (row < rows) ? row : 0;
                const int kkc = (kk_u < o_in) ? kk_u : 0;
                const uint16_t* orow = static_cast<const uint16_t*>(o) + (size_t)rowc * h_stride + (size_t)g * (size_t)o_in + kkc;
                woa_sk_cp_async16(&sx_ring[buf][r][u], orow, ok);
            }
        };
        stage_chunk(0, 0);
        woa_sk_cp_async_commit();

        for (int ci = 0; ci < nchunks; ++ci) {
            const int buf = ci & 1;
            if (ci + 1 < nchunks) {
                stage_chunk(ci + 1, buf ^ 1);
                woa_sk_cp_async_commit();
                woa_sk_cp_async_wait<1>();
            } else {
                woa_sk_cp_async_wait<0>();
            }
            __syncthreads();

            const int k_base = ci * WOA_CHUNK_K;
            const int k_off = k_base + s_off;
            const int kt_idx = k_off >> 5;
            const int k_safe = (k_off < o_in) ? k_off : 0;
            const int kt_safe = (k_off < o_in) ? kt_idx : 0;
            {
#pragma unroll
                for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
                    if (l_base + c < o_lora) {
                        const uint4 qw = (k_off < o_in) ? __ldg(reinterpret_cast<const uint4*>(w_ptr[c] + k_safe)) : make_uint4(0u, 0u, 0u, 0u);
                        const uint8_t sb = (k_off < o_in) ? __ldg(s_ptr[c] + kt_safe) : 0;
                        const float sc = (sb == 0xFFu) ? __int_as_float(0x7fc00000) : __int_as_float((uint32_t)sb << 23);
                        const float4 wf0 = woa_cvt4_e4m3(qw.x);
                        const float4 wf1 = woa_cvt4_e4m3(qw.y);
                        const float4 wf2 = woa_cvt4_e4m3(qw.z);
                        const float4 wf3 = woa_cvt4_e4m3(qw.w);
#pragma unroll
                        for (int r = 0; r < ROWS; ++r) {
                            const uint4 a = sx_ring[buf][r][lane * 2];
                            const uint4 b = sx_ring[buf][r][lane * 2 + 1];
                            const uint2 u0 = make_uint2(a.x, a.y);
                            const uint2 u1 = make_uint2(a.z, a.w);
                            const uint2 u2 = make_uint2(b.x, b.y);
                            const uint2 u3 = make_uint2(b.z, b.w);
                            float4 x0, x1, x2, x3;
                            if (dtype == 0) {
                                const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&u0.x));
                                const float2 b0 = __half22float2(*reinterpret_cast<const __half2*>(&u0.y));
                                x0 = make_float4(a0.x, a0.y, b0.x, b0.y);
                                const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&u1.x));
                                const float2 b1 = __half22float2(*reinterpret_cast<const __half2*>(&u1.y));
                                x1 = make_float4(a1.x, a1.y, b1.x, b1.y);
                                const float2 a2 = __half22float2(*reinterpret_cast<const __half2*>(&u2.x));
                                const float2 b2 = __half22float2(*reinterpret_cast<const __half2*>(&u2.y));
                                x2 = make_float4(a2.x, a2.y, b2.x, b2.y);
                                const float2 a3 = __half22float2(*reinterpret_cast<const __half2*>(&u3.x));
                                const float2 b3 = __half22float2(*reinterpret_cast<const __half2*>(&u3.y));
                                x3 = make_float4(a3.x, a3.y, b3.x, b3.y);
                            } else {
                                const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u0.x));
                                const float2 b0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u0.y));
                                x0 = make_float4(a0.x, a0.y, b0.x, b0.y);
                                const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u1.x));
                                const float2 b1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u1.y));
                                x1 = make_float4(a1.x, a1.y, b1.x, b1.y);
                                const float2 a2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.x));
                                const float2 b2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u2.y));
                                x2 = make_float4(a2.x, a2.y, b2.x, b2.y);
                                const float2 a3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u3.x));
                                const float2 b3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u3.y));
                                x3 = make_float4(a3.x, a3.y, b3.x, b3.y);
                            }
                            float sub0 = 0.0f;
                            sub0 = fmaf(wf0.x, x0.x, sub0);
                            sub0 = fmaf(wf0.y, x0.y, sub0);
                            sub0 = fmaf(wf0.z, x0.z, sub0);
                            sub0 = fmaf(wf0.w, x0.w, sub0);
                            sub0 = fmaf(wf1.x, x1.x, sub0);
                            sub0 = fmaf(wf1.y, x1.y, sub0);
                            sub0 = fmaf(wf1.z, x1.z, sub0);
                            sub0 = fmaf(wf1.w, x1.w, sub0);
                            float sub1 = 0.0f;
                            sub1 = fmaf(wf2.x, x2.x, sub1);
                            sub1 = fmaf(wf2.y, x2.y, sub1);
                            sub1 = fmaf(wf2.z, x2.z, sub1);
                            sub1 = fmaf(wf2.w, x2.w, sub1);
                            sub1 = fmaf(wf3.x, x3.x, sub1);
                            sub1 = fmaf(wf3.y, x3.y, sub1);
                            sub1 = fmaf(wf3.z, x3.z, sub1);
                            sub1 = fmaf(wf3.w, x3.w, sub1);
                            const float sub_pair = __shfl_xor_sync(0xFFFFFFFFu, sub0 + sub1, 1);
                            const float tile_sum = (lane & 1) ? (sub_pair + (sub0 + sub1)) : ((sub0 + sub1) + sub_pair);
                            if ((lane & 1) == 0) {
                                acc[r][c] = fmaf(tile_sum, sc, acc[r][c]);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int c = 0; c < WOA_COLS_PER_WARP; ++c) {
        const int lc = l_base + c;
        if (lc < o_lora) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = r_base + r;
                if (row < rows) {
                    float sum = acc[r][c];
#pragma unroll
                    for (int offset = 16; offset >= 2; offset >>= 1) {
                        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, offset);
                    }
                    if (lane == 0) {
                        st(out, dtype, (size_t)row * out_stride + (size_t)g * (size_t)o_lora + (size_t)lc, sum);
                    }
                }
            }
        }
    }
}


__device__ __forceinline__ void woa_mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void woa_decode_fp8x4(uint32_t packed4, uint32_t& h0, uint32_t& h1) {
    const __half2_raw r0 = __nv_cvt_fp8x2_to_halfraw2((uint16_t)(packed4 & 0xFFFFu), __NV_E4M3);
    const __half2_raw r1 = __nv_cvt_fp8x2_to_halfraw2((uint16_t)(packed4 >> 16), __NV_E4M3);
    h0 = *reinterpret_cast<const uint32_t*>(&r0);
    h1 = *reinterpret_cast<const uint32_t*>(&r1);
}

template <int M_TILES_PER_WARP, int M_WARPS>
__global__ void __launch_bounds__(M_WARPS * 128)
    woa_mma_kernel(const void* __restrict__ o, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                   void* __restrict__ out, int rows, int groups, int o_lora, int o_in, int dtype) {
    constexpr int N_WARPS = 4;
    constexpr int THREADS_CTA = M_WARPS * N_WARPS * 32;
    constexpr int ROWS = M_WARPS * M_TILES_PER_WARP * 16;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int warp_n = warp & 3;
    const int warp_m = warp >> 2;
    const int lane = tid & 31;
    const int g_lane = lane >> 2;
    const int t = lane & 3;
    const int l_cta = (int)blockIdx.x * 32;
    const int l_base = l_cta + warp_n * 8;
    const int g = (int)blockIdx.y;
    const int r_base = (int)blockIdx.z * ROWS;

    const int kt = o_in / 32;
    const size_t h_stride = (size_t)groups * (size_t)o_in;
    const size_t out_stride = (size_t)groups * (size_t)o_lora;
    const uint8_t* wg = w + (size_t)g * (size_t)o_lora * (size_t)o_in;
    const uint8_t* sg = s + (size_t)g * (size_t)(o_lora / 32) * (size_t)kt + (size_t)blockIdx.x * (size_t)kt;

    float acc[M_TILES_PER_WARP][4];
#pragma unroll
    for (int mt = 0; mt < M_TILES_PER_WARP; ++mt) {
        acc[mt][0] = 0.0f;
        acc[mt][1] = 0.0f;
        acc[mt][2] = 0.0f;
        acc[mt][3] = 0.0f;
    }

    __shared__ uint32_t sx[ROWS][36];
    __shared__ uint32_t sw[N_WARPS][8][36];

    const int lc_load = l_cta + (tid >> 2);
    const int t_load = tid & 3;
    const uint8_t* wr_load_base = (tid < 128 && lc_load < o_lora) ? (wg + (size_t)lc_load * (size_t)o_in) : nullptr;

    for (int k_base = 0; k_base < o_in; k_base += 64) {
        for (int idx = tid; idx < ROWS * 8; idx += THREADS_CTA) {
            const int r = idx >> 3;
            const int u = idx & 7;
            const int row = r_base + r;
            const int kk = k_base + u * 8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (row < rows && kk < o_in) {
                const uint16_t* orow = static_cast<const uint16_t*>(o) + (size_t)row * h_stride + (size_t)g * (size_t)o_in + kk;
                v = __ldg(reinterpret_cast<const uint4*>(orow));
                if (dtype != 0) {
                    const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
                    const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
                    const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.z));
                    const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.w));
                    const __half2 h0 = __floats2half2_rn(f0.x, f0.y);
                    const __half2 h1 = __floats2half2_rn(f1.x, f1.y);
                    const __half2 h2 = __floats2half2_rn(f2.x, f2.y);
                    const __half2 h3 = __floats2half2_rn(f3.x, f3.y);
                    v.x = *reinterpret_cast<const uint32_t*>(&h0);
                    v.y = *reinterpret_cast<const uint32_t*>(&h1);
                    v.z = *reinterpret_cast<const uint32_t*>(&h2);
                    v.w = *reinterpret_cast<const uint32_t*>(&h3);
                }
            }
            sx[r][u * 4 + 0] = v.x;
            sx[r][u * 4 + 1] = v.y;
            sx[r][u * 4 + 2] = v.z;
            sx[r][u * 4 + 3] = v.w;
        }
        if (tid < 128) {
            const int kk_w = k_base + t_load * 16;
            const uint4 wv = (wr_load_base != nullptr && kk_w < o_in)
                                 ? __ldg(reinterpret_cast<const uint4*>(wr_load_base + kk_w))
                                 : make_uint4(0u, 0u, 0u, 0u);
            uint32_t h0, h1, h2, h3, h4, h5, h6, h7;
            woa_decode_fp8x4(wv.x, h0, h1);
            woa_decode_fp8x4(wv.y, h2, h3);
            woa_decode_fp8x4(wv.z, h4, h5);
            woa_decode_fp8x4(wv.w, h6, h7);
            uint32_t* dst = &sw[warp_n][g_lane][t_load * 8];
            dst[0] = h0; dst[1] = h1; dst[2] = h2; dst[3] = h3;
            dst[4] = h4; dst[5] = h5; dst[6] = h6; dst[7] = h7;
        }
        __syncthreads();

#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const int kk_tile = k_base + sub * 32;
            if (kk_tile < o_in) {
                const uint8_t sb = __ldg(sg + (kk_tile >> 5));
                const float sc = (sb == 0xFFu) ? __int_as_float(0x7fc00000) : __int_as_float((uint32_t)sb << 23);
                const int c_base = sub * 16;
                const uint32_t b0[2] = {sw[warp_n][g_lane][c_base + t], sw[warp_n][g_lane][c_base + 4 + t]};
                const uint32_t b1[2] = {sw[warp_n][g_lane][c_base + 8 + t], sw[warp_n][g_lane][c_base + 12 + t]};
#pragma unroll
                for (int mt = 0; mt < M_TILES_PER_WARP; ++mt) {
                    const int m_idx = warp_m * M_TILES_PER_WARP + mt;
                    const int r_lo = m_idx * 16 + g_lane;
                    const int r_hi = r_lo + 8;
                    const uint32_t a0[4] = {sx[r_lo][c_base + t], sx[r_hi][c_base + t], sx[r_lo][c_base + 4 + t], sx[r_hi][c_base + 4 + t]};
                    const uint32_t a1[4] = {sx[r_lo][c_base + 8 + t], sx[r_hi][c_base + 8 + t], sx[r_lo][c_base + 12 + t], sx[r_hi][c_base + 12 + t]};
                    float td[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    woa_mma16816(td, a0, b0);
                    woa_mma16816(td, a1, b1);
                    acc[mt][0] = fmaf(td[0], sc, acc[mt][0]);
                    acc[mt][1] = fmaf(td[1], sc, acc[mt][1]);
                    acc[mt][2] = fmaf(td[2], sc, acc[mt][2]);
                    acc[mt][3] = fmaf(td[3], sc, acc[mt][3]);
                }
            }
        }
        __syncthreads();
    }

    const int col_out = l_base + 2 * t;
    if (col_out < o_lora) {
#pragma unroll
        for (int mt = 0; mt < M_TILES_PER_WARP; ++mt) {
            const int m_idx = warp_m * M_TILES_PER_WARP + mt;
            const int rc0 = r_base + m_idx * 16 + g_lane;
            const int rc1 = rc0 + 8;
            if (rc0 < rows) {
                const size_t base0 = (size_t)rc0 * out_stride + (size_t)g * (size_t)o_lora + (size_t)col_out;
                st(out, dtype, base0, acc[mt][0]);
                if (col_out + 1 < o_lora) st(out, dtype, base0 + 1, acc[mt][1]);
            }
            if (rc1 < rows) {
                const size_t base1 = (size_t)rc1 * out_stride + (size_t)g * (size_t)o_lora + (size_t)col_out;
                st(out, dtype, base1, acc[mt][2]);
                if (col_out + 1 < o_lora) st(out, dtype, base1 + 1, acc[mt][3]);
            }
        }
    }
}


constexpr int WOA_MMA_THREADS = 256;
constexpr int WOA_MMA_BM = 64;
constexpr int WOA_MMA_BN = 128;
constexpr int WOA_MMA_BK = 64;
constexpr int WOA_MMA_STAGES = 3;

__device__ __forceinline__ void woa_cp_async16(void* smem, const void* gmem, bool valid) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void woa_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void woa_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void woa_ldsm_x4(uint32_t (&a)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}
template <int IMM_BYTES>
__device__ __forceinline__ void woa_ldsm_x4_imm(uint32_t (&a)[4], uint32_t s_base) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4+%5];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s_base), "n"(IMM_BYTES));
}
__device__ __forceinline__ void woa_ldsm_x4_step(uint32_t (&a)[4], uint32_t s_base, int step_1k) {
    switch (step_1k) {
        case 0: woa_ldsm_x4_imm<0>(a, s_base); break;
        case 1: woa_ldsm_x4_imm<1024>(a, s_base); break;
        case 2: woa_ldsm_x4_imm<2048>(a, s_base); break;
        case 3: woa_ldsm_x4_imm<3072>(a, s_base); break;
        case 4: woa_ldsm_x4_imm<4096>(a, s_base); break;
        case 5: woa_ldsm_x4_imm<5120>(a, s_base); break;
        case 6: woa_ldsm_x4_imm<6144>(a, s_base); break;
        default: woa_ldsm_x4_imm<7168>(a, s_base); break;
    }
}
__device__ __forceinline__ void woa_mma16816_zero(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    const float z = 0.0f;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(z));
}
__device__ __forceinline__ uint32_t woa_cvt2_e4m3_h2(uint16_t b2) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(b2, __NV_E4M3);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ int woa_a_off(int r, int c) { return r * WOA_MMA_BK + ((c ^ (r & 7)) << 3); }
__device__ __forceinline__ int woa_b_off(int nr, int c16) { return nr * 4 + (c16 ^ (nr & 3)); }
__device__ __forceinline__ int woa_a_off32(int r, int c) { return r * 32 + ((c ^ ((r >> 1) & 3)) << 3); }
__device__ __forceinline__ int woa_b_off32(int nr, int c16) { return nr * 2 + (c16 ^ ((nr >> 2) & 1)); }

template <int BM, int STAGES>
__global__ void __launch_bounds__(256, 2)
    woa_cp_mma_bk32_kernel(const void* __restrict__ o, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                           void* __restrict__ out, int rows, int groups, int o_lora, int o_in) {
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int MT = BM / 16;
    constexpr int A_STAGE_HALVES = BM * BK;
    constexpr int B_STAGE_U4 = BN * 2;
    constexpr int A_ITERS = (BM * 4) / 256;
    __shared__ __align__(128) __half sa_ring[STAGES][A_STAGE_HALVES];
    __shared__ __align__(128) uint4 sb_ring[STAGES][B_STAGE_U4];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g_lane = lane >> 2;
    const int t = lane & 3;

    const int m0 = (int)blockIdx.x * BM;
    const int n0 = (int)blockIdx.y * BN;
    const int g = (int)blockIdx.z;

    const int ktiles = o_in / 32;
    const int h_stride = groups * o_in;
    const __half* o_base = static_cast<const __half*>(o) + (size_t)g * (size_t)o_in;
    const uint8_t* wg = w + (size_t)g * (size_t)o_lora * (size_t)o_in;
    const int n_warp = n0 + warp * 16;
    const uint8_t* s_row = s + (size_t)(g * (o_lora / 32) + (n_warp / 32)) * (size_t)ktiles;
    const int KC = o_in / BK;

    auto load_stage = [&](int stg, int kc) {
        __half* sa = sa_ring[stg];
        uint4* sb = sb_ring[stg];
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int idx = i * 256 + tid;
            const int r = idx >> 2;
            const int c = idx & 3;
            const int gr = m0 + r;
            const bool ok = (gr < rows);
            const size_t x_off = (size_t)(ok ? gr : 0) * (size_t)h_stride + kc * BK + c * 8;
            woa_cp_async16(sa + woa_a_off32(r, c), o_base + x_off, ok);
        }
        {
            const int nr = tid >> 1;
            const int c16 = tid & 1;
            const int gn = n0 + nr;
            woa_cp_async16(&sb[woa_b_off32(nr, c16)], wg + (size_t)gn * (size_t)o_in + kc * BK + c16 * 16, true);
        }
    };

    float acc[MT][2][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mt][nt][c] = 0.0f;

#pragma unroll
    for (int s_idx = 0; s_idx < STAGES - 1; ++s_idx) {
        if (s_idx < KC) load_stage(s_idx, s_idx);
        woa_cp_async_commit();
    }

    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int lc = lane >> 4;

#pragma unroll 1
    for (int kc = 0; kc < KC; ++kc) {
        woa_cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int nk = kc + STAGES - 1;
        if (nk < KC) load_stage(nk % STAGES, nk);
        woa_cp_async_commit();

        const __half* sa = sa_ring[kc % STAGES];
        const uint4* sb = sb_ring[kc % STAGES];

        const uint8_t sb_byte = __ldg(s_row + kc);
        const uint32_t sc_bits = (sb_byte == 0u) ? 0x00400000u : ((sb_byte == 0xFFu) ? 0x7fc00000u : ((uint32_t)sb_byte << 23));
        const float sc = __uint_as_float(sc_bits);
        uint32_t b0[2][2], b1[2][2];
        const int nr_base = warp * 16 + g_lane;
        const int b_off0 = woa_b_off32(nr_base, 0);
        const int b_off1 = woa_b_off32(nr_base, 1);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint16_t* r0 = reinterpret_cast<const uint16_t*>(&sb[b_off0 + nt * 16]);
            const uint16_t* r1 = reinterpret_cast<const uint16_t*>(&sb[b_off1 + nt * 16]);
            b0[nt][0] = woa_cvt2_e4m3_h2(r0[t]);
            b0[nt][1] = woa_cvt2_e4m3_h2(r0[t + 4]);
            b1[nt][0] = woa_cvt2_e4m3_h2(r1[t]);
            b1[nt][1] = woa_cvt2_e4m3_h2(r1[t + 4]);
        }
        const uint32_t s_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(sa + woa_a_off32(lr, 0 + lc)));
        const uint32_t s_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(sa + woa_a_off32(lr, 2 + lc)));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            uint32_t a0[4], a1[4];
            woa_ldsm_x4_step(a0, s_a0, mt);
            woa_ldsm_x4_step(a1, s_a1, mt);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                float p[4];
                woa_mma16816_zero(p, a0, b0[nt]);
                woa_mma16816(p, a1, b1[nt]);
                acc[mt][nt][0] = fmaf(p[0], sc, acc[mt][nt][0]);
                acc[mt][nt][1] = fmaf(p[1], sc, acc[mt][nt][1]);
                acc[mt][nt][2] = fmaf(p[2], sc, acc[mt][nt][2]);
                acc[mt][nt][3] = fmaf(p[3], sc, acc[mt][nt][3]);
            }
        }
    }
    woa_cp_async_wait<0>();

    const size_t out_stride = (size_t)groups * (size_t)o_lora;
    __half* y = static_cast<__half*>(out) + (size_t)g * (size_t)o_lora;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int r0 = m0 + mt * 16 + g_lane;
        const int r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = n_warp + nt * 8 + 2 * t;
            if (col < o_lora) {
                if (r0 < rows) {
                    *reinterpret_cast<__half2*>(y + (size_t)r0 * out_stride + col) =
                        __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
                }
                if (r1 < rows) {
                    *reinterpret_cast<__half2*>(y + (size_t)r1 * out_stride + col) =
                        __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
                }
            }
        }
    }
}

template <int BM, int BN, int M_WARPS, int N_WARPS, int STAGES, int MIN_BLOCKS>
__global__ void __launch_bounds__(M_WARPS * N_WARPS * 32, MIN_BLOCKS)
    woa_cp_mma_kernel(const void* __restrict__ o, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                      void* __restrict__ out, int rows, int groups, int o_lora, int o_in) {
    constexpr int THREADS = M_WARPS * N_WARPS * 32;
    constexpr int MT_PER_WARP = (BM / 16) / M_WARPS;
    constexpr int NT_PER_WARP = (BN / 8) / N_WARPS;
    constexpr int A_STAGE_HALVES = BM * WOA_MMA_BK;
    constexpr int B_STAGE_U4 = BN * 4;
    constexpr int A_ITERS = (BM * 8) / THREADS;
    constexpr int B_ITERS = (BN * 4) / THREADS;
    __shared__ __align__(128) __half sa_ring[STAGES][A_STAGE_HALVES];
    __shared__ __align__(128) uint4 sb_ring[STAGES][B_STAGE_U4];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int warp_m = warp / N_WARPS;
    const int warp_n = warp % N_WARPS;
    const int lane = tid & 31;
    const int g_lane = lane >> 2;
    const int t = lane & 3;

    const int m0 = (int)blockIdx.x * BM;
    const int n0 = (int)blockIdx.y * BN;
    const int g = (int)blockIdx.z;

    const int ktiles = o_in / 32;
    const size_t h_stride = (size_t)groups * (size_t)o_in;
    const size_t out_stride = (size_t)groups * (size_t)o_lora;
    const uint8_t* wg = w + (size_t)g * (size_t)o_lora * (size_t)o_in;
    const int n_warp = n0 + warp_n * (NT_PER_WARP * 8);
    const uint8_t* s_row = s + (size_t)g * (size_t)(o_lora / 32) * (size_t)ktiles + (size_t)(n_warp / 32) * (size_t)ktiles;
    const int KC = o_in / WOA_MMA_BK;

    auto load_stage = [&](int stg, int kc) {
        __half* sa = sa_ring[stg];
        uint4* sb = sb_ring[stg];
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int idx = i * THREADS + tid;
            const int r = idx >> 3;
            const int c = idx & 7;
            const int gr = m0 + r;
            const bool ok = (gr < rows);
            const size_t x_off = (size_t)(ok ? gr : 0) * h_stride + (size_t)g * (size_t)o_in + kc * WOA_MMA_BK + c * 8;
            woa_cp_async16(sa + woa_a_off(r, c), static_cast<const __half*>(o) + x_off, ok);
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
            const int idx = i * THREADS + tid;
            const int nr = idx >> 2;
            const int c16 = idx & 3;
            const int gn = n0 + nr;
            woa_cp_async16(&sb[woa_b_off(nr, c16)], wg + (size_t)gn * (size_t)o_in + kc * WOA_MMA_BK + c16 * 16, true);
        }
    };

    float acc[MT_PER_WARP][NT_PER_WARP][4];
#pragma unroll
    for (int mt = 0; mt < MT_PER_WARP; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mt][nt][c] = 0.0f;

#pragma unroll
    for (int s_idx = 0; s_idx < STAGES - 1; ++s_idx) {
        if (s_idx < KC) load_stage(s_idx, s_idx);
        woa_cp_async_commit();
    }

    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int lc = lane >> 4;

#pragma unroll 1
    for (int kc = 0; kc < KC; ++kc) {
        woa_cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int nk = kc + STAGES - 1;
        if (nk < KC) load_stage(nk % STAGES, nk);
        woa_cp_async_commit();

        const __half* sa = sa_ring[kc % STAGES];
        const uint4* sb = sb_ring[kc % STAGES];

#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const uint8_t sb_byte = __ldg(s_row + kc * 2 + sub);
            const uint32_t sc_bits = (sb_byte == 0u) ? 0x00400000u : ((sb_byte == 0xFFu) ? 0x7fc00000u : ((uint32_t)sb_byte << 23));
            const float sc = __uint_as_float(sc_bits);
            const int kk0 = sub * 2;
            const int kk1 = sub * 2 + 1;
            uint32_t b0[NT_PER_WARP][2], b1[NT_PER_WARP][2];
            const int nr_base = warp_n * (NT_PER_WARP * 8) + g_lane;
            const int b_off0 = woa_b_off(nr_base, kk0);
            const int b_off1 = woa_b_off(nr_base, kk1);
#pragma unroll
            for (int nt = 0; nt < NT_PER_WARP; ++nt) {
                const uint16_t* r0 = reinterpret_cast<const uint16_t*>(&sb[b_off0 + nt * 32]);
                const uint16_t* r1 = reinterpret_cast<const uint16_t*>(&sb[b_off1 + nt * 32]);
                b0[nt][0] = woa_cvt2_e4m3_h2(r0[t]);
                b0[nt][1] = woa_cvt2_e4m3_h2(r0[t + 4]);
                b1[nt][0] = woa_cvt2_e4m3_h2(r1[t]);
                b1[nt][1] = woa_cvt2_e4m3_h2(r1[t + 4]);
            }
            const int r_base = warp_m * (MT_PER_WARP * 16) + lr;
            const uint32_t s_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(sa + woa_a_off(r_base, kk0 * 2 + lc)));
            const uint32_t s_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(sa + woa_a_off(r_base, kk1 * 2 + lc)));
#pragma unroll
            for (int mt = 0; mt < MT_PER_WARP; ++mt) {
                uint32_t a0[4], a1[4];
                woa_ldsm_x4_step(a0, s_a0, mt * 2);
                woa_ldsm_x4_step(a1, s_a1, mt * 2);
#pragma unroll
                for (int nt = 0; nt < NT_PER_WARP; ++nt) {
                    float p[4];
                    woa_mma16816_zero(p, a0, b0[nt]);
                    woa_mma16816(p, a1, b1[nt]);
                    acc[mt][nt][0] = fmaf(p[0], sc, acc[mt][nt][0]);
                    acc[mt][nt][1] = fmaf(p[1], sc, acc[mt][nt][1]);
                    acc[mt][nt][2] = fmaf(p[2], sc, acc[mt][nt][2]);
                    acc[mt][nt][3] = fmaf(p[3], sc, acc[mt][nt][3]);
                }
            }
        }
    }
    woa_cp_async_wait<0>();

    __half* y = static_cast<__half*>(out);
#pragma unroll
    for (int mt = 0; mt < MT_PER_WARP; ++mt) {
        const int r0 = m0 + (warp_m * MT_PER_WARP + mt) * 16 + g_lane;
        const int r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP; ++nt) {
            const int col = n_warp + nt * 8 + 2 * t;
            if (col < o_lora) {
                if (r0 < rows) {
                    *reinterpret_cast<__half2*>(y + (size_t)r0 * out_stride + (size_t)g * (size_t)o_lora + col) =
                        __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
                }
                if (r1 < rows) {
                    *reinterpret_cast<__half2*>(y + (size_t)r1 * out_stride + (size_t)g * (size_t)o_lora + col) =
                        __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
                }
            }
        }
    }
}

/// Softmax-gate pooling: `out[j][d] = sum_p softmax_p(score[p][d]) * kv[p][d]` over group `j`'s `ratio` rows, per
/// dimension `d` — the softmax runs along the group axis, so every dimension has its own gate. Prefill only: `j`'s
/// members are rows `j * ratio .. j * ratio + ratio - 1` of `kv`/`score`. The norm is applied by the caller.
__global__ void pool_kernel(const void* kv, const void* score, void* out, int head_dim, int ratio, int dtype) {
    const int j = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }
    float mx = -1e30f;
    for (int p = 0; p < ratio; ++p) {
        mx = fmaxf(mx, ld(score, dtype, (size_t)(j * ratio + p) * head_dim + d));
    }
    float sum = 0.0f, acc = 0.0f;
    for (int p = 0; p < ratio; ++p) {
        const size_t at = (size_t)(j * ratio + p) * head_dim + d;
        const float e = expf(ld(score, dtype, at) - mx);
        sum += e;
        acc += e * ld(kv, dtype, at);
    }
    st(out, dtype, (size_t)j * head_dim + d, acc / sum);
}

/// The decode form: pool the completed group out of the state (which [`state_kernel`] filled slot by slot).
__global__ void pool_state_kernel(const float* kv_state, const float* score_state, void* out, int head_dim,
                                  int ratio, int dtype) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }
    float mx = -1e30f;
    for (int p = 0; p < ratio; ++p) {
        mx = fmaxf(mx, score_state[p * head_dim + d]);
    }
    float sum = 0.0f, acc = 0.0f;
    for (int p = 0; p < ratio; ++p) {
        const size_t at = (size_t)p * head_dim + d;
        const float e = expf(score_state[at] - mx);
        sum += e;
        acc += e * kv_state[at];
    }
    st(out, dtype, d, acc / sum);
}

/// One token into slot `slot` of the carry-over state (both the value and its gate score).
__global__ void state_kernel(const void* kv, const void* score, float* kv_state, float* score_state, int head_dim,
                             int slot, int dtype) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }
    kv_state[(size_t)slot * head_dim + d] = ld(kv, dtype, d);
    score_state[(size_t)slot * head_dim + d] = ld(score, dtype, d);
}

// ---------------------------------------------------------------------------------------------------- indexer

/// `score[r][p] = sum_h relu(q[r][h] . k[p]) * w[r][h]`, `-inf` past the row's reachable positions and outside the
/// level-1 candidate blocks when `mask` is given (one byte per position).
__global__ void index_score_kernel(const void* q, const void* index_k, const float* w, const uint8_t* mask,
                                   int mask_stride, float* score, int rows, int n_pos, int n_heads, int dim,
                                   float scale, int start_pos, int ratio, int dtype) {
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
    float acc = 0.0f;
    for (int h = 0; h < n_heads; ++h) {
        const size_t qb = ((size_t)r * n_heads + h) * dim;
        const size_t kb = (size_t)p * dim;
        float d = 0.0f;
        for (int i = 0; i < dim; ++i) {
            d += ld(q, dtype, qb + i) * ld(index_k, dtype, kb + i);
        }
        acc += fmaxf(d, 0.0f) * w[(size_t)r * n_heads + h];
    }
    score[(size_t)r * n_pos + p] = acc * scale;
}

// ---------------------------------------------------------------------------------------------------- top-k

/// Monotone f32 -> u32: `x >= 0 ? bits ^ 0x80000000 : ~bits`, so the unsigned order is the float order.
__device__ __forceinline__ uint32_t key_of(float x) {
    const uint32_t b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}

/// Inclusive scan of `v` over the block (Hillis-Steele in shared memory); returns the exclusive prefix for `tid`.
__device__ __forceinline__ int block_scan(int* sh, int v, int tid) {
    sh[tid] = v;
    __syncthreads();
    for (int s = 1; s < 256; s <<= 1) {
        const int t = (tid >= s) ? sh[tid - s] : 0;
        __syncthreads();
        sh[tid] += t;
        __syncthreads();
    }
    const int incl = sh[tid];
    __syncthreads();
    return incl - v;
}

/// The `topk` best positions of one row, ascending, `+ offset`; `-1` past the count.
///
/// A radix select over the f32 score bits: four 8-bit passes find the exact `topk`-th value `t`, then every position
/// with `score > t` is emitted in index order and the remaining slots are filled with `score == t` in index order.
/// `topk` is a runtime argument, which is the whole point: the GLM-5.3 kernel this replaces derives its selection
/// length from `DSA_TOPK`/`DSA_SEL_POOLS` and cannot be asked for 512.
///
/// One block per row, 256 threads, each owning a contiguous chunk of the row so the emitted order is the index order.
__global__ void topk_kernel(const float* score, int* out, int n_pos, int topk, int col_offset, int val_offset,
                            int row_stride) {
    __shared__ int hist[256];
    __shared__ int scan[256];
    __shared__ uint32_t thr;
    __shared__ int above;
    __shared__ uint32_t rlo, rhi;
    __shared__ int rneed;
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = score + (size_t)r * n_pos;
    int* dst = out + (size_t)r * row_stride + col_offset;
    const int k = min(topk, n_pos);

    for (int i = tid; i < topk; i += blockDim.x) {
        dst[i] = -1;
    }
    if (k <= 0) {
        return;
    }
    const int per = (n_pos + blockDim.x - 1) / blockDim.x;
    const int i0 = tid * per;
    const int i1 = min(n_pos, i0 + per);

    if (tid == 0) {
        rlo = 0u;
        rhi = 0xFFFFFFFFu;
        rneed = k;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += blockDim.x) {
            hist[b] = 0;
        }
        __syncthreads();
        for (int i = i0; i < i1; ++i) {
            const uint32_t u = key_of(row[i]);
            if (u >= rlo && u <= rhi) {
                atomicAdd(&hist[(u >> shift) & 0xFF], 1);
            }
        }
        __syncthreads();
        if (tid == 0) {
            int b = 255, cum = 0;
            for (; b >= 0; --b) {
                if (cum + hist[b] >= rneed) {
                    break;
                }
                cum += hist[b];
            }
            // `m` is the mask of the undecided low bits BELOW `shift` (0x00FFFFFF, 0x0000FFFF,
            // 0x000000FF, 0); `(uint32_t)b << shift` sits in bits `shift .. shift+7` (inside `~m`),
            // so masking it by `m` zeroed `rlo` on every pass and left the threshold at 0.
            const uint32_t m = (shift == 0) ? 0u : (0xFFFFFFFFu >> (32 - shift));
            rlo |= (uint32_t)b << shift;
            rhi = rlo | m;
            rneed -= cum;
        }
        __syncthreads();
    }
    if (tid == 0) {
        thr = rlo;
        above = k - rneed;
    }
    __syncthreads();
    const uint32_t t = thr;

    int c_gt = 0, c_eq = 0;
    for (int i = i0; i < i1; ++i) {
        const uint32_t u = key_of(row[i]);
        c_gt += (u > t);
        c_eq += (u == t);
    }
    const int off_gt = block_scan(scan, c_gt, tid);
    __syncthreads();
    const int off_eq = block_scan(scan, c_eq, tid);
    __syncthreads();
    int w = off_gt;
    for (int i = i0; i < i1; ++i) {
        if (key_of(row[i]) > t) {
            // defense in depth: a threshold that ever lands below the k-th key must not write past
            // the k slots into whatever the row's tail holds
            if (w < k) {
                dst[w] = (row[i] == -INFINITY) ? -1 : i + val_offset;
            }
            ++w;
        }
    }
    int w2 = above + off_eq;
    for (int i = i0; i < i1; ++i) {
        if (key_of(row[i]) == t) {
            if (w2 < k) {
                dst[w2] = (row[i] == -INFINITY) ? -1 : i + val_offset;
            }
            ++w2;
        }
    }
}

/// `get_window_topk_idxs` (model.py:410) into the first `win_slots` slots of every row: the sliding-window cache slots each
/// query may attend to, `-1` where the ring holds nothing yet. The compressed selection follows in the same row.
__global__ void window_idxs_kernel(int* out, int split, int row_stride, int win, int start_pos, int cmp_topk, int ratio) {
    const int r = blockIdx.x;
    int* row = out + (size_t)r * row_stride;
    const bool chunk_only = (start_pos == 0) || (split <= (int)gridDim.x && (gridDim.x > 1 || split < win));
    const int win_slots = chunk_only ? min(split, win) : win;
    for (int j = threadIdx.x; j < win_slots; j += blockDim.x) {
        int t;
        if (chunk_only) {
            t = max(0, r + 1 - win) + j;
            if (t > r) {
                t = -1;
            }
        } else if (split == win) {
            const int oldest = start_pos % win + 1;
            t = (j < win - oldest) ? oldest + j : j - (win - oldest);
            if (t > start_pos) {
                t = -1;
            }
        } else {
            const int p = (start_pos + r + 1 - win) + j;
            if (p < 0) {
                t = -1;
            } else if (p < start_pos) {
                t = p % win;
            } else {
                t = win + (p - start_pos);
            }
        }
        row[j] = t;
    }
    if (cmp_topk > 0 && ratio > 0) {
        const int len = (start_pos + r + 1) / ratio;
        int* dst = row + win_slots;
        for (int i = threadIdx.x; i < cmp_topk; i += blockDim.x) {
            dst[i] = (i < len) ? (split + i) : -1;
        }
    }
}

// ---------------------------------------------------------------------------------------------------- level 1 blocks

__global__ void candidate_score_kernel(const float* logits, float* bscore, int width, int start_pos,
                                       int ratio, int block_size) {
    // one CTA row of block-max scores into gmem: the per-row block score array scales with the
    // context (n_blocks = width / block_size) and no longer fits shared memory past ~12K blocks
    // (48 KB default; 512K-1M contexts need 256-512 KB), so selection reads it back from gmem.
    const int r = blockIdx.y;
    const int n_blocks = (width + block_size - 1) / block_size;
    const float* row = logits + (size_t)r * width;
    float* brow = bscore + (size_t)r * n_blocks;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < n_blocks; b += gridDim.x * blockDim.x) {
        float m = -INFINITY;
        const int hi = min(width, (b + 1) * block_size);
        for (int i = b * block_size; i < hi; ++i) {
            m = fmaxf(m, row[i]);
        }
        brow[b] = m;
    }
}

__global__ void candidate_select_kernel(const float* bscore, uint8_t* mask, int mask_stride, int width,
                                        int start_pos, int ratio, int topk_blocks, int block_size) {
    __shared__ int hist[256];
    __shared__ uint32_t thr;
    __shared__ int above;
    __shared__ uint32_t rlo, rhi;
    __shared__ int rneed;
    __shared__ int nkept;
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    uint8_t* mrow = mask + (size_t)r * mask_stride;
    const int n_blocks = (width + block_size - 1) / block_size;
    float* brow = const_cast<float*>(bscore) + (size_t)r * n_blocks;

    // the block with this query's newest position is only partly filled, so pin it in: it holds the
    // most recent tokens but could otherwise be outscored by an older, full block
    const int compress_len = (start_pos + r + 1) / ratio;
    const int last = (compress_len > 0) ? (compress_len - 1) / block_size : -1;
    if (tid == 0 && last >= 0 && last < n_blocks) {
        brow[last] = INFINITY;
    }
    __syncthreads();
    const int keep = min(topk_blocks, n_blocks);
    const uint32_t neg_inf_key = key_of(-INFINITY);
    if (keep >= n_blocks) {
        for (int i = tid; i < width; i += blockDim.x) {
            mrow[i] = (brow[i / block_size] > -INFINITY) ? 1u : 0u;
        }
        return;
    }
    if (tid == 0) {
        rlo = 0u;
        rhi = 0xFFFFFFFFu;
        rneed = keep;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += blockDim.x) {
            hist[b] = 0;
        }
        __syncthreads();
        for (int b = tid; b < n_blocks; b += blockDim.x) {
            const uint32_t u = key_of(brow[b]);
            if (u >= rlo && u <= rhi) {
                atomicAdd(&hist[(u >> shift) & 0xFF], 1);
            }
        }
        __syncthreads();
        if (tid == 0) {
            int b = 255, cum = 0;
            for (; b >= 0; --b) {
                if (cum + hist[b] >= rneed) {
                    break;
                }
                cum += hist[b];
            }
            // `m` is the mask of the undecided low bits BELOW `shift` (0x00FFFFFF, 0x0000FFFF,
            // 0x000000FF, 0); `(uint32_t)b << shift` sits in bits `shift .. shift+7` (inside `~m`),
            // so masking it by `m` zeroed `rlo` on every pass and left the threshold at 0.
            const uint32_t m = (shift == 0) ? 0u : (0xFFFFFFFFu >> (32 - shift));
            rlo |= (uint32_t)b << shift;
            rhi = rlo | m;
            rneed -= cum;
        }
        __syncthreads();
    }
    if (tid == 0) {
        thr = rlo;
        int gt = 0;
        for (int b = 0; b < n_blocks; ++b) {
            if (key_of(brow[b]) > rlo) {
                ++gt;
            }
        }
        above = keep - gt;
    }
    __syncthreads();
    const uint32_t t = thr;
    if (tid == 0) {
        nkept = 0;
    }
    __syncthreads();
    // one decision per block: every position of a kept block takes the same bit, so the mask stays
    // block-uniform, and the shared counter clamps the emit at `keep` blocks even if a threshold
    // regression would otherwise publish more than that
    for (int b = tid; b < n_blocks; b += blockDim.x) {
        const uint32_t u = key_of(brow[b]);
        bool ok = false;
        if (u > neg_inf_key) {
            if (u > t) {
                ok = true;
            } else if (u == t && above > 0) {
                int eq_before = 0;
                for (int j = 0; j < b; ++j) {
                    if (key_of(brow[j]) == t) {
                        ++eq_before;
                    }
                }
                ok = (eq_before < above);
            }
        }
        // Tie-break is deterministic: the `eq_before < above` rule keeps the LOWEST-index tied
        // blocks. (The old `atomicAdd(&nkept, 1) < keep` gate made the *set* of kept tied blocks
        // depend on warp arrival order, so identical prompts could sample different candidates
        // across runs once ties straddled the threshold.)
        const int hi = min(width, (b + 1) * block_size);
        for (int i = b * block_size; i < hi; ++i) {
            mrow[i] = ok ? 1u : 0u;
        }
    }
}

// ---------------------------------------------------------------------------------------------------- plane writes

__global__ void plane_write_kernel(const void* src, void* plane, int width, int dst_start, int dtype) {
    // Raw 16-bit copies (f16 or the halves of an f32 row): the f16 convert round-trip
    // (`__half2float` -> `__float2half_rn`) canonicalizes the ~1.56% of random 16-bit lanes
    // that decode as NaN halves, silently mutating f32 buffers (streams, hc pre/post/comb,
    // moe_out) copied through this kernel.
    (void)dtype;
    const int r = blockIdx.y;
    const uint16_t* sp = (const uint16_t*)src;
    uint16_t* dp = (uint16_t*)plane;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < width; c += blockDim.x * gridDim.x) {
        dp[((size_t)dst_start + r) * width + c] = sp[(size_t)r * width + c];
    }
}

__global__ void ring_write_kernel(const void* src, void* ring, int width, int slot0, int n_slots, int dtype) {
    // Raw 16-bit copies; see plane_write_kernel.
    (void)dtype;
    const int r = blockIdx.y;
    const int slot = (slot0 + r) % n_slots;
    const uint16_t* sp = (const uint16_t*)src;
    uint16_t* dp = (uint16_t*)ring;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < width; c += blockDim.x * gridDim.x) {
        dp[(size_t)slot * width + c] = sp[(size_t)r * width + c];
    }
}

// Device-params variant (DSV4.1 speculative round graphs): the ring slot comes from the round's `start_pos` and
// `window` in the params block, so a captured graph replays as `start_pos` advances.
__global__ void ring_write_params_kernel(const void* src, void* ring, int width, const Dsv4MtpRoundParams* p, int dtype) {
    // Raw 16-bit copies; see plane_write_kernel.
    (void)dtype;
    const int r = blockIdx.y;
    const int slot0 = mtp_params_slot0(p);
    const int n_slots = p->window;
    const int slot = (slot0 + r) % n_slots;
    const uint16_t* sp = (const uint16_t*)src;
    uint16_t* dp = (uint16_t*)ring;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < width; c += blockDim.x * gridDim.x) {
        dp[(size_t)slot * width + c] = sp[(size_t)r * width + c];
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- launchers

GB_EXPORT int gb_dsv4_plane_write(const void* src, void* plane, int rows, int width, int dst_start, int dtype,
                                  void* stream);
GB_EXPORT int gb_dsv4_ring_write(const void* src, void* ring, int rows, int width, int slot0, int n_slots, int dtype,
                                 void* stream);

GB_EXPORT int gb_dsv4_rope(void* x, int dtype, int rows, int stride, int dim, int block, int sub, const float* cos,
                           const float* sin, int table_stride, int pos0, int pos_step, int inverse, void* stream) {
    if (rows <= 0 || dim <= 0 || dim % 2 != 0 || sub <= 0 || block < dim || stride < sub * block || pos_step < 1) {
        return (int)cudaErrorInvalidValue;
    }
    const int threads = min(max(dim / 2, 32), 128);
    rope_kernel<<<dim3(rows, sub), threads, 0, gb_stream(stream)>>>(x, dtype, stride, dim, block, sub, cos, sin,
                                                                   table_stride, pos0, pos_step, inverse);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_rope_params(void* x, int dtype, int rows, int stride, int dim, int block, int sub, const float* cos,
                                  const float* sin, int table_stride, int pos_off, int pos_step, int inverse,
                                  const void* params, void* stream) {
    if (rows <= 0 || dim <= 0 || dim % 2 != 0 || sub <= 0 || block < dim || stride < sub * block || pos_step < 1 ||
        !params) {
        return (int)cudaErrorInvalidValue;
    }
    const Dsv4MtpRoundParams* p = (const Dsv4MtpRoundParams*)params;
    const int threads = min(max(dim / 2, 32), 128);
    rope_params_kernel<<<dim3(rows, sub), threads, 0, gb_stream(stream)>>>(
        x, dtype, stride, dim, block, sub, cos, sin, table_stride, pos_off, pos_step, inverse, p);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_rmsnorm_f32_to_f16(const float* x, const float* w, void* y, int rows, int cols, float eps,
                                         void* stream) {
    if (rows <= 0 || cols <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    rmsnorm_f32_to_f16_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>(x, w, y, cols, eps);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_rmsnorm(void* x, const float* w, int dtype, int rows, int width, float eps, void* stream) {
    if (rows <= 0 || width <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    rmsnorm_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>(x, w, dtype, width, eps);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_quant(void* x, int dtype, int rows, int width, int block, int bits, int pow2_scale,
                            void* stream) {
    if (rows <= 0 || block <= 0 || width % block != 0 || (bits != 8 && bits != 4)) {
        return (int)cudaErrorInvalidValue;
    }
    if (dtype == 0 && block == 32) {
        const int n_groups = width / 32;
        const int g_per_cta = (256 / 16) * 4;
        const int gx = (n_groups + g_per_cta - 1) / g_per_cta;
        quant_warp_f16_b32_kernel<<<dim3(gx, rows), 256, 0, gb_stream(stream)>>>(x, width, n_groups, bits, pow2_scale);
        return GB_CHECK_LAUNCH();
    }
    if (block == 16 || block == 32) {
        const int n_groups = width / block;
        const int g_per_cta = 256 / block;
        const int gx = (n_groups + g_per_cta - 1) / g_per_cta;
        quant_warp_kernel<<<dim3(gx, rows), 256, 0, gb_stream(stream)>>>(x, dtype, width, block, n_groups, bits, pow2_scale);
        return GB_CHECK_LAUNCH();
    }
    const int threads = min(block, THREADS);
    quant_kernel<<<dim3(width / block, rows), threads, 0, gb_stream(stream)>>>(x, dtype, width, block, bits,
                                                                              pow2_scale);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_attn(const void* q, const void* kv_win, const void* kv_cmp, int split, const float* sink,
                           const int* idx, int idx_stride, int idx_len, void* out, int rows, int n_heads,
                           int head_dim, float scale, int dtype, void* stream) {
    if (rows <= 0 || n_heads <= 0 || head_dim <= 0 || head_dim > 128 * DSV4_ATTN_VPT) {
        return (int)cudaErrorInvalidValue;
    }
    const int threads = 128;
    attn_kernel<<<dim3(rows, n_heads), threads, 0, gb_stream(stream)>>>(q, kv_win, kv_cmp, split, sink, idx,
                                                                       idx_stride, idx_len, out, n_heads, head_dim,
                                                                       scale, dtype);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_woa_grouped(const void* o, const uint8_t* w, const uint8_t* s, void* out, int rows, int groups,
                                  int o_lora, int o_in, int dtype, int use_classic, int force_mma, void* stream) {
    if (rows <= 0 || groups <= 0 || o_lora % 32 != 0 || o_in % 32 != 0) {
        return (int)cudaErrorInvalidValue;
    }
    const unsigned gx = (unsigned)((o_lora + WOA_TILE_L - 1) / WOA_TILE_L);
    const unsigned gy = (unsigned)groups;
    cudaStream_t st = gb_stream(stream);
    const bool small_scalar = !force_mma;
    if (small_scalar && !use_classic && rows <= 8) {
        // Production decode path: the pipelined woa kernel (same bits as the classic one; see
        // woa_kernel_sk). The classic kernel remains the test-only reference behind `use_classic`.
        switch (rows) {
            case 1: woa_kernel_sk<1><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype); break;
            case 2: woa_kernel_sk<2><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype); break;
            case 3: woa_kernel_sk<3><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype); break;
            case 4: woa_kernel_sk<4><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype); break;
            case 5: woa_kernel_sk<5><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype); break;
            case 6: woa_kernel_sk<6><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype); break;
            default: woa_kernel_sk<8><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype); break;
        }
        return GB_CHECK_LAUNCH();
    }
    if (small_scalar && rows <= 16) {
        switch (rows) {
            case 1:
                woa_kernel<1><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
                break;
            case 2:
                woa_kernel<2><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
                break;
            case 3:
                woa_kernel<3><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
                break;
            case 4:
                woa_kernel<4><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
                break;
            case 5:
                woa_kernel<5><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
                break;
            case 6:
                woa_kernel<6><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
                break;
            case 7:
            case 8: {
                woa_kernel<8><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
                break;
            }
            case 9:
            case 10:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
            case 16: {
                woa_kernel<16><<<dim3(gx, gy, 1), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
                break;
            }
        }
        return GB_CHECK_LAUNCH();
    }
    // Prefill slices (including forced m <= 16) run the tensor-core MMA path, so a row's bits do not
    // depend on the prefill chunk grid. Decode / speculative-verify small rows keep the scalar kernels above.
    if (dtype == 0 && o_in >= 512 && (o_in % WOA_MMA_BK) == 0 && (o_lora % WOA_MMA_BN) == 0) {
        const int ctas_128 = ((rows + 127) / 128) * (o_lora / 128) * groups;
        if (rows >= 256 && ctas_128 >= 48) {
            const unsigned gx_cp = (unsigned)((rows + 127) / 128);
            const unsigned gy_cp = (unsigned)(o_lora / 128);
            woa_cp_mma_bk32_kernel<128, 3><<<dim3(gx_cp, gy_cp, gy), WOA_MMA_THREADS, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in);
        } else {
            const unsigned gx_cp = (unsigned)((rows + WOA_MMA_BM - 1) / WOA_MMA_BM);
            const unsigned gy_cp = (unsigned)(o_lora / WOA_MMA_BN);
            woa_cp_mma_kernel<64, 128, 1, 8, 3, 2><<<dim3(gx_cp, gy_cp, gy), WOA_MMA_THREADS, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in);
        }
    } else if (o_in < 512) {
        const unsigned gz = (unsigned)((rows + 7) / 8);
        woa_kernel<8><<<dim3(gx, gy, gz), WOA_WARPS * 32, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
    } else if (rows <= 16) {
        woa_mma_kernel<1, 1><<<dim3(gx, gy, 1), 128, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
    } else if (rows <= 32) {
        woa_mma_kernel<1, 2><<<dim3(gx, gy, 1), 256, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
    } else if (rows <= 64) {
        woa_mma_kernel<2, 2><<<dim3(gx, gy, 1), 256, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
    } else {
        const unsigned gz = (unsigned)((rows + 127) / 128);
        woa_mma_kernel<4, 2><<<dim3(gx, gy, gz), 256, 0, st>>>(o, w, s, out, rows, groups, o_lora, o_in, dtype);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_pool(const void* kv, const void* score, void* out, int rows, int head_dim, int ratio,
                           int n_out, int dtype, void* stream) {
    if (rows <= 0 || head_dim <= 0 || ratio < 1 || n_out < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (ratio == 1) {
        return gb_dsv4_plane_write(kv, out, rows, head_dim, 0, dtype, stream);
    }
    if (n_out == 0) {
        return 0;
    }
    pool_kernel<<<dim3(n_out, (head_dim + THREADS - 1) / THREADS), THREADS, 0, gb_stream(stream)>>>(kv, score, out,
                                                                                                   head_dim, ratio,
                                                                                                   dtype);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_pool_state(const float* kv_state, const float* score_state, void* out, int head_dim,
                                 int ratio, int dtype, void* stream) {
    if (head_dim <= 0 || ratio < 2) {
        return (int)cudaErrorInvalidValue;
    }
    pool_state_kernel<<<(head_dim + THREADS - 1) / THREADS, THREADS, 0, gb_stream(stream)>>>(kv_state, score_state,
                                                                                             out, head_dim, ratio,
                                                                                             dtype);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_state_write(const void* kv, const void* score, float* kv_state, float* score_state,
                                  int head_dim, int slot, int dtype, void* stream) {
    if (head_dim <= 0 || slot < 0) {
        return (int)cudaErrorInvalidValue;
    }
    state_kernel<<<(head_dim + THREADS - 1) / THREADS, THREADS, 0, gb_stream(stream)>>>(kv, score, kv_state,
                                                                                        score_state, head_dim, slot,
                                                                                        dtype);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_index_score(const void* q, const void* index_k, const float* w, const uint8_t* mask,
                                  int mask_stride, float* score, int rows, int n_pos, int n_heads, int dim,
                                  float scale, int start_pos, int ratio, int dtype, void* stream) {
    if (rows <= 0 || n_pos <= 0 || n_heads <= 0 || dim <= 0 || ratio <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    index_score_kernel<<<dim3(rows, (n_pos + 127) / 128), 128, 0, gb_stream(stream)>>>(
        q, index_k, w, mask, mask_stride, score, rows, n_pos, n_heads, dim, scale, start_pos, ratio, dtype);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_topk_select(const float* score, int* out, int rows, int n_pos, int topk, int col_offset,
                                  int val_offset, int row_stride, void* stream) {
    if (rows <= 0 || n_pos <= 0 || topk <= 0 || row_stride < col_offset + topk) {
        return (int)cudaErrorInvalidValue;
    }
    topk_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>(score, out, n_pos, topk, col_offset, val_offset, row_stride);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_window_idxs(int* out, int rows, int split, int row_stride, int win, int start_pos,
                                  void* stream) {
    const int win_slots = (start_pos == 0 || (rows > 1 && split <= rows)) ? min(split, win) : win;
    if (rows <= 0 || split <= 0 || win <= 0 || row_stride < win_slots) {
        return (int)cudaErrorInvalidValue;
    }
    window_idxs_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>(out, split, row_stride, win, start_pos, 0, 0);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_window_and_cmp_idxs(int* out, int rows, int split, int row_stride, int win, int start_pos,
                                          int cmp_topk, int ratio, void* stream) {
    const int win_slots = (start_pos == 0 || (rows > 1 && split <= rows)) ? min(split, win) : win;
    if (rows <= 0 || split <= 0 || win <= 0 || row_stride < win_slots + max(0, cmp_topk)) {
        return (int)cudaErrorInvalidValue;
    }
    window_idxs_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>(out, split, row_stride, win, start_pos, cmp_topk, ratio);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_candidate_blocks(const float* logits, float* bscore, uint8_t* mask, int mask_stride,
                                       int rows, int width, int start_pos, int ratio, int topk_blocks,
                                       int block_size, void* stream) {
    if (rows <= 0 || width <= 0 || block_size <= 0 || topk_blocks <= 0 || ratio <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (bscore == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    const int n_blocks = (width + block_size - 1) / block_size;
    const int tiles = (n_blocks + 4 * THREADS - 1) / (4 * THREADS);
    candidate_score_kernel<<<dim3(tiles, rows), THREADS, 0, gb_stream(stream)>>>(
        logits, bscore, width, start_pos, ratio, block_size);
    GB_CHECK_LAUNCH();
    candidate_select_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>(
        bscore, mask, mask_stride, width, start_pos, ratio, topk_blocks, block_size);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_plane_write(const void* src, void* plane, int rows, int width, int dst_start, int dtype,
                                  void* stream) {
    if (rows <= 0 || width <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int gx = min((width + 127) / 128, 32);
    plane_write_kernel<<<dim3(gx, rows), 128, 0, gb_stream(stream)>>>(src, plane, width, dst_start, dtype);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_ring_write(const void* src, void* ring, int rows, int width, int slot0, int n_slots, int dtype,
                                 void* stream) {
    if (rows <= 0 || width <= 0 || n_slots <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int gx = min((width + 127) / 128, 32);
    ring_write_kernel<<<dim3(gx, rows), 128, 0, gb_stream(stream)>>>(src, ring, width, slot0, n_slots, dtype);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_ring_write_params(const void* src, void* ring, int rows, int width, const void* params, int dtype,
                                        void* stream) {
    if (rows <= 0 || width <= 0 || !params) {
        return (int)cudaErrorInvalidValue;
    }
    // NOTE: `params` is a DEVICE pointer; never dereference it on the host (the kernels read it on the device).
    const Dsv4MtpRoundParams* p = (const Dsv4MtpRoundParams*)params;
    const int gx = min((width + 127) / 128, 32);
    ring_write_params_kernel<<<dim3(gx, rows), 128, 0, gb_stream(stream)>>>(src, ring, width, p, dtype);
    return GB_CHECK_LAUNCH();
}


GB_EXPORT int gb_dsv4_q_rope_rot_f16(void* x, int rows, int stride, int sub, const float* cos, const float* sin,
                                     int table_stride, int pos0, int pos_step, int do_rot, void* stream) {
    if (rows <= 0 || sub <= 0) return (int)cudaErrorInvalidValue;
    const int warps = (sub >= 8) ? 8 : 4;
    const int gx = (sub + warps - 1) / warps;
    q_rope_rot_f16_kernel<<<dim3(gx, rows), warps * 32, 0, gb_stream(stream)>>>(
        x, stride, sub, cos, sin, table_stride, pos0, pos_step, do_rot);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_o_rot_rope_quant_f16(void* x, int rows, int stride, int sub, const float* cos, const float* sin,
                                           int table_stride, int pos0, int pos_step, int do_rot, void* stream) {
    if (rows <= 0 || sub <= 0) return (int)cudaErrorInvalidValue;
    const int warps = (sub >= 8) ? 8 : 4;
    const int gx = (sub + warps - 1) / warps;
    o_rot_rope_quant_f16_kernel<<<dim3(gx, rows), warps * 32, 0, gb_stream(stream)>>>(
        x, stride, sub, cos, sin, table_stride, pos0, pos_step, do_rot);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_kv_prepare(void* x, const float* w, int rows, int width, int rope_dim, int rope_block,
                                 const float* cos, const float* sin, int table_stride, int pos0, int pos_step,
                                 int do_rot, float eps, void* stream) {
    if (rows <= 0 || width <= 0 || width % 32 != 0 || rope_dim <= 0 || rope_dim % 2 != 0 || rope_block < rope_dim ||
        rope_block % rope_dim != 0 || (width - rope_dim) % 64 != 0 || pos_step < 1) {
        return (int)cudaErrorInvalidValue;
    }
    kv_prepare_kernel<<<rows, THREADS, 0, gb_stream(stream)>>>(
        x, w, width, eps, cos, sin, table_stride, pos0, pos_step, rope_dim, rope_block, do_rot);
    return GB_CHECK_LAUNCH();
}
