// FP8 E4M3 block-scaled linear: y[m, n] = sum_k x[m, k] * W[n, k] * scale[n / 32][k / 32], fp32 accumulate, fp16 out.
#include "common.cuh"
#include <cuda_fp8.h>
#include <stdlib.h>

namespace {

/// 2^(b - 127); 0xFF is NaN in E8M0; 0 is 2^-127 (f32 subnormal).
__device__ __forceinline__ float e8m0_scale(uint8_t b) {
    const uint32_t bits = (b == 0u) ? 0x00400000u : ((b == 0xFFu) ? 0x7fc00000u : ((uint32_t)b << 23));
    return __uint_as_float(bits);
}

/// One E4M3 byte as f32 (the hardware conversion; ±0, subnormals and 448 all exact).
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

__device__ __forceinline__ float4 cvt4_e4m3(uint32_t u) {
    const __half2 h01 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u & 0xFFFFu), __NV_E4M3);
    const __half2 h23 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u >> 16), __NV_E4M3);
    const float2 f01 = __half22float2(h01);
    const float2 f23 = __half22float2(h23);
    return make_float4(f01.x, f01.y, f23.x, f23.y);
}

__device__ __forceinline__ float dot4_f32(float4 w, float4 x) {
    return fmaf(w.x, x.x, fmaf(w.y, x.y, fmaf(w.z, x.z, w.w * x.w)));
}

template <int X_DTYPE>
__device__ __forceinline__ float4 ld4_x(const void* x, size_t base) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(x) + base));
    if constexpr (X_DTYPE == 0) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        return make_float4(a.x, a.y, b.x, b.y);
    } else {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        return make_float4(a.x, a.y, b.x, b.y);
    }
}

constexpr int TILE = 32;     // scale tile width
constexpr int CHUNK_K = 512; // 32 lanes * 16 bytes (uint4) per step

template <int X_DTYPE>
__device__ __forceinline__ float4 cvt_u2_x(uint2 u) {
    if constexpr (X_DTYPE == 0) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        return make_float4(a.x, a.y, b.x, b.y);
    } else {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        return make_float4(a.x, a.y, b.x, b.y);
    }
}

template <int ROWS, int WARPS, int X_DTYPE>
__device__ __forceinline__ void fp8_gemm_cta(const void* __restrict__ x,
                                             const uint8_t* __restrict__ w,
                                             const uint8_t* __restrict__ s,
                                             __half* __restrict__ y,
                                             int m,
                                             int k,
                                             int n,
                                             int bx) {
    constexpr int BLOCK_COLS = WARPS * 4;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n0 = bx * BLOCK_COLS + warp * 4;
    const int r0 = blockIdx.y * ROWS;
    const int ktiles = k / TILE;
    const uint8_t* s_tile_row = s + (size_t)(n0 / TILE) * ktiles;

    float acc[ROWS][4];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    const int k_off = lane * 16;
    if constexpr (ROWS <= 8) {
        for (int k_base = 0; k_base < k; k_base += CHUNK_K) {
            const int kk = k_base + k_off;
            if (kk < k) {
                const float sc = e8m0_scale(__ldg(s_tile_row + (kk >> 5)));
                uint4 w_vec[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int nc = n0 + c;
                    w_vec[c] = (nc < n)
                                   ? __ldg(reinterpret_cast<const uint4*>(w + (size_t)nc * k + kk))
                                   : make_uint4(0u, 0u, 0u, 0u);
                }
                uint4 x_u4[ROWS][2];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    const int rc = r0 + r;
                    if (rc < m) {
                        const uint4* xp = reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(x) + (size_t)rc * k + kk);
                        x_u4[r][0] = __ldg(xp + 0);
                        x_u4[r][1] = __ldg(xp + 1);
                    } else {
                        x_u4[r][0] = make_uint4(0u, 0u, 0u, 0u);
                        x_u4[r][1] = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float4 wf0 = cvt4_e4m3(w_vec[c].x);
                    const float4 wf1 = cvt4_e4m3(w_vec[c].y);
                    const float4 wf2 = cvt4_e4m3(w_vec[c].z);
                    const float4 wf3 = cvt4_e4m3(w_vec[c].w);
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        const float4 x0 = cvt_u2_x<X_DTYPE>(make_uint2(x_u4[r][0].x, x_u4[r][0].y));
                        const float4 x1 = cvt_u2_x<X_DTYPE>(make_uint2(x_u4[r][0].z, x_u4[r][0].w));
                        const float4 x2 = cvt_u2_x<X_DTYPE>(make_uint2(x_u4[r][1].x, x_u4[r][1].y));
                        const float4 x3 = cvt_u2_x<X_DTYPE>(make_uint2(x_u4[r][1].z, x_u4[r][1].w));
                        const float dot = dot4_f32(wf0, x0) + dot4_f32(wf1, x1) +
                                          dot4_f32(wf2, x2) + dot4_f32(wf3, x3);
                        acc[r][c] = fmaf(dot, sc, acc[r][c]);
                    }
                }
            }
        }
    } else {
        __shared__ uint2 sx[ROWS][128];
        for (int k_base = 0; k_base < k; k_base += CHUNK_K) {
            for (int idx = tid; idx < ROWS * 128; idx += WARPS * 32) {
                const int r = idx >> 7;
                const int u = idx & 127;
                const int rc = r0 + r;
                const int kk_u = k_base + u * 4;
                sx[r][u] = (rc < m && kk_u < k)
                               ? __ldg(reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(x) + (size_t)rc * k + kk_u))
                               : make_uint2(0u, 0u);
            }
            __syncthreads();
            const int kk = k_base + k_off;
            if (kk < k) {
                const float sc = e8m0_scale(__ldg(s_tile_row + (kk >> 5)));
                uint4 w_vec[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int nc = n0 + c;
                    w_vec[c] = (nc < n)
                                   ? __ldg(reinterpret_cast<const uint4*>(w + (size_t)nc * k + kk))
                                   : make_uint4(0u, 0u, 0u, 0u);
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float4 wf0 = cvt4_e4m3(w_vec[c].x);
                    const float4 wf1 = cvt4_e4m3(w_vec[c].y);
                    const float4 wf2 = cvt4_e4m3(w_vec[c].z);
                    const float4 wf3 = cvt4_e4m3(w_vec[c].w);
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        const float4 x0 = cvt_u2_x<X_DTYPE>(sx[r][lane * 4 + 0]);
                        const float4 x1 = cvt_u2_x<X_DTYPE>(sx[r][lane * 4 + 1]);
                        const float4 x2 = cvt_u2_x<X_DTYPE>(sx[r][lane * 4 + 2]);
                        const float4 x3 = cvt_u2_x<X_DTYPE>(sx[r][lane * 4 + 3]);
                        const float dot = dot4_f32(wf0, x0) + dot4_f32(wf1, x1) +
                                          dot4_f32(wf2, x2) + dot4_f32(wf3, x3);
                        acc[r][c] = fmaf(dot, sc, acc[r][c]);
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int rc = r0 + r;
        if (rc < m) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float v = acc[r][c];
#pragma unroll
                for (int mask = 16; mask >= 1; mask >>= 1) {
                    v += __shfl_xor_sync(0xffffffffu, v, mask);
                }
                const int nc = n0 + c;
                if (lane == 0 && nc < n) {
                    y[(size_t)rc * n + nc] = __float2half(v);
                }
            }
        }
    }
}

template <int ROWS, int WARPS, int X_DTYPE>
__global__ void __launch_bounds__(WARPS * 32) fp8_gemm_kernel(const void* __restrict__ x,
                                                              const uint8_t* __restrict__ w,
                                                              const uint8_t* __restrict__ s,
                                                              __half* __restrict__ y,
                                                              int m,
                                                              int k,
                                                              int n) {
    fp8_gemm_cta<ROWS, WARPS, X_DTYPE>(x, w, s, y, m, k, n, (int)blockIdx.x);
}

template <int ROWS, int WARPS, int X_DTYPE>
__global__ void __launch_bounds__(WARPS * 32) fp8_gemm_pair_kernel(const void* __restrict__ x,
                                                                   const uint8_t* __restrict__ w1,
                                                                   const uint8_t* __restrict__ s1,
                                                                   __half* __restrict__ y1,
                                                                   int n1,
                                                                   const uint8_t* __restrict__ w2,
                                                                   const uint8_t* __restrict__ s2,
                                                                   __half* __restrict__ y2,
                                                                   int n2,
                                                                   int m,
                                                                   int k,
                                                                   int nb1) {
    const int bx = (int)blockIdx.x;
    if (bx < nb1) {
        fp8_gemm_cta<ROWS, WARPS, X_DTYPE>(x, w1, s1, y1, m, k, n1, bx);
    } else {
        fp8_gemm_cta<ROWS, WARPS, X_DTYPE>(x, w2, s2, y2, m, k, n2, bx - nb1);
    }
}

// ------------------------------------------------------------------ small-M fast path
//
// The decode-shaped FP8 GEMMs (shared expert, MLA projections, dense layers) run at rows 1-8 against
// hundreds of MB of weights. The classic kernel above is memory-latency-bound at these shapes: each CTA
// sweeps all of K with a handful of 16-byte loads in flight, and at rows=1 the grid (N/8 blocks) does not
// even fill the 128 SMs. This variant keeps the EXACT accumulation order (each output is still summed over
// K in ascending tile order by one CTA, with the same per-tile fmaf chain, scale and conversions), but
//   - narrows the per-warp column tile from 4 to CPW (more CTAs per SM, fewer idle SMs at rows=1),
//   - software-pipelines every K chunk: the NEXT chunk's weights and x are prefetched into registers while
//     the current chunk accumulates — no shared memory, no barriers, so the warps free-run exactly like the
//     classic kernel's while each warp has 2x the loads in flight, and
//   - keeps 128-bit loads and the same warp-shuffle reduction and store.
//
// Bit-exactness: only the load timing moves. Each output (r, col) is still summed by one lane over k in
// the same ascending 512-wide chunks with the same dot4 chain, scale multiply, shuffle-reduction order and
// fp16 conversion, so the bits are identical to the classic kernel's by construction.

__device__ __forceinline__ void cp_async16_fp8(void* smem, const void* gmem, bool valid) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit_fp8() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait_fp8() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <int ROWS, int WARPS, int CPW, int X_DTYPE>
__device__ __forceinline__ void fp8_gemm_sk_cta(const void* __restrict__ x,
                                                const uint8_t* __restrict__ w,
                                                const uint8_t* __restrict__ s,
                                                __half* __restrict__ y,
                                                int m,
                                                int k,
                                                int n,
                                                int bx) {
    constexpr int BLOCK_COLS = WARPS * CPW;
    constexpr int THREADS = WARPS * 32;
    constexpr bool PIPE = (ROWS <= 4); // rows>=5 double-buffer the x chunk in registers, no spill
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n0 = bx * BLOCK_COLS + warp * CPW;
    const int r0 = blockIdx.y * ROWS;
    const int ktiles = k / TILE;
    const uint8_t* s_tile_row = s + (size_t)(n0 / TILE) * ktiles;

    float acc[ROWS][CPW];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    const int k_off = lane * 16;
    const int nchunks = (k + CHUNK_K - 1) / CHUNK_K;

    auto load_chunk = [&](int ci, uint4 (&wv)[CPW], uint4 (&xv)[ROWS][2]) {
        const int k_base = ci * CHUNK_K;
        const int kk = k_base + k_off;
        const bool ok = kk < k;
        const int nc_clamp = (n0 < n) ? n0 : 0;
        const int w_off = (ok && nc_clamp + CPW <= n) ? kk : 0;
        const int rc_clamp = (r0 + ROWS <= m) ? r0 : 0;
        const size_t x_off = (size_t)rc_clamp * k + (ok ? kk : 0);
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const int nc = n0 + c;
            const bool col_ok = (nc < n);
            const uint4* wp = reinterpret_cast<const uint4*>(w + (size_t)(col_ok ? nc : nc_clamp) * k + w_off);
            wv[c] = (col_ok && ok) ? __ldg(wp) : make_uint4(0u, 0u, 0u, 0u);
        }
        const uint4* xp = reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(x) + x_off);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int rc = r0 + r;
            const bool row_ok = (rc < m);
            const uint4* xr = row_ok ? xp + (size_t)r * (k / 8) : xp;
            xv[r][0] = (row_ok && ok) ? __ldg(xr) : make_uint4(0u, 0u, 0u, 0u);
            xv[r][1] = (row_ok && ok) ? __ldg(xr + 1) : make_uint4(0u, 0u, 0u, 0u);
        }
    };

    uint4 w_cur[CPW], w_next[CPW];
    uint4 x_cur[ROWS][2], x_next[ROWS][2];
    if constexpr (PIPE) {
        load_chunk(0, w_cur, x_cur);
    }

    for (int ci = 0; ci < nchunks; ++ci) {
        if constexpr (PIPE) {
            if (ci + 1 < nchunks) {
                load_chunk(ci + 1, w_next, x_next);
            }
        } else {
            load_chunk(ci, w_cur, x_cur);
        }
        const int k_base = ci * CHUNK_K;
        const int kk = k_base + k_off;
        if (kk < k) {
            const float sc = e8m0_scale(__ldg(s_tile_row + (kk >> 5)));
#pragma unroll
            for (int c = 0; c < CPW; ++c) {
                const float4 wf0 = cvt4_e4m3(w_cur[c].x);
                const float4 wf1 = cvt4_e4m3(w_cur[c].y);
                const float4 wf2 = cvt4_e4m3(w_cur[c].z);
                const float4 wf3 = cvt4_e4m3(w_cur[c].w);
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    const float4 x0 = cvt_u2_x<X_DTYPE>(make_uint2(x_cur[r][0].x, x_cur[r][0].y));
                    const float4 x1 = cvt_u2_x<X_DTYPE>(make_uint2(x_cur[r][0].z, x_cur[r][0].w));
                    const float4 x2 = cvt_u2_x<X_DTYPE>(make_uint2(x_cur[r][1].x, x_cur[r][1].y));
                    const float4 x3 = cvt_u2_x<X_DTYPE>(make_uint2(x_cur[r][1].z, x_cur[r][1].w));
                    const float dot = dot4_f32(wf0, x0) + dot4_f32(wf1, x1) +
                                      dot4_f32(wf2, x2) + dot4_f32(wf3, x3);
                    acc[r][c] = fmaf(dot, sc, acc[r][c]);
                }
            }
        }
        if (ci + 1 < nchunks && PIPE) {
#pragma unroll
            for (int c = 0; c < CPW; ++c) w_cur[c] = w_next[c];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                x_cur[r][0] = x_next[r][0];
                x_cur[r][1] = x_next[r][1];
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int rc = r0 + r;
        if (rc < m) {
#pragma unroll
            for (int c = 0; c < CPW; ++c) {
                float v = acc[r][c];
#pragma unroll
                for (int mask = 16; mask >= 1; mask >>= 1) {
                    v += __shfl_xor_sync(0xffffffffu, v, mask);
                }
                const int nc = n0 + c;
                if (lane == 0 && nc < n) {
                    y[(size_t)rc * n + nc] = __float2half(v);
                }
            }
        }
    }
}

template <int ROWS, int WARPS, int CPW, int X_DTYPE>
__global__ void __launch_bounds__(WARPS * 32) fp8_gemm_sk_kernel(const void* __restrict__ x,
                                                                  const uint8_t* __restrict__ w,
                                                                  const uint8_t* __restrict__ s,
                                                                  __half* __restrict__ y,
                                                                  int m,
                                                                  int k,
                                                                  int n) {
    fp8_gemm_sk_cta<ROWS, WARPS, CPW, X_DTYPE>(x, w, s, y, m, k, n, (int)blockIdx.x);
}

template <int ROWS, int WARPS, int CPW, int X_DTYPE>
__global__ void __launch_bounds__(WARPS * 32) fp8_gemm_pair_sk_kernel(const void* __restrict__ x,
                                                                      const uint8_t* __restrict__ w1,
                                                                      const uint8_t* __restrict__ s1,
                                                                      __half* __restrict__ y1,
                                                                      int n1,
                                                                      const uint8_t* __restrict__ w2,
                                                                      const uint8_t* __restrict__ s2,
                                                                      __half* __restrict__ y2,
                                                                      int n2,
                                                                      int m,
                                                                      int k,
                                                                      int nb1) {
    const int bx = (int)blockIdx.x;
    if (bx < nb1) {
        fp8_gemm_sk_cta<ROWS, WARPS, CPW, X_DTYPE>(x, w1, s1, y1, m, k, n1, bx);
    } else {
        fp8_gemm_sk_cta<ROWS, WARPS, CPW, X_DTYPE>(x, w2, s2, y2, m, k, n2, bx - nb1);
    }
}

// ---------------------------------------------------------------------------------------------------- Tensor Core MMA prefill kernel

constexpr int MMA_THREADS = 256; // 8 warps
constexpr int MMA_BM = 64;       // 4 m16 tiles
constexpr int MMA_BN = 128;      // 8 warps * 16 cols (2 n8 tiles per warp)
constexpr int MMA_BK = 64;       // 2 scale tiles of 32 (4 k16 MMA steps)
constexpr int MMA_STAGES = 3;

__device__ __forceinline__ void ldsm_x4_fp8(uint32_t (&a)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}
template <int IMM_BYTES>
__device__ __forceinline__ void ldsm_x4_imm_fp8(uint32_t (&a)[4], uint32_t s_base) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4+%5];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s_base), "n"(IMM_BYTES));
}
__device__ __forceinline__ void ldsm_x4_step_fp8(uint32_t (&a)[4], uint32_t s_base, int step_1k) {
    switch (step_1k) {
        case 0: ldsm_x4_imm_fp8<0>(a, s_base); break;
        case 1: ldsm_x4_imm_fp8<1024>(a, s_base); break;
        case 2: ldsm_x4_imm_fp8<2048>(a, s_base); break;
        case 3: ldsm_x4_imm_fp8<3072>(a, s_base); break;
        case 4: ldsm_x4_imm_fp8<4096>(a, s_base); break;
        case 5: ldsm_x4_imm_fp8<5120>(a, s_base); break;
        case 6: ldsm_x4_imm_fp8<6144>(a, s_base); break;
        default: ldsm_x4_imm_fp8<7168>(a, s_base); break;
    }
}
__device__ __forceinline__ void mma16816_fp8_zero(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    const float z = 0.0f;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(z));
}
__device__ __forceinline__ void mma16816_fp8(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t cvt2_e4m3_h2(uint16_t b2) {
    const __half2 h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(b2), __NV_E4M3);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ int fp8_a_off(int r, int c) { return r * MMA_BK + ((c ^ (r & 7)) << 3); }
__device__ __forceinline__ int fp8_b_off(int nr, int c16) { return nr * 4 + (c16 ^ (nr & 3)); }
__device__ __forceinline__ int fp8_a_off32(int r, int c) { return r * 32 + ((c ^ ((r >> 1) & 3)) << 3); }
__device__ __forceinline__ int fp8_b_off32(int nr, int c16) { return nr * 2 + (c16 ^ ((nr >> 2) & 1)); }

__device__ __forceinline__ uint4 cvt8_bf16_to_f16(uint4 u) {
    const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.z));
    const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.w));
    const __half2 h0 = __floats2half2_rn(f0.x, f0.y);
    const __half2 h1 = __floats2half2_rn(f1.x, f1.y);
    const __half2 h2 = __floats2half2_rn(f2.x, f2.y);
    const __half2 h3 = __floats2half2_rn(f3.x, f3.y);
    return make_uint4(*reinterpret_cast<const uint32_t*>(&h0),
                      *reinterpret_cast<const uint32_t*>(&h1),
                      *reinterpret_cast<const uint32_t*>(&h2),
                      *reinterpret_cast<const uint32_t*>(&h3));
}

template <int BM, int BN, int M_WARPS, int N_WARPS, int STAGES, int MIN_BLOCKS, int X_DTYPE>
__global__ void __launch_bounds__(M_WARPS * N_WARPS * 32, MIN_BLOCKS)
    fp8_mma_kernel(const void* __restrict__ x,
                   const uint8_t* __restrict__ w,
                   const uint8_t* __restrict__ s,
                   __half* __restrict__ y,
                   int m,
                   int k,
                   int n) {
    constexpr int THREADS = M_WARPS * N_WARPS * 32;
    constexpr int MT_PER_WARP = (BM / 16) / M_WARPS;
    constexpr int NT_PER_WARP = (BN / 8) / N_WARPS;
    constexpr int A_STAGE_HALVES = BM * MMA_BK;
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
    const int g = lane >> 2;
    const int t = lane & 3;

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int ktiles = k / TILE;
    const int n_warp = n0 + warp_n * (NT_PER_WARP * 8);
    const uint8_t* s_row = s + (size_t)(n_warp / TILE) * ktiles;
    const int KC = k / MMA_BK;

    auto load_stage = [&](int stg, int kc) {
        __half* sa = sa_ring[stg];
        uint4* sb = sb_ring[stg];
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int idx = i * THREADS + tid;
            const int r = idx >> 3;
            const int c = idx & 7;
            const int gr = m0 + r;
            const bool ok = (gr < m);
            const size_t x_off = (size_t)(ok ? gr : 0) * k + kc * MMA_BK + c * 8;
            if constexpr (X_DTYPE == 0) {
                cp_async16_fp8(sa + fp8_a_off(r, c), static_cast<const __half*>(x) + x_off, ok);
            } else {
                const uint4 raw = ok ? __ldg(reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(x) + x_off))
                                     : make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(sa + fp8_a_off(r, c)) = cvt8_bf16_to_f16(raw);
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
            const int idx = i * THREADS + tid;
            const int nr = idx >> 2;
            const int c16 = idx & 3;
            const int gn = n0 + nr;
            cp_async16_fp8(&sb[fp8_b_off(nr, c16)], w + (size_t)gn * k + kc * MMA_BK + c16 * 16, true);
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
        cp_async_commit_fp8();
    }

    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int lc = lane >> 4;

#pragma unroll 1
    for (int kc = 0; kc < KC; ++kc) {
        cp_async_wait_fp8<STAGES - 2>();
        __syncthreads();
        const int nk = kc + STAGES - 1;
        if (nk < KC) load_stage(nk % STAGES, nk);
        cp_async_commit_fp8();

        const __half* sa = sa_ring[kc % STAGES];
        const uint4* sb = sb_ring[kc % STAGES];

#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const float sc = e8m0_scale(__ldg(s_row + kc * 2 + sub));
            const int kk0 = sub * 2;
            const int kk1 = sub * 2 + 1;
            uint32_t b0[NT_PER_WARP][2], b1[NT_PER_WARP][2];
            const int nr_base = warp_n * (NT_PER_WARP * 8) + g;
            const int b_off0 = fp8_b_off(nr_base, kk0);
            const int b_off1 = fp8_b_off(nr_base, kk1);
#pragma unroll
            for (int nt = 0; nt < NT_PER_WARP; ++nt) {
                const uint16_t* r0 = reinterpret_cast<const uint16_t*>(&sb[b_off0 + nt * 32]);
                const uint16_t* r1 = reinterpret_cast<const uint16_t*>(&sb[b_off1 + nt * 32]);
                b0[nt][0] = cvt2_e4m3_h2(r0[t]);
                b0[nt][1] = cvt2_e4m3_h2(r0[t + 4]);
                b1[nt][0] = cvt2_e4m3_h2(r1[t]);
                b1[nt][1] = cvt2_e4m3_h2(r1[t + 4]);
            }
            const int r_base = warp_m * (MT_PER_WARP * 16) + lr;
            const uint32_t s_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(sa + fp8_a_off(r_base, kk0 * 2 + lc)));
            const uint32_t s_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(sa + fp8_a_off(r_base, kk1 * 2 + lc)));
#pragma unroll
            for (int mt = 0; mt < MT_PER_WARP; ++mt) {
                uint32_t a0[4], a1[4];
                ldsm_x4_step_fp8(a0, s_a0, mt * 2);
                ldsm_x4_step_fp8(a1, s_a1, mt * 2);
#pragma unroll
                for (int nt = 0; nt < NT_PER_WARP; ++nt) {
                    float p[4];
                    mma16816_fp8_zero(p, a0, b0[nt]);
                    mma16816_fp8(p, a1, b1[nt]);
                    acc[mt][nt][0] = fmaf(p[0], sc, acc[mt][nt][0]);
                    acc[mt][nt][1] = fmaf(p[1], sc, acc[mt][nt][1]);
                    acc[mt][nt][2] = fmaf(p[2], sc, acc[mt][nt][2]);
                    acc[mt][nt][3] = fmaf(p[3], sc, acc[mt][nt][3]);
                }
            }
        }
    }
    cp_async_wait_fp8<0>();

#pragma unroll
    for (int mt = 0; mt < MT_PER_WARP; ++mt) {
        const int r0 = m0 + (warp_m * MT_PER_WARP + mt) * 16 + g;
        const int r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP; ++nt) {
            const int col = n_warp + nt * 8 + 2 * t;
            if (col < n) {
                if (r0 < m) {
                    *reinterpret_cast<__half2*>(y + (size_t)r0 * n + col) =
                        __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
                }
                if (r1 < m) {
                    *reinterpret_cast<__half2*>(y + (size_t)r1 * n + col) =
                        __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
                }
            }
        }
    }
}

/// out[i] = W[i] * scale[i / k / 32][(i % k) / 32] in fp16, for the tests and for a host-side sanity check of the
/// layout. Flat over an [N, K] weight.
__global__ void fp8_dequant_kernel(const uint8_t* __restrict__ w,
                                   const uint8_t* __restrict__ s,
                                   __half* __restrict__ out,
                                   long long total,
                                   int k,
                                   int ktiles) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int nrow = (int)(i / k);
    const int kcol = (int)(i % k);
    const float v = e4m3_to_f32(w[i]) * e8m0_scale(s[(size_t)(nrow / TILE) * ktiles + kcol / TILE]);
    out[i] = __float2half(v);
}

inline int check_args(int m, int k, int n, int x_dtype) {
    if (m < 1 || k < TILE || n < TILE || (k % TILE) != 0 || (n % TILE) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (x_dtype != 0 && x_dtype != 1) {
        return (int)cudaErrorInvalidValue;
    }
    return 0;
}

inline bool use_fp8_mma(int m, int k, int n, bool force = false) {
    if (!force && m <= 16) return false;
    if (k < 512 || (k % MMA_BK) != 0 || (n % MMA_BN) != 0) return false;
    static int mode = -1;
    if (mode < 0) {
        mode = (getenv("REXL3_DSV41_NO_FP8_MMA") != nullptr) ? 0 : 1;
    }
    return mode == 1;
}

template <int ROWS, int X_DTYPE>
inline void launch_fp8_rows(const void* x, const uint8_t* w, const uint8_t* s, __half* out, int m, int k, int n, cudaStream_t st) {
    const int rtiles = (m + ROWS - 1) / ROWS;
    if (ROWS <= 8) {
        // Production decode path: the pipelined small-M kernel (same bits as the classic kernel; see
        // fp8_gemm_sk_cta). 4 columns per CTA: the grid is >= 256 CTAs on every live decode shape — the
        // smallest live n (512, wkv) only appears inside the pair launch, where nb1+nb2 >= 448. Rows<=4
        // double-buffer the K chunks in registers; rows>=5 keep the classic in-loop loads (no spill).
        constexpr int WARPS = 2, CPW = 2;
        const int bcols = WARPS * CPW;
        fp8_gemm_sk_kernel<ROWS, WARPS, CPW, X_DTYPE><<<dim3((n + bcols - 1) / bcols, rtiles), WARPS * 32, 0, st>>>(x, w, s, out, m, k, n);
        return;
    }
    if (n <= 1280) {
        constexpr int WARPS = 2; // 8 cols/block -> 64 blocks at n=512, 160 blocks at n=1280
        const int bcols = WARPS * 4;
        fp8_gemm_kernel<ROWS, WARPS, X_DTYPE><<<dim3((n + bcols - 1) / bcols, rtiles), WARPS * 32, 0, st>>>(x, w, s, out, m, k, n);
    } else if (n <= 2560) {
        constexpr int WARPS = 4; // 16 cols/block -> 144 blocks at n=2304
        const int bcols = WARPS * 4;
        fp8_gemm_kernel<ROWS, WARPS, X_DTYPE><<<dim3((n + bcols - 1) / bcols, rtiles), WARPS * 32, 0, st>>>(x, w, s, out, m, k, n);
    } else {
        constexpr int WARPS = 8; // 32 cols/block
        const int bcols = WARPS * 4;
        fp8_gemm_kernel<ROWS, WARPS, X_DTYPE><<<dim3((n + bcols - 1) / bcols, rtiles), WARPS * 32, 0, st>>>(x, w, s, out, m, k, n);
    }
}

// The classic small-M dispatch, kept verbatim as the test-only reference path (use_classic FFI). The
// production path above is the unconditional pipelined kernel; this function exists solely so the GPU tests
// can prove the new kernels bit-for-bit against the classic ones without env switches.
template <int ROWS, int X_DTYPE>
inline void launch_fp8_rows_classic(const void* x, const uint8_t* w, const uint8_t* s, __half* out, int m, int k, int n, cudaStream_t st) {
    const int rtiles = (m + ROWS - 1) / ROWS;
    if (n <= 1280) {
        constexpr int WARPS = 2; // 8 cols/block -> 64 blocks at n=512, 160 blocks at n=1280
        const int bcols = WARPS * 4;
        fp8_gemm_kernel<ROWS, WARPS, X_DTYPE><<<dim3((n + bcols - 1) / bcols, rtiles), WARPS * 32, 0, st>>>(x, w, s, out, m, k, n);
    } else if (n <= 2560) {
        constexpr int WARPS = 4; // 16 cols/block -> 144 blocks at n=2304
        const int bcols = WARPS * 4;
        fp8_gemm_kernel<ROWS, WARPS, X_DTYPE><<<dim3((n + bcols - 1) / bcols, rtiles), WARPS * 32, 0, st>>>(x, w, s, out, m, k, n);
    } else {
        constexpr int WARPS = 8; // 32 cols/block
        const int bcols = WARPS * 4;
        fp8_gemm_kernel<ROWS, WARPS, X_DTYPE><<<dim3((n + bcols - 1) / bcols, rtiles), WARPS * 32, 0, st>>>(x, w, s, out, m, k, n);
    }
}

template <int BM, int BN, int M_WARPS, int N_WARPS, int STAGES, int MIN_BLOCKS, int X_DTYPE>
__global__ void __launch_bounds__(M_WARPS * N_WARPS * 32, MIN_BLOCKS)
    fp8_mma_bk32_kernel(const void* __restrict__ x,
                        const uint8_t* __restrict__ w,
                        const uint8_t* __restrict__ s,
                        __half* __restrict__ y,
                        int m,
                        int k,
                        int n) {
    constexpr int THREADS = M_WARPS * N_WARPS * 32;
    constexpr int BK = 32;
    constexpr int MT_PER_WARP = (BM / 16) / M_WARPS;
    constexpr int NT_PER_WARP = (BN / 8) / N_WARPS;
    constexpr int A_STAGE_HALVES = BM * BK;
    constexpr int B_STAGE_U4 = BN * 2;
    constexpr int A_ITERS = (BM * 4) / THREADS;
    constexpr int B_ITERS = (BN * 2) / THREADS;
    __shared__ __align__(128) __half sa_ring[STAGES][A_STAGE_HALVES];
    __shared__ __align__(128) uint4 sb_ring[STAGES][B_STAGE_U4];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int warp_m = warp / N_WARPS;
    const int warp_n = warp % N_WARPS;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int ktiles = k / TILE;
    const int n_warp = n0 + warp_n * (NT_PER_WARP * 8);
    const uint8_t* s_row = s + (size_t)(n_warp / TILE) * ktiles;
    const int KC = k / BK;

    auto load_stage = [&](int stg, int kc) {
        __half* sa = sa_ring[stg];
        uint4* sb = sb_ring[stg];
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int idx = i * THREADS + tid;
            const int r = idx >> 2;
            const int c = idx & 3;
            const int gr = m0 + r;
            const bool ok = (gr < m);
            const size_t x_off = (size_t)(ok ? gr : 0) * k + kc * BK + c * 8;
            if constexpr (X_DTYPE == 0) {
                cp_async16_fp8(sa + fp8_a_off32(r, c), static_cast<const __half*>(x) + x_off, ok);
            } else {
                const uint4 raw = ok ? __ldg(reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(x) + x_off))
                                     : make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(sa + fp8_a_off32(r, c)) = cvt8_bf16_to_f16(raw);
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
            const int idx = i * THREADS + tid;
            const int nr = idx >> 1;
            const int c16 = idx & 1;
            const int gn = n0 + nr;
            cp_async16_fp8(&sb[fp8_b_off32(nr, c16)], w + (size_t)gn * k + kc * BK + c16 * 16, true);
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
        cp_async_commit_fp8();
    }

    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int lc = lane >> 4;

#pragma unroll 1
    for (int kc = 0; kc < KC; ++kc) {
        cp_async_wait_fp8<STAGES - 2>();
        __syncthreads();
        const int nk = kc + STAGES - 1;
        if (nk < KC) load_stage(nk % STAGES, nk);
        cp_async_commit_fp8();

        const __half* sa = sa_ring[kc % STAGES];
        const uint4* sb = sb_ring[kc % STAGES];

        const float sc = e8m0_scale(__ldg(s_row + kc));
        uint32_t b0[NT_PER_WARP][2], b1[NT_PER_WARP][2];
        const int nr_base = warp_n * (NT_PER_WARP * 8) + g;
        const int b_off0 = fp8_b_off32(nr_base, 0);
        const int b_off1 = fp8_b_off32(nr_base, 1);
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP; ++nt) {
            const uint16_t* r0 = reinterpret_cast<const uint16_t*>(&sb[b_off0 + nt * 16]);
            const uint16_t* r1 = reinterpret_cast<const uint16_t*>(&sb[b_off1 + nt * 16]);
            b0[nt][0] = cvt2_e4m3_h2(r0[t]);
            b0[nt][1] = cvt2_e4m3_h2(r0[t + 4]);
            b1[nt][0] = cvt2_e4m3_h2(r1[t]);
            b1[nt][1] = cvt2_e4m3_h2(r1[t + 4]);
        }
        const int r_base = warp_m * (MT_PER_WARP * 16) + lr;
        const uint32_t s_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(sa + fp8_a_off32(r_base, 0 + lc)));
        const uint32_t s_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(sa + fp8_a_off32(r_base, 2 + lc)));
#pragma unroll
        for (int mt = 0; mt < MT_PER_WARP; ++mt) {
            uint32_t a0[4], a1[4];
            ldsm_x4_step_fp8(a0, s_a0, mt);
            ldsm_x4_step_fp8(a1, s_a1, mt);
#pragma unroll
            for (int nt = 0; nt < NT_PER_WARP; ++nt) {
                float p[4];
                mma16816_fp8_zero(p, a0, b0[nt]);
                mma16816_fp8(p, a1, b1[nt]);
                acc[mt][nt][0] = fmaf(p[0], sc, acc[mt][nt][0]);
                acc[mt][nt][1] = fmaf(p[1], sc, acc[mt][nt][1]);
                acc[mt][nt][2] = fmaf(p[2], sc, acc[mt][nt][2]);
                acc[mt][nt][3] = fmaf(p[3], sc, acc[mt][nt][3]);
            }
        }
    }
    cp_async_wait_fp8<0>();

#pragma unroll
    for (int mt = 0; mt < MT_PER_WARP; ++mt) {
        const int r0 = m0 + (warp_m * MT_PER_WARP + mt) * 16 + g;
        const int r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP; ++nt) {
            const int col = n_warp + nt * 8 + 2 * t;
            if (col < n) {
                if (r0 < m) {
                    *reinterpret_cast<__half2*>(y + (size_t)r0 * n + col) =
                        __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
                }
                if (r1 < m) {
                    *reinterpret_cast<__half2*>(y + (size_t)r1 * n + col) =
                        __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
                }
            }
        }
    }
}

template <int X_DTYPE>
inline void launch_fp8_dtype(const void* x, const uint8_t* w, const uint8_t* s, __half* out, int m, int k, int n, cudaStream_t st, bool force_mma = false) {
    if (use_fp8_mma(m, k, n, force_mma)) {
        const int ctas_128 = ((m + 127) / 128) * (n / 128);
        const int ctas_64_128 = ((m + 63) / 64) * (n / 128);
        if (m >= 256 && ctas_128 >= 96) {
            const dim3 grid((unsigned)((m + 127) / 128), (unsigned)(n / 128));
            fp8_mma_bk32_kernel<128, 128, 1, 8, 3, 2, X_DTYPE><<<grid, 256, 0, st>>>(x, w, s, out, m, k, n);
        } else if (ctas_64_128 < 48 && (n % 64) == 0) {
            const dim3 grid((unsigned)((m + 63) / 64), (unsigned)(n / 64));
            fp8_mma_kernel<64, 64, 2, 2, 3, 2, X_DTYPE><<<grid, 128, 0, st>>>(x, w, s, out, m, k, n);
        } else {
            const dim3 grid((unsigned)((m + 63) / 64), (unsigned)(n / 128));
            fp8_mma_kernel<64, 128, 1, 8, 3, 2, X_DTYPE><<<grid, 256, 0, st>>>(x, w, s, out, m, k, n);
        }
        return;
    }
    switch (m) {
        case 1: launch_fp8_rows<1, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 2: launch_fp8_rows<2, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 3: launch_fp8_rows<3, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 4: launch_fp8_rows<4, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 5: launch_fp8_rows<5, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 6: launch_fp8_rows<6, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 7:
        case 8: launch_fp8_rows<8, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        default: launch_fp8_rows<16, X_DTYPE>(x, w, s, out, m, k, n, st); break;
    }
}

template <int ROWS, int X_DTYPE>
inline void launch_fp8_pair_rows(const void* x,
                                 const uint8_t* w1,
                                 const uint8_t* s1,
                                 __half* y1,
                                 int n1,
                                 const uint8_t* w2,
                                 const uint8_t* s2,
                                 __half* y2,
                                 int n2,
                                 int m,
                                 int k,
                                 cudaStream_t st) {
    const int rtiles = (m + ROWS - 1) / ROWS;
    if (ROWS <= 8) {
        // Production decode path (pair): same pipelined small-M kernel, one grid over both matrices. 4
        // columns per CTA; nb1+nb2 >= 256 on every live pair (wq_a+wkv: 448, w1+w3: 1152).
        constexpr int WARPS = 2, CPW = 2;
        const int bcols = WARPS * CPW;
        const int nb1 = (n1 + bcols - 1) / bcols;
        const int nb2 = (n2 + bcols - 1) / bcols;
        fp8_gemm_pair_sk_kernel<ROWS, WARPS, CPW, X_DTYPE><<<dim3(nb1 + nb2, rtiles), WARPS * 32, 0, st>>>(
            x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, nb1);
        return;
    }
    const int max_n = (n1 > n2) ? n1 : n2;
    if (max_n <= 1280) {
        constexpr int WARPS = 2;
        const int bcols = WARPS * 4;
        const int nb1 = (n1 + bcols - 1) / bcols;
        const int nb2 = (n2 + bcols - 1) / bcols;
        fp8_gemm_pair_kernel<ROWS, WARPS, X_DTYPE><<<dim3(nb1 + nb2, rtiles), WARPS * 32, 0, st>>>(
            x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, nb1);
    } else if (max_n <= 2560) {
        constexpr int WARPS = 4;
        const int bcols = WARPS * 4;
        const int nb1 = (n1 + bcols - 1) / bcols;
        const int nb2 = (n2 + bcols - 1) / bcols;
        fp8_gemm_pair_kernel<ROWS, WARPS, X_DTYPE><<<dim3(nb1 + nb2, rtiles), WARPS * 32, 0, st>>>(
            x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, nb1);
    } else {
        constexpr int WARPS = 8;
        const int bcols = WARPS * 4;
        const int nb1 = (n1 + bcols - 1) / bcols;
        const int nb2 = (n2 + bcols - 1) / bcols;
        fp8_gemm_pair_kernel<ROWS, WARPS, X_DTYPE><<<dim3(nb1 + nb2, rtiles), WARPS * 32, 0, st>>>(
            x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, nb1);
    }
}

// The classic pair dispatch, kept as the test-only reference path (use_classic FFI).
template <int ROWS, int X_DTYPE>
inline void launch_fp8_pair_rows_classic(const void* x,
                                         const uint8_t* w1,
                                         const uint8_t* s1,
                                         __half* y1,
                                         int n1,
                                         const uint8_t* w2,
                                         const uint8_t* s2,
                                         __half* y2,
                                         int n2,
                                         int m,
                                         int k,
                                         cudaStream_t st) {
    const int rtiles = (m + ROWS - 1) / ROWS;
    const int max_n = (n1 > n2) ? n1 : n2;
    if (max_n <= 1280) {
        constexpr int WARPS = 2;
        const int bcols = WARPS * 4;
        const int nb1 = (n1 + bcols - 1) / bcols;
        const int nb2 = (n2 + bcols - 1) / bcols;
        fp8_gemm_pair_kernel<ROWS, WARPS, X_DTYPE><<<dim3(nb1 + nb2, rtiles), WARPS * 32, 0, st>>>(
            x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, nb1);
    } else if (max_n <= 2560) {
        constexpr int WARPS = 4;
        const int bcols = WARPS * 4;
        const int nb1 = (n1 + bcols - 1) / bcols;
        const int nb2 = (n2 + bcols - 1) / bcols;
        fp8_gemm_pair_kernel<ROWS, WARPS, X_DTYPE><<<dim3(nb1 + nb2, rtiles), WARPS * 32, 0, st>>>(
            x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, nb1);
    } else {
        constexpr int WARPS = 8;
        const int bcols = WARPS * 4;
        const int nb1 = (n1 + bcols - 1) / bcols;
        const int nb2 = (n2 + bcols - 1) / bcols;
        fp8_gemm_pair_kernel<ROWS, WARPS, X_DTYPE><<<dim3(nb1 + nb2, rtiles), WARPS * 32, 0, st>>>(
            x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, nb1);
    }
}

template <int X_DTYPE>
inline void launch_fp8_pair_dtype(const void* x,
                                  const uint8_t* w1,
                                  const uint8_t* s1,
                                  __half* y1,
                                  int n1,
                                  const uint8_t* w2,
                                  const uint8_t* s2,
                                  __half* y2,
                                  int n2,
                                  int m,
                                  int k,
                                  cudaStream_t st,
                                  bool force_mma = false) {
    if (use_fp8_mma(m, k, n1, force_mma) || use_fp8_mma(m, k, n2, force_mma)) {
        launch_fp8_dtype<X_DTYPE>(x, w1, s1, y1, m, k, n1, st, force_mma);
        launch_fp8_dtype<X_DTYPE>(x, w2, s2, y2, m, k, n2, st, force_mma);
        return;
    }
    switch (m) {
        case 1: launch_fp8_pair_rows<1, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 2: launch_fp8_pair_rows<2, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 3: launch_fp8_pair_rows<3, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 4: launch_fp8_pair_rows<4, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 5: launch_fp8_pair_rows<5, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 6: launch_fp8_pair_rows<6, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 7:
        case 8: launch_fp8_pair_rows<8, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        default: launch_fp8_pair_rows<16, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
    }
}

template <int X_DTYPE>
inline void launch_fp8_dtype_classic(const void* x, const uint8_t* w, const uint8_t* s, __half* out, int m, int k, int n, cudaStream_t st) {
    if (use_fp8_mma(m, k, n)) {
        const int ctas_128 = ((m + 127) / 128) * (n / 128);
        const int ctas_64_128 = ((m + 63) / 64) * (n / 128);
        if (m >= 256 && ctas_128 >= 96) {
            const dim3 grid((unsigned)((m + 127) / 128), (unsigned)(n / 128));
            fp8_mma_bk32_kernel<128, 128, 1, 8, 3, 2, X_DTYPE><<<grid, 256, 0, st>>>(x, w, s, out, m, k, n);
        } else if (ctas_64_128 < 48 && (n % 64) == 0) {
            const dim3 grid((unsigned)((m + 63) / 64), (unsigned)(n / 64));
            fp8_mma_kernel<64, 64, 2, 2, 3, 2, X_DTYPE><<<grid, 128, 0, st>>>(x, w, s, out, m, k, n);
        } else {
            const dim3 grid((unsigned)((m + 63) / 64), (unsigned)(n / 128));
            fp8_mma_kernel<64, 128, 1, 8, 3, 2, X_DTYPE><<<grid, 256, 0, st>>>(x, w, s, out, m, k, n);
        }
        return;
    }
    switch (m) {
        case 1: launch_fp8_rows_classic<1, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 2: launch_fp8_rows_classic<2, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 3: launch_fp8_rows_classic<3, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 4: launch_fp8_rows_classic<4, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 5: launch_fp8_rows_classic<5, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 6: launch_fp8_rows_classic<6, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        case 7:
        case 8: launch_fp8_rows_classic<8, X_DTYPE>(x, w, s, out, m, k, n, st); break;
        default: launch_fp8_rows_classic<16, X_DTYPE>(x, w, s, out, m, k, n, st); break;
    }
}

template <int X_DTYPE>
inline void launch_fp8_pair_dtype_classic(const void* x,
                                          const uint8_t* w1,
                                          const uint8_t* s1,
                                          __half* y1,
                                          int n1,
                                          const uint8_t* w2,
                                          const uint8_t* s2,
                                          __half* y2,
                                          int n2,
                                          int m,
                                          int k,
                                          cudaStream_t st) {
    if (use_fp8_mma(m, k, n1) || use_fp8_mma(m, k, n2)) {
        launch_fp8_dtype_classic<X_DTYPE>(x, w1, s1, y1, m, k, n1, st);
        launch_fp8_dtype_classic<X_DTYPE>(x, w2, s2, y2, m, k, n2, st);
        return;
    }
    switch (m) {
        case 1: launch_fp8_pair_rows_classic<1, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 2: launch_fp8_pair_rows_classic<2, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 3: launch_fp8_pair_rows_classic<3, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 4: launch_fp8_pair_rows_classic<4, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 5: launch_fp8_pair_rows_classic<5, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 6: launch_fp8_pair_rows_classic<6, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        case 7:
        case 8: launch_fp8_pair_rows_classic<8, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
        default: launch_fp8_pair_rows_classic<16, X_DTYPE>(x, w1, s1, y1, n1, w2, s2, y2, n2, m, k, st); break;
    }
}

}  // namespace

GB_EXPORT int gb_fp8_linear(const void* x,
                            int x_dtype,
                            const uint8_t* w,
                            const uint8_t* s,
                            void* y,
                            int m,
                            int k,
                            int n,
                            int force_mma,
                            void* stream) {
    if (x == nullptr || w == nullptr || s == nullptr || y == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    const int bad = check_args(m, k, n, x_dtype);
    if (bad != 0) {
        return bad;
    }
    __half* out = static_cast<__half*>(y);
    cudaStream_t st = gb_stream(stream);
    const bool force = force_mma != 0;
    if (x_dtype == 0) {
        launch_fp8_dtype<0>(x, w, s, out, m, k, n, st, force);
    } else {
        launch_fp8_dtype<1>(x, w, s, out, m, k, n, st, force);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_fp8_linear_pair(const void* x,
                                 int x_dtype,
                                 const uint8_t* w1,
                                 const uint8_t* s1,
                                 void* y1,
                                 int n1,
                                 const uint8_t* w2,
                                 const uint8_t* s2,
                                 void* y2,
                                 int n2,
                                 int m,
                                 int k,
                                 int force_mma,
                                 void* stream) {
    if (x == nullptr || w1 == nullptr || s1 == nullptr || y1 == nullptr ||
        w2 == nullptr || s2 == nullptr || y2 == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    int bad = check_args(m, k, n1, x_dtype);
    if (bad != 0) {
        return bad;
    }
    bad = check_args(m, k, n2, x_dtype);
    if (bad != 0) {
        return bad;
    }
    __half* out1 = static_cast<__half*>(y1);
    __half* out2 = static_cast<__half*>(y2);
    cudaStream_t st = gb_stream(stream);
    const bool force = force_mma != 0;
    if (x_dtype == 0) {
        launch_fp8_pair_dtype<0>(x, w1, s1, out1, n1, w2, s2, out2, n2, m, k, st, force);
    } else {
        launch_fp8_pair_dtype<1>(x, w1, s1, out1, n1, w2, s2, out2, n2, m, k, st, force);
    }
    return GB_CHECK_LAUNCH();
}

// Test-only classic-path entry points, so the A/B tests can compare the new pipelined small-M kernels
// against the classic ones bit-for-bit without any env switch in the production path.
GB_EXPORT int gb_fp8_linear_classic(const void* x,
                                    int x_dtype,
                                    const uint8_t* w,
                                    const uint8_t* s,
                                    void* y,
                                    int m,
                                    int k,
                                    int n,
                                    void* stream) {
    if (x == nullptr || w == nullptr || s == nullptr || y == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    const int bad = check_args(m, k, n, x_dtype);
    if (bad != 0) {
        return bad;
    }
    __half* out = static_cast<__half*>(y);
    cudaStream_t st = gb_stream(stream);
    if (x_dtype == 0) {
        launch_fp8_dtype_classic<0>(x, w, s, out, m, k, n, st);
    } else {
        launch_fp8_dtype_classic<1>(x, w, s, out, m, k, n, st);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_fp8_linear_pair_classic(const void* x,
                                         int x_dtype,
                                         const uint8_t* w1,
                                         const uint8_t* s1,
                                         void* y1,
                                         int n1,
                                         const uint8_t* w2,
                                         const uint8_t* s2,
                                         void* y2,
                                         int n2,
                                         int m,
                                         int k,
                                         void* stream) {
    if (x == nullptr || w1 == nullptr || s1 == nullptr || y1 == nullptr ||
        w2 == nullptr || s2 == nullptr || y2 == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    int bad = check_args(m, k, n1, x_dtype);
    if (bad != 0) {
        return bad;
    }
    bad = check_args(m, k, n2, x_dtype);
    if (bad != 0) {
        return bad;
    }
    __half* out1 = static_cast<__half*>(y1);
    __half* out2 = static_cast<__half*>(y2);
    cudaStream_t st = gb_stream(stream);
    if (x_dtype == 0) {
        launch_fp8_pair_dtype_classic<0>(x, w1, s1, out1, n1, w2, s2, out2, n2, m, k, st);
    } else {
        launch_fp8_pair_dtype_classic<1>(x, w1, s1, out1, n1, w2, s2, out2, n2, m, k, st);
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_fp8_dequant(const uint8_t* w, const uint8_t* s, void* out, int n, int k, void* stream) {
    if (w == nullptr || s == nullptr || out == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    if (n < TILE || k < TILE || (n % TILE) != 0 || (k % TILE) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    const long long total = (long long)n * (long long)k;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    fp8_dequant_kernel<<<(unsigned)blocks, threads, 0, gb_stream(stream)>>>(
        w, s, static_cast<__half*>(out), total, k, k / TILE);
    return GB_CHECK_LAUNCH();
}
