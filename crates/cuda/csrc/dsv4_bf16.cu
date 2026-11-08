// gb-nvcc: --fmad=false
#include "common.cuh"

namespace {

typedef uint16_t u16;
typedef uint32_t u32;

constexpr int ERR_INVALID = 1;
constexpr int MT = 8;
constexpr int NC = 4;

__device__ __forceinline__ float bf2f(u16 b) { return __uint_as_float((u32)b << 16); }

__device__ __forceinline__ u32 cvt2_bf16_to_f16_b(u32 u) {
    const float f0 = bf2f(u & 0xFFFFu);
    const float f1 = bf2f(u >> 16);
    const __half2 h = __floats2half2_rn(f0, f1);
    return *reinterpret_cast<const u32*>(&h);
}

__device__ __forceinline__ uint4 cvt8_bf16_to_f16_b(uint4 u) {
    return make_uint4(cvt2_bf16_to_f16_b(u.x), cvt2_bf16_to_f16_b(u.y), cvt2_bf16_to_f16_b(u.z), cvt2_bf16_to_f16_b(u.w));
}

__device__ __forceinline__ void ldsm_x4_b(u32 (&a)[4], const void* smem) {
    const u32 s = static_cast<u32>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

__device__ __forceinline__ void mma16816_b(float (&d)[4], const u32 (&a)[4], const u32 (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(128) bf16_mma_k(const __half* __restrict__ x,
                                                  const u16* __restrict__ w,
                                                  void* __restrict__ y,
                                                  int m,
                                                  int k,
                                                  int n,
                                                  int y_f32) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 64;
    __shared__ __align__(128) __half sa[BM * BK];
    __shared__ __align__(128) uint4 sb[BN * 8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int lr = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int lc = lane >> 4;

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int col = n0 + warp * 8 + 2 * t;

    float acc[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[mt][c] = 0.0f;

    for (int k_base = 0; k_base < k; k_base += BK) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = i * 128 + tid;
            const int r = idx >> 3;
            const int c = idx & 7;
            const int gr = m0 + r;
            const int dst_off = r * BK + ((c ^ (r & 7)) << 3);
            *reinterpret_cast<uint4*>(sa + dst_off) =
                (gr < m) ? __ldg(reinterpret_cast<const uint4*>(x + (size_t)gr * k + k_base + c * 8))
                         : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = i * 128 + tid;
            const int nr = idx >> 3;
            const int c = idx & 7;
            const int gn = n0 + nr;
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(w + (size_t)gn * k + k_base + c * 8));
            sb[nr * 8 + (c ^ (nr & 7))] = cvt8_bf16_to_f16_b(raw);
        }
        __syncthreads();

        const int nr = warp * 8 + g;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            u32 a0[4], a1[4];
            const int r0 = lr;
            const int r1 = 16 + lr;
            const int c = ks * 2 + lc;
            ldsm_x4_b(a0, sa + r0 * BK + ((c ^ (r0 & 7)) << 3));
            ldsm_x4_b(a1, sa + r1 * BK + ((c ^ (r1 & 7)) << 3));
            const u32* p0 = reinterpret_cast<const u32*>(&sb[nr * 8 + ((ks * 2) ^ (nr & 7))]);
            const u32* p1 = reinterpret_cast<const u32*>(&sb[nr * 8 + ((ks * 2 + 1) ^ (nr & 7))]);
            const u32 b[2] = {p0[t], p1[t]};
            mma16816_b(acc[0], a0, b);
            mma16816_b(acc[1], a1, b);
        }
        __syncthreads();
    }

    if (col < n) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int r0 = m0 + mt * 16 + g;
            const int r1 = r0 + 8;
            if (y_f32) {
                float* yf = static_cast<float*>(y);
                if (r0 < m) *reinterpret_cast<float2*>(yf + (size_t)r0 * n + col) = make_float2(acc[mt][0], acc[mt][1]);
                if (r1 < m) *reinterpret_cast<float2*>(yf + (size_t)r1 * n + col) = make_float2(acc[mt][2], acc[mt][3]);
            } else {
                __half* yh = static_cast<__half*>(y);
                if (r0 < m) *reinterpret_cast<__half2*>(yh + (size_t)r0 * n + col) = __floats2half2_rn(acc[mt][0], acc[mt][1]);
                if (r1 < m) *reinterpret_cast<__half2*>(yh + (size_t)r1 * n + col) = __floats2half2_rn(acc[mt][2], acc[mt][3]);
            }
        }
    }
}

__global__ void bf16_linear_k(const __half* __restrict__ x, const u16* __restrict__ w, void* __restrict__ y, int m,
                              int k, int n, int y_f32) {
    const int n0 = blockIdx.y * NC;
    const int r0 = blockIdx.x * MT;
    float acc[NC][MT];
#pragma unroll
    for (int c = 0; c < NC; ++c)
#pragma unroll
        for (int i = 0; i < MT; ++i) acc[c][i] = 0.f;

    const int rows = min(MT, m - r0);
    const int cols = min(NC, n - n0);

    for (int t = threadIdx.x; t < k; t += blockDim.x) {
        float xv[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            xv[i] = (i < rows) ? __half2float(__ldg(&x[(size_t)(r0 + i) * k + t])) : 0.f;
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c < cols) {
                const float wv = bf2f(__ldg(&w[(size_t)(n0 + c) * k + t]));
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if (i < rows) acc[c][i] = fmaf(xv[i], wv, acc[c][i]);
                }
            }
        }
    }
    __shared__ float sh[NC][MT][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            float v = acc[c][i];
#pragma unroll
            for (int s = 16; s >= 1; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
            if (lane == 0) sh[c][i][warp] = v;
        }
    }
    __syncthreads();
    const int nw = blockDim.x >> 5;
    if (warp < cols && lane < rows) {
        float v = 0.f;
        for (int q = 0; q < nw; ++q) v += sh[warp][lane][q];
        const int r = r0 + lane;
        const int col = n0 + warp;
        if (y_f32) {
            ((float*)y)[(size_t)r * n + col] = v;
        } else {
            ((__half*)y)[(size_t)r * n + col] = __float2half(v);
        }
    }
}

static inline bool al16(const void* p) { return ((uintptr_t)p & 15u) == 0; }

}  // namespace

GB_EXPORT int gb_bf16_linear(const void* x, const void* w, void* y, int m, int k, int n, int y_f32, void* stream) {
    if (m < 0 || k <= 0 || n <= 0) return ERR_INVALID;
    if (!al16(x) || !al16(w) || !al16(y)) return ERR_INVALID;
    if (m == 0) return 0;
    if ((k % 64) == 0 && (n % 32) == 0) {
        const dim3 g((unsigned)((m + 31) / 32), (unsigned)(n / 32));
        bf16_mma_k<<<g, 128, 0, gb_stream(stream)>>>((const __half*)x, (const u16*)w, y, m, k, n, y_f32);
        return GB_CHECK_LAUNCH();
    }
    const dim3 g((unsigned)((m + MT - 1) / MT), (unsigned)((n + NC - 1) / NC));
    bf16_linear_k<<<g, 256, 0, gb_stream(stream)>>>((const __half*)x, (const u16*)w, y, m, k, n, y_f32);
    return GB_CHECK_LAUNCH();
}
