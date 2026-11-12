// gb-nvcc: --fmad=false
// mHC (manifold-constrained hyper-connections) kernels for GLM-5.3-Flash: 4 residual streams of 4096, fp32, row-major
// x[row][stream h][d]. Spec and oracle: docs/GLM53.md "mHC"; ExLlamaV3 `modules/hyperconnections.py` + `exllamav3_ext/hc_mix.cu`
// (MIT); the pinned K split follows TensorFold `families/glm5_next/cuda/glue.py` (`_hc_partial`, fixed `HC_BLOCKS`; MIT).
//
// One site (attention or FFN) is   mix:   S = [x . fn^T (24 dots), sum x^2]   (reduction over the 16384 stream values)
//                                         rmr = 1 / sqrt(sum x^2 / 16384 + rms_eps)
//                                         pre  = sigmoid(fma(S[0:4]  * rmr, scale0, base[0:4]))  + hc_eps
//                                         post = 2 sigmoid(fma(S[4:8]  * rmr, scale1, base[4:8]))
//                                         comb = sinkhorn(softmax_rows(fma(S[8:24] * rmr, scale2, base[8:24])))
//                                         collapsed[d] = sum_h pre[h] x[h][d]  (fma chain in h order), then the layer's RMSNorm
//                                  apply: x[h][d] <- post[h] y[d] + sum_g comb[g][h] x[g][d]   (g = 0..3 fma chain), in place.
//
// ROW INVARIANCE (the contract: row r's output bits are identical for any number of rows sharing the call, 1..65535, and any
// position in the batch). It holds BY CONSTRUCTION because the arithmetic of a row is defined on fixed "atoms" and never on
// the launch shape:
//   * K split: the 16384 values of a row are cut into NATOM = 32 atoms of 128 d x all 4 streams (512 values). Atom a, lane l
//     owns d = 128a + 4l .. +3 of every stream. Its 25 quantities (24 fn dots + sum of squares) are an fma chain over its 16
//     values in (stream, d) order, then a fixed xor butterfly over the 32 lanes (16, 8, 4, 2, 1; the reduce-scatter form
//     below has exactly that addition tree). Each atom's 25 floats are written to the partials workspace.
//   * The finalize re-reduces the 32 atom partials of each quantity as a sequential chain a = 0..31, one thread per quantity.
//   * The number of warps per block, the grid shape, the apply/finalize column chunking only decide WHICH thread computes a
//     value, never in what order its terms are added. No float atomics, no cross-row sharing of any partial result.
//   * Everything after the reduction is per-row scalar work on 4 + 4 + 16 values or per-column work on 4 streams.
// Same atoms for the standalone partials and the apply-fused partials (same device code), so fused == unfused bit for bit.
//
// ARITHMETIC. fn is stored in a lossless narrow type when the checkpoint allows it (fp16 on the GLM-5.3 pack: all 90 fn tensors
// round-trip exactly; bf16 otherwise; fp32 if neither is exact) and widened to fp32 exactly, so the products are the same for
// every storage type. fmaf() is explicit where an FMA is wanted (dots, collapse, apply); --fmad=false keeps every other
// a*b+c as two roundings so the host mirror in the tests is exact. Transcendentals are the ACCURATE ones (expf, IEEE division
// and sqrtf, as glue.cu): the sigmoid/softmax/Sinkhorn tail is 24 values per row, so __expf / __fdividef would save nothing
// and cost accuracy against a float64 reference. (ExLlamaV3 uses __expf, __fdividef and rsqrtf; REXL3 does not.)
//
// Geometry is GLM-5.3-Flash's: H = 4 streams, hidden 4096. Other sizes are refused by the launchers (cudaErrorInvalidValue).
#include "common.cuh"
#include "pdl.cuh"
#include <math.h>

using gbp_pdl::pdl_trigger;
using gbp_pdl::pdl_wait;

namespace {

typedef uint16_t u16;
typedef uint32_t u32;

constexpr int ERR_INVALID = 1;  // cudaErrorInvalidValue

constexpr int NS = 4;                 // streams
constexpr int HD = 4096;              // hidden size
constexpr int ROW = NS * HD;          // 16384 values per row
constexpr int NM = 24;                // fn rows: 4 pre, 4 post, 16 comb
constexpr int NQ = NM + 1;            // + sum of squares (slot 24)
constexpr int QS = 32;                // partials stride per atom (floats)
constexpr int ATOM_D = 128;           // d per atom
constexpr int NATOM = HD / ATOM_D;    // 32
constexpr int FIN_NT = 512;           // finalize: collapse threads (16 warps) ...
constexpr int FIN_THREADS = FIN_NT + 32;  // ... plus one warp for the reduction + Sinkhorn
constexpr int VEC = HD / 4;           // float4 columns per stream (1024)

enum { T_F32 = 0, T_F16 = 1, T_BF16 = 2 };

// ------------------------------------------------------------------------------------------------ typed 4-wide access

__device__ __forceinline__ float bf2f(u32 u16bits) { return __uint_as_float(u16bits << 16); }
__device__ __forceinline__ u16 f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

template <int T>
__device__ __forceinline__ float4 ld4(const void* p, size_t idx) {
    if constexpr (T == T_F32) {
        return *(const float4*)((const float*)p + idx);
    } else if constexpr (T == T_F16) {
        const uint2 u = *(const uint2*)((const u16*)p + idx);
        const float2 a = __half22float2(*(const __half2*)&u.x), b = __half22float2(*(const __half2*)&u.y);
        return make_float4(a.x, a.y, b.x, b.y);
    } else {
        const uint2 u = *(const uint2*)((const u16*)p + idx);
        return make_float4(bf2f(u.x & 0xFFFFu), bf2f(u.x >> 16), bf2f(u.y & 0xFFFFu), bf2f(u.y >> 16));
    }
}

// Raw (unconverted) 4-wide load and its widening: lets a kernel issue all its weight loads before the data they multiply.
template <int T>
struct Raw4 {
    typedef uint2 type;
};
template <>
struct Raw4<T_F32> {
    typedef float4 type;
};
template <int T>
__device__ __forceinline__ typename Raw4<T>::type ld4raw(const void* p, size_t idx) {
    if constexpr (T == T_F32) return *(const float4*)((const float*)p + idx);
    else return *(const uint2*)((const u16*)p + idx);
}
template <int T>
__device__ __forceinline__ float4 cvt4(const typename Raw4<T>::type& u) {
    if constexpr (T == T_F32) {
        return u;
    } else if constexpr (T == T_F16) {
        const float2 a = __half22float2(*(const __half2*)&u.x), b = __half22float2(*(const __half2*)&u.y);
        return make_float4(a.x, a.y, b.x, b.y);
    } else {
        return make_float4(bf2f(u.x & 0xFFFFu), bf2f(u.x >> 16), bf2f(u.y & 0xFFFFu), bf2f(u.y >> 16));
    }
}

template <int T>
__device__ __forceinline__ void st4(void* p, size_t idx, float a, float b, float c, float d) {
    if constexpr (T == T_F32) {
        *(float4*)((float*)p + idx) = make_float4(a, b, c, d);
    } else if constexpr (T == T_F16) {
        __half2 lo = __floats2half2_rn(a, b), hi = __floats2half2_rn(c, d);
        uint2 u;
        u.x = *(u32*)&lo;
        u.y = *(u32*)&hi;
        *(uint2*)((u16*)p + idx) = u;
    } else {
        uint2 u;
        u.x = (u32)f2bf(a) | ((u32)f2bf(b) << 16);
        u.y = (u32)f2bf(c) | ((u32)f2bf(d) << 16);
        *(uint2*)((u16*)p + idx) = u;
    }
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// Sum over a warp with the fixed xor-butterfly 16, 8, 4, 2, 1 (every lane ends with the same bits).
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Reduce 32 values per lane over the 32 lanes with the xor butterfly (16, 8, 4, 2, 1) in its cheap reduce-scatter form: after
// the call lane l holds the sum over lanes of v[l]. The additions are exactly those of 32 independent warp_sum()s (the same
// pairs at the same levels; fp add commutes, so both partners of a pair compute the same bits), with 31 shuffles instead of 160.
__device__ __forceinline__ float reduce_scatter32(const float (&v)[32], int lane) {
    float a[16], b[8], c[4], d[2];
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4, b1 = lane & 2, b0 = lane & 1;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float keep = b4 ? v[i + 16] : v[i], send = b4 ? v[i] : v[i + 16];
        a[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float keep = b3 ? a[i + 8] : a[i], send = b3 ? a[i] : a[i + 8];
        b[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float keep = b2 ? b[i + 4] : b[i], send = b2 ? b[i] : b[i + 4];
        c[i] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float keep = b1 ? c[i + 2] : c[i], send = b1 ? c[i] : c[i + 2];
        d[i] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
    }
    const float keep = b0 ? d[1] : d[0], send = b0 ? d[0] : d[1];
    return keep + __shfl_xor_sync(0xffffffffu, send, 1);
}

// ------------------------------------------------------------------------------------------------ the apply, per float4 of d

// x[h] <- post[h] * y + sum_g comb[g][h] * x[g]: o = post[h] * y (rounded), then g = 0..3 o = fmaf(comb[g][h], x[g], o).
__device__ __forceinline__ void apply_vec(float4 (&xv)[NS], const float4 yv, const float (&post)[NS], const float (&comb)[NS * NS]) {
    float4 nx[NS];
#pragma unroll
    for (int h = 0; h < NS; ++h) {
        float4 o;
        o.x = post[h] * yv.x;
        o.y = post[h] * yv.y;
        o.z = post[h] * yv.z;
        o.w = post[h] * yv.w;
#pragma unroll
        for (int g = 0; g < NS; ++g) {
            o.x = fmaf(comb[g * NS + h], xv[g].x, o.x);
            o.y = fmaf(comb[g * NS + h], xv[g].y, o.y);
            o.z = fmaf(comb[g * NS + h], xv[g].z, o.z);
            o.w = fmaf(comb[g * NS + h], xv[g].w, o.w);
        }
        nx[h] = o;
    }
#pragma unroll
    for (int h = 0; h < NS; ++h) xv[h] = nx[h];
}

// ------------------------------------------------------------------------------------------------ partials (K1)

// One block = one atom of RB rows; its MG warps split the 25 quantities (24 dots + the sum of squares) into contiguous chunks of
// QP = ceil(25 / MG). The arithmetic of every (row, quantity) is the same fma chain over the lane's 16 values and the same xor
// butterfly whatever MG and RB are: MG is a latency knob for small row counts (more warps = more weight loads in flight), RB a
// bandwidth knob for large ones (the weight tile is loaded once for RB rows). Same bits for every (MG, RB).
// WT: fn storage type. APPLY: first apply the PREVIOUS site's update (post/comb/y) to x, store it (warp 0, after a barrier so no
// warp can read an updated value) and take the dots of the updated values (the apply-fused partials of the next site).
template <int WT, int YT, bool APPLY, int MG, int RB>
__global__ void __launch_bounds__(MG * 32) partials_k(float* x, const void* fn, float* part, const void* y, const float* post_g,
                                                      const float* comb_g, int rows) {
    constexpr int QP = (NQ + MG - 1) / MG;
    constexpr bool PF = QP <= 7;  // few quantities per warp: issue every weight load first (registers allow it)
    typedef typename Raw4<WT>::type Raw;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int a = blockIdx.x, r0 = blockIdx.y * RB;
    const int d0 = a * ATOM_D + lane * 4;
    const int qbeg = warp * QP;
    Raw wraw[PF ? QP : 1][PF ? NS : 1];
    if constexpr (PF) {
#pragma unroll
        for (int i = 0; i < QP; ++i)
#pragma unroll
            for (int h = 0; h < NS; ++h)
                if (qbeg + i < NM) wraw[i][h] = ld4raw<WT>(fn, (size_t)(qbeg + i) * ROW + (size_t)h * HD + d0);
    }
    pdl_wait();  // x (the streams), the previous site's post/comb and the layer's y are the predecessor's; fn is a weight (loaded above)
    float4 xv[RB][NS];
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
        const int r = min(r0 + rb, rows - 1);  // rows past the end repeat the last row (their results are never stored)
        const float* xr = x + (size_t)r * ROW;
#pragma unroll
        for (int h = 0; h < NS; ++h) xv[rb][h] = *(const float4*)(xr + (size_t)h * HD + d0);
    }
    if constexpr (APPLY) {
        float p[RB][NS], c[RB][NS * NS];
        float4 yv[RB];
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
            const int r = min(r0 + rb, rows - 1);
#pragma unroll
            for (int h = 0; h < NS; ++h) p[rb][h] = post_g[(size_t)r * NS + h];
#pragma unroll
            for (int k = 0; k < NS * NS; ++k) c[rb][k] = comb_g[(size_t)r * NS * NS + k];
            yv[rb] = ld4<YT>(y, (size_t)r * HD + d0);
        }
        if constexpr (MG > 1) __syncthreads();  // every warp has read the old x
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
            apply_vec(xv[rb], yv[rb], p[rb], c[rb]);
            if (warp == 0 && r0 + rb < rows) {
                float* xr = x + (size_t)(r0 + rb) * ROW;
#pragma unroll
                for (int h = 0; h < NS; ++h) *(float4*)(xr + (size_t)h * HD + d0) = xv[rb][h];
            }
        }
    }
    pdl_trigger();  // every read of the predecessor's output is issued (the apply's store, if any, is above)
    constexpr int NV = MG == 1 ? QS : QP;
    float v[RB][NV];
#pragma unroll
    for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int i = 0; i < NV; ++i) v[rb][i] = 0.f;
#pragma unroll
    for (int i = 0; i < QP; ++i) {
        const int q = qbeg + i;
        if (q >= NQ) break;
        if (q == NM) {
#pragma unroll
            for (int rb = 0; rb < RB; ++rb) {
                float ss = 0.f;
#pragma unroll
                for (int h = 0; h < NS; ++h) {
                    ss = fmaf(xv[rb][h].x, xv[rb][h].x, ss);
                    ss = fmaf(xv[rb][h].y, xv[rb][h].y, ss);
                    ss = fmaf(xv[rb][h].z, xv[rb][h].z, ss);
                    ss = fmaf(xv[rb][h].w, xv[rb][h].w, ss);
                }
                v[rb][i] = ss;
            }
        } else {
            float acc[RB];
#pragma unroll
            for (int rb = 0; rb < RB; ++rb) acc[rb] = 0.f;
#pragma unroll
            for (int h = 0; h < NS; ++h) {
                float4 w;
                if constexpr (PF) w = cvt4<WT>(wraw[i][h]);
                else w = ld4<WT>(fn, (size_t)q * ROW + (size_t)h * HD + d0);
#pragma unroll
                for (int rb = 0; rb < RB; ++rb) {
                    acc[rb] = fmaf(xv[rb][h].x, w.x, acc[rb]);
                    acc[rb] = fmaf(xv[rb][h].y, w.y, acc[rb]);
                    acc[rb] = fmaf(xv[rb][h].z, w.z, acc[rb]);
                    acc[rb] = fmaf(xv[rb][h].w, w.w, acc[rb]);
                }
            }
#pragma unroll
            for (int rb = 0; rb < RB; ++rb) v[rb][i] = acc[rb];
        }
    }
    if constexpr (MG > 1) {
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1)
#pragma unroll
            for (int rb = 0; rb < RB; ++rb)
#pragma unroll
                for (int i = 0; i < QP; ++i) v[rb][i] += __shfl_xor_sync(0xffffffffu, v[rb][i], m);
    }
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
        const bool valid = r0 + rb < rows;
        float* out = part + ((size_t)(r0 + rb) * NATOM + a) * QS;
        if constexpr (MG == 1) {
            const float t = reduce_scatter32(v[rb], lane);  // all lanes take part, only valid rows store
            if (valid) out[lane] = t;
        } else {
#pragma unroll
            for (int i = 0; i < QP; ++i)
                if (valid && lane == i && qbeg + i < NQ) out[qbeg + i] = v[rb][i];
            if (valid && warp == 0 && lane >= NQ) out[lane] = 0.f;  // pad slots 25..31 (never read; no stale bits in the workspace)
        }
    }
}

// ------------------------------------------------------------------------------------------------ finalize (K2)

// 16 warps of 32 threads normalise and collapse (named barrier 1 for their block reduction); the 17th warp reduces the atom
// partials, derives post/comb (Sinkhorn on 16 lanes) and leaves. Thread t owns float4 columns t and t + 512 of every stream.
__device__ __forceinline__ void bar_collapse() { asm volatile("bar.sync 1, 512;" ::: "memory"); }

template <int OT, bool NORM>
__global__ void __launch_bounds__(FIN_THREADS)
    finalize_k(const float* x, const float* part, const float* base, const float* scale, const float* norm_w, float* post, float* comb,
               float* pre_out, void* out, float rms_eps, float hc_eps, float norm_eps, int iters) {
    __shared__ float S[QS];
    __shared__ float red[FIN_NT / 32];
    const int r = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float4* x4 = (const float4*)(x + (size_t)r * ROW);
    float4 xv[2][NS];
    pdl_wait();  // x and the atom partials are the predecessor's
    if (warp < FIN_NT / 32) {
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
            for (int h = 0; h < NS; ++h) xv[k][h] = x4[(size_t)h * VEC + tid + FIN_NT * k];
    } else {
        float s = 0.f;
        if (lane < NQ) {
            const float* p = part + (size_t)r * NATOM * QS + lane;
            float t[NATOM];
#pragma unroll
            for (int a = 0; a < NATOM; ++a) t[a] = p[(size_t)a * QS];
            s = t[0];
#pragma unroll
            for (int a = 1; a < NATOM; ++a) s = s + t[a];  // the fixed chain over atoms
        }
        S[lane] = s;
    }
    __syncthreads();
    pdl_trigger();  // x and the partials are read: the successor (the layer's first linear) may launch and prefetch its weights
    const float rmr = 1.0f / sqrtf(S[NM] / (float)ROW + rms_eps);
    float pre[NS];
#pragma unroll
    for (int h = 0; h < NS; ++h) pre[h] = sigmoid_f(fmaf(S[h] * rmr, scale[0], base[h])) + hc_eps;
    if (pre_out && tid < NS) pre_out[(size_t)r * NS + tid] = pre[tid];

    if (warp == FIN_NT / 32) {  // post, comb: lane l < 16 = (row i, col j) = (l / 4, l % 4); row sums xor 1, 2; column sums xor 4, 8
        if (lane < NS) post[(size_t)r * NS + lane] = 2.0f * sigmoid_f(fmaf(S[NS + lane] * rmr, scale[1], base[NS + lane]));
        if (lane < NS * NS) {
            const unsigned mask = 0xFFFFu;
            float v = fmaf(S[2 * NS + lane] * rmr, scale[2], base[2 * NS + lane]);
            float mx = fmaxf(v, __shfl_xor_sync(mask, v, 1));
            mx = fmaxf(mx, __shfl_xor_sync(mask, mx, 2));
            v = expf(v - mx);
            float s = v;
            s += __shfl_xor_sync(mask, s, 1);
            s += __shfl_xor_sync(mask, s, 2);
            v = v / s + hc_eps;
            float cs = v;
            cs += __shfl_xor_sync(mask, cs, 4);
            cs += __shfl_xor_sync(mask, cs, 8);
            v = v / (cs + hc_eps);
            for (int it = 0; it < iters - 1; ++it) {
                float rs = v;
                rs += __shfl_xor_sync(mask, rs, 1);
                rs += __shfl_xor_sync(mask, rs, 2);
                v = v / (rs + hc_eps);
                cs = v;
                cs += __shfl_xor_sync(mask, cs, 4);
                cs += __shfl_xor_sync(mask, cs, 8);
                v = v / (cs + hc_eps);
            }
            comb[(size_t)r * NS * NS + lane] = v;
        }
        return;
    }

    float4 cv[2];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        float o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int h = 0; h < NS; ++h) {
            o[0] = fmaf(pre[h], xv[k][h].x, o[0]);
            o[1] = fmaf(pre[h], xv[k][h].y, o[1]);
            o[2] = fmaf(pre[h], xv[k][h].z, o[2]);
            o[3] = fmaf(pre[h], xv[k][h].w, o[3]);
        }
        cv[k] = make_float4(o[0], o[1], o[2], o[3]);
        if (NORM) {
            ss = fmaf(o[0], o[0], ss);
            ss = fmaf(o[1], o[1], ss);
            ss = fmaf(o[2], o[2], ss);
            ss = fmaf(o[3], o[3], ss);
        }
    }
    float inv = 0.f;
    if (NORM) {
        ss = warp_sum(ss);
        if (lane == 0) red[warp] = ss;
        bar_collapse();
        float tot = red[0];
#pragma unroll
        for (int w = 1; w < FIN_NT / 32; ++w) tot = tot + red[w];
        inv = 1.0f / sqrtf(tot / (float)HD + norm_eps);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int c4 = tid + FIN_NT * k;
        float4 o = cv[k];
        if (NORM) {
            const float4 w = *(const float4*)(norm_w + (size_t)c4 * 4);
            o.x = (o.x * inv) * w.x;
            o.y = (o.y * inv) * w.y;
            o.z = (o.z * inv) * w.z;
            o.w = (o.w * inv) * w.w;
        }
        st4<OT>(out, (size_t)r * HD + (size_t)c4 * 4, o.x, o.y, o.z, o.w);
    }
}

// ------------------------------------------------------------------------------------------------ apply (standalone)

template <int YT>
__global__ void __launch_bounds__(128) apply_k(float* x, const void* y, const float* post_g, const float* comb_g) {
    const int r = blockIdx.y, c4 = blockIdx.x * blockDim.x + threadIdx.x;  // grid.x * 128 == VEC
    float p[NS], c[NS * NS];
#pragma unroll
    for (int h = 0; h < NS; ++h) p[h] = post_g[(size_t)r * NS + h];
#pragma unroll
    for (int k = 0; k < NS * NS; ++k) c[k] = comb_g[(size_t)r * NS * NS + k];
    float* xr = x + (size_t)r * ROW;
    float4 xv[NS];
#pragma unroll
    for (int h = 0; h < NS; ++h) xv[h] = *(const float4*)(xr + (size_t)h * HD + (size_t)c4 * 4);
    const float4 yv = ld4<YT>(y, (size_t)r * HD + (size_t)c4 * 4);
    apply_vec(xv, yv, p, c);
#pragma unroll
    for (int h = 0; h < NS; ++h) *(float4*)(xr + (size_t)h * HD + (size_t)c4 * 4) = xv[h];
}

// ------------------------------------------------------------------------------------------------ head: stream mean (+ norm)

// c = ((x0 + x1) + x2) + x3, times 0.25 (exact); optionally the model's final RMSNorm in the same pass.
template <int OT, bool NORM>
__global__ void __launch_bounds__(FIN_NT) mean_k(const float* x, const float* norm_w, void* out, float norm_eps) {
    __shared__ float red[FIN_NT / 32];
    const int r = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float4* x4 = (const float4*)(x + (size_t)r * ROW);
    float4 xv[2][NS];
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int h = 0; h < NS; ++h) xv[k][h] = x4[(size_t)h * VEC + tid + FIN_NT * k];
    float4 cv[2];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        float o[4];
        o[0] = ((xv[k][0].x + xv[k][1].x) + xv[k][2].x) + xv[k][3].x;
        o[1] = ((xv[k][0].y + xv[k][1].y) + xv[k][2].y) + xv[k][3].y;
        o[2] = ((xv[k][0].z + xv[k][1].z) + xv[k][2].z) + xv[k][3].z;
        o[3] = ((xv[k][0].w + xv[k][1].w) + xv[k][2].w) + xv[k][3].w;
        o[0] *= 0.25f;
        o[1] *= 0.25f;
        o[2] *= 0.25f;
        o[3] *= 0.25f;
        cv[k] = make_float4(o[0], o[1], o[2], o[3]);
        if (NORM) {
            ss = fmaf(o[0], o[0], ss);
            ss = fmaf(o[1], o[1], ss);
            ss = fmaf(o[2], o[2], ss);
            ss = fmaf(o[3], o[3], ss);
        }
    }
    float inv = 0.f;
    if (NORM) {
        ss = warp_sum(ss);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = red[0];
#pragma unroll
        for (int w = 1; w < FIN_NT / 32; ++w) tot = tot + red[w];
        inv = 1.0f / sqrtf(tot / (float)HD + norm_eps);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int c4 = tid + FIN_NT * k;
        float4 o = cv[k];
        if (NORM) {
            const float4 w = *(const float4*)(norm_w + (size_t)c4 * 4);
            o.x = (o.x * inv) * w.x;
            o.y = (o.y * inv) * w.y;
            o.z = (o.z * inv) * w.z;
            o.w = (o.w * inv) * w.w;
        }
        st4<OT>(out, (size_t)r * HD + (size_t)c4 * 4, o.x, o.y, o.z, o.w);
    }
}

// ------------------------------------------------------------------------------------------------ expand

// x[r][h][d] = float(in[r][d]) for the 4 streams (ExpandStreams). IN: rows of any activation type (exact widening).
template <int IT>
__global__ void __launch_bounds__(256) expand_rows_k(const void* in, float* x) {
    const int r = blockIdx.y, c4 = blockIdx.x * blockDim.x + threadIdx.x;  // grid.x * 256 == VEC
    const float4 v = ld4<IT>(in, (size_t)r * HD + (size_t)c4 * 4);
#pragma unroll
    for (int h = 0; h < NS; ++h) *(float4*)(x + (size_t)r * ROW + (size_t)h * HD + (size_t)c4 * 4) = v;
}

// Same from a bf16 embedding table and token ids; an id outside [0, vocab) gives zeros (as gb_embed).
__global__ void __launch_bounds__(256) embed_expand_k(const u16* table, const int* ids, float* x, int vocab) {
    const int r = blockIdx.y, c4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int id = ids[r];
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if ((unsigned)id < (unsigned)vocab) v = ld4<T_BF16>(table, (size_t)(unsigned)id * HD + (size_t)c4 * 4);
#pragma unroll
    for (int h = 0; h < NS; ++h) *(float4*)(x + (size_t)r * ROW + (size_t)h * HD + (size_t)c4 * 4) = v;
}

// ------------------------------------------------------------------------------------------------ launchers

static inline bool al(const void* p, int n) { return ((uintptr_t)p & (uintptr_t)(n - 1)) == 0; }
static inline int tal(int t) { return t == T_F32 ? 16 : 8; }  // alignment of a 4-wide access of type t
static inline bool ok_type(int t) { return t == T_F32 || t == T_F16 || t == T_BF16; }
static inline bool ok_rows(int rows) { return rows >= 0 && rows <= 65535; }

#define DISPATCH3(T, FN)                                  \
    switch (T) {                                          \
        case T_F32: FN(T_F32); break;                     \
        case T_F16: FN(T_F16); break;                     \
        default: FN(T_BF16); break;                       \
    }

// split = m-groups per atom (warps per block): 1, 2, 4 or 8; rb = rows per block: 1, 2 or 4; 0 = automatic from the row count. Latency
// and bandwidth knobs only: every value is computed by the same operations in the same order for any (split, rb).
int launch_partials(float* x, const void* fn, int fn_type, float* part, int rows, int split, int rb, void* stream, bool apply,
                    const void* y, int y_type, const float* post, const float* comb) {
    if (!ok_rows(rows) || !ok_type(fn_type) || !ok_type(y_type)) return ERR_INVALID;
    // automatic choice from the sweep on a GB10 (tests/mhc.rs bench_mhc_kernels): latency-bound below 5 rows, then L2-traffic-bound
    if (split != 1 && split != 2 && split != 4 && split != 8) split = rows <= 4 ? 4 : 1;
    if (rb != 1 && rb != 2 && rb != 4) rb = rows <= 16 ? 1 : 2;
    if (rows == 0) return 0;
    if (!al(x, 16) || !al(fn, tal(fn_type)) || !al(part, 4)) return ERR_INVALID;
    if (apply && (!al(y, tal(y_type)) || !post || !comb)) return ERR_INVALID;
    const dim3 grid(NATOM, (rows + rb - 1) / rb);
    cudaStream_t s = gb_stream(stream);
    int rc = 0;
#define LK(WT, YT, AP, MG, RB) rc = gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMhc), partials_k<WT, YT, AP, MG, RB>, grid, dim3((MG) * 32), 0, s, x, fn, part, y, post, comb, rows)
#define LR(WT, YT, AP, MG)                               \
    switch (rb) {                                        \
        case 1: LK(WT, YT, AP, MG, 1); break;            \
        case 2: LK(WT, YT, AP, MG, 2); break;            \
        default: LK(WT, YT, AP, MG, 4); break;           \
    }
#define LM(WT, YT, AP)                                   \
    switch (split) {                                     \
        case 1: LR(WT, YT, AP, 1); break;                \
        case 2: LR(WT, YT, AP, 2); break;                \
        case 4: LR(WT, YT, AP, 4); break;                \
        default: LR(WT, YT, AP, 8); break;               \
    }
    if (!apply) {
#define L(WT) LM(WT, T_F32, false)
        DISPATCH3(fn_type, L)
#undef L
    } else {
#define L(WT)                                            \
    switch (y_type) {                                    \
        case T_F32: LM(WT, T_F32, true); break;          \
        case T_F16: LM(WT, T_F16, true); break;          \
        default: LM(WT, T_BF16, true); break;            \
    }
        DISPATCH3(fn_type, L)
#undef L
    }
#undef LM
#undef LR
#undef LK
    return rc;
}

}  // namespace

GB_EXPORT size_t gb_mhc_partials_bytes(int rows) { return rows > 0 ? (size_t)rows * NATOM * QS * sizeof(float) : 0; }

GB_EXPORT int gb_mhc_expand(const void* in, int in_type, float* x, int rows, void* stream) {
    if (!ok_rows(rows) || !ok_type(in_type)) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!al(in, tal(in_type)) || !al(x, 16)) return ERR_INVALID;
    const dim3 grid(VEC / 256, rows);
#define L(T) expand_rows_k<T><<<grid, 256, 0, gb_stream(stream)>>>(in, x)
    DISPATCH3(in_type, L)
#undef L
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_mhc_embed_expand(const void* table, const int* ids, float* x, int rows, int vocab, void* stream) {
    if (!ok_rows(rows) || vocab <= 0) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!al(table, 8) || !al(x, 16)) return ERR_INVALID;
    embed_expand_k<<<dim3(VEC / 256, rows), 256, 0, gb_stream(stream)>>>((const u16*)table, ids, x, vocab);
    return GB_CHECK_LAUNCH();
}

// K1: atom partials of the dots and the sum of squares. `split` (1, 2, 4, 8 m-groups per atom) and `rb` (1, 2, 4 rows per block); 0 =
// automatic. Latency / bandwidth knobs only, they never change a bit of the output.
GB_EXPORT int gb_mhc_partials(const float* x, const void* fn, int fn_type, float* part, int rows, int split, int rb, void* stream) {
    return launch_partials((float*)x, fn, fn_type, part, rows, split, rb, stream, false, nullptr, T_F32, nullptr, nullptr);
}

// Apply the previous site's update in place and take the next site's partials of the updated streams in the same pass.
GB_EXPORT int gb_mhc_apply_partials(float* x, const void* y, int y_type, const float* post, const float* comb, const void* fn_next,
                                    int fn_type, float* part, int rows, int split, int rb, void* stream) {
    return launch_partials(x, fn_next, fn_type, part, rows, split, rb, stream, true, y, y_type, post, comb);
}

// K2: reduce the partials, derive pre/post/comb, collapse the streams and (norm_w != null) RMSNorm the collapsed row with
// weight norm_w (multiplicative, as given) into `out` (f32, f16 or bf16, one rounding). With norm_w == null `out` is the plain
// collapsed row. post [rows][4] and comb [rows][4][4] (comb[g][h]: row g, column h) are written for the apply; pre_out
// (nullable, [rows][4]) is a debug/test output.
GB_EXPORT int gb_mhc_finalize(const float* x, const float* part, const float* base, const float* scale, const float* norm_w, float* post,
                              float* comb, float* pre_out, void* out, int out_type, int rows, float rms_eps, float hc_eps,
                              float norm_eps, int iters, void* stream) {
    if (!ok_rows(rows) || !ok_type(out_type) || iters < 1 || iters > 256) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!al(x, 16) || !al(part, 4) || !al(out, tal(out_type)) || (norm_w && !al(norm_w, 16))) return ERR_INVALID;
    cudaStream_t s = gb_stream(stream);
    const bool norm = norm_w != nullptr;
    int rc = 0;
#define LN(OT, N) rc = gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMhc), finalize_k<OT, N>, dim3(rows), dim3(FIN_THREADS), 0, s, x, part, base, scale, norm_w, post, comb, pre_out, out, rms_eps, hc_eps, norm_eps, iters)
#define L(OT)                                            \
    if (norm) LN(OT, true); else LN(OT, false)
    DISPATCH3(out_type, L)
#undef L
#undef LN
    return rc;
}

// Apply alone: x[h] <- post[h] y + sum_g comb[g][h] x[g], in place (y f32, f16 or bf16).
GB_EXPORT int gb_mhc_apply(float* x, const void* y, int y_type, const float* post, const float* comb, int rows, void* stream) {
    if (!ok_rows(rows) || !ok_type(y_type)) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!al(x, 16) || !al(y, tal(y_type))) return ERR_INVALID;
    const dim3 grid(VEC / 128, rows);
#define L(YT) apply_k<YT><<<grid, 128, 0, gb_stream(stream)>>>(x, y, post, comb)
    DISPATCH3(y_type, L)
#undef L
    return GB_CHECK_LAUNCH();
}

// Head: mean over the 4 streams ((x0 + x1) + x2) + x3, times 0.25; with norm_w the model's final RMSNorm in the same pass.
GB_EXPORT int gb_mhc_mean(const float* x, const float* norm_w, void* out, int out_type, int rows, float norm_eps, void* stream) {
    if (!ok_rows(rows) || !ok_type(out_type)) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!al(x, 16) || !al(out, tal(out_type)) || (norm_w && !al(norm_w, 16))) return ERR_INVALID;
    cudaStream_t s = gb_stream(stream);
    const bool norm = norm_w != nullptr;
#define LN(OT, N) mean_k<OT, N><<<rows, FIN_NT, 0, s>>>(x, norm_w, out, norm_eps)
#define L(OT)                                            \
    if (norm) LN(OT, true); else LN(OT, false)
    DISPATCH3(out_type, L)
#undef L
#undef LN
    return GB_CHECK_LAUNCH();
}
