// gb-nvcc: --fmad=false
// DeepSeek-V4.1 MoE router + hyper-connection mix, at this pack's shapes (hidden 5120, 384 routed experts, top-6,
// 4 residual streams). Mirror of the reference `Gate` / `Expert` / `Block.hc_mixes` in `model.py` (lines 794-996) and of
// `kernel.py: hc_split_sinkhorn_kernel` (lines 407-462).
//
// ---------------------------------------------------------------------------------------------------------------
// ROUTER (`Gate.forward`)
//
//   scores  = (x.float() @ W.float()^T) / gate_temp                       gate_temp = 1.0 for this pack
//   scores  = sqrt(softplus(scores))                                      score_func = "sqrtsoftplus"
//   bias    = bias_vl if (image_mask is not None and this token is in an image span) else bias
//   indices = topk(scores + bias, K)[1]                                   noaux_tc: the bias STEERS SELECTION ONLY
//   weights = scores.gather(indices)                                     ... the returned weight is the UNBIASED score
//   weights = weights / (weights.sum(-1) + 1e-20)                        norm_topk_prob, not norm_eps
//   weights = weights * 1.5                                              routed_scaling_factor
//
// The dual bias is a pure per-token select of which [384] fp32 vector is added for selection; it never enters the weight.
// `noaux_tc` has no auxiliary loss term, so nothing here is trained-through: it is a deterministic function of the logits.
//
// softplus is evaluated in its stable form  log1p(exp(-|x|)) + max(x, 0): for x -> +inf that is x + tiny (no overflow of
// exp(x)), for x -> -inf it is exp(x) to full relative precision (no cancellation in log(1 + exp(x))). sqrt is the
// monotone outer map, so the ordering of scores equals the ordering of softplus scores.
//
// ---------------------------------------------------------------------------------------------------------------
// HYPER-CONNECTIONS (`Block.hc_mixes`, `Block.hc_pre`, `Block.hc_post`)
//
//   x: [rows][stream h][d] fp32, 4 streams of 5120 = 20480 values per row
//   S        = [x . fn^T (24 dots), sum x^2]                             reduction over the whole 20480-value row
//   rmr      = rsqrt(sum x^2 / 20480 + norm_eps)                         model.py uses args.norm_eps, 1e-20 here
//   pre[h]   = sigmoid(fma(S[h]     * rmr, scale0, base[h]))     + hc_eps
//   post[h]  = 2 * sigmoid(fma(S[h + 4] * rmr, scale1, base[h + 4]))
//   comb     = sinkhorn(softmax_rows(fma(S[8..24] * rmr, scale2, base[8..24])))   -- 20 iterations, see below
//   collapsed[d] = sum_h pre[h] x[h][d]                                  then the layer's RMSNorm (caller)
//   x[h][d] <- post[h] y[d] + sum_g comb[g][h] x[g][d]                   in place
//
// Sinkhorn (exactly `hc_split_sinkhorn_kernel`): softmax rows, `+ eps` on the normalised entries, one COLUMN
// normalisation, then `iters - 1` rounds of (row, column), all divisions `/(sum + eps)`. 20 iterations drive a 4x4
// positive matrix to a doubly stochastic one to far below fp32 resolution.
//
// SHAPES. This file is the 5120/20480/40-atom instantiation. The 4096 (GLM-5.3) instantiation lives in `mhc.cu` and is
// NOT modified; the host mirror in `src/dsv4_moe.rs` is written over (hidden, streams) so the tests can prove the two
// agree bit for bit when the 4096 tensors are zero-padded into the 5120 layout (zeros contribute exact +0.0 terms, and
// the atom chain over 40 atoms ends in 8 more exact additions of 0).
//
// ROW INVARIANCE. Row r's bits depend only on row r: the 20480 values are cut into 40 atoms of 128 d x 4 streams
// (lane l of atom a owns d = 128a + 4l .. +3), each atom's 25 quantities are an fma chain over its 16 values in
// (stream, d) order followed by a fixed xor butterfly, and the finalize re-reduces the 40 atom partials as a
// sequential chain a = 0..39. No atomics, no cross-row state, no host synchronisation.
//
// ARITHMETIC. fmaf() is explicit where an FMA is wanted and --fmad=false keeps every other a*b+c as two roundings, so
// the host mirror is exact. Transcendentals are the accurate ones (expf / log1pf / sqrtf / IEEE `/`): the tail is 24
// values per row, so `__expf` would buy nothing and cost bits against a float64 reference.
#include "common.cuh"
#include <math.h>

namespace {

typedef uint16_t u16;
typedef uint32_t u32;

constexpr int ERR_INVALID = 1;  // cudaErrorInvalidValue

// ---- MoE router geometry (this pack)
constexpr int HID = 5120;  // model width
constexpr int E = 384;     // routed experts
constexpr int K = 6;       // activated experts
constexpr int WARPS_G = 8; // gate-logits: experts per block

// ---- hyper-connection geometry
constexpr int NS = 4;                 // streams
constexpr int HD = 5120;              // hidden size of a stream
constexpr int ROW = NS * HD;          // 20480 values per row
constexpr int NM = 24;                // fn rows: 4 pre, 4 post, 16 comb
constexpr int NQ = NM + 1;            // + sum of squares (slot 24)
constexpr int ATOM_D = 128;           // d per atom
constexpr int NATOM = HD / ATOM_D;    // 40
constexpr int PW = 8;                 // partials: warps per block
constexpr int QP = (NQ + PW - 1) / PW;  // 4 quantities per warp
constexpr int FIN_T = 512;            // finalize threads (collapse)
constexpr int VEC = HD / 4;           // float4 columns per stream (1280)
constexpr int APL = 128;              // apply: threads per block

// Dtype codes of the crate's C ABI (`exl3_linear::Dtype`): F16 = 0, BF16 = 1, F32 = 2.
enum { T_F16 = 0, T_BF16 = 1, T_F32 = 2 };

// ------------------------------------------------------------------------------------------------ conversions / helpers

__device__ __forceinline__ float bf2f(u16 b) { return __uint_as_float((u32)b << 16); }

template <int T>
__device__ __forceinline__ float ldx(const void* p, size_t i) {
    if constexpr (T == T_F32) {
        return ((const float*)p)[i];
    } else if constexpr (T == T_F16) {
        return __half2float(((const __half*)p)[i]);
    } else {
        return bf2f(((const u16*)p)[i]);
    }
}

__device__ __forceinline__ float4 ld4f(const void* p, size_t i) {
    return *(const float4*)((const float*)p + i);
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// The scoring function: sqrt(softplus(x)), stable form. Same bits as the host mirror in src/dsv4_moe.rs.
__device__ __forceinline__ float sqrtsoftplus_f(float x) {
    const float m = fabsf(x);
    return sqrtf(log1pf(expf(-m)) + fmaxf(x, 0.0f));
}

// Fixed xor butterfly 16, 8, 4, 2, 1: every lane ends with the same bits.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// ------------------------------------------------------------------------------------------------ 1. router logits

template <int T>
__device__ __forceinline__ void ldx8(const void* p, size_t base, float (&v)[8]) {
    if constexpr (T == T_F32) {
        const float4 a = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(p) + base));
        const float4 b = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(p) + base + 4));
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else if constexpr (T == T_F16) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(static_cast<const __half*>(p) + base));
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&u.z));
        const float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&u.w));
        v[0] = f0.x; v[1] = f0.y; v[2] = f1.x; v[3] = f1.y;
        v[4] = f2.x; v[5] = f2.y; v[6] = f3.x; v[7] = f3.y;
    } else {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(static_cast<const u16*>(p) + base));
        v[0] = bf2f(u.x & 0xFFFFu); v[1] = bf2f(u.x >> 16);
        v[2] = bf2f(u.y & 0xFFFFu); v[3] = bf2f(u.y >> 16);
        v[4] = bf2f(u.z & 0xFFFFu); v[5] = bf2f(u.z >> 16);
        v[6] = bf2f(u.w & 0xFFFFu); v[7] = bf2f(u.w >> 16);
    }
}

__device__ __forceinline__ uint32_t cvt2_bf16_to_f16(uint32_t u) {
    const float f0 = bf2f(u & 0xFFFFu);
    const float f1 = bf2f(u >> 16);
    const __half2 h = __floats2half2_rn(f0, f1);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ uint4 cvt8_bf16_to_f16_rtr(uint4 u) {
    return make_uint4(cvt2_bf16_to_f16(u.x), cvt2_bf16_to_f16(u.y), cvt2_bf16_to_f16(u.z), cvt2_bf16_to_f16(u.w));
}

__device__ __forceinline__ void ldsm_x4_rtr(uint32_t (&a)[4], const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

__device__ __forceinline__ void mma16816_rtr(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int T>
__global__ void __launch_bounds__(128) router_logits_mma_k(const void* __restrict__ x,
                                                           const u16* __restrict__ w,
                                                           float* __restrict__ logits,
                                                           int rows) {
    constexpr int BM = 32;
    constexpr int BN = 64;
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
    const int n_warp = n0 + warp * 16;

    float acc[2][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mt][nt][c] = 0.0f;

    for (int k_base = 0; k_base < HID; k_base += BK) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = i * 128 + tid;
            const int r = idx >> 3;
            const int c = idx & 7;
            const int gr = m0 + r;
            const int dst_off = r * BK + ((c ^ (r & 7)) << 3);
            if (gr < rows) {
                const size_t x_off = (size_t)gr * HID + k_base + c * 8;
                if constexpr (T == T_F16) {
                    *reinterpret_cast<uint4*>(sa + dst_off) =
                        __ldg(reinterpret_cast<const uint4*>(static_cast<const __half*>(x) + x_off));
                } else if constexpr (T == T_BF16) {
                    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(static_cast<const u16*>(x) + x_off));
                    *reinterpret_cast<uint4*>(sa + dst_off) = cvt8_bf16_to_f16_rtr(raw);
                } else {
                    const float4 a = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(x) + x_off));
                    const float4 b = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(x) + x_off + 4));
                    const __half2 h0 = __floats2half2_rn(a.x, a.y);
                    const __half2 h1 = __floats2half2_rn(a.z, a.w);
                    const __half2 h2 = __floats2half2_rn(b.x, b.y);
                    const __half2 h3 = __floats2half2_rn(b.z, b.w);
                    *reinterpret_cast<uint4*>(sa + dst_off) = make_uint4(
                        *reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1),
                        *reinterpret_cast<const uint32_t*>(&h2), *reinterpret_cast<const uint32_t*>(&h3));
                }
            } else {
                *reinterpret_cast<uint4*>(sa + dst_off) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = i * 128 + tid;
            const int nr = idx >> 3;
            const int c = idx & 7;
            const int gn = n0 + nr;
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(w + (size_t)gn * HID + k_base + c * 8));
            sb[nr * 8 + (c ^ (nr & 7))] = cvt8_bf16_to_f16_rtr(raw);
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t a[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r = mt * 16 + lr;
                const int c = ks * 2 + lc;
                ldsm_x4_rtr(a[mt], sa + r * BK + ((c ^ (r & 7)) << 3));
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int nr = warp * 16 + nt * 8 + g;
                const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&sb[nr * 8 + ((ks * 2) ^ (nr & 7))]);
                const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&sb[nr * 8 + ((ks * 2 + 1) ^ (nr & 7))]);
                const uint32_t b[2] = {p0[t], p1[t]};
                mma16816_rtr(acc[0][nt], a[0], b);
                mma16816_rtr(acc[1][nt], a[1], b);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int r0 = m0 + mt * 16 + g;
        const int r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = n_warp + nt * 8 + 2 * t;
            if (col < E) {
                if (r0 < rows) {
                    *reinterpret_cast<float2*>(logits + (size_t)r0 * E + col) =
                        make_float2(acc[mt][nt][0], acc[mt][nt][1]);
                }
                if (r1 < rows) {
                    *reinterpret_cast<float2*>(logits + (size_t)r1 * E + col) =
                        make_float2(acc[mt][nt][2], acc[mt][nt][3]);
                }
            }
        }
    }
}
template <int T, int ROWS>
__global__ void __launch_bounds__(128) router_logits_k(const void* __restrict__ x,
                                                       const u16* __restrict__ w,
                                                       float* __restrict__ logits,
                                                       int rows) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e = blockIdx.y * 4 + warp;
    if (e >= E) return;
    const int r0 = blockIdx.x * ROWS;
    const u16* wr = w + (size_t)e * HID;
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
    for (int k_base = 0; k_base < HID; k_base += 256) {
        const int i = k_base + lane * 8;
        const uint4 uw = __ldg(reinterpret_cast<const uint4*>(wr + i));
        const float wv[8] = {
            bf2f(uw.x & 0xFFFFu), bf2f(uw.x >> 16),
            bf2f(uw.y & 0xFFFFu), bf2f(uw.y >> 16),
            bf2f(uw.z & 0xFFFFu), bf2f(uw.z >> 16),
            bf2f(uw.w & 0xFFFFu), bf2f(uw.w >> 16)
        };
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int rc = r0 + r;
            if (rc < rows) {
                float xv[8];
                ldx8<T>(x, (size_t)rc * HID + i, xv);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[r] = fmaf(xv[j], wv[j], acc[r]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int rc = r0 + r;
        if (rc < rows) {
            float v = acc[r];
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0) logits[(size_t)rc * E + e] = v;
        }
    }
}

// ------------------------------------------------------------------------------------------------ 2. noaux_tc top-k

// One block of 128 threads per row. All 128 threads evaluate sqrtsoftplus_f in parallel (3 experts/thread),
// then warp 0 selects top-K in K=6 warp-parallel argmax passes (ties to lower expert id).
template <bool F16LOG>
__global__ void __launch_bounds__(128) router_select_k(const void* __restrict__ logits,
                                                       const float* __restrict__ tab,
                                                       const float* __restrict__ bias,
                                                       const float* __restrict__ bias_vl,
                                                       const unsigned char* __restrict__ mask,
                                                       int* __restrict__ pick,
                                                       u16* __restrict__ wts,
                                                       int rows,
                                                       float scale,
                                                       float top_p) {
    __shared__ float s_key[E];
    __shared__ float s_sc[E];
    __shared__ uint8_t s_used[E];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;
    const float* b = (mask != nullptr && mask[r]) ? bias_vl : bias;
#pragma unroll
    for (int j = 0; j < E / 128; ++j) {
        const int e = tid + j * 128;
        const float s = F16LOG ? tab[((const u16*)logits)[(size_t)r * E + e]]
                               : sqrtsoftplus_f(((const float*)logits)[(size_t)r * E + e]);
        s_sc[e] = s;
        s_key[e] = s + b[e];
        s_used[e] = 0;
    }
    __syncthreads();
    if (tid < 32) {
        const int lane = tid;
        float ss[K];
        int iv[K];
#pragma unroll
        for (int k = 0; k < K; ++k) {
            float best_k = -INFINITY;
            int best_e = E;
#pragma unroll
            for (int j = 0; j < E / 32; ++j) {
                const int e = lane * (E / 32) + j;
                if (!s_used[e]) {
                    const float kv = s_key[e];
                    if (best_e == E || kv > best_k || (kv == best_k && e < best_e)) {
                        best_k = kv;
                        best_e = e;
                    }
                }
            }
#pragma unroll
            for (int m = 16; m >= 1; m >>= 1) {
                const float ok = __shfl_xor_sync(0xffffffffu, best_k, m);
                const int oe = __shfl_xor_sync(0xffffffffu, best_e, m);
                if (oe < E && (best_e == E || ok > best_k || (ok == best_k && oe < best_e))) {
                    best_k = ok;
                    best_e = oe;
                }
            }
            iv[k] = best_e;
            ss[k] = s_sc[best_e];
            if (lane == 0) {
                s_used[best_e] = 1;
            }
            __syncwarp();
        }
        if (lane == 0) {
            if (top_p < 1.0f) {
#pragma unroll
                for (int i = 0; i < K - 1; ++i) {
#pragma unroll
                    for (int j = i + 1; j < K; ++j) {
                        if (ss[j] > ss[i] || (ss[j] == ss[i] && iv[j] < iv[i])) {
                            const float ts = ss[i];
                            ss[i] = ss[j];
                            ss[j] = ts;
                            const int ti = iv[i];
                            iv[i] = iv[j];
                            iv[j] = ti;
                        }
                    }
                }
                float sum = 0.f;
#pragma unroll
                for (int i = 0; i < K; ++i) sum += ss[i];
                const float sum_den = sum + 1e-20f;
                float cum_p = 0.f;
                float kept_sum = 0.f;
#pragma unroll
                for (int j = 0; j < K; ++j) {
                    if (j < 3 || cum_p < top_p) {
                        cum_p += ss[j] / sum_den;
                        kept_sum += ss[j];
                    } else {
                        iv[j] = -1;
                        ss[j] = 0.f;
                    }
                }
                const float den = kept_sum + 1e-20f;
#pragma unroll
                for (int j = 0; j < K; ++j) {
                    pick[(size_t)r * K + j] = iv[j];
                    if (iv[j] >= 0) {
                        const float w = (ss[j] / den) * scale;
                        wts[(size_t)r * K + j] = __half_as_ushort(__float2half_rn(w));
                    } else {
                        wts[(size_t)r * K + j] = 0u;
                    }
                }
            } else {
                float sum = 0.f;
#pragma unroll
                for (int i = 0; i < K; ++i) sum += ss[i];
                const float den = sum + 1e-20f;
#pragma unroll
                for (int i = 0; i < K; ++i) {
                    pick[(size_t)r * K + i] = iv[i];
                    const float w = (ss[i] / den) * scale;
                    wts[(size_t)r * K + i] = __half_as_ushort(__float2half_rn(w));
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ 3. hc partials

// One block = one atom of ROWS rows: PW warps split the 25 quantities into chunks of QP, reading `fn` once.
template <int ROWS>
__global__ void __launch_bounds__(PW * 32) hc_partials_k(float* __restrict__ x,
                                                         const float* __restrict__ fn,
                                                         float* __restrict__ part,
                                                         const float* __restrict__ pre_in,
                                                         float* __restrict__ collapsed_out,
                                                         const void* __restrict__ y_prev,
                                                         const float* __restrict__ post_prev,
                                                         const float* __restrict__ comb_prev,
                                                         int rows) {
    __shared__ float4 sv[ROWS][NS][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int a = blockIdx.x, r0 = blockIdx.y * ROWS;
    if (r0 >= rows) return;
    const int d0 = a * ATOM_D + lane * 4;
    if (y_prev != nullptr) {
#pragma unroll
        for (int r = warp; r < ROWS; r += PW) {
            const int rc = r0 + r;
            if (rc < rows) {
                float* xr = x + (size_t)rc * ROW;
                float4 xv[NS];
#pragma unroll
                for (int h = 0; h < NS; ++h) xv[h] = ld4f(xr, (size_t)h * HD + d0);
                const size_t yi = (size_t)rc * HD + d0;
                const uint2 u = *(const uint2*)((const __half*)y_prev + yi);
                const float2 a2 = __half22float2(*(const __half2*)&u.x), b2 = __half22float2(*(const __half2*)&u.y);
                const float4 yv = make_float4(a2.x, a2.y, b2.x, b2.y);
                const float4 p4 = *(const float4*)(post_prev + (size_t)rc * NS);
                const float p[4] = {p4.x, p4.y, p4.z, p4.w};
                float c[NS * NS];
#pragma unroll
                for (int k = 0; k < NS * NS; k += 4) {
                    const float4 c4 = *(const float4*)(comb_prev + (size_t)rc * NS * NS + k);
                    c[k] = c4.x; c[k + 1] = c4.y; c[k + 2] = c4.z; c[k + 3] = c4.w;
                }
#pragma unroll
                for (int h = 0; h < NS; ++h) {
                    float4 o = make_float4(p[h] * yv.x, p[h] * yv.y, p[h] * yv.z, p[h] * yv.w);
#pragma unroll
                    for (int g = 0; g < NS; ++g) {
                        o.x = fmaf(c[g * NS + h], xv[g].x, o.x);
                        o.y = fmaf(c[g * NS + h], xv[g].y, o.y);
                        o.z = fmaf(c[g * NS + h], xv[g].z, o.z);
                        o.w = fmaf(c[g * NS + h], xv[g].w, o.w);
                    }
                    *(float4*)(xr + (size_t)h * HD + d0) = o;
                    sv[r][h][lane] = o;
                }
            } else {
#pragma unroll
                for (int h = 0; h < NS; ++h) sv[r][h][lane] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    } else {
#pragma unroll
        for (int idx = warp; idx < ROWS * NS; idx += PW) {
            const int r = idx / NS;
            const int h = idx % NS;
            const int rc = r0 + r;
            const float* xr = x + (size_t)rc * ROW;
            sv[r][h][lane] = (rc < rows) ? ld4f(xr, (size_t)h * HD + d0) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    __syncthreads();
    if (pre_in != nullptr && collapsed_out != nullptr && warp == 7) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int rc = r0 + r;
            if (rc < rows) {
                const float4 p = *(const float4*)(pre_in + (size_t)rc * NS);
                const float4 x0 = sv[r][0][lane];
                const float4 x1 = sv[r][1][lane];
                const float4 x2 = sv[r][2][lane];
                const float4 x3 = sv[r][3][lane];
                float4 out;
                out.x = p.x * x0.x + p.y * x1.x + p.z * x2.x + p.w * x3.x;
                out.y = p.x * x0.y + p.y * x1.y + p.z * x2.y + p.w * x3.y;
                out.z = p.x * x0.z + p.y * x1.z + p.z * x2.z + p.w * x3.z;
                out.w = p.x * x0.w + p.y * x1.w + p.z * x2.w + p.w * x3.w;
                *(float4*)(collapsed_out + (size_t)rc * HD + d0) = out;
            }
        }
        return;
    }
    if (warp < 6) {
        float acc[4][ROWS];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int r = 0; r < ROWS; ++r) acc[i][r] = 0.f;
        const int q0 = warp * 4;
#pragma unroll
        for (int h = 0; h < NS; ++h) {
            const size_t h_off = (size_t)h * HD + d0;
            const float4 f0 = __ldg(reinterpret_cast<const float4*>(fn + (size_t)(q0 + 0) * ROW + h_off));
            const float4 f1 = __ldg(reinterpret_cast<const float4*>(fn + (size_t)(q0 + 1) * ROW + h_off));
            const float4 f2 = __ldg(reinterpret_cast<const float4*>(fn + (size_t)(q0 + 2) * ROW + h_off));
            const float4 f3 = __ldg(reinterpret_cast<const float4*>(fn + (size_t)(q0 + 3) * ROW + h_off));
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const float4 vr = sv[r][h][lane];
                acc[0][r] = fmaf(f0.x, vr.x, acc[0][r]);
                acc[0][r] = fmaf(f0.y, vr.y, acc[0][r]);
                acc[0][r] = fmaf(f0.z, vr.z, acc[0][r]);
                acc[0][r] = fmaf(f0.w, vr.w, acc[0][r]);
                acc[1][r] = fmaf(f1.x, vr.x, acc[1][r]);
                acc[1][r] = fmaf(f1.y, vr.y, acc[1][r]);
                acc[1][r] = fmaf(f1.z, vr.z, acc[1][r]);
                acc[1][r] = fmaf(f1.w, vr.w, acc[1][r]);
                acc[2][r] = fmaf(f2.x, vr.x, acc[2][r]);
                acc[2][r] = fmaf(f2.y, vr.y, acc[2][r]);
                acc[2][r] = fmaf(f2.z, vr.z, acc[2][r]);
                acc[2][r] = fmaf(f2.w, vr.w, acc[2][r]);
                acc[3][r] = fmaf(f3.x, vr.x, acc[3][r]);
                acc[3][r] = fmaf(f3.y, vr.y, acc[3][r]);
                acc[3][r] = fmaf(f3.z, vr.z, acc[3][r]);
                acc[3][r] = fmaf(f3.w, vr.w, acc[3][r]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = q0 + i;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int rc = r0 + r;
                if (rc < rows) {
                    const float s = warp_sum(acc[i][r]);
                    if (lane == 0) part[((size_t)rc * NQ + q) * NATOM + a] = s;
                }
            }
        }
    } else if (warp == 6) {
        float acc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
#pragma unroll
            for (int h = 0; h < NS; ++h) {
                const float4 vr = sv[r][h][lane];
                acc[r] = fmaf(vr.x, vr.x, acc[r]);
                acc[r] = fmaf(vr.y, vr.y, acc[r]);
                acc[r] = fmaf(vr.z, vr.z, acc[r]);
                acc[r] = fmaf(vr.w, vr.w, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int rc = r0 + r;
            if (rc < rows) {
                const float s = warp_sum(acc[r]);
                if (lane == 0) part[((size_t)rc * NQ + NM) * NATOM + a] = s;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ 4. hc finalize

// One warp (32 threads) per row: 25 lanes reduce the 40 atom partials, then 16 lanes run the 4x4 Sinkhorn in parallel.
__global__ void __launch_bounds__(32) hc_finalize_k(const float* __restrict__ x,
                                                    const float* __restrict__ part,
                                                    const float* __restrict__ base,
                                                    const float* __restrict__ scale,
                                                    float* __restrict__ pre_out,
                                                    float* __restrict__ post_g,
                                                    float* __restrict__ comb_g,
                                                    int rows,
                                                    float rms_eps,
                                                    float hc_eps,
                                                    int iters) {
    (void)x;
    __shared__ float S[NQ];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;
    if (tid < NQ) {
        const float* p = part + ((size_t)r * NQ + tid) * NATOM;
        float s = 0.f;
#pragma unroll
        for (int a = 0; a < NATOM; a += 4) {
            const float4 v = __ldg(reinterpret_cast<const float4*>(p + a));
            s = (((s + v.x) + v.y) + v.z) + v.w;
        }
        S[tid] = s;
    }
    __syncwarp();
    const float rmr = 1.0f / sqrtf(S[NM] / (float)ROW + rms_eps);
    if (tid < NS) {
        pre_out[(size_t)r * NS + tid] = sigmoid_f(fmaf(S[tid] * rmr, scale[0], base[tid])) + hc_eps;
        post_g[(size_t)r * NS + tid] = 2.0f * sigmoid_f(fmaf(S[tid + NS] * rmr, scale[1], base[tid + NS]));
    }
    if (tid < NS * NS) {
        const int j = tid >> 2;
        const int k = tid & 3;
        float cv = fmaf(S[NS * 2 + tid] * rmr, scale[2], base[NS * 2 + tid]);
        const float c0 = __shfl_sync(0xffffu, cv, j * 4 + 0);
        const float c1 = __shfl_sync(0xffffu, cv, j * 4 + 1);
        const float c2 = __shfl_sync(0xffffu, cv, j * 4 + 2);
        const float c3 = __shfl_sync(0xffffu, cv, j * 4 + 3);
        const float mx = fmaxf(fmaxf(fmaxf(c0, c1), c2), c3);
        cv = expf(cv - mx);
        const float e0 = __shfl_sync(0xffffu, cv, j * 4 + 0);
        const float e1 = __shfl_sync(0xffffu, cv, j * 4 + 1);
        const float e2 = __shfl_sync(0xffffu, cv, j * 4 + 2);
        const float e3 = __shfl_sync(0xffffu, cv, j * 4 + 3);
        const float rsum = (((0.f + e0) + e1) + e2) + e3;
        cv = cv / rsum + hc_eps;

        // one column normalisation, then iters - 1 rounds of (row, column)
        {
            const float k0 = __shfl_sync(0xffffu, cv, 0 * 4 + k);
            const float k1 = __shfl_sync(0xffffu, cv, 1 * 4 + k);
            const float k2 = __shfl_sync(0xffffu, cv, 2 * 4 + k);
            const float k3 = __shfl_sync(0xffffu, cv, 3 * 4 + k);
            const float csum = (((0.f + k0) + k1) + k2) + k3;
            cv = cv / (csum + hc_eps);
        }
        for (int it = 1; it < iters; ++it) {
            const float r0 = __shfl_sync(0xffffu, cv, j * 4 + 0);
            const float r1 = __shfl_sync(0xffffu, cv, j * 4 + 1);
            const float r2 = __shfl_sync(0xffffu, cv, j * 4 + 2);
            const float r3 = __shfl_sync(0xffffu, cv, j * 4 + 3);
            const float rs = (((0.f + r0) + r1) + r2) + r3;
            cv = cv / (rs + hc_eps);

            const float k0 = __shfl_sync(0xffffu, cv, 0 * 4 + k);
            const float k1 = __shfl_sync(0xffffu, cv, 1 * 4 + k);
            const float k2 = __shfl_sync(0xffffu, cv, 2 * 4 + k);
            const float k3 = __shfl_sync(0xffffu, cv, 3 * 4 + k);
            const float cs = (((0.f + k0) + k1) + k2) + k3;
            cv = cv / (cs + hc_eps);
        }
        comb_g[(size_t)r * NS * NS + tid] = cv;
    }
}

// ------------------------------------------------------------------------------------------------ 5. hc apply

// x[h][d] <- post[h] * y[d] + sum_g comb[g][h] * x[g][d], in place. One float4 of d per thread.
__global__ void hc_apply_k(float* x, const void* y, int yt, const float* post_g, const float* comb_g, int rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int j = blockIdx.y * APL + threadIdx.x;  // float4 index inside a stream
    if (j >= VEC) return;
    const int d0 = j * 4;
    float* xr = x + (size_t)r * ROW;
    float4 xv[NS];
#pragma unroll
    for (int h = 0; h < NS; ++h) xv[h] = ld4f(xr, (size_t)h * HD + d0);
    float4 yv;
    const size_t yi = (size_t)r * HD + d0;
    if (yt == T_F32) yv = *(const float4*)((const float*)y + yi);
    else if (yt == T_F16) {
        const uint2 u = *(const uint2*)((const __half*)y + yi);
        const float2 a = __half22float2(*(const __half2*)&u.x), b = __half22float2(*(const __half2*)&u.y);
        yv = make_float4(a.x, a.y, b.x, b.y);
    } else {
        const uint2 u = *(const uint2*)((const u16*)y + yi);
        yv = make_float4(bf2f(u.x & 0xFFFFu), bf2f(u.x >> 16), bf2f(u.y & 0xFFFFu), bf2f(u.y >> 16));
    }
    float p[NS], c[NS * NS];
#pragma unroll
    for (int h = 0; h < NS; ++h) p[h] = post_g[(size_t)r * NS + h];
#pragma unroll
    for (int k = 0; k < NS * NS; ++k) c[k] = comb_g[(size_t)r * NS * NS + k];
    float4 nx[NS];
#pragma unroll
    for (int h = 0; h < NS; ++h) {
        float4 o;
        o.x = p[h] * yv.x;
        o.y = p[h] * yv.y;
        o.z = p[h] * yv.z;
        o.w = p[h] * yv.w;
#pragma unroll
        for (int g = 0; g < NS; ++g) {
            o.x = fmaf(c[g * NS + h], xv[g].x, o.x);
            o.y = fmaf(c[g * NS + h], xv[g].y, o.y);
            o.z = fmaf(c[g * NS + h], xv[g].z, o.z);
            o.w = fmaf(c[g * NS + h], xv[g].w, o.w);
        }
        nx[h] = o;
    }
#pragma unroll
    for (int h = 0; h < NS; ++h) *(float4*)(xr + (size_t)h * HD + d0) = nx[h];
}

// ------------------------------------------------------------------------------------------------ 6. fused SwiGLU

__global__ void sqrtsoftplus_k(const float* x, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sqrtsoftplus_f(x[i]);
}

// silu(min(gate, limit)) * clamp(up, -limit, limit), times the router weight of the token when `w` is given.
__global__ void swiglu_k(const float* g, const float* u, const float* w, float* out, int n, int inter, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (limit > 0.f) {
        uv = fminf(fmaxf(uv, -limit), limit);
        gv = fminf(gv, limit);
    }
    float v = (gv / (1.0f + expf(-gv))) * uv;
    if (w != nullptr) v *= w[i / inter];
    out[i] = v;
}

// The same arithmetic on the MoE block's native dtype: gate, up, out and the per-row router weight are fp16, the
// activation is computed in fp32 and rounded once on the way out. `w` is `[rows]` fp16 (the router's weights, packed).
__global__ void swiglu_f16_k(const __half* g, const __half* u, const __half* w, __half* out, int n, int inter, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = __half2float(g[i]), uv = __half2float(u[i]);
    if (limit > 0.f) {
        uv = fminf(fmaxf(uv, -limit), limit);
        gv = fminf(gv, limit);
    }
    float v = (gv / (1.0f + expf(-gv))) * uv;
    if (w != nullptr) v *= __half2float(w[i / inter]);
    out[i] = __float2half(v);
}

}  // namespace

namespace {

static inline bool al(const void* p, int n) { return ((uintptr_t)p & (uintptr_t)(n - 1)) == 0; }
static inline int tal(int t) { return t == T_F32 ? 16 : 8; }  // alignment of a 4-wide access of type t
static inline bool ok_type(int t) { return t == T_F32 || t == T_F16 || t == T_BF16; }
static inline bool ok_rows(int rows) { return rows >= 0 && rows <= 65535; }

}  // namespace

template <int XT>
static inline void launch_router_logits(const void* x, const u16* w, float* logits, int rows, cudaStream_t st) {
    if (rows > 16 && al(x, 16) && al(w, 16)) {
        const unsigned gx = (unsigned)((rows + 31) / 32);
        const unsigned gy = (unsigned)(E / 64);
        router_logits_mma_k<XT><<<dim3(gx, gy), 128, 0, st>>>(x, w, logits, rows);
        return;
    }
    const unsigned gy = (unsigned)(E / 4);
    switch (rows) {
        case 1: router_logits_k<XT, 1><<<dim3(1, gy), 128, 0, st>>>(x, w, logits, rows); break;
        case 2: router_logits_k<XT, 2><<<dim3(1, gy), 128, 0, st>>>(x, w, logits, rows); break;
        case 3: router_logits_k<XT, 3><<<dim3(1, gy), 128, 0, st>>>(x, w, logits, rows); break;
        case 4: router_logits_k<XT, 4><<<dim3(1, gy), 128, 0, st>>>(x, w, logits, rows); break;
        case 5: router_logits_k<XT, 5><<<dim3(1, gy), 128, 0, st>>>(x, w, logits, rows); break;
        case 6: router_logits_k<XT, 6><<<dim3(1, gy), 128, 0, st>>>(x, w, logits, rows); break;
        case 7:
        case 8: router_logits_k<XT, 8><<<dim3(1, gy), 128, 0, st>>>(x, w, logits, rows); break;
        default: {
            const unsigned gx = (unsigned)((rows + 15) / 16);
            router_logits_k<XT, 16><<<dim3(gx, gy), 128, 0, st>>>(x, w, logits, rows);
            break;
        }
    }
}

// logits [rows][384] fp32 = x [rows][5120] (f16 / bf16 / f32) times the bf16 gate [384][5120] transposed.
GB_EXPORT int gb_dsv4_router_logits(const void* x, int x_type, const void* w, float* logits, int rows, void* stream) {
    if (!ok_rows(rows) || !ok_type(x_type)) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!al(x, tal(x_type)) || !al(w, 2) || !al(logits, 16)) return ERR_INVALID;
    cudaStream_t st = gb_stream(stream);
    switch (x_type) {
        case T_F32: launch_router_logits<T_F32>(x, (const u16*)w, logits, rows, st); break;
        case T_F16: launch_router_logits<T_F16>(x, (const u16*)w, logits, rows, st); break;
        default: launch_router_logits<T_BF16>(x, (const u16*)w, logits, rows, st); break;
    }
    return GB_CHECK_LAUNCH();
}

// picks [rows][6] i32 and weights [rows][6] fp16 from logits (fp32, or fp16 when `tab` is given and f16_log != 0).
// `bias_vl` may be null (then `mask` is never read and every row uses `bias`).
GB_EXPORT int gb_dsv4_router_select(const void* logits, const float* tab, const float* bias, const float* bias_vl,
                                    const unsigned char* mask, int* pick, unsigned short* wts, int rows, int f16_log,
                                    float scale, float top_p, void* stream) {
    if (!ok_rows(rows)) return ERR_INVALID;
    if (logits == nullptr || bias == nullptr || pick == nullptr || wts == nullptr) return ERR_INVALID;
    if (f16_log ? (tab == nullptr) : !al(logits, 16)) return ERR_INVALID;
    if (!al(pick, 16) || !al(wts, 8) || !al(bias, 16)) return ERR_INVALID;
    if (rows == 0) return 0;
    if (f16_log) router_select_k<true><<<(unsigned)rows, 128, 0, gb_stream(stream)>>>(logits, tab, bias, bias_vl, mask, pick, (u16*)wts, rows, scale, top_p);
    else router_select_k<false><<<(unsigned)rows, 128, 0, gb_stream(stream)>>>(logits, tab, bias, bias_vl, mask, pick, (u16*)wts, rows, scale, top_p);
    return GB_CHECK_LAUNCH();
}

// 25 atom partials per row, part [rows][25][40] fp32, optionally also collapsing streams into collapsed_out [rows][5120] fp32.
GB_EXPORT int gb_dsv4_hc_apply_partials_and_collapse(float* x, const float* fn, float* part, const float* pre_in,
                                                     float* collapsed_out, const void* y_prev, const float* post_prev,
                                                     const float* comb_prev, int rows, void* stream) {
    if (!ok_rows(rows)) return ERR_INVALID;
    if (!al(x, 16) || !al(fn, 16) || !al(part, 16)) return ERR_INVALID;
    if ((pre_in != nullptr && !al(pre_in, 16)) || (collapsed_out != nullptr && !al(collapsed_out, 16))) return ERR_INVALID;
    if (y_prev != nullptr && (!al(y_prev, 8) || !al(post_prev, 16) || !al(comb_prev, 16))) return ERR_INVALID;
    if (rows == 0) return 0;
    cudaStream_t st = gb_stream(stream);
    switch (rows) {
        case 1: hc_partials_k<1><<<dim3(NATOM, 1), PW * 32, 0, st>>>(x, fn, part, pre_in, collapsed_out, y_prev, post_prev, comb_prev, rows); break;
        case 2: hc_partials_k<2><<<dim3(NATOM, 1), PW * 32, 0, st>>>(x, fn, part, pre_in, collapsed_out, y_prev, post_prev, comb_prev, rows); break;
        case 3: hc_partials_k<3><<<dim3(NATOM, 1), PW * 32, 0, st>>>(x, fn, part, pre_in, collapsed_out, y_prev, post_prev, comb_prev, rows); break;
        case 4: hc_partials_k<4><<<dim3(NATOM, 1), PW * 32, 0, st>>>(x, fn, part, pre_in, collapsed_out, y_prev, post_prev, comb_prev, rows); break;
        case 5: hc_partials_k<5><<<dim3(NATOM, 1), PW * 32, 0, st>>>(x, fn, part, pre_in, collapsed_out, y_prev, post_prev, comb_prev, rows); break;
        case 6: hc_partials_k<6><<<dim3(NATOM, 1), PW * 32, 0, st>>>(x, fn, part, pre_in, collapsed_out, y_prev, post_prev, comb_prev, rows); break;
        default: {
            const unsigned gy = (unsigned)((rows + 7) / 8);
            hc_partials_k<8><<<dim3(NATOM, gy), PW * 32, 0, st>>>(x, fn, part, pre_in, collapsed_out, y_prev, post_prev, comb_prev, rows);
            break;
        }
    }
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_hc_partials_and_collapse(const float* x, const float* fn, float* part, const float* pre_in,
                                               float* collapsed_out, int rows, void* stream) {
    return gb_dsv4_hc_apply_partials_and_collapse(const_cast<float*>(x), fn, part, pre_in, collapsed_out, nullptr, nullptr, nullptr, rows, stream);
}

GB_EXPORT int gb_dsv4_hc_partials(const float* x, const float* fn, float* part, int rows, void* stream) {
    return gb_dsv4_hc_partials_and_collapse(x, fn, part, nullptr, nullptr, rows, stream);
}

// pre [rows][4], post [rows][4], comb [rows][4][4] (comb[g][h]).
GB_EXPORT int gb_dsv4_hc_finalize(const float* x, const float* part, const float* base, const float* scale,
                                  float* pre_out, float* post, float* comb, int rows, float rms_eps, float hc_eps,
                                  int iters, void* stream) {
    if (!ok_rows(rows) || iters < 1 || iters > 256) return ERR_INVALID;
    if (!al(x, 16) || !al(part, 16) || !al(base, 16) || !al(scale, 4)) return ERR_INVALID;
    if (!al(pre_out, 16) || !al(post, 16) || !al(comb, 16)) return ERR_INVALID;
    if (rows == 0) return 0;
    hc_finalize_k<<<(unsigned)rows, 32, 0, gb_stream(stream)>>>(x, part, base, scale, pre_out, post, comb, rows, rms_eps,
                                                                 hc_eps, iters);
    return GB_CHECK_LAUNCH();
}

// x[h][d] <- post[h] y[d] + sum_g comb[g][h] x[g][d], in place.
GB_EXPORT int gb_dsv4_hc_apply(float* x, const void* y, int y_type, const float* post, const float* comb, int rows,
                               void* stream) {
    if (!ok_rows(rows) || !ok_type(y_type)) return ERR_INVALID;
    if (!al(x, 16) || !al(y, tal(y_type)) || !al(post, 16) || !al(comb, 16)) return ERR_INVALID;
    if (rows == 0) return 0;
    const dim3 g((unsigned)rows, (unsigned)((VEC + APL - 1) / APL));
#define L(YT) hc_apply_k<<<g, APL, 0, gb_stream(stream)>>>(x, y, YT, post, comb, rows)
    switch (y_type) {
        case T_F32: L(T_F32); break;
        case T_F16: L(T_F16); break;
        default: L(T_BF16); break;
    }
#undef L
    return GB_CHECK_LAUNCH();
}

// out [rows][inter] fp32 = silu(min(gate, limit)) * clamp(up, -limit, limit) * (w ? w[row] : 1).
GB_EXPORT int gb_dsv4_swiglu(const float* g, const float* u, const float* w, float* out, int n, int inter, float limit,
                             void* stream) {
    if (n < 0 || inter <= 0) return ERR_INVALID;
    if (!al(g, 16) || !al(u, 16) || !al(out, 16) || (w != nullptr && !al(w, 4))) return ERR_INVALID;
    if (n == 0) return 0;
    const int threads = 256, blocks = (n + threads - 1) / threads;
    swiglu_k<<<blocks, threads, 0, gb_stream(stream)>>>(g, u, w, out, n, inter, limit);
    return GB_CHECK_LAUNCH();
}

// out [rows][inter] f16 = silu(min(gate, limit)) * clamp(up, -limit, limit) * (w ? w[row] : 1), every operand fp16.
GB_EXPORT int gb_dsv4_swiglu_f16(const void* g, const void* u, const void* w, void* out, int n, int inter, float limit,
                                 void* stream) {
    if (n < 0 || inter <= 0) return ERR_INVALID;
    if (!al(g, 16) || !al(u, 16) || !al(out, 16) || (w != nullptr && !al(w, 16))) return ERR_INVALID;
    if (n == 0) return 0;
    const int threads = 256, blocks = (n + threads - 1) / threads;
    swiglu_f16_k<<<blocks, threads, 0, gb_stream(stream)>>>((const __half*)g, (const __half*)u, (const __half*)w,
                                                            (__half*)out, n, inter, limit);
    return GB_CHECK_LAUNCH();
}

__global__ void engram_gate_add_k(float* __restrict__ x, const __half* __restrict__ kv,
                                  const u16* __restrict__ qw, const u16* __restrict__ kw,
                                  int rows, int hc, int dim, float eps) {
    const int r = blockIdx.x;
    const int c = blockIdx.y;
    if (r >= rows || c >= hc) return;
    float* xr = x + ((size_t)r * hc + c) * dim;
    const __half* kr = kv + ((size_t)r * (hc + 1) + c) * dim;
    const __half* vr = kv + ((size_t)r * (hc + 1) + hc) * dim;
    const u16* qr = qw + (size_t)c * dim;
    const u16* kwr = kw + (size_t)c * dim;

    float x_sq = 0.f, k_sq = 0.f, dot = 0.f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float xv = xr[d];
        const float kv_f = __half2float(kr[d]);
        const float w = bf2f(qr[d]) * bf2f(kwr[d]);
        x_sq = fmaf(xv, xv, x_sq);
        k_sq = fmaf(kv_f, kv_f, k_sq);
        dot = fmaf(xv * w, kv_f, dot);
    }
    x_sq = warp_sum(x_sq);
    k_sq = warp_sum(k_sq);
    dot = warp_sum(dot);
    __shared__ float sx[8], sk[8], sd[8], sgate;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        sx[warp] = x_sq;
        sk[warp] = k_sq;
        sd[warp] = dot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tx = 0.f, tk = 0.f, td = 0.f;
        const int nw = blockDim.x >> 5;
        for (int w = 0; w < nw; ++w) {
            tx += sx[w];
            tk += sk[w];
            td += sd[w];
        }
        const float fdim = (float)dim;
        const float rstd = (1.0f / sqrtf(tx / fdim + eps)) * (1.0f / sqrtf(tk / fdim + eps));
        const float scaled = td * rstd / sqrtf(fdim);
        const float mag = sqrtf(fmaxf(fabsf(scaled), 1e-6f));
        sgate = 1.0f / (1.0f + expf(-copysignf(mag, scaled)));
    }
    __syncthreads();
    const float gate = sgate;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        xr[d] += gate * __half2float(vr[d]);
    }
}

// y = sqrtsoftplus(x) elementwise.
GB_EXPORT int gb_dsv4_sqrtsoftplus(const float* x, float* y, int n, void* stream) {
    if (n < 0) return ERR_INVALID;
    if (!al(x, 16) || !al(y, 16)) return ERR_INVALID;
    if (n == 0) return 0;
    const int threads = 256, blocks = (n + threads - 1) / threads;
    sqrtsoftplus_k<<<blocks, threads, 0, gb_stream(stream)>>>(x, y, n);
    return GB_CHECK_LAUNCH();
}

__global__ void hc_collapse_f32_k(const float* __restrict__ x,
                                  const float* __restrict__ pre,
                                  float* __restrict__ y,
                                  int rows,
                                  int dim) {
    const int r = blockIdx.y;
    const int d0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (r >= rows || d0 >= dim) return;
    const float4 p = *(const float4*)(pre + (size_t)r * 4);
    const float* xr = x + (size_t)r * 4 * dim;
    const float4 x0 = *(const float4*)(xr + 0 * dim + d0);
    const float4 x1 = *(const float4*)(xr + 1 * dim + d0);
    const float4 x2 = *(const float4*)(xr + 2 * dim + d0);
    const float4 x3 = *(const float4*)(xr + 3 * dim + d0);
    float4 out;
    out.x = p.x * x0.x + p.y * x1.x + p.z * x2.x + p.w * x3.x;
    out.y = p.x * x0.y + p.y * x1.y + p.z * x2.y + p.w * x3.y;
    out.z = p.x * x0.z + p.y * x1.z + p.z * x2.z + p.w * x3.z;
    out.w = p.x * x0.w + p.y * x1.w + p.z * x2.w + p.w * x3.w;
    *(float4*)(y + (size_t)r * dim + d0) = out;
}

GB_EXPORT int gb_dsv4_hc_collapse_f32(const float* x, const float* pre, float* y, int rows, int dim, void* stream) {
    if (!x || !pre || !y || rows <= 0 || dim <= 0 || (dim & 3) != 0) return ERR_INVALID;
    const int vec = dim / 4;
    const dim3 grid((vec + 255) / 256, rows);
    hc_collapse_f32_k<<<grid, 256, 0, gb_stream(stream)>>>(x, pre, y, rows, dim);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_dsv4_engram_gate_add(float* x, const void* kv, const void* qw, const void* kw,
                                      int rows, int hc, int dim, float eps, void* stream) {
    if (!ok_rows(rows) || hc <= 0 || dim <= 0) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!al(x, 16) || !al(kv, 8) || !al(qw, 2) || !al(kw, 2)) return ERR_INVALID;
    const dim3 g((unsigned)rows, (unsigned)hc);
    engram_gate_add_k<<<g, 256, 0, gb_stream(stream)>>>(x, (const __half*)kv, (const u16*)qw, (const u16*)kw,
                                                        rows, hc, dim, eps);
    return GB_CHECK_LAUNCH();
}
