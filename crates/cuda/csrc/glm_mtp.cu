// gb-nvcc: --fmad=false
// GLM-5.3-Flash NextN (MTP) layer glue: the four small kernels the plain-residual MTP block needs that the trunk's mHC path does not have, and (5) the
// keyed pick that turns the head's candidates into a draft.
//
//   1. mtp_input:        out[r] = [ f16((e * inv_e) * we) | f16((h * inv_h) * wh) ] (fp16 [rows, 2 * dim], embedding side first), where
//                        e = f16(bf16 embedding row of ids[r]) (an id outside 0..vocab gives a zero row), h = the tapped hidden row
//                        (fp16 [rows, dim]), inv_x = 1 / sqrtf(mean(x^2) + eps). This is the fork's `enorm`, `hnorm` and the concat that
//                        `eh_proj` reads: the embedding is converted to fp16 first (the fork's `Embedding.forward(out_dtype = half)`), then
//                        normed, then the two halves are joined.
//   1b. mtp_embed_f32:   the embedding half of `mtp_input` for rows whose embedding is a vision feature row, not a table row: cat[r][0..dim] =
//                        f16((e * inv_e) * we) with e = f16(feature row) (fp32 [rows, dim]; the fork's `Embedding.forward` writes the image
//                        embeddings `.to(half)` over the table rows of the image placeholder ids). The hnorm half of the row is left as it is.
//   2. rmsnorm_f32_f16:  y = f16((x * inv) * w), x fp32 [rows, dim], y fp16: the norm before the MTP block's attention and the
//                        `shared_head.norm` before the head (the residual of this layer is fp32, plain: no mHC).
//   3. add_rmsnorm_f32:  x += y (fp32, in place), then y_out = rmsnorm(x) in fp16: the residual add of the attention output fused with
//                        the norm in front of the feed-forward block, as the fork's `rms_norm_res_in` does.
//   4. add_f32:          x += y (fp32, in place): the residual add of the feed-forward output.
//
// Compiled --fmad=false, no atomics on data, and every sum of squares in ONE fixed order that is a function of `dim` alone (thread t owns
// the 8-element vectors t, t + 256, ...; it adds f[0]^2 .. f[7]^2 of each in turn; then the fixed xor-butterfly across each warp and the
// warps 0 .. W-1 added in order: exactly `rmsnorm_f16_k` of glm_glue.cu). A row's output bits depend on that row alone, at any row count,
// and a host loop in the same order reproduces them.
#include <math.h>

#include "common.cuh"
#include "keyed.cuh"

namespace {

typedef uint16_t u16;
typedef uint32_t u32;

constexpr int ERR_INVALID = 1;  // cudaErrorInvalidValue

__device__ __forceinline__ float h2f(u16 u) { return __half2float(__ushort_as_half(u)); }
__device__ __forceinline__ u16 f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }
__device__ __forceinline__ float bf2f(u16 u) { return __uint_as_float(((u32)u) << 16); }

__device__ __forceinline__ void unpack8h(const uint4& u, float (&f)[8]) {
    f[0] = h2f((u16)(u.x & 0xFFFF)); f[1] = h2f((u16)(u.x >> 16));
    f[2] = h2f((u16)(u.y & 0xFFFF)); f[3] = h2f((u16)(u.y >> 16));
    f[4] = h2f((u16)(u.z & 0xFFFF)); f[5] = h2f((u16)(u.z >> 16));
    f[6] = h2f((u16)(u.w & 0xFFFF)); f[7] = h2f((u16)(u.w >> 16));
}
// the same eight values from a bf16 row, rounded to fp16 (the embedding table is bf16, the fork widens it through fp16)
__device__ __forceinline__ void unpack8bf_as_h(const uint4& u, float (&f)[8]) {
    f[0] = h2f(f2h(bf2f((u16)(u.x & 0xFFFF)))); f[1] = h2f(f2h(bf2f((u16)(u.x >> 16))));
    f[2] = h2f(f2h(bf2f((u16)(u.y & 0xFFFF)))); f[3] = h2f(f2h(bf2f((u16)(u.y >> 16))));
    f[4] = h2f(f2h(bf2f((u16)(u.z & 0xFFFF)))); f[5] = h2f(f2h(bf2f((u16)(u.z >> 16))));
    f[6] = h2f(f2h(bf2f((u16)(u.w & 0xFFFF)))); f[7] = h2f(f2h(bf2f((u16)(u.w >> 16))));
}
__device__ __forceinline__ u32 pack2h(float a, float b) { return (u32)f2h(a) | ((u32)f2h(b) << 16); }
__device__ __forceinline__ uint4 pack8h(const float (&f)[8]) {
    uint4 u;
    u.x = pack2h(f[0], f[1]); u.y = pack2h(f[2], f[3]); u.z = pack2h(f[4], f[5]); u.w = pack2h(f[6], f[7]);
    return u;
}

static inline __host__ __device__ bool aligned16(const void* p) { return ((uintptr_t)p & 15) == 0; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

template <int NT>
__device__ __forceinline__ float block_sum(float v, float* sm) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    float s = sm[0];
#pragma unroll
    for (int w = 1; w < NT / 32; ++w) s = s + sm[w];
    __syncthreads();
    return s;
}

constexpr int NT = 256;

// ---------------------------------------------------------------------------------------------------------------- mtp_input

__global__ void __launch_bounds__(NT) mtp_input_k(const u16* table, const int* ids, const u16* hid, const float* we, const float* wh, u16* out,
                                                  int dim, int vocab, float eps) {
    __shared__ float sm[NT / 32];
    const int r = blockIdx.x;
    const int id = ids[r];
    const bool ok = id >= 0 && id < vocab;
    const u16* erow = table + (size_t)(ok ? id : 0) * dim;
    const u16* hrow = hid + (size_t)r * dim;
    u16* orow = out + (size_t)r * 2 * dim;
    const int nvec = dim / 8;
    float ssE = 0.f, ssH = 0.f;
    for (int v = threadIdx.x; v < nvec; v += NT) {
        float f[8], g[8];
        if (ok) {
            unpack8bf_as_h(*(const uint4*)(erow + (size_t)v * 8), f);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = 0.f;
        }
        unpack8h(*(const uint4*)(hrow + (size_t)v * 8), g);
#pragma unroll
        for (int i = 0; i < 8; ++i) { ssE = ssE + f[i] * f[i]; }
#pragma unroll
        for (int i = 0; i < 8; ++i) { ssH = ssH + g[i] * g[i]; }
    }
    ssE = block_sum<NT>(ssE, sm);
    ssH = block_sum<NT>(ssH, sm);
    const float invE = 1.0f / sqrtf(ssE / (float)dim + eps);
    const float invH = 1.0f / sqrtf(ssH / (float)dim + eps);
    for (int v = threadIdx.x; v < nvec; v += NT) {
        float f[8], g[8], oe[8], oh[8];
        if (ok) {
            unpack8bf_as_h(*(const uint4*)(erow + (size_t)v * 8), f);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = 0.f;
        }
        unpack8h(*(const uint4*)(hrow + (size_t)v * 8), g);
        const float4 a0 = *(const float4*)(we + (size_t)v * 8), a1 = *(const float4*)(we + (size_t)v * 8 + 4);
        const float4 b0 = *(const float4*)(wh + (size_t)v * 8), b1 = *(const float4*)(wh + (size_t)v * 8 + 4);
        const float wev[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float whv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { oe[i] = (f[i] * invE) * wev[i]; oh[i] = (g[i] * invH) * whv[i]; }
        *(uint4*)(orow + (size_t)v * 8) = pack8h(oe);
        *(uint4*)(orow + (size_t)dim + (size_t)v * 8) = pack8h(oh);
    }
}

// out left half only: see 1b. The sum of squares is the one of mtp_input_k (the same vectors, the same order), so a text row and an image row
// of the same values normalise to the same bits.
__global__ void __launch_bounds__(NT) mtp_embed_f32_k(const float* feat, const float* we, u16* out, int dim, float eps) {
    __shared__ float sm[NT / 32];
    const int r = blockIdx.x;
    const float* frow = feat + (size_t)r * dim;
    u16* orow = out + (size_t)r * 2 * dim;
    const int nvec = dim / 8;
    float ss = 0.f;
    for (int v = threadIdx.x; v < nvec; v += NT) {
        const float4 a = *(const float4*)(frow + (size_t)v * 8), b = *(const float4*)(frow + (size_t)v * 8 + 4);
        const float f[8] = {h2f(f2h(a.x)), h2f(f2h(a.y)), h2f(f2h(a.z)), h2f(f2h(a.w)), h2f(f2h(b.x)), h2f(f2h(b.y)), h2f(f2h(b.z)), h2f(f2h(b.w))};
#pragma unroll
        for (int i = 0; i < 8; ++i) { ss = ss + f[i] * f[i]; }
    }
    ss = block_sum<NT>(ss, sm);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int v = threadIdx.x; v < nvec; v += NT) {
        const float4 a = *(const float4*)(frow + (size_t)v * 8), b = *(const float4*)(frow + (size_t)v * 8 + 4);
        const float f[8] = {h2f(f2h(a.x)), h2f(f2h(a.y)), h2f(f2h(a.z)), h2f(f2h(a.w)), h2f(f2h(b.x)), h2f(f2h(b.y)), h2f(f2h(b.z)), h2f(f2h(b.w))};
        const float4 w0 = *(const float4*)(we + (size_t)v * 8), w1 = *(const float4*)(we + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[i] = (f[i] * inv) * wv[i]; }
        *(uint4*)(orow + (size_t)v * 8) = pack8h(o);
    }
}

// ------------------------------------------------------------------------------------------------------------- fp32-input norms

__device__ __forceinline__ void load8f(const float* p, float (&f)[8]) {
    const float4 a = *(const float4*)p, b = *(const float4*)(p + 4);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
__device__ __forceinline__ void store8f(float* p, const float (&f)[8]) {
    *(float4*)p = make_float4(f[0], f[1], f[2], f[3]);
    *(float4*)(p + 4) = make_float4(f[4], f[5], f[6], f[7]);
}

__global__ void __launch_bounds__(NT) rmsnorm_f32_f16_k(const float* x, const float* w, u16* y, int dim, float eps) {
    __shared__ float sm[NT / 32];
    const size_t base = (size_t)blockIdx.x * dim;
    const int nvec = dim / 8;
    float ss = 0.f;
    for (int v = threadIdx.x; v < nvec; v += NT) {
        float f[8];
        load8f(x + base + (size_t)v * 8, f);
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = ss + f[i] * f[i];
    }
    ss = block_sum<NT>(ss, sm);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int v = threadIdx.x; v < nvec; v += NT) {
        float f[8], o[8], wv[8];
        load8f(x + base + (size_t)v * 8, f);
        load8f(w + (size_t)v * 8, wv);
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = (f[i] * inv) * wv[i];
        *(uint4*)(y + base + (size_t)v * 8) = pack8h(o);
    }
}

// x += y (fp32, stored back), then y_out = f16((x * inv) * w) with the sum of squares of the NEW x. A thread re-reads only the vectors it wrote.
__global__ void __launch_bounds__(NT) add_rmsnorm_f32_k(float* x, const float* y, const float* w, u16* y_out, int dim, float eps) {
    __shared__ float sm[NT / 32];
    const size_t base = (size_t)blockIdx.x * dim;
    const int nvec = dim / 8;
    float ss = 0.f;
    for (int v = threadIdx.x; v < nvec; v += NT) {
        float a[8], b[8], s[8];
        load8f(x + base + (size_t)v * 8, a);
        load8f(y + base + (size_t)v * 8, b);
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i] = a[i] + b[i];
        store8f(x + base + (size_t)v * 8, s);
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = ss + s[i] * s[i];
    }
    ss = block_sum<NT>(ss, sm);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int v = threadIdx.x; v < nvec; v += NT) {
        float f[8], o[8], wv[8];
        load8f(x + base + (size_t)v * 8, f);
        load8f(w + (size_t)v * 8, wv);
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = (f[i] * inv) * wv[i];
        *(uint4*)(y_out + base + (size_t)v * 8) = pack8h(o);
    }
}

__global__ void add_f32_k(float* x, const float* y, long long n4) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
        float4 a = ((float4*)x)[i];
        const float4 b = ((const float4*)y)[i];
        a.x = a.x + b.x; a.y = a.y + b.y; a.z = a.z + b.z; a.w = a.w + b.w;
        ((float4*)x)[i] = a;
    }
}

// ---- 5. keyed_pick: the draft head's proposal as the TARGET's keyed draw over the head's own top candidates (docs/GLM53.md "Keyed-noise drafting").
//
// One row = one sequence at one chain depth. `ids` (i32) and `vals` (f32) are the row's `count` best candidates by (value DESC, id ASC) with the ids already
// token ids (a restricted head's columns mapped back: the noise is keyed by id, never by column). The row's sampling record says whether it is keyed
// (a sampled request: temperature > 0 and a top_k the list covers) or not (the head's argmax = the first candidate, as before); a keyed row takes the
// candidates the target's sampler keeps (the first `top_k`, then top_p, then min_p), adds Gumbel noise keyed on (seed, first_position + depth, id) to
// logit / T and picks the largest: `crates/sampling::try_choose` on the same list. One thread per row, a row reads only its own list and record.
struct KeyedRow {
    unsigned long long seed;
    unsigned long long first_position;  // the absolute position the row's depth-0 draft is compared at (len of the sequence)
    double temperature, top_p, min_p;
    int top_k;
    int keyed;
};

__global__ void keyed_pick_k(const int* __restrict__ ids, const float* __restrict__ vals, int count, const KeyedRow* __restrict__ rows, int depth,
                             int* __restrict__ out_ids, float* __restrict__ out_vals) {
    const int r = blockIdx.x;
    const int* c_row = ids + (size_t)r * count;
    const float* v_row = vals + (size_t)r * count;
    const KeyedRow p = rows[r];
    int pick = 0;
    if (p.keyed && p.temperature > 0.0) {
        int k = p.top_k > 0 && p.top_k < count ? p.top_k : count;
        pick = gb_keyed::keyed_choose(v_row, c_row, k, p.seed, p.temperature, p.top_p, p.min_p, p.first_position + (unsigned long long)depth);
    }
    out_ids[r] = c_row[pick];
    out_vals[r] = v_row[pick];
}

}  // namespace

// ==================================================================================================== launchers

// ids i32 / vals f32 [rows, count] (count <= 64, sorted by (value desc, id asc)), params KeyedRow [rows] (48 bytes each), out_ids i32 / out_vals f32 [rows].
// `depth` is the chain depth of this call (0 for the first draft): the noise of row r is drawn at params[r].first_position + depth.
GB_EXPORT int gb_glm_keyed_pick(const int* ids, const float* vals, int count, int rows, const void* params, int depth, int* out_ids, float* out_vals,
                                void* stream) {
    if (rows < 0 || count < 1 || count > 64 || depth < 0) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!ids || !vals || !params || !out_ids || !out_vals) return ERR_INVALID;
    keyed_pick_k<<<rows, 1, 0, gb_stream(stream)>>>(ids, vals, count, (const KeyedRow*)params, depth, out_ids, out_vals);
    return GB_CHECK_LAUNCH();
}

// out fp16 [rows, 2 * dim]: see the header. table bf16 [vocab, dim], ids i32 [rows], hid fp16 [rows, dim], we / wh fp32 [dim]. dim % 8 == 0.
GB_EXPORT int gb_glm_mtp_input(const void* table, const int* ids, const void* hid, const float* we, const float* wh, void* out, int rows, int dim,
                               int vocab, float eps, void* stream) {
    if (rows < 0 || dim < 8 || dim % 8 != 0 || vocab < 0) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!table || !ids || !hid || !we || !wh || !out) return ERR_INVALID;
    if (!aligned16(table) || !aligned16(hid) || !aligned16(we) || !aligned16(wh) || !aligned16(out)) return ERR_INVALID;
    mtp_input_k<<<rows, NT, 0, gb_stream(stream)>>>((const u16*)table, ids, (const u16*)hid, we, wh, (u16*)out, dim, vocab, eps);
    return GB_CHECK_LAUNCH();
}

// the left halves of `rows` rows of `out` (fp16 [rows, 2 * dim], as mtp_input writes them) from fp32 feature rows `feat` [rows, dim]; we fp32 [dim].
GB_EXPORT int gb_glm_mtp_embed_f32(const void* feat, const float* we, void* out, int rows, int dim, float eps, void* stream) {
    if (rows < 0 || dim < 8 || dim % 8 != 0) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!feat || !we || !out || !aligned16(feat) || !aligned16(we) || !aligned16(out)) return ERR_INVALID;
    mtp_embed_f32_k<<<rows, NT, 0, gb_stream(stream)>>>((const float*)feat, we, (u16*)out, dim, eps);
    return GB_CHECK_LAUNCH();
}

// y fp16 = f16((x * inv) * w), x fp32 [rows, dim], w fp32 [dim]. dim % 8 == 0.
GB_EXPORT int gb_glm_rmsnorm_f32_f16(const void* x, const void* w, void* y, int rows, int dim, float eps, void* stream) {
    if (rows < 0 || dim < 8 || dim % 8 != 0) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!x || !w || !y || !aligned16(x) || !aligned16(w) || !aligned16(y)) return ERR_INVALID;
    rmsnorm_f32_f16_k<<<rows, NT, 0, gb_stream(stream)>>>((const float*)x, (const float*)w, (u16*)y, dim, eps);
    return GB_CHECK_LAUNCH();
}

// x (fp32 [rows, dim]) += y (fp32), then y_out (fp16) = rmsnorm(x). x and y must not overlap; y_out is a different buffer.
GB_EXPORT int gb_glm_add_rmsnorm_f32(void* x, const void* y, const void* w, void* y_out, int rows, int dim, float eps, void* stream) {
    if (rows < 0 || dim < 8 || dim % 8 != 0) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!x || !y || !w || !y_out || !aligned16(x) || !aligned16(y) || !aligned16(w) || !aligned16(y_out)) return ERR_INVALID;
    if (x == y) return ERR_INVALID;
    add_rmsnorm_f32_k<<<rows, NT, 0, gb_stream(stream)>>>((float*)x, (const float*)y, (const float*)w, (u16*)y_out, dim, eps);
    return GB_CHECK_LAUNCH();
}

// x (fp32, n elements) += y (fp32). n % 4 == 0, 16-byte aligned.
GB_EXPORT int gb_glm_add_f32(void* x, const void* y, long long n, void* stream) {
    if (n < 0 || n % 4 != 0) return ERR_INVALID;
    if (n == 0) return 0;
    if (!x || !y || !aligned16(x) || !aligned16(y) || x == y) return ERR_INVALID;
    const long long n4 = n / 4;
    int blocks = (int)((n4 + 255) / 256);
    if (blocks > 1024) blocks = 1024;
    add_f32_k<<<blocks, 256, 0, gb_stream(stream)>>>((float*)x, (const float*)y, n4);
    return GB_CHECK_LAUNCH();
}
