// gb-nvcc: --fmad=false
// GLM-5.3-Flash glue kernels, the two the model needs that the bf16 glue does not have:
//
//   1. rmsnorm_f16: y = f16((x * inv) * w), x and y fp16 [rows, dim], w fp32 [dim], inv = 1 / sqrtf(ss / dim + eps), the sum of
//      squares in ONE fixed order that is a function of dim alone (thread t owns the 8-element vectors t, t + 256, ...; it adds
//      f[0]^2 .. f[7]^2 of each in turn; then a fixed xor-butterfly across each warp and warps 0 .. W-1 added in order). So a row's
//      output bits depend on that row alone, at any row count. This is the norm of the MLA chain (q_a_layernorm: dim 1536,
//      kv_a_layernorm: dim 512), which the model keeps in fp16 end to end (docs/GLM53.md), like the ExLlamaV3 fork.
//   2. topk_f16: the exact per-row top-`count` of fp16 logits by (value DESC, id ASC) over the first `valid` columns of rows
//      that are `stride` apart. The model's logits have 154,880 columns (the lm_head's width) but the tokenizer has 154,856
//      tokens: the last 24 are padding the sampler must never see, so they are not read at all. Same machinery and the same
//      guarantees as `topk` in glue.cu (every element becomes a unique u64 key, so the selected set is independent of how the
//      vocabulary is split; NaN counts as -inf, -0 ties with +0).
//
// Compiled --fmad=false and with no atomics on data: every `a*b+c` is two separately rounded operations, so a host loop with the
// same order reproduces the bits.
#include <math.h>

#include "common.cuh"

namespace {

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

constexpr int ERR_INVALID = 1;  // cudaErrorInvalidValue

__device__ __forceinline__ float h2f(u16 u) { return __half2float(__ushort_as_half(u)); }
__device__ __forceinline__ u16 f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

__device__ __forceinline__ void unpack8h(const uint4& u, float (&f)[8]) {
    f[0] = h2f((u16)(u.x & 0xFFFF)); f[1] = h2f((u16)(u.x >> 16));
    f[2] = h2f((u16)(u.y & 0xFFFF)); f[3] = h2f((u16)(u.y >> 16));
    f[4] = h2f((u16)(u.z & 0xFFFF)); f[5] = h2f((u16)(u.z >> 16));
    f[6] = h2f((u16)(u.w & 0xFFFF)); f[7] = h2f((u16)(u.w >> 16));
}
__device__ __forceinline__ u32 pack2h(float a, float b) { return (u32)f2h(a) | ((u32)f2h(b) << 16); }
__device__ __forceinline__ uint4 pack8h(const float (&f)[8]) {
    uint4 u;
    u.x = pack2h(f[0], f[1]); u.y = pack2h(f[2], f[3]); u.z = pack2h(f[4], f[5]); u.w = pack2h(f[6], f[7]);
    return u;
}

static inline __host__ __device__ bool aligned16(const void* p) { return ((uintptr_t)p & 15) == 0; }
static inline __host__ __device__ bool aligned8(const void* p) { return ((uintptr_t)p & 7) == 0; }

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

// ------------------------------------------------------------------------------------------------------------ rmsnorm

constexpr int RMS_NT = 256;

__global__ void __launch_bounds__(RMS_NT) rmsnorm_f16_k(const u16* x, const float* w, u16* y, int dim, float eps) {
    __shared__ float sm[RMS_NT / 32];
    const size_t base = (size_t)blockIdx.x * dim;
    const int nvec = dim / 8;
    float ss = 0.f;
    for (int v = threadIdx.x; v < nvec; v += RMS_NT) {
        float f[8];
        unpack8h(*(const uint4*)(x + base + (size_t)v * 8), f);
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = ss + f[i] * f[i];
    }
    ss = block_sum<RMS_NT>(ss, sm);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int v = threadIdx.x; v < nvec; v += RMS_NT) {
        float f[8], o[8];
        unpack8h(*(const uint4*)(x + base + (size_t)v * 8), f);
        const float4 w0 = *(const float4*)(w + (size_t)v * 8), w1 = *(const float4*)(w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = (f[i] * inv) * wv[i];
        *(uint4*)(y + base + (size_t)v * 8) = pack8h(o);
    }
}

// ---------------------------------------------------------------------------------------------------------------- top-k

constexpr int TK_NT = 256;
constexpr int TK_TILE = TK_NT * 4;
constexpr int TK_CAP = 2048;
constexpr int TK_FLUSH = 512;
constexpr int TK_PF = 2;
constexpr int TK_CHUNK = 16384;

// key = (ordered16(fp16 bits) << 32) | (0xFFFFFFFF - id): larger is better, ties go to the lower id. NaN -> -inf, -0 -> +0.
__device__ __forceinline__ u64 make_key(u16 u, u32 id) {
    if ((u & 0x7FFF) > 0x7C00) u = 0xFC00;
    else if ((u & 0x7FFF) == 0) u = 0;
    const u32 t = (u & 0x8000) ? (~(u32)u & 0xFFFF) : ((u32)u | 0x8000);
    return ((u64)t << 32) | (u64)(~id);
}

__device__ void bitonic_sort_desc(u64* arr, int M) {
    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int p = threadIdx.x; p < M / 2; p += TK_NT) {
                const int i = ((p & ~(j - 1)) << 1) | (p & (j - 1));
                const int l = i | j;
                const u64 a = arr[i], b = arr[l];
                const bool desc = (i & k) == 0;
                if (desc ? a < b : a > b) { arr[i] = b; arr[l] = a; }
            }
            __syncthreads();
        }
    }
}

template <int P>
struct TkShared {
    u64 pool[P];
    u64 cand[TK_CAP];
    int cnt;
    u64 thr;
};

template <int P>
__device__ void tk_merge(TkShared<P>& s, int count) {
    __syncthreads();
    const int cnt = s.cnt;
    int M = P;
    while (M < cnt) M <<= 1;
    for (int i = cnt + threadIdx.x; i < M; i += TK_NT) s.cand[i] = 0;
    __syncthreads();
    bitonic_sort_desc(s.cand, M);
    if (threadIdx.x < P) {
        const u64 a = s.pool[threadIdx.x], b = s.cand[P - 1 - threadIdx.x];
        s.pool[threadIdx.x] = a > b ? a : b;
    }
    __syncthreads();
    for (int j = P / 2; j > 0; j >>= 1) {
        u64 a = 0, b = 0;
        int i = 0, l = 0;
        const bool act = threadIdx.x < P && (threadIdx.x & j) == 0;
        if (act) {
            i = threadIdx.x; l = i | j;
            a = s.pool[i]; b = s.pool[l];
            if (a < b) { s.pool[i] = b; s.pool[l] = a; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { s.cnt = 0; s.thr = s.pool[count - 1]; }
    __syncthreads();
}

template <int P, class Src>
__device__ void tk_select(TkShared<P>& s, const Src& src, int len, int count) {
    if (threadIdx.x < P) s.pool[threadIdx.x] = 0;
    if (threadIdx.x == 0) { s.cnt = 0; s.thr = 0; }
    __syncthreads();
    const int ntile = (len + TK_TILE - 1) / TK_TILE;
    const int lane = threadIdx.x & 31;
    u64 ring[TK_PF][4];
#pragma unroll
    for (int u = 0; u < TK_PF; ++u) src.load4(u * TK_TILE + threadIdx.x * 4, ring[u]);
    for (int t0 = 0; t0 < ntile; t0 += TK_PF) {
#pragma unroll
        for (int u = 0; u < TK_PF; ++u) {
            const int t = t0 + u;
            if (t < ntile) {
                u64 cur[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) cur[q] = ring[u][q];
                src.load4((t + TK_PF) * TK_TILE + threadIdx.x * 4, ring[u]);
                const u64 thr = s.thr;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const bool pass = cur[q] > thr;
                    const u32 mask = __ballot_sync(0xffffffffu, pass);
                    if (mask) {
                        int base = 0;
                        if (lane == 0) base = atomicAdd(&s.cnt, __popc(mask));
                        base = __shfl_sync(0xffffffffu, base, 0);
                        if (pass) s.cand[base + __popc(mask & ((1u << lane) - 1))] = cur[q];
                    }
                }
                __syncthreads();
                const int cnt = s.cnt;
                if (cnt >= TK_FLUSH || (t == ntile - 1 && cnt > 0)) tk_merge<P>(s, count);
                else __syncthreads();
            }
        }
    }
}

struct Stage1Src {
    const u16* row;
    int lo, hi;
    bool vec;
    __device__ void load4(int idx, u64 (&k)[4]) const {
        const int e = lo + idx;
        if (vec && e + 4 <= hi) {
            const uint2 u = *(const uint2*)(row + e);
            k[0] = make_key(u.x & 0xFFFF, e); k[1] = make_key(u.x >> 16, e + 1);
            k[2] = make_key(u.y & 0xFFFF, e + 2); k[3] = make_key(u.y >> 16, e + 3);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) k[i] = (e + i < hi) ? make_key(row[e + i], e + i) : 0;
        }
    }
};

struct Stage2Src {
    const u64* keys;
    int len;
    __device__ void load4(int idx, u64 (&k)[4]) const {
        if (idx + 4 <= len) {
            const ulonglong2 a = *(const ulonglong2*)(keys + idx), b = *(const ulonglong2*)(keys + idx + 2);
            k[0] = a.x; k[1] = a.y; k[2] = b.x; k[3] = b.y;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) k[i] = (idx + i < len) ? keys[idx + i] : 0;
        }
    }
};

template <int P>
__global__ void __launch_bounds__(TK_NT) topk_stage1_k(const u16* logits, u64* ws, int stride, int valid, int nchunks, int count) {
    __shared__ TkShared<P> s;
    const int chunk = blockIdx.x, row = blockIdx.y;
    Stage1Src src;
    src.row = logits + (size_t)row * stride;
    src.lo = chunk * TK_CHUNK;
    src.hi = min(valid, src.lo + TK_CHUNK);
    src.vec = (stride % 4 == 0) && aligned8(logits);
    tk_select<P>(s, src, src.hi - src.lo, count);
    if (threadIdx.x < P) ws[((size_t)row * nchunks + chunk) * P + threadIdx.x] = s.pool[threadIdx.x];
}

template <int P>
__global__ void __launch_bounds__(TK_NT) topk_stage2_k(const u64* ws, int* ids, float* vals, int nchunks, int count) {
    __shared__ TkShared<P> s;
    const int row = blockIdx.x;
    Stage2Src src;
    src.keys = ws + (size_t)row * nchunks * P;
    src.len = nchunks * P;
    tk_select<P>(s, src, src.len, count);
    if (threadIdx.x < count) {
        const u64 key = s.pool[threadIdx.x];
        int id = -1;
        float v = -INFINITY;
        if (key != 0) {
            id = (int)~(u32)key;
            const u32 t = (u32)(key >> 32);
            const u32 u = (t & 0x8000) ? (t & 0x7FFF) : (~t & 0xFFFF);
            v = h2f((u16)u);
        }
        ids[(size_t)row * count + threadIdx.x] = id;
        vals[(size_t)row * count + threadIdx.x] = v;
    }
}

static int topk_p(int count) { return count <= 16 ? 16 : count <= 32 ? 32 : 64; }
static size_t topk_nchunks(int valid) { return ((size_t)valid + TK_CHUNK - 1) / TK_CHUNK; }

template <int P>
int launch_topk(const void* logits, int rows, int stride, int valid, int count, int* ids, float* vals, u64* ws, void* stream) {
    const int nch = (int)topk_nchunks(valid);
    topk_stage1_k<P><<<dim3(nch, rows), TK_NT, 0, gb_stream(stream)>>>((const u16*)logits, ws, stride, valid, nch, count);
    topk_stage2_k<P><<<rows, TK_NT, 0, gb_stream(stream)>>>(ws, ids, vals, nch, count);
    return GB_CHECK_LAUNCH();
}

}  // namespace

// y = f16((x * (1 / sqrtf(mean(x^2) + eps))) * w) per row; x, y fp16 [rows, dim] (y may alias x), w fp32 [dim]. dim % 8 == 0 and
// 16-byte aligned pointers, or cudaErrorInvalidValue.
GB_EXPORT int gb_glm_rmsnorm_f16(const void* x, const void* w, void* y, int rows, int dim, float eps, void* stream) {
    if (rows < 0 || dim < 8 || dim % 8 != 0) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!x || !w || !y || !aligned16(x) || !aligned16(w) || !aligned16(y)) return ERR_INVALID;
    rmsnorm_f16_k<<<rows, RMS_NT, 0, gb_stream(stream)>>>((const u16*)x, (const float*)w, (u16*)y, dim, eps);
    return GB_CHECK_LAUNCH();
}

// Bytes of workspace gb_glm_topk_f16 needs (0 for an unsupported shape).
GB_EXPORT size_t gb_glm_topk_f16_workspace_bytes(int rows, int valid, int count) {
    if (rows <= 0 || valid <= 0 || count <= 0 || count > 64) return 0;
    return (size_t)rows * topk_nchunks(valid) * topk_p(count) * sizeof(u64);
}

// Exact per-row top-`count` of fp16 logits `[rows, stride]` over columns 0..valid by (value DESC, id ASC): ids i32 [rows, count],
// vals f32 [rows, count] (fp16 widened). 1 <= count <= min(64, valid), valid <= stride. `workspace` is
// gb_glm_topk_f16_workspace_bytes(rows, valid, count) bytes, 16-byte aligned.
GB_EXPORT int gb_glm_topk_f16(const void* logits, int rows, int stride, int valid, int count, int* ids, float* vals, void* workspace,
                              size_t ws_bytes, void* stream) {
    if (rows < 0 || valid < 1 || valid > stride || count < 1 || count > 64 || count > valid) return ERR_INVALID;
    if (rows == 0) return 0;
    if (!logits || !ids || !vals || !workspace || !aligned16(workspace)) return ERR_INVALID;
    if (ws_bytes < gb_glm_topk_f16_workspace_bytes(rows, valid, count)) return ERR_INVALID;
    u64* ws = (u64*)workspace;
    const int p = topk_p(count);
    if (p == 16) return launch_topk<16>(logits, rows, stride, valid, count, ids, vals, ws, stream);
    if (p == 32) return launch_topk<32>(logits, rows, stride, valid, count, ids, vals, ws, stream);
    return launch_topk<64>(logits, rows, stride, valid, count, ids, vals, ws, stream);
}
