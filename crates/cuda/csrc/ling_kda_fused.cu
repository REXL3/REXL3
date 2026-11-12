// gb-nvcc: --fmad=false
// Ling-3.1 (bailing_moe_v3) KDA decode-stage fusion: the three independent per-layer stages that follow the qkv/f/g
// projections — prep (conv+silu+l2norm of the qkv rows), the b_proj gate (beta) and the full-rank f/g gate (g_decay + z)
// — run as ONE launch of one kernel instead of three sequential launches. Each block computes exactly the expressions of
// its source kernel (kda.cu's prep_kernel<true> with tile_rows=1, beta_from_b_kernel<X_BF16,1>, full_fg_gate_kernel<1>,
// the decode/verify profile rows <= 8), same operand order, same rounding, so every output bit is unchanged; only the
// launch count and the inter-kernel serialization differ. Ling-owned: no other family calls this entry.
#include "common.cuh"
#include "exl3_linear_decode.cuh"

namespace {

constexpr int D = 128;                 // key dim == value dim (kda.cu)
constexpr int KEEP = 3;                // conv history rows
constexpr int W_STEP = 4;              // warps per state-kernel block (kda.cu)
constexpr float Q_SCALE = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float L2_EPS = 1e-6f;

__device__ __forceinline__ float ling_warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    return x;
}

template <int N>
__device__ __forceinline__ int ling_log2c() {
    return N == 32 ? 5 : N == 16 ? 4 : N == 8 ? 3 : N == 4 ? 2 : 1;
}

template <int N>
__device__ __forceinline__ int ling_owned_index(int lane) {
    int idx = 0;
#pragma unroll
    for (int i = 0; i < ling_log2c<N>(); ++i) idx += ((lane >> (4 - i)) & 1) * (N >> (i + 1));
    return idx;
}

template <int N>
__device__ __forceinline__ float ling_halve_reduce(float (&v)[N], int lane) {
#pragma unroll
    for (int step = 0; step < ling_log2c<N>(); ++step) {
        const int half = N >> (step + 1);
        const int off = 16 >> step;
        const bool hi = (lane & off) != 0;
#pragma unroll
        for (int j = 0; j < half; ++j) {
            const float keep = hi ? v[half + j] : v[j];
            const float send = hi ? v[j] : v[half + j];
            v[j] = keep + __shfl_xor_sync(0xffffffffu, send, off);
        }
    }
    float x = v[0];
#pragma unroll
    for (int off = 16 >> ling_log2c<N>(); off; off >>= 1) x += __shfl_xor_sync(0xffffffffu, x, off);
    return x;
}

__device__ __forceinline__ float ling_round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ling_sigmoid_ieee(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ void ling_load4_bf16(const __nv_bfloat16* p, float (&x)[4]) {
    const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(p);
    const __nv_bfloat162 a = __ldg(p2), b = __ldg(p2 + 1);
    x[0] = __low2float(a);
    x[1] = __high2float(a);
    x[2] = __low2float(b);
    x[3] = __high2float(b);
}

__device__ __forceinline__ void ling_load_raw4_f32(const void* base, size_t idx, float (&x)[4]) {
    const float4 t = __ldg(reinterpret_cast<const float4*>(reinterpret_cast<const float*>(base) + idx));
    x[0] = ling_round_bf16(t.x);
    x[1] = ling_round_bf16(t.y);
    x[2] = ling_round_bf16(t.z);
    x[3] = ling_round_bf16(t.w);
}

__device__ __forceinline__ int ling_find_seq(const int* __restrict__ seq_start, int n_seq, int row) {
    int lo = 0, hi = n_seq - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (seq_start[mid] <= row) lo = mid; else hi = mid - 1;
    }
    return lo;
}

__device__ __forceinline__ void ling_load_x8_bf16(const void* x, size_t idx8, float (&f)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(x) + idx8);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
    f[0] = __low2float(b2[0]); f[1] = __high2float(b2[0]);
    f[2] = __low2float(b2[1]); f[3] = __high2float(b2[1]);
    f[4] = __low2float(b2[2]); f[5] = __high2float(b2[2]);
    f[6] = __low2float(b2[3]); f[7] = __high2float(b2[3]);
}

// Phase 0: prep_kernel<FROM_F32=true> with tile_rows = 1 (decode/verify batches of single rows), blockDim 128 in the
// source; here the block is 256 threads and threads 128..255 are idle (the grid mapping and per-thread math are the
// source's). One block per (1-row tile, 4-unit group).
__device__ __forceinline__ void ling_prep_block(
        const void* __restrict__ qkv, const __nv_bfloat16* __restrict__ conv_w,
        const __nv_bfloat16* __restrict__ conv_pool, const int* __restrict__ seq_start,
        const int* __restrict__ slot, const int* __restrict__ keep, int n_seq, int rows, int heads,
        bool emit_q, bool skip_full, __nv_bfloat16* __restrict__ raw_out, float* __restrict__ q,
        float* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ sq_sm,
        float* __restrict__ sk_sm, __nv_bfloat16* __restrict__ sv_sm, int unit, int row0, int row_last,
        int lr0) {
    const int lane = threadIdx.x & 31;
    if (!emit_q && !raw_out && unit < heads) return;
    const int row_end = min(row_last, rows);
    const size_t conv_dim = (size_t)3 * heads * D;
    const int ch0 = unit * D + lane * 4;

    const __nv_bfloat162* w02 = reinterpret_cast<const __nv_bfloat162*>(conv_w + (size_t)ch0 * 4);
    const __nv_bfloat162 wa0 = __ldg(w02), wa1 = __ldg(w02 + 1), wa2 = __ldg(w02 + 2), wa3 = __ldg(w02 + 3);
    const __nv_bfloat162* w12 = w02 + 4;
    const __nv_bfloat162 wb0 = __ldg(w12), wb1 = __ldg(w12 + 1), wb2 = __ldg(w12 + 2), wb3 = __ldg(w12 + 3);
    float wt[16];
    wt[0] = __low2float(wa0); wt[1] = __high2float(wa0);
    wt[2] = __low2float(wa1); wt[3] = __high2float(wa1);
    wt[4] = __low2float(wa2); wt[5] = __high2float(wa2);
    wt[6] = __low2float(wa3); wt[7] = __high2float(wa3);
    wt[8] = __low2float(wb0); wt[9] = __high2float(wb0);
    wt[10] = __low2float(wb1); wt[11] = __high2float(wb1);
    wt[12] = __low2float(wb2); wt[13] = __high2float(wb2);
    wt[14] = __low2float(wb3); wt[15] = __high2float(wb3);
    int seq = 0, start = 0, send = 0, n_lim = 0;
    size_t pool = 0;
    bool have = false;
    float xw[3][4];
    for (int row = row0; row < row_end; ++row) {
        const int lr = lr0 + (row - row0);
        if (row >= send) {
            seq = ling_find_seq(seq_start, n_seq, row);
            start = seq_start[seq];
            send = seq_start[seq + 1];
            pool = (size_t)slot[seq] * KEEP;
            n_lim = send - start;
            if (keep) {
                const int n = min(max(keep[seq], 0), n_lim);
                n_lim = (skip_full && n >= n_lim) ? 0 : n;
            }
            have = false;
        }
        const int lt = row - start;
        if (lt >= n_lim) continue;
        if (!have) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int idx = lt - 3 + j;
                if (idx >= 0) ling_load_raw4_f32(qkv, (size_t)(start + idx) * conv_dim + ch0, xw[j]);
                else ling_load4_bf16(conv_pool + (pool + 3 + idx) * conv_dim + ch0, xw[j]);
            }
            have = true;
        }
        float x0[4];
        ling_load_raw4_f32(qkv, (size_t)row * conv_dim + ch0, x0);
        if (raw_out) {
            const __nv_bfloat162 a = __floats2bfloat162_rn(x0[0], x0[1]), b = __floats2bfloat162_rn(x0[2], x0[3]);
            *reinterpret_cast<__nv_bfloat162*>(raw_out + (size_t)row * conv_dim + ch0) = a;
            *(reinterpret_cast<__nv_bfloat162*>(raw_out + (size_t)row * conv_dim + ch0) + 1) = b;
        }
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[c] = __fmaf_rn(wt[4 * c + j], j < 3 ? xw[j][c] : x0[c], acc[c]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            xw[0][c] = xw[1][c];
            xw[1][c] = xw[2][c];
            xw[2][c] = x0[c];
        }
        float c[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = ling_round_bf16(acc[i] / (1.0f + expf(-acc[i])));

        if (unit < 2 * heads) {
            const bool is_q = unit < heads;
            if (!is_q || emit_q) {
                float ss = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) ss = ss + c[i] * c[i];
                ss = ling_warp_sum(ss);
                const float inv = 1.0f / sqrtf(ss + L2_EPS);
                const float t0 = c[0] * inv, t1 = c[1] * inv, t2 = c[2] * inv, t3 = c[3] * inv;
                const float4 out = is_q ? make_float4(t0 * Q_SCALE, t1 * Q_SCALE, t2 * Q_SCALE, t3 * Q_SCALE)
                                        : make_float4(t0, t1, t2, t3);
                if (sq_sm && is_q)
                    *reinterpret_cast<float4*>(sq_sm + (size_t)lr * D + lane * 4) = out;
                else if (sk_sm && !is_q)
                    *reinterpret_cast<float4*>(sk_sm + (size_t)lr * D + lane * 4) = out;
                if (q && is_q) {
                    float* dst = q + ((size_t)row * heads + unit) * D + lane * 4;
                    *reinterpret_cast<float4*>(dst) = out;
                } else if (k && !is_q) {
                    float* dst = k + ((size_t)row * heads + (unit - heads)) * D + lane * 4;
                    *reinterpret_cast<float4*>(dst) = out;
                }
            }
        } else {
            const int hv = unit - 2 * heads;
            const __nv_bfloat162 a = __floats2bfloat162_rn(c[0], c[1]), b = __floats2bfloat162_rn(c[2], c[3]);
            if (sv_sm) {
                *reinterpret_cast<__nv_bfloat162*>(sv_sm + (size_t)lr * D + lane * 4) = a;
                *(reinterpret_cast<__nv_bfloat162*>(sv_sm + (size_t)lr * D + lane * 4) + 1) = b;
            }
            if (v) {
                *reinterpret_cast<__nv_bfloat162*>(v + ((size_t)row * heads + hv) * D + lane * 4) = a;
                *(reinterpret_cast<__nv_bfloat162*>(v + ((size_t)row * heads + hv) * D + lane * 4) + 1) = b;
            }
        }
    }
}

// Phase 1: beta_from_b_kernel<X_BF16, 1> (the decode profile's rows <= 8 instantiation): one head per block, BG_RT = 8
// rows, BG_NT = 256 threads.
__device__ __forceinline__ void ling_beta_block(
        const void* x, const __nv_bfloat16* w_b, int rows, int hidden, int heads, float* beta, int h, int rblk) {
    __shared__ float sm[8][8];  // [BG_RT][BG_NT/32]
    const int r0 = rblk * 8;
    const int nvec = hidden / 8;
    const __nv_bfloat16* wrow = w_b + (size_t)h * hidden;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[r] = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float wf[8];
        const uint4 wu = __ldg(reinterpret_cast<const uint4*>(wrow) + v);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&wu);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 t = __bfloat1622float2(b2[i]);
            wf[2 * i] = t.x;
            wf[2 * i + 1] = t.y;
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r0 + r < rows) {
                float xf[8];
                ling_load_x8_bf16(x, (size_t)(r0 + r) * nvec + v, xf);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r] = __fmaf_rn(xf[i], wf[i], acc[r]);
            }
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    float flat[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) flat[r] = acc[r];
    const float part = ling_halve_reduce<8>(flat, lane);
    const int idx = ling_owned_index<8>(lane);
    if ((lane & 3) == 0) sm[idx % 8][wid] = part;
    __syncthreads();
    if (threadIdx.x < 8) {
        const int rr = threadIdx.x;
        if (r0 + rr < rows) {
            float s = sm[rr][0];
            for (int wv = 1; wv < 8; ++wv) s = s + sm[rr][wv];
            beta[(size_t)(r0 + rr) * heads + h] = ling_round_bf16(ling_sigmoid_ieee(s));
        }
    }
}

// Phase 2: full_fg_gate_kernel<1>: 4 columns per thread, RT = 1 row per block row.
__device__ __forceinline__ void ling_fgg_block(
        const half* f, const half* g, const float* a_log, const float* dt_bias, float lower, int rows, int heads,
        float* g_decay, float* z, int cblock, int r) {
    const int C = heads * 128;
    const int idx = (cblock * 256 + threadIdx.x) * 4;
    if (idx >= C) return;
    if (r >= rows) return;
    const float decay = expf(__ldg(a_log + (idx >> 7)));
    const float4 dtb = __ldg(reinterpret_cast<const float4*>(dt_bias + idx));
    const size_t off = (size_t)r * C + idx;
    const uint2 uf = __ldg(reinterpret_cast<const uint2*>(f + off));
    const uint2 ug = __ldg(reinterpret_cast<const uint2*>(g + off));
    const half2* hf = reinterpret_cast<const half2*>(&uf);
    const half2* hg = reinterpret_cast<const half2*>(&ug);
    const float2 fa = __half22float2(hf[0]);
    const float2 fb = __half22float2(hf[1]);
    const float2 za = __half22float2(hg[0]);
    const float2 zb = __half22float2(hg[1]);
    const float4 fv = make_float4(fa.x, fa.y, fb.x, fb.y);
    const float4 zv = make_float4(za.x, za.y, zb.x, zb.y);
    const float4 d = make_float4(
        expf(lower * ling_sigmoid_ieee(decay * (fv.x + dtb.x))),
        expf(lower * ling_sigmoid_ieee(decay * (fv.y + dtb.y))),
        expf(lower * ling_sigmoid_ieee(decay * (fv.z + dtb.z))),
        expf(lower * ling_sigmoid_ieee(decay * (fv.w + dtb.w))));
    *reinterpret_cast<float4*>(g_decay + off) = d;
    *reinterpret_cast<float4*>(z + off) = zv;
}

}  // namespace

// Fused decode-stage kernel: blockDim 256. grid.x = prep blocks + beta blocks + fgg blocks (in that order). Only the
// rows <= 8 profile (decode/verify windows of up to 8 rows in one call) is implemented; the caller falls back to the
// three separate kernels above it.
__global__ void __launch_bounds__(256) ling_kda_prep_beta_fgg_kernel(
        const void* __restrict__ qkv, const void* __restrict__ x, const __nv_bfloat16* __restrict__ b_weight,
        const __nv_bfloat16* __restrict__ conv_w, const __nv_bfloat16* __restrict__ conv_pool,
        const int* __restrict__ seq_start, const int* __restrict__ slot, const int* __restrict__ keep,
        const half* __restrict__ f_out, const half* __restrict__ g_out, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float lower, int n_seq, int rows, int heads, int hidden,
        int emit_q, int skip_full, __nv_bfloat16* __restrict__ raw_out, float* __restrict__ q,
        float* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ g_decay,
        float* __restrict__ z_gate, float* __restrict__ beta) {
    const int prep_units = 3 * heads / 4;
    const int prep_blocks = rows * prep_units;      // tile_rows = 1: grid (rows, prep_units)
    const int beta_blocks = heads * ((rows + 7) / 8);
    const int C = heads * 128;
    const int cx = (C / 4 + 255) / 256;
    const int fgg_blocks = cx * rows;
    const int b = blockIdx.x;
    if (b < prep_blocks) {
        if (threadIdx.x < 128) {
            const int row = b % rows;
            const int unit = (b / rows) * 4 + (threadIdx.x >> 5);
            ling_prep_block(qkv, conv_w, conv_pool, seq_start, slot, keep, n_seq, rows, heads, emit_q != 0,
                            skip_full != 0, raw_out, q, k, v, nullptr, nullptr, nullptr, unit, row,
                            min(row + 1, rows), 0);
        }
    } else if (b < prep_blocks + beta_blocks) {
        const int bb = b - prep_blocks;
        const int h = bb % heads;
        const int rblk = bb / heads;
        ling_beta_block(x, b_weight, rows, hidden, heads, beta, h, rblk);
    } else {
        const int fb = b - prep_blocks - beta_blocks;
        const int cblock = fb % cx;
        const int r = fb / cx;
        ling_fgg_block(f_out, g_out, a_log, dt_bias, lower, rows, heads, g_decay, z_gate, cblock, r);
    }
}

GB_EXPORT int gb_ling_kda_prep_beta_fgg(const void* qkv, const void* x, const void* b_weight, const void* conv_w,
                                        const void* conv_pool, const int* seq_start, const int* slot, const int* keep,
                                        const void* f_out, const void* g_out, const float* a_log, const float* dt_bias,
                                        float lower, int n_seq, int rows, int heads, int hidden, int emit_q,
                                        int skip_full, void* raw_out, float* q, float* k, void* v, float* g_decay,
                                        float* z_gate, float* beta, void* stream) {
    if (rows < 1 || rows > 8 || n_seq < 1 || heads < 1 || heads % 4 != 0 || hidden <= 0 || hidden % 8 != 0)
        return (int)cudaErrorInvalidValue;
    const int C = heads * 128;
    const int prep_units = 3 * heads / 4;
    const int prep_blocks = rows * prep_units;
    const int beta_blocks = heads * ((rows + 7) / 8);
    const int cx = (C / 4 + 255) / 256;
    const int fgg_blocks = cx * rows;
    const int grid = prep_blocks + beta_blocks + fgg_blocks;
    ling_kda_prep_beta_fgg_kernel<<<grid, 256, 0, gb_stream(stream)>>>(
        qkv, x, (const __nv_bfloat16*)b_weight, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_pool,
        seq_start, slot, keep, (const half*)f_out, (const half*)g_out, a_log, dt_bias, lower, n_seq, rows, heads,
        hidden, emit_q, skip_full, (__nv_bfloat16*)raw_out, q, k, (__nv_bfloat16*)v, g_decay, z_gate, beta);
    return GB_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Batched commit tail (`gb_ling_kda_commit_tail_all`, `kda::commit_tail_all`):
// one launch replacing the `gb_kda_commit_tail` launch of every KDA layer of a
// flush (the conv shift of the record's accepted rows + the parity flip). One
// (sequence, channel) thread per block; blockIdx.z selects the layer from a
// small descriptor table built once at model init (fixed device pointers).
// Bit-identical per layer to `commit_tail_kernel` (kda.cu): the same thread ->
// element mapping, the same KEEP history read/write order, the same parity flip.
// ---------------------------------------------------------------------------
struct LingKdaCommitLayer {
    const void* raw;
    void* conv_pool;
    int* parity;
};

__global__ void __launch_bounds__(256) ling_kda_commit_tail_all_kernel(
        const LingKdaCommitLayer* __restrict__ layers, const int* __restrict__ seq_start,
        const int* __restrict__ slot, const int* __restrict__ keep, int conv_dim, bool flip) {
    const LingKdaCommitLayer L = layers[blockIdx.z];
    const __nv_bfloat16* __restrict__ raw = (const __nv_bfloat16*)L.raw;
    __nv_bfloat16* conv_pool = (__nv_bfloat16*)L.conv_pool;
    int* parity = L.parity;
    const int seq = blockIdx.y, ch = blockIdx.x * 256 + threadIdx.x;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = keep ? min(max(keep[seq], 0), len) : len;
    if (n <= 0 || ch >= conv_dim) return;
    __nv_bfloat16* pool = conv_pool + (size_t)slot[seq] * KEEP * conv_dim + ch;
    __nv_bfloat16 nw[KEEP];
#pragma unroll
    for (int i = 0; i < KEEP; ++i) {
        const int e = n + i;
        nw[i] = e < KEEP ? pool[(size_t)e * conv_dim] : raw[(size_t)(begin + e - KEEP) * conv_dim + ch];
    }
#pragma unroll
    for (int i = 0; i < KEEP; ++i) pool[(size_t)i * conv_dim] = nw[i];
    if (flip && blockIdx.x == 0 && threadIdx.x == 0) parity[slot[seq]] ^= 1;
}

GB_EXPORT int gb_ling_kda_commit_tail_all(const void* layers, int n_layers, const int* seq_start,
                                          const int* slot, const int* keep, int n_seq, int conv_dim,
                                          int flip, void* stream) {
    if (n_layers < 1 || n_seq < 1 || conv_dim < 1 || conv_dim % 256 != 0) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)((conv_dim + 255) / 256), (unsigned)n_seq, (unsigned)n_layers);
    ling_kda_commit_tail_all_kernel<<<grid, 256, 0, gb_stream(stream)>>>(
        (const LingKdaCommitLayer*)layers, seq_start, slot, keep, conv_dim, flip != 0);
    return GB_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Fused linear epilogue + residual add + RMSNorm (`gb_ling_linear_add_rmsnorm`,
// ling_kda_proj_fused.cu): one launch of one 256-thread block per row replaces the
// `rot_out_kernel` + `rmsnorm_k<true>` pair of the KDA o_proj, MLA dense and dense
// FFN down_proj decode projections.
//
//   * Step A (8 warps, each 4 128-column blocks) computes the exact `rot_out_kernel`
//     output of this row into shared memory: the SK fp32 split partials added in
//     split order (`__ldcg` loads, the same q0 = 1, 9, ... chunking), the `svh`
//     scale, `fwht128`, `HAD_SCALE`, rounded to bf16. The rot_out math has no
//     FMA expression (pure FADD / FSUB / FMUL), so its bits do not depend on the
//     --fmad flag.
//   * Step B runs glue.cu's exact `rmsnorm_k<true>` loop over `dim / 8` vectors
//     (this file is compiled with --fmad=false like glue.cu, so the `ss = ss +
//     f[i] * f[i]` accumulation stays two roundings): hidden + res -> h_out (bf16),
//     then y = rmsnorm(h_out; w, eps). `h_out` may alias `x` (the model's residual
//     stream is updated in place), exactly as `gb_add_rmsnorm` allows.
//   * The hidden stream (`x`) is read from global memory while `res` is read from
//     shared memory; the bf16 codes are the same bytes the separate rot_out launch
//     would have written to global, so every output bit is the separate launches'.
//
// Grid: `m` blocks of 256. `n` must be <= 4096 (the shared `s_res`).
// One 128-column block of the next sub-layer's input rotation, computed from the bf16 codes this epilogue just
// wrote to `y` (staged in `s_y`): exactly `ling_rot_in_kernel`'s per-element math (`load4` of the bf16 row, the
// `suh` scale, `fwht128`, `HAD_SCALE`), stored back as fp16. Bit-identical per element to the separate launch.
__device__ __forceinline__ void ling_prerot_rot4(const uint16_t* sy, const half* suh, int col, int lane,
                                                 __half2 (&oh)[2]) {
    float v[4];
    v[0] = __uint_as_float((uint32_t)sy[col] << 16);
    v[1] = __uint_as_float((uint32_t)sy[col + 1] << 16);
    v[2] = __uint_as_float((uint32_t)sy[col + 2] << 16);
    v[3] = __uint_as_float((uint32_t)sy[col + 3] << 16);
    const uint2 us = __ldg(reinterpret_cast<const uint2*>(suh + col));
    const half2* hs = reinterpret_cast<const half2*>(&us);
    const float2 sa = __half22float2(hs[0]);
    const float2 sb = __half22float2(hs[1]);
    float s[4] = {sa.x, sa.y, sb.x, sb.y};
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    gbx::fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= gbx::HAD_SCALE;
    oh[0] = __floats2half2_rn(v[0], v[1]);
    oh[1] = __floats2half2_rn(v[2], v[3]);
}

__global__ void __launch_bounds__(256) ling_rot_out_add_rmsnorm_k(
    const float* __restrict__ c, const half* __restrict__ svh,
    const uint16_t* __restrict__ x, const float* __restrict__ w,
    uint16_t* __restrict__ h_out, uint16_t* __restrict__ y,
    int n, int sk, int dim, float eps,
    const half* __restrict__ suh_p0, const half* __restrict__ suh_p1, const half* __restrict__ suh_p2,
    half* __restrict__ xh_p0, half* __restrict__ xh_p1, half* __restrict__ xh_p2) {
    __shared__ uint16_t s_res[4096];
    __shared__ uint16_t s_y[4096];
    __shared__ float sm[8];
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t base = (size_t)row * dim;
    const size_t m = gridDim.x;

    // Step A: the rot_out_kernel finish of this row. One warp per 128-wide block,
    // each warp covering 4 blocks (the grid.y = 8 CTA layout of rot_out_kernel
    // folded into the 8 warps of this block). Per-column math identical.
#pragma unroll
    for (int bb = 0; bb < 4; ++bb) {
        const int blk = warp + 8 * bb;
        const int col = blk * 128 + 4 * lane;
        const size_t at = (size_t)row * n + col;
        float4 s = __ldcg(reinterpret_cast<const float4*>(c + at));
        for (int q0 = 1; q0 < sk; q0 += 8) {
            float4 u[8];
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) u[d] = __ldcg(reinterpret_cast<const float4*>(c + (size_t)(q0 + d) * m * n + at));
#pragma unroll
            for (int d = 0; d < 8; ++d)
                if (q0 + d < sk) { s.x += u[d].x; s.y += u[d].y; s.z += u[d].z; s.w += u[d].w; }
        }
        float v[4] = {s.x, s.y, s.z, s.w};
        gbx::fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = v[j] * gbx::HAD_SCALE * __half2float(__ldg(svh + col + j));
        const uint32_t x0 = __bfloat16_as_ushort(__float2bfloat16_rn(v[0]));
        const uint32_t x1 = __bfloat16_as_ushort(__float2bfloat16_rn(v[1]));
        const uint32_t x2 = __bfloat16_as_ushort(__float2bfloat16_rn(v[2]));
        const uint32_t x3 = __bfloat16_as_ushort(__float2bfloat16_rn(v[3]));
        uint4 u;
        u.x = x0 | (x1 << 16);
        u.y = x2 | (x3 << 16);
        *reinterpret_cast<uint4*>(s_res + col) = u;
    }

    __syncthreads();

    // Step B: glue.cu's rmsnorm_k<true>, verbatim (unpack8/rbf/pack8/block_sum).
    const int nvec = dim / 8;
    float ss = 0.f;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float f[8], r[8];
        const uint4 xu = *(const uint4*)(x + base + (size_t)v * 8);
        f[0] = __uint_as_float(xu.x << 16); f[1] = __uint_as_float(xu.x & 0xFFFF0000u);
        f[2] = __uint_as_float(xu.y << 16); f[3] = __uint_as_float(xu.y & 0xFFFF0000u);
        f[4] = __uint_as_float(xu.z << 16); f[5] = __uint_as_float(xu.z & 0xFFFF0000u);
        f[6] = __uint_as_float(xu.w << 16); f[7] = __uint_as_float(xu.w & 0xFFFF0000u);
        const uint4 ru = *(const uint4*)(s_res + (size_t)v * 8);
        r[0] = __uint_as_float(ru.x << 16); r[1] = __uint_as_float(ru.x & 0xFFFF0000u);
        r[2] = __uint_as_float(ru.y << 16); r[3] = __uint_as_float(ru.y & 0xFFFF0000u);
        r[4] = __uint_as_float(ru.z << 16); r[5] = __uint_as_float(ru.z & 0xFFFF0000u);
        r[6] = __uint_as_float(ru.w << 16); r[7] = __uint_as_float(ru.w & 0xFFFF0000u);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint16_t bi = __bfloat16_as_ushort(__float2bfloat16_rn(f[i] + r[i]));
            f[i] = __uint_as_float((uint32_t)bi << 16);
        }
        uint4 ho;
        ho.x = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f[0])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f[1])) << 16);
        ho.y = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f[2])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f[3])) << 16);
        ho.z = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f[4])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f[5])) << 16);
        ho.w = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f[6])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(f[7])) << 16);
        *(uint4*)(h_out + base + (size_t)v * 8) = ho;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss = ss + f[i] * f[i];
    }
    ss = ling_warp_sum(ss);
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    float s = sm[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) s = s + sm[w];
    __syncthreads();
    const float inv = 1.0f / sqrtf(s / (float)dim + eps);
    for (int v = threadIdx.x; v < nvec; v += 256) {
        uint4 src = *(uint4*)(h_out + base + (size_t)v * 8);
        float f[8];
        f[0] = __uint_as_float(src.x << 16); f[1] = __uint_as_float(src.x & 0xFFFF0000u);
        f[2] = __uint_as_float(src.y << 16); f[3] = __uint_as_float(src.y & 0xFFFF0000u);
        f[4] = __uint_as_float(src.z << 16); f[5] = __uint_as_float(src.z & 0xFFFF0000u);
        f[6] = __uint_as_float(src.w << 16); f[7] = __uint_as_float(src.w & 0xFFFF0000u);
        const float4 w0 = *(const float4*)(w + (size_t)v * 8), w1 = *(const float4*)(w + (size_t)v * 8 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = (f[i] * inv) * wv[i];
        uint4 out;
        out.x = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(o[0])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(o[1])) << 16);
        out.y = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(o[2])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(o[3])) << 16);
        out.z = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(o[4])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(o[5])) << 16);
        out.w = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(o[6])) | ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(o[7])) << 16);
        *(uint4*)(y + base + (size_t)v * 8) = out;
        *(uint4*)(s_y + (size_t)v * 8) = out;
    }

    // Next sub-layer input rotation fused into this epilogue: rotate this row's rmsnorm output (`y`, staged in
    // shared memory as the bf16 codes the separate rot_in launches read back) with the next stage's suh vectors
    // into its rotated inputs (`xh_p*`). Same grid partition as `ling_rot_in_kernel` (one warp per 128-column
    // block, 4 elements per lane), so every output element's bits are the separate launch's. With no prerot
    // arguments the single `__syncthreads()` is the only added cost; `s_y` is not written.
    __syncthreads();
    const bool p0 = suh_p0 && xh_p0;
    const bool p1 = suh_p1 && xh_p1;
    const bool p2 = suh_p2 && xh_p2;
    if (p0 || p1 || p2) {
#pragma unroll
        for (int bb = 0; bb < 4; ++bb) {
            const int blk = warp + 8 * bb;
            const int col = blk * 128 + 4 * lane;
            if (col >= dim) break;
            __half2 oh[2];
            if (p0) {
                ling_prerot_rot4(s_y, suh_p0, col, lane, oh);
                uint2 ou;
                ou.x = *reinterpret_cast<const unsigned*>(&oh[0]);
                ou.y = *reinterpret_cast<const unsigned*>(&oh[1]);
                *reinterpret_cast<uint2*>(xh_p0 + base + col) = ou;
            }
            if (p1) {
                ling_prerot_rot4(s_y, suh_p1, col, lane, oh);
                uint2 ou;
                ou.x = *reinterpret_cast<const unsigned*>(&oh[0]);
                ou.y = *reinterpret_cast<const unsigned*>(&oh[1]);
                *reinterpret_cast<uint2*>(xh_p1 + base + col) = ou;
            }
            if (p2) {
                ling_prerot_rot4(s_y, suh_p2, col, lane, oh);
                uint2 ou;
                ou.x = *reinterpret_cast<const unsigned*>(&oh[0]);
                ou.y = *reinterpret_cast<const unsigned*>(&oh[1]);
                *reinterpret_cast<uint2*>(xh_p2 + base + col) = ou;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused KDA window + gated norm (`gb_ling_kda_window_and_norm`, `kda::window`
// + `kda::norm` on the decode profile: rows <= 8, no keep, write_final). The
// state step is `state_kernel<R, true, RZ>` verbatim (same helpers, same
// expressions, same order, same --fmad=false); after the G_X blocks of a
// (seq, head) pair have written their y slices, the last of them — found with
// a per-(seq, head) atomic counter (`head_cnt`, zeroed at allocation,
// self-reset here) — runs the `norm_kernel<1>` math for every row of that
// sequence, while the non-last blocks overlap their store_state DRAM writes.
// y, S and out are bit-identical to the two separate launches.
// ---------------------------------------------------------------------------
template <int N>
__device__ __forceinline__ int ling_lane_of_index(int i) {
    int l = 0;
#pragma unroll
    for (int b = 0; b < ling_log2c<N>(); ++b) l |= ((i >> (ling_log2c<N>() - 1 - b)) & 1) << (4 - b);
    return l;
}

template <int R>
__device__ __forceinline__ int ling_row_of_lane(int lane) {
    return ling_owned_index<R>(lane);
}

template <int R>
__device__ __forceinline__ int ling_lane_of_row(int r) {
    return ling_lane_of_index<R>(r);
}

template <int R>
__device__ __forceinline__ float ling_reduce_rows(const float (&v)[R], int lane) {
    float a[R];
#pragma unroll
    for (int i = 0; i < R; ++i) a[i] = v[i];
    return ling_halve_reduce<R>(a, lane);
}

template <int R>
__device__ __forceinline__ void ling_update(float (&s)[R][4], const float (&k)[4], const float (&g)[4], float v_mine,
                                            float beta, int lane) {
    float mem[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        mem[r] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[r][i] = s[r][i] * g[i];
            mem[r] = mem[r] + s[r][i] * k[i];
        }
    }
    const float delta_mine = (v_mine - ling_reduce_rows<R>(mem, lane)) * beta;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float delta = __shfl_sync(0xffffffffu, delta_mine, ling_lane_of_row<R>(r));
#pragma unroll
        for (int i = 0; i < 4; ++i) s[r][i] = s[r][i] + k[i] * delta;
    }
}

template <int R>
__device__ __forceinline__ float ling_readout(const float (&s)[R][4], const float (&q)[4], int lane) {
    float out[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        out[r] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) out[r] = out[r] + s[r][i] * q[i];
    }
    return ling_reduce_rows<R>(out, lane);
}

template <int R>
struct ling_RowIn {
    float q[4], k[4], g[4], v, beta;

    __device__ __forceinline__ void load(const float* __restrict__ qp, const float* __restrict__ kp,
                                         const __nv_bfloat16* __restrict__ vp, const float* __restrict__ gp,
                                         const float* __restrict__ bp, int row, int head, int heads, int value0,
                                         int lane) {
        const size_t key = ((size_t)row * heads + head) * D + lane * 4;
        const float4 kk = __ldg(reinterpret_cast<const float4*>(kp + key));
        const float4 gg = __ldg(reinterpret_cast<const float4*>(gp + key));
        k[0] = kk.x; k[1] = kk.y; k[2] = kk.z; k[3] = kk.w;
        g[0] = gg.x; g[1] = gg.y; g[2] = gg.z; g[3] = gg.w;
        const float4 qq = __ldg(reinterpret_cast<const float4*>(qp + key));
        q[0] = qq.x; q[1] = qq.y; q[2] = qq.z; q[3] = qq.w;
        v = __bfloat162float(vp[((size_t)row * heads + head) * D + value0 + ling_row_of_lane<R>(lane)]);
        beta = bp[(size_t)row * heads + head];
    }
};

template <int R>
__device__ __forceinline__ void ling_load_state(float (&s)[R][4], const float* __restrict__ base, int value0, int lane) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float4 t = __ldg(reinterpret_cast<const float4*>(base + (size_t)(value0 + r) * D + lane * 4));
        s[r][0] = t.x; s[r][1] = t.y; s[r][2] = t.z; s[r][3] = t.w;
    }
}

template <int R>
__device__ __forceinline__ void ling_store_state(float* __restrict__ base, const float (&s)[R][4], int value0, int lane) {
#pragma unroll
    for (int r = 0; r < R; ++r)
        *reinterpret_cast<float4*>(base + (size_t)(value0 + r) * D + lane * 4) = make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
}

template <int R, bool RZ>
__global__ void __launch_bounds__(32 * W_STEP) ling_kda_window_norm_kernel(
        const float* __restrict__ q, const float* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, float* S, const int* __restrict__ parity,
        const int* __restrict__ seq_start, const int* __restrict__ slot, int heads, int* __restrict__ head_cnt,
        float eps, __nv_bfloat16* __restrict__ y, const float* __restrict__ z,
        const __nv_bfloat16* __restrict__ norm_w, __half* __restrict__ out,
        const half* __restrict__ o_suh, half* __restrict__ o_xh) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y, seq = blockIdx.z;
    const int value0 = (blockIdx.x * W_STEP + warp) * R;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    if (len <= 0) return;
    const size_t s_head = (size_t)D * D;
    const int sl = slot[seq], cur = parity[sl] & 1;
    const float* rd = S + (((size_t)sl * 2 + cur) * heads + head) * s_head;
    float* wr = S + (((size_t)sl * 2 + (cur ^ 1)) * heads + head) * s_head;
    float s[R][4];
    ling_load_state<R>(s, rd, value0, lane);
    ling_RowIn<R> next;
    next.load(q, k, v, g, beta, begin, head, heads, value0, lane);
    for (int j = 0; j < len; ++j) {
        const ling_RowIn<R> in = next;
        if (j + 1 < len) next.load(q, k, v, g, beta, begin + j + 1, head, heads, value0, lane);
        ling_update<R>(s, in.k, in.g, in.v, in.beta, lane);
        const float yv = ling_readout<R>(s, in.q, lane);
        if ((lane & (R == 8 ? 3 : 7)) == 0)
            y[((size_t)(begin + j) * heads + head) * D + value0 + ling_row_of_lane<R>(lane)] =
                RZ ? __float2bfloat16_rz(yv) : __float2bfloat16_rn(yv);
    }
    ling_store_state<R>(wr, s, value0, lane);

    // last-block handshake per (seq, head): the block whose atomicAdd returns G_X - 1
    // resets the counter and runs the norm (y is fully visible: every block fenced and
    // added before this one observed the count).
    __syncthreads();
    __threadfence();
    __shared__ int is_last_blk;
    if (threadIdx.x == 0) {
        const int idx = seq * heads + head;
        const int prev = atomicAdd(head_cnt + idx, 1);
        if (prev == (int)gridDim.x - 1) {
            head_cnt[idx] = 0;
            is_last_blk = 1;
        } else {
            is_last_blk = 0;
        }
    }
    __syncthreads();
    if (!is_last_blk) return;

    float wt[4];
    ling_load4_bf16(norm_w + lane * 4, wt);
    for (int j = warp; j < len; j += W_STEP) {
        const int row = begin + j;
        const size_t off = ((size_t)row * heads + head) * D + lane * 4;
        float x[4];
        ling_load4_bf16(y + off, x);
        const float4 zv = __ldg(reinterpret_cast<const float4*>(z + off));
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) ss = ss + x[i] * x[i];
        ss = ling_warp_sum(ss);
        const float rmf = 1.0f / sqrtf(ss / (float)D + eps);
        const float zz[4] = {zv.x, zv.y, zv.z, zv.w};
        __half2 h[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float o[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int c = 2 * i + e;
                o[e] = ((x[c] * wt[c]) * rmf) * ling_sigmoid_ieee(zz[c]);
            }
            h[i] = __floats2half2_rn(o[0], o[1]);
        }
        uint2 u;
        u.x = *reinterpret_cast<const unsigned*>(&h[0]);
        u.y = *reinterpret_cast<const unsigned*>(&h[1]);
        *reinterpret_cast<uint2*>(out + off) = u;
        // The next consumer of `out` (the KDA o_proj's input rotation) fused into this epilogue: rotate the
        // fp16 codes just written with `o_suh` into `o_xh` (the o_proj's rotated input), exactly
        // `ling_rot_in_kernel`'s math (`load4` of the f16 row, the `suh` scale, `fwht128`, `HAD_SCALE`).
        // Bit-identical per element to the separate launch; with `o_suh == nullptr` nothing is written.
        if (o_suh) {
            float v[4];
            const float2 f0 = __half22float2(h[0]);
            const float2 f1 = __half22float2(h[1]);
            v[0] = f0.x; v[1] = f0.y; v[2] = f1.x; v[3] = f1.y;
            const uint2 us = __ldg(reinterpret_cast<const uint2*>(o_suh + (size_t)head * D + lane * 4));
            const half2* hs = reinterpret_cast<const half2*>(&us);
            const float2 sa = __half22float2(hs[0]);
            const float2 sb = __half22float2(hs[1]);
            float s[4] = {sa.x, sa.y, sb.x, sb.y};
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] *= s[j];
            gbx::fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] *= gbx::HAD_SCALE;
            __half2 oh[2] = {__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3])};
            uint2 ou;
            ou.x = *reinterpret_cast<const unsigned*>(&oh[0]);
            ou.y = *reinterpret_cast<const unsigned*>(&oh[1]);
            *reinterpret_cast<uint2*>(o_xh + off) = ou;
        }
    }
}

GB_EXPORT int gb_ling_kda_window_and_norm(const float* q, const float* k, const void* v, const float* g,
                                          const float* beta, float* S, const int* parity, const int* seq_start,
                                          const int* slot, int n_seq, int heads, int rz, int r_step, void* y,
                                          int* head_cnt, float eps, const float* z, const void* norm_w, void* out,
                                          const void* o_suh, void* o_xh, void* stream) {
    if (n_seq < 0 || heads < 1 || heads % 4 != 0 || (r_step != 4 && r_step != 8)) return (int)cudaErrorInvalidValue;
    if ((o_suh == nullptr) != (o_xh == nullptr)) return (int)cudaErrorInvalidValue;
    if (n_seq == 0) return 0;
    const int gx = D / (r_step * W_STEP);
    cudaStream_t st = gb_stream(stream);
    const int block = 32 * W_STEP;
    if (r_step == 8) {
        if (rz)
            ling_kda_window_norm_kernel<8, true><<<dim3(gx, heads, n_seq), block, 0, st>>>(
                q, k, (const __nv_bfloat16*)v, g, beta, S, parity, seq_start, slot, heads, head_cnt, eps,
                (__nv_bfloat16*)y, z, (const __nv_bfloat16*)norm_w, (__half*)out, (const half*)o_suh, (half*)o_xh);
        else
            ling_kda_window_norm_kernel<8, false><<<dim3(gx, heads, n_seq), block, 0, st>>>(
                q, k, (const __nv_bfloat16*)v, g, beta, S, parity, seq_start, slot, heads, head_cnt, eps,
                (__nv_bfloat16*)y, z, (const __nv_bfloat16*)norm_w, (__half*)out, (const half*)o_suh, (half*)o_xh);
    } else {
        if (rz)
            ling_kda_window_norm_kernel<4, true><<<dim3(gx, heads, n_seq), block, 0, st>>>(
                q, k, (const __nv_bfloat16*)v, g, beta, S, parity, seq_start, slot, heads, head_cnt, eps,
                (__nv_bfloat16*)y, z, (const __nv_bfloat16*)norm_w, (__half*)out, (const half*)o_suh, (half*)o_xh);
        else
            ling_kda_window_norm_kernel<4, false><<<dim3(gx, heads, n_seq), block, 0, st>>>(
                q, k, (const __nv_bfloat16*)v, g, beta, S, parity, seq_start, slot, heads, head_cnt, eps,
                (__nv_bfloat16*)y, z, (const __nv_bfloat16*)norm_w, (__half*)out, (const half*)o_suh, (half*)o_xh);
    }
    return GB_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Fully fused decode/verify stage (`gb_ling_kda_prep_beta_fgg_window_norm`,
// replacing the sequential `prep_gate_fused` + `window_and_norm` pair): prep +
// beta + fgg + window + norm in ONE launch. Grid `(gx, heads, n_seq)`, 128
// threads per block, zero cross-block barriers (the per-`(seq, head)` last-block
// norm handshake is unchanged). Step 1: warps 0..2 run `ling_prep_block` for
// `head`'s q/k/v and warp 3 the fgg of `head`, staging the per-`(seq, head)`
// slices `s_q/s_k/s_v/s_g`; the global q/k/v/raw_out/g_decay/z writes that keep
// the buffers bit-identical to the separate launches are gated on
// `blockIdx.x == 0`. Step 2: `beta` for `head` over all rows, the source beta
// kernel's 256 logical threads mapped onto our 128 (each thread runs `vt = tid`
// and `vt = tid + 128`, the same acc/reduce math per logical thread and the same
// `sm_beta[wid]` sum order). Step 3: the exact
// `ling_kda_window_norm_kernel<R, RZ>` body reads q/k/v/g/beta from shared.
// Every output element computes the same expressions in the same order as the
// two separate launches.
// ---------------------------------------------------------------------------
template <int R>
__device__ __forceinline__ void ling_row_in_shared(
        const float (&sq)[8][128], const float (&sk)[8][128],
        const __nv_bfloat16 (&sv)[8][128], const float (&sg)[8][128],
        const float (&sb)[8], int begin, int j, int value0, int lane,
        float (&q)[4], float (&k)[4], float (&g)[4], float& v, float& beta) {
    const float4 kk = *reinterpret_cast<const float4*>(&sk[j][lane * 4]);
    const float4 gg = *reinterpret_cast<const float4*>(&sg[j][lane * 4]);
    const float4 qq = *reinterpret_cast<const float4*>(&sq[j][lane * 4]);
    k[0] = kk.x; k[1] = kk.y; k[2] = kk.z; k[3] = kk.w;
    g[0] = gg.x; g[1] = gg.y; g[2] = gg.z; g[3] = gg.w;
    q[0] = qq.x; q[1] = qq.y; q[2] = qq.z; q[3] = qq.w;
    v = __bfloat162float(sv[j][value0 + ling_row_of_lane<R>(lane)]);
    beta = sb[begin + j];
}

template <int R, bool RZ>
__global__ void __launch_bounds__(128) ling_kda_prep_beta_fgg_window_norm_kernel(
        const void* __restrict__ qkv, const void* __restrict__ x,
        const __nv_bfloat16* __restrict__ b_weight, const __nv_bfloat16* __restrict__ conv_w,
        const __nv_bfloat16* __restrict__ conv_pool, const int* __restrict__ seq_start,
        const int* __restrict__ slot, const int* __restrict__ keep, const half* __restrict__ f_out,
        const half* __restrict__ g_out, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float lower, int n_seq, int rows, int heads, int hidden,
        int emit_q, int skip_full, __nv_bfloat16* __restrict__ raw_out, float* __restrict__ q,
        float* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ g_decay,
        float* __restrict__ z_gate, float* __restrict__ beta, float* S,
        const int* __restrict__ parity, int* __restrict__ head_cnt, float eps,
        __nv_bfloat16* __restrict__ y, float* __restrict__ z,
        const __nv_bfloat16* __restrict__ norm_w, __half* __restrict__ out,
        const half* __restrict__ o_suh, half* __restrict__ o_xh) {
    __shared__ float s_q[8][128];
    __shared__ float s_k[8][128];
    __shared__ float s_g[8][128];
    __shared__ __nv_bfloat16 s_v[8][128];
    __shared__ float s_beta[8];
    __shared__ float sm_beta[8][8];

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y, seq = blockIdx.z;
    const int value0 = (blockIdx.x * W_STEP + warp) * R;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    if (len <= 0) return;
    const bool g0 = blockIdx.x == 0;

    // Step 1: prep q/k/v of `head` (warps 0..2), fgg of `head` (warp 3).
    if (warp < 3) {
        const int unit = warp * heads + head;
        ling_prep_block(qkv, conv_w, conv_pool, seq_start, slot, keep, n_seq, rows, heads,
                        emit_q != 0, skip_full != 0, g0 ? raw_out : nullptr, g0 ? q : nullptr,
                        g0 ? k : nullptr, g0 ? v : nullptr, (float*)s_q, (float*)s_k,
                        (__nv_bfloat16*)s_v, unit, begin, begin + len, 0);
    } else {
        const int C = heads * 128;
        const int ch0 = head * 128 + lane * 4;
        const float decay = expf(__ldg(a_log + head));
        const float4 dtb = __ldg(reinterpret_cast<const float4*>(dt_bias + ch0));
        for (int j = 0; j < len; ++j) {
            const size_t off = (size_t)(begin + j) * C + ch0;
            const uint2 uf = __ldg(reinterpret_cast<const uint2*>(f_out + off));
            const uint2 ug = __ldg(reinterpret_cast<const uint2*>(g_out + off));
            const half2* hf = reinterpret_cast<const half2*>(&uf);
            const half2* hg = reinterpret_cast<const half2*>(&ug);
            const float2 fa = __half22float2(hf[0]);
            const float2 fb = __half22float2(hf[1]);
            const float2 za = __half22float2(hg[0]);
            const float2 zb = __half22float2(hg[1]);
            const float4 fv = make_float4(fa.x, fa.y, fb.x, fb.y);
            const float4 zv = make_float4(za.x, za.y, zb.x, zb.y);
            const float4 d = make_float4(
                expf(lower * ling_sigmoid_ieee(decay * (fv.x + dtb.x))),
                expf(lower * ling_sigmoid_ieee(decay * (fv.y + dtb.y))),
                expf(lower * ling_sigmoid_ieee(decay * (fv.z + dtb.z))),
                expf(lower * ling_sigmoid_ieee(decay * (fv.w + dtb.w))));
            *reinterpret_cast<float4*>(&s_g[j][lane * 4]) = d;
            if (g0) {
                *reinterpret_cast<float4*>(g_decay + off) = d;
                *reinterpret_cast<float4*>(z + off) = zv;
            }
        }
    }

    // Step 2: beta for `head` over all rows (source beta kernel's 256 threads on
    // our 128: each thread runs vt = tid and vt = tid + 128).
    {
        const int nvec = hidden / 8;
        const __nv_bfloat16* wrow = b_weight + (size_t)head * hidden;
#pragma unroll
        for (int half_w = 0; half_w < 2; ++half_w) {
            const int vt = threadIdx.x + half_w * 128;
            const int wid = vt >> 5;
            const int vlane = vt & 31;
            float acc[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) acc[r] = 0.0f;
            for (int v = vt; v < nvec; v += 256) {
                float wf[8];
                const uint4 wu = __ldg(reinterpret_cast<const uint4*>(wrow) + v);
                const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&wu);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 t = __bfloat1622float2(b2[i]);
                    wf[2 * i] = t.x;
                    wf[2 * i + 1] = t.y;
                }
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    if (r < rows) {
                        float xf[8];
                        ling_load_x8_bf16(x, (size_t)r * nvec + v, xf);
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[r] = __fmaf_rn(xf[i], wf[i], acc[r]);
                    }
                }
            }
            float flat[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) flat[r] = acc[r];
            const float part = ling_halve_reduce<8>(flat, vlane);
            const int idx = ling_owned_index<8>(vlane);
            if ((vlane & 3) == 0) sm_beta[idx % 8][wid] = part;
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            const int rr = threadIdx.x;
            if (rr < rows) {
                float s = sm_beta[rr][0];
                for (int wv = 1; wv < 8; ++wv) s = s + sm_beta[rr][wv];
                const float bv = ling_round_bf16(ling_sigmoid_ieee(s));
                s_beta[rr] = bv;
                if (g0 && rr >= begin && rr < begin + len) beta[(size_t)rr * heads + head] = bv;
            }
        }
    }

    __syncthreads();

    // Step 3: the exact `ling_kda_window_norm_kernel<R, RZ>` body, reading
    // q/k/v/g/beta from shared.
    const size_t s_head = (size_t)D * D;
    const int sl = slot[seq], cur = parity[sl] & 1;
    const float* rd = S + (((size_t)sl * 2 + cur) * heads + head) * s_head;
    float* wr = S + (((size_t)sl * 2 + (cur ^ 1)) * heads + head) * s_head;
    float s[R][4];
    ling_load_state<R>(s, rd, value0, lane);

    float cq[4], ck[4], cg[4], cv, cb;
    ling_row_in_shared<R>(s_q, s_k, s_v, s_g, s_beta, begin, 0, value0, lane, cq, ck, cg, cv, cb);
    for (int j = 0; j < len; ++j) {
        // capture the current row first (like the original `const in = next`), then load the next
        float qq[4], kk[4], gg[4], vv, bb;
        qq[0] = cq[0]; qq[1] = cq[1]; qq[2] = cq[2]; qq[3] = cq[3];
        kk[0] = ck[0]; kk[1] = ck[1]; kk[2] = ck[2]; kk[3] = ck[3];
        gg[0] = cg[0]; gg[1] = cg[1]; gg[2] = cg[2]; gg[3] = cg[3];
        vv = cv;
        bb = cb;
        if (j + 1 < len)
            ling_row_in_shared<R>(s_q, s_k, s_v, s_g, s_beta, begin, j + 1, value0, lane, cq, ck, cg, cv, cb);
        ling_update<R>(s, kk, gg, vv, bb, lane);
        const float yv = ling_readout<R>(s, qq, lane);
        if ((lane & (R == 8 ? 3 : 7)) == 0)
            y[((size_t)(begin + j) * heads + head) * D + value0 + ling_row_of_lane<R>(lane)] =
                RZ ? __float2bfloat16_rz(yv) : __float2bfloat16_rn(yv);
    }
    ling_store_state<R>(wr, s, value0, lane);

    __syncthreads();
    __threadfence();
    __shared__ int is_last_blk;
    if (threadIdx.x == 0) {
        const int idx = seq * heads + head;
        const int prev = atomicAdd(head_cnt + idx, 1);
        if (prev == (int)gridDim.x - 1) {
            head_cnt[idx] = 0;
            is_last_blk = 1;
        } else {
            is_last_blk = 0;
        }
    }
    __syncthreads();
    if (!is_last_blk) return;

    float wt[4];
    ling_load4_bf16(norm_w + lane * 4, wt);
    for (int j = warp; j < len; j += W_STEP) {
        const int row = begin + j;
        const size_t off = ((size_t)row * heads + head) * D + lane * 4;
        float x[4];
        ling_load4_bf16(y + off, x);
        const float4 zv = __ldg(reinterpret_cast<const float4*>(z + off));
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) ss = ss + x[i] * x[i];
        ss = ling_warp_sum(ss);
        const float rmf = 1.0f / sqrtf(ss / (float)D + eps);
        const float zz[4] = {zv.x, zv.y, zv.z, zv.w};
        __half2 h[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float o[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int c = 2 * i + e;
                o[e] = ((x[c] * wt[c]) * rmf) * ling_sigmoid_ieee(zz[c]);
            }
            h[i] = __floats2half2_rn(o[0], o[1]);
        }
        uint2 u;
        u.x = *reinterpret_cast<const unsigned*>(&h[0]);
        u.y = *reinterpret_cast<const unsigned*>(&h[1]);
        *reinterpret_cast<uint2*>(out + off) = u;
        // The next consumer of `out` (the KDA o_proj's input rotation) fused into this epilogue: rotate the
        // fp16 codes just written with `o_suh` into `o_xh`, exactly `ling_rot_in_kernel`'s math. Bit-identical
        // per element to the separate launch; with `o_suh == nullptr` nothing is written.
        if (o_suh) {
            float v[4];
            const float2 f0 = __half22float2(h[0]);
            const float2 f1 = __half22float2(h[1]);
            v[0] = f0.x; v[1] = f0.y; v[2] = f1.x; v[3] = f1.y;
            const uint2 us = __ldg(reinterpret_cast<const uint2*>(o_suh + (size_t)head * D + lane * 4));
            const half2* hs = reinterpret_cast<const half2*>(&us);
            const float2 sa = __half22float2(hs[0]);
            const float2 sb = __half22float2(hs[1]);
            float ssuh[4] = {sa.x, sa.y, sb.x, sb.y};
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) v[jj] *= ssuh[jj];
            gbx::fwht128(v, lane);
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) v[jj] *= gbx::HAD_SCALE;
            __half2 oh[2] = {__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3])};
            uint2 ou;
            ou.x = *reinterpret_cast<const unsigned*>(&oh[0]);
            ou.y = *reinterpret_cast<const unsigned*>(&oh[1]);
            *reinterpret_cast<uint2*>(o_xh + off) = ou;
        }
    }
}

GB_EXPORT int gb_ling_kda_prep_beta_fgg_window_norm(
        const void* qkv, const void* x, const void* b_weight, const void* conv_w, const void* conv_pool,
        const int* seq_start, const int* slot, const int* keep, const void* f_out, const void* g_out,
        const float* a_log, const float* dt_bias, float lower, int n_seq, int rows, int heads, int hidden,
        int emit_q, int skip_full, void* raw_out, float* q, float* k, void* v, float* g_decay,
        float* z_gate, float* beta, float* S, const int* parity, int rz, int r_step, void* y,
        int* head_cnt, float eps, float* z, const void* norm_w, void* out, const void* o_suh,
        void* o_xh, void* stream) {
    if (rows < 1 || rows > 8 || n_seq < 1 || heads < 1 || heads % 4 != 0 || hidden <= 0 || hidden % 8 != 0)
        return (int)cudaErrorInvalidValue;
    if (r_step != 4 && r_step != 8) return (int)cudaErrorInvalidValue;
    if ((o_suh == nullptr) != (o_xh == nullptr)) return (int)cudaErrorInvalidValue;
    const int gx = D / (r_step * W_STEP);
    cudaStream_t st = gb_stream(stream);
    const dim3 grid(gx, heads, n_seq);
    if (r_step == 8) {
        if (rz)
            ling_kda_prep_beta_fgg_window_norm_kernel<8, true><<<grid, 128, 0, st>>>(
                qkv, x, (const __nv_bfloat16*)b_weight, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_pool,
                seq_start, slot, keep, (const half*)f_out, (const half*)g_out, a_log, dt_bias, lower,
                n_seq, rows, heads, hidden, emit_q, skip_full, (__nv_bfloat16*)raw_out, q, k,
                (__nv_bfloat16*)v, g_decay, z_gate, beta, S, parity, head_cnt, eps, (__nv_bfloat16*)y,
                z, (const __nv_bfloat16*)norm_w, (__half*)out, (const half*)o_suh, (half*)o_xh);
        else
            ling_kda_prep_beta_fgg_window_norm_kernel<8, false><<<grid, 128, 0, st>>>(
                qkv, x, (const __nv_bfloat16*)b_weight, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_pool,
                seq_start, slot, keep, (const half*)f_out, (const half*)g_out, a_log, dt_bias, lower,
                n_seq, rows, heads, hidden, emit_q, skip_full, (__nv_bfloat16*)raw_out, q, k,
                (__nv_bfloat16*)v, g_decay, z_gate, beta, S, parity, head_cnt, eps, (__nv_bfloat16*)y,
                z, (const __nv_bfloat16*)norm_w, (__half*)out, (const half*)o_suh, (half*)o_xh);
    } else {
        if (rz)
            ling_kda_prep_beta_fgg_window_norm_kernel<4, true><<<grid, 128, 0, st>>>(
                qkv, x, (const __nv_bfloat16*)b_weight, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_pool,
                seq_start, slot, keep, (const half*)f_out, (const half*)g_out, a_log, dt_bias, lower,
                n_seq, rows, heads, hidden, emit_q, skip_full, (__nv_bfloat16*)raw_out, q, k,
                (__nv_bfloat16*)v, g_decay, z_gate, beta, S, parity, head_cnt, eps, (__nv_bfloat16*)y,
                z, (const __nv_bfloat16*)norm_w, (__half*)out, (const half*)o_suh, (half*)o_xh);
        else
            ling_kda_prep_beta_fgg_window_norm_kernel<4, false><<<grid, 128, 0, st>>>(
                qkv, x, (const __nv_bfloat16*)b_weight, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_pool,
                seq_start, slot, keep, (const half*)f_out, (const half*)g_out, a_log, dt_bias, lower,
                n_seq, rows, heads, hidden, emit_q, skip_full, (__nv_bfloat16*)raw_out, q, k,
                (__nv_bfloat16*)v, g_decay, z_gate, beta, S, parity, head_cnt, eps, (__nv_bfloat16*)y,
                z, (const __nv_bfloat16*)norm_w, (__half*)out, (const half*)o_suh, (half*)o_xh);
    }
    return GB_CHECK_LAUNCH();
}

// launches `kda::gate_from_full_fg` (beta_from_b_kernel<X_BF16, 4> +
// full_fg_gate_kernel<8>) and `kda::prep_prefill_bf16` (prep_kernel<false>) run as
// ONE launch of one 256-thread kernel. Each block runs the exact body of its source
// kernel for the same (unit / head-group / row-block), just dispatched across
// blockIdx.y instead of separate grids, so every output bit is the separate
// launches'. blockIdx.y < P runs the prep tile (tile_rows = 64, a speed choice that
// moves no bit: each row computes the source's expressions from the same raw rows),
// P <= blockIdx.y < P + B the beta head-group blocks, the rest the full_fg row
// blocks (RZ = 8 rows each).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ling_prefill_prep_block(
        const void* __restrict__ qkv, const __nv_bfloat16* __restrict__ conv_w,
        const __nv_bfloat16* __restrict__ conv_pool, const int* __restrict__ seq_start,
        const int* __restrict__ slot, int n_seq, int rows, int heads,
        __nv_bfloat16* __restrict__ raw_out, float* __restrict__ q,
        float* __restrict__ k, __nv_bfloat16* __restrict__ v, int unit, int row0, int tile_rows) {
    const int lane = threadIdx.x & 31;
    if (unit >= 3 * heads) return;
    const int row_end = min(row0 + tile_rows, rows);
    const size_t conv_dim = (size_t)3 * heads * D;
    const int ch0 = unit * D + lane * 4;

    const __nv_bfloat162* w02 = reinterpret_cast<const __nv_bfloat162*>(conv_w + (size_t)ch0 * 4);
    const __nv_bfloat162 wa0 = __ldg(w02), wa1 = __ldg(w02 + 1), wa2 = __ldg(w02 + 2), wa3 = __ldg(w02 + 3);
    const __nv_bfloat162* w12 = w02 + 4;
    const __nv_bfloat162 wb0 = __ldg(w12), wb1 = __ldg(w12 + 1), wb2 = __ldg(w12 + 2), wb3 = __ldg(w12 + 3);
    float wt[16];
    wt[0] = __low2float(wa0); wt[1] = __high2float(wa0);
    wt[2] = __low2float(wa1); wt[3] = __high2float(wa1);
    wt[4] = __low2float(wa2); wt[5] = __high2float(wa2);
    wt[6] = __low2float(wa3); wt[7] = __high2float(wa3);
    wt[8] = __low2float(wb0); wt[9] = __high2float(wb0);
    wt[10] = __low2float(wb1); wt[11] = __high2float(wb1);
    wt[12] = __low2float(wb2); wt[13] = __high2float(wb2);
    wt[14] = __low2float(wb3); wt[15] = __high2float(wb3);
    int seq = 0, start = 0, send = 0, n_lim = 0;
    size_t pool = 0;
    bool have = false;
    float xw[3][4];
    for (int row = row0; row < row_end; ++row) {
        if (row >= send) {
            seq = ling_find_seq(seq_start, n_seq, row);
            start = seq_start[seq];
            send = seq_start[seq + 1];
            pool = (size_t)slot[seq] * KEEP;
            n_lim = send - start;
            have = false;
        }
        const int lt = row - start;
        if (lt >= n_lim) continue;
        if (!have) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int idx = lt - 3 + j;
                if (idx >= 0) ling_load4_bf16((const __nv_bfloat16*)qkv + (size_t)(start + idx) * conv_dim + ch0, xw[j]);
                else ling_load4_bf16(conv_pool + (pool + 3 + idx) * conv_dim + ch0, xw[j]);
            }
            have = true;
        }
        float x0[4];
        ling_load4_bf16((const __nv_bfloat16*)qkv + (size_t)row * conv_dim + ch0, x0);
        if (raw_out) {
            const __nv_bfloat162 a = __floats2bfloat162_rn(x0[0], x0[1]), b = __floats2bfloat162_rn(x0[2], x0[3]);
            *reinterpret_cast<__nv_bfloat162*>(raw_out + (size_t)row * conv_dim + ch0) = a;
            *(reinterpret_cast<__nv_bfloat162*>(raw_out + (size_t)row * conv_dim + ch0) + 1) = b;
        }
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[c] = __fmaf_rn(wt[4 * c + j], j < 3 ? xw[j][c] : x0[c], acc[c]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            xw[0][c] = xw[1][c];
            xw[1][c] = xw[2][c];
            xw[2][c] = x0[c];
        }
        float c[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = ling_round_bf16(acc[i] / (1.0f + expf(-acc[i])));

        if (unit < 2 * heads) {
            const bool is_q = unit < heads;
            {
                float ss = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) ss = ss + c[i] * c[i];
                ss = ling_warp_sum(ss);
                const float inv = 1.0f / sqrtf(ss + L2_EPS);
                const float t0 = c[0] * inv, t1 = c[1] * inv, t2 = c[2] * inv, t3 = c[3] * inv;
                const float4 out = is_q ? make_float4(t0 * Q_SCALE, t1 * Q_SCALE, t2 * Q_SCALE, t3 * Q_SCALE)
                                        : make_float4(t0, t1, t2, t3);
                float* dst = (is_q ? q : k) + ((size_t)row * heads + (is_q ? unit : unit - heads)) * D + lane * 4;
                *reinterpret_cast<float4*>(dst) = out;
            }
        } else {
            const int hv = unit - 2 * heads;
            const __nv_bfloat162 a = __floats2bfloat162_rn(c[0], c[1]), b = __floats2bfloat162_rn(c[2], c[3]);
            *reinterpret_cast<__nv_bfloat162*>(v + ((size_t)row * heads + hv) * D + lane * 4) = a;
            *(reinterpret_cast<__nv_bfloat162*>(v + ((size_t)row * heads + hv) * D + lane * 4) + 1) = b;
        }
    }
}

// beta_from_b_kernel<X_BF16, 4> verbatim: NH = 4 heads per block.
__device__ __forceinline__ void ling_prefill_beta4_block(
        const void* x, const __nv_bfloat16* w_b, int rows, int hidden, int heads, float* beta, int h0, int r0) {
    __shared__ float sm[4][8][8];
    const int nvec = hidden / 8;
    const __nv_bfloat16* wrow = w_b + (size_t)h0 * hidden;
    float acc[4][8];
#pragma unroll
    for (int hh = 0; hh < 4; ++hh)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[hh][r] = 0.0f;
    for (int v = threadIdx.x; v < nvec; v += 256) {
        float wf[4][8];
#pragma unroll
        for (int hh = 0; hh < 4; ++hh) {
            const uint4 wu = __ldg(reinterpret_cast<const uint4*>(wrow + (size_t)hh * hidden) + v);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&wu);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 t = __bfloat1622float2(b2[i]);
                wf[hh][2 * i] = t.x;
                wf[hh][2 * i + 1] = t.y;
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r0 + r < rows) {
                float xf[8];
                ling_load_x8_bf16(x, (size_t)(r0 + r) * nvec + v, xf);
#pragma unroll
                for (int hh = 0; hh < 4; ++hh)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[hh][r] = __fmaf_rn(xf[i], wf[hh][i], acc[hh][r]);
            }
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
        constexpr int N = 8;
        float flat[N];
#pragma unroll
        for (int r = 0; r < N; ++r) flat[r] = acc[hh][r];
        const float part = ling_halve_reduce<N>(flat, lane);
        const int idx = ling_owned_index<N>(lane);
        if ((lane & ((1 << (5 - ling_log2c<N>())) - 1)) == 0) sm[hh][idx % 8][wid] = part;
    }
    __syncthreads();
    if (threadIdx.x < 4 * 8) {
        const int hh = threadIdx.x / 8, rr = threadIdx.x % 8;
        if (r0 + rr < rows) {
            float s = sm[hh][rr][0];
            for (int wv = 1; wv < 8; ++wv) s = s + sm[hh][rr][wv];
            beta[(size_t)(r0 + rr) * heads + h0 + hh] = ling_round_bf16(ling_sigmoid_ieee(s));
        }
    }
}

// full_fg_gate_kernel<8> verbatim.
__device__ __forceinline__ void ling_prefill_fgg8_block(
        const half* f, const half* g, const float* a_log, const float* dt_bias, float lower, int rows, int heads,
        float* g_decay, float* z, int cblock, int r0) {
    const int C = heads * 128;
    const int idx = (cblock * 256 + threadIdx.x) * 4;
    if (idx >= C || r0 >= rows) return;
    const float decay = expf(__ldg(a_log + (idx >> 7)));
    const float4 dtb = __ldg(reinterpret_cast<const float4*>(dt_bias + idx));
#pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
        const int r = r0 + rr;
        if (r < rows) {
            const size_t off = (size_t)r * C + idx;
            const uint2 uf = __ldg(reinterpret_cast<const uint2*>(f + off));
            const uint2 ug = __ldg(reinterpret_cast<const uint2*>(g + off));
            const half2* hf = reinterpret_cast<const half2*>(&uf);
            const half2* hg = reinterpret_cast<const half2*>(&ug);
            const float2 fa = __half22float2(hf[0]);
            const float2 fb = __half22float2(hf[1]);
            const float2 za = __half22float2(hg[0]);
            const float2 zb = __half22float2(hg[1]);
            const float4 fv = make_float4(fa.x, fa.y, fb.x, fb.y);
            const float4 zv = make_float4(za.x, za.y, zb.x, zb.y);
            const float4 d = make_float4(
                expf(lower * ling_sigmoid_ieee(decay * (fv.x + dtb.x))),
                expf(lower * ling_sigmoid_ieee(decay * (fv.y + dtb.y))),
                expf(lower * ling_sigmoid_ieee(decay * (fv.z + dtb.z))),
                expf(lower * ling_sigmoid_ieee(decay * (fv.w + dtb.w))));
            *reinterpret_cast<float4*>(g_decay + off) = d;
            *reinterpret_cast<float4*>(z + off) = zv;
        }
    }
}

__global__ void __launch_bounds__(256) ling_kda_prefill_prep_gate_fgg_kernel(
        const void* __restrict__ qkv, const void* __restrict__ x, const __nv_bfloat16* __restrict__ b_weight,
        const __nv_bfloat16* __restrict__ conv_w, const __nv_bfloat16* __restrict__ conv_pool,
        const int* __restrict__ seq_start, const int* __restrict__ slot,
        const half* __restrict__ f_out, const half* __restrict__ g_out, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float lower, int n_seq, int rows, int heads, int hidden,
        __nv_bfloat16* __restrict__ raw_out, float* __restrict__ q,
        float* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ g_decay,
        float* __restrict__ z_gate, float* __restrict__ beta) {
    const int P = 3 * heads / 4;      // prep unit rows (tile_rows = 64)
    const int B = heads / 4;          // beta head-group rows
    const int y = blockIdx.y;
    if (y < P) {
        if (threadIdx.x >= 128) return;
        const int row0 = blockIdx.x * 64;
        if (row0 >= rows) return;
        const int unit = y * 4 + (threadIdx.x >> 5);
        ling_prefill_prep_block(qkv, conv_w, conv_pool, seq_start, slot, n_seq, rows, heads, raw_out, q, k, v,
                                unit, row0, 64);
    } else if (y < P + B) {
        const int h0 = (y - P) * 4;
        const int r0 = blockIdx.x * 8;
        if (r0 >= rows) return;
        ling_prefill_beta4_block(x, b_weight, rows, hidden, heads, beta, h0, r0);
    } else {
        const int head = y - P - B;
        const int r0 = blockIdx.x * 8;
        if (r0 >= rows) return;
        ling_prefill_fgg8_block(f_out, g_out, a_log, dt_bias, lower, rows, heads, g_decay, z_gate, head, r0);
    }
}

GB_EXPORT int gb_ling_kda_prefill_prep_gate_fgg(const void* qkv, const void* x, const void* b_weight, const void* conv_w,
                                                const void* conv_pool, const int* seq_start, const int* slot,
                                                const void* f_out, const void* g_out, const float* a_log, const float* dt_bias,
                                                float lower, int n_seq, int rows, int heads, int hidden,
                                                void* raw_out, float* q, float* k, void* v, float* g_decay,
                                                float* z_gate, float* beta, void* stream) {
    if (rows < 1 || n_seq < 1 || heads < 1 || heads % 4 != 0 || hidden <= 0 || hidden % 8 != 0)
        return (int)cudaErrorInvalidValue;
    const int P = 3 * heads / 4;
    const int B = heads / 4;
    const int F = heads;
    const dim3 grid((unsigned)((rows + 7) / 8), (unsigned)(P + B + F));
    ling_kda_prefill_prep_gate_fgg_kernel<<<grid, 256, 0, gb_stream(stream)>>>(
        qkv, x, (const __nv_bfloat16*)b_weight, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_pool,
        seq_start, slot, (const half*)f_out, (const half*)g_out, a_log, dt_bias, lower, n_seq, rows, heads, hidden,
        (__nv_bfloat16*)raw_out, q, k, (__nv_bfloat16*)v, g_decay, z_gate, beta);
    return GB_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// P2 prefill fused chunk + gated norm + commit tail
// (`gb_ling_kda_prefill_chunk_norm_commit`, `window_chunked` + `kda::norm` +
// `kda::commit_from` on `prefill && !p1_dense`): one launch of `chunk_kernel<64>`
// (glm_p2_kda.cu, verbatim) whose LAST CTA per head (gridDim.x = 2, a
// per-head self-resetting atomic counter) additionally runs the exact
// `norm_kernel<4>` math for that head over every row and the exact
// `commit_tail_kernel` body for the head's channels (whole-chunk prefill: keep =
// null, seq 0, rows accepted, conv shift off the bf16 projection buffer + parity
// flip). Every element computes the source kernels' expressions, so all outputs
// are bit-identical; only the launch count changes.
// ---------------------------------------------------------------------------
constexpr int P2_C = 16;    // rows a chunk
constexpr int P2_LD = 140;
__device__ __forceinline__ int p2_kof(int i) { return i + ((i >> 5) << 2); }
__device__ __forceinline__ float4 p2_ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }

template <int VS>
struct P2Smem {
    float xk[P2_C][P2_LD];
    float xq[P2_C][P2_LD];
    float ks[P2_C][P2_LD];
    float akk[P2_C][P2_C];
    float aqk[P2_C][P2_C];
    float dl[P2_C][VS];
    float vv[P2_C][VS];
    float beta[P2_C];
    float pend[D];
};

__device__ __forceinline__ float p2_tree16(const float (&c)[16]) {
    float t[8], u[4];
#pragma unroll
    for (int g = 0; g < 8; ++g) t[g] = c[g] + c[g + 8];
#pragma unroll
    for (int g = 0; g < 4; ++g) u[g] = t[g] + t[g + 4];
    const float w0 = u[0] + u[2], w1 = u[1] + u[3];
    return w0 + w1;
}

__device__ __forceinline__ float p2_dot128(const float* a, const float* b) {
    float c[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const float4 a0 = p2_ld4(a + p2_kof(8 * g)), a1 = p2_ld4(a + p2_kof(8 * g) + 4);
        const float4 b0 = p2_ld4(b + p2_kof(8 * g)), b1 = p2_ld4(b + p2_kof(8 * g) + 4);
        float x = 0.0f;
        x = __fmaf_rn(a0.x, b0.x, x);
        x = __fmaf_rn(a0.y, b0.y, x);
        x = __fmaf_rn(a0.z, b0.z, x);
        x = __fmaf_rn(a0.w, b0.w, x);
        x = __fmaf_rn(a1.x, b1.x, x);
        x = __fmaf_rn(a1.y, b1.y, x);
        x = __fmaf_rn(a1.z, b1.z, x);
        x = __fmaf_rn(a1.w, b1.w, x);
        c[g] = x;
    }
    return p2_tree16(c);
}

__device__ __forceinline__ void p2_dot128x2(const float* a1, const float* a2, const float* b, float& o1, float& o2) {
    float c1[16], c2[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const float4 b0 = p2_ld4(b + p2_kof(8 * g)), b1 = p2_ld4(b + p2_kof(8 * g) + 4);
        const float4 x0 = p2_ld4(a1 + p2_kof(8 * g)), x1 = p2_ld4(a1 + p2_kof(8 * g) + 4);
        const float4 y0 = p2_ld4(a2 + p2_kof(8 * g)), y1 = p2_ld4(a2 + p2_kof(8 * g) + 4);
        float x = 0.0f, y = 0.0f;
        x = __fmaf_rn(x0.x, b0.x, x);
        x = __fmaf_rn(x0.y, b0.y, x);
        x = __fmaf_rn(x0.z, b0.z, x);
        x = __fmaf_rn(x0.w, b0.w, x);
        x = __fmaf_rn(x1.x, b1.x, x);
        x = __fmaf_rn(x1.y, b1.y, x);
        x = __fmaf_rn(x1.z, b1.z, x);
        x = __fmaf_rn(x1.w, b1.w, x);
        y = __fmaf_rn(y0.x, b0.x, y);
        y = __fmaf_rn(y0.y, b0.y, y);
        y = __fmaf_rn(y0.z, b0.z, y);
        y = __fmaf_rn(y0.w, b0.w, y);
        y = __fmaf_rn(y1.x, b1.x, y);
        y = __fmaf_rn(y1.y, b1.y, y);
        y = __fmaf_rn(y1.z, b1.z, y);
        y = __fmaf_rn(y1.w, b1.w, y);
        c1[g] = x;
        c2[g] = y;
    }
    o1 = p2_tree16(c1);
    o2 = p2_tree16(c2);
}

__device__ __forceinline__ int p2_owned(int lane) { return ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1); }

__device__ __forceinline__ float p2_reduce8(float (&v)[8], int lane) {
    {
        const bool hi = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float keep = hi ? v[4 + j] : v[j], send = hi ? v[j] : v[4 + j];
            v[j] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
        }
    }
    {
        const bool hi = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float keep = hi ? v[2 + j] : v[j], send = hi ? v[j] : v[2 + j];
            v[j] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
        }
    }
    {
        const bool hi = lane & 2;
        const float keep = hi ? v[1] : v[0], send = hi ? v[0] : v[1];
        v[0] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
    }
    return v[0] + __shfl_xor_sync(0xffffffffu, v[0], 1);
}

template <int VS, bool RZ>
__global__ void __launch_bounds__(VS * 2) ling_kda_prefill_chunk_norm_commit_kernel(
        const float* __restrict__ q, const float* __restrict__ k,
        const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, float* S, int* __restrict__ parity, int slot,
        int rows, int heads, __nv_bfloat16* __restrict__ y,
        const float* __restrict__ z, const __nv_bfloat16* __restrict__ norm_w, float eps, __half* __restrict__ out,
        const __nv_bfloat16* __restrict__ raw, __nv_bfloat16* __restrict__ conv_pool, int* __restrict__ head_cnt,
        const __half* __restrict__ ko_suh, __half* __restrict__ ko_xh) {
    constexpr int NT = VS * 2;
    __shared__ __align__(16) P2Smem<VS> sm;
    const int head = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int c = lane & 15;
    const int jl0 = warp * 16 + (lane >> 4) * 8;
    const int jown = jl0 + p2_owned(lane);
    const int j0 = blockIdx.x * VS;
    const int cur = parity[slot] & 1;
    const size_t sh = (size_t)D * D;
    const float* rd = S + (((size_t)slot * 2 + cur) * heads + head) * sh;
    float* wr = S + (((size_t)slot * 2 + (cur ^ 1)) * heads + head) * sh;
    float s[8][8];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const float* p = rd + (size_t)(j0 + jl0 + m) * D + 8 * c;
        const float4 a = p2_ld4(p), b = p2_ld4(p + 4);
        s[m][0] = a.x, s[m][1] = a.y, s[m][2] = a.z, s[m][3] = a.w;
        s[m][4] = b.x, s[m][5] = b.y, s[m][6] = b.z, s[m][7] = b.w;
    }
    const int n_chunks = (rows + P2_C - 1) / P2_C;
    constexpr int VPT = (P2_C * VS) / (NT >= 2 * D ? NT - D : NT);
    const bool chan = tid < D;
    const bool vload = NT >= 2 * D ? tid >= D : true;
    const int vt = NT >= 2 * D ? tid - D : tid;
    float pg[P2_C], pk[P2_C], pq[P2_C], pb = 0.0f;
    unsigned short pv[VPT];
    auto fetch = [&](int r0, int n) {
        if (chan) {
#pragma unroll
            for (int r = 0; r < P2_C; ++r) {
                const bool ok = r < n;
                const size_t o = ((size_t)(r0 + r) * heads + head) * D + tid;
                pg[r] = ok ? __ldg(g + o) : 1.0f;
                pk[r] = ok ? __ldg(k + o) : 0.0f;
                pq[r] = ok ? __ldg(q + o) : 0.0f;
            }
        }
        if (vload) {
#pragma unroll
            for (int e = 0; e < VPT; ++e) {
                const int idx = vt * VPT + e, r = idx / VS, jj = idx % VS;
                pv[e] = r < n ? __ldg(&reinterpret_cast<const unsigned short*>(v)[((size_t)(r0 + r) * heads + head) * D + j0 + jj]) : (unsigned short)0;
            }
            if (vt < P2_C) pb = vt < n ? __ldg(&beta[(size_t)(r0 + vt) * heads + head]) : 0.0f;
        }
    };
    fetch(0, min(P2_C, rows));
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int r0 = ci * P2_C;
        const int n = min(P2_C, rows - r0);
        if (chan) {
            float P = 1.0f;
#pragma unroll
            for (int r = 0; r < P2_C; ++r) {
                P = P * pg[r];
                sm.xk[r][p2_kof(tid)] = pk[r] * P;
                sm.xq[r][p2_kof(tid)] = pq[r] * P;
                sm.ks[r][p2_kof(tid)] = __fdiv_rn(pk[r], P);
            }
            sm.pend[tid] = P;
        }
        if (vload) {
#pragma unroll
            for (int e = 0; e < VPT; ++e) {
                const int idx = vt * VPT + e;
                sm.vv[idx / VS][idx % VS] = __uint_as_float((unsigned)pv[e] << 16);
            }
            if (vt < P2_C) sm.beta[vt] = pb;
        }
        __syncthreads();
        for (int e = tid; e < 136; e += NT) {
            if (e < 120) {
                int r = 1;
                while (r * (r + 1) / 2 <= e) ++r;
                const int t = e - r * (r - 1) / 2;
                float a, b;
                p2_dot128x2(sm.xk[r], sm.xq[r], sm.ks[t], a, b);
                sm.akk[r][t] = a;
                sm.aqk[r][t] = b;
            } else {
                const int r = e - 120;
                sm.aqk[r][r] = p2_dot128(sm.xq[r], sm.ks[r]);
            }
        }
        __syncthreads();
        for (int e = tid; e < P2_C * D; e += NT) {
            const int r = e / D, ch = e % D;
            sm.ks[r][p2_kof(ch)] = sm.ks[r][p2_kof(ch)] * sm.pend[ch];
        }
        float x[P2_C], yq[P2_C];
#pragma unroll
        for (int r = 0; r < P2_C; r += 2) {
            float part[4][8];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float* a = u & 1 ? &sm.xq[r + (u >> 1)][p2_kof(8 * c)] : &sm.xk[r + (u >> 1)][p2_kof(8 * c)];
                const float4 a0 = p2_ld4(a), a1 = p2_ld4(a + 4);
                const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    float acc = 0.0f;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc = __fmaf_rn(av[e], s[m][e], acc);
                    part[u][m] = acc;
                }
            }
            x[r] = p2_reduce8(part[0], lane);
            yq[r] = p2_reduce8(part[1], lane);
            x[r + 1] = p2_reduce8(part[2], lane);
            yq[r + 1] = p2_reduce8(part[3], lane);
        }
        float d[P2_C];
#pragma unroll
        for (int r = 0; r < P2_C; ++r) {
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < r; ++t) acc = __fmaf_rn(sm.akk[r][t], d[t], acc);
            d[r] = (sm.vv[r][jown] - (x[r] + acc)) * sm.beta[r];
        }
        const bool writer = (lane & 1) == 0;
#pragma unroll
        for (int r = 0; r < P2_C; ++r) {
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t <= r; ++t) acc = __fmaf_rn(sm.aqk[r][t], d[t], acc);
            if (writer) {
                if (r < n) {
                    const float yr = yq[r] + acc;
                    y[((size_t)(r0 + r) * heads + head) * D + j0 + jown] = RZ ? __float2bfloat16_rz(yr) : __float2bfloat16_rn(yr);
                }
                sm.dl[r][jown] = d[r];
            }
        }
        if (ci + 1 < n_chunks) fetch(r0 + P2_C, min(P2_C, rows - r0 - P2_C));
        __syncthreads();
        {
            const float4 p0 = p2_ld4(&sm.pend[8 * c]), p1 = p2_ld4(&sm.pend[8 * c + 4]);
            const float pv[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int m = 0; m < 8; ++m)
#pragma unroll
                for (int e = 0; e < 8; ++e) s[m][e] = s[m][e] * pv[e];
        }
#pragma unroll
        for (int t = 0; t < P2_C; ++t) {
            const float* kh = &sm.ks[t][p2_kof(8 * c)];
            const float4 h0 = p2_ld4(kh), h1 = p2_ld4(kh + 4);
            const float hv[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            const float4 d0 = p2_ld4(&sm.dl[t][jl0]), d1 = p2_ld4(&sm.dl[t][jl0 + 4]);
            const float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
            for (int m = 0; m < 8; ++m)
#pragma unroll
                for (int e = 0; e < 8; ++e) s[m][e] = __fmaf_rn(dv[m], hv[e], s[m][e]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        float* p = wr + (size_t)(j0 + jl0 + m) * D + 8 * c;
        *reinterpret_cast<float4*>(p) = make_float4(s[m][0], s[m][1], s[m][2], s[m][3]);
        *reinterpret_cast<float4*>(p + 4) = make_float4(s[m][4], s[m][5], s[m][6], s[m][7]);
    }

    // ---- epilogue: the last CTA of head `head` (gridDim.x = 2) runs the gated norm of that head over every row,
    // then the commit tail of the head's channels (conv shift off `raw` + parity flip).
    __syncthreads();
    __threadfence();
    __shared__ int s_last;
    if (threadIdx.x == 0) {
        const int prev = atomicAdd(head_cnt + head, 1);
        if (prev == (int)gridDim.x - 1) {
            head_cnt[head] = 0;
            s_last = 1;
        } else {
            s_last = 0;
        }
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();

    float wt[4];
    ling_load4_bf16(norm_w + lane * 4, wt);
    for (int rb = warp; rb * 4 < rows; rb += 4) {
        const int r0 = rb * 4;
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const int row = r0 + rr;
            if (row >= rows) break;
            const size_t off = ((size_t)row * heads + head) * D + lane * 4;
            float x[4];
            ling_load4_bf16(y + off, x);
            const float4 zv = __ldg(reinterpret_cast<const float4*>(z + off));
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) ss = ss + x[i] * x[i];
            ss = ling_warp_sum(ss);
            const float rmf = 1.0f / sqrtf(ss / (float)D + eps);
            const float zz[4] = {zv.x, zv.y, zv.z, zv.w};
            __half2 h[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                float o[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int c = 2 * i + e;
                    o[e] = ((x[c] * wt[c]) * rmf) * ling_sigmoid_ieee(zz[c]);
                }
                h[i] = __floats2half2_rn(o[0], o[1]);
            }
            uint2 u;
            u.x = *reinterpret_cast<const unsigned*>(&h[0]);
            u.y = *reinterpret_cast<const unsigned*>(&h[1]);
            *reinterpret_cast<uint2*>(out + off) = u;
            // Fused k.o input rotation: rotate the rounded fp16 gated row by ko_suh into ko_xh (the exact
            // gb_glm_p2_rotate math on the bytes `out` carries: fp16 values converted back to fp32, the suh
            // scale, the 128-point Walsh-Hadamard butterfly, HAD_SCALE, fp16 stores), so the P2 k.o GEMM
            // below skips its rotate launch.
            if (ko_suh && ko_xh) {
                float v[4];
                v[0] = __half2float(h[0].x);
                v[1] = __half2float(h[0].y);
                v[2] = __half2float(h[1].x);
                v[3] = __half2float(h[1].y);
                const int col = lane * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] *= __half2float(__ldg(ko_suh + head * 128 + col + j));
                gbx::fwht128(v, lane);
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] *= gbx::HAD_SCALE;
                __half2 q0 = __floats2half2_rn(v[0], v[1]), q1 = __floats2half2_rn(v[2], v[3]);
                uint2 qu;
                qu.x = *reinterpret_cast<const unsigned*>(&q0);
                qu.y = *reinterpret_cast<const unsigned*>(&q1);
                *reinterpret_cast<uint2*>(ko_xh + ((size_t)row * heads + head) * 128 + col) = qu;
            }
        }
    }

    const int conv_dim = 3 * heads * D;
    const int ch_base = head * (conv_dim / heads);
    __nv_bfloat16* pool = conv_pool + (size_t)slot * KEEP * conv_dim;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int ch = ch_base + threadIdx.x + 128 * k;
        if (ch >= conv_dim) continue;
        __nv_bfloat16 nw[KEEP];
#pragma unroll
        for (int i = 0; i < KEEP; ++i) {
            const int e = rows + i;
            nw[i] = e < KEEP ? pool[(size_t)e * conv_dim + ch] : raw[(size_t)(e - KEEP) * conv_dim + ch];
        }
#pragma unroll
        for (int i = 0; i < KEEP; ++i) pool[(size_t)i * conv_dim + ch] = nw[i];
    }
    if (head == 0 && threadIdx.x == 0) parity[slot] ^= 1;
}

GB_EXPORT int gb_ling_kda_prefill_chunk_norm_commit(const float* q, const float* k, const void* v, const float* g,
                                                    const float* beta, float* S, int* parity, int slot, int rows,
                                                    int heads, int rz, void* y, const float* z, const void* norm_w,
                                                    float eps, void* out, const void* raw, void* conv_pool,
                                                    int* head_cnt, void* stream, const void* ko_suh, void* ko_xh) {
    if (rows < 1 || heads < 1 || heads % 4 != 0 || slot < 0) return (int)cudaErrorInvalidValue;
    if ((ko_suh == nullptr) != (ko_xh == nullptr)) return (int)cudaErrorInvalidValue;
    const dim3 grid(D / 64, heads);
    cudaStream_t st = gb_stream(stream);
    if (rz)
        ling_kda_prefill_chunk_norm_commit_kernel<64, true><<<grid, 128, 0, st>>>(
            q, k, (const __nv_bfloat16*)v, g, beta, S, parity, slot, rows, heads,
            (__nv_bfloat16*)y, z, (const __nv_bfloat16*)norm_w, eps, (__half*)out,
            (const __nv_bfloat16*)raw, (__nv_bfloat16*)conv_pool, head_cnt,
            (const __half*)ko_suh, (__half*)ko_xh);
    else
        ling_kda_prefill_chunk_norm_commit_kernel<64, false><<<grid, 128, 0, st>>>(
            q, k, (const __nv_bfloat16*)v, g, beta, S, parity, slot, rows, heads,
            (__nv_bfloat16*)y, z, (const __nv_bfloat16*)norm_w, eps, (__half*)out,
            (const __nv_bfloat16*)raw, (__nv_bfloat16*)conv_pool, head_cnt,
            (const __half*)ko_suh, (__half*)ko_xh);
    return GB_CHECK_LAUNCH();
}
