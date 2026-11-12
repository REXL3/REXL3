// gb-nvcc: --fmad=false
// GLM-5.3-Flash prefill under contract P2 (Lane P step (e); docs/GLM_PREFILL_P2.md section 8.1): the KDA (Kimi delta attention)
// recurrence of a prompt piece in CHUNKS OF 16 ROWS, fp32 throughout. The Rust side is `kda::prefill_chunked` (crates/cuda/src/kda.rs);
// the inputs are those of P1's state kernel (kda.cu: q, k f32 from `prep`, v bf16, the per-channel decay g and beta from `project`).
//
// The recurrence, per head, value row j and key channel i (P1's `update()`): s *= g_r; mem = s . k_r; delta = (v_r - mem) beta_r;
// s += k_r delta; y_r = s . q_r. Over a chunk of rows r = 0..15 with S0 the state at its first row and P_r = g_0 g_1 ... g_r (per key
// channel: the chunk-relative decay, at least e^-80 because the log decay is above `lower` = -5, so it and 1 / P_r stay normal in fp32):
//
//   delta_r = beta_r (v_r - [S0 (k_r o P_r) + sum_{t<r} A_kk[r][t] delta_t]),   A_kk[r][t] = (k_r o P_r) . (k_t / P_t)
//   y_r     = S0 (q_r o P_r) + sum_{t<=r} A_qk[r][t] delta_t,                     A_qk[r][t] = (q_r o P_r) . (k_t / P_t)
//   S_end   = S0 o P_15 + sum_t delta_t k^_t,   k^_t = (k_t / P_t) o P_15         (row j: S0[j][i] P_15[i] + sum_t delta_t[j] k^_t[i])
//
// A partial chunk (a prefill segment's last) is the full one with the rows past its end at g = 1, k = q = v = beta = 0: every term they add
// is an exact zero and no valid row reads them, so a row has the bits it has in any longer chunk.
//
// THE ARITHMETIC (P2's `kda=glm-kda-chunk16-v1`; a function of the chunk's rows and S0 alone, so of the prefill grid only through its
// 16-row chunks): every op rounded alone (--fmad=false) except the explicit fmaf; P_r = P_{r-1} * g_r from 1; k_t / P_t an IEEE
// division; a dot product over the 128 keys is sixteen fmaf chains over the key groups [8 c, 8 c + 8) in key order from +0, combined as
// t_c = c_c + c_{c+8} (c < 8), u_c = t_c + t_{c+4} (c < 4), w_c = u_c + u_{c+2} (c < 2), w_0 + w_1 (the xor butterfly over the group
// index with offsets 8, 4, 2, 1); the delta and y sums are fmaf chains over t in order from +0, mem = X + sum; the state update is
// s * P_15 then fmaf(delta_t, k^_t, s) for t = 0..15; y is rounded to bf16 (nearest even; toward zero for the oracle-rounding test knob,
// as P1's). The block shape (VS value rows a CTA) moves no bit.
//
// One CTA per (head, VS value rows), 16 value rows a warp: lane l = 16 jg + c owns the 8 value rows 16 w + 8 jg + m (m = 0..7) and the key
// group c (keys 8 c .. 8 c + 7) of the state, 64 values in registers through the whole piece (each shared operand then feeds 8 fmaf: the
// kernel is bound by shared memory and DRAM, not by the FMA pipe). The 16 lanes of a half-warp reduce the 8 rows' group partials with
// the halving butterfly (P1's `halve_reduce`, offsets 8, 4, 2, 1): lane l ends with the full sum of row m = owned(l). S0 is read from
// S[slot][cur], the end state written to S[slot][1 - cur] (the commit tail flips `cur` after it, as after P1's prefill pieces). REXL3's own design; the chunked (WY / UT) form of the gated delta rule is
// public literature (the oracle's Triton `chunk_kda`, FLA were read for the algorithm only, no code taken).
#include "common.cuh"

namespace {

constexpr int D = 128;  // key dim == value dim
constexpr int C = 16;   // rows a chunk
// A key vector in shared memory: key group c (8 keys) at 8 c + 4 (c / 4), so the 8 groups a quarter-warp reads are in distinct banks.
constexpr int LD = 140;
__device__ __forceinline__ int kof(int i) { return i + ((i >> 5) << 2); }

template <int VS>
struct Smem {
    float xk[C][LD];  // k_r o P_r: the row side of A_kk and the vector S0 multiplies for mem
    float xq[C][LD];  // q_r o P_r
    float ks[C][LD];  // k_t / P_t: the column side of A; then k^_t = (k_t / P_t) o P_15 for the state update
    float akk[C][C];
    float aqk[C][C];
    float dl[C][VS];  // delta_t of the CTA's value rows
    float vv[C][VS];  // v_r of the CTA's value rows (0 past the chunk's end)
    float beta[C];
    float pend[D];  // P_15 per key channel
};

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }

// The canonical combination of the 16 key-group partials (the xor butterfly over the group index, offsets 8, 4, 2, 1).
__device__ __forceinline__ float tree16(const float (&c)[16]) {
    float t[8], u[4];
#pragma unroll
    for (int g = 0; g < 8; ++g) t[g] = c[g] + c[g + 8];
#pragma unroll
    for (int g = 0; g < 4; ++g) u[g] = t[g] + t[g + 4];
    const float w0 = u[0] + u[2], w1 = u[1] + u[3];
    return w0 + w1;
}

// The dot product of two shared key vectors (kof layout), in one thread: the 16 group chains, then tree16.
__device__ __forceinline__ float dot128(const float* a, const float* b) {
    float c[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const float4 a0 = ld4(a + kof(8 * g)), a1 = ld4(a + kof(8 * g) + 4);
        const float4 b0 = ld4(b + kof(8 * g)), b1 = ld4(b + kof(8 * g) + 4);
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
    return tree16(c);
}

// Two dot products sharing the column vector `b`: dot128(a1, b) and dot128(a2, b), each with dot128's arithmetic.
__device__ __forceinline__ void dot128x2(const float* a1, const float* a2, const float* b, float& o1, float& o2) {
    float c1[16], c2[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const float4 b0 = ld4(b + kof(8 * g)), b1 = ld4(b + kof(8 * g) + 4);
        const float4 x0 = ld4(a1 + kof(8 * g)), x1 = ld4(a1 + kof(8 * g) + 4);
        const float4 y0 = ld4(a2 + kof(8 * g)), y1 = ld4(a2 + kof(8 * g) + 4);
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
    o1 = tree16(c1);
    o2 = tree16(c2);
}

// The 8 rows' partials of this lane's key group, reduced over the 16 lanes of its half-warp: every addition has the operands of the
// plain xor butterfly over the lanes with offsets 8, 4, 2, 1 (the halves of the live values are exchanged first, P1's halve_reduce), so
// the row lane l returns, owned(l), is the tree16 of its 16 group partials. `v` is clobbered.
__device__ __forceinline__ int owned(int lane) { return ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1); }

__device__ __forceinline__ float reduce8(float (&v)[8], int lane) {
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
__global__ void __launch_bounds__(VS * 2) chunk_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                       const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                                       const float* __restrict__ beta, float* S, const int* __restrict__ parity, int slot,
                                                       int rows, int heads, __nv_bfloat16* __restrict__ y) {
    constexpr int NT = VS * 2;
    __shared__ __align__(16) Smem<VS> sm;
    const int head = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int c = lane & 15;                       // key group
    const int jl0 = warp * 16 + (lane >> 4) * 8;  // the lane's first value row within the CTA
    const int jown = jl0 + owned(lane);           // the value row whose sums this lane holds after a reduction
    const int j0 = blockIdx.x * VS;
    const int cur = parity[slot] & 1;
    const size_t sh = (size_t)D * D;
    const float* rd = S + (((size_t)slot * 2 + cur) * heads + head) * sh;
    float* wr = S + (((size_t)slot * 2 + (cur ^ 1)) * heads + head) * sh;
    float s[8][8];  // s[m][e] = S[j0 + jl0 + m][8 c + e]
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const float* p = rd + (size_t)(j0 + jl0 + m) * D + 8 * c;
        const float4 a = ld4(p), b = ld4(p + 4);
        s[m][0] = a.x, s[m][1] = a.y, s[m][2] = a.z, s[m][3] = a.w;
        s[m][4] = b.x, s[m][5] = b.y, s[m][6] = b.z, s[m][7] = b.w;
    }
    const int n_chunks = (rows + C - 1) / C;
    // The next chunk's inputs, loaded into registers while the current chunk's state update runs (software pipelining): a thread below
    // D holds key channel `tid`'s g, k, q of the chunk's 16 rows; v and beta are spread over the threads (VS = 128: the upper 128
    // threads; VS = 64: every thread, 8 values a thread).
    constexpr int VPT = (C * VS) / (NT >= 2 * D ? NT - D : NT);  // v values a loading thread holds
    const bool chan = tid < D;
    const bool vload = NT >= 2 * D ? tid >= D : true;
    const int vt = NT >= 2 * D ? tid - D : tid;
    float pg[C], pk[C], pq[C], pb = 0.0f;
    unsigned short pv[VPT];  // bf16 bits: converted when stored, so the loads never stall the thread that issued them
    auto fetch = [&](int r0, int n) {
        if (chan) {
#pragma unroll
            for (int r = 0; r < C; ++r) {
                const bool ok = r < n;
                const size_t o = ((size_t)(r0 + r) * heads + head) * D + tid;
                pg[r] = ok ? g[o] : 1.0f;
                pk[r] = ok ? k[o] : 0.0f;
                pq[r] = ok ? q[o] : 0.0f;
            }
        }
        if (vload) {
#pragma unroll
            for (int e = 0; e < VPT; ++e) {
                const int idx = vt * VPT + e, r = idx / VS, jj = idx % VS;
                pv[e] = r < n ? reinterpret_cast<const unsigned short*>(v)[((size_t)(r0 + r) * heads + head) * D + j0 + jj] : (unsigned short)0;
            }
            if (vt < C) pb = vt < n ? beta[(size_t)(r0 + vt) * heads + head] : 0.0f;
        }
    };
    fetch(0, min(C, rows));
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int r0 = ci * C;
        const int n = min(C, rows - r0);
        // ---- the chunk-relative decay products and the scaled key / query vectors (from the prefetched registers)
        if (chan) {
            float P = 1.0f;
#pragma unroll
            for (int r = 0; r < C; ++r) {
                P = P * pg[r];
                sm.xk[r][kof(tid)] = pk[r] * P;
                sm.xq[r][kof(tid)] = pq[r] * P;
                sm.ks[r][kof(tid)] = __fdiv_rn(pk[r], P);
            }
            sm.pend[tid] = P;
        }
        if (vload) {
#pragma unroll
            for (int e = 0; e < VPT; ++e) {
                const int idx = vt * VPT + e;
                sm.vv[idx / VS][idx % VS] = __uint_as_float((unsigned)pv[e] << 16);
            }
            if (vt < C) sm.beta[vt] = pb;
        }
        __syncthreads();
        // ---- A_qk (t <= r: 136 entries) and A_kk (t < r: 120 entries): 256 dot products over the CTA's threads
        for (int e = tid; e < 136; e += NT) {
            if (e < 120) {  // (r, t < r): A_kk[r][t] and A_qk[r][t] share the column vector
                int r = 1;
                while (r * (r + 1) / 2 <= e) ++r;
                const int t = e - r * (r - 1) / 2;
                float a, b;
                dot128x2(sm.xk[r], sm.xq[r], sm.ks[t], a, b);
                sm.akk[r][t] = a;
                sm.aqk[r][t] = b;
            } else {  // the diagonal A_qk[r][r]
                const int r = e - 120;
                sm.aqk[r][r] = dot128(sm.xq[r], sm.ks[r]);
            }
        }
        __syncthreads();
        // ---- k^_t = (k_t / P_t) o P_15 in place (A is formed; the state update reads it after the next barrier)
        for (int e = tid; e < C * D; e += NT) {
            const int r = e / D, ch = e % D;
            sm.ks[r][kof(ch)] = sm.ks[r][kof(ch)] * sm.pend[ch];
        }
        // ---- X_r = S0 (k_r o P_r) and Yq_r = S0 (q_r o P_r) for the lane's owned value row
        float x[C], yq[C];
#pragma unroll
        for (int r = 0; r < C; r += 2) {
            float part[4][8];  // (row r, k), (row r, q), (row r + 1, k), (row r + 1, q)
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float* a = u & 1 ? &sm.xq[r + (u >> 1)][kof(8 * c)] : &sm.xk[r + (u >> 1)][kof(8 * c)];
                const float4 a0 = ld4(a), a1 = ld4(a + 4);
                const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    float acc = 0.0f;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc = __fmaf_rn(av[e], s[m][e], acc);
                    part[u][m] = acc;
                }
            }
            x[r] = reduce8(part[0], lane);
            yq[r] = reduce8(part[1], lane);
            x[r + 1] = reduce8(part[2], lane);
            yq[r + 1] = reduce8(part[3], lane);
        }
        // ---- delta_r = beta_r (v_r - (X_r + sum_{t<r} A_kk[r][t] delta_t)), then y_r = Yq_r + sum_{t<=r} A_qk[r][t] delta_t
        float d[C];
#pragma unroll
        for (int r = 0; r < C; ++r) {
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < r; ++t) acc = __fmaf_rn(sm.akk[r][t], d[t], acc);
            d[r] = (sm.vv[r][jown] - (x[r] + acc)) * sm.beta[r];
        }
        const bool writer = (lane & 1) == 0;
#pragma unroll
        for (int r = 0; r < C; ++r) {
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
        if (ci + 1 < n_chunks) fetch(r0 + C, min(C, rows - r0 - C));  // consumed by the next chunk; the loads overlap the update
        __syncthreads();  // k^ and every warp's deltas are in shared memory
        // ---- the state update: s * P_15, then fmaf(delta_t, k^_t, s) for t in order
        {
            const float4 p0 = ld4(&sm.pend[8 * c]), p1 = ld4(&sm.pend[8 * c + 4]);
            const float pv[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int m = 0; m < 8; ++m)
#pragma unroll
                for (int e = 0; e < 8; ++e) s[m][e] = s[m][e] * pv[e];
        }
#pragma unroll
        for (int t = 0; t < C; ++t) {
            const float* kh = &sm.ks[t][kof(8 * c)];
            const float4 h0 = ld4(kh), h1 = ld4(kh + 4);
            const float hv[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            const float4 d0 = ld4(&sm.dl[t][jl0]), d1 = ld4(&sm.dl[t][jl0 + 4]);
            const float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
            for (int m = 0; m < 8; ++m)
#pragma unroll
                for (int e = 0; e < 8; ++e) s[m][e] = __fmaf_rn(dv[m], hv[e], s[m][e]);
        }
        __syncthreads();  // the next chunk overwrites the shared arrays
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        float* p = wr + (size_t)(j0 + jl0 + m) * D + 8 * c;
        *reinterpret_cast<float4*>(p) = make_float4(s[m][0], s[m][1], s[m][2], s[m][3]);
        *reinterpret_cast<float4*>(p + 4) = make_float4(s[m][4], s[m][5], s[m][6], s[m][7]);
    }
}

template <int VS>
void launch(const float* q, const float* k, const __nv_bfloat16* v, const float* g, const float* b, float* S, const int* p, int slot,
            int rows, int heads, bool rz, __nv_bfloat16* y, cudaStream_t st) {
    const dim3 grid(D / VS, heads);
    if (rz)
        chunk_kernel<VS, true><<<grid, VS * 2, 0, st>>>(q, k, v, g, b, S, p, slot, rows, heads, y);
    else
        chunk_kernel<VS, false><<<grid, VS * 2, 0, st>>>(q, k, v, g, b, S, p, slot, rows, heads, y);
}

}  // namespace

// The chunked recurrence of `rows` prompt rows of one sequence in `slot` (rows 0, 16, 32, ... of the call start the chunks; the call starts
// on a chunk boundary of its prefill segment, the caller's rule). q, k f32 [rows][heads][128], v bf16 [rows][heads][128], g f32
// [rows][heads * 128], beta f32 [rows][heads], S f32 [slots][2][heads][128][128] (read S[slot][cur], write S[slot][1 - cur]), parity i32
// [slots], y bf16 [rows][heads][128]. vs = value rows a CTA (128 or 64): a speed choice, never a bit. rz: round y toward zero (the
// oracle-rounding test knob). Returns a cudaError_t.
extern "C" int gb_glm_p2_kda_chunk(const void* q, const void* k, const void* v, const void* g, const void* beta, void* S,
                                   const void* parity, int slot, int rows, int heads, int vs, int rz, void* y, void* stream) {
    if (rows <= 0) return 0;
    if (heads <= 0 || (vs != 128 && vs != 64) || slot < 0) return (int)cudaErrorInvalidValue;
    const cudaStream_t st = gb_stream(stream);
    const float* qq = static_cast<const float*>(q);
    const float* kk = static_cast<const float*>(k);
    const __nv_bfloat16* vv = static_cast<const __nv_bfloat16*>(v);
    const float* gg = static_cast<const float*>(g);
    const float* bb = static_cast<const float*>(beta);
    float* ss = static_cast<float*>(S);
    const int* pp = static_cast<const int*>(parity);
    __nv_bfloat16* yy = static_cast<__nv_bfloat16*>(y);
    if (vs == 128) launch<128>(qq, kk, vv, gg, bb, ss, pp, slot, rows, heads, rz != 0, yy, st);
    else launch<64>(qq, kk, vv, gg, bb, ss, pp, slot, rows, heads, rz != 0, yy, st);
    return (int)cudaGetLastError();
}
