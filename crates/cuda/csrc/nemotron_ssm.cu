// gb-nvcc: --fmad=false
// Mamba2 (SSD) mixer kernels of the Nemotron 3.5 Super VL text backbone: the causal conv over x | B | C with a 3-row
// history, the discretization (dt, g), the fp32 recurrence, and the grouped gated RMSNorm. Geometry: 128 heads x head_dim
// 64, state 128, 8 B/C groups (head h reads group h / 16), conv kernel 4 over 10240 channels. Reference semantics:
// crates/model/src/families/nemotron_h_omni/ssm_ref.rs. The bf16 rounding points are those of the fork's live path
// (exllamav3-dflash2: modules/mamba2.py, exllamav3_ext/gdn.cu, exllamav3_ext/norm.cu), as the plan decides them.
//
// Rounding points, each one bf16 round-to-nearest-even:
//   - the conv input xBC (the fork's mixed_xbc.to(bf16) before the conv),
//   - the conv output after silu: the bf16 x, B and C that the recurrence and the readout read,
//   - dt, stored as bf16 and used only as the x * dt input scale; g and the decay exp(g) come from the unrounded fp32 dt,
//   - z before the gated norm (the fork's z.to(bf16)),
//   - the core output y = C . S + D * x, before the norm,
//   - the gated norm output, which is the bf16 input of out_proj.
// Everything else is fp32. --fmad=false rounds every multiply and add alone, and one `advance` and one `readout` serve the
// verify window, the serial decode step and the commit replay, so their bits agree.
//
// Where the fork's live path differs from the decided semantics above (open for the plan to confirm):
//   - the core output: the fork's MAMBA2 rule kernel rounds y with __float2bfloat16_rz (toward zero); this file rounds to nearest,
//   - the update: the fork forms (B * x) * dt; this file forms B * (x * dt), as the plan's "x * dt input scale" reads,
//   - z: the fork's eager path rounds z to bf16 (modules/mamba2.py); its BC-graph path (mamba2_fused_op) keeps z in fp32,
//   - exp: the fork uses __expf in the rule kernel and in the silu gates; this file uses expf, as gdn.cu's pre kernel does.
//
// Verify (pre, then window) reads the committed pools and writes only the Record and the core output: the S pool is
// read-only and the conv pool is read by pre. Commit (commit_state, commit_conv) replays the accepted prefix of a Record
// into the pools. A serial decode step is a 1-row verify followed by commit(keep = 1).
//
// Pools (per layer, indexed by slot):
//   S    f32  [slots][128 heads][128 state i][64 head_dim j]  (S_SLOT floats per slot; j is contiguous)
//   conv bf16 [slots][3][10240]                              (the last 3 pre-conv xBC rows, oldest first)
// Sequences: seq_start i32 [n_seq + 1] (row offsets) and slot i32 [n_seq], both device arrays, so the launch parameters
// that change per step can be graph-captured. Rows of a sequence are consecutive and run in order.
// Preconditions the device does not check: slots within one commit launch are distinct, seq_start is non-decreasing,
// and every keep[i] is at most the length of sequence i (clamped on the device, never read out of range).

#include <stdlib.h>

#include "common.cuh"

namespace {

constexpr int HEADS = 128;
constexpr int STATE = 128;                   // N: the state rows of one head
constexpr int HDIM = 64;                     // P: the head_dim columns of one head; one thread per column
constexpr int GROUPS = 8;
constexpr int HPG = HEADS / GROUPS;          // heads per B/C group
constexpr int INNER = HEADS * HDIM;          // 8192: x and z, the gated norm's channels
constexpr int BCW = GROUPS * STATE;          // 1024: B, and C
constexpr int CONV = INNER + 2 * BCW;        // 10240: conv channels x | B | C
constexpr int KEEP = 3;                      // conv history rows
constexpr int CONV_TAPS_K = 4;               // conv taps: the history rows plus the current one
constexpr int NORM_GROUP = INNER / GROUPS;   // 1024 channels per gated-norm group
constexpr size_t S_HEAD = (size_t)STATE * HDIM;
constexpr size_t S_SLOT = (size_t)HEADS * S_HEAD;

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// The pre-conv xBC rows and z are read through these (lane S5): f32 rows (rounded to bf16 on read, as always) or, under P2's fused in_proj
// split, the same values already stored as bf16 (rbf of a bf16 value is that value): the same floats either way.
__device__ __forceinline__ float rd_bf(float x) { return rbf(x); }
__device__ __forceinline__ float rd_bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 to_bf(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ __nv_bfloat16 to_bf(__nv_bfloat16 x) { return x; }

// The sequence a global row belongs to: the largest s with seq_start[s] <= row (empty sequences are skipped).
__device__ __forceinline__ int find_seq(const int* __restrict__ seq_start, int n_seq, int row) {
    int lo = 0, hi = n_seq - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (seq_start[mid] <= row) lo = mid; else hi = mid - 1;
    }
    return lo;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    return x;
}

// The state update every path runs: s[i] = s[i] * decay + B[i] * xdt, with xdt = x * dt. Each op is rounded alone.
__device__ __forceinline__ void advance(float (&s)[STATE], const float* __restrict__ Bs, float decay, float xdt) {
#pragma unroll
    for (int i = 0; i < STATE; ++i) {
        const float kept = s[i] * decay;
        const float inj = Bs[i] * xdt;
        s[i] = kept + inj;
    }
}

// y = sum_i C[i] * s[i] + D * x on the (already advanced) state. Four interleaved partial sums, combined in a fixed order.
__device__ __forceinline__ float readout(const float (&s)[STATE], const float* __restrict__ Cs, float x, float d) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int i = 0; i < STATE; i += 4) {
        a0 = a0 + Cs[i] * s[i];
        a1 = a1 + Cs[i + 1] * s[i + 1];
        a2 = a2 + Cs[i + 2] * s[i + 2];
        a3 = a3 + Cs[i + 3] * s[i + 3];
    }
    return ((a0 + a1) + (a2 + a3)) + d * x;
}

// B (and C) of one row's group, widened from bf16 to fp32 (exact), into shared memory by the HDIM threads of a head.
__device__ __forceinline__ void stage_b(const __nv_bfloat16* __restrict__ xrow, int grp, float* __restrict__ Bs, int tid) {
    for (int i = tid; i < STATE; i += HDIM) Bs[i] = bf2f(xrow[INNER + grp * STATE + i]);
}
__device__ __forceinline__ void stage_bc(const __nv_bfloat16* __restrict__ xrow, int grp, float* __restrict__ Bs,
                                         float* __restrict__ Cs, int tid) {
    stage_b(xrow, grp, Bs, tid);
    for (int i = tid; i < STATE; i += HDIM) Cs[i] = bf2f(xrow[INNER + BCW + grp * STATE + i]);
}

// ---------------------------------------------------------------------------------------------------------------------
// pre: the depthwise causal conv over [3 history rows; the window's rows] with bias, silu, one rounding to bf16. Tap j
// multiplies x[t - 3 + j], oldest first, and the accumulation starts from the bias (the fork's order). A row before the
// window reads the conv pool. One thread per channel, so the reads of a warp are contiguous.
// ---------------------------------------------------------------------------------------------------------------------
template <typename TX>
__global__ void __launch_bounds__(256) pre_conv_kernel(
        const TX* __restrict__ xin, const __nv_bfloat16* __restrict__ conv_w, const __nv_bfloat16* __restrict__ conv_b,
        const __nv_bfloat16* __restrict__ conv_pool, const int* __restrict__ seq_start, const int* __restrict__ slot,
        int n_seq, __nv_bfloat16* __restrict__ out) {
    const int row = blockIdx.x, ch = blockIdx.y * 256 + threadIdx.x;
    const int seq = find_seq(seq_start, n_seq, row);
    const int start = seq_start[seq], lt = row - start;
    const size_t pool = (size_t)slot[seq] * KEEP;
    float acc = bf2f(conv_b[ch]);
#pragma unroll
    for (int j = 0; j < CONV_TAPS_K; ++j) {
        const int idx = lt - 3 + j;  // window row x[t - 3 + j]
        const float x = idx >= 0 ? rd_bf(xin[(size_t)(start + idx) * CONV + ch])
                                 : bf2f(conv_pool[(pool + (size_t)(3 + idx)) * CONV + ch]);
        acc = __fmaf_rn(bf2f(conv_w[(size_t)ch * CONV_TAPS_K + j]), x, acc);
    }
    out[(size_t)row * CONV + ch] = __float2bfloat16_rn(acc / (1.0f + expf(-acc)));
}

// dt and g for every row and head: dt = clamp(softplus(dt_raw + dt_bias), 0, +inf) with the linear threshold 20, g = -exp(A_log) * dt
// from the unrounded dt, decay = exp(g). The stored dt is bf16 and only scales x. A NaN input stays NaN.
__global__ void __launch_bounds__(HEADS) pre_dt_kernel(
        const float* __restrict__ dt_raw, const float* __restrict__ dt_bias, const float* __restrict__ a_log,
        __nv_bfloat16* __restrict__ dt_out, float* __restrict__ g_out, float* __restrict__ decay_out) {
    const int row = blockIdx.x, h = threadIdx.x;
    const size_t at = (size_t)row * HEADS + h;
    const float av = dt_raw[at] + dt_bias[h];
    const float sp = av > 20.0f ? av : log1pf(expf(av));
    const float dt = sp < 0.0f ? 0.0f : sp;
    const float g = -expf(a_log[h]) * dt;
    dt_out[at] = __float2bfloat16_rn(dt);
    g_out[at] = g;
    decay_out[at] = expf(g);
}

// ---------------------------------------------------------------------------------------------------------------------
// window (verify, or a decode step): one block per (head, sequence), one thread per head_dim column. The thread holds the
// column's STATE rows in registers, loaded once from the committed S (read-only here) and advanced row by row. Each row
// writes its core output y (bf16). Nothing is stored to S.
// ---------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(HDIM) window_kernel(
        const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ dt, const float* __restrict__ decay,
        const float* __restrict__ S, const float* __restrict__ D, const int* __restrict__ seq_start, const int* __restrict__ slot,
        __nv_bfloat16* __restrict__ o) {
    __shared__ float Bs[STATE], Cs[STATE];
    const int head = blockIdx.x, seq = blockIdx.y, j = threadIdx.x;
    const int begin = seq_start[seq], end = seq_start[seq + 1];
    if (end <= begin) return;  // uniform across the block
    const int grp = head / HPG;
    const float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD + j;
    float s[STATE];
#pragma unroll
    for (int i = 0; i < STATE; ++i) s[i] = base[(size_t)i * HDIM];
    const float d = D[head];
    for (int r = begin; r < end; ++r) {
        const __nv_bfloat16* xr = xbc + (size_t)r * CONV;
        stage_bc(xr, grp, Bs, Cs, j);
        __syncthreads();
        const float x = bf2f(xr[head * HDIM + j]);
        advance(s, Bs, decay[(size_t)r * HEADS + head], x * bf2f(dt[(size_t)r * HEADS + head]));
        const float y = readout(s, Cs, x, d);
        o[(size_t)r * INNER + head * HDIM + j] = __float2bfloat16_rn(y);
        __syncthreads();
    }
}

// window + commit of every row (a prefill slice): window_kernel's recurrence, and the column's state after each sequence's last row is
// stored back in place. The state after the last row is what commit_state_kernel reaches with keep = the sequence's length (the same
// advance() over the same rows in the same order), so the pool ends with the same bits and S is read once instead of twice.
__global__ void __launch_bounds__(HDIM) window_commit_kernel(
        const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ dt, const float* __restrict__ decay,
        float* __restrict__ S, const float* __restrict__ D, const int* __restrict__ seq_start, const int* __restrict__ slot,
        __nv_bfloat16* __restrict__ o) {
    __shared__ float Bs[STATE], Cs[STATE];
    const int head = blockIdx.x, seq = blockIdx.y, j = threadIdx.x;
    const int begin = seq_start[seq], end = seq_start[seq + 1];
    if (end <= begin) return;  // uniform across the block
    const int grp = head / HPG;
    float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD + j;
    float s[STATE];
#pragma unroll
    for (int i = 0; i < STATE; ++i) s[i] = base[(size_t)i * HDIM];
    const float d = D[head];
    for (int r = begin; r < end; ++r) {
        const __nv_bfloat16* xr = xbc + (size_t)r * CONV;
        stage_bc(xr, grp, Bs, Cs, j);
        __syncthreads();
        const float x = bf2f(xr[head * HDIM + j]);
        advance(s, Bs, decay[(size_t)r * HEADS + head], x * bf2f(dt[(size_t)r * HEADS + head]));
        const float y = readout(s, Cs, x, d);
        o[(size_t)r * INNER + head * HDIM + j] = __float2bfloat16_rn(y);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < STATE; ++i) base[(size_t)i * HDIM] = s[i];
}

// window_commit_kernel for a PREFILL slice, faster and BIT-IDENTICAL (lane S3). Two changes of schedule, none of arithmetic:
//  * four threads a column: thread q of column j holds the state rows i = q, q + 4, .. (32 of them). advance() is independent per i, so
//    each s[i] gets exactly its old operations; readout()'s four interleaved partial sums a0..a3 ARE these four threads (a_q sums
//    C[i] s[i] over i = q, q + 4, .. in increasing i, as before), and ((a0 + a1) + (a2 + a3)) + d * x is formed by two xor shuffles
//    (IEEE addition is commutative bit for bit, so the lane that adds a1 + a0 holds a0 + a1). 256 threads a (head, sequence) block
//    instead of 64: four times the parallelism on a slice's 128 blocks.
//  * the rows' B, C, x, dt and decay are staged into shared memory kScanChunk rows at a time, so a row costs no global-memory latency
//    and no barrier (two barriers a chunk instead of two a row).
// The state carried from slice to slice is the same advance() over the same rows in the same order, so it is identical for any slice
// width (row invariance of the scan), and identical to window_commit_kernel's.
constexpr int kScanChunk = 16;
__global__ void __launch_bounds__(4 * HDIM) window_commit4_kernel(
        const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ dt, const float* __restrict__ decay,
        float* __restrict__ S, const float* __restrict__ D, const int* __restrict__ seq_start, const int* __restrict__ slot,
        __nv_bfloat16* __restrict__ o) {
    // B and C of a row, transposed by thread: entry (i % 4) * kQ + i / 4 holds row i, so thread q reads its 32 rows i = q, q + 4, ..
    // as 8 float4 (kQ = 36, not 32: the four threads' runs start in different banks)
    constexpr int kQ = 36;
    __shared__ __align__(16) float Bs[kScanChunk][4 * kQ], Cs[kScanChunk][4 * kQ];
    __shared__ float xs[kScanChunk][HDIM], xdts[kScanChunk], decs[kScanChunk];
    const int head = blockIdx.x, seq = blockIdx.y, t = threadIdx.x;
    const int j = t >> 2, q = t & 3;
    const int begin = seq_start[seq], end = seq_start[seq + 1];
    if (end <= begin) return;  // uniform across the block
    const int grp = head / HPG;
    float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD + j;
    float s[STATE / 4];
#pragma unroll
    for (int k = 0; k < STATE / 4; ++k) s[k] = base[(size_t)(4 * k + q) * HDIM];
    const float d = D[head];
    for (int r0 = begin; r0 < end; r0 += kScanChunk) {
        const int nr = min(kScanChunk, end - r0);
        __syncthreads();  // the previous chunk's rows are read
        for (int e = t; e < nr * STATE; e += 4 * HDIM) {
            const int rr = e / STATE, i = e % STATE;
            const __nv_bfloat16* xr = xbc + (size_t)(r0 + rr) * CONV;
            Bs[rr][(i & 3) * kQ + (i >> 2)] = bf2f(xr[INNER + grp * STATE + i]);
            Cs[rr][(i & 3) * kQ + (i >> 2)] = bf2f(xr[INNER + BCW + grp * STATE + i]);
        }
        for (int e = t; e < nr * HDIM; e += 4 * HDIM) {
            const int rr = e / HDIM, c = e % HDIM;
            xs[rr][c] = bf2f(xbc[(size_t)(r0 + rr) * CONV + head * HDIM + c]);
        }
        if (t < nr) {
            xdts[t] = bf2f(dt[(size_t)(r0 + t) * HEADS + head]);
            decs[t] = decay[(size_t)(r0 + t) * HEADS + head];
        }
        __syncthreads();
        for (int rr = 0; rr < nr; ++rr) {
            const float x = xs[rr][j];
            const float xdt = x * xdts[rr];
            const float dc = decs[rr];
            float a = 0.0f;
            const float4* B4 = reinterpret_cast<const float4*>(&Bs[rr][q * kQ]);
            const float4* C4 = reinterpret_cast<const float4*>(&Cs[rr][q * kQ]);
#pragma unroll
            for (int m = 0; m < STATE / 16; ++m) {
                const float4 b4 = B4[m], c4 = C4[m];
                const float bb[4] = {b4.x, b4.y, b4.z, b4.w}, cc[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int k = 4 * m + u;  // state row i = 4 k + q
                    const float kept = s[k] * dc;
                    const float inj = bb[u] * xdt;
                    s[k] = kept + inj;
                    a = a + cc[u] * s[k];
                }
            }
            const float pair = a + __shfl_xor_sync(0xffffffffu, a, 1);     // q 0, 1: a0 + a1; q 2, 3: a2 + a3
            const float all = pair + __shfl_xor_sync(0xffffffffu, pair, 2);  // q 0: (a0 + a1) + (a2 + a3)
            if (q == 0) o[(size_t)(r0 + rr) * INNER + head * HDIM + j] = __float2bfloat16_rn(all + d * x);
        }
    }
#pragma unroll
    for (int k = 0; k < STATE / 4; ++k) base[(size_t)(4 * k + q) * HDIM] = s[k];
}

// window_commit4_kernel with two columns a thread (j and j + 32; lane S5): the B and C loads of a row serve both, half the shared
// memory traffic (the bound of window_commit4_kernel, MEASURED with ncu); every column's operations are window_commit4_kernel's.
__global__ void __launch_bounds__(2 * HDIM) window_commit4x2_kernel(
        const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ dt, const float* __restrict__ decay,
        float* __restrict__ S, const float* __restrict__ D, const int* __restrict__ seq_start, const int* __restrict__ slot,
        __nv_bfloat16* __restrict__ o) {
    // B and C of a row, transposed by thread: entry (i % 4) * kQ + i / 4 holds row i, so thread q reads its 32 rows i = q, q + 4, ..
    // as 8 float4 (kQ = 36, not 32: the four threads' runs start in different banks)
    constexpr int kQ = 36;
    __shared__ __align__(16) float Bs[kScanChunk][4 * kQ], Cs[kScanChunk][4 * kQ];
    __shared__ float xs[kScanChunk][HDIM], xdts[kScanChunk], decs[kScanChunk];
    const int head = blockIdx.x, seq = blockIdx.y, t = threadIdx.x;
    const int j = t >> 2, q = t & 3;  // this thread's columns: j and j + HDIM / 2
    const int begin = seq_start[seq], end = seq_start[seq + 1];
    if (end <= begin) return;  // uniform across the block
    const int grp = head / HPG;
    float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD + j;
    float s[2][STATE / 4];
#pragma unroll
    for (int k = 0; k < STATE / 4; ++k) {
        s[0][k] = base[(size_t)(4 * k + q) * HDIM];
        s[1][k] = base[(size_t)(4 * k + q) * HDIM + HDIM / 2];
    }
    const float d = D[head];
    for (int r0 = begin; r0 < end; r0 += kScanChunk) {
        const int nr = min(kScanChunk, end - r0);
        __syncthreads();  // the previous chunk's rows are read
        for (int e = t; e < nr * STATE; e += 2 * HDIM) {
            const int rr = e / STATE, i = e % STATE;
            const __nv_bfloat16* xr = xbc + (size_t)(r0 + rr) * CONV;
            Bs[rr][(i & 3) * kQ + (i >> 2)] = bf2f(xr[INNER + grp * STATE + i]);
            Cs[rr][(i & 3) * kQ + (i >> 2)] = bf2f(xr[INNER + BCW + grp * STATE + i]);
        }
        for (int e = t; e < nr * HDIM; e += 2 * HDIM) {
            const int rr = e / HDIM, c = e % HDIM;
            xs[rr][c] = bf2f(xbc[(size_t)(r0 + rr) * CONV + head * HDIM + c]);
        }
        if (t < nr) {
            xdts[t] = bf2f(dt[(size_t)(r0 + t) * HEADS + head]);
            decs[t] = decay[(size_t)(r0 + t) * HEADS + head];
        }
        __syncthreads();
        for (int rr = 0; rr < nr; ++rr) {
            const float* B4f = &Bs[rr][q * kQ];
            const float* C4f = &Cs[rr][q * kQ];
            const float dc = decs[rr];
#pragma unroll
            for (int h = 0; h < 2; ++h) {  // window_commit4_kernel's row step for column j + h HDIM / 2, operation for operation
                const int jj = j + h * (HDIM / 2);
                const float x = xs[rr][jj];
                const float xdt = x * xdts[rr];
                float a = 0.0f;
                const float4* B4 = reinterpret_cast<const float4*>(B4f);
                const float4* C4 = reinterpret_cast<const float4*>(C4f);
#pragma unroll
                for (int m = 0; m < STATE / 16; ++m) {
                    const float4 b4 = B4[m], c4 = C4[m];
                    const float bb[4] = {b4.x, b4.y, b4.z, b4.w}, cc[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int k = 4 * m + u;
                        const float kept = s[h][k] * dc;
                        const float inj = bb[u] * xdt;
                        s[h][k] = kept + inj;
                        a = a + cc[u] * s[h][k];
                    }
                }
                const float pair = a + __shfl_xor_sync(0xffffffffu, a, 1);
                const float all = pair + __shfl_xor_sync(0xffffffffu, pair, 2);
                if (q == 0) o[(size_t)(r0 + rr) * INNER + head * HDIM + jj] = __float2bfloat16_rn(all + d * x);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < STATE / 4; ++k) {
        base[(size_t)(4 * k + q) * HDIM] = s[0][k];
        base[(size_t)(4 * k + q) * HDIM + HDIM / 2] = s[1][k];
    }
}

// window + commit of the sequences that ask for it (a decode window of single-row streams): window_kernel's recurrence for every
// sequence, and for each sequence with apply[seq] > 0 the column's state after its last row is stored back in place, in the pass that
// read it. That state is what commit_state_kernel reaches with keep = the sequence's length (the same advance() over the same rows in
// the same order), so the pool ends with the bits of window + commit and S is read once instead of twice and the commit launches go.
// A sequence with apply[seq] <= 0 is window_kernel's: nothing is stored. The core output is the same either way.
__global__ void __launch_bounds__(HDIM) window_commit_sel_kernel(
        const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ dt, const float* __restrict__ decay,
        float* __restrict__ S, const float* __restrict__ D, const int* __restrict__ seq_start, const int* __restrict__ slot,
        const int* __restrict__ apply, __nv_bfloat16* __restrict__ o) {
    __shared__ float Bs[STATE], Cs[STATE];
    const int head = blockIdx.x, seq = blockIdx.y, j = threadIdx.x;
    const int begin = seq_start[seq], end = seq_start[seq + 1];
    if (end <= begin) return;  // uniform across the block
    const int grp = head / HPG;
    float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD + j;
    float s[STATE];
#pragma unroll
    for (int i = 0; i < STATE; ++i) s[i] = base[(size_t)i * HDIM];
    const float d = D[head];
    for (int r = begin; r < end; ++r) {
        const __nv_bfloat16* xr = xbc + (size_t)r * CONV;
        stage_bc(xr, grp, Bs, Cs, j);
        __syncthreads();
        const float x = bf2f(xr[head * HDIM + j]);
        advance(s, Bs, decay[(size_t)r * HEADS + head], x * bf2f(dt[(size_t)r * HEADS + head]));
        const float y = readout(s, Cs, x, d);
        o[(size_t)r * INNER + head * HDIM + j] = __float2bfloat16_rn(y);
        __syncthreads();
    }
    if (apply[seq] > 0) {
#pragma unroll
        for (int i = 0; i < STATE; ++i) base[(size_t)i * HDIM] = s[i];
    }
}

// LAGGED decode window: window_commit_sel_kernel for single-row sequences, with the store of the state deferred. The S pool holds the
// state of a sequence as of its last FOLD; the rows decoded since (p of them, 0 <= p < K) wait in a per-slot ring of their records (the
// post-conv bf16 x and B of the row, dt and decay: all an advance() reads). The kernel loads S, advances it through the p pending rows in
// order (the same advance() over the same bits the serial decode ran: the state it reaches is the serial decode's state after those rows),
// then through the window's row, writes the row's core output as window_kernel does, and then either FOLDS (p + 1 == K: the state after
// the row is stored to S and the ring is empty again) or PUSHES the row's record into ring row p. The sequence's state is read once and
// written once every K steps instead of once and written once every step; the pending rows cost a few KB. A sequence with apply <= 0 is
// window_kernel's (the caller folded it first: p = 0): nothing is stored, nothing is pushed. `lag[seq]` is p of the sequence's slot,
// maintained by the host (device arrays read by the kernel, so the launch is graph-capturable).
__global__ void __launch_bounds__(HDIM) window_lag_kernel(
        const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ dt, const float* __restrict__ decay,
        float* __restrict__ S, const float* __restrict__ D, const int* __restrict__ seq_start, const int* __restrict__ slot,
        const int* __restrict__ apply, const int* __restrict__ lag, __nv_bfloat16* __restrict__ ring_xbc,
        __nv_bfloat16* __restrict__ ring_dt, float* __restrict__ ring_decay, int K, __nv_bfloat16* __restrict__ o) {
    __shared__ float Bs[STATE], Cs[STATE];
    const int head = blockIdx.x, seq = blockIdx.y, j = threadIdx.x;
    const int begin = seq_start[seq], end = seq_start[seq + 1];
    if (end <= begin) return;  // uniform across the block
    const int grp = head / HPG;
    const int sl = slot[seq];
    const bool lagged = apply[seq] > 0;
    float* base = S + (size_t)sl * S_SLOT + (size_t)head * S_HEAD + j;
    float s[STATE];
#pragma unroll
    for (int i = 0; i < STATE; ++i) s[i] = base[(size_t)i * HDIM];
    const float d = D[head];
    const int p = lagged ? min(max(lag[seq], 0), K - 1) : 0;
    const size_t ring0 = (size_t)sl * (size_t)(K - 1);
    for (int q = 0; q < p; ++q) {
        const __nv_bfloat16* rx = ring_xbc + (ring0 + q) * CONV;
        stage_b(rx, grp, Bs, j);
        __syncthreads();
        const float x = bf2f(rx[head * HDIM + j]);
        advance(s, Bs, ring_decay[(ring0 + q) * HEADS + head], x * bf2f(ring_dt[(ring0 + q) * HEADS + head]));
        __syncthreads();
    }
    for (int r = begin; r < end; ++r) {
        const __nv_bfloat16* xr = xbc + (size_t)r * CONV;
        stage_bc(xr, grp, Bs, Cs, j);
        __syncthreads();
        const float x = bf2f(xr[head * HDIM + j]);
        advance(s, Bs, decay[(size_t)r * HEADS + head], x * bf2f(dt[(size_t)r * HEADS + head]));
        const float y = readout(s, Cs, x, d);
        o[(size_t)r * INNER + head * HDIM + j] = __float2bfloat16_rn(y);
        __syncthreads();
    }
    if (!lagged) return;
    if (p + 1 >= K) {
#pragma unroll
        for (int i = 0; i < STATE; ++i) base[(size_t)i * HDIM] = s[i];
    } else {
        // push the row's record: its x slice, its group's B (the heads of a group write the same bytes), dt and decay
        const __nv_bfloat16* xr = xbc + (size_t)begin * CONV;
        __nv_bfloat16* rx = ring_xbc + (ring0 + p) * CONV;
        rx[head * HDIM + j] = xr[head * HDIM + j];
        for (int i = j; i < STATE; i += HDIM) rx[INNER + grp * STATE + i] = xr[INNER + grp * STATE + i];
        if (j == 0) {
            ring_dt[(ring0 + p) * HEADS + head] = dt[(size_t)begin * HEADS + head];
            ring_decay[(ring0 + p) * HEADS + head] = decay[(size_t)begin * HEADS + head];
        }
    }
}

// FOLD of lagged slots: the S pool of every layer ends with the state after the slot's pending rows. One launch for all layers and the
// listed slots (`lags[i]` pending rows of `slots[i]`; the same advance() as the window's, so the state is the serial decode's).
struct LagLayer {
    float* S;                          // the layer's S pool, all slots
    const __nv_bfloat16* ring_xbc;     // [slots][K - 1][CONV]
    const __nv_bfloat16* ring_dt;      // [slots][K - 1][HEADS]
    const float* ring_decay;           // [slots][K - 1][HEADS]
};

__global__ void __launch_bounds__(HDIM) fold_lag_all_kernel(const LagLayer* __restrict__ layers, const int* __restrict__ slots,
                                                            const int* __restrict__ lags, int K) {
    __shared__ float Bs[STATE];
    const LagLayer L = layers[blockIdx.z];
    const int head = blockIdx.x, i = blockIdx.y, j = threadIdx.x;
    const int p = min(max(lags[i], 0), K - 1);
    if (p <= 0) return;  // uniform across the block
    const int sl = slots[i];
    const int grp = head / HPG;
    float* base = L.S + (size_t)sl * S_SLOT + (size_t)head * S_HEAD + j;
    float s[STATE];
#pragma unroll
    for (int a = 0; a < STATE; ++a) s[a] = base[(size_t)a * HDIM];
    const size_t ring0 = (size_t)sl * (size_t)(K - 1);
    for (int q = 0; q < p; ++q) {
        const __nv_bfloat16* rx = L.ring_xbc + (ring0 + q) * CONV;
        stage_b(rx, grp, Bs, j);
        __syncthreads();
        const float x = bf2f(rx[head * HDIM + j]);
        advance(s, Bs, L.ring_decay[(ring0 + q) * HEADS + head], x * bf2f(L.ring_dt[(ring0 + q) * HEADS + head]));
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < STATE; ++a) base[(size_t)a * HDIM] = s[a];
}

// commit (state): replay the first keep[i] rows of each sequence with the same advance(), and store the column back in place.
// keep <= 0 leaves the sequence's state untouched. Each thread stores exactly the floats it loaded, so blocks never overlap.
__global__ void __launch_bounds__(HDIM) commit_state_kernel(
        const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ dt, const float* __restrict__ decay,
        float* __restrict__ S, const int* __restrict__ seq_start, const int* __restrict__ slot, const int* __restrict__ keep) {
    __shared__ float Bs[STATE];
    const int head = blockIdx.x, seq = blockIdx.y, j = threadIdx.x;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = min(max(keep[seq], 0), len);
    if (n <= 0) return;  // uniform across the block
    const int grp = head / HPG;
    float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD + j;
    float s[STATE];
#pragma unroll
    for (int i = 0; i < STATE; ++i) s[i] = base[(size_t)i * HDIM];
    for (int k = 0; k < n; ++k) {
        const int r = begin + k;
        const __nv_bfloat16* xr = xbc + (size_t)r * CONV;
        stage_b(xr, grp, Bs, j);
        __syncthreads();
        const float x = bf2f(xr[head * HDIM + j]);
        advance(s, Bs, decay[(size_t)r * HEADS + head], x * bf2f(dt[(size_t)r * HEADS + head]));
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < STATE; ++i) base[(size_t)i * HDIM] = s[i];
}

// commit (conv): the new history is the last 3 rows of concat(old 3 rows, the first keep input rows). One thread owns one
// (sequence, channel) and reads every source before it writes. keep <= 0 leaves the history untouched.
template <typename TX>
__global__ void __launch_bounds__(256) commit_conv_kernel(
        const TX* __restrict__ xin, __nv_bfloat16* __restrict__ conv_pool, const int* __restrict__ seq_start,
        const int* __restrict__ slot, const int* __restrict__ keep) {
    const int seq = blockIdx.y, ch = blockIdx.x * 256 + threadIdx.x;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = min(max(keep[seq], 0), len);
    if (n <= 0 || ch >= CONV) return;
    __nv_bfloat16* pool = conv_pool + (size_t)slot[seq] * KEEP * CONV + ch;
    __nv_bfloat16 nw[KEEP];
#pragma unroll
    for (int i = 0; i < KEEP; ++i) {
        const int e = n + i;  // index into concat(old 3 rows, the kept input rows)
        nw[i] = e < KEEP ? pool[(size_t)e * CONV] : to_bf(xin[(size_t)(begin + e - KEEP) * CONV + ch]);
    }
#pragma unroll
    for (int i = 0; i < KEEP; ++i) pool[(size_t)i * CONV] = nw[i];
}

// ---------------------------------------------------------------------------------------------------------------------
// commit of every Mamba layer in one launch (the replay of a decode step or a verify window): commit_state_kernel and
// commit_conv_kernel with the layer taken from blockIdx.z and its record, retained rows and pools from a device table the caller builds
// once. Each (layer, head, sequence) block runs the code of the single-layer kernel on its own layer's data, so the pools end with the
// bits of 40 separate launches, and the 40 layers fill the GPU together instead of one at a time.
// ---------------------------------------------------------------------------------------------------------------------
struct CommitLayer {
    const __nv_bfloat16* xbc;  // the layer's window record: post-conv rows, bf16 [rows][10240]
    const __nv_bfloat16* dt;   // bf16 [rows][128]
    const float* decay;        // f32 [rows][128]
    float* S;                  // the layer's S pool, all slots
    const float* xin;          // the layer's retained pre-conv xBC rows, f32 [rows][10240]
    __nv_bfloat16* conv;       // the layer's conv pool, all slots
};

__global__ void __launch_bounds__(HDIM) commit_state_all_kernel(const CommitLayer* __restrict__ layers, const int* __restrict__ seq_start,
                                                                const int* __restrict__ slot, const int* __restrict__ keep) {
    __shared__ float Bs[STATE];
    const CommitLayer L = layers[blockIdx.z];
    const int head = blockIdx.x, seq = blockIdx.y, j = threadIdx.x;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = min(max(keep[seq], 0), len);
    if (n <= 0) return;  // uniform across the block
    const int grp = head / HPG;
    float* base = L.S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD + j;
    float s[STATE];
#pragma unroll
    for (int i = 0; i < STATE; ++i) s[i] = base[(size_t)i * HDIM];
    for (int k = 0; k < n; ++k) {
        const int r = begin + k;
        const __nv_bfloat16* xr = L.xbc + (size_t)r * CONV;
        stage_b(xr, grp, Bs, j);
        __syncthreads();
        const float x = bf2f(xr[head * HDIM + j]);
        advance(s, Bs, L.decay[(size_t)r * HEADS + head], x * bf2f(L.dt[(size_t)r * HEADS + head]));
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < STATE; ++i) base[(size_t)i * HDIM] = s[i];
}

__global__ void __launch_bounds__(256) commit_conv_all_kernel(const CommitLayer* __restrict__ layers, const int* __restrict__ seq_start,
                                                              const int* __restrict__ slot, const int* __restrict__ keep) {
    const CommitLayer L = layers[blockIdx.z];
    const int seq = blockIdx.y, ch = blockIdx.x * 256 + threadIdx.x;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = min(max(keep[seq], 0), len);
    if (n <= 0 || ch >= CONV) return;
    __nv_bfloat16* pool = L.conv + (size_t)slot[seq] * KEEP * CONV + ch;
    __nv_bfloat16 nw[KEEP];
#pragma unroll
    for (int i = 0; i < KEEP; ++i) {
        const int e = n + i;  // index into concat(old 3 rows, the kept input rows)
        nw[i] = e < KEEP ? pool[(size_t)e * CONV] : __float2bfloat16_rn(L.xin[(size_t)(begin + e - KEEP) * CONV + ch]);
    }
#pragma unroll
    for (int i = 0; i < KEEP; ++i) pool[(size_t)i * CONV] = nw[i];
}

// ---------------------------------------------------------------------------------------------------------------------
// gated grouped RMSNorm: one block per (row, group of 1024 channels). v = y * silu(z) with z rounded to bf16 first, the mean
// square of v over the group in a fixed order, then out = (v * w) * rsqrt(mean + eps) rounded to bf16 (the fork's apply4 order).
// ---------------------------------------------------------------------------------------------------------------------
template <typename TZ>
__global__ void __launch_bounds__(256) gated_norm_kernel(
        const __nv_bfloat16* __restrict__ y, const TZ* __restrict__ z, const float* __restrict__ w, float eps,
        __nv_bfloat16* __restrict__ out) {
    __shared__ float red[8];
    const int row = blockIdx.x, grp = blockIdx.y, t = threadIdx.x;
    const size_t base = (size_t)row * INNER + (size_t)grp * NORM_GROUP;
    float v[4];
    float sq = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = t + 256 * k;
        const float zb = rd_bf(z[base + c]);
        v[k] = bf2f(y[base + c]) * (zb / (1.0f + expf(-zb)));
        sq = sq + v[k] * v[k];
    }
    sq = warp_sum(sq);
    if ((t & 31) == 0) red[t >> 5] = sq;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) total = total + red[i];
    const float inv = rsqrtf(total / (float)NORM_GROUP + eps);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = t + 256 * k;
        out[base + c] = __float2bfloat16_rn((v[k] * w[grp * NORM_GROUP + c]) * inv);
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Lane S, contract P2 at two TP ranks: the same kernels on a sub-range (a rank's head groups). Each is a verbatim copy of the kernel
// above it with ONE change: the channel / head / group index is offset by the range's start (`ch_base`, `head_lo`, `grp_lo`), so an
// element computed here is computed by the same code from the same inputs as by the full kernel. The full-geometry buffers are kept.
// ---------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) pre_conv_range_kernel(
        const float* __restrict__ xin, const __nv_bfloat16* __restrict__ conv_w, const __nv_bfloat16* __restrict__ conv_b,
        const __nv_bfloat16* __restrict__ conv_pool, const int* __restrict__ seq_start, const int* __restrict__ slot,
        int n_seq, __nv_bfloat16* __restrict__ out, int ch_base) {
    const int row = blockIdx.x, ch = ch_base + blockIdx.y * 256 + threadIdx.x;
    const int seq = find_seq(seq_start, n_seq, row);
    const int start = seq_start[seq], lt = row - start;
    const size_t pool = (size_t)slot[seq] * KEEP;
    float acc = bf2f(conv_b[ch]);
#pragma unroll
    for (int j = 0; j < CONV_TAPS_K; ++j) {
        const int idx = lt - 3 + j;  // window row x[t - 3 + j]
        const float x = idx >= 0 ? rbf(xin[(size_t)(start + idx) * CONV + ch])
                                 : bf2f(conv_pool[(pool + (size_t)(3 + idx)) * CONV + ch]);
        acc = __fmaf_rn(bf2f(conv_w[(size_t)ch * CONV_TAPS_K + j]), x, acc);
    }
    out[(size_t)row * CONV + ch] = __float2bfloat16_rn(acc / (1.0f + expf(-acc)));
}

__global__ void __launch_bounds__(4 * HDIM) window_commit4_range_kernel(
        const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ dt, const float* __restrict__ decay,
        float* __restrict__ S, const float* __restrict__ D, const int* __restrict__ seq_start, const int* __restrict__ slot,
        __nv_bfloat16* __restrict__ o, int head_lo) {
    // B and C of a row, transposed by thread: entry (i % 4) * kQ + i / 4 holds row i, so thread q reads its 32 rows i = q, q + 4, ..
    // as 8 float4 (kQ = 36, not 32: the four threads' runs start in different banks)
    constexpr int kQ = 36;
    __shared__ __align__(16) float Bs[kScanChunk][4 * kQ], Cs[kScanChunk][4 * kQ];
    __shared__ float xs[kScanChunk][HDIM], xdts[kScanChunk], decs[kScanChunk];
    const int head = head_lo + blockIdx.x, seq = blockIdx.y, t = threadIdx.x;
    const int j = t >> 2, q = t & 3;
    const int begin = seq_start[seq], end = seq_start[seq + 1];
    if (end <= begin) return;  // uniform across the block
    const int grp = head / HPG;
    float* base = S + (size_t)slot[seq] * S_SLOT + (size_t)head * S_HEAD + j;
    float s[STATE / 4];
#pragma unroll
    for (int k = 0; k < STATE / 4; ++k) s[k] = base[(size_t)(4 * k + q) * HDIM];
    const float d = D[head];
    for (int r0 = begin; r0 < end; r0 += kScanChunk) {
        const int nr = min(kScanChunk, end - r0);
        __syncthreads();  // the previous chunk's rows are read
        for (int e = t; e < nr * STATE; e += 4 * HDIM) {
            const int rr = e / STATE, i = e % STATE;
            const __nv_bfloat16* xr = xbc + (size_t)(r0 + rr) * CONV;
            Bs[rr][(i & 3) * kQ + (i >> 2)] = bf2f(xr[INNER + grp * STATE + i]);
            Cs[rr][(i & 3) * kQ + (i >> 2)] = bf2f(xr[INNER + BCW + grp * STATE + i]);
        }
        for (int e = t; e < nr * HDIM; e += 4 * HDIM) {
            const int rr = e / HDIM, c = e % HDIM;
            xs[rr][c] = bf2f(xbc[(size_t)(r0 + rr) * CONV + head * HDIM + c]);
        }
        if (t < nr) {
            xdts[t] = bf2f(dt[(size_t)(r0 + t) * HEADS + head]);
            decs[t] = decay[(size_t)(r0 + t) * HEADS + head];
        }
        __syncthreads();
        for (int rr = 0; rr < nr; ++rr) {
            const float x = xs[rr][j];
            const float xdt = x * xdts[rr];
            const float dc = decs[rr];
            float a = 0.0f;
            const float4* B4 = reinterpret_cast<const float4*>(&Bs[rr][q * kQ]);
            const float4* C4 = reinterpret_cast<const float4*>(&Cs[rr][q * kQ]);
#pragma unroll
            for (int m = 0; m < STATE / 16; ++m) {
                const float4 b4 = B4[m], c4 = C4[m];
                const float bb[4] = {b4.x, b4.y, b4.z, b4.w}, cc[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int k = 4 * m + u;  // state row i = 4 k + q
                    const float kept = s[k] * dc;
                    const float inj = bb[u] * xdt;
                    s[k] = kept + inj;
                    a = a + cc[u] * s[k];
                }
            }
            const float pair = a + __shfl_xor_sync(0xffffffffu, a, 1);     // q 0, 1: a0 + a1; q 2, 3: a2 + a3
            const float all = pair + __shfl_xor_sync(0xffffffffu, pair, 2);  // q 0: (a0 + a1) + (a2 + a3)
            if (q == 0) o[(size_t)(r0 + rr) * INNER + head * HDIM + j] = __float2bfloat16_rn(all + d * x);
        }
    }
#pragma unroll
    for (int k = 0; k < STATE / 4; ++k) base[(size_t)(4 * k + q) * HDIM] = s[k];
}

__global__ void __launch_bounds__(256) gated_norm_range_kernel(
        const __nv_bfloat16* __restrict__ y, const float* __restrict__ z, const float* __restrict__ w, float eps,
        __nv_bfloat16* __restrict__ out, int grp_lo) {
    __shared__ float red[8];
    const int row = blockIdx.x, grp = grp_lo + blockIdx.y, t = threadIdx.x;
    const size_t base = (size_t)row * INNER + (size_t)grp * NORM_GROUP;
    float v[4];
    float sq = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = t + 256 * k;
        const float zb = rbf(z[base + c]);
        v[k] = bf2f(y[base + c]) * (zb / (1.0f + expf(-zb)));
        sq = sq + v[k] * v[k];
    }
    sq = warp_sum(sq);
    if ((t & 31) == 0) red[t >> 5] = sq;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) total = total + red[i];
    const float inv = rsqrtf(total / (float)NORM_GROUP + eps);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = t + 256 * k;
        out[base + c] = __float2bfloat16_rn((v[k] * w[grp * NORM_GROUP + c]) * inv);
    }
}

__global__ void __launch_bounds__(256) commit_conv_range_kernel(
        const float* __restrict__ xin, __nv_bfloat16* __restrict__ conv_pool, const int* __restrict__ seq_start,
        const int* __restrict__ slot, const int* __restrict__ keep, int ch_base) {
    const int seq = blockIdx.y, ch = ch_base + blockIdx.x * 256 + threadIdx.x;
    const int begin = seq_start[seq], len = seq_start[seq + 1] - begin;
    const int n = min(max(keep[seq], 0), len);
    if (n <= 0 || ch >= CONV) return;
    __nv_bfloat16* pool = conv_pool + (size_t)slot[seq] * KEEP * CONV + ch;
    __nv_bfloat16 nw[KEEP];
#pragma unroll
    for (int i = 0; i < KEEP; ++i) {
        const int e = n + i;  // index into concat(old 3 rows, the kept input rows)
        nw[i] = e < KEEP ? pool[(size_t)e * CONV] : __float2bfloat16_rn(xin[(size_t)(begin + e - KEEP) * CONV + ch]);
    }
#pragma unroll
    for (int i = 0; i < KEEP; ++i) pool[(size_t)i * CONV] = nw[i];
}

}  // namespace

GB_EXPORT int gb_nssm_pre(const float* xin, const float* dt_raw, const void* conv_w, const void* conv_b, const float* a_log,
                          const float* dt_bias, const void* conv_pool, const int* seq_start, const int* slot, int n_seq, int rows,
                          void* xbc, void* dt, float* g, float* decay, void* stream) {
    if (rows <= 0 || n_seq <= 0) return 0;
    cudaStream_t st = gb_stream(stream);
    pre_conv_kernel<float><<<dim3(rows, CONV / 256), 256, 0, st>>>(xin, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_b,
                                                            (const __nv_bfloat16*)conv_pool, seq_start, slot, n_seq,
                                                            (__nv_bfloat16*)xbc);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    pre_dt_kernel<<<rows, HEADS, 0, st>>>(dt_raw, dt_bias, a_log, (__nv_bfloat16*)dt, g, decay);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_window(const void* xbc, const void* dt, const float* decay, const float* S, const float* D,
                             const int* seq_start, const int* slot, int n_seq, void* o, void* stream) {
    if (n_seq <= 0) return 0;
    window_kernel<<<dim3(HEADS, n_seq), HDIM, 0, gb_stream(stream)>>>((const __nv_bfloat16*)xbc, (const __nv_bfloat16*)dt, decay,
                                                                     S, D, seq_start, slot, (__nv_bfloat16*)o);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_window_commit(const void* xbc, const void* dt, const float* decay, float* S, const float* D,
                                    const int* seq_start, const int* slot, int n_seq, void* o, void* stream) {
    if (n_seq <= 0) return 0;
    window_commit_kernel<<<dim3(HEADS, n_seq), HDIM, 0, gb_stream(stream)>>>((const __nv_bfloat16*)xbc, (const __nv_bfloat16*)dt,
                                                                            decay, S, D, seq_start, slot, (__nv_bfloat16*)o);
    return GB_CHECK_LAUNCH();
}

// window_commit for a prefill slice on the four-threads-a-column kernel: the same bits (window_commit4_kernel), faster at many rows.
GB_EXPORT int gb_nssm_window_commit4(const void* xbc, const void* dt, const float* decay, float* S, const float* D,
                                     const int* seq_start, const int* slot, int n_seq, void* o, void* stream) {
    if (n_seq <= 0) return 0;
    static int x2 = -1;  // REXL3_NEMO_P2_SCAN_X2=0: one column a thread (read once)
    if (x2 < 0) {
        const char* v = getenv("REXL3_NEMO_P2_SCAN_X2");
        x2 = (v && v[0] == '0') ? 0 : 1;
    }
    if (x2)
        window_commit4x2_kernel<<<dim3(HEADS, n_seq), 2 * HDIM, 0, gb_stream(stream)>>>(
            (const __nv_bfloat16*)xbc, (const __nv_bfloat16*)dt, decay, S, D, seq_start, slot, (__nv_bfloat16*)o);
    else
        window_commit4_kernel<<<dim3(HEADS, n_seq), 4 * HDIM, 0, gb_stream(stream)>>>((const __nv_bfloat16*)xbc, (const __nv_bfloat16*)dt,
                                                                                    decay, S, D, seq_start, slot, (__nv_bfloat16*)o);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_window_commit_sel(const void* xbc, const void* dt, const float* decay, float* S, const float* D,
                                        const int* seq_start, const int* slot, const int* apply, int n_seq, void* o,
                                        void* stream) {
    if (n_seq <= 0) return 0;
    window_commit_sel_kernel<<<dim3(HEADS, n_seq), HDIM, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)xbc, (const __nv_bfloat16*)dt, decay, S, D, seq_start, slot, apply, (__nv_bfloat16*)o);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_window_lag(const void* xbc, const void* dt, const float* decay, float* S, const float* D, const int* seq_start,
                                 const int* slot, const int* apply, const int* lag, void* ring_xbc, void* ring_dt, float* ring_decay,
                                 int K, int n_seq, void* o, void* stream) {
    if (n_seq <= 0 || K <= 0) return 0;
    window_lag_kernel<<<dim3(HEADS, n_seq), HDIM, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)xbc, (const __nv_bfloat16*)dt, decay, S, D, seq_start, slot, apply, lag, (__nv_bfloat16*)ring_xbc,
        (__nv_bfloat16*)ring_dt, ring_decay, K, (__nv_bfloat16*)o);
    return GB_CHECK_LAUNCH();
}

// `layers` is a device array of n_layers LagLayer (four pointers each, in the struct's order).
GB_EXPORT int gb_nssm_fold_lag_all(const void* layers, int n_layers, const int* slots, const int* lags, int n, int K, void* stream) {
    if (n <= 0 || n_layers <= 0 || K <= 1) return 0;
    fold_lag_all_kernel<<<dim3(HEADS, n, n_layers), HDIM, 0, gb_stream(stream)>>>((const LagLayer*)layers, slots, lags, K);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_commit_state(const void* xbc, const void* dt, const float* decay, float* S, const int* seq_start,
                                   const int* slot, const int* keep, int n_seq, void* stream) {
    if (n_seq <= 0) return 0;
    commit_state_kernel<<<dim3(HEADS, n_seq), HDIM, 0, gb_stream(stream)>>>((const __nv_bfloat16*)xbc, (const __nv_bfloat16*)dt,
                                                                           decay, S, seq_start, slot, keep);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_commit_conv(const float* xin, void* conv_pool, const int* seq_start, const int* slot, const int* keep,
                                  int n_seq, void* stream) {
    if (n_seq <= 0) return 0;
    commit_conv_kernel<float><<<dim3(CONV / 256, n_seq), 256, 0, gb_stream(stream)>>>(xin, (__nv_bfloat16*)conv_pool, seq_start, slot,
                                                                              keep);
    return GB_CHECK_LAUNCH();
}

// `layers` is a device array of n_layers CommitLayer (six pointers each, in the struct's order).
GB_EXPORT int gb_nssm_commit_all(const void* layers, int n_layers, const int* seq_start, const int* slot, const int* keep, int n_seq,
                                 void* stream) {
    if (n_seq <= 0 || n_layers <= 0) return 0;
    const CommitLayer* L = (const CommitLayer*)layers;
    cudaStream_t st = gb_stream(stream);
    commit_state_all_kernel<<<dim3(HEADS, n_seq, n_layers), HDIM, 0, st>>>(L, seq_start, slot, keep);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    commit_conv_all_kernel<<<dim3(CONV / 256, n_seq, n_layers), 256, 0, st>>>(L, seq_start, slot, keep);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_gated_norm(const void* y, const float* z, const float* w, float eps, int rows, void* out, void* stream) {
    if (rows <= 0) return 0;
    gated_norm_kernel<float><<<dim3(rows, GROUPS), 256, 0, gb_stream(stream)>>>((const __nv_bfloat16*)y, z, w, eps,
                                                                        (__nv_bfloat16*)out);
    return GB_CHECK_LAUNCH();
}

// The P2 prefill variants (lane S5) of gb_nssm_pre, gb_nssm_commit_conv and gb_nssm_gated_norm whose pre-conv xBC rows (`xin`) and z are
// bf16 (the fused in_proj split stores them on the bf16 grid they are read on): the same results bit for bit.
GB_EXPORT int gb_nssm_pre_b(const void* xin, const float* dt_raw, const void* conv_w, const void* conv_b, const float* a_log,
                            const float* dt_bias, const void* conv_pool, const int* seq_start, const int* slot, int n_seq, int rows,
                            void* xbc, void* dt, float* g, float* decay, void* stream) {
    if (rows <= 0 || n_seq <= 0) return 0;
    cudaStream_t st = gb_stream(stream);
    pre_conv_kernel<__nv_bfloat16><<<dim3(rows, CONV / 256), 256, 0, st>>>(
        (const __nv_bfloat16*)xin, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_b, (const __nv_bfloat16*)conv_pool, seq_start,
        slot, n_seq, (__nv_bfloat16*)xbc);
    int rc = GB_CHECK_LAUNCH();
    if (rc) return rc;
    pre_dt_kernel<<<rows, HEADS, 0, st>>>(dt_raw, dt_bias, a_log, (__nv_bfloat16*)dt, g, decay);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_commit_conv_b(const void* xin, void* conv_pool, const int* seq_start, const int* slot, const int* keep, int n_seq,
                                    void* stream) {
    if (n_seq <= 0) return 0;
    commit_conv_kernel<__nv_bfloat16><<<dim3(CONV / 256, n_seq), 256, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)xin, (__nv_bfloat16*)conv_pool, seq_start, slot, keep);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_gated_norm_b(const void* y, const void* z, const float* w, float eps, int rows, void* out, void* stream) {
    if (rows <= 0) return 0;
    gated_norm_kernel<__nv_bfloat16><<<dim3(rows, GROUPS), 256, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)y, (const __nv_bfloat16*)z, w, eps, (__nv_bfloat16*)out);
    return GB_CHECK_LAUNCH();
}

// ---- lane S: the sub-range launchers (P2 at two TP ranks; see the range kernels) ----
// pre over up to three channel ranges (base, count; count % 256 == 0) and dt for every head.
GB_EXPORT int gb_nssm_pre_range(const float* xin, const float* dt_raw, const void* conv_w, const void* conv_b, const float* a_log,
                                const float* dt_bias, const void* conv_pool, const int* seq_start, const int* slot, int n_seq, int rows,
                                void* xbc, void* dt, float* g, float* decay, int c0, int n0, int c1, int n1, int c2, int n2, void* stream) {
    if (rows <= 0 || n_seq <= 0) return 0;
    cudaStream_t st = gb_stream(stream);
    const int cs[3] = {c0, c1, c2}, ns[3] = {n0, n1, n2};
    for (int i = 0; i < 3; ++i) {
        if (ns[i] <= 0) continue;
        if (ns[i] % 256 != 0 || cs[i] < 0 || cs[i] + ns[i] > CONV) return (int)cudaErrorInvalidValue;
        pre_conv_range_kernel<<<dim3(rows, ns[i] / 256), 256, 0, st>>>(xin, (const __nv_bfloat16*)conv_w, (const __nv_bfloat16*)conv_b,
                                                                      (const __nv_bfloat16*)conv_pool, seq_start, slot, n_seq,
                                                                      (__nv_bfloat16*)xbc, cs[i]);
        int rc = GB_CHECK_LAUNCH();
        if (rc) return rc;
    }
    pre_dt_kernel<<<rows, HEADS, 0, st>>>(dt_raw, dt_bias, a_log, (__nv_bfloat16*)dt, g, decay);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_window_commit4_range(const void* xbc, const void* dt, const float* decay, float* S, const float* D,
                                           const int* seq_start, const int* slot, int n_seq, void* o, int head_lo, int n_heads,
                                           void* stream) {
    if (n_seq <= 0) return 0;
    if (head_lo < 0 || n_heads < 1 || head_lo + n_heads > HEADS) return (int)cudaErrorInvalidValue;
    window_commit4_range_kernel<<<dim3(n_heads, n_seq), 4 * HDIM, 0, gb_stream(stream)>>>(
        (const __nv_bfloat16*)xbc, (const __nv_bfloat16*)dt, decay, S, D, seq_start, slot, (__nv_bfloat16*)o, head_lo);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_gated_norm_range(const void* y, const float* z, const float* w, float eps, int rows, void* out, int grp_lo, int n_grp,
                                       void* stream) {
    if (rows <= 0) return 0;
    if (grp_lo < 0 || n_grp < 1 || grp_lo + n_grp > GROUPS) return (int)cudaErrorInvalidValue;
    gated_norm_range_kernel<<<dim3(rows, n_grp), 256, 0, gb_stream(stream)>>>((const __nv_bfloat16*)y, z, w, eps, (__nv_bfloat16*)out,
                                                                             grp_lo);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_nssm_commit_conv_range(const float* xin, void* conv_pool, const int* seq_start, const int* slot, const int* keep,
                                        int n_seq, int c0, int n0, int c1, int n1, int c2, int n2, void* stream) {
    if (n_seq <= 0) return 0;
    const int cs[3] = {c0, c1, c2}, ns[3] = {n0, n1, n2};
    for (int i = 0; i < 3; ++i) {
        if (ns[i] <= 0) continue;
        if (ns[i] % 256 != 0 || cs[i] < 0 || cs[i] + ns[i] > CONV) return (int)cudaErrorInvalidValue;
        commit_conv_range_kernel<<<dim3(ns[i] / 256, n_seq), 256, 0, gb_stream(stream)>>>(xin, (__nv_bfloat16*)conv_pool, seq_start, slot,
                                                                                         keep, cs[i]);
        int rc = GB_CHECK_LAUNCH();
        if (rc) return rc;
    }
    return 0;
}
