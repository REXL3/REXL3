// Kernels of the two-rank tensor-parallel Nemotron engine (docs/NEMOTRON_TP.md, "TP-exact").
//
// The rule of that design: every output element is computed by one rank with the single-node kernel, and the other rank receives it by a
// sum over buffers of DISJOINT SUPPORT (a rank holds its own elements and the additive identity elsewhere; with two ranks x + id == x,
// so the sum is a gather and no arithmetic touches a value). The kernels here build those buffers. They do no arithmetic: a bit-exact
// copy and the identity, so there is nothing for an order of operations to change, and every element depends on its own (row, column)
// only (row invariance by construction: a row's bits do not depend on how many rows share the call).
//
// The identity is -0.0f (0x80000000), not +0.0f: IEEE addition gives x + (-0.0) == x for EVERY x, the sign of a zero included
// (-0.0 + -0.0 = -0.0, +0.0 + -0.0 = +0.0), where x + (+0.0) turns a -0.0 into +0.0. A -0.0 is a real value of these buffers (an exact
// zero times a negative scale), and the step digest hashes raw bits, so the gathered row is bit-identical to the single-node one only
// with -0.0 as the filler.
//
//   nemo_tp_scatter_cols   dst[r, c] = src[r, c - off] for c in [off, off + n_local), -0.0f for every other column of the row
//   nemo_tp_scatter_cols4  the same on float4, when the widths, the offset and the base pointers are 16-byte aligned

#include <stdint.h>

#include "common.cuh"
#include <cuda_bf16.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxGridX = 1024;  // column blocks of one row (grid-stride beyond it)

// One element a thread: dst[r, c] = src[r, c - off] inside [off, off + nl), -0.0f (the additive identity) outside. Blocks (x) stride over the columns of row
// blockIdx.y.
__global__ void __launch_bounds__(kThreads) nemo_tp_scatter_cols(const float* __restrict__ src, float* __restrict__ dst, int nl, int nt,
                                                                int off) {
    const size_t r = blockIdx.y;
    const float* s = src + r * (size_t)nl;
    float* d = dst + r * (size_t)nt;
    for (int c = blockIdx.x * kThreads + threadIdx.x; c < nt; c += gridDim.x * kThreads) {
        const int lc = c - off;
        d[c] = (lc >= 0 && lc < nl) ? s[lc] : -0.0f;
    }
}

// One float4 a thread; nl, nt and off are multiples of 4, so a float4 of dst lies wholly inside or wholly outside the source range.
__global__ void __launch_bounds__(kThreads) nemo_tp_scatter_cols4(const float4* __restrict__ src, float4* __restrict__ dst, int nl4,
                                                                 int nt4, int off4) {
    const size_t r = blockIdx.y;
    const float4* s = src + r * (size_t)nl4;
    float4* d = dst + r * (size_t)nt4;
    for (int c = blockIdx.x * kThreads + threadIdx.x; c < nt4; c += gridDim.x * kThreads) {
        const int lc = c - off4;
        d[c] = (lc >= 0 && lc < nl4) ? s[lc] : make_float4(-0.0f, -0.0f, -0.0f, -0.0f);
    }
}

// ---- the packed column exchange (lane S, contract P2 prefill only; docs/NEMOTRON_PREFILL_SPEED.md) ----------------------------------
// A rank's columns [off, off + nl) of a row, packed for the peer: the columns below `bend` as bf16 (round to nearest even, the rounding
// every consumer of those columns applies itself: the in_proj z and xBC), the others as f32 (bit-exact). Row layout: bf16[nb] then
// f32[nl - nb], nb = clamp(bend - off, 0, nl) (even, so the f32 part is 4-byte aligned).
__global__ void __launch_bounds__(kThreads) nemo_tp_pack_cols(const float* __restrict__ src, int nl, int off, int bend, int row_bytes,
                                                             unsigned char* __restrict__ out) {
    const size_t r = blockIdx.y;
    const int nb = min(max(bend - off, 0), nl);
    const float* s = src + r * (size_t)nl;
    unsigned char* o = out + r * (size_t)row_bytes;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(o);
    float* of = reinterpret_cast<float*>(o + (size_t)nb * 2);
    for (int c = blockIdx.x * kThreads + threadIdx.x; c < nl; c += gridDim.x * kThreads) {
        const float v = s[c];
        if (c < nb) ob[c] = __float2bfloat16_rn(v);
        else of[c - nb] = v;
    }
}

// dst [rows, nt] f32: this rank's columns [off, off + nl) from `own` (f32, bit-exact), the peer's [poff, poff + pnl) from its packed row
// (bf16 widened exactly to f32 below `bend`, f32 above), every other column -0.0f (there is none at two ranks).
__global__ void __launch_bounds__(kThreads) nemo_tp_place_cols(const float* __restrict__ own, int nl, int off,
                                                              const unsigned char* __restrict__ peer, int pnl, int poff, int prow_bytes,
                                                              int bend, float* __restrict__ dst, int nt) {
    const size_t r = blockIdx.y;
    const int pnb = min(max(bend - poff, 0), pnl);
    const float* s = own + r * (size_t)nl;
    const unsigned char* p = peer + r * (size_t)prow_bytes;
    const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(p);
    const float* pf = reinterpret_cast<const float*>(p + (size_t)pnb * 2);
    float* d = dst + r * (size_t)nt;
    for (int c = blockIdx.x * kThreads + threadIdx.x; c < nt; c += gridDim.x * kThreads) {
        const int lc = c - off, pc = c - poff;
        float v = -0.0f;
        if (lc >= 0 && lc < nl) v = s[lc];
        else if (pc >= 0 && pc < pnl) v = pc < pnb ? __bfloat162float(pb[pc]) : pf[pc - pnb];
        d[c] = v;
    }
}

// The in_proj split of `split_inproj_kernel` (nemotron_glue.cu) fused with nemo_tp_place_cols (lane S, P2 under TP): every column c of a
// row is first gathered as nemo_tp_place_cols would write it, then written where split_inproj_kernel writes it: z (c < 8,192) rounded
// by round_bf16 (the glue's own rounding, copied bit for bit), xBC (c < 18,432) and dt as they are. The same bits in z, xBC and dt as
// place_cols followed by split_inproj, without the [rows, 18,560] round trip.
__device__ __forceinline__ float tp_round_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return __uint_as_float(0x7fc00000u);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(u);
}

constexpr int kZCols = 8192, kXbcCols = 10240, kDtCols = 128, kInprojCols = kZCols + kXbcCols + kDtCols;

__global__ void __launch_bounds__(kThreads) nemo_tp_place_split(const float* __restrict__ own, int nl, int off,
                                                               const unsigned char* __restrict__ peer, int pnl, int poff, int prow_bytes,
                                                               int bend, float* __restrict__ z, float* __restrict__ xbc,
                                                               float* __restrict__ dt) {
    const size_t r = blockIdx.y;
    const int pnb = min(max(bend - poff, 0), pnl);
    const float* s = own + r * (size_t)nl;
    const unsigned char* p = peer + r * (size_t)prow_bytes;
    const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(p);
    const float* pf = reinterpret_cast<const float*>(p + (size_t)pnb * 2);
    for (int c = blockIdx.x * kThreads + threadIdx.x; c < kInprojCols; c += gridDim.x * kThreads) {
        const int lc = c - off, pc = c - poff;
        float v = -0.0f;
        if (lc >= 0 && lc < nl) v = s[lc];
        else if (pc >= 0 && pc < pnl) v = pc < pnb ? __bfloat162float(pb[pc]) : pf[pc - pnb];
        if (c < kZCols) z[r * kZCols + c] = tp_round_bf16(v);
        else if (c < kZCols + kXbcCols) xbc[r * kXbcCols + (c - kZCols)] = v;
        else dt[r * kDtCols + (c - kZCols - kXbcCols)] = v;
    }
}

// The head-group in_proj split (lane S, P2 at two ranks): `src` [rows, nz + nx + nbc + nbc + 128] is the rank's merged in_proj output
// (z of its heads | x of its heads | B of its groups | C of its groups | dt of every head). z goes to z[:, zo .. zo + nz) rounded by the
// glue's round_bf16 (as split_inproj does), x to xbc[:, xo ..), B to xbc[:, 8192 + bo ..), C to xbc[:, 9216 + bo ..), dt to dt[:, 0 .. 128)
// unchanged: split_inproj's writes for these columns, from the same values.
__global__ void __launch_bounds__(kThreads) nemo_tp_split_hg(const float* __restrict__ src, int nz, int nx, int nbc, int zo, int xo, int bo,
                                                            float* __restrict__ z, float* __restrict__ xbc, float* __restrict__ dt) {
    const size_t r = blockIdx.y;
    const int w = nz + nx + 2 * nbc + kDtCols;
    const float* s = src + r * (size_t)w;
    for (int c = blockIdx.x * kThreads + threadIdx.x; c < w; c += gridDim.x * kThreads) {
        const float v = s[c];
        if (c < nz) z[r * kZCols + zo + c] = tp_round_bf16(v);
        else if (c < nz + nx) xbc[r * kXbcCols + xo + (c - nz)] = v;
        else if (c < nz + nx + nbc) xbc[r * kXbcCols + 8192 + bo + (c - nz - nx)] = v;
        else if (c < nz + nx + 2 * nbc) xbc[r * kXbcCols + 9216 + bo + (c - nz - nx - nbc)] = v;
        else dt[r * kDtCols + (c - nz - nx - 2 * nbc)] = v;
    }
}

// ---- the routed-pair gather (lane S, contract P2): the MoE `y` rows of the pairs a rank's experts own, packed in (expert, list order) ----
// Block e (an expert of [lo, hi)): its cnt[e] pairs (lists[e * lcap + i], ascending pair ids) go to rows off(e) + i of the packed buffer,
// off(e) = the pairs of the range's experts before e. `to_packed`: y -> packed (the sender); else packed -> y (the receiver, the peer's
// range). A bit-exact copy of N-float rows (N % 4 == 0).
__global__ void __launch_bounds__(kThreads) nemo_tp_pairs(const int* __restrict__ cnt, const int* __restrict__ lists, int lcap, int lo,
                                                         int N, float* __restrict__ y, float* __restrict__ packed, int to_packed) {
    const int e = lo + blockIdx.x;
    __shared__ int s_off;
    if (threadIdx.x == 0) {
        int o = 0;
        for (int j = lo; j < e; ++j) o += cnt[j];
        s_off = o;
    }
    __syncthreads();
    const int n = cnt[e], off = s_off, n4 = N / 4;
    for (int i = blockIdx.y; i < n; i += gridDim.y) {
        const size_t pair = (size_t)lists[(size_t)e * lcap + i];
        float4* yr = reinterpret_cast<float4*>(y + pair * N);
        float4* pr = reinterpret_cast<float4*>(packed + (size_t)(off + i) * N);
        for (int q = threadIdx.x; q < n4; q += kThreads) {
            if (to_packed) pr[q] = yr[q];
            else yr[q] = pr[q];
        }
    }
}

// map[pair] = the row of the packed buffer that carries it, for the pairs of the experts [lo, hi) (the peer's), in the order
// nemo_tp_pairs packs them; the caller fills map with -1 first (own pairs: read from y).
__global__ void __launch_bounds__(kThreads) nemo_tp_pair_map(const int* __restrict__ cnt, const int* __restrict__ lists, int lcap, int lo,
                                                            int* __restrict__ map) {
    const int e = lo + blockIdx.x;
    __shared__ int s_off;
    if (threadIdx.x == 0) {
        int o = 0;
        for (int j = lo; j < e; ++j) o += cnt[j];
        s_off = o;
    }
    __syncthreads();
    const int n = cnt[e];
    for (int i = threadIdx.x; i < n; i += kThreads) map[lists[(size_t)e * lcap + i]] = s_off + i;
}

// nm_combine (nemotron_moe.cu) with each slot's y row read from `y` (map < 0) or from the received rows (`recv[map]`): the same
// operations in the same order on the same values, so the same bits as the combine of the gathered `y`.
constexpr int kTopK = 22;
__global__ void __launch_bounds__(128) nemo_tp_combine_gather(const float* __restrict__ y, const float* __restrict__ recv,
                                                             const int* __restrict__ map, const float* __restrict__ wts, int N,
                                                             float* __restrict__ routed) {
    const int t = blockIdx.y, n = blockIdx.x * 128 + threadIdx.x;
    float acc = 0.f;
    for (int s = 0; s < kTopK; ++s) {
        const size_t p = (size_t)t * kTopK + s;
        const int m = map[p];
        const float v = m < 0 ? y[p * N + n] : recv[(size_t)m * N + n];
        acc = __fadd_rn(acc, __fmul_rn(wts[p], v));
    }
    routed[(size_t)t * N + n] = acc;
}

}  // namespace

// map[pair] for the peer's pairs (see nemo_tp_pair_map); the caller fills map with -1 (0xFF bytes) first.
GB_EXPORT int gb_nemo_tp_pair_map(const void* cnt, const void* lists, int lcap, int lo, int hi, void* map, void* stream) {
    if (lo < 0 || hi <= lo || lcap < 1 || !cnt || !lists || !map) return (int)cudaErrorInvalidValue;
    nemo_tp_pair_map<<<(unsigned)(hi - lo), kThreads, 0, gb_stream(stream)>>>(static_cast<const int*>(cnt), static_cast<const int*>(lists),
                                                                             lcap, lo, static_cast<int*>(map));
    return GB_CHECK_LAUNCH();
}

// routed [rows, N] = sum over the 22 slots, in slot order from +0.0, of wts * (y or the received row): nm_combine's arithmetic.
GB_EXPORT int gb_nemo_tp_combine_gather(const void* y, const void* recv, const void* map, const void* wts, int rows, int N, void* routed,
                                        void* stream) {
    if (rows < 1 || N % 128 != 0 || !y || !recv || !map || !wts || !routed) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(N / 128), (unsigned)rows);
    nemo_tp_combine_gather<<<grid, 128, 0, gb_stream(stream)>>>(static_cast<const float*>(y), static_cast<const float*>(recv),
                                                               static_cast<const int*>(map), static_cast<const float*>(wts), N,
                                                               static_cast<float*>(routed));
    return GB_CHECK_LAUNCH();
}

// The routed pairs of the experts [lo, hi): to_packed 1 copies their y rows into `packed` (expert order, then list order), 0 copies
// `packed` back into their y rows. cnt [512] and lists [512 * lcap] are the prefill path's bucket outputs.
GB_EXPORT int gb_nemo_tp_pairs(const void* cnt, const void* lists, int lcap, int lo, int hi, int N, void* y, void* packed, int to_packed,
                               void* stream) {
    if (lo < 0 || hi <= lo || N < 4 || N % 4 != 0 || lcap < 1 || !cnt || !lists || !y || !packed) return (int)cudaErrorInvalidValue;
    const dim3 grid((unsigned)(hi - lo), 16u);
    nemo_tp_pairs<<<grid, kThreads, 0, gb_stream(stream)>>>(static_cast<const int*>(cnt), static_cast<const int*>(lists), lcap, lo, N,
                                                           static_cast<float*>(y), static_cast<float*>(packed), to_packed);
    return GB_CHECK_LAUNCH();
}

// z [rows, 8192], xbc [rows, 10240], dt [rows, 128] f32 straight from this rank's columns and the peer's packed rows (nl + pnl == 18,560).
GB_EXPORT int gb_nemo_tp_place_split(const void* own, int nl, int off, const void* peer, int pnl, int poff, int bend, void* z, void* xbc,
                                     void* dt, int rows, void* stream) {
    const int pnb = min(max(bend - poff, 0), pnl);
    if (rows < 1 || rows > 65535 || nl < 1 || pnl < 1 || nl + pnl != kInprojCols || off < 0 || poff < 0 || pnb % 2 != 0 || !own || !peer ||
        !z || !xbc || !dt)
        return (int)cudaErrorInvalidValue;
    const int prow = pnb * 2 + (pnl - pnb) * 4;
    const int bx = (kInprojCols + kThreads - 1) / kThreads;
    const dim3 grid((unsigned)(bx < kMaxGridX ? bx : kMaxGridX), (unsigned)rows);
    nemo_tp_place_split<<<grid, kThreads, 0, gb_stream(stream)>>>(static_cast<const float*>(own), nl, off,
                                                                 static_cast<const unsigned char*>(peer), pnl, poff, prow, bend,
                                                                 static_cast<float*>(z), static_cast<float*>(xbc), static_cast<float*>(dt));
    return GB_CHECK_LAUNCH();
}

// The head-group split (see nemo_tp_split_hg).
GB_EXPORT int gb_nemo_tp_split_hg(const void* src, int rows, int nz, int nx, int nbc, int zo, int xo, int bo, void* z, void* xbc, void* dt,
                                  void* stream) {
    if (rows < 1 || rows > 65535 || nz < 0 || nx < 0 || nbc < 0 || zo < 0 || zo + nz > kZCols || xo < 0 || xo + nx > 8192 || bo < 0 ||
        bo + nbc > 1024 || !src || !z || !xbc || !dt)
        return (int)cudaErrorInvalidValue;
    const int w = nz + nx + 2 * nbc + kDtCols;
    const int bx = (w + kThreads - 1) / kThreads;
    const dim3 grid((unsigned)(bx < kMaxGridX ? bx : kMaxGridX), (unsigned)rows);
    nemo_tp_split_hg<<<grid, kThreads, 0, gb_stream(stream)>>>(static_cast<const float*>(src), nz, nx, nbc, zo, xo, bo, static_cast<float*>(z),
                                                              static_cast<float*>(xbc), static_cast<float*>(dt));
    return GB_CHECK_LAUNCH();
}

// Bytes of one packed row of nl columns starting at off (bf16 below bend, f32 from it).
GB_EXPORT int gb_nemo_tp_packed_row_bytes(int nl, int off, int bend) {
    const int nb = min(max(bend - off, 0), nl);
    return nb * 2 + (nl - nb) * 4;
}

// Pack rows x [off, off + nl) of src (f32 [rows, nl]) into out ([rows, packed_row_bytes]). nb must be even.
GB_EXPORT int gb_nemo_tp_pack_cols(const void* src, int rows, int nl, int off, int bend, void* out, void* stream) {
    const int nb = min(max(bend - off, 0), nl);
    if (rows < 1 || rows > 65535 || nl < 1 || off < 0 || nb % 2 != 0 || !src || !out) return (int)cudaErrorInvalidValue;
    const int row_bytes = nb * 2 + (nl - nb) * 4;
    const int bx = (nl + kThreads - 1) / kThreads;
    const dim3 grid((unsigned)(bx < kMaxGridX ? bx : kMaxGridX), (unsigned)rows);
    nemo_tp_pack_cols<<<grid, kThreads, 0, gb_stream(stream)>>>(static_cast<const float*>(src), nl, off, bend, row_bytes,
                                                               static_cast<unsigned char*>(out));
    return GB_CHECK_LAUNCH();
}

// dst [rows, nt]: own columns from own (f32 [rows, nl] at off), the peer's from its packed rows (pnl columns at poff).
GB_EXPORT int gb_nemo_tp_place_cols(const void* own, int nl, int off, const void* peer, int pnl, int poff, int bend, void* dst, int rows,
                                    int nt, void* stream) {
    const int pnb = min(max(bend - poff, 0), pnl);
    if (rows < 1 || rows > 65535 || nl < 1 || pnl < 1 || off < 0 || poff < 0 || off + nl > nt || poff + pnl > nt || pnb % 2 != 0 ||
        !own || !peer || !dst)
        return (int)cudaErrorInvalidValue;
    const int prow = pnb * 2 + (pnl - pnb) * 4;
    const int bx = (nt + kThreads - 1) / kThreads;
    const dim3 grid((unsigned)(bx < kMaxGridX ? bx : kMaxGridX), (unsigned)rows);
    nemo_tp_place_cols<<<grid, kThreads, 0, gb_stream(stream)>>>(static_cast<const float*>(own), nl, off,
                                                                static_cast<const unsigned char*>(peer), pnl, poff, prow, bend,
                                                                static_cast<float*>(dst), nt);
    return GB_CHECK_LAUNCH();
}

// dst [rows, n_total] f32: the columns [col_off, col_off + n_local) are src [rows, n_local], every other element is -0.0f. src and dst
// must not overlap. Returns cudaErrorInvalidValue (without launching) for an empty or out-of-range call.
GB_EXPORT int gb_nemo_tp_scatter_cols_f32(const void* src, void* dst, int rows, int n_local, int n_total, int col_off, void* stream) {
    if (rows < 1 || rows > 65535 || n_local < 0 || n_total < 1 || col_off < 0 || col_off > n_total - n_local || !dst ||
        (n_local > 0 && !src))
        return (int)cudaErrorInvalidValue;
    cudaStream_t st = gb_stream(stream);
    const bool vec = n_local % 4 == 0 && n_total % 4 == 0 && col_off % 4 == 0 && ((uintptr_t)dst & 15) == 0 &&
                     (n_local == 0 || ((uintptr_t)src & 15) == 0);
    const int width = vec ? n_total / 4 : n_total;
    const int bx = (width + kThreads - 1) / kThreads;
    const dim3 grid((unsigned)(bx < kMaxGridX ? bx : kMaxGridX), (unsigned)rows);
    if (vec) {
        nemo_tp_scatter_cols4<<<grid, kThreads, 0, st>>>(static_cast<const float4*>(src), static_cast<float4*>(dst), n_local / 4,
                                                          n_total / 4, col_off / 4);
    } else {
        nemo_tp_scatter_cols<<<grid, kThreads, 0, st>>>(static_cast<const float*>(src), static_cast<float*>(dst), n_local, n_total,
                                                        col_off);
    }
    return GB_CHECK_LAUNCH();
}
