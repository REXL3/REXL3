// gb-nvcc: --fmad=false
// GLM-5.3-Flash prefill under contract P2 (Lane P step (d); docs/GLM_PREFILL_P2.md section 12): the MLA absorb and expand of a prompt
// chunk's rows on tensor cores. The Rust side is crates/cuda/src/glm_p2_mla.rs.
//
//   absorb  q_lat[r][h][c]     = sum_{d < 256} q[r][h * 256 + d] * W_UK[h][d][c]       (c < 512)
//   expand  out[r][h * 256 + n] = sum_{c < 512} o_lat[r][h][c] * W_UVt[h][c][n]        (n < 256)
//
// P1 computes every output as one fp32 __fmaf_rn chain over the contraction index on CUDA cores (csrc/mla.cu, skinny_kernel). Under P2
// the same products run on mma.m16n8k16 (fp16 operands, fp32 accumulate) in the fold128 K order of the dense P2 GEMM
// (csrc/glm_p2_dense.cu): per 128-k block b = 0, 1, .. in order, the chain of its 8 k16 MMAs from +0.0, then tot = tot + part_b (one IEEE
// fp32 add, round to nearest; tot starts at +0.0); one round to fp16 at the end. Nothing depends on the rows of the call, the M tile, the
// grid or the heads of the call (no split K, no stream K, no atomics); rows past the call are zero-filled in shared memory and never
// stored, and an mma row's result depends on that row's A fragment and the B fragment only. So a row's bits depend on that row's operands
// and the weights alone (P2 (a), (b)); fp16 operands, fp32 accumulation (P2 (c)); deterministic (P2 (d)).
//
// Layouts (fp16 only: GLM binds kv_b as fp16; P1's): x [rows][xs] with head h's K operands at h * K; w [heads][K][N] (N contiguous);
// y [rows][heads * N] with head h's N outputs at h * N. One CTA = 64 rows x 128 columns of one head, 8 warps of 32 x 32 (2 x 4), BK = 32
// in a 3-stage cp.async ring, padded shared rows (A 80 bytes, B 272 bytes: the eight 16-byte rows of every ldmatrix hit distinct banks),
// B fragments by ldmatrix.trans from the N-contiguous weight rows.
#include "common.cuh"

namespace {

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int STAGES = 3;
constexpr int THREADS = 256;
constexpr int APAD = BK + 8;   // halves per A shared row
constexpr int BPAD = BN + 8;   // halves per B shared row
constexpr int FOLD_K = 128;    // the fold block

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// 16 bytes global -> shared, zero-filled without reading the source's bytes when !valid (src must still be a mapped address)
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
// d = a b + d (fp16 operands, fp32 accumulate)
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int K, int N>
__global__ void __launch_bounds__(THREADS) p2_mla_kernel(const half* __restrict__ x, int64_t xs, const half* __restrict__ w,
                                                         half* __restrict__ y, int64_t ys, int rows) {
    static_assert(K % FOLD_K == 0 && FOLD_K % BK == 0 && N % BN == 0, "tiling");
    __shared__ __align__(16) half sa[STAGES][BM * APAD];
    __shared__ __align__(16) half sb[STAGES][BK * BPAD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2;  // rows 32 wm .. 32 wm + 31 of the tile
    const int wn = warp & 3;   // columns 32 wn .. 32 wn + 31
    // the M tiles of one (head, N tile) are consecutive CTAs: they share the weight tile through L2
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, h = blockIdx.z;
    const half* xh = x + (int64_t)h * K;
    const half* wh = w + (int64_t)h * K * N;

    auto load = [&](int stage, int kt) {
        const int k0 = kt * BK;
        {
            // A: BM rows x BK halves = 256 pieces of 16 bytes, one a thread; a row past the call is zeros
            const int r = tid >> 2, c = tid & 3, gr = m0 + r;
            const bool v = gr < rows;
            cp16(&sa[stage][r * APAD + c * 8], xh + (int64_t)(v ? gr : 0) * xs + k0 + c * 8, v);
        }
#pragma unroll
        for (int i = tid; i < BK * (BN / 8); i += THREADS) {
            // B: BK rows x BN halves = 512 pieces, two a thread
            const int kr = i / (BN / 8), c = i % (BN / 8);
            cp16(&sb[stage][kr * BPAD + c * 8], wh + (int64_t)(k0 + kr) * N + n0 + c * 8, true);
        }
    };

    float part[2][4][4], tot[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int nj = 0; nj < 4; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) part[mi][nj][e] = 0.0f, tot[mi][nj][e] = 0.0f;

    constexpr int KT = K / BK;
    constexpr int FOLD = FOLD_K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        wait_group<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < KT) load(nk % STAGES, nk);
            commit();
        }
        const half* A = sa[kt % STAGES];
        const half* B = sb[kt % STAGES];
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) ldmatrix4(af[mi], A + (wm * 32 + mi * 16 + (lane & 15)) * APAD + kk + (lane >> 4) * 8);
            uint32_t bf[4][2];
#pragma unroll
            for (int nj = 0; nj < 4; nj += 2) {
                // matrices (k 0-7, n 0-7), (k 8-15, n 0-7), (k 0-7, n 8-15), (k 8-15, n 8-15): b0 / b1 of two n8 tiles
                const int q = lane >> 3, j = lane & 7;
                uint32_t r[4];
                ldmatrix4t(r, B + (kk + (q & 1) * 8 + j) * BPAD + wn * 32 + nj * 8 + (q >> 1) * 8);
                bf[nj][0] = r[0];
                bf[nj][1] = r[1];
                bf[nj + 1][0] = r[2];
                bf[nj + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int nj = 0; nj < 4; ++nj) mma(part[mi][nj], af[mi], bf[nj][0], bf[nj][1]);
        }
        if ((kt + 1) % FOLD == 0) {
            // the end of a 128-k block: fold its chain into the total, restart the chain from +0.0
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int nj = 0; nj < 4; ++nj)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[mi][nj][e] = __fadd_rn(tot[mi][nj][e], part[mi][nj][e]);
                        part[mi][nj][e] = 0.0f;
                    }
        }
    }
    wait_group<0>();

    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = m0 + wm * 32 + mi * 16 + g;
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
            const int col = h * N + n0 + wn * 32 + nj * 8 + 2 * t;
            if (r0 < rows)
                *reinterpret_cast<half2*>(y + (int64_t)r0 * ys + col) = __floats2half2_rn(tot[mi][nj][0], tot[mi][nj][1]);
            if (r0 + 8 < rows)
                *reinterpret_cast<half2*>(y + (int64_t)(r0 + 8) * ys + col) = __floats2half2_rn(tot[mi][nj][2], tot[mi][nj][3]);
        }
    }
}

template <int K, int N>
int launch(const void* x, int64_t xs, const void* w, void* y, int rows, int nh, cudaStream_t s) {
    if (rows < 0 || nh < 1 || xs < (int64_t)nh * K || (xs & 7)) return (int)cudaErrorInvalidValue;
    if (rows == 0) return 0;
    const dim3 grid((rows + BM - 1) / BM, N / BN, nh);
    p2_mla_kernel<K, N><<<grid, THREADS, 0, s>>>((const half*)x, xs, (const half*)w, (half*)y, (int64_t)nh * N, rows);
    return GB_CHECK_LAUNCH();
}

}  // namespace

// q [rows][q_row_stride] fp16 (head h at h * 256), w_uk [nh][256][512], q_lat [rows][nh][512].
GB_EXPORT int gb_glm_p2_mla_absorb(const void* q, int64_t q_row_stride, const void* w_uk, void* q_lat, int rows, int nh,
                                   void* stream) {
    return launch<256, 512>(q, q_row_stride, w_uk, q_lat, rows, nh, (cudaStream_t)stream);
}

// o_lat [rows][nh][512] fp16, w_uvt [nh][512][256], out [rows][nh * 256].
GB_EXPORT int gb_glm_p2_mla_expand(const void* o_lat, const void* w_uvt, void* out, int rows, int nh, void* stream) {
    return launch<512, 256>(o_lat, (int64_t)nh * 512, w_uvt, out, rows, nh, (cudaStream_t)stream);
}
