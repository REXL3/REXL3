// gb-nvcc: --fmad=false
// Kernels of the per-tensor quantizer core (crates/quant/src/core). C ABI, raw device pointers, asynchronous on the given
// stream, 0 or a cudaError_t on return. Matrices are row-major fp32 with a leading dimension (elements per row).
//
// Reductions accumulate in fp64 in a fixed order (one block per output, a fixed thread count, a shared-memory tree), so
// they are deterministic; the caller rounds the fp64 result. Built with --fmad=false: every fp32 operation rounds on its
// own, as one PyTorch element-wise kernel per operation does.

#include <cuda_runtime.h>
#include <cstdint>

#define RQ_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

constexpr int EW_THREADS = 256;
constexpr int RED_THREADS = 256;

inline int blocks_for(long long n) {
    long long b = (n + EW_THREADS - 1) / EW_THREADS;
    return (int)(b < 1 ? 1 : (b > 65535LL * 32 ? 65535LL * 32 : b));
}

// The trellis position p of a 16 x 16 tile reads row r, column c (crates/exl3 tile_positions; the MMA B-fragment order).
__device__ __forceinline__ void pos_rc(int p, int* r, int* c) {
    const int lane = p >> 3, j = p & 7;
    *r = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1);
    *c = (lane >> 2) + 8 * (j >> 2);
}

__device__ __forceinline__ double block_sum(double v, double* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = RED_THREADS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    return sh[0];
}

// mode 0: x[i, j] *= v[j]; 1: x[i, j] *= v[i]; 2: x[i, j] /= v[j]; 3: x[i, j] /= v[i]
__global__ void scale_kernel(float* x, long long rows, long long cols, long long ld, const float* v, int mode) {
    const long long n = rows * cols;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / cols, j = t % cols;
        float* p = x + i * ld + j;
        switch (mode) {
            case 0: *p = *p * v[j]; break;
            case 1: *p = *p * v[i]; break;
            case 2: *p = *p / v[j]; break;
            default: *p = *p / v[i]; break;
        }
    }
}

// op 0: out = a - b; 1: out = a + b; 2: out = a * b
__global__ void binary_kernel(float* out, long long ldo, const float* a, long long lda, const float* b, long long ldb,
                              long long rows, long long cols, int op) {
    const long long n = rows * cols;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / cols, j = t % cols;
        const float x = a[i * lda + j], y = b[i * ldb + j];
        out[i * ldo + j] = op == 0 ? x - y : op == 1 ? x + y : x * y;
    }
}

// out = x * a (op 0) or x + a (op 1), element-wise over n contiguous values
__global__ void scalar_kernel(float* out, const float* x, long long n, float a, int op) {
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n; t += (long long)gridDim.x * blockDim.x)
        out[t] = op == 0 ? x[t] * a : x[t] + a;
}

__global__ void copy2d_kernel(float* dst, long long ldd, const float* src, long long lds, long long rows, long long cols) {
    const long long n = rows * cols;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / cols, j = t % cols;
        dst[i * ldd + j] = src[i * lds + j];
    }
}

// dst (cols x rows) = src (rows x cols)^T
__global__ void transpose_kernel(float* dst, const float* src, long long rows, long long cols) {
    const long long n = rows * cols;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / cols, j = t % cols;
        dst[j * rows + i] = src[i * cols + j];
    }
}

// dst row-major (n x n) = the lower triangle of the column-major src, zeros above
__global__ void tril_from_colmajor_kernel(float* dst, const float* src, long long n) {
    const long long m = n * n;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / n, j = t % n;
        dst[t] = j <= i ? src[i + j * n] : 0.0f;
    }
}

// x[i, i] += v (op 0), x[i, i] -= v (op 1), x[i, i] = v (op 2)
__global__ void diag_kernel(float* x, long long n, long long ld, float v, int op) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        float* p = x + i * ld + i;
        *p = op == 0 ? *p + v : op == 1 ? *p - v : v;
    }
}

__global__ void zero_diag_blocks_kernel(float* x, long long n, int block) {
    const long long m = n * block;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / block, j = (i / block) * block + t % block;
        x[i * n + j] = 0.0f;
    }
}

// out[b, r + 16 c] = x[16 b + r, 16 b + c]: the diagonal 16 x 16 blocks of an (n x n) matrix, each column-major
__global__ void diag_blocks_cm_kernel(float* out, const float* x, long long n) {
    const long long m = n * 16;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x) {
        const long long b = t >> 8, e = t & 255, r = e & 15, c = e >> 4;
        out[t] = x[(16 * b + r) * n + 16 * b + c];
    }
}

__global__ void identity_blocks_kernel(float* out, long long blocks) {
    const long long m = blocks * 256;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x) {
        const long long e = t & 255;
        out[t] = (e & 15) == (e >> 4) ? 1.0f : 0.0f;
    }
}

// tiles[t, p] = rows[r, 16 t + c] for the 16 rows of a block row (ld elements apart), (r, c) the position of p
__global__ void rows_to_tiles_kernel(float* tiles, const float* rows, long long ld, long long n_tiles) {
    const long long m = n_tiles * 256;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x) {
        int r, c;
        pos_rc((int)(t & 255), &r, &c);
        tiles[t] = rows[r * ld + (t >> 8) * 16 + c];
    }
}

__global__ void tiles_to_rows_kernel(float* rows, long long ld, const float* tiles, long long n_tiles) {
    const long long m = n_tiles * 256;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x) {
        int r, c;
        pos_rc((int)(t & 255), &r, &c);
        rows[r * ld + (t >> 8) * 16 + c] = tiles[t];
    }
}

// out[s, p] = w[(16 a_s + r) * n + 16 c_s + col] (sampled tiles of a (k, n) matrix, trellis order)
__global__ void gather_tiles_kernel(float* out, const float* w, long long n, const int* ta, const int* tc, long long count) {
    const long long m = count * 256;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x) {
        int r, c;
        pos_rc((int)(t & 255), &r, &c);
        const long long s = t >> 8;
        out[t] = w[(16LL * ta[s] + r) * n + 16LL * tc[s] + c];
    }
}

// dst[i, :] = src[idx[i], :] (rows) or dst[i, j] = src[idx[i], idx[j]] (sym)
__global__ void gather_rows_kernel(float* dst, const float* src, const int* idx, long long rows, long long cols) {
    const long long m = rows * cols;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / cols, j = t % cols;
        dst[t] = src[(long long)idx[i] * cols + j];
    }
}

__global__ void gather_sym_kernel(float* dst, const float* src, const int* idx, long long n) {
    const long long m = n * n;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / n, j = t % n;
        dst[t] = src[(long long)idx[i] * n + idx[j]];
    }
}

// state block rows: dst[i] = src[idx[i]] for blocks of `width` int16
__global__ void gather_blocks_i16_kernel(int16_t* dst, const int16_t* src, const int* idx, long long blocks, long long width) {
    const long long m = blocks * width;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x)
        dst[t] = src[(long long)idx[t / width] * width + t % width];
}

// ---- fp64 reductions -------------------------------------------------------------------------------------------

// kind 0: out[o] = sum (double) x^2; 1: out[o] = sum (double)(a * b) (the product rounded to fp32 first); 2: sum x.
// Output o covers `len` elements at base o * ostride, `step` apart.
__global__ void reduce_kernel(double* out, const float* a, const float* b, long long len, long long step, long long ostride,
                              int kind) {
    __shared__ double sh[RED_THREADS];
    const long long base = blockIdx.x * ostride;
    double acc = 0.0;
    for (long long i = threadIdx.x; i < len; i += RED_THREADS) {
        const long long at = base + i * step;
        if (kind == 0) {
            const double v = (double)a[at];
            acc += v * v;
        } else if (kind == 1) {
            acc += (double)(a[at] * b[at]);
        } else {
            acc += (double)a[at];
        }
    }
    const double s = block_sum(acc, sh);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// a single sum over a long contiguous range: per-block partials over fixed chunks, then a second pass
__global__ void reduce_partial_kernel(double* part, const float* a, const float* b, long long n, long long chunk, int kind) {
    __shared__ double sh[RED_THREADS];
    const long long lo = blockIdx.x * chunk;
    const long long hi = lo + chunk < n ? lo + chunk : n;
    double acc = 0.0;
    for (long long i = lo + threadIdx.x; i < hi; i += RED_THREADS) {
        if (kind == 0) {
            const double v = (double)a[i];
            acc += v * v;
        } else if (kind == 1) {
            acc += (double)(a[i] * b[i]);
        } else {
            acc += (double)a[i];
        }
    }
    const double s = block_sum(acc, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = s;
}

__global__ void reduce_final_kernel(double* out, const double* part, long long n) {
    __shared__ double sh[RED_THREADS];
    double acc = 0.0;
    for (long long i = threadIdx.x; i < n; i += RED_THREADS) acc += part[i];
    const double s = block_sum(acc, sh);
    if (threadIdx.x == 0) out[0] = s;
}

// per 16 x 16 tile of a (k, n) matrix: sum (double) x^2
__global__ void tile_sumsq_kernel(double* out, const float* w, long long n) {
    __shared__ double sh[RED_THREADS];
    const long long tiles_n = n / 16;
    const long long a = blockIdx.x / tiles_n, c = blockIdx.x % tiles_n;
    const int r = threadIdx.x >> 4, col = threadIdx.x & 15;
    const double v = (double)w[(16 * a + r) * n + 16 * c + col];
    const double s = block_sum(v * v, sh);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// g-scale error terms: e = dec * inv_s - t (fp32), out[set] = sum (double) e^2 over `len` values of each set
__global__ void gs_err_kernel(double* out, const float* dec, const float* t, const float* inv_s, long long len) {
    __shared__ double sh[RED_THREADS];
    const long long base = blockIdx.x * len;
    const float inv = inv_s[blockIdx.x];
    double acc = 0.0;
    for (long long i = threadIdx.x; i < len; i += RED_THREADS) {
        const float e = dec[base + i] * inv - t[i];
        acc += (double)e * (double)e;
    }
    const double s = block_sum(acc, sh);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// out[s, :] = x[:] * a[s] for `sets` scaled copies of n values
__global__ void scaled_copies_kernel(float* out, const float* x, const float* a, long long n, long long sets) {
    const long long m = n * sets;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < m; t += (long long)gridDim.x * blockDim.x)
        out[t] = x[t % n] * a[t / n];
}

}  // namespace

#define RQ_LAUNCH(kernel, n, ...)                                                                      \
    do {                                                                                               \
        kernel<<<blocks_for(n), EW_THREADS, 0, (cudaStream_t)stream>>>(__VA_ARGS__);                   \
        return (int)cudaGetLastError();                                                                \
    } while (0)

RQ_EXPORT int rq_scale(float* x, long long rows, long long cols, long long ld, const float* v, int mode, void* stream) {
    RQ_LAUNCH(scale_kernel, rows * cols, x, rows, cols, ld, v, mode);
}

RQ_EXPORT int rq_binary(float* out, long long ldo, const float* a, long long lda, const float* b, long long ldb,
                        long long rows, long long cols, int op, void* stream) {
    RQ_LAUNCH(binary_kernel, rows * cols, out, ldo, a, lda, b, ldb, rows, cols, op);
}

RQ_EXPORT int rq_scalar(float* out, const float* x, long long n, float a, int op, void* stream) {
    RQ_LAUNCH(scalar_kernel, n, out, x, n, a, op);
}

RQ_EXPORT int rq_copy2d(float* dst, long long ldd, const float* src, long long lds, long long rows, long long cols,
                        void* stream) {
    RQ_LAUNCH(copy2d_kernel, rows * cols, dst, ldd, src, lds, rows, cols);
}

RQ_EXPORT int rq_transpose(float* dst, const float* src, long long rows, long long cols, void* stream) {
    RQ_LAUNCH(transpose_kernel, rows * cols, dst, src, rows, cols);
}

RQ_EXPORT int rq_tril_from_colmajor(float* dst, const float* src, long long n, void* stream) {
    RQ_LAUNCH(tril_from_colmajor_kernel, n * n, dst, src, n);
}

RQ_EXPORT int rq_diag(float* x, long long n, long long ld, float v, int op, void* stream) {
    RQ_LAUNCH(diag_kernel, n, x, n, ld, v, op);
}

RQ_EXPORT int rq_zero_diag_blocks(float* x, long long n, int block, void* stream) {
    RQ_LAUNCH(zero_diag_blocks_kernel, n * block, x, n, block);
}

RQ_EXPORT int rq_diag_blocks_colmajor(float* out, const float* x, long long n, void* stream) {
    RQ_LAUNCH(diag_blocks_cm_kernel, n * 16, out, x, n);
}

RQ_EXPORT int rq_identity_blocks(float* out, long long blocks, void* stream) {
    RQ_LAUNCH(identity_blocks_kernel, blocks * 256, out, blocks);
}

RQ_EXPORT int rq_rows_to_tiles(float* tiles, const float* rows, long long ld, long long n_tiles, void* stream) {
    RQ_LAUNCH(rows_to_tiles_kernel, n_tiles * 256, tiles, rows, ld, n_tiles);
}

RQ_EXPORT int rq_tiles_to_rows(float* rows, long long ld, const float* tiles, long long n_tiles, void* stream) {
    RQ_LAUNCH(tiles_to_rows_kernel, n_tiles * 256, rows, ld, tiles, n_tiles);
}

RQ_EXPORT int rq_gather_tiles(float* out, const float* w, long long n, const int* ta, const int* tc, long long count,
                              void* stream) {
    RQ_LAUNCH(gather_tiles_kernel, count * 256, out, w, n, ta, tc, count);
}

RQ_EXPORT int rq_gather_rows(float* dst, const float* src, const int* idx, long long rows, long long cols, void* stream) {
    RQ_LAUNCH(gather_rows_kernel, rows * cols, dst, src, idx, rows, cols);
}

RQ_EXPORT int rq_gather_sym(float* dst, const float* src, const int* idx, long long n, void* stream) {
    RQ_LAUNCH(gather_sym_kernel, n * n, dst, src, idx, n);
}

RQ_EXPORT int rq_gather_blocks_i16(int16_t* dst, const int16_t* src, const int* idx, long long blocks, long long width,
                                   void* stream) {
    RQ_LAUNCH(gather_blocks_i16_kernel, blocks * width, dst, src, idx, blocks, width);
}

RQ_EXPORT int rq_scaled_copies(float* out, const float* x, const float* a, long long n, long long sets, void* stream) {
    RQ_LAUNCH(scaled_copies_kernel, n * sets, out, x, a, n, sets);
}

RQ_EXPORT int rq_reduce(double* out, const float* a, const float* b, long long outputs, long long len, long long step,
                        long long ostride, int kind, void* stream) {
    if (outputs <= 0) return 0;
    reduce_kernel<<<(unsigned)outputs, RED_THREADS, 0, (cudaStream_t)stream>>>(out, a, b, len, step, ostride, kind);
    return (int)cudaGetLastError();
}

// out[0] = the fp64 sum over n contiguous values; part holds at least rq_reduce_all_parts(n) doubles
RQ_EXPORT long long rq_reduce_all_parts(long long n) { return (n + (1 << 16) - 1) >> 16; }

RQ_EXPORT int rq_reduce_all(double* out, double* part, const float* a, const float* b, long long n, int kind, void* stream) {
    const long long chunk = 1 << 16;
    const long long parts = rq_reduce_all_parts(n);
    if (parts <= 0) return (int)cudaMemsetAsync(out, 0, sizeof(double), (cudaStream_t)stream);
    reduce_partial_kernel<<<(unsigned)parts, RED_THREADS, 0, (cudaStream_t)stream>>>(part, a, b, n, chunk, kind);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    reduce_final_kernel<<<1, RED_THREADS, 0, (cudaStream_t)stream>>>(out, part, parts);
    return (int)cudaGetLastError();
}

RQ_EXPORT int rq_tile_sumsq(double* out, const float* w, long long k, long long n, void* stream) {
    const long long tiles = (k / 16) * (n / 16);
    if (tiles <= 0) return 0;
    tile_sumsq_kernel<<<(unsigned)tiles, RED_THREADS, 0, (cudaStream_t)stream>>>(out, w, n);
    return (int)cudaGetLastError();
}

RQ_EXPORT int rq_gs_err(double* out, const float* dec, const float* t, const float* inv_s, long long sets, long long len,
                        void* stream) {
    if (sets <= 0) return 0;
    gs_err_kernel<<<(unsigned)sets, RED_THREADS, 0, (cudaStream_t)stream>>>(out, dec, t, inv_s, len);
    return (int)cudaGetLastError();
}
