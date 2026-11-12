// gb-nvcc: --fmad=false
// Kernels of the quantizer's teacher forward and capture (crates/quant/src/teacher, capture.rs). C ABI, raw device
// pointers, asynchronous on the given stream, 0 or a cudaError_t on return.
//
// Element-wise fp32 operations follow PyTorch's formulas one rounding per operation (built with --fmad=false).
// Reductions other than GEMMs are fp64 pairwise trees: zero-padded to a power of two, level by level
// a[i] += a[i + s] for i a multiple of 2s, so element i of a level is a[2i] + a[2i+1] of the level below; the result
// rounds to fp32 once where the forward keeps fp32.

#include <cuda_runtime.h>
#include <cstdint>

#define TQ_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

constexpr int EW = 256;

inline int grid_for(long long n) {
    long long b = (n + EW - 1) / EW;
    return (int)(b < 1 ? 1 : (b > 65535LL * 32 ? 65535LL * 32 : b));
}

inline int pow2_at_least(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// In-place pairwise tree over a[0, p) (p a power of two) by the whole block; the sum ends in a[0].
__device__ void tree_block(double* a, int p) {
    for (int s = 1; s < p; s <<= 1) {
        for (int i = threadIdx.x * 2 * s; i < p; i += blockDim.x * 2 * s) a[i] += a[i + s];
        __syncthreads();
    }
}

// Sequential form of the same tree for one thread: leaves in order through a binary counter (exact for any count; the
// caller feeds zero leaves up to a power of two). depth <= 16.
struct TreeSeq {
    double v[17];
    int lvl[17];
    int n = 0;
    __device__ void push(double x) {
        int l = 0;
        while (n > 0 && lvl[n - 1] == l) {
            x = v[n - 1] + x;
            --n;
            ++l;
        }
        v[n] = x;
        lvl[n] = l;
        ++n;
    }
    __device__ double total() const {
        double t = v[n - 1];
        for (int i = n - 2; i >= 0; --i) t = v[i] + t;
        return t;
    }
};

__global__ void bf16_to_f32_kernel(float* out, const uint16_t* in, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = __uint_as_float(((uint32_t)in[i]) << 16);
}

__global__ void embed_kernel(float* out, const uint16_t* table, const int32_t* ids, long long n, long long H) {
    const long long total = n * H;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
        const long long r = t / H, j = t % H;
        out[t] = __uint_as_float(((uint32_t)table[(long long)ids[r] * H + j]) << 16);
    }
}

// mode 0: (x * rsqrt(mean(x^2) + eps)) * (1 + w)          (Qwen3.5 RMSNorm, weight stored as w - 1)
// mode 1: (w * (x * rsqrt(mean(x^2) + eps))) * silu(gate)  (the GDN output norm)
// Row r reads x + r * ldx, gate + r * ldg, writes out + r * ldo. One block per row.
__global__ void rmsnorm_kernel(float* out, long long ldo, const float* x, long long ldx, const float* w, const float* gate,
                               long long ldg, int n, int p, float eps, int mode) {
    extern __shared__ double sh[];
    const long long r = blockIdx.x;
    const float* xr = x + r * ldx;
    for (int i = threadIdx.x; i < p; i += blockDim.x) {
        const double v = i < n ? (double)xr[i] : 0.0;
        sh[i] = v * v;
    }
    __syncthreads();
    tree_block(sh, p);
    const float m = (float)(sh[0] * (1.0 / (double)n));
    const float inv = rsqrtf(m + eps);
    float* o = out + r * ldo;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xn = xr[i] * inv;
        if (mode == 0) {
            o[i] = xn * (1.0f + w[i]);
        } else {
            const float t = w[i] * xn;
            o[i] = t * silu_f(gate[r * ldg + i]);
        }
    }
}

// Causal depthwise convolution (kernel K, zero left padding) and SiLU over (B, T, C) rows; w is (C, K).
__global__ void conv_silu_kernel(float* out, const float* x, const float* w, int B, int T, int C, int K) {
    const long long total = (long long)B * T * C;
    for (long long id = blockIdx.x * (long long)blockDim.x + threadIdx.x; id < total; id += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(id % C);
        const long long bt = id / C;
        const int t = (int)(bt % T);
        TreeSeq tr;
        int np = 1;
        while (np < K) np <<= 1;
        for (int j = 0; j < np; ++j) {
            double p = 0.0;
            if (j < K) {
                const int tt = t + j - (K - 1);
                const float xv = tt >= 0 ? x[(bt - t + tt) * C + c] : 0.0f;
                p = (double)xv * (double)w[c * K + j];
            }
            tr.push(p);
        }
        out[id] = silu_f((float)tr.total());
    }
}

// beta = sigmoid(b); g = -exp(a_log) * softplus(a + dt_bias); rows x heads.
__global__ void gdn_gates_kernel(float* g, float* beta, const float* a, const float* b, const float* a_log,
                                 const float* dt_bias, long long rows, int heads) {
    const long long total = rows * heads;
    for (long long id = blockIdx.x * (long long)blockDim.x + threadIdx.x; id < total; id += (long long)gridDim.x * blockDim.x) {
        const int h = (int)(id % heads);
        beta[id] = sigmoid_f(b[id]);
        const float z = a[id] + dt_bias[h];
        const float sp = z > 20.0f ? z : log1pf(expf(z));
        const float na = -expf(a_log[h]);
        g[id] = na * sp;
    }
}

// L2 normalization in fp64 of rows (bt, h): x + bt * ld + off + h * n, then times `scale`; out (bt, h, n) fp64.
__global__ void l2norm_f64_kernel(double* out, const float* x, long long ld, long long off, int heads, int n, int p,
                                  double eps, double scale) {
    extern __shared__ double sh[];
    const long long r = blockIdx.x;
    const long long bt = r / heads;
    const int h = (int)(r % heads);
    const float* xr = x + bt * ld + off + (long long)h * n;
    for (int i = threadIdx.x; i < p; i += blockDim.x) {
        const double v = i < n ? (double)xr[i] : 0.0;
        sh[i] = v * v;
    }
    __syncthreads();
    tree_block(sh, p);
    const double inv = rsqrt(sh[0] + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double v = (double)xr[i] * inv;
        if (scale != 1.0) v = v * scale;
        out[r * n + i] = v;
    }
}

// Gated delta rule, token by token in fp64. Block (b, value head hv, column group of CG): thread i owns row i of the
// state's CG columns. q, k: (B, T, Hk, dk) fp64 (normalized; q scaled); v rows of x (B, T, ldx) at v_off;
// g, beta: (B, T, Hv); o: (B, T, Hv, dv) fp32.
template <int CG>
__global__ void gdn_f64_kernel(float* o, const double* q, const double* k, const float* x, long long ldx, long long v_off,
                               const float* g, const float* beta, int T, int Hk, int Hv, int dk, int dv) {
    extern __shared__ double sh[];
    double* prod = sh;               // [CG][dk]
    double* red = sh + CG * dk;      // [CG]
    const int groups = dv / CG;
    const int cg = blockIdx.x % groups;
    const int hv = (blockIdx.x / groups) % Hv;
    const int b = blockIdx.x / (groups * Hv);
    const int hk = hv / (Hv / Hk);
    const int i = threadIdx.x;
    double S[CG];
#pragma unroll
    for (int j = 0; j < CG; ++j) S[j] = 0.0;
    for (int t = 0; t < T; ++t) {
        const long long bt = (long long)b * T + t;
        const double decay = exp((double)g[bt * Hv + hv]);
        const double bet = (double)beta[bt * Hv + hv];
        const double ki = k[(bt * Hk + hk) * dk + i];
        const double qi = q[(bt * Hk + hk) * dk + i];
#pragma unroll
        for (int j = 0; j < CG; ++j) {
            S[j] = S[j] * decay;
            prod[j * dk + i] = S[j] * ki;
        }
        __syncthreads();
        for (int s = 1; s < dk; s <<= 1) {
            const int per = dk / (2 * s);
            for (int w = i; w < CG * per; w += blockDim.x) {
                const int j = w / per, ii = (w % per) * 2 * s;
                prod[j * dk + ii] += prod[j * dk + ii + s];
            }
            __syncthreads();
        }
        if (i < CG) {
            const int col = cg * CG + i;
            const double vj = (double)x[bt * ldx + v_off + (long long)hv * dv + col];
            red[i] = (vj - prod[i * dk]) * bet;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < CG; ++j) {
            S[j] = S[j] + ki * red[j];
            prod[j * dk + i] = S[j] * qi;
        }
        __syncthreads();
        for (int s = 1; s < dk; s <<= 1) {
            const int per = dk / (2 * s);
            for (int w = i; w < CG * per; w += blockDim.x) {
                const int j = w / per, ii = (w % per) * 2 * s;
                prod[j * dk + ii] += prod[j * dk + ii + s];
            }
            __syncthreads();
        }
        if (i < CG) o[(bt * Hv + hv) * dv + cg * CG + i] = (float)prod[i * dk];
        __syncthreads();
    }
}

// out[r, i] = silu(a[r * lda + i]) * b[r * ldb + i]
__global__ void silu_mul_kernel(float* out, long long ldo, const float* a, long long lda, const float* b, long long ldb,
                                long long rows, int n) {
    const long long total = rows * n;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
        const long long r = t / n;
        const int i = (int)(t % n);
        out[r * ldo + i] = silu_f(a[r * lda + i]) * b[r * ldb + i];
    }
}

// out = a * sigmoid(g), n contiguous values.
__global__ void mul_sigmoid_kernel(float* out, const float* a, const float* g, long long n) {
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n; t += (long long)gridDim.x * blockDim.x)
        out[t] = a[t] * sigmoid_f(g[t]);
}

// out[r, j] = moe[r, j] + sigmoid(gl[r]) * sh[r, j]
__global__ void shared_combine_kernel(float* out, const float* moe, const float* shv, const float* gl, long long rows, int H) {
    const long long total = rows * H;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
        const long long r = t / H;
        out[t] = moe[t] + sigmoid_f(gl[r]) * shv[t];
    }
}

// Router: probs = softmax(logits) in fp64 (rounded to fp32), the top k by value (ties: lower index), weights
// top / sum(top) with the sum in fp64 over the top k in descending order. One block of E threads per row.
__global__ void route_kernel(int32_t* top_i, float* top_w, uint32_t* counts, const float* logits, int E, int K, int p) {
    extern __shared__ double sh[];
    float* probs = (float*)(sh + p);
    const long long r = blockIdx.x;
    const float* l = logits + r * E;
    // max (order-free)
    if (threadIdx.x == 0) {
        double m = -INFINITY;
        for (int e = 0; e < E; ++e) m = fmax(m, (double)l[e]);
        sh[p + (E + 1) / 2 + 1] = m;   // scratch past the probs
    }
    __syncthreads();
    const double m = sh[p + (E + 1) / 2 + 1];
    double* ex = (double*)(probs + E + (E & 1)) + 2;
    for (int e = threadIdx.x; e < p; e += blockDim.x) {
        const double v = e < E ? exp((double)l[e] - m) : 0.0;
        sh[e] = v;
        if (e < E) ex[e] = v;
    }
    __syncthreads();
    tree_block(sh, p);
    const double den = sh[0];
    for (int e = threadIdx.x; e < E; e += blockDim.x) probs[e] = (float)(ex[e] / den);
    __syncthreads();
    if (threadIdx.x == 0) {
        int sel[16];
        float val[16];
        for (int s = 0; s < K; ++s) {
            int best = -1;
            float bv = 0.0f;
            for (int e = 0; e < E; ++e) {
                bool taken = false;
                for (int u = 0; u < s; ++u) taken |= sel[u] == e;
                if (taken) continue;
                if (best < 0 || probs[e] > bv) {
                    best = e;
                    bv = probs[e];
                }
            }
            sel[s] = best;
            val[s] = bv;
        }
        int kp = 1;
        while (kp < K) kp <<= 1;
        TreeSeq tr;
        for (int s = 0; s < kp; ++s) tr.push(s < K ? (double)val[s] : 0.0);
        const float sum = (float)tr.total();
        for (int s = 0; s < K; ++s) {
            top_i[r * K + s] = sel[s];
            top_w[r * K + s] = val[s] / sum;
            if (counts) atomicAdd(&counts[sel[s]], 1u);
        }
    }
}

// out[tok[r], j] = out[tok[r], j] + y[r, j] * w[r]
__global__ void scale_add_rows_kernel(float* out, const float* y, const int32_t* tok, const float* w, long long rows, int H) {
    const long long total = rows * H;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
        const long long r = t / H;
        const int j = (int)(t % H);
        float* o = out + (long long)tok[r] * H + j;
        *o = *o + y[t] * w[r];
    }
}

// dst[r, :] = src[idx[r], :]
__global__ void gather_rows_kernel(float* dst, const float* src, const int32_t* idx, long long rows, int cols) {
    const long long total = rows * cols;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
        const long long r = t / cols;
        const int j = (int)(t % cols);
        dst[t] = src[(long long)idx[r] * cols + j];
    }
}

// In place over (rows = B * T * H) heads of hd values, the first rd rotated: x' = x cos + rotate_half(x) sin with
// angle (float)t * inv_freq[i mod rd/2], rotate_half(x)_i = -x_{i + rd/2} (i < rd/2), x_{i - rd/2} (else).
__global__ void rope_kernel(float* x, long long rows, int T, int H, int hd, int rd, const float* inv_freq) {
    const int half = rd / 2;
    const long long total = rows * half;
    for (long long id = blockIdx.x * (long long)blockDim.x + threadIdx.x; id < total; id += (long long)gridDim.x * blockDim.x) {
        const long long r = id / half;
        const int i = (int)(id % half);
        const int t = (int)((r / H) % T);
        const float f = (float)t * inv_freq[i];
        const float c = cosf(f), s = sinf(f);
        float* p = x + r * hd;
        const float a = p[i], b = p[i + half];
        p[i] = a * c + (-b) * s;
        p[i + half] = b * c + a * s;
    }
}

// Causal attention in fp64 for one (b, query head, t) per block; q (B, T, Hq, d), k, v (B, T, Hkv, d) fp32;
// o (B, T, Hq, d) fp32. Scores and weights in shared memory (T_max doubles each).
__global__ void attn_f64_kernel(float* o, const float* q, const float* k, const float* v, int T, int Hq, int Hkv, int d,
                                double scale) {
    extern __shared__ double sh[];
    const int t = blockIdx.x % T;
    const int h = (blockIdx.x / T) % Hq;
    const int b = blockIdx.x / (T * Hq);
    const int hk = h / (Hq / Hkv);
    const int n = t + 1;
    const int p = [&] { int x = 1; while (x < n) x <<= 1; return x; }();
    double* sc = sh;          // [p]
    double* tr = sh + p;      // [p]
    const float* qr = q + (((long long)b * T + t) * Hq + h) * d;
    int dp = 1;
    while (dp < d) dp <<= 1;
    for (int j = threadIdx.x; j < p; j += blockDim.x) {
        if (j < n) {
            const float* kr = k + (((long long)b * T + j) * Hkv + hk) * d;
            TreeSeq ts;
            for (int u = 0; u < dp; ++u) ts.push(u < d ? (double)qr[u] * (double)kr[u] : 0.0);
            sc[j] = ts.total() * scale;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double m = -INFINITY;
        for (int j = 0; j < n; ++j) m = fmax(m, sc[j]);
        tr[0] = m;
    }
    __syncthreads();
    const double m = tr[0];
    __syncthreads();
    for (int j = threadIdx.x; j < p; j += blockDim.x) {
        const double e = j < n ? exp(sc[j] - m) : 0.0;
        sc[j] = e;
        tr[j] = e;
    }
    __syncthreads();
    tree_block(tr, p);
    const double den = tr[0];
    __syncthreads();
    for (int j = threadIdx.x; j < p; j += blockDim.x) sc[j] = sc[j] / den;
    __syncthreads();
    for (int u = threadIdx.x; u < d; u += blockDim.x) {
        TreeSeq ts;
        for (int j = 0; j < p; ++j) {
            const int jj = j < n ? j : n - 1;
            const double vv = (double)v[(((long long)b * T + jj) * Hkv + hk) * d + u];
            ts.push(sc[j] * vv);
        }
        o[(((long long)b * T + t) * Hq + h) * d + u] = (float)ts.total();
    }
}

__global__ void f32_to_f64_kernel(double* out, const float* x, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = (double)x[i];
}

__global__ void add_f64_kernel(double* a, const double* b, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        a[i] = a[i] + b[i];
}

__global__ void f64_to_f32_kernel(float* out, const double* x, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = (float)x[i];
}

__global__ void nonfinite_rows_kernel(uint32_t* bad, const float* x, long long rows, int cols) {
    for (long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x; r < rows; r += (long long)gridDim.x * blockDim.x) {
        const float* p = x + r * cols;
        bool ok = true;
        for (int j = 0; j < cols && ok; ++j) ok = isfinite(p[j]);
        if (!ok) atomicAdd(bad, 1u);
    }
}

}  // namespace

#define TQ_LAUNCHED() return (int)cudaGetLastError()

TQ_EXPORT int tq_bf16_to_f32(float* out, const uint16_t* in, long long n, cudaStream_t s) {
    bf16_to_f32_kernel<<<grid_for(n), EW, 0, s>>>(out, in, n);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_embed_bf16(float* out, const uint16_t* table, const int32_t* ids, long long n, long long H, cudaStream_t s) {
    embed_kernel<<<grid_for(n * H), EW, 0, s>>>(out, table, ids, n, H);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_rmsnorm(float* out, long long ldo, const float* x, long long ldx, const float* w, const float* gate,
                         long long ldg, long long rows, int n, float eps, int mode, cudaStream_t s) {
    if (rows == 0) return 0;
    const int p = pow2_at_least(n);
    if (rows > 0x7fffffffLL || p > 16384) return (int)cudaErrorInvalidValue;
    const size_t smem = (size_t)p * sizeof(double);
    if (smem > 48 * 1024) cudaFuncSetAttribute(rmsnorm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    rmsnorm_kernel<<<(unsigned)rows, 256, smem, s>>>(out, ldo, x, ldx, w, gate, ldg, n, p, eps, mode);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_conv_silu(float* out, const float* x, const float* w, int B, int T, int C, int K, cudaStream_t s) {
    if (K > 16) return (int)cudaErrorInvalidValue;
    conv_silu_kernel<<<grid_for((long long)B * T * C), EW, 0, s>>>(out, x, w, B, T, C, K);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_gdn_gates(float* g, float* beta, const float* a, const float* b, const float* a_log, const float* dt_bias,
                           long long rows, int heads, cudaStream_t s) {
    gdn_gates_kernel<<<grid_for(rows * heads), EW, 0, s>>>(g, beta, a, b, a_log, dt_bias, rows, heads);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_l2norm_f64(double* out, const float* x, long long ld, long long off, long long bt, int heads, int n,
                            double eps, double scale, cudaStream_t s) {
    const int p = pow2_at_least(n);
    if (p > 16384) return (int)cudaErrorInvalidValue;
    const size_t smem = (size_t)p * sizeof(double);
    if (smem > 48 * 1024) cudaFuncSetAttribute(l2norm_f64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    l2norm_f64_kernel<<<(unsigned)(bt * heads), 128, smem, s>>>(out, x, ld, off, heads, n, p, eps, scale);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_gdn_f64(float* o, const double* q, const double* k, const float* x, long long ldx, long long v_off,
                         const float* g, const float* beta, int B, int T, int Hk, int Hv, int dk, int dv, cudaStream_t s) {
    constexpr int CG = 16;
    if (dv % CG || dk > 1024 || (dk & (dk - 1)) || Hv % Hk || dk < CG) return (int)cudaErrorInvalidValue;
    const size_t smem = (size_t)(CG * dk + CG) * sizeof(double);
    if (smem > 48 * 1024) cudaFuncSetAttribute(gdn_f64_kernel<CG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gdn_f64_kernel<CG><<<(unsigned)(B * Hv * (dv / CG)), dk, smem, s>>>(o, q, k, x, ldx, v_off, g, beta, T, Hk, Hv, dk, dv);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_silu_mul(float* out, long long ldo, const float* a, long long lda, const float* b, long long ldb,
                          long long rows, int n, cudaStream_t s) {
    silu_mul_kernel<<<grid_for(rows * n), EW, 0, s>>>(out, ldo, a, lda, b, ldb, rows, n);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_mul_sigmoid(float* out, const float* a, const float* g, long long n, cudaStream_t s) {
    mul_sigmoid_kernel<<<grid_for(n), EW, 0, s>>>(out, a, g, n);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_shared_combine(float* out, const float* moe, const float* shv, const float* gl, long long rows, int H,
                                cudaStream_t s) {
    shared_combine_kernel<<<grid_for(rows * H), EW, 0, s>>>(out, moe, shv, gl, rows, H);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_route(int32_t* top_i, float* top_w, uint32_t* counts, const float* logits, long long rows, int E, int K,
                       cudaStream_t s) {
    if (K > 16 || E > 4096 || rows == 0) return rows == 0 ? 0 : (int)cudaErrorInvalidValue;
    const int p = pow2_at_least(E);
    // sh: tree [p] doubles, probs [E] floats (padded to an even count), the max, the exp values [E] doubles
    const size_t smem = (size_t)p * 8 + (size_t)(E + (E & 1)) * 4 + 16 + (size_t)E * 8 + 16;
    if (smem > 48 * 1024) cudaFuncSetAttribute(route_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    route_kernel<<<(unsigned)rows, 256, smem, s>>>(top_i, top_w, counts, logits, E, K, p);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_scale_add_rows(float* out, const float* y, const int32_t* tok, const float* w, long long rows, int H,
                                cudaStream_t s) {
    if (rows == 0) return 0;
    scale_add_rows_kernel<<<grid_for(rows * H), EW, 0, s>>>(out, y, tok, w, rows, H);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_gather_rows(float* dst, const float* src, const int32_t* idx, long long rows, int cols, cudaStream_t s) {
    if (rows == 0) return 0;
    gather_rows_kernel<<<grid_for(rows * cols), EW, 0, s>>>(dst, src, idx, rows, cols);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_rope(float* x, long long rows, int T, int H, int hd, int rd, const float* inv_freq, cudaStream_t s) {
    rope_kernel<<<grid_for(rows * (rd / 2)), EW, 0, s>>>(x, rows, T, H, hd, rd, inv_freq);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_attn_f64(float* o, const float* q, const float* k, const float* v, int B, int T, int Hq, int Hkv, int d,
                          double scale, cudaStream_t s) {
    const int p = pow2_at_least(T);
    const size_t smem = (size_t)2 * p * sizeof(double);
    if (smem > 200 * 1024 || Hq % Hkv) return (int)cudaErrorInvalidValue;
    if (smem > 48 * 1024) cudaFuncSetAttribute(attn_f64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attn_f64_kernel<<<(unsigned)((long long)B * Hq * T), 256, smem, s>>>(o, q, k, v, T, Hq, Hkv, d, scale);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_f32_to_f64(double* out, const float* x, long long n, cudaStream_t s) {
    f32_to_f64_kernel<<<grid_for(n), EW, 0, s>>>(out, x, n);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_add_f64(double* a, const double* b, long long n, cudaStream_t s) {
    add_f64_kernel<<<grid_for(n), EW, 0, s>>>(a, b, n);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_f64_to_f32(float* out, const double* x, long long n, cudaStream_t s) {
    f64_to_f32_kernel<<<grid_for(n), EW, 0, s>>>(out, x, n);
    TQ_LAUNCHED();
}

TQ_EXPORT int tq_nonfinite_rows(uint32_t* bad, const float* x, long long rows, int cols, cudaStream_t s) {
    nonfinite_rows_kernel<<<grid_for(rows), EW, 0, s>>>(bad, x, rows, cols);
    TQ_LAUNCHED();
}
