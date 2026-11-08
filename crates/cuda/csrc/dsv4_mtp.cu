// gb-nvcc: --fmad=false
// DeepSeek-V4.1-Flash NextN / DSpark draft chain (`src/dsv4_mtp.rs`, `/tmp/ds41ref/model.py`:1023-1160):
//   * `topk_idxs_k`:  the draft block's slot list, `cat([0..reach), window + 0..block_size)`, broadcast over every query;
//   * `markov_bias_k`: `logits[r, :] += head @ embed[token[r]]` -- the rank-256 Markov head's full-vocabulary bias;
//   * `confidence_k`:  `sigmoid(dot([h[r] ; m[r]], w) + b)` in fp32, one block per row;
//   * `accept_k`:      the accepted prefix length of a block, one thread per sequence.
//
// Conventions as glue.cu / dspark.cu: fp32 math, `--fmad=false` with explicit `fmaf`, a fixed reduction order that depends
// on the row length only, so a host loop with the same order reproduces the bits.
#include "common.cuh"
#include "dsv4_round_params.cuh"
#include <math.h>

namespace {

constexpr int TOPK_MAX_KV = 160;  // window 128 + block 5, with room; mirrors MAX_KV in dsv4_mtp_gpu.cu

constexpr int CONF_NT = 256;  // the confidence block; the host mirror reproduces this reduction order

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

// out[b][q][j] = j < reach ? j : window + (j - reach), for every query q. The reference expands one row over
// (bsz, block_size), so every query in a batch gets the same list -- including slots past its own position.
__global__ void topk_idxs_k(int* __restrict__ out, int bsz, int block_size, int reach, int window, int n_kv) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_kv) return;
    const int v = j < reach ? j : window + (j - reach);
    const long total = (long)bsz * block_size;
    for (long r = 0; r < total; ++r) out[r * n_kv + j] = v;
}

// The device-params variant: `reach` / `window` / `n_kv` come from the round params block (updated once per round
// by an async H2D copy before a CUDA-graph replay), so a graph captured once replays unchanged as `start_pos`
// advances. The grid is fixed at the maximum `n_kv` (window + block_size); threads past the round's `n_kv` exit
// at the guard, so the writes are identical to the scalar launch.
__global__ void topk_idxs_params_k(int* __restrict__ out, int bsz, int block_size, const Dsv4MtpRoundParams* p) {
    const int n_kv = p->n_kv;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_kv) return;
    const int reach = p->reach;
    const int window = p->window;
    const int v = j < reach ? j : window + (j - reach);
    const long total = (long)bsz * block_size;
    for (long r = 0; r < total; ++r) out[r * n_kv + j] = v;
}

// logits[r, v] += sum_i head[v, i] * embed[token[r], i]. One block per (row, vocab tile); the tile walks `vocab` in
// NT-wide strides so the reduction order is a function of `rank` alone.
constexpr int MB_VOCAB_TILE = 8;

__global__ void __launch_bounds__(CONF_NT)
    markov_bias_k(float* __restrict__ logits, const int* __restrict__ tokens, const float* __restrict__ embed,
                  const float* __restrict__ head, int rows, int vocab, int rank) {
    __shared__ float sm[CONF_NT / 32];
    const int row = blockIdx.x / ((vocab + MB_VOCAB_TILE - 1) / MB_VOCAB_TILE);
    const int tile = blockIdx.x - row * ((vocab + MB_VOCAB_TILE - 1) / MB_VOCAB_TILE);
    if (row >= rows) return;
    const int t = tokens[row];
    const bool ok = (unsigned)t < (unsigned)vocab;
    const float* e = ok ? embed + (size_t)t * rank : nullptr;
    for (int dv = 0; dv < MB_VOCAB_TILE; ++dv) {
        const int v = tile * MB_VOCAB_TILE + dv;
        if (v >= vocab) return;
        float acc = 0.f;
        for (int i = threadIdx.x; i < rank; i += CONF_NT) acc = fmaf(ok ? e[i] : 0.f, head[(size_t)v * rank + i], acc);
        acc = block_sum<CONF_NT>(acc, sm);
        if (threadIdx.x == 0) logits[(size_t)row * vocab + v] += acc;
        __syncthreads();
    }
}

// out[r] = sigmoid(sum_i h[r, i] * w[i] + sum_i m[r, i] * w[dim + i] + b); thread t owns elements t, t + 256, ...
__global__ void __launch_bounds__(CONF_NT)
    confidence_k(const float* __restrict__ h, const float* __restrict__ m, const float* __restrict__ w,
                 const float* __restrict__ b, float* __restrict__ out, int dim, int rank) {
    __shared__ float sm[CONF_NT / 32];
    const int r = blockIdx.x;
    float acc = 0.f;
    for (int i = threadIdx.x; i < dim; i += CONF_NT) acc = fmaf(h[(size_t)r * dim + i], w[i], acc);
    for (int i = threadIdx.x; i < rank; i += CONF_NT) acc = fmaf(m[(size_t)r * rank + i], w[dim + i], acc);
    acc = block_sum<CONF_NT>(acc, sm);
    if (threadIdx.x == 0) out[r] = 1.f / (1.f + expf(-(acc + b[0])));
}

// The accepted prefix: the first k with conf[s, k] < threshold, else block_size. A prefix, not a subset -- see
// `accept_prefix` in src/dsv4_mtp.rs for why.
__global__ void accept_k(const float* __restrict__ conf, int* __restrict__ out, int n_seq, int block_size, float threshold) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_seq) return;
    int k = 0;
    while (k < block_size && conf[(size_t)s * block_size + k] >= threshold) ++k;
    out[s] = k;
}

}  // namespace

extern "C" {

int gb_dsv4_mtp_topk_idxs(void* out, int bsz, int block_size, int reach, int window, void* stream) {
    if (!out || bsz <= 0 || block_size <= 0 || reach < 0 || window <= 0) return 1;
    const int n_kv = reach + block_size;
    const int nt = 128, nb = (n_kv + nt - 1) / nt;
    topk_idxs_k<<<nb, nt, 0, (cudaStream_t)stream>>>((int*)out, bsz, block_size, reach, window, n_kv);
    return (int)cudaGetLastError();
}

int gb_dsv4_mtp_topk_idxs_params(void* out, int bsz, int block_size, const void* params, void* stream) {
    if (!out || !params || bsz <= 0 || block_size <= 0) return 1;
    // NOTE: `params` is a DEVICE pointer; never dereference it on the host (the kernels read it on the device).
    const Dsv4MtpRoundParams* p = (const Dsv4MtpRoundParams*)params;
    const int nt = 128;
    // grid fixed at the max n_kv the config can reach (window + block_size)
    const int nb = (TOPK_MAX_KV + nt - 1) / nt;
    topk_idxs_params_k<<<nb, nt, 0, (cudaStream_t)stream>>>((int*)out, bsz, block_size, p);
    return (int)cudaGetLastError();
}

int gb_dsv4_mtp_markov_bias(void* logits, const void* tokens, const void* embed, const void* head, int rows, int vocab, int rank,
                            void* stream) {
    if (!logits || !tokens || !embed || !head || rows <= 0 || vocab <= 0 || rank <= 0) return 1;
    const int tiles = (vocab + MB_VOCAB_TILE - 1) / MB_VOCAB_TILE;
    markov_bias_k<<<rows * tiles, CONF_NT, 0, (cudaStream_t)stream>>>((float*)logits, (const int*)tokens, (const float*)embed,
                                                                     (const float*)head, rows, vocab, rank);
    return (int)cudaGetLastError();
}

int gb_dsv4_mtp_confidence(const void* h, const void* m, const void* w, const void* b, void* out, int rows, int dim, int rank,
                           void* stream) {
    if (!h || !m || !w || !b || !out || rows <= 0 || dim <= 0 || rank <= 0) return 1;
    confidence_k<<<rows, CONF_NT, 0, (cudaStream_t)stream>>>((const float*)h, (const float*)m, (const float*)w, (const float*)b,
                                                             (float*)out, dim, rank);
    return (int)cudaGetLastError();
}

int gb_dsv4_mtp_accept(const void* conf, void* out, int n_seq, int block_size, float threshold, void* stream) {
    if (!conf || !out || n_seq <= 0 || block_size <= 0) return 1;
    const int nt = 128, nb = (n_seq + nt - 1) / nt;
    accept_k<<<nb, nt, 0, (cudaStream_t)stream>>>((const float*)conf, (int*)out, n_seq, block_size, threshold);
    return (int)cudaGetLastError();
}

}  // extern "C"