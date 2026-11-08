// GLM-5.3-Flash DSA (DeepSeek sparse attention) indexer: geometry and the one definition of the per-row regime.
// Included by dsa_index.cu, dsa_score.cu and dsa_select.cu. Spec and oracle: docs/GLM53.md, tools/dsa_oracle.py.
//
// Semantics (the ExLlamaV3 fork's k-pool path, exllamav3/modules/mla_attn.py `_indexer_topk_kpool`):
//   index key  k_t  = LayerNorm(wk(x_t))            fp16, 128 values, biased LayerNorm (eps 1e-6)
//   gate       g_t  = x_t @ index_kpool_compress_gate^T   fp16, 128 values
//   pool key   K_p  = sum_{j<4} softmax_j(g_{4p+j} + ape_j) * k_{4p+j}   per dimension, fp32 math, one fp16 rounding
//   score      S[r,p] = fp16( 2^-6 * sum_{h<32} relu(q[r,h] . K_p) * w[r,h] )     for p < (pos_r + 1) / 4, else -inf
//   selection  the 512 highest fp16 scores, ties to the LOWEST pool index, ascending; each pool expands to its 4 tokens;
//              then the 0..3 raw tokens of the incomplete last pool (the tail)
//   regime     a row with pos + 1 > 2048 (its OWN context length) is sparse; every other row is dense (no selection)
//
// Row invariance: nothing in these kernels depends on how many rows share a call, on which rows they are, or on the order of
// the rows: each output element is a pure function of one row's own inputs and the planes, computed in an order that is fixed by
// the geometry. No atomics anywhere; no FMA contraction (every file is compiled with --fmad=false).

#pragma once
#include "common.cuh"

#define DSA_D 128              // index_head_dim
#define DSA_H 32               // index_n_heads
#define DSA_P 4                // index_kpool: tokens per pool
#define DSA_TOPK 2048          // index_topk: tokens attended per sparse row (pools x P)
#define DSA_SEL_POOLS 512      // DSA_TOPK / DSA_P
#define DSA_PAGE_TOKENS 256    // tokens per cache page (shared with the latent cache's block table)
#define DSA_POOLS_PER_PAGE 64  // DSA_PAGE_TOKENS / DSA_P
#define DSA_K_OUT 2051         // DSA_TOPK + DSA_P - 1: selected tokens plus the largest possible tail
#define DSA_K_STRIDE 2056      // row stride of the selection (multiple of 4 ints = 16 bytes)
#define DSA_MAX_ROWS 128       // rows per launch
#define DSA_RING_MIN 132       // DSA_MAX_ROWS + DSA_P: the smallest ring (multiple of DSA_P) that survives a full window

static_assert(DSA_TOPK == DSA_SEL_POOLS * DSA_P, "geometry");
static_assert(DSA_PAGE_TOKENS == DSA_POOLS_PER_PAGE * DSA_P, "geometry");
static_assert(DSA_K_STRIDE >= DSA_K_OUT && DSA_K_STRIDE % 4 == 0, "geometry");

// The regime of a row, defined once. `pos` is the row's absolute position; its context is the pos + 1 tokens 0..pos.
// For one decode row this is exactly the fork's `max(seqlens) + q_len > index_topk`; for a row of a verify window it is the
// same rule applied to that row alone (the fork applies it to the whole window).
__host__ __device__ __forceinline__ bool dsa_row_is_sparse(int pos) { return pos + 1 > DSA_TOPK; }
// Complete pools visible to the row, and the raw tokens of its incomplete last pool.
__host__ __device__ __forceinline__ int dsa_row_pools(int pos) { return (pos + 1) / DSA_P; }
__host__ __device__ __forceinline__ int dsa_row_tail(int pos) { return (pos + 1) % DSA_P; }
