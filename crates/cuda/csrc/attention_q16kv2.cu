// The paged KV cache and attention kernels of attention.cu, built for 16 query heads over 2 KV heads (groups of 8: the full-attention
// layers of the Qwen3.5/3.6-35B-A3B text backbone of MiniCPM-V-4.7, head dim 256). One source: this file only sets the shape and
// suffixes every launcher `_q16kv2`; the kernels, their provenance (ported in part from TensorFold, see THIRD_PARTY_NOTICES.md) and
// their tests are attention.cu's. The prefill variants whose heads a CTA do not divide a group of 8 (2 and 3) are refused.
#define GB_ATTN_NKV 2
#define GB_ATTN_NQ 16
#define GB_ATTN_EXPORT(name) name##_q16kv2
#include "attention.cu"
