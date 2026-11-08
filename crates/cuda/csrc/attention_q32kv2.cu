// The paged KV cache and attention kernels of attention.cu, built for 32 query heads over 2 KV heads (groups of 16: the full-attention
// layers of Qwen3.5-122B-A10B / Ornith-1.5-122B, head dim 256).
#define GB_ATTN_NKV 2
#define GB_ATTN_NQ 32
#define GB_ATTN_EXPORT(name) name##_q32kv2
#include "attention.cu"
