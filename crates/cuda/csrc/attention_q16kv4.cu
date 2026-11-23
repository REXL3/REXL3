// The paged KV cache and attention kernels of attention.cu, built for 16 query heads over 4 KV heads (groups of 4: Qwen3.5-4B
// and other generic Qwen hybrids, head dim 256). The original Qwen3.8-27B source and launchers are untouched; this shim only
// selects its compile-time shape and suffixes every launcher `_q16kv4`.
#define GB_ATTN_NKV 4
#define GB_ATTN_NQ 16
#define GB_ATTN_EXPORT(name) name##_q16kv4
#include "attention.cu"
