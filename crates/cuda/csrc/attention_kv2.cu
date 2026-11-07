// The paged KV cache and attention kernels of attention.cu, built for 2 KV heads (24 query heads in groups of 12:
// Qwen3.8-Flash-Next). One source: this file only sets the shape and suffixes every launcher `_kv2`; the kernels, their
// provenance (ported in part from TensorFold, see THIRD_PARTY_NOTICES.md) and their tests are attention.cu's.
#define GB_ATTN_NKV 2
#define GB_ATTN_EXPORT(name) name##_kv2
#include "attention.cu"
