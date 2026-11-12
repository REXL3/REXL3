// The paged KV cache and attention kernels of attention.cu, built for 8 query heads over 2 KV heads (groups of 4: the five layers of
// MiniCPM-V-4.7's DSpark draft, head dim 256). One source: this file only sets the shape and suffixes every launcher `_q8kv2`; the
// kernels, their provenance (ported in part from TensorFold, see THIRD_PARTY_NOTICES.md) and their tests are attention.cu's. The draft
// block runs as decode rows of one slot that all see the same keys (every row's position is the block's last): no kernel change. The
// prefill variants whose heads a CTA do not divide a group of 4 (2 and 3) are refused.
#define GB_ATTN_NKV 2
#define GB_ATTN_NQ 8
#define GB_ATTN_EXPORT(name) name##_q8kv2
#include "attention.cu"
