// gb-nvcc: --fmad=false
// The gated delta net kernels of gdn.cu built for 64 value heads (16 key heads x 128, 64 value heads x 128, conv kernel 4 over 12288
// channels: the Qwen3.5-122B-A10B / Ornith-1.5-122B text backbone; value head h reads key head h / 4).
#define GB_GDN_HV 64
#define GB_GDN_EXPORT(name) name##_hv64
#include "gdn.cu"
