// gb-nvcc: --fmad=false
// The gated delta net kernels of gdn.cu built for 32 value heads (16 key heads x 128, 32 value heads x 128, conv kernel 4 over 8192
// channels: the Qwen3.5/3.6-35B-A3B text backbone of MiniCPM-V-4.7; value head h reads key head h / 2). One source: this file only
// sets the shape and suffixes every launcher `_hv32`; the kernels, their provenance (ported in part from TensorFold, see
// THIRD_PARTY_NOTICES.md) and the replay design are gdn.cu's. Line 1 repeats gdn.cu's nvcc flag: build.rs reads only the first line of
// a file, and the single-rounding arithmetic of the replay (one step() giving the same bits wherever it is inlined) needs --fmad=false.
#define GB_GDN_HV 32
#define GB_GDN_EXPORT(name) name##_hv32
#include "gdn.cu"
