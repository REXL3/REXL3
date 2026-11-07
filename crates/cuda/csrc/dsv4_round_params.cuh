// Device-resident per-round parameters for the DSV4.1-Flash speculative round graphs.
//
// Unit A (the DFlash/MTP draft round) bakes no per-round host scalars into its graphs: every
// value that varies round-to-round — the anchor token, the advancing `start_pos`, the derived
// `reach` / `n_kv` and the keyed-walk sampling parameters — lives in one of these structs on the
// device, rewritten before each capture/replay by one async H2D copy from pinned staging. A tuned
// graph then replays the same kernel sequence for any round with the same (slot, seeded, walk)
// shape, reading the fresh values from this block.
//
// Layout must match `crate::dsv4_mtp_gpu::Dsv4MtpRoundParams` (Rust `#[repr(C)]`) exactly.
#pragma once
#include <stdint.h>

typedef struct {
    int anchor;        // the trunk token the draft block anchors at (ids[0])
    int start_pos;     // absolute position this round advances from
    int reach;         // min(window, start_pos + 1)
    int n_kv;          // reach + block_size
    int window;        // the per-stage KV ring width
    int block_size;    // draft width (5 on the shipped pack)
    int cand_k;        // keyed-walk candidate count
    int _pad;
    unsigned long long seed;     // keyed-walk SplitMix64 seed
    unsigned long long base_pos; // absolute position the walk's first draft predicts
    double temperature;
    double top_p;
    double min_p;
} Dsv4MtpRoundParams;

__device__ __forceinline__ int mtp_params_slot0(const Dsv4MtpRoundParams* p) {
    const int w = p->window > 0 ? p->window : 1;
    return p->start_pos % w;
}
