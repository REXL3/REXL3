#include "common.cuh"
#include "roce_dev.cuh"

#include <stddef.h>
#include <stdlib.h>

// The standalone exchange of REXL3's two-rank RoCE all-gather (docs/ROCE_GATHER.md): ONE kernel stages this rank's bytes into its
// half of the site (skipped when the source already is that half: zero-copy), publishes them, waits for the peer's, and writes the
// result (rank-major gather, the peer's bytes only, or the f32 sum in rank order), or nothing (a fused consumer reads the halves).
//
// Two shapes (gb_roce_shape): below bulk_min one descriptor for the whole message (all blocks stage, the last to arrive publishes,
// every block waits for every piece and copies its share); from bulk_min the message is chunked and each block owns whole chunks:
// it stages one, publishes it on its own (one descriptor, one flag), and later waits for the peer's chunk of the same index and copies
// it out, so the copies overlap the wire.
//
// Exactness: a gather and a swap are byte copies. The sum is PTX add.rn.f32 WITHOUT .ftz (as tp_oneshot.cu: x + (-0.0) == x for
// every x, subnormals included), rank 0's operand first, so both ranks compute the same bits and a disjoint-support sum is a copy of
// the owner's element (the same bits as NCCL's ring with two ranks).
//
// Reads of the peer's half are plain weak loads issued after the flag's ld.acquire.sys and a block barrier; never .nc / __ldg.
//
// No abort check before the publish (it would put a host-memory read on the critical path): a poisoned rank's descriptor is ignored
// (its proxy has stopped), and every wait sees the abort word within 256 polls and skips the result.

namespace {

enum Mode : int { kGather = 0, kSwap = 1, kSumF32 = 2, kNone = 3 };

__device__ __forceinline__ float add_keep_denormals(float a, float b) {
    float r;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// Memory-level parallelism: host memory (the region) is not cached in the GPU L2 and a load from it waits ~1-2 us, so every copy
// issues kMlp 16-byte loads a thread before it stores any (the compiler cannot hoist a load above a store to a pointer that may
// alias it: a plain loop keeps ONE load in flight a thread, which made the copy-out of 64 KiB by 2 blocks cost ~5 us, measured).
// The loads stay plain weak loads (no __restrict__ / __ldg: the NIC writes this memory behind the non-coherent path's back).
constexpr int kMlp = 4;
// The bulk kernel's depth: its copies share the memory system with the NIC's DMA of the same exchange, and more pressure from the
// GPU slows the wire (measured: 48 staging blocks 3.8-4.4 ms for 64 MiB against 3.17 with 8, N1+N4; depth 4 with 8 blocks 4.19 ms
// against 3.53 with depth 1, N3+N2).
#ifndef GB_ROCE_BULK_MLP
#define GB_ROCE_BULK_MLP 1
#endif

// dst[0..n) = src[0..n) with this block range's threads, uint4 where both are 16-byte aligned, bytes otherwise.
template <int kMlp = kMlp>
__device__ __forceinline__ void copy_range(char* dst, const char* src, uint64_t n, uint64_t tid, uint64_t nthreads) {
    if (((uintptr_t)dst | (uintptr_t)src) % 16 == 0) {
        const uint64_t n16 = n / 16;
        uint4* d = reinterpret_cast<uint4*>(dst);
        const uint4* s = reinterpret_cast<const uint4*>(src);
        uint64_t i = tid;
        for (; i + (kMlp - 1) * nthreads < n16; i += kMlp * nthreads) {
            uint4 v[kMlp];
#pragma unroll
            for (int k = 0; k < kMlp; ++k) v[k] = s[i + k * nthreads];
#pragma unroll
            for (int k = 0; k < kMlp; ++k) d[i + k * nthreads] = v[k];
        }
        for (; i < n16; i += nthreads) d[i] = s[i];
        for (uint64_t j = n16 * 16 + tid; j < n; j += nthreads) dst[j] = src[j];
    } else {
        for (uint64_t i = tid; i < n; i += nthreads) dst[i] = src[i];
    }
}

__device__ __forceinline__ float4 add4_keep_denormals(float4 x, float4 y) {
    float4 r;
    r.x = add_keep_denormals(x.x, y.x);
    r.y = add_keep_denormals(x.y, y.y);
    r.z = add_keep_denormals(x.z, y.z);
    r.w = add_keep_denormals(x.w, y.w);
    return r;
}

// out[i] = a[i] + b[i] (a = rank 0's operand), f32, over n floats; vectorised when all three are 16-byte aligned.
__device__ __forceinline__ void sum_range(float* out, const float* a, const float* b, uint64_t n, uint64_t tid, uint64_t nthreads) {
    if ((((uintptr_t)out | (uintptr_t)a | (uintptr_t)b) % 16) == 0) {
        const uint64_t n4 = n / 4;
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* b4 = reinterpret_cast<const float4*>(b);
        float4* o4 = reinterpret_cast<float4*>(out);
        uint64_t i = tid;
        for (; i + nthreads < n4; i += 2 * nthreads) {
            const float4 x0 = a4[i], y0 = b4[i], x1 = a4[i + nthreads], y1 = b4[i + nthreads];
            o4[i] = add4_keep_denormals(x0, y0);
            o4[i + nthreads] = add4_keep_denormals(x1, y1);
        }
        for (; i < n4; i += nthreads) o4[i] = add4_keep_denormals(a4[i], b4[i]);
        for (uint64_t j = n4 * 4 + tid; j < n; j += nthreads) out[j] = add_keep_denormals(a[j], b[j]);
    } else {
        for (uint64_t i = tid; i < n; i += nthreads) out[i] = add_keep_denormals(a[i], b[i]);
    }
}

// Stage bytes [lo, hi) of this rank's message into its half (when the source is not already there) and, for a gather, into its rank's
// slot of dst in the same pass (the source is read once).
template <int kMlp = kMlp>
__device__ __forceinline__ void stage_range(int mode, const char* src, char* own, char* dst, uint64_t stride, uint32_t rank,
                                            uint64_t lo, uint64_t hi, uint64_t tid, uint64_t nthreads) {
    if (hi <= lo) return;
    char* dst_own = (mode == kGather && dst != nullptr) ? dst + (uint64_t)rank * stride : nullptr;
    const char* from = src != nullptr ? src : own;
    const bool stage = src != nullptr && src != own;
    if (!stage && dst_own == nullptr) return;
    const uint64_t n = hi - lo;
    if (((uintptr_t)from | (uintptr_t)own | (uintptr_t)(dst_own ? dst_own : own)) % 16 == 0) {
        const uint4* a = reinterpret_cast<const uint4*>(from + lo);
        uint4* b = reinterpret_cast<uint4*>(own + lo);
        uint4* c = dst_own ? reinterpret_cast<uint4*>(dst_own + lo) : nullptr;
        const uint64_t n16 = n / 16;
        uint64_t i = tid;
        for (; i + (kMlp - 1) * nthreads < n16; i += kMlp * nthreads) {
            uint4 v[kMlp];
#pragma unroll
            for (int k = 0; k < kMlp; ++k) v[k] = a[i + k * nthreads];
#pragma unroll
            for (int k = 0; k < kMlp; ++k) {
                if (stage) b[i + k * nthreads] = v[k];
                if (c) c[i + k * nthreads] = v[k];
            }
        }
        for (; i < n16; i += nthreads) {
            const uint4 v = a[i];
            if (stage) b[i] = v;
            if (c) c[i] = v;
        }
        for (uint64_t j = n16 * 16 + tid; j < n; j += nthreads) {
            const char v = from[lo + j];
            if (stage) own[lo + j] = v;
            if (dst_own) dst_own[lo + j] = v;
        }
    } else {
        for (uint64_t i = tid; i < n; i += nthreads) {
            const char v = from[lo + i];
            if (stage) own[lo + i] = v;
            if (dst_own) dst_own[lo + i] = v;
        }
    }
}

// The result for bytes [lo, hi) once the peer's are in: gather: the peer's slot of dst (this rank's was written while staging);
// swap: dst = the peer's bytes; sum: dst = rank 0 + rank 1 (own: this rank's operand, the source or its half).
template <int kMlp = kMlp>
__device__ __forceinline__ void finish_range(int mode, char* dst, uint64_t stride, const char* own, const char* peer, uint32_t rank,
                                             uint64_t lo, uint64_t hi, uint64_t tid, uint64_t nthreads) {
    if (hi <= lo || dst == nullptr || mode == kNone) return;
    const uint64_t n = hi - lo;
    if (mode == kGather) {
        copy_range<kMlp>(dst + (uint64_t)(1 - rank) * stride + lo, peer + lo, n, tid, nthreads);
    } else if (mode == kSwap) {
        copy_range<kMlp>(dst + lo, peer + lo, n, tid, nthreads);
    } else {
        const float* a = reinterpret_cast<const float*>((rank == 0 ? own : peer) + lo);
        const float* b = reinterpret_cast<const float*>((rank == 0 ? peer : own) + lo);
        sum_range(reinterpret_cast<float*>(dst + lo), a, b, n / 4, tid, nthreads);
    }
}

// Below bulk_min: one descriptor (one or two pieces). Block 0 polls the flags (threads 0..pieces-1) and broadcasts the arrival in
// device memory; the other blocks wait on that word, so only one block ever spins on host memory the NIC is writing.
// `send` / `recv`: this rank's / the peer's message bytes (equal for a gather and a sum).
__global__ void __launch_bounds__(512) roce_small_k(GbRoceDev d, GbRoceSite s, const char* src, char* dst, uint64_t send,
                                                    uint64_t recv, uint64_t stride, int mode) {
    gb_roce_pdl_wait();
    const uint32_t seq = gb_roce_seq(d, s);
    const uint32_t par = seq & 1u;
    char* own = gb_roce_half(d, s, seq, d.rank);
    const char* peer = gb_roce_half(d, s, seq, 1 - d.rank);
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, nthreads = (uint64_t)gridDim.x * blockDim.x;
    stage_range(mode, src, own, dst, stride, d.rank, 0, send, tid, nthreads);
    gb_roce_producer_epilogue(d, s, seq, send);
    gb_roce_pdl_trigger();  // the next kernel of the stream may launch now (it waits in its own prologue)
    uint32_t pieces, rail0, bulk;
    uint64_t piece;
    gb_roce_shape(&d, s.id, seq, recv, &pieces, &piece, &rail0, &bulk);
    __shared__ int bad;
    uint32_t* ready = d.ctl + d.ctl_ready + s.flag[par];
    if (blockIdx.x == 0) {
        bad = gb_roce_wait_all(d, s, seq, pieces);
        if (threadIdx.x == 0 && gridDim.x > 1 && !bad) gb_roce_set_ready(ready, seq);
    } else {
        if (threadIdx.x == 0) bad = gb_roce_wait_ready(d, ready, seq) != 0u;
    }
    __syncthreads();
    if (!bad)
        finish_range(mode, dst, stride, src != nullptr ? src : own, peer, d.rank, 0, mode == kSwap ? recv : send, tid, nthreads);
    gb_roce_finish(d, s, seq);
}

// From bulk_min: chunks (gb_roce_chunk). With two blocks or more and a result to write, the grid splits: the first half STAGES the
// chunks in order (each published by the last staging block to finish its share: a per-chunk arrive counter), the second half
// DRAINS at the same time (its first block polls the peer's chunk k and broadcasts the arrival in device memory; every draining
// block copies its share of chunk k), so neither the wire, the staging nor the copy-out waits for the whole of another (the earlier
// form staged every chunk before it copied any out: 64 MiB took 3.53 ms against 2.77 for the exchange alone, N3+N2). A message
// whose source overlaps its destination (an in-place sum or swap) keeps the sequential form: a drained chunk must not overwrite
// source bytes a stager has yet to read.
__global__ void __launch_bounds__(512) roce_bulk_k(GbRoceDev d, GbRoceSite s, const char* src, char* dst, uint64_t send,
                                                   uint64_t recv, uint64_t stride, int mode) {
    gb_roce_pdl_wait();
    const uint32_t seq = gb_roce_seq(d, s);
    const uint32_t par = seq & 1u;
    char* own = gb_roce_half(d, s, seq, d.rank);
    const char* peer = gb_roce_half(d, s, seq, 1 - d.rank);
    const uint64_t cm = gb_roce_chunk(&d, send), ct = gb_roce_chunk(&d, recv);
    const uint32_t mine = (uint32_t)((send + cm - 1) / cm), theirs = (uint32_t)((recv + ct - 1) / ct);
    const uint64_t total = mode == kSwap ? recv : send;
    const uint64_t out_bytes = mode == kGather ? stride + total : total;  // the destination range this exchange writes
    const bool overlap = src != nullptr && dst != nullptr && src < dst + out_bytes && dst < src + send;
    const bool split = gridDim.x >= 2 && dst != nullptr && mode != kNone && !overlap;
    const uint32_t gs = split ? gridDim.x / 2 : gridDim.x;  // staging blocks [0, gs); draining blocks [gs, grid) (all when not split)
    const uint32_t d0 = split ? gs : 0, nd = gridDim.x - d0;
    if (blockIdx.x < gs) {
        const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, nthreads = (uint64_t)gs * blockDim.x;
        for (uint32_t k = 0; k < mine; ++k) {
            const uint64_t lo = (uint64_t)k * cm, hi = send < lo + cm ? send : lo + cm;
            stage_range<GB_ROCE_BULK_MLP>(mode, src, own, dst, stride, d.rank, lo, hi, tid, nthreads);
            __syncthreads();
            if (threadIdx.x == 0 && gb_roce_arrive(d.ctl + d.ctl_chunks + s.flag[par] + k, gs)) {
                const uint64_t off = s.half[par][d.rank] + lo;
                gb_roce_post(d, off, off, hi - lo, hi - lo, 1, s.flag[par] + k, seq, k % d.rails, 1);
            }
        }
    }
    gb_roce_pdl_trigger();  // per block: the next kernel launches once every block got here (after its share of the publishing)
    if (blockIdx.x >= d0) {
        __shared__ int bad;
        const uint64_t tid = (uint64_t)(blockIdx.x - d0) * blockDim.x + threadIdx.x, nthreads = (uint64_t)nd * blockDim.x;
        for (uint32_t k = 0; k < theirs; ++k) {
            uint32_t* ready = d.ctl + d.ctl_ready + s.flag[par] + k;
            if (threadIdx.x == 0) {
                if (blockIdx.x == d0) {
                    bad = gb_roce_wait_slot(d, s.flag[par] + k, seq, s.id) != 0u;
                    if (!bad && nd > 1) gb_roce_set_ready(ready, seq);
                } else {
                    bad = gb_roce_wait_ready(d, ready, seq) != 0u;
                }
            }
            __syncthreads();
            if (bad) break;
            const uint64_t lo = (uint64_t)k * ct, hi = total < lo + ct ? total : lo + ct;
            finish_range<GB_ROCE_BULK_MLP>(mode, dst, stride, src != nullptr ? src : own, peer, d.rank, lo, hi, tid, nthreads);
        }
    }
    gb_roce_finish(d, s, seq);
}

}  // namespace

// Launch the standalone exchange at `site` on `stream`. src: this rank's bytes (device or host-mapped memory; may be its own half of
// the site's current parity for zero-copy, or null when it already wrote that half), dst: where the result goes (null: nowhere).
// send / recv: this rank's and the peer's message bytes (both <= site cap); stride: the rank stride of a gather's dst; mode: 0 gather,
// 1 swap, 2 f32 sum, 3 none; sms: the device's SM count; bulk_blocks: blocks of a bulk (chunked) exchange. Launched with PDL. Returns a
// cudaError_t (0 on success).
extern "C" int gb_roce_exchange(const GbRoceDev* d, const GbRoceSite* s, const void* src, void* dst, uint64_t send, uint64_t recv,
                                uint64_t stride, int mode, int sms, int bulk_blocks, void* stream) {
    if (send > s->cap || recv > s->cap || send == 0 || recv == 0) return (int)cudaErrorInvalidValue;
    const bool bulk = send >= d->bulk_min;
    if (bulk != (recv >= d->bulk_min)) return (int)cudaErrorInvalidValue;  // both ranks must take the same path
    const uint64_t big = send > recv ? send : recv;
    int grid;
    if (src == nullptr && dst == nullptr) {
        grid = 1;  // nothing to stage or copy: the exchange alone
    } else if (bulk) {
        // the chunks' staging and copy-out, by few blocks: they share DRAM with the NIC's DMA, and more blocks slow the wire
        // (measured: 8 blocks 3.17 ms for 64 MiB, 48 blocks 3.8 ms)
        grid = bulk_blocks > 0 && bulk_blocks < sms ? bulk_blocks : sms;
    } else {
        // one block per 32 KiB, at most one per SM: few enough to stay resident beside other work
        const uint64_t want = (big + 32767) / 32768;
        grid = (int)(want < 1 ? 1 : (want > (uint64_t)sms ? (uint64_t)sms : want));
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid);
    cfg.blockDim = dim3(512);
    cfg.stream = static_cast<cudaStream_t>(stream);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    const char* source = static_cast<const char*>(src);
    char* out = static_cast<char*>(dst);
    const cudaError_t e = bulk ? cudaLaunchKernelEx(&cfg, roce_bulk_k, *d, *s, source, out, send, recv, stride, mode)
                               : cudaLaunchKernelEx(&cfg, roce_small_k, *d, *s, source, out, send, recv, stride, mode);
    return (int)e;
}

// sizeof / offsetof of the records the Rust binding mirrors (src/roce.rs checks its #[repr(C)] copies against these).
extern "C" int gb_roce_abi(uint64_t* out, int n) {
    const uint64_t v[] = {sizeof(GbRoceDev), sizeof(GbRoceSite), sizeof(GbRoceOpen), sizeof(GbRoceError), sizeof(GbRoceProxyStat),
                          sizeof(GbRoceBlob), offsetof(GbRoceDev, ctl_ready), offsetof(GbRoceSite, max_pieces),
                          offsetof(GbRoceOpen, rank), GB_ROCE_H_ABORT, GB_ROCE_H_ERROR, GB_ROCE_H_BELL, GB_ROCE_H_PROXY,
                          GB_ROCE_C_SITE0, GB_ROCE_C_PER_SITE};
    const int k = (int)(sizeof v / sizeof v[0]);
    for (int i = 0; i < n && i < k; ++i) out[i] = v[i];
    return k;
}
