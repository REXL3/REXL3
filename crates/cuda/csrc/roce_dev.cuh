// Device side of REXL3's two-rank RoCE all-gather (docs/ROCE_GATHER.md): what a kernel calls to publish its output to the peer and to
// wait for the peer's. Included by roce_gather.cu (the standalone gather) and by any fused producer / consumer kernel.
//
// A SITE is one exchange point: per parity (the exchange's sequence & 1) a rank-major pair of halves [rank 0 | rank 1] of `cap` bytes
// in the registered region, and the flag slots the peer's pieces land on. Sequence numbers live in device memory (GbRoceDev.ctl), so
// a CUDA graph replays exchanges with new sequences, and eager and graph launches mix.
//
// Protocol, per exchange of a site (both ranks run the same kernels in the same order on one stream):
//   producer: write my half of parity seq&1 (in place: zero-copy, or staged by the standalone kernel); every block arrives at
//             GPU scope; the LAST block's thread 0 writes one 64-byte descriptor into the ring and releases its ticket at SYSTEM
//             scope (the only system-scope fence of the exchange); this rank's proxy posts the RDMA writes, a flag after each piece.
//   consumer: one thread per piece polls its flag (ld.acquire.sys) for (generation << 32 | seq), with the abort word and a
//             %globaltimer deadline checked while it spins; a block barrier; then the block reads both halves with plain loads
//             (never ld.global.nc / __ldg: the NIC writes this memory behind the non-coherent path's back).
//   finish:   the last block to finish the consumer advances the site's sequence (the next exchange uses the other parity).
// Slot reuse is safe because on both ranks consumer(site, t) precedes producer(site, t + 1) in stream order: the peer cannot write
// parity p again before it has consumed exchange t, which needs my data of t, which my producer of t only publishes after my
// consumer of t - 1 finished reading parity p (docs/ROCE_GATHER.md, "safety").
#pragma once
#include <stdint.h>

#include "roce_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Passed by value to every kernel that takes part (it fits in the parameter space with room to spare). Mirrored in src/roce.rs.
typedef struct GbRoceDev {
    char* region;          // the region's device address (on a GB10 the host address itself)
    uint32_t* ctl;         // device memory: GB_ROCE_C_TICKET, then GB_ROCE_C_PER_SITE words per site from GB_ROCE_C_SITE0, ...
    uint64_t gen_hi;       // generation << 32: the high word of every flag this rank waits for
    uint64_t timeout_ns;   // a wait gives up (records GB_ROCE_E_TIMEOUT, raises abort) after this long
    uint64_t ring_off;
    uint64_t flags_off;
    uint64_t bulk_min;     // a message of at least this many bytes is chunked onto the bulk QPs
    uint64_t chunk;        // the largest chunk (one piece, one flag, one descriptor each)
    uint64_t chunk_min;    // the smallest: a bulk message is cut into about 8 chunks, a power of two in [chunk_min, chunk]
    uint32_t ring_mask;
    uint32_t rails;
    uint32_t rank;
    uint32_t stripe_min;   // a message of at least this many bytes (and below bulk_min) is split over both rails
    uint32_t ctl_chunks;   // word offset in ctl of the per-flag-slot arrive counters (chunks staged by a group of blocks)
    uint32_t ctl_ready;    // word offset in ctl of the per-flag-slot ready words (one poller's "arrived", broadcast in device memory)
} GbRoceDev;

typedef struct GbRoceSite {
    uint64_t half[2][2];   // byte offsets [parity][rank]
    uint64_t cap;          // bytes per half
    uint32_t flag[2];      // first incoming flag slot per parity
    uint32_t id;           // site index (its ctl words)
    uint32_t max_pieces;   // flag slots reserved per parity
} GbRoceSite;

// The chunk of a bulk message of `bytes`: about 8 chunks, a power of two clamped to [chunk_min, chunk].
#ifdef __CUDACC__
__host__ __device__
#endif
static inline uint64_t gb_roce_chunk(const GbRoceDev* d, uint64_t bytes) {
    uint64_t want = (bytes + 7) / 8, c = d->chunk_min;
    while (c < want && c < d->chunk) c <<= 1;
    return c;
}

// The shape of one rank's message of `bytes` at `site`, exchange `seq`: the same function on both ranks, so the receiver knows which
// flags to wait for. Single rail below stripe_min (alternating rails by site + seq), both rails up to bulk_min, chunks above.
#ifdef __CUDACC__
__host__ __device__
#endif
static inline void gb_roce_shape(const GbRoceDev* d, uint32_t site, uint32_t seq, uint64_t bytes, uint32_t* pieces,
                                 uint64_t* piece_bytes, uint32_t* rail0, uint32_t* bulk) {
    if (bytes >= d->bulk_min) {
        const uint64_t c = gb_roce_chunk(d, bytes);
        *pieces = (uint32_t)((bytes + c - 1) / c);
        *piece_bytes = c;
        *rail0 = 0;
        *bulk = 1;
    } else if (d->rails > 1 && bytes >= d->stripe_min) {
        *pieces = 2;
        *piece_bytes = (((bytes + 1) / 2) + 63) & ~63ull;
        *rail0 = 0;
        *bulk = 0;
    } else {
        *pieces = 1;
        *piece_bytes = bytes;
        *rail0 = d->rails > 1 ? (site + seq) % d->rails : 0;
        *bulk = 0;
    }
}

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__

// Programmatic dependent launch (PDL): a kernel launched with cudaLaunchAttributeProgrammaticStreamSerialization may start before
// the previous kernel of its stream ends; it must call gb_roce_pdl_wait() before it reads anything the previous kernel wrote (and
// before it reads a site's sequence). gb_roce_pdl_trigger() lets the next kernel launch early (it still waits for this one to end).
__device__ __forceinline__ void gb_roce_pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void gb_roce_pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ uint64_t gb_roce_now() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint32_t gb_roce_ld_relaxed_sys(const volatile void* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint64_t gb_roce_ld_acquire_sys64(const volatile void* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t* gb_roce_ctl(const GbRoceDev& d, uint32_t site, uint32_t word) {
    return d.ctl + GB_ROCE_C_SITE0 + site * GB_ROCE_C_PER_SITE + word;
}

// 1 when this rank's transport is aborted or poisoned (a kernel that sees it does nothing; the host reports the error).
__device__ __forceinline__ bool gb_roce_aborted(const GbRoceDev& d) {
    return gb_roce_ld_relaxed_sys(d.region + GB_ROCE_H_ABORT) != 0u;
}

// The sequence of the exchange in progress at `site` (every block of the producer and the consumer reads the same value: only the
// consumer's last block advances it, at its very end).
__device__ __forceinline__ uint32_t gb_roce_seq(const GbRoceDev& d, const GbRoceSite& s) {
    return *(volatile uint32_t*)gb_roce_ctl(d, s.id, 0) + 1u;
}

__device__ __forceinline__ char* gb_roce_half(const GbRoceDev& d, const GbRoceSite& s, uint32_t seq, uint32_t rank) {
    return d.region + s.half[seq & 1u][rank];
}

// Thread 0 of each block, after a __syncthreads: count this block in at GPU scope (release of the block's writes, acquire of the
// earlier blocks'); true in the last block, which also resets the counter for the next launch.
__device__ __forceinline__ bool gb_roce_arrive(uint32_t* counter, uint32_t blocks) {
    if (blocks == 1) return true;  // one block: it is the last (no atomic, no fence: its writes are ordered by the barrier)
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(counter) : "memory");
    if (old + 1u == blocks) {
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(counter), "r"(0u) : "memory");
        return true;
    }
    return false;
}

// One thread: put a descriptor in the ring and release its ticket at system scope. Every write this thread has acquired (its own,
// its block's through the barrier, the earlier blocks' through gb_roce_arrive) is visible to the proxy and the NIC before the ticket.
__device__ __forceinline__ void gb_roce_post(const GbRoceDev& d, uint64_t src, uint64_t dst, uint64_t bytes, uint64_t piece_bytes,
                                             uint32_t pieces, uint32_t flag, uint32_t seq, uint32_t rail0, uint32_t bulk) {
    const uint32_t t = atomicAdd(d.ctl + GB_ROCE_C_TICKET, 1u) + 1u;
    uint4* e = reinterpret_cast<uint4*>(reinterpret_cast<GbRoceDesc*>(d.region + d.ring_off) + (t & d.ring_mask));
    // the GbRoceDesc layout as four 16-byte stores (roce_abi.h; static_asserts in roce_shim.cu)
    e[0] = make_uint4((uint32_t)src, (uint32_t)(src >> 32), (uint32_t)dst, (uint32_t)(dst >> 32));
    e[1] = make_uint4((uint32_t)piece_bytes, (uint32_t)(piece_bytes >> 32), (uint32_t)bytes, (uint32_t)(bytes >> 32));
    e[2] = make_uint4(flag, seq, (pieces & 0xffffu) | ((rail0 & 0xffu) << 16) | ((bulk & 0xffu) << 24), 0u);
    // the entry's ticket with RELEASE (the proxy acquires it before it reads the body), then the doorbell, relaxed: a doorbell seen
    // before its entry's ticket only makes the proxy wait on that entry for a moment
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(&reinterpret_cast<GbRoceDesc*>(e)->ticket), "r"(t) : "memory");
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(d.region + GB_ROCE_H_BELL), "r"(t) : "memory");
}

// One thread: publish this rank's whole message of `bytes` at `site` (my half of parity seq & 1), shaped by gb_roce_shape.
__device__ __forceinline__ void gb_roce_publish(const GbRoceDev& d, const GbRoceSite& s, uint32_t seq, uint64_t bytes) {
    uint32_t pieces, rail0, bulk;
    uint64_t piece;
    gb_roce_shape(&d, s.id, seq, bytes, &pieces, &piece, &rail0, &bulk);
    const uint64_t off = s.half[seq & 1u][d.rank];
    gb_roce_post(d, off, off, bytes, piece, pieces, s.flag[seq & 1u], seq, rail0, bulk);
}

// Record the first error and raise the abort word (fail-stop): the host sees it in async_error().
__device__ __forceinline__ void gb_roce_fail(const GbRoceDev& d, uint32_t code, uint32_t site, uint32_t seq, uint32_t detail) {
    GbRoceError* e = reinterpret_cast<GbRoceError*>(d.region + GB_ROCE_H_ERROR);
    if (atomicCAS_system(&e->code, 0u, code) == 0u) {
        e->site = site;
        e->seq = seq;
        e->detail = detail;
    }
    __threadfence_system();
    atomicExch_system(reinterpret_cast<unsigned int*>(d.region + GB_ROCE_H_ABORT), 1u);
}

// One thread: spin until incoming flag slot `slot` holds (generation << 32 | seq). 0 when it does; an error code when the abort word
// is raised or the deadline passes (the error is recorded). The acquire orders every later load of this block (after a barrier) after
// the flag, so the NIC's data, placed before its flag, is what they read.
__device__ __forceinline__ uint32_t gb_roce_wait_slot(const GbRoceDev& d, uint32_t slot, uint32_t seq, uint32_t site) {
    const volatile uint64_t* f = reinterpret_cast<const volatile uint64_t*>(d.region + d.flags_off) + slot;
    const uint64_t want = d.gen_hi | seq;
    if (gb_roce_ld_acquire_sys64(f) == want) return 0;
    const uint64_t t0 = gb_roce_now();
    for (uint32_t spins = 1;; ++spins) {
        if (gb_roce_ld_acquire_sys64(f) == want) return 0;
        if ((spins & 255u) == 0u) {
            if (gb_roce_aborted(d)) return GB_ROCE_E_ABORTED;
            if (gb_roce_now() - t0 > d.timeout_ns) {
                gb_roce_fail(d, GB_ROCE_E_TIMEOUT, site, seq, slot);
                return GB_ROCE_E_TIMEOUT;
            }
        }
    }
}

// One thread: spin until device word `w` holds `seq` (another block's poller saw the flags and broadcast it); the abort word is
// checked every 256 polls (a poller that failed raised it). 0 when it does.
__device__ __forceinline__ uint32_t gb_roce_wait_ready(const GbRoceDev& d, const uint32_t* w, uint32_t seq) {
    for (uint32_t spins = 1;; ++spins) {
        uint32_t v;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(w) : "memory");
        if (v == seq) return 0;
        if ((spins & 255u) == 0u && gb_roce_aborted(d)) return GB_ROCE_E_ABORTED;
    }
}

__device__ __forceinline__ void gb_roce_set_ready(uint32_t* w, uint32_t seq) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(w), "r"(seq) : "memory");
}

// Every thread of a block: wait for all `pieces` flags of the peer's message at `site` (threads 0..pieces-1 poll one each). Returns
// nonzero in every thread when any wait failed.
__device__ __forceinline__ int gb_roce_wait_all(const GbRoceDev& d, const GbRoceSite& s, uint32_t seq, uint32_t pieces) {
    __shared__ int gb_roce_bad;
    if (threadIdx.x == 0) gb_roce_bad = 0;
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < pieces; k += blockDim.x)
        if (gb_roce_wait_slot(d, s.flag[seq & 1u] + k, seq, s.id) != 0u) gb_roce_bad = 1;
    __syncthreads();
    return gb_roce_bad;
}

// Every thread of every block of the consumer, at its end: the last block advances the site's sequence.
__device__ __forceinline__ void gb_roce_finish(const GbRoceDev& d, const GbRoceSite& s, uint32_t seq) {
    __syncthreads();
    if (threadIdx.x == 0 && gb_roce_arrive(gb_roce_ctl(d, s.id, 2), gridDim.x * gridDim.y * gridDim.z))
        *(volatile uint32_t*)gb_roce_ctl(d, s.id, 0) = seq;
}

// The fused producer's epilogue, every thread of every block, after the block's last write into gb_roce_half(d, s, seq, d.rank):
// the last block publishes the message.
__device__ __forceinline__ void gb_roce_producer_epilogue(const GbRoceDev& d, const GbRoceSite& s, uint32_t seq, uint64_t bytes) {
    __syncthreads();
    if (threadIdx.x == 0 && gb_roce_arrive(gb_roce_ctl(d, s.id, 1), gridDim.x * gridDim.y * gridDim.z))
        gb_roce_publish(d, s, seq, bytes);
}

// The fused consumer's prologue, every thread of every block: wait for the peer's message of `peer_bytes`. Nonzero: do not read
// the halves (the exchange failed; the host reports it), but still call gb_roce_finish.
// ONE block polls the flags in host memory (the first to arrive: it swaps the site's election word from the previous sequence to
// this one, so the poller is always a resident block and no grid-size assumption is made) and broadcasts the arrival through the
// first flag slot's ready word in device memory; every other block spins on that word. A grid of consumer blocks all polling memory
// the NIC is writing costs the NIC's DMA (the microbenchmark-2/5 finding behind the standalone kernel's single poller).
__device__ __forceinline__ int gb_roce_consumer_prologue(const GbRoceDev& d, const GbRoceSite& s, uint32_t seq, uint64_t peer_bytes) {
    uint32_t pieces, rail0, bulk;
    uint64_t piece;
    gb_roce_shape(&d, s.id, seq, peer_bytes, &pieces, &piece, &rail0, &bulk);
    if (gridDim.x * gridDim.y * gridDim.z == 1) return gb_roce_wait_all(d, s, seq, pieces);
    __shared__ int gb_roce_poller, gb_roce_late;
    uint32_t* ready = d.ctl + d.ctl_ready + s.flag[seq & 1u];
    if (threadIdx.x == 0) gb_roce_poller = atomicExch(gb_roce_ctl(d, s.id, 3), seq) != seq;
    __syncthreads();
    if (gb_roce_poller) {  // block-uniform
        const int bad = gb_roce_wait_all(d, s, seq, pieces);
        if (threadIdx.x == 0 && !bad) gb_roce_set_ready(ready, seq);
        return bad;
    }
    if (threadIdx.x == 0) gb_roce_late = gb_roce_wait_ready(d, ready, seq) != 0u;
    __syncthreads();
    return gb_roce_late;
}

#endif  // __CUDACC__
