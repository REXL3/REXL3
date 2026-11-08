// The two-rank RoCE all-gather of REXL3 (docs/ROCE_GATHER.md): the layout and records shared by the host shim (roce_shim.cu), the
// device side (roce_dev.cuh) and the Rust binding (src/roce.rs, which mirrors every struct here with #[repr(C)] and a size test).
//
// One registered region per rank (cudaHostAlloc Mapped|Portable, registered with ibv_reg_mr on every rail, never with
// IBV_ACCESS_RELAXED_ORDERING). Both ranks build the SAME layout from the same site list, so an offset names the same place on both:
//
//   [0, 4096)                 header (GbRoceHeader): abort word, the first error, the proxy's progress
//   [ring_off, +64 * ring)    descriptor ring (GbRoceDesc), written by the GPU, read by this rank's proxy thread
//   [flags_off, +8 * flags)   flags (u64), written by the PEER's NIC: (generation << 32) | sequence
//   [sites_off, ...)          site buffers: per site, 2 parities x 2 ranks halves of `cap` bytes, rank-major ([rank 0 | rank 1])
#pragma once
#include <stdint.h>

#define GB_ROCE_VERSION 1u
#define GB_ROCE_MAX_RAILS 2
#define GB_ROCE_HEADER_BYTES 4096u
#define GB_ROCE_DESC_BYTES 64u
#define GB_ROCE_BLOB_BYTES 512u

// header words (byte offsets inside the header page; each on its own 64-byte line)
#define GB_ROCE_H_ABORT 64u      // u32: nonzero = abort; written by the host (abort()), read by every spinning kernel
#define GB_ROCE_H_ERROR 128u     // GbRoceError: the first error, from the GPU (a wait that timed out) or the proxy (a CQ error)
#define GB_ROCE_H_BELL 256u      // u32: the doorbell, the last ticket the GPU released (its own line: the only line the proxy spins on)
#define GB_ROCE_H_PROXY 1024u    // GbRoceProxyStat: what the proxy has done (diagnostics, written by the proxy, read by the host)

// error codes (GbRoceError.code)
#define GB_ROCE_E_TIMEOUT 1u     // a GPU wait passed its deadline: site, seq and the missing piece recorded
#define GB_ROCE_E_ABORTED 2u     // a GPU wait saw the abort word
#define GB_ROCE_E_CQ 3u          // a send completed with an error status (the peer is gone, or the QP broke): detail = wc status
#define GB_ROCE_E_POST 4u        // ibv_post_send failed: detail = its return value
#define GB_ROCE_E_DESC 5u        // a descriptor the proxy refused (out of the region): detail = the ticket
#define GB_ROCE_E_SQ 6u          // the send queue did not drain within the proxy's bound

// device control words (u32, device memory, GbRoceDev.ctl): the ticket counter, then per site: seq, arrive, tail
#define GB_ROCE_C_TICKET 0u
#define GB_ROCE_C_SITE0 16u
#define GB_ROCE_C_PER_SITE 4u    // [seq, arrive, tail, elect]: elect = the last sequence a fused consumer grid elected its poller for

#ifdef __cplusplus
extern "C" {
#endif

// One send request: `bytes` from local region offset `src` to peer region offset `dst`, as `pieces` pieces of `piece_bytes` (the last
// one shorter), piece k on rail (rail0 + k) % rails, each followed on the same QP by an inline write of the flag value into the
// peer's flag slot `flag + k`. The GPU writes the body as three 16-byte stores, releases the entry's ticket at system scope, then
// writes the ticket into the doorbell (GB_ROCE_H_BELL). The proxy spins on the doorbell, never on an entry the GPU may be writing: a CPU
// spinning on the line the GPU writes in pieces turns every store into a coherence round trip (measured: 4.7 us per publish).
typedef struct GbRoceDesc {
    uint64_t src;
    uint64_t dst;
    uint64_t piece_bytes;
    uint32_t bytes_lo;      // total bytes (low 32)
    uint32_t bytes_hi;      // total bytes (high 32)
    uint32_t flag;          // first flag slot of the peer
    uint32_t seq;           // the flag value's low word (the high word is the generation)
    uint16_t pieces;
    uint8_t rail0;
    uint8_t bulk;           // 1: the bulk QPs (prefill-sized messages), 0: the latency QPs
    uint32_t spare;
    uint32_t pad[3];
    uint32_t ticket;        // last word, written last
} GbRoceDesc;

typedef struct GbRoceError {
    uint32_t code;
    uint32_t site;
    uint32_t seq;
    uint32_t detail;
} GbRoceError;

typedef struct GbRoceProxyStat {
    uint64_t descs;         // descriptors posted
    uint64_t wrs;           // work requests posted
    uint64_t completions;   // signaled completions retired
    uint64_t max_sq;        // the largest number of unretired work requests seen on one QP
    uint32_t running;       // 1 while the thread serves
    uint32_t cpu;           // the core it runs on (-1: not pinned)
} GbRoceProxyStat;

// What one rank tells the other (the side channel carries it as GB_ROCE_BLOB_BYTES opaque bytes).
typedef struct GbRoceBlob {
    uint32_t magic;         // 'RXLT'
    uint32_t version;
    uint64_t region_addr;   // this rank's region base (the peer's remote_addr base)
    uint64_t region_bytes;
    uint64_t layout_hash;   // FNV-1a of the layout (site list, ring, flags): must match
    uint32_t nonce;         // this rank's half of the generation
    uint32_t rails;
    uint32_t tclass;
    uint32_t mtu;           // the smaller active MTU of the rails (enum ibv_mtu)
    struct {
        uint32_t rkey;
        uint32_t qpn[2];    // [latency, bulk]
        uint32_t psn[2];
        uint8_t gid[16];
        uint32_t mtu;
    } rail[GB_ROCE_MAX_RAILS];
    uint8_t pad[GB_ROCE_BLOB_BYTES - 48 - GB_ROCE_MAX_RAILS * 40];
} GbRoceBlob;

typedef struct GbRoceOpen {
    void* region;           // host address of the registered region (cudaHostAlloc'd by the caller)
    uint64_t region_bytes;
    uint64_t ring_off;      // byte offset of the ring
    uint64_t flags_off;     // byte offset of the flag slots (u64 each)
    uint32_t ring_entries;  // a power of two
    uint32_t flag_slots;
    uint64_t layout_hash;
    const char* hcas;       // comma-separated device names, at most GB_ROCE_MAX_RAILS ("rocep1s0f0,roceP2p1s0f0")
    int32_t gid_index;      // -1: find the RoCE v2 IPv4-mapped GID of each device in sysfs
    int32_t tclass;         // GRH traffic class (106 = DSCP 26 | ECN on the CRS812 fabric)
    int32_t sq_depth;       // send queue depth per QP
    int32_t rank;
} GbRoceOpen;

// The host shim (roce_shim.cu). Every call returns 0 or a negative errno-like code with a message in `err` (NUL-terminated).
typedef struct GbRoceCtx GbRoceCtx;
int gb_roce_available(char* err, int errlen);
int gb_roce_open(const GbRoceOpen* o, GbRoceCtx** out, char* err, int errlen);
int gb_roce_blob(GbRoceCtx* c, void* blob, int len);
int gb_roce_connect(GbRoceCtx* c, const void* peer_blob, int len, uint64_t* generation, char* err, int errlen);
int gb_roce_start(GbRoceCtx* c, int cpu, char* err, int errlen);
int gb_roce_isolate(GbRoceCtx* c, char* err, int errlen);  // the proxy's core taken from every other thread (count changed)
void gb_roce_stop(GbRoceCtx* c);
void gb_roce_close(GbRoceCtx* c);
int gb_roce_rails(const GbRoceCtx* c);

#ifdef __cplusplus
}
#endif
