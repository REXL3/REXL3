// gb-nvcc: -Xcompiler -mno-outline-atomics
// The host side of REXL3's two-rank RoCE all-gather (docs/ROCE_GATHER.md): libibverbs opened at run time (no link-time dependency, as
// NCCL is in nccl.rs), the queue pairs of every rail, and the proxy thread that turns GPU-written descriptors into RDMA writes.
//
// Host code only (nvcc hands it to the host compiler). The verbs structs, enums and the two calls verbs.h inlines through the context's
// ops table (ibv_post_send, ibv_poll_cq) come from the installed <infiniband/verbs.h> at build time, so no struct offset is hard-coded
// here; a build host without the header compiles the stubs at the bottom and gb_roce_available() says why.
//
// Design (ideas credited in THIRD_PARTY_NOTICES.md and docs/ROCE_GATHER.md; no code taken from any of them): one-shot RDMA writes into
// the peer's pinned region with a flag written after the data on the same reliable-connected QP (the scheme of b12x's RoCEnante and
// TensorFold's port of it), with REXL3's own protocol around it: a ring of 64-byte descriptors instead of a single doorbell word (any
// number of sites and chunked prefill transfers in flight, each naming its own source, destination and flag slots), one doorbell per
// rail per descriptor, selective signaling, a proxy that never naps while serving and is pinned to one core, separate QPs for bulk
// transfers, and fail-stop errors written into the region's header where the GPU and the host both see them.
#include "roce_abi.h"

#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void gb_roce_fail(char* err, int errlen, const char* fmt, ...) {
    if (!err || errlen <= 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, (size_t)errlen, fmt, ap);
    va_end(ap);
}

static_assert(sizeof(GbRoceDesc) == GB_ROCE_DESC_BYTES, "a descriptor is one 64-byte line");
static_assert(offsetof(GbRoceDesc, ticket) == 60, "the ticket is the last word");
static_assert(offsetof(GbRoceDesc, piece_bytes) == 16 && offsetof(GbRoceDesc, bytes_lo) == 24 && offsetof(GbRoceDesc, flag) == 32 &&
                  offsetof(GbRoceDesc, seq) == 36 && offsetof(GbRoceDesc, pieces) == 40 && offsetof(GbRoceDesc, rail0) == 42 &&
                  offsetof(GbRoceDesc, bulk) == 43,
              "the descriptor layout roce_dev.cuh writes as four 16-byte stores");
static_assert(sizeof(GbRoceBlob) == GB_ROCE_BLOB_BYTES, "the blob is GB_ROCE_BLOB_BYTES");

#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#include <dlfcn.h>
#include <pthread.h>
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>

namespace {

struct Verbs {
    void* lib = nullptr;
    ibv_device** (*get_device_list)(int*);
    void (*free_device_list)(ibv_device**);
    const char* (*get_device_name)(ibv_device*);
    ibv_context* (*open_device)(ibv_device*);
    int (*close_device)(ibv_context*);
    int (*query_port)(ibv_context*, uint8_t, ibv_port_attr*);
    int (*query_gid)(ibv_context*, uint8_t, int, ibv_gid*);
    ibv_pd* (*alloc_pd)(ibv_context*);
    int (*dealloc_pd)(ibv_pd*);
    ibv_mr* (*reg_mr)(ibv_pd*, void*, size_t, int);
    int (*dereg_mr)(ibv_mr*);
    ibv_cq* (*create_cq)(ibv_context*, int, void*, ibv_comp_channel*, int);
    int (*destroy_cq)(ibv_cq*);
    ibv_qp* (*create_qp)(ibv_pd*, ibv_qp_init_attr*);
    int (*destroy_qp)(ibv_qp*);
    int (*modify_qp)(ibv_qp*, ibv_qp_attr*, int);
    const char* (*wc_status_str)(ibv_wc_status);
};

template <class T>
bool sym(void* lib, const char* name, T* out) {
    *out = reinterpret_cast<T>(dlsym(lib, name));
    return *out != nullptr;
}

// loaded once and never unloaded (every context keeps pointers into it)
const Verbs* verbs(char* err, int errlen) {
    static Verbs v;
    static int state = 0;  // 0 not tried, 1 loaded, -1 failed
    static char why[256];
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (state == 0) {
        v.lib = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
        bool ok = v.lib != nullptr;
        if (!ok) snprintf(why, sizeof why, "dlopen(libibverbs.so.1): %s", dlerror());
#define GB_SYM(f)                                                     \
    if (ok && !sym(v.lib, "ibv_" #f, &v.f)) {                          \
        snprintf(why, sizeof why, "libibverbs.so.1 lacks ibv_%s", #f); \
        ok = false;                                                   \
    }
        GB_SYM(get_device_list)
        GB_SYM(free_device_list)
        GB_SYM(get_device_name)
        GB_SYM(open_device)
        GB_SYM(close_device)
        GB_SYM(query_port)
        GB_SYM(query_gid)
        GB_SYM(alloc_pd)
        GB_SYM(dealloc_pd)
        GB_SYM(reg_mr)
        GB_SYM(dereg_mr)
        GB_SYM(create_cq)
        GB_SYM(destroy_cq)
        GB_SYM(create_qp)
        GB_SYM(destroy_qp)
        GB_SYM(modify_qp)
        GB_SYM(wc_status_str)
#undef GB_SYM
        state = ok ? 1 : -1;
    }
    pthread_mutex_unlock(&mu);
    if (state != 1) {
        gb_roce_fail(err, errlen, "%s", why);
        return nullptr;
    }
    return &v;
}

constexpr int kQps = 2;           // per rail: [0] latency, [1] bulk
constexpr int kSignalEvery = 32;  // one signaled work request in this many per QP (it retires the ones before it)
constexpr int kMaxPieces = 64;    // pieces of one descriptor

struct Qp {
    ibv_qp* qp = nullptr;
    uint32_t psn = 0;
    uint32_t unsignaled = 0;   // work requests since the last signaled one
    uint32_t outstanding = 0;  // work requests posted and not yet retired
};

struct Rail {
    char name[64] = {0};
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_mr* mr = nullptr;
    ibv_cq* cq = nullptr;
    Qp q[kQps];
    int gid_index = -1;
    ibv_gid gid;
    ibv_mtu mtu = IBV_MTU_4096;
    uint32_t peer_rkey = 0;
};

struct Chain {
    ibv_send_wr wr[2 * kMaxPieces];
    ibv_sge sge[2 * kMaxPieces];
    uint64_t flag_val[kMaxPieces];
    int n = 0;
    int flags = 0;
};

}  // namespace

struct GbRoceCtx {
    const Verbs* v = nullptr;
    GbRoceOpen o;
    char hcas[256];
    Rail rail[GB_ROCE_MAX_RAILS];
    int rails = 0;
    uint32_t nonce = 0;
    uint64_t peer_addr = 0;
    uint64_t generation = 0;  // (generation << 32): the high word of every flag value this rank writes
    bool connected = false;
    pthread_t thread;
    bool thread_live = false;
    std::atomic<int> run{0};
    int cpu = -1;
    std::atomic<long> proxy_tid{0};  // the proxy thread's kernel tid (gb_roce_isolate leaves it alone)
    Chain chain[GB_ROCE_MAX_RAILS];
    // REXL3_ROCE_TRACE=<file>: per descriptor (ticket, CLOCK_MONOTONIC ns when seen, ns when posted), written at close
    char trace_path[256] = {0};
    uint64_t* trace = nullptr;
    uint32_t trace_n = 0, trace_cap = 0;
};

namespace {

char* base(GbRoceCtx* c) { return static_cast<char*>(c->o.region); }
volatile uint32_t* abort_word(GbRoceCtx* c) { return reinterpret_cast<volatile uint32_t*>(base(c) + GB_ROCE_H_ABORT); }
GbRoceProxyStat* stat(GbRoceCtx* c) { return reinterpret_cast<GbRoceProxyStat*>(base(c) + GB_ROCE_H_PROXY); }

// The first error wins; the abort word is raised with it, so every spinning kernel of this rank stops (fail-stop).
void record_error(GbRoceCtx* c, uint32_t code, uint32_t site, uint32_t seq, uint32_t detail) {
    GbRoceError* e = reinterpret_cast<GbRoceError*>(base(c) + GB_ROCE_H_ERROR);
    uint32_t zero = 0;
    if (__atomic_compare_exchange_n(&e->code, &zero, code, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        e->site = site;
        e->seq = seq;
        __atomic_store_n(&e->detail, detail, __ATOMIC_RELEASE);
    }
    __atomic_store_n(abort_word(c), 1u, __ATOMIC_RELEASE);
}

int find_gid_index(const char* dev) {
    for (int i = 0; i < 64; ++i) {
        char p[256], t[64] = {0}, g[64] = {0};
        snprintf(p, sizeof p, "/sys/class/infiniband/%s/ports/1/gid_attrs/types/%d", dev, i);
        FILE* f = fopen(p, "r");
        if (!f) continue;
        size_t n = fread(t, 1, sizeof t - 1, f);
        fclose(f);
        if (n == 0 || strncmp(t, "RoCE v2", 7) != 0) continue;
        snprintf(p, sizeof p, "/sys/class/infiniband/%s/ports/1/gids/%d", dev, i);
        f = fopen(p, "r");
        if (!f) continue;
        n = fread(g, 1, sizeof g - 1, f);
        fclose(f);
        if (n >= 30 && strncmp(g, "0000:0000:0000:0000:0000:ffff:", 30) == 0) return i;
    }
    return -1;
}

void close_rail(const Verbs* v, Rail& r) {
    for (Qp& q : r.q)
        if (q.qp) {
            v->destroy_qp(q.qp);
            q.qp = nullptr;
        }
    if (r.cq) v->destroy_cq(r.cq);
    if (r.mr) v->dereg_mr(r.mr);
    if (r.pd) v->dealloc_pd(r.pd);
    if (r.ctx) v->close_device(r.ctx);
    r.cq = nullptr;
    r.mr = nullptr;
    r.pd = nullptr;
    r.ctx = nullptr;
}

int open_rail(GbRoceCtx* c, Rail& r, const char* name, char* err, int errlen) {
    const Verbs* v = c->v;
    snprintf(r.name, sizeof r.name, "%s", name);
    int n = 0;
    ibv_device** list = v->get_device_list(&n);
    if (!list) {
        gb_roce_fail(err, errlen, "ibv_get_device_list: %s", strerror(errno));
        return -ENODEV;
    }
    for (int i = 0; i < n && !r.ctx; ++i)
        if (strcmp(v->get_device_name(list[i]), name) == 0) r.ctx = v->open_device(list[i]);
    v->free_device_list(list);
    if (!r.ctx) {
        gb_roce_fail(err, errlen, "RDMA device %s not found or not openable", name);
        return -ENODEV;
    }
    ibv_port_attr pa;
    memset(&pa, 0, sizeof pa);
    if (v->query_port(r.ctx, 1, &pa) != 0 || pa.state != IBV_PORT_ACTIVE) {
        gb_roce_fail(err, errlen, "%s port 1 is not active", name);
        return -ENODEV;
    }
    r.mtu = pa.active_mtu;
    r.gid_index = c->o.gid_index >= 0 ? c->o.gid_index : find_gid_index(name);
    if (r.gid_index < 0 || v->query_gid(r.ctx, 1, r.gid_index, &r.gid) != 0) {
        gb_roce_fail(err, errlen, "%s: no RoCE v2 IPv4-mapped GID (set REXL3_ROCE_GID_INDEX)", name);
        return -ENODEV;
    }
    r.pd = v->alloc_pd(r.ctx);
    if (!r.pd) {
        gb_roce_fail(err, errlen, "%s: ibv_alloc_pd failed", name);
        return -ENOMEM;
    }
    // NEVER IBV_ACCESS_RELAXED_ORDERING: flag-after-data needs the NIC to place one QP's writes in order (PCI_WR_ORDERING per_mkey)
    r.mr = v->reg_mr(r.pd, c->o.region, (size_t)c->o.region_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!r.mr) {
        const int e = errno;
        struct rlimit rl;
        getrlimit(RLIMIT_MEMLOCK, &rl);
        gb_roce_fail(err, errlen,
                     "%s: ibv_reg_mr of the %llu-byte region failed: %s (RLIMIT_MEMLOCK soft %lld, hard %lld bytes; each rail "
                     "registers the region once. An ssh session may have an 8 MiB limit: start the process from a systemd user "
                     "service, e.g. systemd-run --user --wait --pipe)",
                     name, (unsigned long long)c->o.region_bytes, strerror(e), (long long)rl.rlim_cur, (long long)rl.rlim_max);
        return -ENOMEM;
    }
    r.cq = v->create_cq(r.ctx, 4 * c->o.sq_depth, nullptr, nullptr, 0);
    if (!r.cq) {
        gb_roce_fail(err, errlen, "%s: ibv_create_cq failed", name);
        return -ENOMEM;
    }
    for (int k = 0; k < kQps; ++k) {
        ibv_qp_init_attr a;
        memset(&a, 0, sizeof a);
        a.send_cq = r.cq;
        a.recv_cq = r.cq;
        a.qp_type = IBV_QPT_RC;
        a.sq_sig_all = 0;
        a.cap.max_send_wr = (uint32_t)c->o.sq_depth;
        a.cap.max_recv_wr = 1;
        a.cap.max_send_sge = 1;
        a.cap.max_recv_sge = 1;
        a.cap.max_inline_data = 32;
        r.q[k].qp = v->create_qp(r.pd, &a);
        if (!r.q[k].qp) {
            gb_roce_fail(err, errlen, "%s: ibv_create_qp failed: %s", name, strerror(errno));
            return -ENOMEM;
        }
        ibv_qp_attr q;
        memset(&q, 0, sizeof q);
        q.qp_state = IBV_QPS_INIT;
        q.port_num = 1;
        q.pkey_index = 0;
        q.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        if (v->modify_qp(r.q[k].qp, &q, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            gb_roce_fail(err, errlen, "%s: QP to INIT failed: %s", name, strerror(errno));
            return -EIO;
        }
        r.q[k].psn = (uint32_t)(lrand48() & 0xffffff);
    }
    return 0;
}

// Retire completions on one rail. A failed one is recorded (fail-stop) and returns nonzero.
int drain(GbRoceCtx* c, Rail& r) {
    ibv_wc wc[16];
    const int n = ibv_poll_cq(r.cq, 16, wc);
    if (n < 0) {
        record_error(c, GB_ROCE_E_CQ, 0, 0, 0xffffu);
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            record_error(c, GB_ROCE_E_CQ, (uint32_t)(wc[i].wr_id >> 32), (uint32_t)(wc[i].wr_id & 0xffffffffu),
                         (uint32_t)wc[i].status);
            return -1;
        }
        Qp& q = r.q[(wc[i].wr_id >> 31) & 1];
        q.outstanding = q.outstanding > (uint32_t)kSignalEvery ? q.outstanding - kSignalEvery : 0;
        stat(c)->completions++;
    }
    return 0;
}

// Wait (bounded, ~2 s) until `need` more work requests fit in the send queue of `q`.
int make_room(GbRoceCtx* c, Rail& r, Qp& q, uint32_t need) {
    for (uint64_t spins = 0; q.outstanding + need > (uint32_t)c->o.sq_depth - 2; ++spins) {
        if (drain(c, r)) return -1;
        if (spins > 200000000ull || __atomic_load_n(abort_word(c), __ATOMIC_ACQUIRE)) {
            record_error(c, GB_ROCE_E_SQ, 0, 0, q.outstanding);
            return -1;
        }
    }
    return 0;
}

// Post one descriptor: per rail ONE chain of [data piece, inline flag] pairs, so each rail rings its doorbell once.
int post_desc(GbRoceCtx* c, const GbRoceDesc& d) {
    const uint64_t bytes = (uint64_t)d.bytes_lo | ((uint64_t)d.bytes_hi << 32);
    const uint64_t region = c->o.region_bytes;
    if (bytes == 0 || d.pieces == 0 || d.pieces > kMaxPieces || bytes > region || d.src > region - bytes ||
        d.dst > region - bytes || d.piece_bytes == 0 || (uint64_t)d.pieces * d.piece_bytes < bytes ||
        (uint64_t)d.flag + d.pieces > c->o.flag_slots) {
        record_error(c, GB_ROCE_E_DESC, d.flag, d.seq, d.ticket);
        return -1;
    }
    const int qk = d.bulk ? 1 : 0;
    for (int r = 0; r < c->rails; ++r) c->chain[r].n = c->chain[r].flags = 0;
    uint64_t off = 0;
    for (int k = 0; k < d.pieces && off < bytes; ++k) {
        const int r = (d.rail0 + k) % c->rails;
        Rail& rail = c->rail[r];
        Chain& h = c->chain[r];
        const uint64_t len = bytes - off < d.piece_bytes ? bytes - off : d.piece_bytes;
        // the piece
        ibv_sge& s = h.sge[h.n];
        s.addr = (uint64_t)(uintptr_t)c->o.region + d.src + off;
        s.length = (uint32_t)len;
        s.lkey = rail.mr->lkey;
        ibv_send_wr& w = h.wr[h.n];
        memset(&w, 0, sizeof w);
        w.wr_id = ((uint64_t)(d.flag + k) << 32) | ((uint64_t)qk << 31) | (d.seq & 0x7fffffffu);
        w.sg_list = &s;
        w.num_sge = 1;
        w.opcode = IBV_WR_RDMA_WRITE;
        w.wr.rdma.remote_addr = c->peer_addr + d.dst + off;
        w.wr.rdma.rkey = rail.peer_rkey;
        h.n++;
        // its flag, after it on the same QP, copied inline at post time
        uint64_t& fv = h.flag_val[h.flags++];
        fv = c->generation | d.seq;
        ibv_sge& fs = h.sge[h.n];
        fs.addr = (uint64_t)(uintptr_t)&fv;
        fs.length = 8;
        fs.lkey = 0;
        ibv_send_wr& f = h.wr[h.n];
        memset(&f, 0, sizeof f);
        f.wr_id = w.wr_id;
        f.sg_list = &fs;
        f.num_sge = 1;
        f.opcode = IBV_WR_RDMA_WRITE;
        f.send_flags = IBV_SEND_INLINE;
        f.wr.rdma.remote_addr = c->peer_addr + c->o.flags_off + 8ull * (d.flag + k);
        f.wr.rdma.rkey = rail.peer_rkey;
        h.n++;
        off += len;
    }
    for (int r = 0; r < c->rails; ++r) {
        Chain& h = c->chain[r];
        if (h.n == 0) continue;
        Rail& rail = c->rail[r];
        Qp& q = rail.q[qk];
        if (make_room(c, rail, q, (uint32_t)h.n)) return -1;
        // selective signaling: mark the flag write that crosses the kSignalEvery boundary
        for (int i = 0; i < h.n; ++i) {
            h.wr[i].next = i + 1 < h.n ? &h.wr[i + 1] : nullptr;
            if (++q.unsignaled >= (uint32_t)kSignalEvery) {
                q.unsignaled = 0;
                h.wr[i].send_flags |= IBV_SEND_SIGNALED;
            }
        }
        ibv_send_wr* bad = nullptr;
        const int rc = ibv_post_send(q.qp, &h.wr[0], &bad);
        if (rc != 0) {
            record_error(c, GB_ROCE_E_POST, d.flag, d.seq, (uint32_t)rc);
            return -1;
        }
        q.outstanding += (uint32_t)h.n;
        if (q.outstanding > stat(c)->max_sq) stat(c)->max_sq = q.outstanding;
        stat(c)->wrs += (uint64_t)h.n;
    }
    stat(c)->descs++;
    return 0;
}

// The proxy: spins on the next ring entry's ticket; posts entries in ticket order; retires completions while idle. It never naps:
// a gap between two collectives of a decode step is microseconds, and a sleep costs a scheduler wake-up on the critical path.
void* proxy_main(void* arg) {
    GbRoceCtx* c = static_cast<GbRoceCtx*>(arg);
    if (c->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(c->cpu, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0) c->cpu = -1;
    }
    stat(c)->cpu = (uint32_t)c->cpu;
    c->proxy_tid.store((long)syscall(SYS_gettid));
    __atomic_store_n(&stat(c)->running, 1u, __ATOMIC_RELEASE);
    GbRoceDesc* ring = reinterpret_cast<GbRoceDesc*>(base(c) + c->o.ring_off);
    const volatile uint32_t* bell = reinterpret_cast<const volatile uint32_t*>(base(c) + GB_ROCE_H_BELL);
    const uint32_t mask = c->o.ring_entries - 1;
    uint32_t expect = 1;       // the next ticket to post
    uint32_t newest = 0;       // the newest ticket the doorbell has shown (tickets of concurrent publishers may land out of order)
    uint32_t last_bell = 0;
    uint32_t idle = 0;
    while (c->run.load(std::memory_order_relaxed)) {
        const uint32_t b = __atomic_load_n(bell, __ATOMIC_ACQUIRE);
        if (b != last_bell) {
            last_bell = b;
            if ((int32_t)(b - newest) > 0) newest = b;
        }
        // entries up to the newest ticket exist; post every one that is complete, in ticket order. Every 4096 idle spins the next
        // entry is looked at even without a doorbell (a doorbell overwritten by an older ticket before this thread saw it).
        bool any = false;
        while ((int32_t)(newest - expect) >= 0 || (idle & 4095u) == 4095u) {
            GbRoceDesc* e = &ring[expect & mask];
            if (__atomic_load_n(&e->ticket, __ATOMIC_ACQUIRE) != expect) break;
            uint64_t t_seen = 0;
            if (c->trace) {
                timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                t_seen = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
            }
            GbRoceDesc d = *e;  // after the acquire: every field the GPU wrote before its release
            if (post_desc(c, d)) {
                c->run.store(0);
                break;
            }
            if (c->trace && c->trace_n < c->trace_cap) {
                timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                uint64_t* r = c->trace + 3ull * c->trace_n++;
                r[0] = ((uint64_t)d.flag << 32) | d.seq;
                r[1] = t_seen;
                r[2] = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
            }
            if ((int32_t)(expect - newest) >= 0) newest = expect;
            ++expect;
            any = true;
            idle = 0;
        }
        if (any) continue;
        if ((++idle & 31u) == 0) {
            bool bad = false;
            for (int r = 0; r < c->rails; ++r) bad = bad || drain(c, c->rail[r]) != 0;
            if (bad || __atomic_load_n(abort_word(c), __ATOMIC_ACQUIRE)) break;
        }
    }
    __atomic_store_n(&stat(c)->running, 0u, __ATOMIC_RELEASE);
    return nullptr;
}

}  // namespace

extern "C" {

int gb_roce_available(char* err, int errlen) { return verbs(err, errlen) ? 0 : -ENOSYS; }

int gb_roce_rails(const GbRoceCtx* c) { return c ? c->rails : 0; }

int gb_roce_open(const GbRoceOpen* o, GbRoceCtx** out, char* err, int errlen) {
    *out = nullptr;
    const Verbs* v = verbs(err, errlen);
    if (!v) return -ENOSYS;
    if (!o->region || o->region_bytes < GB_ROCE_HEADER_BYTES || (o->ring_entries & (o->ring_entries - 1)) != 0 ||
        o->ring_off + 64ull * o->ring_entries > o->region_bytes || o->flags_off + 8ull * o->flag_slots > o->region_bytes ||
        o->sq_depth < 16) {
        gb_roce_fail(err, errlen, "gb_roce_open: inconsistent layout");
        return -EINVAL;
    }
    GbRoceCtx* c = new GbRoceCtx();
    c->v = v;
    c->o = *o;
    snprintf(c->hcas, sizeof c->hcas, "%s", o->hcas ? o->hcas : "");
    c->o.hcas = c->hcas;
    srand48((long)time(nullptr) ^ (long)getpid() ^ ((long)o->rank << 20));
    c->nonce = (uint32_t)lrand48() | 1u;
    char names[256];
    snprintf(names, sizeof names, "%s", c->hcas);
    for (char *save = nullptr, *tok = strtok_r(names, ",", &save); tok; tok = strtok_r(nullptr, ",", &save)) {
        while (*tok == ' ' || *tok == '=' || *tok == '^') ++tok;
        char* colon = strchr(tok, ':');
        if (colon) *colon = 0;
        if (!*tok) continue;
        if (c->rails == GB_ROCE_MAX_RAILS) break;
        const int rc = open_rail(c, c->rail[c->rails], tok, err, errlen);
        c->rails++;
        if (rc != 0) {
            gb_roce_close(c);
            return rc;
        }
    }
    if (c->rails == 0) {
        gb_roce_fail(err, errlen, "no RDMA device named (REXL3_ROCE_HCA / NCCL_IB_HCA is empty)");
        gb_roce_close(c);
        return -ENODEV;
    }
    *out = c;
    return 0;
}

int gb_roce_blob(GbRoceCtx* c, void* blob, int len) {
    if (len < (int)sizeof(GbRoceBlob)) return -EINVAL;
    GbRoceBlob b;
    memset(&b, 0, sizeof b);
    b.magic = 0x544c5852u;  // "RXLT"
    b.version = GB_ROCE_VERSION;
    b.region_addr = (uint64_t)(uintptr_t)c->o.region;
    b.region_bytes = c->o.region_bytes;
    b.layout_hash = c->o.layout_hash;
    b.nonce = c->nonce;
    b.rails = (uint32_t)c->rails;
    b.tclass = (uint32_t)c->o.tclass;
    for (int r = 0; r < c->rails; ++r) {
        b.rail[r].rkey = c->rail[r].mr->rkey;
        for (int k = 0; k < kQps; ++k) {
            b.rail[r].qpn[k] = c->rail[r].q[k].qp->qp_num;
            b.rail[r].psn[k] = c->rail[r].q[k].psn;
        }
        memcpy(b.rail[r].gid, c->rail[r].gid.raw, 16);
        b.rail[r].mtu = (uint32_t)c->rail[r].mtu;
    }
    memcpy(blob, &b, sizeof b);
    return (int)sizeof b;
}

int gb_roce_connect(GbRoceCtx* c, const void* peer_blob, int len, uint64_t* generation, char* err, int errlen) {
    if (len < (int)sizeof(GbRoceBlob)) {
        gb_roce_fail(err, errlen, "the peer's blob is %d bytes, expected %d", len, (int)sizeof(GbRoceBlob));
        return -EINVAL;
    }
    GbRoceBlob p;
    memcpy(&p, peer_blob, sizeof p);
    if (p.magic != 0x544c5852u || p.version != GB_ROCE_VERSION) {
        gb_roce_fail(err, errlen, "the peer runs another transport build (magic %08x version %u)", p.magic, p.version);
        return -EPROTO;
    }
    if (p.layout_hash != c->o.layout_hash || p.region_bytes != c->o.region_bytes) {
        gb_roce_fail(err, errlen, "the peer's region layout differs (hash %016llx vs %016llx, %llu vs %llu bytes)",
                     (unsigned long long)p.layout_hash, (unsigned long long)c->o.layout_hash, (unsigned long long)p.region_bytes,
                     (unsigned long long)c->o.region_bytes);
        return -EPROTO;
    }
    if ((int)p.rails != c->rails || (int)p.tclass != c->o.tclass) {
        gb_roce_fail(err, errlen, "the peer has %u rails with traffic class %u, this rank %d with %d", p.rails, p.tclass, c->rails,
                     c->o.tclass);
        return -EPROTO;
    }
    c->peer_addr = p.region_addr;
    // the generation both ranks derive alike: a flag of an earlier communicator never equals one of this one
    const uint32_t gen = (c->nonce ^ p.nonce) | 1u;
    c->generation = (uint64_t)gen << 32;
    *generation = gen;
    for (int r = 0; r < c->rails; ++r) {
        Rail& rail = c->rail[r];
        rail.peer_rkey = p.rail[r].rkey;
        const ibv_mtu mtu = (ibv_mtu)(p.rail[r].mtu < (uint32_t)rail.mtu ? p.rail[r].mtu : (uint32_t)rail.mtu);
        for (int k = 0; k < kQps; ++k) {
            ibv_qp_attr q;
            memset(&q, 0, sizeof q);
            q.qp_state = IBV_QPS_RTR;
            q.path_mtu = mtu;
            q.dest_qp_num = p.rail[r].qpn[k];
            q.rq_psn = p.rail[r].psn[k];
            q.max_dest_rd_atomic = 1;
            q.min_rnr_timer = 12;
            q.ah_attr.is_global = 1;
            q.ah_attr.port_num = 1;
            memcpy(q.ah_attr.grh.dgid.raw, p.rail[r].gid, 16);
            q.ah_attr.grh.sgid_index = (uint8_t)rail.gid_index;
            q.ah_attr.grh.hop_limit = 64;
            q.ah_attr.grh.traffic_class = (uint8_t)c->o.tclass;
            if (c->v->modify_qp(rail.q[k].qp, &q,
                                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                    IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
                gb_roce_fail(err, errlen, "%s: QP %d to RTR failed: %s", rail.name, k, strerror(errno));
                return -EIO;
            }
            memset(&q, 0, sizeof q);
            q.qp_state = IBV_QPS_RTS;
            q.timeout = 14;  // 4.096 us x 2^14 = 67 ms per try; 7 tries: a lost peer surfaces as a CQ error in about 0.5 s
            q.retry_cnt = 7;
            q.rnr_retry = 7;
            q.sq_psn = rail.q[k].psn;
            q.max_rd_atomic = 1;
            if (c->v->modify_qp(rail.q[k].qp, &q,
                                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                    IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
                gb_roce_fail(err, errlen, "%s: QP %d to RTS failed: %s", rail.name, k, strerror(errno));
                return -EIO;
            }
        }
    }
    c->connected = true;
    return 0;
}

int gb_roce_start(GbRoceCtx* c, int cpu, char* err, int errlen) {
    if (!c->connected) {
        gb_roce_fail(err, errlen, "gb_roce_start before gb_roce_connect");
        return -EINVAL;
    }
    if (c->thread_live) return 0;
    c->cpu = cpu;
    if (const char* tp = getenv("REXL3_ROCE_TRACE")) {
        snprintf(c->trace_path, sizeof c->trace_path, "%s", tp);
        c->trace_cap = 1u << 20;
        c->trace = static_cast<uint64_t*>(calloc(3ull * c->trace_cap, 8));
    }
    c->run.store(1);
    if (pthread_create(&c->thread, nullptr, proxy_main, c) != 0) {
        c->run.store(0);
        gb_roce_fail(err, errlen, "pthread_create: %s", strerror(errno));
        return -EAGAIN;
    }
    c->thread_live = true;
    (void)0;
    for (int i = 0; i < 2000 && !__atomic_load_n(&stat(c)->running, __ATOMIC_ACQUIRE); ++i) usleep(500);
    return 0;
}

// Keep the proxy's core to the proxy: every OTHER thread of this process (the engine's, CUDA's, NCCL's) gets its affinity mask without
// that core, so a launching thread is never time-sliced against the spinning proxy. Threads created later inherit the mask of the
// thread that creates them (all of which now exclude the core). Returns the number of threads changed (0 when the proxy is not
// pinned), or a negative errno with a message.
int gb_roce_isolate(GbRoceCtx* c, char* err, int errlen) {
    if (!c || !c->thread_live || c->cpu < 0) return 0;
    const long proxy = c->proxy_tid.load();
    DIR* d = opendir("/proc/self/task");
    if (!d) {
        gb_roce_fail(err, errlen, "gb_roce_isolate: /proc/self/task: %s", strerror(errno));
        return -errno;
    }
    int changed = 0;
    while (dirent* e = readdir(d)) {
        const long tid = strtol(e->d_name, nullptr, 10);
        if (tid <= 0 || tid == proxy) continue;
        cpu_set_t m;
        CPU_ZERO(&m);
        if (sched_getaffinity((pid_t)tid, sizeof m, &m) != 0 || !CPU_ISSET(c->cpu, &m)) continue;
        CPU_CLR(c->cpu, &m);
        if (CPU_COUNT(&m) == 0) continue;  // a thread pinned to that core alone keeps it
        if (sched_setaffinity((pid_t)tid, sizeof m, &m) == 0) ++changed;
    }
    closedir(d);
    return changed;
}

void gb_roce_stop(GbRoceCtx* c) {
    if (!c || !c->thread_live) return;
    c->run.store(0);
    pthread_join(c->thread, nullptr);
    c->thread_live = false;
}

void gb_roce_close(GbRoceCtx* c) {
    if (!c) return;
    gb_roce_stop(c);
    if (c->trace) {
        if (FILE* f = fopen(c->trace_path, "w")) {
            for (uint32_t i = 0; i < c->trace_n; ++i)
                fprintf(f, "%llu %llu %llu %llu\n", (unsigned long long)(c->trace[3 * i] >> 32),
                        (unsigned long long)(c->trace[3 * i] & 0xffffffffu), (unsigned long long)c->trace[3 * i + 1],
                        (unsigned long long)c->trace[3 * i + 2]);
            fclose(f);
        }
        free(c->trace);
    }
    for (int r = 0; r < c->rails; ++r) close_rail(c->v, c->rail[r]);
    delete c;
}

}  // extern "C"

#else  // no <infiniband/verbs.h> on the build host: the transport reports itself unavailable

struct GbRoceCtx {
    int unused;
};

extern "C" {
int gb_roce_available(char* err, int errlen) {
    gb_roce_fail(err, errlen, "built without <infiniband/verbs.h> (install libibverbs-dev on the build host)");
    return -38;  // ENOSYS
}
int gb_roce_open(const GbRoceOpen*, GbRoceCtx** out, char* err, int errlen) {
    *out = nullptr;
    return gb_roce_available(err, errlen);
}
int gb_roce_blob(GbRoceCtx*, void*, int) { return -38; }
int gb_roce_connect(GbRoceCtx*, const void*, int, uint64_t*, char* err, int errlen) { return gb_roce_available(err, errlen); }
int gb_roce_start(GbRoceCtx*, int, char* err, int errlen) { return gb_roce_available(err, errlen); }
int gb_roce_isolate(GbRoceCtx*, char*, int) { return 0; }
void gb_roce_stop(GbRoceCtx*) {}
void gb_roce_close(GbRoceCtx*) {}
int gb_roce_rails(const GbRoceCtx*) { return 0; }
}
#endif
