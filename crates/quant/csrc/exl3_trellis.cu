// gb-nvcc: --expt-relaxed-constexpr
// REXL3's EXL3 trellis encoder: a tail-biting Viterbi search over all 65536 16-bit states per position, for tiles of
// 256 weights in trellis order. Rust API: src/exl3_trellis.rs. Our own code.
//
// One thread block works on one job (a tile, or a (tile, wrap node) pair in the exact search) at a time and loops over
// jobs (persistent blocks). Per position p the block evaluates all 65536 16-bit states. Notation for a step:
//   B = D(p) new bits of s_p, A = D(p+1) bits the step reduces over, M = 16 - A - B middle bits;
//   predecessor node = s >> B = (k << M) | mid (k = top A bits of s); successor node e = s & (2^(16-A) - 1) = (mid << B) | j.
// A butterfly (fixed mid) links 2^A predecessors to 2^B successors; lanes own consecutive butterflies.
//
// Arithmetic contract (reference.py reproduces it bit for bit): v = fp16 codebook value (exact in fp32); d = v - x;
// c = fma(d, d, N_prev) (weighted: fma(d * w, d, N_prev)); key = (bits(c) & ~(2^A - 1)) | k; the smallest key wins, so
// the cost is truncated by A mantissa bits and ties go to the smallest top bits k. End-node arg-min: smallest cost,
// ties to the smallest node index. Precision mode H (fp16 costs, ExLlamaV3-like precision): the inputs x and every new
// cost c are rounded to fp16 (nearest even) before use; the key truncation is then a no-op.
//
// Node storage, all in shared memory: fp32 double buffers from ka = 3; one fp32 array updated in place in two phases
// (compute into registers, barrier, write, barrier) at ka = 2; at ka = 1 (2^15 nodes) one in-place array of 24-bit
// costs (fp32 with the low 8 mantissa bits dropped: a u16 high half and a u8 low byte), 96 KB.
// History: per step and successor node, the arg-min top bits in a slot of 1/2/4/8 bits, rows indexed by absolute
// position. Traceback runs on one thread at the end of the pass.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <type_traits>

#include "common.cuh"

namespace {

constexpr int NPOS = 256;
constexpr int THREADS = 512;

// MODE_FREE: no tail-biting. Free start and free end: one pass from position 0 with every start node at cost 0, the
// end node with the smallest cost, traced back. s_0 is then any 16-bit state; its top 16 - D(0) bits (the start node,
// written to `wrap`) are not in the tile's stream and must be stored next to it (a start-state side channel).
enum Mode : int { MODE_ROLLED = 0, MODE_GIVEN = 1, MODE_COST = 2, MODE_DUMP = 3, MODE_FREE = 4 };

struct Params {
    const float* x;         // [n_tiles, 256], trellis order
    const float* w;         // per-position weights: [n_tiles, 256] (w_stride 256) or [256] (w_stride 0); null = 1
    long long w_stride;
    const int* job_tile;    // [n_jobs] tile of each job; null = job index
    const int* job_node;    // [n_jobs] wrap node (MODE_GIVEN, MODE_COST)
    short* states;          // [n_tiles, 256] out (MODE_ROLLED, MODE_GIVEN)
    float* values;          // [n_tiles, 256] out, decoded fp16 values as fp32
    float* cost;            // [n_jobs] out: DP cost at the end node (fp32 contract)
    int* wrap;              // [n_jobs] out: wrap node used (MODE_ROLLED, MODE_GIVEN)
    float* dump;            // [n_jobs, 2^(16 - D(0))] out (MODE_DUMP): final node costs by node index, free start
    unsigned char* hist;    // scratch [gridDim.x, hist_bytes]
    long long hist_bytes;
    int n_jobs;
    int mode;
    int pre;                // MODE_ROLLED: wrap estimate from a free start `pre` positions before the wrap (any
    int post;               // length, laps allowed) through `post` positions after it (<= 256), traced back to the wrap
    int ka;
    unsigned mask;
    int cb;                 // codebook id; the encoder takes 2 (mul1) only
};

__device__ __forceinline__ int dsh(int ka, unsigned mask, int p) { return ka + ((mask >> (p & 15)) & 1); }

__host__ __device__ constexpr int slot_bits(int a) { return a <= 1 ? 1 : a <= 2 ? 2 : a <= 4 ? 4 : 8; }

__device__ __forceinline__ __half mul1_h(uint32_t s) {
    uint32_t x = s * 0x83DCD12Du;
    uint32_t h = __dp4a(x, 0x01010101u, 0x6400u);
    return __hfma(__ushort_as_half((unsigned short)h), __ushort_as_half((unsigned short)0x1EEE),
                  __ushort_as_half((unsigned short)0xC931));
}

__device__ __forceinline__ float mul1_f(uint32_t s) { return __half2float(mul1_h(s)); }

// Node storage: natural node order with an XOR bank swizzle. Lanes own consecutive butterflies, so for a fixed k the
// predecessor reads (k << M) | mid hit consecutive nodes (conflict-free) and the successor writes (mid << B) | j spread
// over all 32 banks once bits 5.. are folded into the bank bits.
__device__ __forceinline__ int sw(int n) { return n ^ ((n >> 5) & 31); }

struct F32Nodes {
    float* p;
    __device__ __forceinline__ float ld(int i) const { return p[i]; }
    __device__ __forceinline__ void st(int i, float v) const { p[i] = v; }
};

// 24-bit costs: the stored value is bits(v) & 0xFFFFFF00 (sign, exponent, 15 mantissa bits)
struct S24Nodes {
    uint16_t* hi;
    uint8_t* lo;
    __device__ __forceinline__ float ld(int i) const {
        return __uint_as_float(((uint32_t)hi[i] << 16) | ((uint32_t)lo[i] << 8));
    }
    __device__ __forceinline__ void st(int i, float v) const {
        const uint32_t b = __float_as_uint(v);
        hi[i] = (uint16_t)(b >> 16);
        lo[i] = (uint8_t)(b >> 8);
    }
};

// Per-step history slot: 2^B * slot >= 8 so a thread's chunk of one butterfly is whole bytes.
__host__ __device__ constexpr int slot_ab(int a, int b) {
    return slot_bits(a) > (8 >> b) ? slot_bits(a) : (8 >> b);
}

template <int A, int B>
struct Geo {
    static constexpr int M = 16 - A - B;
    static constexpr int NMID = 1 << M;
    static constexpr int SUCC = 1 << B;
    static constexpr int MPT = NMID >= THREADS ? NMID / THREADS : 1;  // butterflies per thread
    static constexpr int TPM = NMID >= THREADS ? 1 : THREADS / NMID;  // threads per butterfly
    // successors per thread and butterfly: at least 4 where possible, so one predecessor load serves several states
    static constexpr int JS = TPM * 4 <= SUCC ? SUCC / TPM : (SUCC < 4 ? SUCC : 4);
    static constexpr int G = TPM * JS / SUCC;   // lanes sharing a successor group (their minima are combined)
    static constexpr int KC = (1 << A) / G;     // candidates per lane and successor
    static constexpr int SLOT = slot_ab(A, B);
    static constexpr int CHUNK = JS * SLOT;     // history bits per writer lane and butterfly
    static_assert(MPT * JS * KC * THREADS == 65536, "every state once");
    static_assert(G >= 1 && G <= 32 && (G & (G - 1)) == 0, "lane groups");
    static_assert(CHUNK == 8 || CHUNK == 16 || CHUNK == 32 || CHUNK == 64, "history chunk");
};

// Codebook value cache. At integer rates every step has the same geometry, so a thread evaluates the same 128 states at
// every position of every pass: their fp16 codebook values are computed once per launch and kept in registers as 64
// half2 words. The values are the same bits mul1_f computes, so the search is unchanged; only the work per state
// drops (no hash, no codebook FMA in the inner loop).
constexpr int VC_STATES = 65536 / THREADS;
struct VCache {
    __half2 v[VC_STATES / 2];
};

// (cost bits & ~m) | k as one LOP3. With k a compile-time constant the compiler splits it into an AND and an OR; the
// step keeps k in a register (an opaque mov per predecessor) so the key costs one instruction (same value).
template <uint32_t M>
__device__ __forceinline__ uint32_t key_lop3(uint32_t cbits, uint32_t k) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(cbits), "n"(~M), "r"(k));
    return r;
}

template <int A, int B>
__device__ __forceinline__ void geo_thread(int tid, int i, int& mid, int& j0, int& kbase) {
    using Gm = Geo<A, B>;
    if constexpr (Gm::TPM == 1) {
        mid = i * THREADS + tid;
        j0 = 0;
        kbase = 0;
    } else {
        mid = tid / Gm::TPM;
        const int r = tid % Gm::TPM;
        j0 = (r / Gm::G) * Gm::JS;
        kbase = (r % Gm::G) * Gm::KC;
    }
}

// One trellis step. nodes_in: predecessor costs; nodes_out: successor costs (== nodes_in when INPLACE: two phases).
// Candidate arithmetic: d = v - x; c = fma(d, d, P) (weighted: fma(d * w, d, P)); key = (bits(c) & ~m) | k with
// m = 2^A - 1, so one unsigned min gives the cost (truncated by A mantissa bits) and the arg-min, ties to the smallest k.
template <int A, int B, bool INPLACE, bool W, bool H, bool VC, class N>
__device__ __forceinline__ void step(const N nodes_in, const N nodes_out, float x, float w, unsigned char* hist_row,
                                     int tid, const VCache& vc) {
    using Gm = Geo<A, B>;
    constexpr uint32_t m = (1u << A) - 1u;
    float newc[INPLACE ? Gm::MPT * Gm::JS : 1];
#pragma unroll
    for (int i = 0; i < Gm::MPT; ++i) {
        int mid, j0, kbase;
        geo_thread<A, B>(tid, i, mid, j0, kbase);
        uint32_t best[Gm::JS];
#pragma unroll
        for (int jj = 0; jj < Gm::JS; ++jj) best[jj] = 0xFFFFFFFFu;
        const uint32_t lo = ((uint32_t)mid << B) | (uint32_t)j0;
#pragma unroll
        for (int kq = 0; kq < Gm::KC; ++kq) {
            const uint32_t k = (uint32_t)(kbase + kq);
            const float pk = nodes_in.ld(sw((int)(k << Gm::M) | mid));
            uint32_t kr;
            asm volatile("mov.b32 %0, %1;" : "=r"(kr) : "r"(k));
#pragma unroll
            for (int jj = 0; jj < Gm::JS; ++jj) {
                float v;
                if constexpr (VC) {
                    const int idx = (i * Gm::KC + kq) * Gm::JS + jj;
                    v = (idx & 1) ? __high2float(vc.v[idx >> 1]) : __low2float(vc.v[idx >> 1]);
                } else {
                    v = mul1_f((k << (16 - A)) | (lo + (uint32_t)jj));
                }
                const float d = __fsub_rn(v, x);
                float c = W ? __fmaf_rn(__fmul_rn(d, w), d, pk) : __fmaf_rn(d, d, pk);
                if constexpr (H) c = __half2float(__float2half_rn(c));
                best[jj] = min(best[jj], key_lop3<m>(__float_as_uint(c), kr));
            }
        }
        if constexpr (Gm::G > 1) {
#pragma unroll
            for (int jj = 0; jj < Gm::JS; ++jj) {
#pragma unroll
                for (int off = 1; off < Gm::G; off <<= 1) best[jj] = min(best[jj], __shfl_xor_sync(0xffffffffu, best[jj], off));
            }
        }
        const bool writer = Gm::G == 1 || (tid % Gm::G) == 0;
        uint64_t chunk = 0;
#pragma unroll
        for (int jj = 0; jj < Gm::JS; ++jj) {
            chunk |= (uint64_t)(best[jj] & m) << (jj * Gm::SLOT);
            const float cost = __uint_as_float(best[jj] & ~m);
            if constexpr (INPLACE) {
                newc[i * Gm::JS + jj] = cost;
            } else if (writer) {
                nodes_out.st(sw((int)(lo + jj)), cost);
            }
        }
        if (hist_row != nullptr && writer) {
            unsigned char* dst = hist_row + ((lo * Gm::SLOT) >> 3);  // the chunk sits at bit lo * SLOT, size-aligned
            if constexpr (Gm::CHUNK == 64) *reinterpret_cast<uint64_t*>(dst) = chunk;
            else if constexpr (Gm::CHUNK == 32) *reinterpret_cast<uint32_t*>(dst) = (uint32_t)chunk;
            else if constexpr (Gm::CHUNK == 16) *reinterpret_cast<uint16_t*>(dst) = (uint16_t)chunk;
            else *dst = (unsigned char)chunk;
        }
    }
    if constexpr (INPLACE) {
        __syncthreads();
        if (Gm::G == 1 || (tid % Gm::G) == 0) {
#pragma unroll
            for (int i = 0; i < Gm::MPT; ++i) {
                int mid, j0, kbase;
                geo_thread<A, B>(tid, i, mid, j0, kbase);
                const int lo = (mid << B) | j0;
#pragma unroll
                for (int jj = 0; jj < Gm::JS; ++jj) nodes_out.st(sw(lo + jj), newc[i * Gm::JS + jj]);
            }
        }
    }
    __syncthreads();
}

template <int KA, bool MASKED>
struct Cfg {
    static constexpr bool S24 = KA == 1;     // 2^15 nodes: 24-bit costs, in place
    static constexpr bool DOUBLE = KA >= 3;  // fp32 double buffers fit from ka = 3; ka = 2 updates in place
    // the codebook value cache (integer rates from ka = 3; at ka = 1 and 2 the in-place step needs its registers: spills)
    static constexpr bool VC = !MASKED && DOUBLE;
    static constexpr int MAXW = 1 << (16 - KA);
    static constexpr size_t NODE_BYTES = S24 ? (size_t)MAXW * 3 : (size_t)MAXW * 4 * (DOUBLE ? 2 : 1);
    using Nodes = typename std::conditional<S24, S24Nodes, F32Nodes>::type;
};

// dispatch one step at position p
template <int KA, bool MASKED, bool W, bool H, class N>
__device__ __forceinline__ void do_step(const Params& P, const N& buf0, const N& buf1, int& cur, float x, float w, int p,
                                        unsigned char* hist_row, int tid, const VCache& vc) {
    using C = Cfg<KA, MASKED>;
    const int b = dsh(P.ka, P.mask, p), a = dsh(P.ka, P.mask, p + 1);
    const N in = cur ? buf1 : buf0;
    const N out = C::DOUBLE ? (cur ? buf0 : buf1) : in;
    if constexpr (!MASKED) {
        step<KA, KA, !C::DOUBLE, W, H, C::VC>(in, out, x, w, hist_row, tid, vc);
    } else {
        constexpr int K1 = KA < 8 ? KA + 1 : KA;
        const int sel = (a - KA) * 2 + (b - KA);
        switch (sel) {
            case 0: step<KA, KA, !C::DOUBLE, W, H, false>(in, out, x, w, hist_row, tid, vc); break;
            case 1: step<KA, K1, !C::DOUBLE, W, H, false>(in, out, x, w, hist_row, tid, vc); break;
            case 2: step<K1, KA, !C::DOUBLE, W, H, false>(in, out, x, w, hist_row, tid, vc); break;
            default: step<K1, K1, !C::DOUBLE, W, H, false>(in, out, x, w, hist_row, tid, vc); break;
        }
    }
    if (C::DOUBLE) cur ^= 1;
}

struct Shared {
    float x[NPOS];
    float w[NPOS];
    int rowpre[17];  // history row offsets: row p starts at (p >> 4) * rowpre[16] + rowpre[p & 15] (period 16)
    float rbest[THREADS / 32];
    int rarg[THREADS / 32];
    int bcast;
};

__device__ __forceinline__ int rowoff(const Shared& sh, int p) { return (p >> 4) * sh.rowpre[16] + sh.rowpre[p & 15]; }

// block-wide arg-min over the final nodes (width 16 - b): ties -> smallest node index
template <class N>
__device__ void argmin_nodes(const N nodes, int b, Shared& sh, int tid, float* out_cost, int* out_node) {
    const int width = 1 << (16 - b);
    float best = __int_as_float(0x7f800000);
    int arg = 0x7fffffff;
    for (int i = tid; i < width; i += THREADS) {
        float c = nodes.ld(i);
        int e = sw(i);  // the swizzle is an involution
        if (c < best || (c == best && e < arg)) {
            best = c;
            arg = e;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ob = __shfl_xor_sync(0xffffffffu, best, off);
        int oa = __shfl_xor_sync(0xffffffffu, arg, off);
        if (ob < best || (ob == best && oa < arg)) {
            best = ob;
            arg = oa;
        }
    }
    if ((tid & 31) == 0) {
        sh.rbest[tid >> 5] = best;
        sh.rarg[tid >> 5] = arg;
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < THREADS / 32; ++i) {
            if (sh.rbest[i] < best || (sh.rbest[i] == best && sh.rarg[i] < arg)) {
                best = sh.rbest[i];
                arg = sh.rarg[i];
            }
        }
        *out_cost = best;
        *out_node = arg;
    }
    __syncthreads();
}

// trace back `count` positions starting at p_first (ring order), from end node e; returns the node before p_first
__device__ int traceback(const Params& P, const unsigned char* hist, const Shared& sh, int p_first, int count, int e,
                         short* st_out) {
    for (int i = count - 1; i >= 0; --i) {
        const int p = (p_first + i) & (NPOS - 1);
        const int a = dsh(P.ka, P.mask, p + 1);
        const int sb = slot_ab(a, dsh(P.ka, P.mask, p));
        const unsigned char* row = hist + rowoff(sh, p);
        const int bit = e * sb;
        const int k = (row[bit >> 3] >> (bit & 7)) & ((1 << a) - 1);
        const int s = (k << (16 - a)) | e;
        if (st_out) st_out[p] = (short)s;
        e = s >> dsh(P.ka, P.mask, p);
    }
    return e;
}

template <int KA, bool MASKED, bool W, bool H>
__global__ void __launch_bounds__(THREADS, (KA >= 4 && !Cfg<KA, MASKED>::VC ? 2 : 1)) viterbi_kernel(Params P) {
    using C = Cfg<KA, MASKED>;
    using N = typename C::Nodes;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Shared& sh = *reinterpret_cast<Shared*>(smem_raw);
    unsigned char* sbuf = smem_raw + ((sizeof(Shared) + 15) / 16) * 16;
    const int tid = threadIdx.x;
    N buf0, buf1;
    if constexpr (C::S24) {
        buf0.hi = reinterpret_cast<uint16_t*>(sbuf);
        buf0.lo = reinterpret_cast<uint8_t*>(sbuf + (size_t)C::MAXW * 2);
        buf1 = buf0;
    } else {
        buf0.p = reinterpret_cast<float*>(sbuf);
        buf1.p = C::DOUBLE ? buf0.p + C::MAXW : buf0.p;
    }
    unsigned char* hist = P.hist + (size_t)blockIdx.x * P.hist_bytes;

    if (tid == 0) {
        int off = 0;
        for (int p = 0; p < 16; ++p) {
            sh.rowpre[p] = off;
            const int a = dsh(P.ka, P.mask, p + 1);
            off += ((1 << (16 - a)) * slot_ab(a, dsh(P.ka, P.mask, p))) >> 3;
        }
        sh.rowpre[16] = off;
    }

    VCache vc;
    if constexpr (C::VC) {
        using Gm = Geo<KA, KA>;
        static_assert(Gm::MPT * Gm::KC * Gm::JS == VC_STATES, "cache covers the thread's states");
#pragma unroll
        for (int i = 0; i < Gm::MPT; ++i) {
            int mid, j0, kbase;
            geo_thread<KA, KA>(tid, i, mid, j0, kbase);
            const uint32_t lo = ((uint32_t)mid << KA) | (uint32_t)j0;
#pragma unroll
            for (int kq = 0; kq < Gm::KC; ++kq) {
#pragma unroll
                for (int jj = 0; jj < Gm::JS; jj += 2) {
                    const uint32_t s = ((uint32_t)(kbase + kq) << (16 - KA)) | (lo + (uint32_t)jj);
                    vc.v[((i * Gm::KC + kq) * Gm::JS + jj) >> 1] = __halves2half2(mul1_h(s), mul1_h(s + 1));
                }
            }
        }
    }

    for (int job = blockIdx.x; job < P.n_jobs; job += gridDim.x) {
        const int tile = P.job_tile ? P.job_tile[job] : job;
        __syncthreads();
        if (tid < NPOS) {
            const float xv = P.x[(size_t)tile * NPOS + tid];
            sh.x[tid] = H ? __half2float(__float2half_rn(xv)) : xv;
            sh.w[tid] = W ? P.w[(size_t)tile * P.w_stride + tid] : 1.0f;
        }
        int e0 = 0;
        int cur = 0;
        // ---- wrap estimate (MODE_ROLLED): free start `pre` positions before the wrap, history on [0, post)
        if (P.mode == MODE_ROLLED) {
            const int p0 = (NPOS - (P.pre % NPOS)) % NPOS;
            {
                const int width = 1 << (16 - dsh(P.ka, P.mask, p0));
                __syncthreads();
                for (int i = tid; i < width; i += THREADS) buf0.st(i, 0.0f);
                __syncthreads();
            }
            for (int i = 0; i < P.pre + P.post; ++i) {
                const int p = (p0 + i) & (NPOS - 1);
                unsigned char* row = i >= P.pre ? hist + rowoff(sh, p) : nullptr;
                do_step<KA, MASKED, W, H>(P, buf0, buf1, cur, sh.x[p], sh.w[p], p, row, tid, vc);
            }
            const int pl = (P.post - 1) & (NPOS - 1);
            float c;
            argmin_nodes(cur ? buf1 : buf0, dsh(P.ka, P.mask, pl + 1), sh, tid, &c, &sh.bcast);
            if (tid == 0) sh.bcast = traceback(P, hist, sh, 0, P.post, sh.bcast, nullptr);
            __syncthreads();
            e0 = sh.bcast;
        } else if (P.mode == MODE_GIVEN || P.mode == MODE_COST) {
            e0 = P.job_node[job];
        }
        // ---- main pass over positions 0..255
        {
            cur = 0;
            const int width = 1 << (16 - dsh(P.ka, P.mask, 0));
            __syncthreads();
            const bool free_start = P.mode == MODE_DUMP || P.mode == MODE_FREE;
            for (int i = tid; i < width; i += THREADS) buf0.st(i, free_start ? 0.0f : __int_as_float(0x7f800000));
            __syncthreads();
            if (!free_start && tid == 0) buf0.st(sw(e0), 0.0f);
            __syncthreads();
        }
        const bool keep = P.mode == MODE_ROLLED || P.mode == MODE_GIVEN || P.mode == MODE_FREE;
        for (int p = 0; p < NPOS; ++p) {
            unsigned char* row = keep ? hist + rowoff(sh, p) : nullptr;
            do_step<KA, MASKED, W, H>(P, buf0, buf1, cur, sh.x[p], sh.w[p], p, row, tid, vc);
        }
        const N fin = cur ? buf1 : buf0;
        // after position 255 the nodes have width 16 - D(0)
        const int bf = dsh(P.ka, P.mask, 0);
        if (P.mode == MODE_DUMP) {
            const int width = 1 << (16 - bf);
            for (int i = tid; i < width; i += THREADS) P.dump[(size_t)job * width + sw(i)] = fin.ld(i);
            continue;
        }
        if (P.mode == MODE_FREE) {
            float c = 0.0f;
            argmin_nodes(fin, bf, sh, tid, &c, &sh.bcast);  // ends on a barrier: every thread sees the end node
            if (tid == 0) {
                const int start = traceback(P, hist, sh, 0, NPOS, sh.bcast, P.states + (size_t)tile * NPOS);
                P.cost[job] = c;
                P.wrap[job] = start;
            }
            __syncthreads();
            if (tid < NPOS) P.values[(size_t)tile * NPOS + tid] = mul1_f((uint16_t)P.states[(size_t)tile * NPOS + tid]);
            continue;
        }
        if (tid == 0) P.cost[job] = fin.ld(sw(e0));
        if (P.mode == MODE_COST) continue;
        if (tid == 0) {
            short* st = P.states + (size_t)tile * NPOS;
            traceback(P, hist, sh, 0, NPOS, e0, st);
            P.wrap[job] = e0;
        }
        __syncthreads();
        if (tid < NPOS) {
            const uint32_t s = (uint16_t)P.states[(size_t)tile * NPOS + tid];
            P.values[(size_t)tile * NPOS + tid] = mul1_f(s);
        }
    }
}

template <int KA, bool MASKED, bool W, bool H>
struct Inst {
    static size_t smem() { return ((sizeof(Shared) + 15) / 16) * 16 + Cfg<KA, MASKED>::NODE_BYTES; }
    static const void* fn() { return (const void*)viterbi_kernel<KA, MASKED, W, H>; }
};

typedef size_t (*SmemFn)();
typedef const void* (*FnFn)();

struct Entry {
    SmemFn smem;
    FnFn fn;
};

template <bool MASKED, bool W, bool H>
Entry pick_ka(int ka) {
    switch (ka) {
        case 1: return {Inst<1, MASKED, W, H>::smem, Inst<1, MASKED, W, H>::fn};
        case 2: return {Inst<2, MASKED, W, H>::smem, Inst<2, MASKED, W, H>::fn};
        case 3: return {Inst<3, MASKED, W, H>::smem, Inst<3, MASKED, W, H>::fn};
        case 4: return {Inst<4, MASKED, W, H>::smem, Inst<4, MASKED, W, H>::fn};
        case 5: return {Inst<5, MASKED, W, H>::smem, Inst<5, MASKED, W, H>::fn};
        case 6: return {Inst<6, MASKED, W, H>::smem, Inst<6, MASKED, W, H>::fn};
        case 7: return {Inst<7, MASKED, W, H>::smem, Inst<7, MASKED, W, H>::fn};
        default: return {Inst<8, false, W, H>::smem, Inst<8, false, W, H>::fn};
    }
}

template <bool H>
Entry pick_h(int ka, bool masked, bool weighted) {
    if (masked) return weighted ? pick_ka<true, true, H>(ka) : pick_ka<true, false, H>(ka);
    return weighted ? pick_ka<false, true, H>(ka) : pick_ka<false, false, H>(ka);
}

bool pick(int ka, unsigned mask, int weighted, int fp16, Entry* e) {
    if (ka < 1 || ka > 8 || mask > 0xFFFFu || (ka == 8 && mask)) return false;
    *e = fp16 ? pick_h<true>(ka, mask != 0, weighted != 0) : pick_h<false>(ka, mask != 0, weighted != 0);
    return true;
}

long long hist_bytes_for(int ka, unsigned mask) {
    long long off = 0;
    for (int p = 0; p < NPOS; ++p) {
        int a = ka + (int)((mask >> ((p + 1) & 15)) & 1);
        int b = ka + (int)((mask >> (p & 15)) & 1);
        off += ((1LL << (16 - a)) * slot_ab(a, b)) >> 3;
    }
    return off;
}

}  // namespace

// Sizes and occupancy for one configuration. Returns a cudaError_t (0 = success).
GB_EXPORT int gb_exl3_trellis_query(int ka, unsigned mask, int weighted, int fp16, int* threads, long long* smem_bytes,
                            int* blocks_per_sm, long long* hist_bytes, int* binary_version, int* num_regs) {
    Entry e;
    if (!pick(ka, mask, weighted, fp16, &e)) return (int)cudaErrorInvalidValue;
    const void* fn = e.fn();
    size_t sm = e.smem();
    cudaError_t err = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (err != cudaSuccess) return (int)err;
    int nb = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, THREADS, sm);
    if (err != cudaSuccess) return (int)err;
    cudaFuncAttributes attr;
    err = cudaFuncGetAttributes(&attr, fn);
    if (err != cudaSuccess) return (int)err;
    *threads = THREADS;
    *smem_bytes = (long long)sm;
    *blocks_per_sm = nb;
    *hist_bytes = hist_bytes_for(ka, mask);
    *binary_version = attr.binaryVersion;
    *num_regs = attr.numRegs;
    return 0;
}

// Launch. Pointers are device pointers; hist must hold grid * hist_bytes (0 for MODE_COST and MODE_DUMP).
GB_EXPORT int gb_exl3_trellis_launch(const float* x, const float* w, long long w_stride, const int* job_tile,
                             const int* job_node, short* states, float* values, float* cost, int* wrap, float* dump,
                             unsigned char* hist, long long hist_bytes, int n_jobs, int mode, int pre,
                             int post, int ka, unsigned mask, int cb, int fp16, int grid, void* stream) {
    Entry e;
    if (!pick(ka, mask, w != nullptr, fp16, &e)) return (int)cudaErrorInvalidValue;
    if (mode < MODE_ROLLED || mode > MODE_FREE || n_jobs <= 0 || grid <= 0) return (int)cudaErrorInvalidValue;
    if (mode == MODE_ROLLED && (pre < 0 || pre > 4096 || post < 1 || post > NPOS)) return (int)cudaErrorInvalidValue;
    if (cb != 2) return (int)cudaErrorInvalidValue;  // the encoder searches the mul1 codebook
    const bool keep = mode == MODE_ROLLED || mode == MODE_GIVEN || mode == MODE_FREE;
    if (!x || (keep && (!states || !values || !hist || !cost || !wrap)) ||
        (mode != MODE_ROLLED && mode != MODE_DUMP && mode != MODE_FREE && !job_node) ||
        (mode == MODE_COST && !cost) || (mode == MODE_DUMP && !dump))
        return (int)cudaErrorInvalidValue;
    Params P;
    P.x = x;
    P.w = w;
    P.w_stride = w_stride;
    P.job_tile = job_tile;
    P.job_node = job_node;
    P.states = states;
    P.values = values;
    P.cost = cost;
    P.wrap = wrap;
    P.dump = dump;
    P.hist = hist;
    P.hist_bytes = hist_bytes;
    P.n_jobs = n_jobs;
    P.mode = mode;
    P.pre = pre;
    P.post = post;
    P.ka = ka;
    P.mask = mask;
    P.cb = cb;
    size_t sm = e.smem();
    cudaError_t err = cudaFuncSetAttribute(e.fn(), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (err != cudaSuccess) return (int)err;
    void* args[] = {&P};
    err = cudaLaunchKernel(e.fn(), dim3(grid), dim3(THREADS), args, sm, (cudaStream_t)stream);
    if (err != cudaSuccess) return (int)err;
    return (int)cudaGetLastError();
}

GB_EXPORT const char* gb_exl3_trellis_error_string(int code) { return cudaGetErrorString((cudaError_t)code); }
