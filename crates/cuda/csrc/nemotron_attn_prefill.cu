// Flash-style tensor-core prefill attention of the Nemotron 3.5 Super VL NoPE layers: 32 query heads of 128 over 2 kv heads, no RoPE,
// scale 128^-0.5, the fp16 KV cache of csrc/nemotron_attn.cu in 256-token pages behind a block table.
//
// This is a SECOND arithmetic. csrc/nemotron_attn.cu is the exact reduction tree (512-key chunks, scalar fp32). The engine serves a row
// with the tree below an absolute position threshold and with this file at and above it, whichever entry point the row arrives by
// (decode, verify window or prefill: nemotron_attn_prefill.rs, FLASH_FROM_POS), so one arithmetic, one path id, per position: the same
// attention, summed in another order above the threshold, in exchange for the tensor cores (docs/NEMOTRON_ATTN_LONG.md).
//
//   gb_nemo_flash_prefill  one call = three launches (all of them read device buffers only, so a CUDA graph can capture the call):
//     flash_plan_kernel     one thread: the call's rows at position >= `from` as row blocks (up to 8 consecutive rows of one slot).
//     flash_partial_kernel  one CTA per (row block, kv head, split). A row block is up to 8 rows of one slot; each of the 8 warps owns
//                           ONE row and all 16 query heads of the kv head (the m16 of m16n8k16), so a K or V tile is read once per CTA
//                           and used by 16 heads x 8 rows. The split is FL_KC = 8192 absolute keys; inside it the 64-key tiles go in
//                           ascending order through an online softmax in the exp2 domain: S = Q K^T by mma (fp16 in, fp32 out; q is
//                           carried as an fp16 hi + lo pair, so the scores see q at fp32 accuracy), fp32 running max and denominator,
//                           P rounded to fp16 once per tile for O += P V by mma. Writes (m, l, o[128]) per (row, split, query head).
//     flash_merge_kernel    one CTA per (row, query head): the splits fold left to right in ascending split index, o / l.
//
// A row below `from` is touched by none of the three (its output belongs to the tree), so a call may hold rows of both and be launched
// for the deepest row's split count and an upper bound of row blocks whatever the rows are: a launch made for more splits or blocks than
// a call needs gives the same bits, which is what lets one captured graph serve a range of contexts.
//
// Bits. A row's output is a function of its own query, its slot's keys up to its position, and nothing else: the tile and split
// boundaries are absolute key indices (multiples of 64 and of FL_KC from key 0), the warp that owns the row runs exactly the tiles
// 0..pos/64 of each of its splits in order (a tile wholly past the row's position is skipped by that warp, a diagonal tile is masked by
// absolute key index, masked scores are -inf and contribute exact zeros), and the splits merge in index order. A row's m16 tile holds
// only that row's heads, and one mma output element depends only on its own A row and B column, so the other rows of the block, how
// many there are, and where a slice happened to start change no bit. The conditional rescale of O (skipped when no row's max moved)
// multiplies by exactly 1.0 when it is skipped.
#include <stdint.h>

#include "common.cuh"
#include "nemotron_kv8.cuh"

namespace {

constexpr int NQ = 32, NKV = 2, GRP = 16, D = 128, PAGE = 256, TILE = 64;
constexpr int KV_ROW = NKV * D;                  // halves per token, all kv heads
constexpr int FL_ROWS = 8;                       // rows (warps) a CTA serves
constexpr int FL_THREADS = FL_ROWS * 32;
constexpr int FL_KC = 8192;                      // keys of one split: nemotron_attn_prefill.rs FLASH_CHUNK_KEYS
constexpr int ROW_BYTES = D * 2;                 // one key's 128 halves of one kv head: 256 bytes, 16 chunks of 16
constexpr int TILE_BYTES = TILE * ROW_BYTES;     // one K or V tile, 16 KiB
constexpr int STAGES = 3;                        // tiles in flight: the tile being used and two loading
constexpr int SMEM_BYTES = STAGES * 2 * TILE_BYTES;  // 96 KiB: a block may take 99 KiB on this part
// `--kv-dtype int8` (nemotron_kv8.cuh): the stages hold int8 tiles (64 keys x 128 bytes of K and of V) and the 64 keys' exponents of both
// kv heads (128 bytes each); after a tile lands the CTA dequantises it, exactly, into ONE fp16 K tile and ONE fp16 V tile in the swizzled
// layout above, and the score and P.V code reads those: everything after the load is the fp16 kernel's arithmetic.
constexpr int ROW8 = D;                                // one key's 128 int8 values of one kv head
constexpr int TILE8 = TILE * ROW8;                     // 8 KiB
constexpr int EXP_TILE = TILE * NKV;                   // 128 bytes: the tile's exponents, both kv heads
constexpr int STAGE8 = 2 * TILE8 + 2 * EXP_TILE;       // 16,640 bytes
constexpr int SMEM8_BYTES = STAGES * STAGE8 + 2 * TILE_BYTES;  // 82,688 bytes

// Shared tiles are stored with the 16-byte chunk c of key row r at chunk c ^ (r & 7): the 8 rows one ldmatrix reads (consecutive keys,
// one logical chunk) fall in 8 different bank groups, and a cp.async store of a row is a permutation of its chunks.
__device__ __forceinline__ uint32_t swz(int key, int chunk) { return (uint32_t)(key * ROW_BYTES + ((chunk ^ (key & 7)) << 4)); }
constexpr float SCALE_LOG2 = 0.08838834764831845f * 1.4426950408889634f;  // 128^-0.5 * log2(e)

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// x and y as an fp16 pair and the fp16 pair of what the first rounding left (x - hi): hi + lo carries 22 mantissa bits, so the scores
// take q at fp32 accuracy, as the tree does, at the cost of a second mma per score tile.
__device__ __forceinline__ void split_h2(float x, float y, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(x, y);
    const float2 hf = __half22float2(h);
    const __half2 l = __floats2half2_rn(x - hf.x, y - hf.y);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// 16 bytes global -> shared; `valid` false zero-fills the 16 bytes without reading global memory.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    const int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// The row blocks of a call: blocks[0].x is their count, blocks[1 + b] = (first row, rows <= 8) of block b. A block holds consecutive rows
// of one slot that are all at position >= `from`; a row below it ends the block. One thread: a call is at most a few thousand rows.
__global__ void flash_plan_kernel(const int* __restrict__ slot, const int* __restrict__ pos, int rows, int from,
                                  int2* __restrict__ blocks) {
    if (threadIdx.x != 0) return;
    int nb = 0, first = 0, n = 0, cur = -1;
    for (int r = 0; r < rows; ++r) {
        if (pos[r] < from) {
            if (n > 0) blocks[1 + nb++] = make_int2(first, n);
            n = 0;
            continue;
        }
        const int s = slot[r];
        if (n > 0 && n < FL_ROWS && s == cur) {
            ++n;
        } else {
            if (n > 0) blocks[1 + nb++] = make_int2(first, n);
            first = r;
            n = 1;
            cur = s;
        }
    }
    if (n > 0) blocks[1 + nb++] = make_int2(first, n);
    blocks[0] = make_int2(nb, 0);
}

// grid (row block bound, kv head, split); blocks past the plan's count exit. `q`, `slot`, `pos` are indexed by the call's row (the
// launcher offsets them to the pass's first row); the partials of row r sit at (r * stride + split) * 32 + head.
template <bool Q8>
__global__ void __launch_bounds__(FL_THREADS, 1)
flash_partial_kernel(const float* __restrict__ q, const void* __restrict__ kc_, const void* __restrict__ vc_,
                     const int* __restrict__ table, int max_blocks, const int* __restrict__ slot, const int* __restrict__ pos,
                     const int2* __restrict__ blocks, int stride, float2* __restrict__ part_ml, float* __restrict__ part_o) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const __half* __restrict__ kc = (const __half*)kc_;
    const __half* __restrict__ vc = (const __half*)vc_;
    // fp16: the K stages then the V stages. int8: the int8 stages, then the one dequantised fp16 K tile and V tile (kb0, vb0)
    const uint32_t kb0 = smem_u32(smem_raw) + (Q8 ? STAGES * STAGE8 : 0);
    const uint32_t vb0 = kb0 + (Q8 ? TILE_BYTES : STAGES * TILE_BYTES);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
    if (blockIdx.x >= blocks[0].x) return;  // CTA-uniform: the call has fewer row blocks than the launch was made for
    const int2 blk = blocks[1 + blockIdx.x];
    const int g = blockIdx.y, c = blockIdx.z;
    const int r0 = blk.x, nr = blk.y;
    int pmax = 0;
    for (int i = 0; i < nr; ++i) pmax = max(pmax, pos[r0 + i]);
    const int c_start = c * FL_KC;
    if (c_start > pmax) return;  // CTA-uniform: the split lies past every row of the block
    const int key_end = min(c_start + FL_KC, pmax + 1);
    const int ntiles = (key_end - c_start + TILE - 1) / TILE;
    const int* tab = table + slot[r0] * max_blocks;
    const bool active = warp < nr;
    const int row = r0 + (active ? warp : 0);
    const int my_pos = active ? pos[row] : -1;

    // the warp's A operand: Q of its row, 16 heads of the kv head, 8 k steps of 16 over the head dim, as fp16 hi + lo
    uint32_t qa[8][4], ql[8][4];
    if (active) {
        const float* qr = q + ((size_t)row * NQ + g * GRP) * D;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int c0 = ks * 16 + 2 * t4, c1 = c0 + 8;
            const float2 x0 = *reinterpret_cast<const float2*>(qr + (size_t)g8 * D + c0);
            const float2 x1 = *reinterpret_cast<const float2*>(qr + (size_t)(g8 + 8) * D + c0);
            const float2 x2 = *reinterpret_cast<const float2*>(qr + (size_t)g8 * D + c1);
            const float2 x3 = *reinterpret_cast<const float2*>(qr + (size_t)(g8 + 8) * D + c1);
            split_h2(x0.x, x0.y, qa[ks][0], ql[ks][0]);
            split_h2(x1.x, x1.y, qa[ks][1], ql[ks][1]);
            split_h2(x2.x, x2.y, qa[ks][2], ql[ks][2]);
            split_h2(x3.x, x3.y, qa[ks][3], ql[ks][3]);
        }
    }

    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = neg_inf(), m1 = neg_inf(), l0 = 0.f, l1 = 0.f;

    // one tile of K and one of V (64 keys of the kv head, 256 bytes each) into `stage`; keys past the block's deepest row are zero-filled
    auto load_tile = [&](int t, int stage) {
        const int ts = c_start + t * TILE;
        if constexpr (Q8) {
            const size_t pg = (size_t)tab[ts / PAGE] * kv8::PAGE_BYTES;  // a tile never straddles a page
            const int kt0 = ts % PAGE;
            const uint32_t sb = smem_u32(smem_raw) + stage * STAGE8;
            const uint8_t* k8 = (const uint8_t*)kc_ + pg;
            const uint8_t* v8 = (const uint8_t*)vc_ + pg;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int c = tid + FL_THREADS * i, key = c >> 3, ch = c & 7;
                const bool ok = ts + key <= pmax;
                const size_t src = (size_t)(kt0 + (ok ? key : 0)) * KV_ROW + g * D + ch * 16;
                cp_async16(sb + key * ROW8 + ch * 16, k8 + src, ok);
                cp_async16(sb + TILE8 + key * ROW8 + ch * 16, v8 + src, ok);
            }
            // the exponents of the tile's 64 keys, both kv heads (in the page, so in bounds; a key past pmax is zeroed at dequantisation)
            if (tid < 8) cp_async16(sb + 2 * TILE8 + tid * 16, k8 + kv8::EXP_OFF + kt0 * NKV + tid * 16, true);
            else if (tid < 16) cp_async16(sb + 2 * TILE8 + EXP_TILE + (tid - 8) * 16, v8 + kv8::EXP_OFF + kt0 * NKV + (tid - 8) * 16, true);
            return;
        }
        const size_t tok0 = (size_t)tab[ts / PAGE] * PAGE + (ts % PAGE);  // a tile never straddles a page
        const uint32_t kb = kb0 + stage * TILE_BYTES, vb = vb0 + stage * TILE_BYTES;
        const int ch = tid & 15;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int key = (tid >> 4) + 16 * i;
            const bool ok = ts + key <= pmax;
            const size_t src = (tok0 + (ok ? key : 0)) * KV_ROW + g * D + ch * 8;
            cp_async16(kb + swz(key, ch), kc + src, ok);
            cp_async16(vb + swz(key, ch), vc + src, ok);
        }
    };

    // a pipeline of STAGES - 1 tiles ahead: every iteration waits for tile t, then starts tile t + 2 into the stage tile t - 1 used.
    // Empty groups are committed too, so `cp_wait<STAGES - 2>` always means "tile t has landed".
    load_tile(0, 0);
    cp_commit();
    if (1 < ntiles) load_tile(1, 1);
    cp_commit();
    for (int t = 0; t < ntiles; ++t) {
        cp_wait<STAGES - 2>();
        __syncthreads();  // tile t is resident; every warp is done with the stage the next load overwrites
        if (t + STAGES - 1 < ntiles) load_tile(t + STAGES - 1, (t + STAGES - 1) % STAGES);
        cp_commit();
        const int ts = c_start + t * TILE;
        if constexpr (Q8) {
            // every warp dequantises a quarter of the tile into the fp16 tiles (their last readers passed the barrier above); keys past
            // the block's deepest row are zeros, as the fp16 path's zero-fill
            const uint8_t* sb = smem_raw + (t % STAGES) * STAGE8;
            uint8_t* kh = smem_raw + STAGES * STAGE8;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c = tid + FL_THREADS * i, key = c >> 4, ch = c & 15;
                const bool ok = ts + key <= pmax;
                const uint2 rk = *reinterpret_cast<const uint2*>(sb + key * ROW8 + ch * 8);
                const uint2 rv = *reinterpret_cast<const uint2*>(sb + TILE8 + key * ROW8 + ch * 8);
                const int ek = (int)(int8_t)sb[2 * TILE8 + key * NKV + g];
                const int ev = (int)(int8_t)sb[2 * TILE8 + EXP_TILE + key * NKV + g];
                const uint4 z = make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(kh + swz(key, ch)) = ok ? kv8::dq8_i8(rk, ek) : z;
                *reinterpret_cast<uint4*>(kh + TILE_BYTES + swz(key, ch)) = ok ? kv8::dq8_i8(rv, ev) : z;
            }
            __syncthreads();  // the fp16 tiles are complete
        }
        if (!active || ts > my_pos) continue;  // warp-uniform: this row has no key in the tile
        const uint32_t kb = Q8 ? kb0 : kb0 + (t % STAGES) * TILE_BYTES, vb = Q8 ? vb0 : vb0 + (t % STAGES) * TILE_BYTES;

        // S = Q K^T: 8 key n-tiles of 8, 8 k steps of 16 over the head dim; one ldmatrix gives two n-tiles' fragments
        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int jp = 0; jp < 4; ++jp) {
                const int krow = jp * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = ks * 2 + ((lane >> 3) & 1);
                uint32_t b[4];
                ldsm_x4(b, kb + swz(krow, chunk));
                mma16816(s[2 * jp], qa[ks], b[0], b[1]);
                mma16816(s[2 * jp + 1], qa[ks], b[2], b[3]);
                mma16816(s[2 * jp], ql[ks], b[0], b[1]);
                mma16816(s[2 * jp + 1], ql[ks], b[2], b[3]);
            }
        }
        // the diagonal tile: keys above the row's position take no part (absolute key index)
        if (ts + TILE - 1 > my_pos) {
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int key = ts + nt * 8 + 2 * t4;
                if (key > my_pos) s[nt][0] = s[nt][2] = neg_inf();
                if (key + 1 > my_pos) s[nt][1] = s[nt][3] = neg_inf();
            }
        }
        float tm0 = neg_inf(), tm1 = neg_inf();
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            tm0 = fmaxf(tm0, fmaxf(s[nt][0], s[nt][1]));
            tm1 = fmaxf(tm1, fmaxf(s[nt][2], s[nt][3]));
        }
#pragma unroll
        for (int o2 = 1; o2 <= 2; o2 <<= 1) {
            tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, o2));
            tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, o2));
        }
        // the running max in the exp2 domain; the tile's first key is the row's own or earlier, so n0 and n1 are finite
        const float n0 = fmaxf(m0, tm0 * SCALE_LOG2), n1 = fmaxf(m1, tm1 * SCALE_LOG2);
        const float c0 = fast_exp2(m0 - n0), c1 = fast_exp2(m1 - n1);  // exp2(-inf) = 0 on the first tile
        m0 = n0;
        m1 = n1;
        l0 *= c0;
        l1 *= c1;
        if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                o[i][0] *= c0;
                o[i][1] *= c0;
                o[i][2] *= c1;
                o[i][3] *= c1;
            }
        }
        // P = exp2(s * scale - max): the denominator takes it in fp32, the P.V product takes it rounded to fp16 once
        uint32_t pk[8][2];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const float p0 = fast_exp2(fmaf(s[nt][0], SCALE_LOG2, -n0));
            const float p1 = fast_exp2(fmaf(s[nt][1], SCALE_LOG2, -n0));
            const float p2 = fast_exp2(fmaf(s[nt][2], SCALE_LOG2, -n1));
            const float p3 = fast_exp2(fmaf(s[nt][3], SCALE_LOG2, -n1));
            l0 += p0 + p1;
            l1 += p2 + p3;
            pk[nt][0] = pack_h2(p0, p1);
            pk[nt][1] = pack_h2(p2, p3);
        }
        // O += P V: 4 k steps of 16 keys, 16 n-tiles of 8 head dims; V fragments by ldmatrix.trans (V is [key][d])
#pragma unroll
        for (int kc16 = 0; kc16 < 4; ++kc16) {
            const uint32_t pa[4] = {pk[2 * kc16][0], pk[2 * kc16][1], pk[2 * kc16 + 1][0], pk[2 * kc16 + 1][1]};
            const int vrow = kc16 * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
            for (int dp = 0; dp < 8; ++dp) {
                const int chunk = dp * 2 + (lane >> 4);
                uint32_t b[4];
                ldsm_x4_t(b, vb + swz(vrow, chunk));
                mma16816(o[2 * dp], pa, b[0], b[1]);
                mma16816(o[2 * dp + 1], pa, b[2], b[3]);
            }
        }
    }

    if (!active || c_start > my_pos) return;  // this row has no key in the split: nothing is written, nothing is merged
#pragma unroll
    for (int o2 = 1; o2 <= 2; o2 <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, o2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, o2);
    }
    const size_t base = ((size_t)row * stride + c) * NQ + g * GRP;
    if (t4 == 0) {
        part_ml[base + g8] = make_float2(m0, l0);
        part_ml[base + g8 + 8] = make_float2(m1, l1);
    }
    float* pa = part_o + (base + g8) * D;
    float* pb = part_o + (base + g8 + 8) * D;
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
        const int d = dn * 8 + 2 * t4;
        *reinterpret_cast<float2*>(pa + d) = make_float2(o[dn][0], o[dn][1]);
        *reinterpret_cast<float2*>(pb + d) = make_float2(o[dn][2], o[dn][3]);
    }
}

// grid (row, query head), one thread per dimension. The splits of the row (position / FL_KC + 1 of them) fold left to right.
__global__ void __launch_bounds__(D) flash_merge_kernel(const int* __restrict__ pos, const float2* __restrict__ part_ml,
                                                       const float* __restrict__ part_o, int stride, int from,
                                                       float* __restrict__ out) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (pos[r] < from) return;  // CTA-uniform: the tree's row
    const int chunks = pos[r] / FL_KC + 1;
    float M = neg_inf(), L = 0.f, O = 0.f;
    for (int c = 0; c < chunks; ++c) {
        const size_t base = ((size_t)r * stride + c) * NQ + h;
        const float2 ml = part_ml[base];
        const float oc = part_o[base * D + d];
        const float Mn = fmaxf(M, ml.x);
        const float a = (M == neg_inf()) ? 0.f : fast_exp2(M - Mn);
        const float b = fast_exp2(ml.x - Mn);
        L = L * a + ml.y * b;
        O = O * a + oc * b;
        M = Mn;
    }
    out[((size_t)r * NQ + h) * D + d] = O / L;
}

}  // namespace

template <bool Q8>
static int flash_launch(const float* q, const void* kc, const void* vc, const int* table, int max_blocks, const int* slot,
                        const int* pos, void* blocks, int rows, int from, int row_blocks, int chunks, int stride, void* part_ml,
                        void* part_o, float* out, void* stream) {
    if (rows <= 0 || row_blocks <= 0 || chunks <= 0) return 0;
    constexpr int smem = Q8 ? SMEM8_BYTES : SMEM_BYTES;
    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(flash_partial_kernel<Q8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return (int)e;
        attr_set = true;
    }
    cudaStream_t s = gb_stream(stream);
    flash_plan_kernel<<<1, 32, 0, s>>>(slot, pos, rows, from, (int2*)blocks);
    int rc = GB_CHECK_LAUNCH();
    if (rc != 0) return rc;
    dim3 grid(row_blocks, NKV, chunks);
    flash_partial_kernel<Q8><<<grid, FL_THREADS, smem, s>>>(q, kc, vc, table, max_blocks, slot, pos, (const int2*)blocks, stride,
                                                            (float2*)part_ml, (float*)part_o);
    rc = GB_CHECK_LAUNCH();
    if (rc != 0) return rc;
    dim3 mgrid(rows, NQ);
    flash_merge_kernel<<<mgrid, D, 0, s>>>(pos, (const float2*)part_ml, (const float*)part_o, stride, from, out);
    return GB_CHECK_LAUNCH();
}

// `blocks`: device scratch for the plan (`1 + row_blocks` int2). `rows` rows of one pass, `from` the first position this arithmetic
// serves, `row_blocks` an upper bound of the row blocks `rows` can make, `chunks` the split count to launch for (at least the deepest
// row's), `stride` the splits one row's partials are spaced by (at least `chunks`). Row indices are relative to `q`, `slot`, `pos` and
// `out` as passed (the caller offsets the pointers to the pass's first row).
GB_EXPORT int gb_nemo_flash_prefill(const float* q, const void* kc, const void* vc, const int* table, int max_blocks, const int* slot,
                                    const int* pos, void* blocks, int rows, int from, int row_blocks, int chunks, int stride,
                                    void* part_ml, void* part_o, float* out, void* stream) {
    return flash_launch<false>(q, kc, vc, table, max_blocks, slot, pos, blocks, rows, from, row_blocks, chunks, stride, part_ml, part_o,
                               out, stream);
}

// The same over an int8 page (`--kv-dtype int8`).
GB_EXPORT int gb_nemo_flash_prefill_i8(const float* q, const void* kc, const void* vc, const int* table, int max_blocks,
                                       const int* slot, const int* pos, void* blocks, int rows, int from, int row_blocks, int chunks,
                                       int stride, void* part_ml, void* part_o, float* out, void* stream) {
    return flash_launch<true>(q, kc, vc, table, max_blocks, slot, pos, blocks, rows, from, row_blocks, chunks, stride, part_ml,
                                        part_o, out, stream);
}
