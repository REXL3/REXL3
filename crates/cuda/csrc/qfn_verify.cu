// gb-nvcc: --fmad=false
// Flash-Next greedy verification gather (`src/qfn_verify.rs`): packs the argmax column of an
// already-computed top-k window (glue.cu's gb_topk, sorted by (value DESC, id ASC)) into dense
// per-row arrays, so a greedy round reads back rows * 4 bytes instead of rows * count * 8. A row
// whose candidates contain a NaN is reported as NaN, mirroring `try_choose`'s NanLogit refusal.
// One thread per row; no cross-row interaction, so row bits depend on the row alone.
#include "common.cuh"

namespace {

__global__ void qfn_greedy_gather_k(const int* __restrict__ ids, const float* __restrict__ vals,
                                    int count, int rows, int* __restrict__ out_ids,
                                    float* __restrict__ out_vals) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const size_t base = (size_t)r * count;
    bool nan = false;
    for (int c = 0; c < count; ++c) {
        if (isnan(vals[base + c])) {
            nan = true;
            break;
        }
    }
    out_ids[r] = ids[base];
    out_vals[r] = nan ? NAN : vals[base];
}

}  // namespace

extern "C" {

GB_EXPORT int gb_qfn_greedy_gather(const void* ids, const void* vals, int count, int rows,
                                   void* out_ids, void* out_vals, void* stream) {
    if (!ids || !vals || !out_ids || !out_vals || rows < 0 || count <= 0 || count > 64) return 1;
    if (rows == 0) return 0;
    qfn_greedy_gather_k<<<(rows + 127) / 128, 128, 0, gb_stream(stream)>>>(
        (const int*)ids, (const float*)vals, count, rows, (int*)out_ids, (float*)out_vals);
    return GB_CHECK_LAUNCH();
}

}  // extern "C"
