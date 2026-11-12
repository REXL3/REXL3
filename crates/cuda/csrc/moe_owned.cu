// gb-nvcc: --expt-relaxed-constexpr --fmad=false
// Whole-expert ownership for the routed experts of moe_experts.cu (decode) and moe_prefill.cu (batched): a rank holds a bank of
// E_local experts (ascending global id) and runs the UNCHANGED expert kernels on remapped picks. This file holds the one kernel that
// is new: the remap of the replicated router picks (global ids) to the rank's local ids, and the -0.0 filler of the pair rows the rank
// does not compute.
//
//   pick_local[p] = local_of[pick[p]] when 0 <= pick[p] < E_global, else -1   (-1: absent here; every expert kernel skips it)
//   y[p, :]       = -0.0 for every pair with pick_local[p] < 0                (absent here or invalid everywhere)
//
// Why -0.0: the ranks' y buffers are summed with disjoint support (f32 add, no flush to zero), and x + (-0.0) == x for every x, -0.0
// and subnormals included, so the sum equals the single-node y bit for bit (docs/NEMOTRON_TP.md section 2). Rows the rank owns are
// written later by the down stage of either path and are not touched here, so the order against those launches does not matter.
// Integer work and constant stores only: no arithmetic of the experts happens here.

#include <stdint.h>

#include "common.cuh"
#include "pdl.cuh"

using gbp_pdl::pdl_trigger;
using gbp_pdl::pdl_wait;

namespace {

// Block (pair): thread 0 remaps; every thread writes its share of the filler row when the pair is absent.
__global__ void __launch_bounds__(128) own_remap_kernel(const int* __restrict__ pick, const int* __restrict__ local_of,
                                                        int* __restrict__ pick_local, float* __restrict__ y, int D, int E) {
    pdl_wait();                                  // pick is the predecessor's output
    const int p = blockIdx.x;
    const int e = pick[p];
    const int l = (e >= 0 && e < E) ? local_of[e] : -1;
    pdl_trigger();
    if (threadIdx.x == 0) pick_local[p] = l;
    if (l >= 0 || !y) return;
    const float nz = __int_as_float(0x80000000);
    float4* row = reinterpret_cast<float4*>(y + (size_t)p * D);
    for (int i = threadIdx.x; i < D / 4; i += blockDim.x) row[i] = make_float4(nz, nz, nz, nz);
}

}  // namespace

// pick_local [P] i32 from pick [P] (global ids) and local_of [E] (local id or -1); y [P, D] fp32 gets -0.0 in every row whose
// pick_local is -1 (y null: no filler). P = rows * topk. D a multiple of 4.
GB_EXPORT int gb_moe_own_remap(const void* pick, const void* local_of, void* pick_local, void* y, int P, int D, int E, void* stream) {
    if (P < 1 || E < 1 || D <= 0 || D % 4 || !pick || !local_of || !pick_local) return (int)cudaErrorInvalidValue;
    return gbp_pdl::launch_pdl(gbp_pdl::on(gbp_pdl::kMoe), own_remap_kernel, dim3((unsigned)P), dim3(128), 0, gb_stream(stream),
                               static_cast<const int*>(pick), static_cast<const int*>(local_of), static_cast<int*>(pick_local),
                               static_cast<float*>(y), D, E);
}
