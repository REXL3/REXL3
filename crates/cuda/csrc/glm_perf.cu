// Kernels of the GLM decode-performance work that are not part of the model's arithmetic.
//
// gb_spin: a one-thread kernel that keeps its stream busy for a number of clock cycles. It exists for the mutation tests of the stream joins: a side
// branch that is made slow on purpose turns "the consumer did not wait for it" from a race that usually works into a result that is wrong every time.

#include "common.cuh"

namespace {

__global__ void spin_kernel(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
}

}  // namespace

// gb_map_ids: ids[i] = map[ids[i]] for i < n (a draft chosen among the columns of a restricted vocabulary head becomes the token id of that column;
// an index outside the map is left as it is: the top-k never produces one).
namespace {

__global__ void map_ids_kernel(int* ids, const int* map, int n, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int c = ids[i];
        if (c >= 0 && c < cols) ids[i] = map[c];
    }
}

}  // namespace

GB_EXPORT int gb_map_ids(int* ids, const int* map, int n, int cols, void* stream) {
    if (n < 0 || cols < 0 || (n > 0 && (!ids || !map))) return (int)cudaErrorInvalidValue;
    if (n == 0) return 0;
    map_ids_kernel<<<(n + 127) / 128, 128, 0, gb_stream(stream)>>>(ids, map, n, cols);
    return GB_CHECK_LAUNCH();
}

GB_EXPORT int gb_spin(long long cycles, void* stream) {
    if (cycles < 0) return (int)cudaErrorInvalidValue;
    spin_kernel<<<1, 1, 0, gb_stream(stream)>>>(cycles);
    return GB_CHECK_LAUNCH();
}
