#include "common.cuh"

__global__ void scale_add_kernel(const float* a, float* b, int n, float s, float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * s + t;
}

GB_EXPORT int gb_smoke_scale_add(const float* a, float* b, int n, float s, float t, void* stream) {
    scale_add_kernel<<<(n + 255) / 256, 256, 0, gb_stream(stream)>>>(a, b, n, s, t);
    return GB_CHECK_LAUNCH();
}
