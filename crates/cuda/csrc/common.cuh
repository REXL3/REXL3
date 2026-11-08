// Shared helpers for all kernels. Every exported launcher is `extern "C"`, takes raw device pointers, and returns
// 0 on success or a cudaError_t (checked right after the launch). Streams arrive as `void*` (a cudaStream_t).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define GB_EXPORT extern "C" __attribute__((visibility("default")))
#define GB_CHECK_LAUNCH() (int)cudaGetLastError()

static inline cudaStream_t gb_stream(void* s) { return (cudaStream_t)s; }
