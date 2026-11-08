// EXL3 encode kernels: trellis-quantization (circular Viterbi) of tiles, at integer and fractional rates, and
// trellis bit-packing/unpacking. Ported from ExLlamaV3's quantize_tiles_kernel.cuh, quantize_tiles_frac_kernel.cuh,
// pack.cu and frac.cu (MIT, Copyright (c) 2025 Turboderp; see THIRD_PARTY_NOTICES.md) with the ATen/PyTorch
// wrappers stripped: self-contained over <cuda_runtime.h>, <cuda_fp16.h>, <cuda_bf16.h>, <cstdint>, common.cuh
// and exl3_linear_decode.cuh. The device code is the reference code, bit for bit; launchers live in
// exl3_encode.cu.
//
// Buffers (all device pointers): input_tiles/output_tiles/output_indices are [tiles, L] rows (L = 256 or 160);
// output_indices hold 16-bit codebook / trellis indices. temp_costs is [batch, 2, edges] half (only K = 1 uses
// it; K >= 2 keep the costs in shared memory) and temp_edges is [batch, L, edges] uint16_t, edges = 65536 >> K
// (integer) or 65536 >> KA (fractional). Both scratch layouts match ExLlamaV3's quantize_tiles, so a caller's
// scratch sizing aligns with the reference.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "common.cuh"
#include "exl3_linear_decode.cuh"
#include "exl3_encode_codebook.cuh"
#include "exl3_encode_quantize_int.cuh"
#include "exl3_encode_quantize_frac.cuh"
#include "exl3_encode_pack.cuh"
