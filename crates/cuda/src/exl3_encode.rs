//! The EXL3 quantizer kernels on the GPU (`csrc/exl3_encode.cu`): the forward path of
//! quantization — tiles of floats to fp16 + trellis indices — and the device-side trellis
//! packing/unpacking the encoder uses. The C ABI functions are declared with
//! [`crate::launchers!`], so a stub build (no nvcc, `cfg(gb_no_cuda)`) compiles the same
//! wrappers with failing stubs and every call returns an error instead of the crate needing a
//! cfg on each wrapper.
//!
//! Tile geometry (as the CPU reference [`gb_exl3::encode`] packs): a 16x16 tile is 256 values;
//! its trellis is `16 * bits` `i16` words, where `bits` is the width (`k_bits` for integer
//! widths, `k2 / 2` for half-integer ones — see [`gb_exl3::Bits`]). `codebook` codes are the
//! `gb_exl3::Codebook` discriminants: 0 = 3inst, 1 = mcg, 2 = mul1.

use crate::rt::{CudaError, DevBuf, Stream};

#[cfg(not(gb_no_cuda))]
mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_exl3_quantize_tiles(
            input_tiles: *const f32,
            output_tiles: *mut f32,
            output_indices: *mut u16,
            temp_costs: *mut c_void,
            temp_edges: *mut u16,
            num_tiles: i32,
            tile_len: i32,
            k_bits: i32,
            codebook: i32,
            max_batch: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_exl3_quantize_tiles_frac(
            input_tiles: *const f32,
            output_tiles: *mut f32,
            output_indices: *mut u16,
            temp_costs: *mut c_void,
            temp_edges: *mut u16,
            num_tiles: i32,
            ka: i32,
            mask: u32,
            max_batch: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_exl3_pack_trellis(
            packed: *mut u16,
            unpacked: *const u16,
            kt: i32,
            nt: i32,
            k2: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_exl3_unpack_trellis(
            unpacked: *mut u16,
            packed: *const u16,
            kt: i32,
            nt: i32,
            k2: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// The error a stub launcher returns (no CUDA in this build).
const NO_CUDA: CudaError = CudaError(-1);

/// Quantize `num_tiles` tiles of `tile_len` floats each at integer width `k_bits`: fp16
/// reconstructions into `output_tiles`, trellis indices into `output_indices`, on `stream`
/// (asynchronous). `temp_costs` / `temp_edges` are scratch the kernel writes; `max_batch`
/// caps the tiles one kernel launch processes. `codebook` is the [`gb_exl3::Codebook`]
/// discriminant (0 = 3inst, 1 = mcg, 2 = mul1).
#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_tiles_raw(
    input_tiles: *const f32,
    output_tiles: *mut f32,
    output_indices: *mut u16,
    temp_costs: *mut core::ffi::c_void,
    temp_edges: *mut u16,
    num_tiles: i32,
    tile_len: i32,
    k_bits: i32,
    codebook: i32,
    max_batch: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_exl3_quantize_tiles(
            input_tiles,
            output_tiles,
            output_indices,
            temp_costs,
            temp_edges,
            num_tiles,
            tile_len,
            k_bits,
            codebook,
            max_batch,
            stream,
        ))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (
            input_tiles,
            output_tiles,
            output_indices,
            temp_costs,
            temp_edges,
            num_tiles,
            tile_len,
            k_bits,
            codebook,
            max_batch,
            stream,
        );
        Err(NO_CUDA)
    }
}

/// [`quantize_tiles_raw`] on device buffers. `temp_costs` and `temp_edges` are caller-owned
/// scratch (see the kernel's sizing contract in `csrc/exl3_encode.cu`).
#[allow(clippy::too_many_arguments)]
pub fn quantize_tiles(
    input_tiles: &DevBuf,
    output_tiles: &mut DevBuf,
    output_indices: &mut DevBuf,
    temp_costs: &mut DevBuf,
    temp_edges: &mut DevBuf,
    num_tiles: i32,
    tile_len: i32,
    k_bits: i32,
    codebook: i32,
    max_batch: i32,
    stream: &Stream,
) -> Result<(), CudaError> {
    unsafe {
        quantize_tiles_raw(
            input_tiles.ptr() as *const f32,
            output_tiles.ptr() as *mut f32,
            output_indices.ptr() as *mut u16,
            temp_costs.ptr(),
            temp_edges.ptr() as *mut u16,
            num_tiles,
            tile_len,
            k_bits,
            codebook,
            max_batch,
            stream.raw(),
        )
    }
}

/// Quantize `num_tiles` tiles at half-integer width: `ka` is the integer part (1 for 1.5,
/// 2 for 2.5, 3 for 3.5 bits) and `mask` the fractional step mask (`0xaaaa`: alternating
/// `ka` / `ka + 1` bit steps), mul1 codebook only, `tile_len` 256. Same buffers and contract
/// as [`quantize_tiles_raw`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_tiles_frac_raw(
    input_tiles: *const f32,
    output_tiles: *mut f32,
    output_indices: *mut u16,
    temp_costs: *mut core::ffi::c_void,
    temp_edges: *mut u16,
    num_tiles: i32,
    ka: i32,
    mask: u32,
    max_batch: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_exl3_quantize_tiles_frac(
            input_tiles,
            output_tiles,
            output_indices,
            temp_costs,
            temp_edges,
            num_tiles,
            ka,
            mask,
            max_batch,
            stream,
        ))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (input_tiles, output_tiles, output_indices, temp_costs, temp_edges, num_tiles, ka, mask, max_batch, stream);
        Err(NO_CUDA)
    }
}

/// [`quantize_tiles_frac_raw`] on device buffers.
#[allow(clippy::too_many_arguments)]
pub fn quantize_tiles_frac(
    input_tiles: &DevBuf,
    output_tiles: &mut DevBuf,
    output_indices: &mut DevBuf,
    temp_costs: &mut DevBuf,
    temp_edges: &mut DevBuf,
    num_tiles: i32,
    ka: i32,
    mask: u32,
    max_batch: i32,
    stream: &Stream,
) -> Result<(), CudaError> {
    unsafe {
        quantize_tiles_frac_raw(
            input_tiles.ptr() as *const f32,
            output_tiles.ptr() as *mut f32,
            output_indices.ptr() as *mut u16,
            temp_costs.ptr(),
            temp_edges.ptr() as *mut u16,
            num_tiles,
            ka,
            mask,
            max_batch,
            stream.raw(),
        )
    }
}

/// Pack a trellis of unpacked 16-bit window states (`[kt, nt, 256]` `u16`, stream order) into
/// its packed `[kt, nt, 16 * bits]` `i16` words (`k2` = twice the width: 3 for 1.5, 5 for
/// 2.5, 7 for 3.5, and `2 * bits` for integer widths). Asynchronous on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn pack_trellis_raw(
    packed: *mut u16,
    unpacked: *const u16,
    kt: i32,
    nt: i32,
    k2: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_exl3_pack_trellis(packed, unpacked, kt, nt, k2, stream))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (packed, unpacked, kt, nt, k2, stream);
        Err(NO_CUDA)
    }
}

/// [`pack_trellis_raw`] on device buffers.
pub fn pack_trellis(packed: &mut DevBuf, unpacked: &DevBuf, kt: i32, nt: i32, k2: i32, stream: &Stream) -> Result<(), CudaError> {
    unsafe { pack_trellis_raw(packed.ptr() as *mut u16, unpacked.ptr() as *const u16, kt, nt, k2, stream.raw()) }
}

/// Unpack a packed trellis `[kt, nt, 16 * bits]` `i16` words into the unpacked 16-bit window
/// states `[kt, nt, 256]` (`k2` = twice the width, as [`pack_trellis_raw`]). The device mirror
/// of [`gb_exl3::encode::unpack_trellis_states`]. Asynchronous on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn unpack_trellis_raw(
    unpacked: *mut u16,
    packed: *const u16,
    kt: i32,
    nt: i32,
    k2: i32,
    stream: *mut core::ffi::c_void,
) -> Result<(), CudaError> {
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(ffi::gb_exl3_unpack_trellis(unpacked, packed, kt, nt, k2, stream))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (unpacked, packed, kt, nt, k2, stream);
        Err(NO_CUDA)
    }
}

/// [`unpack_trellis_raw`] on device buffers.
pub fn unpack_trellis(
    unpacked: &mut DevBuf,
    packed: &DevBuf,
    kt: i32,
    nt: i32,
    k2: i32,
    stream: &Stream,
) -> Result<(), CudaError> {
    unsafe { unpack_trellis_raw(unpacked.ptr() as *mut u16, packed.ptr() as *const u16, kt, nt, k2, stream.raw()) }
}
