//! Tile encoding through `crate::exl3_trellis` (tail-biting, fp32 costs, the (128, 128) wrap window).

use crate::core::dev::{Gpu, Result};
use crate::exl3_trellis::{Encoded, Options};
use gb_cuda::rt::DevBuf;
use gb_exl3::Bits;

/// Encodes `n_tiles` tiles of `x` into `out` (sized for at least `n_tiles`), asynchronously on the context's stream.
pub fn encode_into(g: &mut Gpu, x: &DevBuf, n_tiles: usize, bits: Bits, out: &mut Encoded) -> Result<()> {
    let Gpu { encoder, stream, .. } = g;
    encoder.encode(x, n_tiles, bits, &Options::default(), out, stream)?;
    Ok(())
}

/// Encodes `n_tiles` tiles (256 fp32 each, trellis order) of `x`: the states and decoded values on the device.
pub fn encode_tiles(g: &mut Gpu, x: &DevBuf, n_tiles: usize, bits: Bits) -> Result<Encoded> {
    let mut out = Encoded::alloc(n_tiles)?;
    let Gpu { encoder, stream, .. } = g;
    encoder.encode(x, n_tiles, bits, &Options::default(), &mut out, stream)?;
    Ok(out)
}
