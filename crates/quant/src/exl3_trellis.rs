//! REXL3's EXL3 trellis encoder (`csrc/exl3_trellis.cu`): tiles of 256 fp32 weights in trellis order become the 16-bit
//! state of every position and the fp16 values those states decode to, by a tail-biting Viterbi search over all 65,536
//! states a position. It is the encoder of our quantizer: the Python quantizer (`tools/rexl3_quant/trellis`) and this
//! API run the same compiled kernels.
//!
//! **Arithmetic contract** (bit for bit with the NumPy reference `tools/rexl3_quant/trellis/reference.py`): `v` is the
//! mul1 codebook value of a state (fp16, exact in fp32), `d = v - x`, the new cost is `c = fma(d, d, N_prev)` in fp32,
//! and a step keeps the smallest key `(bits(c) & !(2^A - 1)) | k`, so costs are truncated by `A = D(p + 1)` mantissa
//! bits and ties go to the smallest top bits `k`; the end node is the arg-min cost, ties to the smallest node.
//! [`Costs::Fp16`] rounds the inputs and every new cost to fp16 instead (ExLlamaV3-like precision, for ablations; fp16
//! costs overflow above 65504, so unit-scale tiles only).
//!
//! **Tail biting**: a free-start pass from `pre` positions before the wrap to `post` positions after it, traced back
//! to the wrap, estimates the wrap node; one pass constrained to that node gives the ring ([`Window`]: (128, 128) is
//! the rotate-by-half two-pass scheme and the default, (256, 256) a full-lap warm-up for about 1.2x the time and ~0.3%
//! less error).
//!
//! **Independence**: one thread block per tile with all scratch reinitialised, so a tile's result does not depend on
//! its batch, the grid or call order, and repeated calls are bit-identical.
//!
//! The safe API takes the format's widths ([`Bits`]: 1, 1.5, 2, 2.5, 3, 3.5 and 4..8 bits; a half width shifts
//! `ka + 1` bits into the odd positions, mask `0xAAAA`). [`raw`] is the whole C ABI: any even-popcount 16-bit mask,
//! per-position weights, and the given-wrap, cost-only and free-start dump modes the Python exact search is built on.
//!
//! ```text
//! let out = gb_quant::exl3_trellis::encode_host(&tiles, Bits::from_twice(5).unwrap(), &Options::default())?;
//! // out.states[t * 256 + p]: the 16-bit state of position p of tile t; out.values: its fp16 value as f32
//! ```

use gb_cuda::rt::{CudaError, DevBuf, Stream};
use gb_exl3::Bits;
use std::fmt;

/// Weights (positions) in a tile.
pub const POSITIONS: usize = 256;
/// The codebook the encoder searches: mul1 (`gb_exl3::Codebook::Mul1`, id 2).
pub const CODEBOOK_MUL1: i32 = 2;
/// The mask of the half widths: odd positions shift `ka + 1` bits.
pub const HALF_MASK: u32 = 0xAAAA;
/// Largest `pre` of a [`Window`] (the warm-up may lap the ring).
pub const MAX_PRE: u32 = 4096;
/// Default cap on the history scratch, as the Python encoder's `DEFAULT_SCRATCH`.
pub const DEFAULT_MAX_SCRATCH_BYTES: usize = 2 << 30;

/// The C ABI of `csrc/exl3_trellis.cu`. Device pointers, asynchronous on `stream`; every call returns 0 or a
/// `cudaError_t`. A stub build (no nvcc) gets functions that return `cudaErrorNoDevice`.
pub mod raw {
    /// Wrap node estimated from a free-start window pass, then one constrained pass (states, values, cost, wrap).
    pub const MODE_ROLLED: i32 = 0;
    /// Constrained to `job_node[job]` (states, values, cost, wrap).
    pub const MODE_GIVEN: i32 = 1;
    /// The DP cost of the best ring through `job_node[job]` only (cost).
    pub const MODE_COST: i32 = 2;
    /// Free start: every final node's cost, `dump[job, node]` (no history scratch).
    pub const MODE_DUMP: i32 = 3;

    crate::gpu::foreign! {
        crate::gpu::CUDA_ERROR_NO_DEVICE;
        /// Threads, dynamic shared memory, occupancy, history bytes per job, the kernel's binary version (SASS arch,
        /// e.g. 121) and its registers, for one configuration. `weighted` and `fp16` are 0 or 1.
        pub fn gb_exl3_trellis_query(
            ka: i32,
            mask: u32,
            weighted: i32,
            fp16: i32,
            threads: *mut i32,
            smem_bytes: *mut i64,
            blocks_per_sm: *mut i32,
            hist_bytes: *mut i64,
            binary_version: *mut i32,
            num_regs: *mut i32,
        ) -> i32;
        /// One launch of `grid` persistent blocks over `n_jobs` jobs. `x` [tiles, 256] f32; `w` null or weights with
        /// `w_stride` 256 ([tiles, 256]) or 0 ([256]); `job_tile` null (job = tile) or [n_jobs]; `job_node` [n_jobs]
        /// (MODE_GIVEN, MODE_COST); `states` i16 and `values` f32 [tiles, 256]; `cost` f32 and `wrap` i32 [n_jobs];
        /// `dump` f32 [n_jobs, 2^(16 - D(0))]; `hist` at least `grid * hist_bytes` bytes (MODE_ROLLED, MODE_GIVEN).
        /// `pre` 0..=4096 and `post` 1..=256 (MODE_ROLLED); `cb` must be 2 (mul1).
        pub fn gb_exl3_trellis_launch(
            x: *const f32,
            w: *const f32,
            w_stride: i64,
            job_tile: *const i32,
            job_node: *const i32,
            states: *mut i16,
            values: *mut f32,
            cost: *mut f32,
            wrap: *mut i32,
            dump: *mut f32,
            hist: *mut u8,
            hist_bytes: i64,
            n_jobs: i32,
            mode: i32,
            pre: i32,
            post: i32,
            ka: i32,
            mask: u32,
            cb: i32,
            fp16: i32,
            grid: i32,
            stream: *mut core::ffi::c_void,
        ) -> i32;
    }

    #[cfg(not(gb_quant_no_cuda))]
    extern "C" {
        /// `cudaGetErrorString` of a code (a static string).
        pub fn gb_exl3_trellis_error_string(code: i32) -> *const core::ffi::c_char;
    }

    /// `cudaGetErrorString` of a code (a static string); a stub build has no CUDA runtime to ask.
    ///
    /// # Safety
    /// None beyond the GPU build's signature (it returns a pointer to a static string).
    #[cfg(gb_quant_no_cuda)]
    pub unsafe fn gb_exl3_trellis_error_string(code: i32) -> *const core::ffi::c_char {
        let _ = code;
        c"no CUDA in this build".as_ptr()
    }
}

// cudart's names (a stub build turns them into Rust functions of the same names)
#[allow(non_snake_case)]
mod dev {
    crate::gpu::foreign! {
        crate::gpu::CUDA_ERROR_NO_DEVICE;
        pub fn cudaGetDevice(dev: *mut i32) -> i32;
        pub fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, dev: i32) -> i32;
    }
}

/// Why an encode was refused or failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// A CUDA call failed.
    Cuda(CudaError),
    /// The arguments break the API contract (sizes, window).
    Invalid(String),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::Cuda(e) => write!(f, "trellis encoder: {e}"),
            EncodeError::Invalid(m) => write!(f, "trellis encoder: {m}"),
        }
    }
}

impl std::error::Error for EncodeError {}

impl From<CudaError> for EncodeError {
    fn from(e: CudaError) -> Self {
        EncodeError::Cuda(e)
    }
}

/// Precision of the search costs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Costs {
    /// fp32 costs, truncated by `D(p + 1)` mantissa bits (the default; lower error than fp16 costs).
    #[default]
    Fp32,
    /// Inputs and every new cost rounded to fp16 (ExLlamaV3-like precision, for ablations).
    Fp16,
}

/// The wrap estimate's free-start pass: `pre` positions before the wrap (any length up to [`MAX_PRE`], laps allowed)
/// through `post` positions after it (1..=256).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Window {
    pub pre: u32,
    pub post: u32,
}

impl Window {
    /// The rotate-by-half two-pass scheme (the default).
    pub const HALF: Window = Window { pre: 128, post: 128 };
    /// A full lap of warm-up on both sides: about 1.2x the time of [`Window::HALF`] and ~0.3% less error.
    pub const FULL: Window = Window { pre: 256, post: 256 };

    pub fn new(pre: u32, post: u32) -> Result<Window, EncodeError> {
        let w = Window { pre, post };
        w.check()?;
        Ok(w)
    }

    fn check(&self) -> Result<(), EncodeError> {
        if self.pre > MAX_PRE || self.post < 1 || self.post as usize > POSITIONS {
            return Err(EncodeError::Invalid(format!(
                "window ({}, {}): pre must be 0..={MAX_PRE} and post 1..={POSITIONS}",
                self.pre, self.post
            )));
        }
        Ok(())
    }
}

impl Default for Window {
    fn default() -> Self {
        Window::HALF
    }
}

/// Encoder options. The default is the quantizer's: fp32 costs, the rotate-by-half window, a 2 GiB scratch cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub costs: Costs,
    pub window: Window,
    /// Cap on the history scratch (`grid * hist_bytes`); a smaller cap runs fewer resident blocks, same results.
    pub max_scratch_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { costs: Costs::Fp32, window: Window::HALF, max_scratch_bytes: DEFAULT_MAX_SCRATCH_BYTES }
    }
}

/// `(ka, mask)` of a format width: integer K is (K, 0), K + 0.5 is (K, [`HALF_MASK`]).
pub fn rate(bits: Bits) -> (u32, u32) {
    let t = bits.twice() as u32;
    (t / 2, if t % 2 == 0 { 0 } else { HALF_MASK })
}

/// The kernel's rate check: `ka` in 1..=8, a 16-bit `mask` with an even popcount (whole uint32 words a tile), and no
/// extra bits at `ka` = 8.
pub fn check_rate(ka: u32, mask: u32) -> Result<(), EncodeError> {
    if !(1..=8).contains(&ka) || mask > 0xFFFF || mask.count_ones() % 2 == 1 || (ka == 8 && mask != 0) {
        return Err(EncodeError::Invalid(format!("rate ka={ka} mask={mask:#06x} is not one the encoder takes")));
    }
    Ok(())
}

/// `D(p)`: the new bits position `p` shifts in.
fn shift(ka: u32, mask: u32, p: usize) -> u32 {
    ka + ((mask >> (p & 15)) & 1)
}

fn slot_bits(a: u32) -> usize {
    match a {
        0..=1 => 1,
        2 => 2,
        3..=4 => 4,
        _ => 8,
    }
}

/// Bits of one history slot: wide enough for the arg-min's `a` top bits, and `2^b` slots fill whole bytes.
fn slot_ab(a: u32, b: u32) -> usize {
    slot_bits(a).max(8 >> b)
}

/// History scratch one job keeps (the kernel's `hist_bytes_for`; the GPU tests check the two agree): per position,
/// a slot for each of the `2^(16 - D(p + 1))` successor nodes.
pub fn hist_bytes(ka: u32, mask: u32) -> usize {
    (0..POSITIONS)
        .map(|p| {
            let (a, b) = (shift(ka, mask, p + 1), shift(ka, mask, p));
            ((1usize << (16 - a)) * slot_ab(a, b)) >> 3
        })
        .sum()
}

/// What [`raw::gb_exl3_trellis_query`] reports for one configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelInfo {
    pub threads: usize,
    pub smem_bytes: usize,
    pub blocks_per_sm: usize,
    pub hist_bytes: usize,
    /// The SASS architecture of the loaded kernel (121 on the GB10).
    pub binary_version: i32,
    pub num_regs: i32,
}

/// [`KernelInfo`] of the kernel for a raw rate.
pub fn kernel_info_raw(ka: u32, mask: u32, weighted: bool, costs: Costs) -> Result<KernelInfo, EncodeError> {
    check_rate(ka, mask)?;
    let (mut threads, mut smem, mut blocks, mut hist, mut binv, mut regs) = (0i32, 0i64, 0i32, 0i64, 0i32, 0i32);
    // SAFETY: plain out-parameters into locals
    CudaError::check(unsafe {
        raw::gb_exl3_trellis_query(
            ka as i32,
            mask,
            weighted as i32,
            (costs == Costs::Fp16) as i32,
            &mut threads,
            &mut smem,
            &mut blocks,
            &mut hist,
            &mut binv,
            &mut regs,
        )
    })?;
    Ok(KernelInfo {
        threads: threads as usize,
        smem_bytes: smem as usize,
        blocks_per_sm: blocks as usize,
        hist_bytes: hist as usize,
        binary_version: binv,
        num_regs: regs,
    })
}

/// [`KernelInfo`] of the unweighted kernel for a format width.
pub fn kernel_info(bits: Bits, costs: Costs) -> Result<KernelInfo, EncodeError> {
    let (ka, mask) = rate(bits);
    kernel_info_raw(ka, mask, false, costs)
}

/// Streaming multiprocessors of the current device.
pub fn sm_count() -> Result<usize, CudaError> {
    let (mut d, mut sms) = (0i32, 0i32);
    // SAFETY: plain attribute queries into locals
    unsafe {
        CudaError::check(dev::cudaGetDevice(&mut d))?;
        CudaError::check(dev::cudaDeviceGetAttribute(&mut sms, 16 /* cudaDevAttrMultiProcessorCount */, d))?;
    }
    Ok(sms.max(1) as usize)
}

/// Device outputs of an encode of `n_tiles` tiles.
#[derive(Debug)]
pub struct Encoded {
    pub n_tiles: usize,
    /// u16 [n_tiles, 256]: the 16-bit state of every position (pack with `gb_exl3::encode::pack_tile`).
    pub states: DevBuf,
    /// f32 [n_tiles, 256]: the fp16 values the states decode to, exactly the tile the decoder reconstructs.
    pub values: DevBuf,
    /// f32 [n_tiles]: the search's fp32 squared error of each tile (its DP cost).
    pub cost: DevBuf,
    /// i32 [n_tiles]: the wrap node (low `16 - D(0)` bits of the last state) each ring closes through.
    pub wrap: DevBuf,
}

impl Encoded {
    pub fn alloc(n_tiles: usize) -> Result<Encoded, CudaError> {
        let n = n_tiles * POSITIONS;
        Ok(Encoded {
            n_tiles,
            states: DevBuf::alloc(n * 2)?,
            values: DevBuf::alloc(n * 4)?,
            cost: DevBuf::alloc(n_tiles * 4)?,
            wrap: DevBuf::alloc(n_tiles * 4)?,
        })
    }

    /// Copy the outputs to the host (synchronous; the encode's stream must have reached them).
    pub fn to_host(&self) -> Result<HostEncoded, CudaError> {
        let n = self.n_tiles * POSITIONS;
        let mut out =
            HostEncoded { states: vec![0; n], values: vec![0.0; n], cost: vec![0.0; self.n_tiles], wrap: vec![0; self.n_tiles] };
        if self.n_tiles > 0 {
            self.states.download(&mut out.states)?;
            self.values.download(&mut out.values)?;
            self.cost.download(&mut out.cost)?;
            self.wrap.download(&mut out.wrap)?;
        }
        Ok(out)
    }
}

/// Host copy of [`Encoded`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HostEncoded {
    pub states: Vec<u16>,
    pub values: Vec<f32>,
    pub cost: Vec<f32>,
    pub wrap: Vec<i32>,
}

/// The encoder with its history scratch, kept between calls. Launches are asynchronous and reuse the scratch, so use
/// one `Encoder` per stream and keep it alive until the stream has passed its launches.
#[derive(Debug)]
pub struct Encoder {
    scratch: Option<DevBuf>,
    sms: usize,
}

impl Encoder {
    pub fn new() -> Result<Encoder, CudaError> {
        Ok(Encoder { scratch: None, sms: sm_count()? })
    }

    /// The grid a launch of `n_jobs` jobs runs: every resident block (occupancy x SMs), at most one a job, within the
    /// scratch cap.
    pub fn grid(&self, info: &KernelInfo, n_jobs: usize, max_scratch_bytes: usize) -> usize {
        let full = (info.blocks_per_sm * self.sms).max(1);
        let by_scratch = (max_scratch_bytes / info.hist_bytes.max(1)).max(1);
        full.min(by_scratch).min(n_jobs.max(1))
    }

    /// Encode `n_tiles` tiles of `x` (f32 [n_tiles, 256], trellis order, on the device) at `bits` into `out`
    /// (sized for at least `n_tiles`), asynchronously on `stream`.
    pub fn encode(
        &mut self,
        x: &DevBuf,
        n_tiles: usize,
        bits: Bits,
        opts: &Options,
        out: &mut Encoded,
        stream: &Stream,
    ) -> Result<(), EncodeError> {
        opts.window.check()?;
        if n_tiles == 0 {
            return Ok(());
        }
        if n_tiles > i32::MAX as usize {
            return Err(EncodeError::Invalid(format!("{n_tiles} tiles exceed one launch")));
        }
        let n = n_tiles * POSITIONS;
        if x.bytes() < n * 4 {
            return Err(EncodeError::Invalid(format!("x holds {} bytes, {n_tiles} tiles need {}", x.bytes(), n * 4)));
        }
        if out.n_tiles < n_tiles
            || out.states.bytes() < n * 2
            || out.values.bytes() < n * 4
            || out.cost.bytes() < n_tiles * 4
            || out.wrap.bytes() < n_tiles * 4
        {
            return Err(EncodeError::Invalid(format!("outputs sized for {} tiles, {n_tiles} requested", out.n_tiles)));
        }
        let (ka, mask) = rate(bits);
        let info = kernel_info_raw(ka, mask, false, opts.costs)?;
        let grid = self.grid(&info, n_tiles, opts.max_scratch_bytes);
        let need = grid * info.hist_bytes;
        if self.scratch.as_ref().is_none_or(|s| s.bytes() < need) {
            // earlier launches (on any stream) may still read the old scratch: let them finish before it is freed
            if self.scratch.is_some() {
                gb_cuda::rt::device_sync()?;
            }
            self.scratch = None;
            self.scratch = Some(DevBuf::alloc(need)?);
        }
        let hist = self.scratch.as_ref().expect("scratch allocated above");
        // SAFETY: every buffer was checked to hold what the kernel reads or writes for n_tiles jobs, the scratch holds
        // grid * hist_bytes, and the window and rate were validated
        CudaError::check(unsafe {
            raw::gb_exl3_trellis_launch(
                x.ptr() as *const f32,
                core::ptr::null(),
                0,
                core::ptr::null(),
                core::ptr::null(),
                out.states.ptr() as *mut i16,
                out.values.ptr() as *mut f32,
                out.cost.ptr() as *mut f32,
                out.wrap.ptr() as *mut i32,
                core::ptr::null_mut(),
                hist.ptr() as *mut u8,
                info.hist_bytes as i64,
                n_tiles as i32,
                raw::MODE_ROLLED,
                opts.window.pre as i32,
                opts.window.post as i32,
                ka as i32,
                mask,
                CODEBOOK_MUL1,
                (opts.costs == Costs::Fp16) as i32,
                grid as i32,
                stream.raw(),
            )
        })?;
        Ok(())
    }
}

/// Encode host tiles (f32, `[n, 256]` flattened, trellis order) at `bits`: one synchronous call on a stream of its own.
pub fn encode_host(x: &[f32], bits: Bits, opts: &Options) -> Result<HostEncoded, EncodeError> {
    if x.len() % POSITIONS != 0 {
        return Err(EncodeError::Invalid(format!("{} values are not whole tiles of {POSITIONS}", x.len())));
    }
    opts.window.check()?;
    let n_tiles = x.len() / POSITIONS;
    if n_tiles == 0 {
        return Ok(HostEncoded::default());
    }
    let stream = Stream::new()?;
    let dx = DevBuf::from_slice(x)?;
    let mut out = Encoded::alloc(n_tiles)?;
    let mut enc = Encoder::new()?;
    enc.encode(&dx, n_tiles, bits, opts, &mut out, &stream)?;
    stream.sync()?;
    Ok(out.to_host()?)
}

/// The 16-bit states of a tile encoded without tail-biting (a free start), from its 16 * bits int16 words and its
/// start state `start` = s_0. The stream is read as preceded by s_0's top 16 - D(0) bits instead of wrapping: a window
/// that starts before stream bit 0 (end E(p) < 16) takes its high bits from `start`, every other window is read as
/// [`gb_exl3::states`] reads it. Reference decoder for measuring a stored start state; the serving format has no such field.
pub fn states_seeded(tile: &[i16], bits: Bits, ends: &[u32; 256], start: u16) -> [u16; 256] {
    let mut out = gb_exl3::states(tile, bits, ends);
    let d0 = ends[0];
    let prefix = (start as u32) >> d0; // the 16 - D(0) bits before stream bit 0
    let head32 = (tile[0] as u16 as u32) | ((tile[1] as u16 as u32) << 16); // stream bits 0..31, MSB first
    for p in 0..256 {
        let e = ends[p];
        if e >= 16 {
            break;
        }
        let head = head32 >> (32 - e);
        out[p] = (((prefix & ((1u32 << (16 - e)) - 1)) << e) | head) as u16;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A free-start path: a random prefix and stream define every state as a 16-bit window of prefix ++ stream;
    /// `states_seeded` must give them back from the words and s_0, at every width. Its windows that start inside the
    /// stream are the tail-biting decoder's (`states`).
    #[test]
    fn seeded_states_read_the_start_state_before_the_stream() {
        let mut rng = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for twice in Bits::ALL_TWICE {
            let bits = Bits::from_twice(twice).unwrap();
            let ends = bits.stream_ends();
            let r = 16 * 16 * twice as usize / 2; // stream bits: 256 * bits
            for _case in 0..20 {
                let d0 = ends[0] as usize;
                let pre: Vec<u8> = (0..16 - d0).map(|_| (next() & 1) as u8).collect();
                let stream: Vec<u8> = (0..r).map(|_| (next() & 1) as u8).collect();
                let ext: Vec<u8> = pre.iter().chain(stream.iter()).copied().collect();
                let want: Vec<u16> = (0..256)
                    .map(|p| {
                        let end = ends[p] as usize + (16 - d0); // in ext
                        ext[end - 16..end].iter().fold(0u16, |a, &b| (a << 1) | b as u16)
                    })
                    .collect();
                // words: stream bit i is bit 31 - (i % 32) of u32 word i / 32; u32 word w = int16[2w] | int16[2w + 1] << 16
                let mut tile = vec![0i16; bits.tile_words()];
                for w in 0..r / 32 {
                    let v = (0..32).fold(0u32, |a, j| (a << 1) | stream[32 * w + j] as u32);
                    tile[2 * w] = (v & 0xFFFF) as u16 as i16;
                    tile[2 * w + 1] = (v >> 16) as u16 as i16;
                }
                let got = states_seeded(&tile, bits, &ends, want[0]);
                assert_eq!(got.to_vec(), want, "{} bits", bits.as_f64());
                // positions whose window lies inside the stream read as the tail-biting decoder reads them
                let ring = gb_exl3::states(&tile, bits, &ends);
                for p in 0..256 {
                    if ends[p] >= 16 {
                        assert_eq!(ring[p], got[p]);
                    }
                }
            }
        }
    }

    fn all_bits() -> impl Iterator<Item = Bits> {
        Bits::ALL_TWICE.iter().map(|&t| Bits::from_twice(t).unwrap())
    }

    #[test]
    fn every_format_width_maps_to_its_rate() {
        for bits in all_bits() {
            let (ka, mask) = rate(bits);
            check_rate(ka, mask).unwrap();
            assert_eq!(ka as f64 + mask.count_ones() as f64 / 16.0, bits.as_f64(), "{}", bits.as_f64());
            // the shifts sum to the tile's bitstream, and end where the decoder's windows end
            let ends = bits.stream_ends();
            let mut e = 0u32;
            for (p, &end) in ends.iter().enumerate() {
                e += shift(ka, mask, p);
                assert_eq!(e, end, "{} bits: E({p})", bits.as_f64());
            }
            assert_eq!(e as usize, 16 * bits.tile_words());
        }
    }

    #[test]
    fn rate_check_follows_the_kernel() {
        for (ka, mask, ok) in [
            (1, 0, true),
            (8, 0, true),
            (7, HALF_MASK, true),
            (2, 0x8888, true),
            (5, 0x0101, true),
            (0, 0, false),
            (9, 0, false),
            (8, HALF_MASK, false),
            (3, 0x0001, false),
            (3, 0x1_0000, false),
        ] {
            assert_eq!(check_rate(ka, mask).is_ok(), ok, "ka={ka} mask={mask:#x}");
        }
    }

    #[test]
    fn history_scratch_sizes() {
        // integer K: 256 rows of 2^(16 - K) slots; K = 1 keeps 4-bit slots (two bytes of 2^1 successors), K >= 5 bytes
        assert_eq!(hist_bytes(1, 0), 256 * (1 << 15) * 4 / 8);
        assert_eq!(hist_bytes(2, 0), 256 * (1 << 14) * 2 / 8);
        assert_eq!(hist_bytes(4, 0), 256 * (1 << 12) * 4 / 8);
        assert_eq!(hist_bytes(8, 0), 256 * (1 << 8));
        // 2.5 bits: rows alternate a = 3 (slot max(4, 8 >> 2) = 4) and a = 2 (slot max(2, 8 >> 3) = 2)
        assert_eq!(hist_bytes(2, HALF_MASK), 128 * ((1 << 13) * 4 / 8) + 128 * ((1 << 14) * 2 / 8));
    }

    #[test]
    fn window_bounds() {
        assert_eq!(Window::default(), Window::HALF);
        assert!(Window::new(0, 1).is_ok());
        assert!(Window::new(MAX_PRE, 256).is_ok());
        assert!(Window::new(MAX_PRE + 1, 128).is_err());
        assert!(Window::new(128, 0).is_err());
        assert!(Window::new(128, 257).is_err());
        let o = Options::default();
        assert_eq!((o.costs, o.window, o.max_scratch_bytes), (Costs::Fp32, Window::HALF, 2 << 30));
    }

    #[test]
    fn host_input_must_be_whole_tiles() {
        let bits = Bits::from_twice(4).unwrap();
        assert!(matches!(encode_host(&[0.0; 255], bits, &Options::default()), Err(EncodeError::Invalid(_))));
        let bad = Options { window: Window { pre: 0, post: 0 }, ..Options::default() };
        assert!(matches!(encode_host(&[0.0; 256], bits, &bad), Err(EncodeError::Invalid(_))));
        assert_eq!(encode_host(&[], bits, &Options::default()).unwrap(), HostEncoded::default());
    }

    #[cfg(gb_quant_no_cuda)]
    #[test]
    fn a_stub_build_refuses_to_encode() {
        let r = encode_host(&[0.0; 256], Bits::from_twice(4).unwrap(), &Options::default());
        assert!(matches!(r, Err(EncodeError::Cuda(_))), "{r:?}");
    }
}
