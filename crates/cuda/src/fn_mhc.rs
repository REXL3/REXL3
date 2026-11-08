//! Multi-HyperConnection (mHC) mixer for Qwen4_Exp / Flash-Next models: the reference kernels (`csrc/mhc.cu`).
//!
//! Shapes, for `t` rows: streams f32 `[t, hc, h]`; mixer weights f16 as stored: `norm [hc * h]`, `down [rank, hc * h]`,
//! `up [hc * h, rank]`, `inject [hc, hc * h]`; mixed rows bf16 `[t, h]`; inject weights f32 `[t, hc]`.
#![allow(clippy::too_many_arguments)]

use crate::rt::{CudaError, DevBuf, Stream};

/// Refuses a buffer smaller than what a launch touches, before anything is queued (`CudaError(1)`, as an invalid value).
fn need(b: &DevBuf, bytes: usize, what: &str) -> Result<(), CudaError> {
    if b.bytes() < bytes {
        eprintln!("{what}: buffer of {} bytes, the launch touches {bytes}", b.bytes());
        return Err(CudaError(1));
    }
    Ok(())
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_fn_hc_expand(
            hidden: *const c_void,
            streams: *mut c_void,
            hidden_size: u32,
            hc_mult: u32,
            num_tokens: u32,
            stream: *mut c_void,
        ) -> i32;

        pub fn gb_fn_hc_pre(
            streams: *const c_void,
            hc_norm_w: *const c_void,
            down_w: *const c_void,
            up_w: *const c_void,
            inject_w: *const c_void,
            normed_scratch: *mut c_void,
            low_scratch: *mut f32,
            y_out: *mut c_void,
            inj_out: *mut f32,
            hidden_size: u32,
            hc_mult: u32,
            rank: u32,
            norm_eps: f32,
            num_tokens: u32,
            stream: *mut c_void,
        ) -> i32;

        pub fn gb_fn_hc_head(
            streams: *const c_void,
            hc_norm_w: *const c_void,
            down_w: *const c_void,
            up_w: *const c_void,
            normed_scratch: *mut c_void,
            low_scratch: *mut f32,
            y_out: *mut c_void,
            hidden_size: u32,
            hc_mult: u32,
            rank: u32,
            norm_eps: f32,
            num_tokens: u32,
            stream: *mut c_void,
        ) -> i32;

        pub fn gb_fn_hc_post(
            block_out: *const c_void,
            residual: *const c_void,
            inj: *const f32,
            out: *mut c_void,
            hidden_size: u32,
            hc_mult: u32,
            num_tokens: u32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// The mixer weights of one hyper-connection (`inject` is absent for the head collapse).
#[derive(Clone, Copy)]
pub struct Mixer<'a> {
    pub norm: &'a DevBuf,
    pub down: &'a DevBuf,
    pub up: &'a DevBuf,
    pub inject: Option<&'a DevBuf>,
}

impl Mixer<'_> {
    /// Checks every weight against `h`, `hc`, `rank`.
    pub(crate) fn check(&self, h: usize, hc: usize, rank: usize, what: &str) -> Result<(), CudaError> {
        need(self.norm, hc * h * 2, &format!("{what} norm"))?;
        need(self.down, rank * hc * h * 2, &format!("{what} down"))?;
        need(self.up, hc * h * rank * 2, &format!("{what} up"))?;
        if let Some(inj) = self.inject {
            need(inj, hc * hc * h * 2, &format!("{what} inject"))?;
        }
        Ok(())
    }
}

/// `streams[t, s, :] = f32(hidden[t, :])` for every stream `s`.
pub fn hc_expand(
    hidden: &DevBuf,
    streams: &mut DevBuf,
    hidden_size: usize,
    hc_mult: usize,
    num_tokens: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    need(hidden, num_tokens * hidden_size * 2, "hc_expand hidden")?;
    need(streams, num_tokens * hc_mult * hidden_size * 4, "hc_expand streams")?;
    let sp = streams.ptr();
    {
        let rc = unsafe {
            ffi::gb_fn_hc_expand(hidden.ptr(), sp, hidden_size as u32, hc_mult as u32, num_tokens as u32, stream.raw())
        };
        CudaError::check(rc)
    }
}

/// Scratch of the reference pre-mix: `normed` f32 `[t, hc * h]`, `low` f32 `[t, rank]`.
pub struct PreScratch<'a> {
    pub normed: &'a mut DevBuf,
    pub low: &'a mut DevBuf,
}

/// Reference pre-mix: `y_out` bf16 `[t, h]` and, with `mix.inject`, `inj_out` f32 `[t, hc]` (both or neither).
pub fn hc_pre(
    streams: &DevBuf,
    mix: &Mixer<'_>,
    scratch: PreScratch<'_>,
    y_out: &mut DevBuf,
    inj_out: &mut DevBuf,
    hidden_size: usize,
    hc_mult: usize,
    rank: usize,
    norm_eps: f32,
    num_tokens: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let (h, hc, t) = (hidden_size, hc_mult, num_tokens);
    let Some(inject) = mix.inject else {
        eprintln!("rexl3 level=error launch contract: hc_pre needs the inject weights (use hc_head without them)");
        return Err(CudaError(1));
    };
    mix.check(h, hc, rank, "hc_pre")?;
    need(streams, t * hc * h * 4, "hc_pre streams")?;
    need(scratch.normed, t * hc * h * 4, "hc_pre normed")?;
    let np = scratch.normed.ptr();
    need(scratch.low, t * rank * 4, "hc_pre low")?;
    let lp = scratch.low.ptr() as *mut f32;
    need(y_out, t * h * 2, "hc_pre y")?;
    let yp = y_out.ptr();
    need(inj_out, t * hc * 4, "hc_pre inj")?;
    let ip = inj_out.ptr() as *mut f32;
    {
        let rc = unsafe {
            ffi::gb_fn_hc_pre(
                streams.ptr(),
                mix.norm.ptr(),
                mix.down.ptr(),
                mix.up.ptr(),
                inject.ptr(),
                np,
                lp,
                yp,
                ip,
                h as u32,
                hc as u32,
                rank as u32,
                norm_eps,
                t as u32,
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }
}

/// Reference head collapse (no inject): `y_out` bf16 `[t, h]` from the streams.
pub fn hc_head(
    streams: &DevBuf,
    mix: &Mixer<'_>,
    scratch: PreScratch<'_>,
    y_out: &mut DevBuf,
    hidden_size: usize,
    hc_mult: usize,
    rank: usize,
    norm_eps: f32,
    num_tokens: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let (h, hc, t) = (hidden_size, hc_mult, num_tokens);
    mix.check(h, hc, rank, "hc_head")?;
    need(streams, t * hc * h * 4, "hc_head streams")?;
    need(scratch.normed, t * hc * h * 4, "hc_head normed")?;
    let np = scratch.normed.ptr();
    need(scratch.low, t * rank * 4, "hc_head low")?;
    let lp = scratch.low.ptr() as *mut f32;
    need(y_out, t * h * 2, "hc_head y")?;
    let yp = y_out.ptr();
    {
        let rc = unsafe {
            ffi::gb_fn_hc_head(
                streams.ptr(),
                mix.norm.ptr(),
                mix.down.ptr(),
                mix.up.ptr(),
                np,
                lp,
                yp,
                h as u32,
                hc as u32,
                rank as u32,
                norm_eps,
                t as u32,
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }
}

/// `out[t, s, :] = residual[t, s, :] + inj[t, s] * block_out[t, :]` (f32 streams). `residual` may be None for an in-place update: each
/// element is read before it is written, by the same thread.
pub fn hc_post(
    block_out: &DevBuf,
    residual: Option<&DevBuf>,
    inj: &DevBuf,
    out: &mut DevBuf,
    hidden_size: usize,
    hc_mult: usize,
    num_tokens: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    let (h, hc, t) = (hidden_size, hc_mult, num_tokens);
    need(block_out, t * h * 2, "hc_post block_out")?;
    need(inj, t * hc * 4, "hc_post inj")?;

    need(out, t * hc * h * 4, "hc_post out")?;
    let op = out.ptr();
    let rp = if let Some(b) = residual {
        need(b, t * hc * h * 4, "hc_post residual")?;
        b.ptr()
    } else {
        op
    };
    {
        let rc = unsafe {
            ffi::gb_fn_hc_post(block_out.ptr(), rp, inj.ptr() as *const f32, op, h as u32, hc as u32, t as u32, stream.raw())
        };
        CudaError::check(rc)
    }
}
