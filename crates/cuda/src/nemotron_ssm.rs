//! Mamba2 (SSD) mixer kernels of the Nemotron 3.5 Super VL text backbone (`csrc/nemotron_ssm.cu`). Geometry: 128 heads of
//! head_dim 64, state 128 per B/C group, 8 groups (head h reads group h / 16), a depthwise causal conv of kernel 4 over the
//! 10240 channels x | B | C. Semantics and the bf16 rounding points are documented in the .cu file.
//!
//! The launch shape follows `crate::gdn`: sequences come from device arrays ([`Batch`], shared with the GDN kernels), and
//! nothing here synchronizes the host or allocates device memory per launch, so the launches can be graph-captured.
//!
//! Per-layer pools, allocated by the caller and indexed by slot:
//! * S pool, f32 `[slots][128 heads][128 state][64 head_dim]` (`S_SLOT_ELEMS` floats per slot); read by [`window`], written by
//!   [`commit_state`] only,
//! * conv pool, bf16 `[slots][3][10240]` (the last 3 pre-conv xBC rows, oldest first; `CONV_SLOT_ELEMS` per slot); read by
//!   [`pre`], written by [`commit_conv`] only.
//!
//! Verify and commit: [`pre`] and [`window`] are the verify pass. They read the pools, write the [`Record`] and the core
//! output, and never write S. [`commit_state`] and [`commit_conv`] replay the accepted prefix of the record into the pools.
//! A serial decode step is `pre` + `window` of one row, then the two commits with `keep = 1`.

use crate::gdn::Batch;
use crate::rt::{CudaError, DevBuf, Stream};

pub const HEADS: usize = 128;
/// State rows per head (N).
pub const STATE: usize = 128;
pub const HEAD_DIM: usize = 64;
pub const GROUPS: usize = 8;
/// The x and z width (`heads * head_dim`), also the gated norm's channel count.
pub const INNER: usize = HEADS * HEAD_DIM;
/// The width of B, and of C (`groups * state`).
pub const BC: usize = GROUPS * STATE;
/// The conv's channels: x | B | C.
pub const CONV_DIM: usize = INNER + 2 * BC;
pub const CONV_TAPS: usize = 4;
/// Pre-conv xBC rows kept per slot.
pub const CONV_STATE_ROWS: usize = 3;
/// The gated RMSNorm's epsilon.
pub const NORM_EPS: f32 = 1e-5;
/// f32 elements of one slot of the S pool.
pub const S_SLOT_ELEMS: usize = HEADS * STATE * HEAD_DIM;
/// bf16 elements of one slot of the conv pool.
pub const CONV_SLOT_ELEMS: usize = CONV_STATE_ROWS * CONV_DIM;

/// The per-layer parameters. `conv_w` is bf16 `[10240][4]` (the pack's `conv1d.weight[:, 0, :]`), `conv_b` bf16 `[10240]`, and
/// `a_log`, `dt_bias` and `d` f32 `[128]` (the loader converts the pack's bf16).
pub struct Weights<'a> {
    pub conv_w: &'a DevBuf,
    pub conv_b: &'a DevBuf,
    pub a_log: &'a DevBuf,
    pub dt_bias: &'a DevBuf,
    pub d: &'a DevBuf,
}

/// The post-conv rows of a window (what the commits and the readout need), one row per global row of the window.
pub struct Record {
    pub rows: usize,
    /// bf16 `[rows][10240]`: the conv output after silu, x | B | C.
    pub xbc: DevBuf,
    /// bf16 `[rows][128]`: dt, rounded to bf16. It is only the x * dt input scale.
    pub dt: DevBuf,
    /// f32 `[rows][128]`: g = -exp(A_log) * dt, from the unrounded fp32 dt.
    pub g: DevBuf,
    /// f32 `[rows][128]`: exp(g), the per-head decay of the state.
    pub decay: DevBuf,
}

impl Record {
    pub fn alloc(rows: usize) -> Result<Self, CudaError> {
        Ok(Self {
            rows,
            xbc: DevBuf::alloc(rows * CONV_DIM * 2)?,
            dt: DevBuf::alloc(rows * HEADS * 2)?,
            g: DevBuf::alloc(rows * HEADS * 4)?,
            decay: DevBuf::alloc(rows * HEADS * 4)?,
        })
    }
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_nssm_pre_range(xin: *const f32, dt_raw: *const f32, conv_w: *const c_void, conv_b: *const c_void, a_log: *const f32, dt_bias: *const f32, conv_pool: *const c_void, seq_start: *const i32, slot: *const i32, n_seq: i32, rows: i32, xbc: *mut c_void, dt: *mut c_void, g: *mut f32, decay: *mut f32, c0: i32, n0: i32, c1: i32, n1: i32, c2: i32, n2: i32, stream: *mut c_void) -> i32;
        pub fn gb_nssm_window_commit4_range(xbc: *const c_void, dt: *const c_void, decay: *const f32, s: *mut f32, d: *const f32, seq_start: *const i32, slot: *const i32, n_seq: i32, o: *mut c_void, head_lo: i32, n_heads: i32, stream: *mut c_void) -> i32;
        pub fn gb_nssm_gated_norm_range(y: *const c_void, z: *const f32, w: *const f32, eps: f32, rows: i32, out: *mut c_void, grp_lo: i32, n_grp: i32, stream: *mut c_void) -> i32;
        pub fn gb_nssm_commit_conv_range(xin: *const f32, conv_pool: *mut c_void, seq_start: *const i32, slot: *const i32, keep: *const i32, n_seq: i32, c0: i32, n0: i32, c1: i32, n1: i32, c2: i32, n2: i32, stream: *mut c_void) -> i32;
        pub fn gb_nssm_pre(
            xin: *const f32,
            dt_raw: *const f32,
            conv_w: *const c_void,
            conv_b: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_pool: *const c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            xbc: *mut c_void,
            dt: *mut c_void,
            g: *mut f32,
            decay: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_window(
            xbc: *const c_void,
            dt: *const c_void,
            decay: *const f32,
            s: *const f32,
            d: *const f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_window_commit4(
            xbc: *const c_void,
            dt: *const c_void,
            decay: *const f32,
            s: *mut f32,
            d: *const f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_window_commit(
            xbc: *const c_void,
            dt: *const c_void,
            decay: *const f32,
            s: *mut f32,
            d: *const f32,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_window_commit_sel(
            xbc: *const c_void,
            dt: *const c_void,
            decay: *const f32,
            s: *mut f32,
            d: *const f32,
            seq_start: *const i32,
            slot: *const i32,
            apply: *const i32,
            n_seq: i32,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_window_lag(
            xbc: *const c_void,
            dt: *const c_void,
            decay: *const f32,
            s: *mut f32,
            d: *const f32,
            seq_start: *const i32,
            slot: *const i32,
            apply: *const i32,
            lag: *const i32,
            ring_xbc: *mut c_void,
            ring_dt: *mut c_void,
            ring_decay: *mut f32,
            k: i32,
            n_seq: i32,
            o: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_fold_lag_all(
            layers: *const c_void,
            n_layers: i32,
            slots: *const i32,
            lags: *const i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_commit_state(
            xbc: *const c_void,
            dt: *const c_void,
            decay: *const f32,
            s: *mut f32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_commit_conv(
            xin: *const f32,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_commit_all(
            layers: *const c_void,
            n_layers: i32,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_pre_b(
            xin: *const c_void,
            dt_raw: *const f32,
            conv_w: *const c_void,
            conv_b: *const c_void,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_pool: *const c_void,
            seq_start: *const i32,
            slot: *const i32,
            n_seq: i32,
            rows: i32,
            xbc: *mut c_void,
            dt: *mut c_void,
            g: *mut f32,
            decay: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_commit_conv_b(
            xin: *const c_void,
            conv_pool: *mut c_void,
            seq_start: *const i32,
            slot: *const i32,
            keep: *const i32,
            n_seq: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_nssm_gated_norm_b(y: *const c_void, z: *const c_void, w: *const f32, eps: f32, rows: i32, out: *mut c_void, stream: *mut c_void) -> i32;
        pub fn gb_nssm_gated_norm(
            y: *const c_void,
            z: *const f32,
            w: *const f32,
            eps: f32,
            rows: i32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
    }
}

fn need(buf: &DevBuf, bytes: usize, what: &str) {
    assert!(buf.bytes() >= bytes, "nemotron_ssm: {what} holds {} bytes, needs {bytes}", buf.bytes());
}

fn check_weights(w: &Weights) {
    need(w.conv_w, CONV_DIM * CONV_TAPS * 2, "conv_w");
    need(w.conv_b, CONV_DIM * 2, "conv_b");
    need(w.a_log, HEADS * 4, "a_log");
    need(w.dt_bias, HEADS * 4, "dt_bias");
    need(w.d, HEADS * 4, "d");
}

/// Verify, part 1: the causal conv (bias, silu, one rounding to bf16) over every row of the batch, from the conv pool for the
/// rows before each sequence's first row, and dt, g and decay per row and head. Reads `conv_pool` and never writes it. `xbc_in` is
/// the in_proj output for x | B | C, f32 `[rows][10240]` (rounded to bf16 for the conv, as the fork does). `dt_raw` is f32
/// `[rows][128]`. Fills `rec` (`rec.rows >= batch.rows`).
pub fn pre(
    stream: &Stream,
    xbc_in: &DevBuf,
    dt_raw: &DevBuf,
    w: &Weights,
    conv_pool: &DevBuf,
    batch: &Batch<'_>,
    rec: &mut Record,
) -> Result<(), CudaError> {
    assert!(rec.rows >= batch.rows, "nemotron_ssm::pre: record has {} rows, batch {}", rec.rows, batch.rows);
    need(xbc_in, batch.rows * CONV_DIM * 4, "xbc_in");
    need(dt_raw, batch.rows * HEADS * 4, "dt_raw");
    check_weights(w);
    need(conv_pool, batch.n_slots * CONV_SLOT_ELEMS * 2, "conv pool");
    let rc = unsafe {
        ffi::gb_nssm_pre(
            xbc_in.ptr() as *const f32,
            dt_raw.ptr() as *const f32,
            w.conv_w.ptr(),
            w.conv_b.ptr(),
            w.a_log.ptr() as *const f32,
            w.dt_bias.ptr() as *const f32,
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            batch.rows as i32,
            rec.xbc.ptr(),
            rec.dt.ptr(),
            rec.g.ptr() as *mut f32,
            rec.decay.ptr() as *mut f32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Verify, part 2: run the recurrence over every sequence's rows from the committed state (`s_pool` is only read) and write the
/// core output `o` bf16 `[rows][8192]`, `y = C . S + D * x` before the gate. `d` is f32 `[128]`.
pub fn window(
    stream: &Stream,
    rec: &Record,
    s_pool: &DevBuf,
    d: &DevBuf,
    batch: &Batch<'_>,
    o: &mut DevBuf,
) -> Result<(), CudaError> {
    assert!(rec.rows >= batch.rows);
    need(o, batch.rows * INNER * 2, "o");
    need(s_pool, batch.n_slots * S_SLOT_ELEMS * 4, "S pool");
    need(d, HEADS * 4, "d");
    let rc = unsafe {
        ffi::gb_nssm_window(
            rec.xbc.ptr(),
            rec.dt.ptr(),
            rec.decay.ptr() as *const f32,
            s_pool.ptr() as *const f32,
            d.ptr() as *const f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            o.ptr(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`window`] followed by [`commit_state`] with `keep[i]` = the length of sequence i (every row kept), in one pass over `s_pool`: the
/// core output `o` and the pool are bit for bit what the two calls give. For a prefill slice, which commits all its rows at once; a
/// verify window uses [`window`], which never writes the pool. The conv history is still [`commit_conv`].
pub fn window_commit(
    stream: &Stream,
    rec: &Record,
    s_pool: &mut DevBuf,
    d: &DevBuf,
    batch: &Batch<'_>,
    o: &mut DevBuf,
) -> Result<(), CudaError> {
    assert!(rec.rows >= batch.rows);
    need(o, batch.rows * INNER * 2, "o");
    need(s_pool, batch.n_slots * S_SLOT_ELEMS * 4, "S pool");
    need(d, HEADS * 4, "d");
    // Lane S3: with the prefill opt-in (`nemotron_dense_mma::enabled`), the four-threads-a-column kernel. It is BIT-IDENTICAL to the
    // one-thread-a-column kernel (the same operations on every state row, the readout's four partial sums on four threads, combined in
    // the same order; gated by `tests/nemotron_ssm.rs`), so only the speed depends on the switch.
    let commit = if crate::nemotron_dense_mma::enabled() { ffi::gb_nssm_window_commit4 } else { ffi::gb_nssm_window_commit };
    let rc = unsafe {
        commit(
            rec.xbc.ptr(),
            rec.dt.ptr(),
            rec.decay.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            d.ptr() as *const f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            o.ptr(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`window`] that also stores the state of every sequence `i` with `apply[i] > 0` (i32 `[n_seq]` on the device), after its last row,
/// in the pass that reads it: for such a sequence the core output `o` and the pool are bit for bit what [`window`] and [`commit_state`]
/// with `keep[i]` = its length give, with S read once. A sequence with `apply[i] <= 0` is [`window`]'s: its state is untouched and a
/// later [`commit_state`] moves it. For a decode window of single-row streams (nothing to roll back: the one row is always kept). The
/// conv history of an applied sequence is still [`commit_conv`] (with `apply` as `keep`, which moves exactly the applied ones).
pub fn window_commit_sel(
    stream: &Stream,
    rec: &Record,
    s_pool: &mut DevBuf,
    d: &DevBuf,
    batch: &Batch<'_>,
    apply: &DevBuf,
    o: &mut DevBuf,
) -> Result<(), CudaError> {
    assert!(rec.rows >= batch.rows);
    need(o, batch.rows * INNER * 2, "o");
    need(s_pool, batch.n_slots * S_SLOT_ELEMS * 4, "S pool");
    need(d, HEADS * 4, "d");
    need(apply, batch.n_seq * 4, "apply");
    let rc = unsafe {
        ffi::gb_nssm_window_commit_sel(
            rec.xbc.ptr(),
            rec.dt.ptr(),
            rec.decay.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            d.ptr() as *const f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            apply.ptr() as *const i32,
            batch.n_seq as i32,
            o.ptr(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The ring of one Mamba layer of a LAGGED decode ([`window_lag`]): per slot the records of the up to `k - 1` rows decoded since the
/// slot's state was last folded into the S pool, in the form an `advance` reads them (the post-conv bf16 row, dt, decay).
pub struct LagRing {
    pub slots: usize,
    /// Fold period: a sequence's state is stored to S once every `k` steps.
    pub k: usize,
    /// bf16 `[slots][k - 1][10240]` (only the x and B parts of a row are written and read).
    pub xbc: DevBuf,
    /// bf16 `[slots][k - 1][128]`.
    pub dt: DevBuf,
    /// f32 `[slots][k - 1][128]`.
    pub decay: DevBuf,
}

impl LagRing {
    /// Device bytes of one layer's ring.
    pub fn bytes(slots: usize, k: usize) -> usize {
        let rows = slots * k.saturating_sub(1);
        rows * (CONV_DIM * 2 + HEADS * 2 + HEADS * 4)
    }

    pub fn alloc(slots: usize, k: usize) -> Result<Self, CudaError> {
        assert!(k >= 2, "a lag ring needs a fold period of at least 2");
        let rows = slots.max(1) * (k - 1);
        Ok(Self {
            slots,
            k,
            xbc: DevBuf::zeroed(rows * CONV_DIM * 2)?,
            dt: DevBuf::zeroed(rows * HEADS * 2)?,
            decay: DevBuf::zeroed(rows * HEADS * 4)?,
        })
    }
}

/// [`window_commit_sel`] with the store of the state deferred: for a sequence with `apply[i] > 0` (one row) whose slot has `lag[i]`
/// rows pending in `ring`, the S pool (the state as of the last fold) is advanced through the pending rows and then the window's row,
/// the core output is the serial decode's, and the state after the row is stored to S when `lag[i] + 1 == ring.k` (a FOLD: the ring is
/// empty again) or the row's record is pushed to ring row `lag[i]`. A sequence with `apply[i] <= 0` is [`window`]'s and must have no
/// pending rows. `apply` and `lag` are i32 `[n_seq]` on the device. The conv history of an applied sequence is [`commit_conv`] as ever.
#[allow(clippy::too_many_arguments)]
pub fn window_lag(
    stream: &Stream,
    rec: &Record,
    s_pool: &mut DevBuf,
    d: &DevBuf,
    batch: &Batch<'_>,
    apply: &DevBuf,
    lag: &DevBuf,
    ring: &mut LagRing,
    o: &mut DevBuf,
) -> Result<(), CudaError> {
    assert!(rec.rows >= batch.rows);
    need(o, batch.rows * INNER * 2, "o");
    need(s_pool, batch.n_slots * S_SLOT_ELEMS * 4, "S pool");
    need(d, HEADS * 4, "d");
    need(apply, batch.n_seq * 4, "apply");
    need(lag, batch.n_seq * 4, "lag");
    assert!(ring.slots >= batch.n_slots, "the lag ring holds {} slots, the batch's pool {}", ring.slots, batch.n_slots);
    let rc = unsafe {
        ffi::gb_nssm_window_lag(
            rec.xbc.ptr(),
            rec.dt.ptr(),
            rec.decay.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            d.ptr() as *const f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            apply.ptr() as *const i32,
            lag.ptr() as *const i32,
            ring.xbc.ptr(),
            ring.dt.ptr(),
            ring.decay.ptr() as *mut f32,
            ring.k as i32,
            batch.n_seq as i32,
            o.ptr(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The device table [`fold_lag_all`] reads: for every layer the addresses of its S pool and its lag ring (four pointers each). Valid
/// while the buffers are alive and unmoved.
pub struct LagTable {
    table: DevBuf,
    layers: usize,
    k: usize,
}

impl LagTable {
    pub fn new(layers: &[(&DevBuf, &LagRing)]) -> Result<Self, CudaError> {
        assert!(!layers.is_empty(), "nemotron_ssm::LagTable: no layers");
        let mut k = layers[0].1.k;
        let mut words: Vec<u64> = Vec::with_capacity(layers.len() * 4);
        for (s, ring) in layers {
            assert_eq!(ring.k, k, "every layer's ring has the same fold period");
            assert!(ring.slots <= s.bytes() / (S_SLOT_ELEMS * 4), "a lag ring for more slots than the S pool holds");
            k = ring.k;
            words.extend([s.ptr() as u64, ring.xbc.ptr() as u64, ring.dt.ptr() as u64, ring.decay.ptr() as u64]);
        }
        Ok(Self { table: DevBuf::from_slice(&words)?, layers: layers.len(), k })
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn k(&self) -> usize {
        self.k
    }
}

/// Folds the pending rows of the listed slots into the S pool of every layer, in one launch: `slots[i]` has `lags[i]` rows pending
/// (i32 `[n]` on the device; `lags[i] <= 0` leaves the slot alone). The state it leaves is the serial decode's after those rows.
pub fn fold_lag_all(stream: &Stream, table: &LagTable, slots: &DevBuf, lags: &DevBuf, n: usize) -> Result<(), CudaError> {
    need(slots, n * 4, "slots");
    need(lags, n * 4, "lags");
    let rc = unsafe {
        ffi::gb_nssm_fold_lag_all(
            table.table.ptr(),
            table.layers as i32,
            slots.ptr() as *const i32,
            lags.ptr() as *const i32,
            n as i32,
            table.k as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Commit, state half: replay the first `keep[i]` rows of sequence i (i32 `[n_seq]` on the device, `0 <= keep[i] <= len_i`, clamped
/// on the device) into `s_pool` in place. `keep = 0` leaves the sequence's state untouched. The conv history is [`commit_conv`].
pub fn commit_state(
    stream: &Stream,
    rec: &Record,
    s_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
) -> Result<(), CudaError> {
    assert!(rec.rows >= batch.rows);
    need(keep, batch.n_seq * 4, "keep");
    need(s_pool, batch.n_slots * S_SLOT_ELEMS * 4, "S pool");
    let rc = unsafe {
        ffi::gb_nssm_commit_state(
            rec.xbc.ptr(),
            rec.dt.ptr(),
            rec.decay.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            batch.n_seq as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// Commit, conv half: the new history of each sequence is the last 3 rows of concat(old 3 rows, its first `keep[i]` rows of
/// `xbc_in`, rounded to bf16). `keep = 0` leaves the history untouched.
pub fn commit_conv(
    stream: &Stream,
    xbc_in: &DevBuf,
    conv_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
) -> Result<(), CudaError> {
    need(xbc_in, batch.rows * CONV_DIM * 4, "xbc_in");
    need(keep, batch.n_seq * 4, "keep");
    need(conv_pool, batch.n_slots * CONV_SLOT_ELEMS * 2, "conv pool");
    let rc = unsafe {
        ffi::gb_nssm_commit_conv(
            xbc_in.ptr() as *const f32,
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            batch.n_seq as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// One Mamba layer as [`commit_all`] reads and writes it.
pub struct CommitLayer<'a> {
    /// The layer's window record (what [`commit_state`] replays).
    pub rec: &'a Record,
    /// The layer's retained pre-conv xBC rows, f32 `[rows][10240]` (what [`commit_conv`] takes as `xbc_in`).
    pub xin: &'a DevBuf,
    /// The layer's S pool.
    pub s: &'a DevBuf,
    /// The layer's conv pool.
    pub conv: &'a DevBuf,
}

/// The device table [`commit_all`] reads: for every layer the addresses of its record, retained rows and pools (six pointers each).
/// The table holds addresses, not the buffers: it is valid while the buffers it was built from are alive and unmoved, and the caller
/// keeps them (a model's pools and records live as long as the model).
pub struct CommitTable {
    table: DevBuf,
    layers: usize,
    /// Rows every layer's record and retained rows hold.
    rows: usize,
    /// Slots every layer's pools hold.
    slots: usize,
}

impl CommitTable {
    pub fn new(layers: &[CommitLayer<'_>]) -> Result<Self, CudaError> {
        assert!(!layers.is_empty(), "nemotron_ssm::CommitTable: no layers");
        let mut words: Vec<u64> = Vec::with_capacity(layers.len() * 6);
        let (mut rows, mut slots) = (usize::MAX, usize::MAX);
        for l in layers {
            rows = rows.min(l.rec.rows).min(l.xin.bytes() / (CONV_DIM * 4));
            slots = slots.min(l.s.bytes() / (S_SLOT_ELEMS * 4)).min(l.conv.bytes() / (CONV_SLOT_ELEMS * 2));
            // the order of `CommitLayer` in the .cu file
            words.extend([
                l.rec.xbc.ptr() as u64,
                l.rec.dt.ptr() as u64,
                l.rec.decay.ptr() as u64,
                l.s.ptr() as u64,
                l.xin.ptr() as u64,
                l.conv.ptr() as u64,
            ]);
        }
        Ok(Self { table: DevBuf::from_slice(&words)?, layers: layers.len(), rows, slots })
    }

    /// Layers in the table.
    pub fn layers(&self) -> usize {
        self.layers
    }
}

/// [`commit_state`] and [`commit_conv`] of every layer of `table`, with the same `keep`, in two launches. Each layer's pools end with
/// the bits the single-layer calls give; the layers are independent. The batch's rows must be inside the rows every layer holds and
/// its slots inside the slots of every pool.
pub fn commit_all(stream: &Stream, table: &CommitTable, batch: &Batch<'_>, keep: &DevBuf) -> Result<(), CudaError> {
    assert!(batch.rows <= table.rows, "nemotron_ssm::commit_all: batch of {} rows, layers hold {}", batch.rows, table.rows);
    assert!(batch.n_slots <= table.slots, "nemotron_ssm::commit_all: {} slots, pools hold {}", batch.n_slots, table.slots);
    need(keep, batch.n_seq * 4, "keep");
    let rc = unsafe {
        ffi::gb_nssm_commit_all(
            table.table.ptr(),
            table.layers as i32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            batch.n_seq as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// The gate and the grouped RMSNorm: `out = (v * w) * rsqrt(mean_group(v^2) + NORM_EPS)` with `v = y * silu(z)`, `z` rounded to
/// bf16 first. `y` is the bf16 core output `[rows][8192]`, `z` f32 `[rows][8192]`, `w` f32 `[8192]`, and `out` bf16 `[rows][8192]`.
pub fn gated_norm(stream: &Stream, y: &DevBuf, z: &DevBuf, w: &DevBuf, rows: usize, out: &mut DevBuf) -> Result<(), CudaError> {
    need(y, rows * INNER * 2, "y");
    need(z, rows * INNER * 4, "z");
    need(w, INNER * 4, "norm weight");
    need(out, rows * INNER * 2, "out");
    let rc = unsafe {
        ffi::gb_nssm_gated_norm(
            y.ptr(),
            z.ptr() as *const f32,
            w.ptr() as *const f32,
            NORM_EPS,
            rows as i32,
            out.ptr(),
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`pre`], [`commit_conv`] and [`gated_norm`] with the pre-conv xBC rows (`xbc_in`, `[rows][10240]`) and `z` (`[rows][8192]`) stored
/// as bf16 (lane S5: P2's fused in_proj split writes them on the bf16 grid every consumer reads them on): the same results bit for bit
/// (`rbf` of a bf16 value is that value; the conv history takes the same bf16).
pub mod bf16_in {
    use super::*;
    use core::ffi::c_void;

    pub fn pre(
        stream: &Stream,
        xbc_in: &DevBuf,
        dt_raw: &DevBuf,
        w: &Weights,
        conv_pool: &DevBuf,
        batch: &Batch<'_>,
        rec: &mut Record,
    ) -> Result<(), CudaError> {
        assert!(rec.rows >= batch.rows, "nemotron_ssm::pre: record has {} rows, batch {}", rec.rows, batch.rows);
        need(xbc_in, batch.rows * CONV_DIM * 2, "xbc_in");
        need(dt_raw, batch.rows * HEADS * 4, "dt_raw");
        check_weights(w);
        need(conv_pool, batch.n_slots * CONV_SLOT_ELEMS * 2, "conv pool");
        let rc = unsafe {
            ffi::gb_nssm_pre_b(
                xbc_in.ptr() as *const c_void,
                dt_raw.ptr() as *const f32,
                w.conv_w.ptr(),
                w.conv_b.ptr(),
                w.a_log.ptr() as *const f32,
                w.dt_bias.ptr() as *const f32,
                conv_pool.ptr(),
                batch.seq_start.ptr() as *const i32,
                batch.slot.ptr() as *const i32,
                batch.n_seq as i32,
                batch.rows as i32,
                rec.xbc.ptr(),
                rec.dt.ptr(),
                rec.g.ptr() as *mut f32,
                rec.decay.ptr() as *mut f32,
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }

    pub fn commit_conv(
        stream: &Stream,
        xbc_in: &DevBuf,
        conv_pool: &mut DevBuf,
        batch: &Batch<'_>,
        keep: &DevBuf,
    ) -> Result<(), CudaError> {
        need(xbc_in, batch.rows * CONV_DIM * 2, "xbc_in");
        need(keep, batch.n_seq * 4, "keep");
        need(conv_pool, batch.n_slots * CONV_SLOT_ELEMS * 2, "conv pool");
        let rc = unsafe {
            ffi::gb_nssm_commit_conv_b(
                xbc_in.ptr() as *const c_void,
                conv_pool.ptr(),
                batch.seq_start.ptr() as *const i32,
                batch.slot.ptr() as *const i32,
                keep.ptr() as *const i32,
                batch.n_seq as i32,
                stream.raw(),
            )
        };
        CudaError::check(rc)
    }

    pub fn gated_norm(
        stream: &Stream,
        y: &DevBuf,
        z: &DevBuf,
        w: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
    ) -> Result<(), CudaError> {
        need(y, rows * INNER * 2, "y");
        need(z, rows * INNER * 2, "z");
        need(w, INNER * 4, "norm weight");
        need(out, rows * INNER * 2, "out");
        let rc = unsafe {
            ffi::gb_nssm_gated_norm_b(y.ptr(), z.ptr(), w.ptr() as *const f32, NORM_EPS, rows as i32, out.ptr(), stream.raw())
        };
        CudaError::check(rc)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Lane S, contract P2 at two TP ranks: one rank's head groups (`HeadRange`) through the same kernels on a sub-range. Every element a
// rank computes is computed by the same code from the same inputs as by the whole-range call; the buffers keep the full geometry
// (the other rank's columns are neither read nor written).

/// A rank's share of the Mamba heads: whole B/C groups (`grp_lo .. grp_lo + n_grp`, 16 heads each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadRange {
    pub grp_lo: usize,
    pub n_grp: usize,
}

impl HeadRange {
    /// Rank `r` of two: groups [4 r, 4 r + 4).
    pub fn of_rank(rank: usize) -> Self {
        Self { grp_lo: GROUPS / 2 * rank, n_grp: GROUPS / 2 }
    }
    pub fn head_lo(&self) -> usize {
        self.grp_lo * (HEADS / GROUPS)
    }
    pub fn n_heads(&self) -> usize {
        self.n_grp * (HEADS / GROUPS)
    }
    /// The xBC channel ranges (base, count) of the range: its x channels, its B groups, its C groups.
    pub fn conv_ranges(&self) -> [(usize, usize); 3] {
        let x = (self.head_lo() * HEAD_DIM, self.n_heads() * HEAD_DIM);
        let b = (INNER + self.grp_lo * STATE, self.n_grp * STATE);
        let c = (INNER + GROUPS * STATE + self.grp_lo * STATE, self.n_grp * STATE);
        [x, b, c]
    }
    /// The z / gated channels of the range: [head_lo * 64, + n_heads * 64).
    pub fn inner_range(&self) -> (usize, usize) {
        (self.head_lo() * HEAD_DIM, self.n_heads() * HEAD_DIM)
    }
}

/// [`pre`] for the channels of `hr` (and dt for every head).
#[allow(clippy::too_many_arguments)]
pub fn pre_range(
    stream: &Stream,
    xbc_in: &DevBuf,
    dt_raw: &DevBuf,
    w: &Weights,
    conv_pool: &DevBuf,
    batch: &Batch<'_>,
    rec: &mut Record,
    hr: HeadRange,
) -> Result<(), CudaError> {
    assert!(rec.rows >= batch.rows);
    need(xbc_in, batch.rows * CONV_DIM * 4, "xbc_in");
    need(dt_raw, batch.rows * HEADS * 4, "dt_raw");
    check_weights(w);
    need(conv_pool, batch.n_slots * CONV_SLOT_ELEMS * 2, "conv pool");
    let [(c0, n0), (c1, n1), (c2, n2)] = hr.conv_ranges();
    let rc = unsafe {
        ffi::gb_nssm_pre_range(
            xbc_in.ptr() as *const f32,
            dt_raw.ptr() as *const f32,
            w.conv_w.ptr(),
            w.conv_b.ptr(),
            w.a_log.ptr() as *const f32,
            w.dt_bias.ptr() as *const f32,
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            batch.rows as i32,
            rec.xbc.ptr(),
            rec.dt.ptr(),
            rec.g.ptr() as *mut f32,
            rec.decay.ptr() as *mut f32,
            c0 as i32,
            n0 as i32,
            c1 as i32,
            n1 as i32,
            c2 as i32,
            n2 as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`window_commit`] (the four-threads-a-column kernel) for the heads of `hr`.
pub fn window_commit_range(
    stream: &Stream,
    rec: &Record,
    s_pool: &mut DevBuf,
    d: &DevBuf,
    batch: &Batch<'_>,
    o: &mut DevBuf,
    hr: HeadRange,
) -> Result<(), CudaError> {
    assert!(rec.rows >= batch.rows);
    need(o, batch.rows * INNER * 2, "o");
    need(s_pool, batch.n_slots * S_SLOT_ELEMS * 4, "S pool");
    need(d, HEADS * 4, "d");
    let rc = unsafe {
        ffi::gb_nssm_window_commit4_range(
            rec.xbc.ptr(),
            rec.dt.ptr(),
            rec.decay.ptr() as *const f32,
            s_pool.ptr() as *mut f32,
            d.ptr() as *const f32,
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            batch.n_seq as i32,
            o.ptr(),
            hr.head_lo() as i32,
            hr.n_heads() as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`gated_norm`] for the groups of `hr`.
pub fn gated_norm_range(
    stream: &Stream,
    y: &DevBuf,
    z: &DevBuf,
    w: &DevBuf,
    rows: usize,
    out: &mut DevBuf,
    hr: HeadRange,
) -> Result<(), CudaError> {
    need(y, rows * INNER * 2, "y");
    need(z, rows * INNER * 4, "z");
    need(w, INNER * 4, "norm weight");
    need(out, rows * INNER * 2, "out");
    let rc = unsafe {
        ffi::gb_nssm_gated_norm_range(
            y.ptr(),
            z.ptr() as *const f32,
            w.ptr() as *const f32,
            NORM_EPS,
            rows as i32,
            out.ptr(),
            hr.grp_lo as i32,
            hr.n_grp as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`commit_conv`] for the channels of `hr`.
pub fn commit_conv_range(
    stream: &Stream,
    xbc_in: &DevBuf,
    conv_pool: &mut DevBuf,
    batch: &Batch<'_>,
    keep: &DevBuf,
    hr: HeadRange,
) -> Result<(), CudaError> {
    need(xbc_in, batch.rows * CONV_DIM * 4, "xbc_in");
    need(keep, batch.n_seq * 4, "keep");
    need(conv_pool, batch.n_slots * CONV_SLOT_ELEMS * 2, "conv pool");
    let [(c0, n0), (c1, n1), (c2, n2)] = hr.conv_ranges();
    let rc = unsafe {
        ffi::gb_nssm_commit_conv_range(
            xbc_in.ptr() as *const f32,
            conv_pool.ptr(),
            batch.seq_start.ptr() as *const i32,
            batch.slot.ptr() as *const i32,
            keep.ptr() as *const i32,
            batch.n_seq as i32,
            c0 as i32,
            n0 as i32,
            c1 as i32,
            n1 as i32,
            c2 as i32,
            n2 as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}
