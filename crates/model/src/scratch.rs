//! Activation buffers for one forward pass, allocated once and reused: every step (decode, verify window, prefill
//! chunk) runs in the same buffers, sized for the largest row count.

use crate::config::QwenConfig;
use crate::error::ModelError;
use gb_cuda::exl3_linear::{linear_workspace_bytes, prefill_scratch_bytes};
use gb_cuda::DevBuf;

/// Rows a decode or verify step may carry (all sequences together). The EXL3 decode kernel takes up to 128.
pub const MAX_DECODE_ROWS: usize = 128;

pub struct Scratch {
    pub rows: usize,
    /// bf16 [rows, hidden]: the residual stream, and a second one for the add+norm outputs.
    pub x: DevBuf,
    pub r: DevBuf,
    /// bf16 [rows, hidden]: normed input of the next sublayer.
    pub h: DevBuf,
    /// bf16 [rows, hidden]: sublayer output (mixer output, mlp output).
    pub a: DevBuf,
    /// bf16 [rows, intermediate]: gate, up, swiglu output.
    pub g: DevBuf,
    pub u: DevBuf,
    pub s: DevBuf,
    /// bf16 [rows, conv_channels]: raw in_proj_qkv output (also the q_proj output for full attention, which is smaller).
    pub qkv: DevBuf,
    /// bf16 [rows, 6144]: z / mixer output before out_proj.
    pub z: DevBuf,
    pub o: DevBuf,
    /// GDN record: q, k bf16 [rows,16,128]; v bf16 [rows,48,128]; decay, beta f32 [rows,48]; a, b bf16 [rows,48].
    pub gq: DevBuf,
    pub gk: DevBuf,
    pub gv: DevBuf,
    pub decay: DevBuf,
    pub beta: DevBuf,
    pub ga: DevBuf,
    pub gb: DevBuf,
    /// Full attention: k, v bf16 [rows, 4, 256]; roped q [rows,24,256]; gate [rows,24,256]; k after norm+rope.
    pub fk: DevBuf,
    pub fv: DevBuf,
    pub fq: DevBuf,
    pub fgate: DevBuf,
    pub fk2: DevBuf,
    /// Positions / ids / slot arrays as i32 on the device (uploaded each step).
    pub ids: DevBuf,
    pub positions: DevBuf,
    /// M-RoPE positions `[rows, 3]` (only prefill rows inside an image use them).
    pub rope3: DevBuf,
    pub row_slot: DevBuf,
    pub seq_start: DevBuf,
    pub seq_slot: DevBuf,
    pub keep: DevBuf,
    /// Per sequence of the window about to run: the first row of its previous window's record and the rows of it still owed to
    /// the recurrent state (i32 `[rows]` each; folded into the window kernel).
    pub owed_row0: DevBuf,
    pub owed_keep: DevBuf,
    /// A previous window's sequence layout and owed rows, for the standalone replay of rows no window kernel folds in.
    pub mat_seq_start: DevBuf,
    pub mat_slot: DevBuf,
    pub mat_keep: DevBuf,
    /// bf16 [rows, 5, hidden]: DFlash2 tap rows (outputs of layers 5, 19, 33, 47, 61).
    pub taps: DevBuf,
    /// EXL3 decode workspace (largest over every layer shape at MAX_DECODE_ROWS) and prefill scratch.
    pub ws: DevBuf,
    pub prefill_ws: DevBuf,
}

fn bf(n: usize) -> usize {
    n * 2
}

impl Scratch {
    pub fn new(c: &QwenConfig, rows: usize) -> Result<Self, ModelError> {
        let d = c.hidden;
        let conv = c.conv_channels();
        let z = c.lin_value_heads * c.lin_value_dim;
        let qdim = c.heads * c.head_dim;
        let kvdim = c.kv_heads * c.head_dim;
        let kd = c.lin_key_heads * c.lin_key_dim;
        let alloc = |bytes: usize| DevBuf::zeroed(bytes);
        // largest decode workspace over every EXL3 shape of the target at the widest decode step
        let shapes = [
            (d, c.intermediate),
            (c.intermediate, d),
            (d, conv),
            (d, z),
            (z, d),
            (d, 2 * qdim),
            (d, kvdim),
            (qdim, d),
            (d, c.vocab),
        ];
        let ws_bytes = shapes
            .iter()
            .map(|&(k, n)| {
                linear_workspace_bytes(MAX_DECODE_ROWS.min(rows), k, n, 4).max(linear_workspace_bytes(
                    MAX_DECODE_ROWS.min(rows),
                    k,
                    n,
                    6,
                ))
            })
            .max()
            .unwrap_or(0);
        // the head never takes the prefill path: logits are wanted for a few rows only, and its W_q alone would be 2.5 GB
        let pf_bytes =
            shapes.iter().filter(|&&(_, n)| n != c.vocab).map(|&(k, n)| prefill_scratch_bytes(rows, k, n)).max().unwrap_or(0);
        Ok(Self {
            rows,
            x: alloc(bf(rows * d))?,
            r: alloc(bf(rows * d))?,
            h: alloc(bf(rows * d))?,
            a: alloc(bf(rows * d))?,
            g: alloc(bf(rows * c.intermediate))?,
            u: alloc(bf(rows * c.intermediate))?,
            s: alloc(bf(rows * c.intermediate))?,
            qkv: alloc(bf(rows * conv.max(2 * qdim)))?,
            z: alloc(bf(rows * z.max(qdim)))?,
            o: alloc(bf(rows * z.max(qdim)))?,
            gq: alloc(bf(rows * kd))?,
            gk: alloc(bf(rows * kd))?,
            gv: alloc(bf(rows * z))?,
            decay: alloc(rows * c.lin_value_heads * 4)?,
            beta: alloc(rows * c.lin_value_heads * 4)?,
            ga: alloc(bf(rows * c.lin_value_heads))?,
            gb: alloc(bf(rows * c.lin_value_heads))?,
            fk: alloc(bf(rows * kvdim))?,
            fv: alloc(bf(rows * kvdim))?,
            fq: alloc(bf(rows * qdim))?,
            fgate: alloc(bf(rows * qdim))?,
            fk2: alloc(bf(rows * kvdim))?,
            ids: alloc(rows * 4)?,
            positions: alloc(rows * 4)?,
            rope3: alloc(rows * 12)?,
            row_slot: alloc(rows * 4)?,
            seq_start: alloc((rows + 1) * 4)?,
            seq_slot: alloc(rows * 4)?,
            keep: alloc(rows * 4)?,
            owed_row0: alloc(rows * 4)?,
            owed_keep: alloc(rows * 4)?,
            mat_seq_start: alloc((rows + 1) * 4)?,
            mat_slot: alloc(rows * 4)?,
            mat_keep: alloc(rows * 4)?,
            taps: alloc(bf(rows * 5 * d))?,
            ws: alloc(ws_bytes)?,
            prefill_ws: alloc(pf_bytes)?,
        })
    }

    /// Bytes of device memory held.
    pub fn bytes(&self) -> usize {
        [
            &self.x,
            &self.r,
            &self.h,
            &self.a,
            &self.g,
            &self.u,
            &self.s,
            &self.qkv,
            &self.z,
            &self.o,
            &self.gq,
            &self.gk,
            &self.gv,
            &self.decay,
            &self.beta,
            &self.ga,
            &self.gb,
            &self.fk,
            &self.fv,
            &self.fq,
            &self.fgate,
            &self.fk2,
            &self.ids,
            &self.positions,
            &self.row_slot,
            &self.seq_start,
            &self.seq_slot,
            &self.keep,
            &self.owed_row0,
            &self.owed_keep,
            &self.mat_seq_start,
            &self.mat_slot,
            &self.mat_keep,
            &self.taps,
            &self.ws,
            &self.prefill_ws,
        ]
        .iter()
        .map(|b| b.bytes())
        .sum()
    }
}
