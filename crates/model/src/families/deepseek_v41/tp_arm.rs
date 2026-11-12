//! The TP-exact combine arms of the DeepSeek-V4.1-Flash model step (`families::deepseek_v41::tp`'s seam, filled): the two
//! places a rank of a tensor-parallel group combines its half of a block's output with its peer's, mirroring the VERIFIED
//! nemotron arms (`families::nemotron_h_omni::tp_arm`, `docs/NEMOTRON_TP.md` section 3).
//!
//! **The one rule, unchanged.** Every output element is computed by exactly ONE rank with the same instruction sequence the
//! single-node engine runs — this rank's experts' `y` segments, this rank's heads' attention-output columns — and the peer's
//! elements arrive whole, never re-computed and never re-rounded, so the combined buffer is the single-node one BYTE for
//! byte and every kernel downstream of a combine (`gb_dsv4_moe_scatter`, `MlaDsv4::o_proj`, the head, the mHC sites) runs
//! replicated and identically on every rank. The design is the nemotron `moe_pair_gather`'s (its lane-S MoE-y form), not its
//! `-0.0`-fill-and-sum form, for a reason the buffers force: both of this family's combine buffers are **f16** (`y` is the
//! EXL3 gemv's operand type, the attention output `o` is the sink kernel's), and the group's collective is f32
//! (`TpComm::allgather_sum_f32`). A sum over f16 halves would need a widen-and-narrow round trip no kernel of this tree
//! makes; a byte exchange of the two disjoint halves needs no arithmetic at all, which is the stronger form of the same
//! invariant — there is no `+0.0` anywhere to flip a `-0.0`. The `-0.0` fill discipline is the f32 arms' (`tp`'s loopback
//! test pins it); these arms pin the byte form.
//!
//! * **The MoE y combine** ([`TpArms::y_gather`]): the router is replicated (identical picks, identical grouping), so the
//!   packed slot axis is the single-node one and a rank's owned experts own a CONTIGUOUS range of it. One `swap_bytes` of
//!   the rank's slot range against the peer's mirror range, in place: `y` is the whole buffer, and the unchanged scatter
//!   adds the shared expert (replicated, the nemotron rule: computed on every rank, added once by the scatter each rank
//!   runs) exactly as on one node.
//! * **The MLA o combine** ([`TpArms::o_gather`]): each rank's `forward_slot_to_o` leaves its `q_heads` heads' attention
//!   output compact at the start of the scratch `o`; the arm swaps the two compacts and places both halves at their global
//!   column ranges of a full-width staging buffer (one row copy a row, `plane_write_dev`), and `MlaDsv4::o_proj` — the
//!   unchanged `wo_a`+`wo_b` over every head's columns, replicated weights — runs on the result. The latent KV plane is
//!   written identically by both ranks before any of this and carries no collective.
//!
//! Both arms are TWO-RANK for now (`TpComm::swap_bytes` is a pairwise exchange): a wider group is refused by name at
//! [`Dsv41Model::enable_tp`], the seam a later lane widens (a strided place kernel makes the o combine rank-count-free;
//! the y combine chains one swap a neighbour).
//!
//! Nothing here runs on a single rank: the model holds `tp: None` and every op is the single-node engine's, unchanged.

use super::tp::{ShardTable, TpComm};
use crate::error::ModelError;
use gb_cuda::{DevBuf, Stream};
use std::ops::Range;
use std::sync::Arc;

/// The combine arms of one rank of a group, and their staging: the full-width attention-output buffer
/// `[rows_cap, heads * head_dim]` f16 and the peer's compact half `[rows_cap, o_row_range().len()]` f16 the swap receives
/// into. Allocated once at [`super::model::Dsv41Model::enable_tp`] for the model's widest step, so no forward allocates.
pub struct TpArms {
    comm: Arc<dyn TpComm>,
    table: ShardTable,
    /// `[rows_cap][heads * head_dim]` f16: the combined attention output every rank's `o_proj` reads.
    o_full: DevBuf,
    /// `[rows_cap][local heads * head_dim]` f16: the peer's compact half, the swap's receive buffer.
    o_peer: DevBuf,
    /// `[rows_cap * topk][hidden]` f16: the peer's packed routed-expert output buffer for `moe_experts_resident_tp`.
    y_peer: DevBuf,
    /// Bytes this rank put through the collective, and how many exchanges it issued (the boot log's `event=dsv41_tp`).
    pub exchanged_bytes: u64,
    pub exchanges: u64,
}

impl TpArms {
    /// The arms of `comm`'s rank over `table`, with staging for steps of up to `rows_cap` rows. Fails closed when the
    /// collective cannot exchange bytes (the arms' whole contract) — never at the first forward.
    pub fn new(comm: Arc<dyn TpComm>, table: ShardTable, rows_cap: usize, o_rows_total: usize) -> Result<Self, ModelError> {
        if !comm.supports_swap() {
            return Err(ModelError::Config(
                "deepseek_v41: the group's collective cannot exchange bytes, which the tensor-parallel combine arms need \
                 (an NCCL group or the in-process loopback; REXL3_TP_COMM=none boots no shards)"
                    .into(),
            ));
        }
        let local = table.o_row_range().len();
        // `zeroed_view`, not `zeroed`: the full-width staging buffer is cut into per-step subviews at every `o_gather`
        // (`subview` works only on a view into a block — the device-gated arm test caught this: the first gather panicked
        // with "subview of a buffer that is not a view into a block"). The same one `cudaMalloc` and the same memset.
        let mk = |bytes: usize| DevBuf::zeroed_view(bytes.max(1)).map_err(|e| ModelError::Config(format!("the TP staging: {e}")));
        Ok(Self {
            comm,
            table,
            o_full: mk(rows_cap * o_rows_total * 2)?,
            o_peer: mk(rows_cap * local * 2)?,
            y_peer: mk(rows_cap * 12 * 5120 * 2)?,
            exchanged_bytes: 0,
            exchanges: 0,
        })
    }

    /// This rank's place in the group and its split table.
    pub fn table(&self) -> &ShardTable {
        &self.table
    }

    /// The rank's compact o half of `rows` rows at `o_compact` (the scratch `o` after [`gb_cuda::dsv4_attn::MlaDsv4::forward_slot_to_o`]),
    /// placed with the peer's into rows `[row0, row0 + rows)` of the full-width staging buffer this returns a view of: the
    /// single-rank attention output, byte for byte, at the o-projection boundary. `row0` is the rows' offset in the step's
    /// buffer (0 for a single-sequence step; a multi-sequence window's rank gathers each sequence's rows as its own call,
    /// the same sizes on every rank because the batch is).
    pub fn o_gather(&mut self, o_compact: &DevBuf, row0: usize, rows: usize, st: &Stream) -> Result<DevBuf, ModelError> {
        let total = self.table.q_heads * self.table.head_dim;
        let local = self.table.o_row_range().len();
        let mine = self.table.o_row_range();
        let peers = self.peer_o_range();
        let row_bytes = local * 2;
        if o_compact.bytes() < rows * row_bytes || self.o_full.bytes() < (row0 + rows) * total * 2 {
            return Err(ModelError::Request(format!(
                "tensor parallelism: the attention output of {rows} rows needs {} B of compact and the staging holds {} B at row {row0}",
                rows * row_bytes,
                self.o_full.bytes()
            )));
        }
        // the peer's compact half, one byte exchange (no arithmetic: the bits are the peer's, exactly)
        self.comm
            .swap_bytes(o_compact, 0, rows * row_bytes, &mut self.o_peer, 0, rows * row_bytes, st)
            .map_err(|e| ModelError::Config(format!("the attention-output exchange: {e}")))?;
        let mut dst = self.o_full.subview(row0 * total * 2, rows * total * 2);
        if (local & 7) == 0 && mine.start == 0 && peers.start == local {
            gb_cuda::dsv4_moe::interleave_o_halves(o_compact, &self.o_peer, &mut dst, rows, local, st)?;
        } else if (local & 7) == 0 && peers.start == 0 && mine.start == local {
            gb_cuda::dsv4_moe::interleave_o_halves(&self.o_peer, o_compact, &mut dst, rows, local, st)?;
        } else {
            for r in 0..rows {
                let dst_row = row0 + r;
                gb_cuda::dsv4_attn::plane_write_dev(o_compact, r, &self.o_full, 1, local, dst_row * total + mine.start, st)?;
                gb_cuda::dsv4_attn::plane_write_dev(&self.o_peer, r, &self.o_full, 1, local, dst_row * total + peers.start, st)?;
            }
        }
        self.exchanged_bytes += (rows * row_bytes) as u64;
        self.exchanges += 1;
        Ok(dst)
    }

    /// Gather the rank's compact `olora` half (`[rows, groups_local * o_lora]` f16, produced by
    /// [`gb_cuda::dsv4_attn::MlaDsv4::woa_local`]) with the peer's into rows `[row0, row0 + rows)` of the full-width
    /// staging buffer (`[rows, o_groups * o_lora]` f16). Because `wo_a` reduces each 4,096-dim head group to `o_lora`
    /// (256) before the exchange, this transfers 16x fewer bytes than [`Self::o_gather`] and keeps even 2,048-row
    /// prefill chunks under `RoceTp`'s 8 MiB RDMA threshold while remaining bit-for-bit identical.
    pub fn olora_gather(
        &mut self,
        olora_compact: &DevBuf,
        row0: usize,
        rows: usize,
        o_groups: usize,
        o_lora: usize,
        st: &Stream,
    ) -> Result<DevBuf, ModelError> {
        let total = o_groups * o_lora;
        let (g0, g_local) = self.table.o_group_range(o_groups);
        let local = g_local * o_lora;
        let mine_start = g0 * o_lora;
        let peer_start = if mine_start == 0 { local } else { 0 };
        let row_bytes = local * 2;
        if olora_compact.bytes() < rows * row_bytes || self.o_full.bytes() < (row0 + rows) * total * 2 {
            return Err(ModelError::Request(format!(
                "tensor parallelism: the olora output of {rows} rows needs {} B of compact and the staging holds {} B at row {row0}",
                rows * row_bytes,
                self.o_full.bytes()
            )));
        }
        self.comm
            .swap_bytes(olora_compact, 0, rows * row_bytes, &mut self.o_peer, 0, rows * row_bytes, st)
            .map_err(|e| ModelError::Config(format!("the olora attention exchange: {e}")))?;
        let mut dst = self.o_full.subview(row0 * total * 2, rows * total * 2);
        if (local & 7) == 0 && mine_start == 0 && peer_start == local {
            gb_cuda::dsv4_moe::interleave_o_halves(olora_compact, &self.o_peer, &mut dst, rows, local, st)?;
        } else if (local & 7) == 0 && peer_start == 0 && mine_start == local {
            gb_cuda::dsv4_moe::interleave_o_halves(&self.o_peer, olora_compact, &mut dst, rows, local, st)?;
        } else {
            for r in 0..rows {
                let dst_row = row0 + r;
                gb_cuda::dsv4_attn::plane_write_dev(olora_compact, r, &self.o_full, 1, local, dst_row * total + mine_start, st)?;
                gb_cuda::dsv4_attn::plane_write_dev(&self.o_peer, r, &self.o_full, 1, local, dst_row * total + peer_start, st)?;
            }
        }
        self.exchanged_bytes += (rows * row_bytes) as u64;
        self.exchanges += 1;
        Ok(dst)
    }

    /// Fast grouped MoE for a layer whose owned experts (`table.trunk_expert_range()`) are all resident in `descs`:
    /// zero host stream syncs, 5 compute launches + 1 fixed-size `slots * h * 2` byte swap into `y_peer` + 1 scatter launch.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn moe_fast(
        &mut self,
        s: &mut gb_cuda::dsv4_moe::MoeScratch,
        descs: &DevBuf,
        join_shared: Option<&mut dyn FnMut(&Stream) -> Result<(), gb_cuda::rt::CudaError>>,
        rows: usize,
        gemm_only: bool,
        hidden_in: &DevBuf,
        shared_f16: &mut DevBuf,
        picks: &DevBuf,
        wts: &DevBuf,
        moe_out_f32: &mut DevBuf,
        limit: f32,
        st: &Stream,
    ) -> Result<gb_cuda::dsv4_moe::MoeStats, ModelError> {
        let own_range = self.table.trunk_expert_range();
        let comm = &self.comm;
        let exchanged_bytes = &mut self.exchanged_bytes;
        let exchanges = &mut self.exchanges;
        let y_peer = &mut self.y_peer;
        let mut join_opt = join_shared;
        gb_cuda::dsv4_moe::moe_experts_resident_tp(
            s,
            descs,
            own_range,
            |y, yp, bytes, stream| {
                const MAX_SWAP: usize = 8 << 20;
                let mut off = 0usize;
                while off < bytes {
                    let step = (bytes - off).min(MAX_SWAP);
                    comm.swap_bytes(y, off, step, yp, off, step, stream)?;
                    *exchanged_bytes += (2 * step) as u64;
                    *exchanges += 1;
                    off += step;
                }
                if let Some(j) = join_opt.as_mut() {
                    j(stream)?;
                }
                Ok(())
            },
            y_peer,
            rows,
            gemm_only,
            hidden_in,
            shared_f16,
            picks,
            wts,
            moe_out_f32,
            limit,
            st,
        )
        .map_err(ModelError::from)
    }

    /// The rank's routed-expert `y` combine: the slot range `[own.start, own.end)` of `y` (`[slots, hidden]` f16, the
    /// single-node packing) holds this rank's experts' down projections and the peer's range holds scratch garbage — one
    /// `swap_bytes` of each rank's range into the other's place, in the same buffer, and `y` is the whole buffer the
    /// unchanged scatter reads. `hidden` is the layer's hidden width; the ranges are the layout's own (`moe_experts_tp`'s
    /// hook passes what the grouping produced), and this rank's range must be the mirror of the peer's.
    pub fn y_gather(
        &mut self,
        y: &mut DevBuf,
        own: Range<usize>,
        slots: usize,
        hidden: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let peers = self.peer_slot_range(slots, own.clone());
        let (send_off, send_len) = (own.start * hidden * 2, own.len() * hidden * 2);
        let (recv_off, recv_len) = (peers.start * hidden * 2, peers.len() * hidden * 2);
        if send_off + send_len > y.bytes() || recv_off + recv_len > y.bytes() {
            return Err(ModelError::Request(format!(
                "tensor parallelism: the routed-expert exchange of [{}, {}) + [{}, {}) slots does not fit the {}-slot y buffer",
                own.start,
                own.end,
                peers.start,
                peers.end,
                y.bytes() / (hidden * 2)
            )));
        }
        // the send range is a VIEW of the same buffer the receive range lands in (the two are disjoint), because the
        // collective's signature takes the send by share: same bytes, no second buffer
        let send = y.subview(send_off, send_len);
        self.comm
            .swap_bytes(&send, 0, send_len, y, recv_off, recv_len, st)
            .map_err(|e| ModelError::Config(format!("the routed-expert exchange: {e}")))?;
        self.exchanged_bytes += (send_len + recv_len) as u64;
        self.exchanges += 1;
        Ok(())
    }

    /// The peer's o-projection column range (the other half of the heads at two ranks).
    fn peer_o_range(&self) -> Range<usize> {
        let total = self.table.q_heads * self.table.head_dim;
        let mine = self.table.o_row_range();
        total - mine.end..total - mine.start
    }

    /// The peer's slot range of a `slots`-slot packing: the complement of this rank's `own`. `slots` is the PACKING's
    /// own slot count (the offsets' last entry) — never the scratch buffer's capacity, which is sized for the worst
    /// case and would have this rank wait for bytes the peer never sends. At two ranks the owned range is a prefix
    /// or a suffix of the packing (the split table's contiguous expert range makes it one), so the complement is one
    /// range; the two are disjoint and cover the packing exactly, which is what the in-place swap needs.
    fn peer_slot_range(&self, slots: usize, own: Range<usize>) -> Range<usize> {
        if own.start == 0 {
            own.end..slots
        } else {
            0..own.start
        }
    }

    /// [`Self::peer_slot_range`] over an explicit table, for the tests.
    #[cfg(test)]
    fn peer_slot_range_for_test(slots: usize, own: Range<usize>) -> Range<usize> {
        if own.start == 0 {
            own.end..slots
        } else {
            0..own.start
        }
    }
}

/// The refusals of [`super::model::Dsv41Model::enable_tp`], as one check the model and the tests share: the arms are
/// two-rank byte exchanges, so a wider group is refused by name (the plan seam still admits it — this is the model-step
/// choke, where a wider group's exchange would otherwise wait on a collective that never comes).
pub fn check_two_rank(topo: crate::families::nemotron_h_omni::tp::TpTopology) -> Result<(), ModelError> {
    if topo.ranks == 2 {
        return Ok(());
    }
    Err(ModelError::Config(format!(
        "deepseek_v41: the model step's tensor-parallel combine arms exchange bytes pairwise and are built for two ranks, \
         not {} (the plan, the load and the admission already shard to any count; a wider group needs the strided-place \
         kernel of a later lane)",
        topo.ranks
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::deepseek_v41::config::DeepseekV41Config;
    use crate::families::nemotron_h_omni::tp::TpTopology;

    fn table(rank: usize) -> ShardTable {
        let cfg = DeepseekV41Config::from_json(
            &serde_json::from_str(include_str!("../../../../../testdata/deepseek_v41/config.json")).unwrap(),
        )
        .unwrap();
        ShardTable::for_config(TpTopology { rank, ranks: 2 }, &cfg).unwrap()
    }

    /// The two ranks' o halves are adjacent and complementary — the invariant the placement writes its two destination
    /// column ranges by (each element of a full-width row is written by exactly one rank's copy).
    #[test]
    fn the_o_halves_are_adjacent_and_complementary() {
        let (t0, t1) = (table(0), table(1));
        let (a, b) = (t0.o_row_range(), t1.o_row_range());
        let total = t0.q_heads * t0.head_dim;
        assert_eq!((a.start, a.end, b.start, b.end), (0, total / 2, total / 2, total));
        assert_eq!(a.end, b.start, "rank 0's half ends where rank 1's begins");
        assert_eq!(a.len() + b.len(), total, "the two halves are the whole width");
    }

    /// The slot ranges mirror the same way: whatever this rank does not own of a packing is the peer's, and the two are
    /// disjoint (the in-place swap's source and destination must not overlap).
    #[test]
    fn the_slot_ranges_mirror_and_never_overlap() {
        for (slots, own) in [(22usize, 0..13usize), (22, 13..22), (8, 0..0), (8, 0..8), (64, 0..64)] {
            let peers = TpArms::peer_slot_range_for_test(slots, own.clone());
            assert_eq!(peers.len() + own.len(), slots, "the two ranges cover the packing");
            assert!(own.start >= peers.end || own.end <= peers.start, "disjoint at {own:?} vs {peers:?}");
        }
    }

    /// A wider group is refused by name, a two-rank group passes: the model-step choke the plan seam deliberately does
    /// not make (the split table shards to four).
    #[test]
    fn the_arms_refuse_a_wider_group_by_name() {
        assert!(check_two_rank(TpTopology { rank: 0, ranks: 2 }).is_ok());
        for ranks in [1usize, 4] {
            let e = check_two_rank(TpTopology { rank: 0, ranks }).unwrap_err().to_string();
            assert!(e.contains("two ranks") && e.contains(&ranks.to_string()), "names the count: {e}");
        }
    }
}
