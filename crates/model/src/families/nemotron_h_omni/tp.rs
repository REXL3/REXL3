//! Tensor parallelism for Nemotron 3.5 Super VL ("TP-exact"), for any rank count: the seam every TP piece codes against.
//!
//! See `docs/NEMOTRON_TP.md` for the design. The one rule: every output element is computed by exactly ONE rank with the same
//! instruction sequence the single-node engine runs, and the other ranks receive it by a sum over buffers of disjoint support (the
//! other ranks' elements are exactly `-0.0`, the additive identity; `x + (-0.0) == x`, so the sum is exact in any order and at any rank
//! count). So the TP engine is byte-identical to the single-node engine and every existing golden (step digest, replies baseline,
//! drafted == serial, the oracle) applies to it unchanged. The sharding (which matrix is sliced, which replicated, where the cuts
//! are, what a rank holds) is `tp_shard`, generic over the rank count; two ranks is the verified and measured group.
//!
//! This file is the interface only: the trait, the topology, and the split table. The NCCL implementation, the in-process loopback and
//! the lockstep channel are `TP-C`'s; the sliced weights are `TP-W`'s; the arms and the model wiring are `TP-M`'s.

use gb_cuda::{CudaError, DevBuf, Stream};

/// Where this process sits in the group. `ranks == 1` is the single-node engine: no collective is ever issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpTopology {
    pub rank: usize,
    pub ranks: usize,
}

impl TpTopology {
    pub const SINGLE: TpTopology = TpTopology { rank: 0, ranks: 1 };

    pub fn is_single(&self) -> bool {
        self.ranks == 1
    }

    /// The contiguous half-open range `[lo, hi)` of `n` units that `rank` owns when `n` is split evenly over the ranks. `n` must divide
    /// by `ranks` (every split in [`splits`] does, and a tile-aligned one stays a multiple of 128 columns).
    pub fn slice(&self, n: usize) -> (usize, usize) {
        debug_assert_eq!(n % self.ranks, 0, "{n} does not split {} ways", self.ranks);
        let w = n / self.ranks;
        (self.rank * w, (self.rank + 1) * w)
    }
}

/// The collective the TP engine needs, and only this one. Implementations: NCCL (two nodes), an in-process loopback (two shards on one
/// GPU, for the development loop and the tests), and the single-rank no-op.
///
/// `allreduce_sum_f32` sums `count` f32 elements of `buf` starting at byte offset `off_bytes`, in place, across all ranks, queued on
/// `st`. With two ranks the result is exactly `a + b` (commutative), so it does not depend on which rank is which. It must be safe to
/// capture into a CUDA graph and to replay. The caller guarantees that on every element, at most one rank holds a non-zero value.
pub trait TpComm: Send + Sync {
    fn topology(&self) -> TpTopology;
    fn allreduce_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError>;

    /// The same sum by the ONE-SHOT exchange: every rank sends its range to every peer, receives theirs into staging slots, and one
    /// kernel adds the R operands in fixed rank order (one network hop whatever R is; `(R - 1) x` the bytes). Over buffers of disjoint
    /// support it is bit-identical to [`allreduce_sum_f32`](Self::allreduce_sum_f32). The default IS the allreduce (a communicator with
    /// no one-shot form, such as [`SingleRank`] or a test double, is unaffected); `NcclTp` and the loopback group override it. The
    /// same call-order and same-count contract as the allreduce, and the same choice of form on every rank.
    fn allgather_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        self.allreduce_sum_f32(buf, off_bytes, count, st)
    }

    /// Gives up on the group: whatever collective a thread of this process is stuck in (the peer is gone) returns with an error, and every
    /// later collective fails at once. Called from another thread (the lockstep runtime's watchdog) when the peer is declared lost, so it
    /// must not wait for a lock a stuck collective could hold. For a process on its way out: the communicator is unusable afterwards. The
    /// default is a no-op (a single rank has nothing to abort).
    fn abort(&self) {}

    /// An error the communicator recorded asynchronously (NCCL's `ncclCommGetAsyncError`: a network transport that failed under a
    /// collective), as text naming it, or `None` while it is healthy. Polled by the lockstep runtime's watchdog (every
    /// `TpTimeouts::health_poll`), so it must return at once and must not take a lock a stuck collective could hold. The default is
    /// `None` (a communicator with nothing to poll).
    fn async_error(&self) -> Option<String> {
        None
    }

    /// Whether a caller may capture this communicator's collectives into a CUDA graph and replay them: every call is stream-ordered
    /// and synchronises nothing on the host. The default is `false`, so a communicator that does not say so keeps its callers'
    /// graphs off (GLM's ranks ask it before they capture a window, `families::glm::model::tp`). `NcclTp` says `true` (captured and
    /// replayed bit-exactly across two nodes, `docs/NEMOTRON_TP.md` 10.2); the loopback group synchronises the host inside every
    /// call and stays `false`.
    fn capturable(&self) -> bool {
        false
    }

    /// Two ranks (lane S, contract P2): whether [`swap_bytes`](Self::swap_bytes) is available (NCCL and the loopback group).
    fn supports_swap(&self) -> bool {
        false
    }

    /// Two ranks: send `send_bytes` of `send` (from `send_off`) to the peer and receive the peer's `recv_bytes` into `recv` (from
    /// `recv_off`), queued on `st`. A byte copy, no arithmetic. The same call sequence on both ranks, sizes matching crosswise.
    #[allow(clippy::too_many_arguments)]
    fn swap_bytes(
        &self,
        _send: &DevBuf,
        _send_off: usize,
        _send_bytes: usize,
        _recv: &mut DevBuf,
        _recv_off: usize,
        _recv_bytes: usize,
        _st: &Stream,
    ) -> Result<(), CudaError> {
        Err(CudaError(-1))
    }

    /// Two ranks (lane S, contract P2): the group's SECOND communicator, for the shared-expert split's exchanges on the MoE block's side
    /// stream (`docs/NEMOTRON_PREFILL_SPEED.md` 6.13). `None` (the default): there is none. Created at boot only when both ranks agreed
    /// on it before NCCL init (the lever label's `S` at the Hello, then the bootstrap length in `tp_boot::make_comm`); [`abort`](Self::abort) aborts it too.
    fn aux(&self) -> Option<&dyn TpComm> {
        None
    }

    /// [`swap_bytes`](Self::swap_bytes) for a named user of the exchange (lane T: REXL3's RoCE transport keeps each user's own
    /// sites, `tp_roce::SiteUser`); every other communicator ignores the user.
    #[allow(clippy::too_many_arguments)]
    fn swap_bytes_for(
        &self,
        user: super::tp_roce::SiteUser,
        send: &DevBuf,
        send_off: usize,
        send_bytes: usize,
        recv: &mut DevBuf,
        recv_off: usize,
        recv_bytes: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let _ = user;
        self.swap_bytes(send, send_off, send_bytes, recv, recv_off, recv_bytes, st)
    }

    /// REXL3's RoCE transport behind this group (lane T: its zero-copy swap, `tp_roce::RoceTp::zero_copy_swap`), `None` (the
    /// default) for every other communicator.
    fn roce_tp(&self) -> Option<&super::tp_roce::RoceTp> {
        None
    }
}

/// The no-op group of one rank.
pub struct SingleRank;

impl TpComm for SingleRank {
    fn topology(&self) -> TpTopology {
        TpTopology::SINGLE
    }

    fn allreduce_sum_f32(&self, _buf: &mut DevBuf, _off_bytes: usize, _count: usize, _st: &Stream) -> Result<(), CudaError> {
        Ok(())
    }
}

/// The split table: which dimension of which tensor is cut, and into what. The widths are the matrices' own; the cuts are derived for
/// any rank count by ONE rule (`tp_shard::column_range`: `floor(tiles * rank / ranks)` over 128-column tiles, the engine's
/// [`gb_exl3::tp::SplitMode::Floor`]; `tp_shard::head_row_cut` for the `lm_head` rows), so each is a multiple of 128 columns (the EXL3
/// trellis tile and the output Hadamard block): a rank's slice is a valid EXL3 matrix on its own and its output columns are
/// bit-identical to the same columns of the whole matrix whenever the kernel's K-reduction plan of the slice is the whole matrix's
/// (`tp_shard::slice_keeps_plan`, proved per shape on the device by `nemotron_omni_tp_weights.rs`, not assumed).
pub mod splits {
    /// Mamba `in_proj` output columns: z 8192 | xBC 10240 | dt 128 = 18560 = 145 tiles of 128, an odd count, so the cut is uneven: at two
    /// ranks rank 0 computes columns `[0, 9216)` (72 tiles) and rank 1 `[9216, 18560)` (73 tiles) ([`in_proj_cut`]). The SSM kernels run
    /// replicated on the whole row, so where the cut falls inside z | xBC | dt does not matter; only the 128 alignment does.
    pub const IN_PROJ_COLS: usize = 18560;

    /// The `in_proj` columns rank 0 of `ranks` computes, the end of its range under the engine's cut rule: 9,216 at two ranks (what the
    /// constant `IN_PROJ_CUT` used to say). Derived, not a constant: the cut moves with the rank count. At three ranks and more the
    /// serving plan replicates the matrix (a slice below 64 tiles runs another K reduction, `ShardPlan::for_config`), so this is the
    /// geometry a slice WOULD have there.
    pub fn in_proj_cut(ranks: usize) -> usize {
        super::super::tp_shard::column_range(IN_PROJ_COLS, 0, ranks).1
    }

    /// Mamba `out_proj` output columns (hidden).
    pub const HIDDEN: usize = 4096;
    /// Attention q, k, v, o output columns.
    pub const Q_COLS: usize = 4096;
    pub const KV_COLS: usize = 256;
    pub const O_COLS: usize = 4096;
    /// Vocabulary rows of `lm_head`.
    pub const VOCAB: usize = 131072;
    /// Routed experts of a MoE layer: rank `r` of `R` owns experts `[512 r / R, 512 (r + 1) / R)` (at two ranks `[256 r, 256 (r + 1))`);
    /// the others are absent on its device (the MoE kernels write zeros for an absent expert's pairs, which is the disjoint support the
    /// sum needs).
    pub const EXPERTS: usize = 512;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_ranks_split_a_dimension_into_two_contiguous_halves() {
        let (a, b) = (TpTopology { rank: 0, ranks: 2 }, TpTopology { rank: 1, ranks: 2 });
        assert_eq!(a.slice(8192), (0, 4096));
        assert_eq!(b.slice(8192), (4096, 8192));
        assert_eq!(TpTopology::SINGLE.slice(512), (0, 512));
        assert!(TpTopology::SINGLE.is_single() && !a.is_single());
    }
}
