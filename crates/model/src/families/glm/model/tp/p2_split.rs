//! P2 on lane E's KDA head split (Lane P step P5b; `docs/GLM_PREFILL_P2.md` section 10): a P2 prompt chunk's KDA layer on a rank that
//! runs half the heads.
//!
//! * **qkv**: the rank's stored strip cut (`PlannedExl3::w`, its 32 heads' q, k and v strips laid out as a layer of the kept columns)
//!   through the P2 GEMM (`gb_cuda::glm_p2::linear`). The P2 K order is a function of K alone and its N tiles sit on strip boundaries, so
//!   every column is the P2 single node's bit for bit (`crates/cuda/tests/glm_p2_dense.rs`, `p2_on_a_strip_cut_...`).
//! * **the recurrence**: the chunked KDA (`kda::prefill_chunked`) on the rank's 32 heads and its 32-head pool. Heads are independent in
//!   every KDA kernel (lane E's `kda_split.rs`), the chunk kernel's arithmetic per head is the single node's (`kda=glm-kda-chunk16-v1`).
//! * **o_proj**: option (i) of MAIN's P5b brief: the fp16 core outputs gathered (this rank's 4,096 columns, the peer's swapped in through
//!   `TpComm::swap_bytes_for(SiteUser::PrefillGather, ..)`: byte copies, no arithmetic), then the full P2 o_proj, replicated. So the
//!   layer's output is the P2 single node's by construction (no new P2 fingerprint id).

use super::GlmTp;
use super::KdaIo;
use crate::error::ModelError;
use crate::families::nemotron_h_omni::tp_roce::SiteUser;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::kda;
use gb_cuda::{DevBuf, Stream};

impl GlmTp {
    /// `full` `[rows, 2 w]` fp16 from this rank's half `own` `[rows, w]` (its columns `[r w, (r + 1) w)`) and the peer's, swapped in through
    /// the prefill gather's own RoCE sites (`SiteUser::PrefillGather`; NCCL on every other communicator): byte copies.
    fn gather_cols_f16_prefill(
        &mut self,
        own: &DevBuf,
        full: &mut DevBuf,
        rows: usize,
        w: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let r = self.split.as_ref().ok_or_else(|| ModelError::Config("no head split".into()))?.rank;
        let bytes = rows * w * 2;
        let mut peer = self.split.as_ref().unwrap().peer.subview(0, bytes);
        self.comm.swap_bytes_for(SiteUser::PrefillGather, own, 0, bytes, &mut peer, 0, bytes, st)?;
        self.exchanged_bytes += bytes as u64;
        self.exchanges += 1;
        full.copy_2d_from(r * w * 2, 2 * w * 2, own, 0, w * 2, w * 2, rows, st)?;
        full.copy_2d_from((1 - r) * w * 2, 2 * w * 2, &peer, 0, w * 2, w * 2, rows, st)?;
        Ok(())
    }

    /// One KDA layer of a P2 prompt chunk on this rank's heads (see the module): the qkv strips through the P2 GEMM, the chunked
    /// recurrence on the rank's pool (pieces of `piece` rows, `desc` from `kda::prefill_descriptor_pieces`), the gathered core output
    /// (`io.out` `[rows, 64 * 128]`) and the full P2 o_proj into `io.mixer_out`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::families::glm::model) fn kda_split_layer_p2(
        &mut self,
        ord: usize,
        o: &Exl3Weights,
        dims: kda::Dims,
        params: &kda::Params,
        pool: &mut kda::Pool,
        io: KdaIo<'_>,
        slot: usize,
        rows: usize,
        rec: &mut kda::Record,
        desc: &DevBuf,
        piece: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let KdaIo { hidden, qkv, out, mixer_out, ws, kws } = io;
        let c = dims.v_dim();
        let mut own = self.split.as_ref().ok_or_else(|| ModelError::Config("no head split".into()))?.own.subview(0, rows * c * 2);
        {
            let kr = self.kda_rank(ord).ok_or_else(|| ModelError::Config(format!("no KDA head split for KDA layer {ord}")))?;
            gb_cuda::glm_p2::linear(&kr.qkv.w, hidden, qkv, rows, Dtype::F16, Dtype::F32, ws, st)?;
            kda::prefill_chunked(
                st,
                dims,
                &kr.w.view(),
                params,
                hidden,
                Dtype::F16,
                qkv,
                Dtype::F32,
                pool,
                slot,
                rows,
                rec,
                kws,
                &mut own,
                desc,
                piece,
            )?;
        }
        self.gather_cols_f16_prefill(&own, out, rows, c, st)?;
        gb_cuda::glm_p2::linear(o, out, mixer_out, rows, Dtype::F16, Dtype::F32, ws, st)?;
        Ok(())
    }
}
