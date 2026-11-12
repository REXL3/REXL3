//! The Bailing Moe V3 device image: every weight of the pack on the device, typed for the forward. Built from the loaded
//! regions (`load.rs`) and the typed pack map (`weights.rs`): each integer-width EXL3 group becomes an [`Exl3Weights`] over
//! the device region that holds its trellis / suh / svh, each plain tensor a `DevBuf` view (widened norms are fp32, the KDA
//! conv is the fused `[3 * channels, 1, 4]` slot, the MLA `kv_b_proj` is the packed `uk` / `uvt` pair), and each MoE
//! layer's routed and shared experts the pointer tables `fmoe::forward` indexes by expert id (the mixed-K routed experts run
//! through those tables alone, as half-integer widths have no `Exl3Weights` decode path).

use super::config::BailingConfig;
use super::load::Regions;
use super::weights::{BailingWeights, Exl3Linear, Exl3Part, FfnWeights, MixerWeights, PlainTensor};
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::fn_moe::ExpertTables;
use gb_cuda::mla_gated::MlaGatedWeights;
use gb_cuda::rt::{CudaError, DevBuf, Stream};
use gb_exl3::Codebook;

/// One EXL3 group on the device, at any bit width (integer or half): the raw parts and the dims. `forward` runs the
/// `k2` decode path (`k2 = 2 * bits`), which covers both.
pub struct Lin {
    pub k: usize,
    pub n: usize,
    pub k2: usize,
    pub codebook: Codebook,
    pub trellis: DevBuf,
    pub suh: DevBuf,
    pub svh: DevBuf,
}

impl Lin {
    /// A second handle to the same bytes (`DevBuf` is not `Clone`; nothing is copied, the owner is shared).
    pub fn dup(&self) -> Lin {
        Lin {
            k: self.k,
            n: self.n,
            k2: self.k2,
            codebook: self.codebook,
            trellis: self.trellis.subview(0, self.trellis.bytes()),
            suh: self.suh.subview(0, self.suh.bytes()),
            svh: self.svh.subview(0, self.svh.bytes()),
        }
    }
    fn new(r: &Regions, p: &Exl3Linear) -> Result<Self, String> {
        let part = |x: &Exl3Part| -> Result<DevBuf, String> {
            let b = r.bufs.get(&x.dst.region).ok_or_else(|| format!("{}: missing region {}", p.prefix, x.dst.region))?;
            Ok(b.subview(x.dst.offset as usize, x.bytes as usize))
        };
        Ok(Self {
            k: p.k,
            n: p.n,
            k2: p.bits.twice() as usize,
            codebook: p.codebook,
            trellis: part(&p.trellis)?,
            suh: part(&p.suh)?,
            svh: part(&p.svh)?,
        })
    }

    /// `y [m, n] = layer(x [m, k])` (row-invariant decode, any bit width).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        Exl3Weights::k2_forward(
            x,
            y,
            m,
            x_dtype,
            y_dtype,
            self.k,
            self.n,
            self.k2,
            self.codebook,
            &self.trellis,
            &self.suh,
            &self.svh,
            None,
            ws,
            st,
        )
    }

    /// `y [m, n] = layer(x [m, k])` on the pinned prefill GEMM (any `m >= 1`, row-invariant, any `x_dtype`/`y_dtype`
    /// pair the rotate kernels support — e.g. BF16 -> F32 for the KDA qkv, F16 -> BF16 for `o`): the prefill chunk's
    /// batched path instead of the 8-row decode slicing. `scratch` needs `prefill_scratch_bytes(m, k, n)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_pinned(
        &self,
        x: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        x_dtype: Dtype,
        y_dtype: Dtype,
        scratch: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        Exl3Weights::prefill_pinned_parts(
            x,
            y,
            m,
            x_dtype,
            y_dtype,
            self.k,
            self.n,
            self.k2,
            self.codebook,
            &self.trellis,
            &self.suh,
            &self.svh,
            scratch,
            st,
        )
    }
}

/// A plain tensor view, honoring the load-time `Xform` (a widened norm is fp32; everything else keeps the pack's bytes).
fn plain_view(r: &Regions, t: &PlainTensor) -> Result<DevBuf, String> {
    let b = r.bufs.get(&t.dst.region).ok_or_else(|| format!("{}: missing region {}", t.name, t.dst.region))?;
    if b.bytes() < t.dst.offset as usize + t.dst_bytes as usize {
        return Err(format!("{}: region too small", t.name));
    }
    Ok(b.subview(t.dst.offset as usize, t.dst_bytes as usize))
}

/// The three EXL3 parts of one expert projection, as device pointers (for the fmoe tables).
fn part_ptrs(r: &Regions, p: &Exl3Linear, what: &str) -> Result<[u64; 3], String> {
    let part = |x: &Exl3Part| -> Result<u64, String> {
        let b = r.bufs.get(&x.dst.region).ok_or_else(|| format!("{what}: missing region {}", x.dst.region))?;
        Ok((b.ptr_at(x.dst.offset as usize) as u64))
    };
    Ok([part(&p.trellis)?, part(&p.suh)?, part(&p.svh)?])
}

/// The KDA low-rank projection weights the full-rank gate kernel never reads (`kda::Weights` validates their sizes; they are
/// not touched by `gate_from_full_fg` / `prep` / `norm`).
pub struct KdaDummies {
    pub f_a: DevBuf,
    pub g_a: DevBuf,
    pub f_b: DevBuf,
    pub g_b: DevBuf,
}

impl KdaDummies {
    pub fn new(cfg: &BailingConfig) -> Result<Self, String> {
        let (h, c) = (cfg.hidden, cfg.kda.channels());
        let mk = |bytes: usize| DevBuf::zeroed(bytes).map_err(|e| format!("kda dummy alloc: {e}"));
        Ok(Self { f_a: mk(128 * h * 2)?, g_a: mk(128 * h * 2)?, f_b: mk(c * 128 * 2)?, g_b: mk(c * 128 * 2)? })
    }
}

pub struct KdaDevice {
    pub qkv: Lin,
    pub f: Lin,
    pub g: Lin,
    pub o: Lin,
    pub b: DevBuf,
    pub a_log: DevBuf,
    pub dt_bias: DevBuf,
    pub conv_w: DevBuf,
    pub o_norm: DevBuf,
}

impl KdaDevice {
    pub fn view<'a>(&'a self, d: &'a KdaDummies) -> gb_cuda::kda::Weights<'a> {
        gb_cuda::kda::Weights {
            b: &self.b,
            f_a: &d.f_a,
            g_a: &d.g_a,
            f_b: &d.f_b,
            g_b: &d.g_b,
            conv_w: &self.conv_w,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            o_norm: &self.o_norm,
        }
    }
}

pub struct MlaDevice {
    pub q_a: Lin,
    pub q_a_norm: DevBuf,
    pub q_b: Lin,
    pub kv_a: Lin,
    pub kv_a_norm: DevBuf,
    pub w: MlaGatedWeights,
    pub dense: Lin,
}

pub enum MixerDevice {
    Kda(KdaDevice),
    Mla(MlaDevice),
}

pub struct DenseDevice {
    pub gate: Lin,
    pub up: Lin,
    pub down: Lin,
}

pub struct MoeDevice {
    pub router: DevBuf,
    pub expert_bias: DevBuf,
    pub g_t: DevBuf,
    pub g_su: DevBuf,
    pub g_sv: DevBuf,
    pub u_t: DevBuf,
    pub u_su: DevBuf,
    pub u_sv: DevBuf,
    pub d_t: DevBuf,
    pub d_su: DevBuf,
    pub d_sv: DevBuf,
    pub ktab: DevBuf,
}

impl MoeDevice {
    pub fn tables(&self) -> ExpertTables<'_> {
        ExpertTables {
            g_t: &self.g_t,
            g_su: &self.g_su,
            g_sv: &self.g_sv,
            u_t: &self.u_t,
            u_su: &self.u_su,
            u_sv: &self.u_sv,
            d_t: &self.d_t,
            d_su: &self.d_su,
            d_sv: &self.d_sv,
            ktab: &self.ktab,
        }
    }
}

struct RawExpert<'a> {
    gate: &'a Exl3Linear,
    up: &'a Exl3Linear,
    down: &'a Exl3Linear,
}

fn build_tables(r: &Regions, experts: &[RawExpert<'_>], shared: Option<&RawExpert<'_>>) -> Result<(Vec<DevBuf>, DevBuf), String> {
    let n = experts.len() + usize::from(shared.is_some());
    let mut t: Vec<Vec<u64>> = (0..9).map(|_| Vec::with_capacity(n)).collect();
    let mut kt: Vec<i32> = Vec::with_capacity(n * 3);
    for (slot, e) in experts.iter().map(|e| (e.gate, e.up, e.down)).chain(shared.map(|s| (s.gate, s.up, s.down))).enumerate() {
        let ps = [e.0, e.1, e.2];
        for (j, p) in ps.iter().enumerate() {
            let [tr, su, sv] = part_ptrs(r, p, &format!("expert slot {slot} projection {j}"))?;
            t[3 * j].push(tr);
            t[3 * j + 1].push(su);
            t[3 * j + 2].push(sv);
            kt.push(p.bits.twice() as i32);
        }
    }
    let mut bufs = Vec::with_capacity(9);
    for v in t {
        bufs.push(DevBuf::from_slice(&v).map_err(|e| format!("expert table upload: {e}"))?);
    }
    let ktab = DevBuf::from_slice(&kt).map_err(|e| format!("expert ktab upload: {e}"))?;
    let _ = n;
    Ok((bufs, ktab))
}

pub struct LayerDevice {
    pub input_norm: DevBuf,
    pub post_norm: DevBuf,
    pub mixer: MixerDevice,
    pub ffn: FfnDevice,
}

pub enum FfnDevice {
    Dense(DenseDevice),
    Moe(MoeDevice),
}

pub struct MtpDevice {
    pub eh_proj: DevBuf,
    pub enorm: DevBuf,
    pub hnorm: DevBuf,
    pub final_layernorm: DevBuf,
    pub layer: LayerDevice,
}

pub struct BailingDevice {
    pub hidden: usize,
    pub vocab: usize,
    pub final_norm: DevBuf,
    pub lm_head: Lin,
    pub layers: Vec<LayerDevice>,
    pub mtp: Option<MtpDevice>,
    pub kda_dummies: KdaDummies,
    /// The loaded device regions, kept alive for the lifetime of the device: the MoE expert tables hold raw pointers into
    /// them, so dropping `Regions` would free the memory under the tables (`cudaFree` on drop).
    pub regions: Regions,
}

fn mixer_device(cfg: &BailingConfig, r: &Regions, m: &MixerWeights) -> Result<MixerDevice, String> {
    match m {
        MixerWeights::Kda(k) => {
            let conv = match &k.conv1d_weight {
                super::weights::CatQkvConv { fused, .. } => plain_view(r, fused)?,
            };
            Ok(MixerDevice::Kda(KdaDevice {
                qkv: Lin::new(r, &k.qkv_proj)?,
                f: Lin::new(r, &k.f_proj)?,
                g: Lin::new(r, &k.g_proj)?,
                o: Lin::new(r, &k.o_proj)?,
                b: plain_view(r, &k.b_proj)?,
                a_log: plain_view(r, &k.a_log)?,
                dt_bias: plain_view(r, &k.dt_bias)?,
                conv_w: conv,
                o_norm: plain_view(r, &k.o_norm)?,
            }))
        }
        MixerWeights::Mla(ml) => {
            let kv_b = plain_view(r, &ml.kv_b_proj)?;
            let half = cfg.mla.heads * cfg.mla.qk_nope_head_dim * cfg.mla.kv_lora_rank * 2;
            let uk = kv_b.subview(0, half);
            let uvt = kv_b.subview(half, half);
            let g = plain_view(r, &ml.g_proj)?;
            let wmla = MlaGatedWeights { uk, uvt, g, dtype: gb_cuda::exl3_linear::Dtype::BF16 };
            Ok(MixerDevice::Mla(MlaDevice {
                q_a: Lin::new(r, &ml.q_a_proj)?,
                q_a_norm: plain_view(r, &ml.q_a_layernorm)?,
                q_b: Lin::new(r, &ml.q_b_proj)?,
                kv_a: Lin::new(r, &ml.kv_a_proj_with_mqa)?,
                kv_a_norm: plain_view(r, &ml.kv_a_layernorm)?,
                w: wmla,
                dense: Lin::new(r, &ml.dense)?,
            }))
        }
    }
}

fn ffn_device(r: &Regions, f: &FfnWeights) -> Result<FfnDevice, String> {
    match f {
        FfnWeights::Dense(d) => Ok(FfnDevice::Dense(DenseDevice {
            gate: Lin::new(r, &d.gate_proj)?,
            up: Lin::new(r, &d.up_proj)?,
            down: Lin::new(r, &d.down_proj)?,
        })),
        FfnWeights::Moe(m) => {
            let shared = m.shared.as_ref().ok_or("shared expert missing")?;
            let shared_raw = RawExpert { gate: &shared.gate_proj, up: &shared.up_proj, down: &shared.down_proj };
            let experts: Vec<RawExpert<'_>> =
                m.experts.experts.iter().map(|e| RawExpert { gate: &e.gate_proj, up: &e.up_proj, down: &e.down_proj }).collect();
            let (bufs, ktab) = build_tables(r, &experts, Some(&shared_raw))?;
            let mut it = bufs.into_iter();
            Ok(FfnDevice::Moe(MoeDevice {
                router: plain_view(r, &m.router)?,
                expert_bias: plain_view(r, &m.expert_bias)?,
                g_t: it.next().unwrap(),
                g_su: it.next().unwrap(),
                g_sv: it.next().unwrap(),
                u_t: it.next().unwrap(),
                u_su: it.next().unwrap(),
                u_sv: it.next().unwrap(),
                d_t: it.next().unwrap(),
                d_su: it.next().unwrap(),
                d_sv: it.next().unwrap(),
                ktab,
            }))
        }
    }
}

fn layer_device(cfg: &BailingConfig, r: &Regions, l: &super::weights::BailingLayerWeights) -> Result<LayerDevice, String> {
    let mixer = mixer_device(cfg, r, &l.mixer)?;
    let ffn = ffn_device(r, &l.ffn)?;
    Ok(LayerDevice {
        input_norm: plain_view(r, &l.input_layernorm)?,
        post_norm: plain_view(r, &l.post_attention_layernorm)?,
        mixer,
        ffn,
    })
}

impl BailingDevice {
    pub fn build(cfg: &BailingConfig, w: &BailingWeights, regions: Regions) -> Result<Self, String> {
        let kda_dummies = KdaDummies::new(cfg)?;
        let final_norm = plain_view(&regions, &w.final_norm)?;
        let lm_head = Lin::new(&regions, &w.lm_head)?;
        let mut layers = Vec::with_capacity(w.layers.len());
        for l in &w.layers {
            layers.push(layer_device(cfg, &regions, l)?);
        }
        let mtp = match w.mtp.first() {
            Some(m) => Some(MtpDevice {
                eh_proj: plain_view(&regions, &m.eh_proj)?,
                enorm: plain_view(&regions, &m.enorm)?,
                hnorm: plain_view(&regions, &m.hnorm)?,
                final_layernorm: plain_view(&regions, &m.final_layernorm)?,
                layer: layer_device(cfg, &regions, &m.layer)?,
            }),
            None => None,
        };
        Ok(Self { hidden: cfg.hidden, vocab: cfg.vocab, final_norm, lm_head, layers, mtp, kda_dummies, regions })
    }
}
