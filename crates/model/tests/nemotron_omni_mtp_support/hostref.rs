//! The host side of the head's gate: tensors of the pack read straight from its shards (bf16 widened to f32, exact), the head's
//! `mtp_ref::MtpWeights` built from the REAL weights, and the CPU logits through the trunk's `lm_head`. Nothing here prints tensor values:
//! the tests report scalars (relative errors, maxima, counts) only.
#![allow(dead_code)]

use gb_exl3::f16::f16_to_f64;
use gb_exl3::{Bits, Codebook, MARKER_MUL1};
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::mtp_ref::{hf_from_exl3, ExpertWeights, MtpGeom, MtpWeights};
use gb_model::families::nemotron_h_omni::weights::LM;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub struct Pack {
    dir: PathBuf,
    inv: Inventory,
}

fn widen(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

impl Pack {
    pub fn open(dir: &Path) -> Pack {
        let inv = Inventory::read(dir).unwrap_or_else(|e| panic!("{e}"));
        Pack { dir: dir.to_path_buf(), inv }
    }

    fn read_at(&self, name: &str, byte_off: u64, len: usize) -> Vec<u8> {
        let t = self.inv.get(name).unwrap_or_else(|| panic!("`{name}` is not in the pack"));
        assert!(byte_off + len as u64 <= t.nbytes, "`{name}`: read past the tensor");
        let mut f = File::open(self.dir.join(&t.shard)).unwrap_or_else(|e| panic!("{}: {e}", t.shard));
        f.seek(SeekFrom::Start(t.offset + byte_off)).unwrap();
        let mut v = vec![0u8; len];
        f.read_exact(&mut v).unwrap();
        v
    }

    /// A bf16 tensor widened to f32 (exact).
    pub fn bf16(&self, name: &str) -> Vec<f32> {
        let t = self.inv.get(name).unwrap_or_else(|| panic!("`{name}` is not in the pack"));
        assert_eq!(t.dtype, "BF16", "{name}");
        widen(&self.read_at(name, 0, t.nbytes as usize))
    }

    /// A router gate widened to f32 (exact): BF16, or F16 (v0.1.2 decision D1: a MixedK pack's F16 gate, which the engine widens to f32
    /// on the device; never rounded to bf16).
    pub fn gate(&self, name: &str) -> Vec<f32> {
        let t = self.inv.get(name).unwrap_or_else(|| panic!("`{name}` is not in the pack"));
        let raw = self.read_at(name, 0, t.nbytes as usize);
        match t.dtype.as_str() {
            "BF16" => widen(&raw),
            "F16" => raw.chunks_exact(2).map(|c| gb_exl3::f16::f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect(),
            d => panic!("{name}: a router gate is BF16 or F16, not {d}"),
        }
    }

    /// An f32 tensor.
    pub fn f32(&self, name: &str) -> Vec<f32> {
        let t = self.inv.get(name).unwrap_or_else(|| panic!("`{name}` is not in the pack"));
        assert_eq!(t.dtype, "F32", "{name}");
        self.read_at(name, 0, t.nbytes as usize).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Rows `rows` of a bf16 matrix `[_, width]`, widened.
    pub fn bf16_rows(&self, name: &str, rows: &[u32], width: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows.len() * width);
        for &r in rows {
            out.extend(widen(&self.read_at(name, r as u64 * width as u64 * 2, width * 2)));
        }
        out
    }

    /// `logits[r][v] = sum_k lm_head[v][k] * x[r][k]` for the rows of `x` (f32 `[rows, hidden]`), streaming the head in blocks of `block`
    /// vocabulary rows (f32 accumulation in a fixed order).
    pub fn lm_head_logits(&self, x: &[f32], rows: usize, hidden: usize, vocab: usize) -> Vec<f32> {
        if self.is_exl3(&format!("{LM}.lm_head")) {
            return self.lm_head_logits_exl3(x, rows, hidden, vocab);
        }
        let name = format!("{LM}.lm_head.weight");
        let block = 2048;
        let mut out = vec![0f32; rows * vocab];
        let mut v0 = 0;
        while v0 < vocab {
            let n = block.min(vocab - v0);
            let w = widen(&self.read_at(&name, (v0 * hidden * 2) as u64, n * hidden * 2));
            for r in 0..rows {
                let xr = &x[r * hidden..(r + 1) * hidden];
                for (j, wr) in w.chunks_exact(hidden).enumerate() {
                    let mut acc = [0f32; 8];
                    for (a, b) in xr.chunks_exact(8).zip(wr.chunks_exact(8)) {
                        for l in 0..8 {
                            acc[l] += a[l] * b[l];
                        }
                    }
                    out[r * vocab + v0 + j] = acc.iter().sum();
                }
            }
            v0 += n;
        }
        out
    }

    /// [`Pack::lm_head_logits`] of an EXL3 `lm_head` (the MixedK pack's 8-bit head): each 128-column strip decoded on the host
    /// (`gb_exl3::decode_strip`, float64), every logit a float64 dot product, on up to 16 threads.
    fn lm_head_logits_exl3(&self, x: &[f32], rows: usize, hidden: usize, vocab: usize) -> Vec<f32> {
        let prefix = format!("{LM}.lm_head");
        let part = |p: &str| {
            let name = format!("{prefix}.{p}");
            let t = self.inv.get(&name).unwrap_or_else(|| panic!("`{name}` is not in the pack"));
            (t.shape.clone(), self.read_at(&name, 0, t.nbytes as usize))
        };
        let (shape, tr) = part("trellis");
        assert_eq!((shape[0] * 16, shape[1] * 16), (hidden, vocab), "{prefix}.trellis");
        let bits = Bits::from_trellis_last(shape[2]).unwrap_or_else(|| panic!("{prefix}: trellis width {}", shape[2]));
        let f16s = |b: &[u8]| -> Vec<f64> { b.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect() };
        let (suh, svh) = (f16s(&part("suh").1), f16s(&part("svh").1));
        let trellis: Vec<i16> = tr.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let table = gb_exl3::codebook(Codebook::Mul1);
        let (kt, nt, strips) = (shape[0], shape[1], vocab / 128);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 16);
        let parts: Vec<Vec<(usize, Vec<f32>)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let (trellis, table, suh, svh) = (&trellis, &table, &suh, &svh);
                    sc.spawn(move || {
                        let mut out = Vec::new();
                        for s in (t..strips).step_by(threads) {
                            let d = gb_exl3::decode_strip(trellis, kt, nt, bits, table, suh, svh, s);
                            let mut y = vec![0f32; rows * 128];
                            for r in 0..rows {
                                for c in 0..128 {
                                    let acc: f64 = (0..hidden).map(|i| x[r * hidden + i] as f64 * d[i * 128 + c]).sum();
                                    y[r * 128 + c] = acc as f32;
                                }
                            }
                            out.push((s, y));
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut out = vec![0f32; rows * vocab];
        for (s, y) in parts.into_iter().flatten() {
            for r in 0..rows {
                out[r * vocab + s * 128..r * vocab + (s + 1) * 128].copy_from_slice(&y[r * 128..(r + 1) * 128]);
            }
        }
        out
    }

    /// Whether the pack stores `prefix` as an EXL3 linear (its `.trellis`): a 4-bit head's projection.
    pub fn is_exl3(&self, prefix: &str) -> bool {
        self.inv.get(&format!("{prefix}.trellis")).is_some()
    }

    /// A head projection `[out, inp]` (HF layout) as f32: the bf16 `{prefix}.weight` widened (exact), or, for a 4-bit head, its EXL3
    /// linear decoded on the host (`mtp_ref::hf_from_exl3`), the codebook marker checked.
    pub fn proj(&self, prefix: &str, out: usize, inp: usize) -> Vec<f32> {
        if !self.is_exl3(prefix) {
            let w = self.bf16(&format!("{prefix}.weight"));
            assert_eq!(w.len(), out * inp, "{prefix}");
            return w;
        }
        let part = |p: &str| {
            let name = format!("{prefix}.{p}");
            let t = self.inv.get(&name).unwrap_or_else(|| panic!("`{name}` is not in the pack"));
            (t.dtype.clone(), t.shape.clone(), self.read_at(&name, 0, t.nbytes as usize))
        };
        let (dt, shape, tr) = part("trellis");
        assert_eq!((dt.as_str(), shape[0] * 16, shape[1] * 16), ("I16", inp, out), "{prefix}.trellis");
        let bits = Bits::from_trellis_last(shape[2]).unwrap_or_else(|| panic!("{prefix}: trellis width {}", shape[2]));
        let (_, _, marker) = part("mul1");
        assert_eq!(u32::from_le_bytes([marker[0], marker[1], marker[2], marker[3]]), MARKER_MUL1, "{prefix}.mul1");
        let f16s = |b: &[u8]| -> Vec<f64> { b.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect() };
        let (sdt, _, suh) = part("suh");
        let (vdt, _, svh) = part("svh");
        assert_eq!((sdt.as_str(), vdt.as_str()), ("F16", "F16"), "{prefix} scales");
        let trellis: Vec<i16> = tr.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        hf_from_exl3(&trellis, inp, out, bits, Codebook::Mul1, &f16s(&suh), &f16s(&svh))
    }

    /// The head's weights, f32, ALL 512 experts (about 12 GB on the host). `mtp_ref`'s own layout. A 4-bit head's EXL3 projections (the
    /// MixedK pack's attention too) are decoded on the host ([`Pack::proj`]); everything else is the pack's bf16, widened.
    pub fn head_weights(&self, cfg: &OmniConfig) -> MtpWeights {
        let b0 = format!("{LM}.mtp.layers.0");
        let b1 = format!("{LM}.mtp.layers.1");
        let m = format!("{b1}.mixer");
        let h = cfg.hidden;
        let experts = (0..cfg.experts)
            .map(|e| {
                Some(ExpertWeights {
                    up: self.proj(&format!("{m}.experts.{e}.up_proj"), cfg.expert_inter, cfg.latent),
                    down: self.proj(&format!("{m}.experts.{e}.down_proj"), cfg.latent, cfg.expert_inter),
                })
            })
            .collect();
        MtpWeights {
            geom: MtpGeom::from_config(cfg),
            eh_proj: self.proj(&format!("{b0}.eh_proj"), h, 2 * h),
            enorm: self.bf16(&format!("{b0}.enorm.weight")),
            hnorm: self.bf16(&format!("{b0}.hnorm.weight")),
            attn_norm: self.bf16(&format!("{b0}.norm.weight")),
            // the attention bf16, or EXL3 decoded on the host (the MixedK pack's head)
            wq: self.proj(&format!("{b0}.mixer.q_proj"), cfg.heads * cfg.head_dim, h),
            wk: self.proj(&format!("{b0}.mixer.k_proj"), cfg.kv_heads * cfg.head_dim, h),
            wv: self.proj(&format!("{b0}.mixer.v_proj"), cfg.kv_heads * cfg.head_dim, h),
            wo: self.proj(&format!("{b0}.mixer.o_proj"), h, cfg.heads * cfg.head_dim),
            moe_norm: self.bf16(&format!("{b1}.norm.weight")),
            gate: self.gate(&format!("{m}.gate.weight")),
            gate_bias: self.f32(&format!("{m}.gate.e_score_correction_bias")),
            fc1: self.proj(&format!("{m}.fc1_latent_proj"), cfg.latent, h),
            fc2: self.proj(&format!("{m}.fc2_latent_proj"), h, cfg.latent),
            shared_up: self.proj(&format!("{m}.shared_experts.up_proj"), cfg.shared_inter, h),
            shared_down: self.proj(&format!("{m}.shared_experts.down_proj"), h, cfg.shared_inter),
            experts,
            final_norm: self.bf16(&format!("{b1}.final_layernorm.weight")),
        }
    }
}

/// Relative L2 error of `a` against the reference `b` over equal-length slices: `|a - b| / |b|` (0 for two zero vectors).
pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        if num == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (num / den).sqrt()
    }
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Rounds an f32 to the bf16 grid (round to nearest even), as a bf16 model's activations are.
pub fn round_bf16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    f32::from_bits(r & 0xffff_0000)
}
