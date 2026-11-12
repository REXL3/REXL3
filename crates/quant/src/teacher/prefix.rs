//! Quantizing a captured layer and decoding it back: the weights the quantized-prefix stream advances through.

use super::config::{qmap_of, Target};
pub use super::Hessians;
use super::Teacher;
use crate::core::dev::{transpose, Gpu, Mat, QuantError, Result};
use crate::core::{self, Options, OutScales, Output};
use gb_exl3::{codebook, f16::f16_to_f64, tile_positions, Codebook};
use std::collections::BTreeMap;

pub struct Quantized {
    pub key: String,
    pub out: Output,
    /// the stored weight decoded, (out x in) fp32 on the device
    pub weight: Mat,
}

/// W_r (k x n) of a tensor's states ([k/16, n/16, 256], trellis order): each state's codebook value.
pub fn decode_wr(states: &[i16], k: usize, n: usize, table: &[u16]) -> Vec<f32> {
    let (r, c) = tile_positions();
    let (tk, tn) = (k / 16, n / 16);
    let mut w = vec![0f32; k * n];
    for a in 0..tk {
        for b in 0..tn {
            let t = &states[(a * tn + b) * 256..(a * tn + b + 1) * 256];
            for p in 0..256 {
                let v = f16_to_f64(table[t[p] as u16 as usize]) as f32;
                w[(16 * a + r[p] as usize) * n + 16 * b + c[p] as usize] = v;
            }
        }
    }
    w
}

/// diag(suh) P W_r P diag(svh) of a stored tensor, as (out x in) (the converter's dequantization).
pub fn dequant(g: &Gpu, o: &Output, k: usize, n: usize, inp: usize, out: usize, table: &[u16]) -> Result<Mat> {
    let wr = Mat::from_host(&decode_wr(&o.states, k, n, table), k, n)?;
    let su: Vec<f32> = o.suh.iter().map(|h| f16_to_f64(*h) as f32).collect();
    let sv: Vec<f32> = o.svh.iter().map(|h| f16_to_f64(*h) as f32).collect();
    let q = core::reconstruct(g, &wr, &su, &sv)?;
    let t = transpose(g, &q)?;
    if (inp, out) == (k, n) {
        return Ok(t);
    }
    let c = Mat::alloc(out, inp)?;
    crate::core::dev::copy2d(g, c.ptr(), inp, t.ptr(), k, out, inp)?;
    g.sync()?;
    Ok(c)
}

fn pad(v: &[f32], rows: usize, cols: usize, rp: usize, cp: usize) -> Vec<f32> {
    if (rows, cols) == (rp, cp) {
        return v.to_vec();
    }
    let mut out = vec![0f32; rp * cp];
    for i in 0..rows {
        out[i * cp..i * cp + cols].copy_from_slice(&v[i * cols..(i + 1) * cols]);
    }
    out
}

/// Quantizes layer l's targets (`bits` each, mul1, output scales always, parity core options) against their
/// captured H, with module index `seed` (layer + 1) seeding the named sign streams.
pub fn quantize_layer(g: &mut Gpu, t: &Teacher, l: usize, hess: &Hessians, bits: f64, seed: u64) -> Result<Vec<Quantized>> {
    let specs: Vec<Target> = super::config::targets(&t.cfg, l);
    let table = codebook(Codebook::Mul1);
    let mut opts = Options::parity(bits);
    opts.out_scales = OutScales::Always;
    // targets grouped by qmap in first-use order, so each H is prepared once
    let mut order: Vec<String> = Vec::new();
    let mut by_q: BTreeMap<String, Vec<&Target>> = BTreeMap::new();
    for s in &specs {
        let q = qmap_of(&t.cfg, s, true);
        let e = by_q.entry(q.clone()).or_default();
        if e.is_empty() {
            order.push(q);
        }
        e.push(s);
    }
    let mut out = Vec::with_capacity(specs.len());
    for q in order {
        let (k0, h, count) = hess.get(&q).ok_or_else(|| QuantError::Shape(format!("no captured H for {q}")))?;
        let k = k0.div_ceil(128) * 128;
        let hp = pad(h, *k0, *k0, k, k);
        let prep = core::prepare(g, Some(&hp), k, *count, &core::signs_named(seed, &format!("su:{q}"), k), &opts)?;
        for s in &by_q[&q] {
            let w = t.target_weight(s)?;
            let (o0, i0) = (s.out_features, s.in_features);
            if w.len() != o0 * i0 || i0 != *k0 {
                return Err(QuantError::Shape(format!("{}: {} values for {o0} x {i0} (H k = {k0})", s.key, w.len())));
            }
            let mut wt = vec![0f32; i0 * o0];
            for o in 0..o0 {
                for i in 0..i0 {
                    wt[i * o0 + o] = w[o * i0 + i];
                }
            }
            let n = o0.div_ceil(128) * 128;
            let wt = pad(&wt, i0, o0, k, n);
            let r = core::quantize(
                g,
                &prep,
                &core::Tensor { weight: &wt, k, n, out_sensitivity: None },
                &core::signs_named(seed, &format!("sv:{}", s.key), n),
                &opts,
            )?;
            let weight = dequant(g, &r, k, n, i0, o0, &table)?;
            out.push(Quantized { key: s.key.clone(), out: r, weight });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_places_tile_positions() {
        // states equal to their position index in tile (0, 1) of a 16 x 32 matrix
        let mut states = vec![0i16; 2 * 256];
        for p in 0..256 {
            states[256 + p] = p as i16;
        }
        let table: Vec<u16> = (0..65536u32).map(|s| gb_exl3::f16::f64_to_f16(s as f64 % 2048.0)).collect();
        let w = decode_wr(&states, 16, 32, &table);
        let (r, c) = tile_positions();
        for p in 0..256 {
            assert_eq!(w[r[p] as usize * 32 + 16 + c[p] as usize], p as f32);
            assert_eq!(w[r[p] as usize * 32 + c[p] as usize], 0.0);
        }
    }
}
