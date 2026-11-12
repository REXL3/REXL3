//! The candidate dump: the GLM forward's per-layer residual streams, logits and routing for one prompt, written in the file layout of
//! `tools/dump_glm_reference.py` so that `tools/compare_glm_dump.py ORACLE CANDIDATE` judges it against the oracle's dump.
//!
//! The prompt runs through the production path (128-row prefill chunks of one slot) with the model's capture on. Rows are stored for every
//! position of a prompt of up to 512 tokens and for the last 8 otherwise (`stored_rows`), as the oracle does; the top-64 and the teacher-forced
//! NLL are for every position, computed from the raw fp16 logits over the model's full 154,880 columns (the oracle's NLL and top-64 are
//! unmasked: the vocabulary mask belongs to the sampler, not to the comparison).
//!
//! `routing_mode` in `meta.json` is `forced` when the model replayed the oracle's routing (`GlmModel::forced`) and `natural` otherwise;
//! the compare tool picks its tolerance set by it.

use super::model::{Capture, GlmModel, Head, Mode};
use super::npy;
use super::vision_serve::{GlmImageFeatures, SpliceOp};
use crate::rows::RowBatch;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

/// Prompts up to this many tokens dump every position (as the oracle's dumper does).
pub const FULL_MAX: usize = 512;
/// Rows of the tail that every prompt dumps.
pub const TAIL: usize = 8;
const TOPK: usize = 64;

/// What a dumped prompt came to.
#[derive(Clone, Debug)]
pub struct Dumped {
    pub tokens: usize,
    /// The greedy next token (top-1 of the last position over all 154,880 columns).
    pub greedy_next_token: i64,
    pub seconds: f64,
}

/// Top-`TOPK` (value descending, ties to the lower id) of one row of fp16 logits, with the row's logsumexp (f64) and the logit of `target`.
fn row_stats(row: &[u16], target: Option<usize>) -> (Vec<i32>, Vec<f32>, f64, f32) {
    let f = |b: u16| gb_exl3::f16::f16_to_f64(b) as f32;
    let mut best: Vec<(f32, i32)> = Vec::with_capacity(TOPK + 1);
    let mut min_at = 0usize;
    let mut max = f32::NEG_INFINITY;
    for (i, &b) in row.iter().enumerate() {
        let v = f(b);
        if v > max {
            max = v;
        }
        if best.len() < TOPK {
            best.push((v, i as i32));
            if best.len() == TOPK {
                min_at = (0..TOPK).min_by(|&a, &b| best[a].0.total_cmp(&best[b].0).then(best[b].1.cmp(&best[a].1))).unwrap();
            }
        } else if v > best[min_at].0 {
            // ids ascend, so an equal value never displaces an earlier id
            best[min_at] = (v, i as i32);
            min_at = (0..TOPK).min_by(|&a, &b| best[a].0.total_cmp(&best[b].0).then(best[b].1.cmp(&best[a].1))).unwrap();
        }
    }
    best.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let sum: f64 = row.iter().map(|&b| ((f(b) - max) as f64).exp()).sum();
    let lse = max as f64 + sum.ln();
    let tl = target.map_or(f32::NAN, |t| f(row[t]));
    (best.iter().map(|x| x.1).collect(), best.iter().map(|x| x.0).collect(), lse, tl)
}

fn desc(shape: &[usize], dtype: &str, v: &[f64], raw: &[u8]) -> serde_json::Value {
    let l2 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let absmax = v.iter().fold(0f64, |m, x| m.max(x.abs()));
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    let sha: String = Sha256::digest(raw).iter().map(|b| format!("{b:02x}")).collect();
    serde_json::json!({ "shape": shape, "dtype": dtype, "l2": l2, "absmax": absmax, "mean": mean, "sha256": sha })
}

fn f16_rows_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect()
}

/// One image of a dumped prompt: where its placeholders start and its features (on the device).
pub struct DumpImage {
    pub start: usize,
    pub feats: Arc<GlmImageFeatures>,
}

/// Run `ids` (HF-form token ids of prompt `name`) through `m` from an empty slot 0 and write its dump under `out/name/`.
pub fn dump_prompt(m: &mut GlmModel, name: &str, kind: &str, ids: &[u32], out: &Path) -> Result<Dumped, String> {
    dump_prompt_mm(m, name, kind, ids, &[], out)
}

/// [`dump_prompt`] for a prompt with images: the rows of each image are spliced over the embeddings of its placeholders (`ids` holds the
/// placeholder id at those positions, as the oracle's HF-form ids do), the teacher-forced NLL is NaN before an image row (the oracle's rule), and
/// the dump gains `vision/features.npy` (f32 `[N, 4096]`), `vision/positions.npy` and an `image_span` in `meta.json`, the files the compare tool
/// reads. One image per dump (the oracle's prompts have one).
pub fn dump_prompt_mm(
    m: &mut GlmModel,
    name: &str,
    kind: &str,
    ids: &[u32],
    images: &[DumpImage],
    out: &Path,
) -> Result<Dumped, String> {
    let t0 = std::time::Instant::now();
    let t = ids.len();
    if t == 0 {
        return Err("an empty prompt".into());
    }
    let full = t <= FULL_MAX;
    let tail = TAIL.min(t);
    let rows_kept = if full { t } else { tail };
    let v = m.cfg.vocab;
    let hidden = m.cfg.hidden;
    let layers = m.layers();
    let forced = m.forced.is_some();
    m.capture = Some(Capture { tail: (!full).then_some(TAIL), ..Capture::default() });
    m.begin_slot(0).map_err(|e| e.to_string())?;
    // the chunk the dump prefills in: REXL3_GLM_CHUNK rows (default 128, the decode kernels' window; above it the model's large-row path, which needs a
    // model built with that many chunk rows and the all-row logits buffer)
    let chunk: usize = std::env::var("REXL3_GLM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(128).clamp(1, m.chunk_rows());
    if chunk > 128 {
        m.enable_dump_logits().map_err(|e| e.to_string())?;
    }
    let (mut top_ids, mut top_vals, mut nll) =
        (Vec::with_capacity(t * TOPK), Vec::with_capacity(t * TOPK), Vec::with_capacity(t));
    let mut last_logits: Vec<u16> = Vec::new();
    let mut at = 0;
    while at < t {
        let end = (at + chunk).min(t);
        let b = RowBatch::from_requests(&[gb_engine::RowRequest {
            slot: 0,
            tokens: ids[at..end].to_vec(),
            first_position: at as u64,
        }]);
        let splice: Vec<SpliceOp> = images
            .iter()
            .filter_map(|im| {
                let (lo, hi) = (im.start.max(at), (im.start + im.feats.tokens).min(end));
                (lo < hi).then(|| SpliceOp { feats: im.feats.clone(), dst_row: lo - at, src_row: lo - im.start, n: hi - lo })
            })
            .collect();
        m.forward_mm(&b, Mode::Prefill, Head::AllRows, splice).map_err(|e| format!("rows {at}..{end}: {e}"))?;
        let logits = m.logits_host(0, end - at).map_err(|e| e.to_string())?;
        for r in 0..end - at {
            let pos = at + r;
            // the oracle's NLL is NaN where the next token is an image row
            let next_is_image = images.iter().any(|im| (im.start..im.start + im.feats.tokens).contains(&(pos + 1)));
            let target = (pos + 1 < t && (ids[pos + 1] as usize) < v && !next_is_image).then(|| ids[pos + 1] as usize);
            let (i, x, lse, tl) = row_stats(&logits[r * v..(r + 1) * v], target);
            top_ids.extend(i);
            top_vals.extend(x);
            nll.push(if tl.is_nan() { f32::NAN } else { (lse - tl as f64) as f32 });
        }
        // the last `tail` rows over the chunks: a last chunk can hold fewer than `tail` rows (391 = 3 x 128 + 7), so the earlier chunk's end is kept too
        last_logits.extend_from_slice(&logits[..(end - at) * v]);
        if last_logits.len() > tail * v {
            last_logits.drain(..last_logits.len() - tail * v);
        }
        at = end;
    }
    let cap = m.capture.take().ok_or("the capture vanished")?;
    let d = out.join(name);
    std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    let w = |n: &str| d.join(n);
    let last_rows = |all: &[f32], row: usize, k: usize| -> Vec<f32> { all[all.len() - k * row..].to_vec() };

    npy::write_i64(&w("ids.npy"), &[t], &ids.iter().map(|&x| x as i64).collect::<Vec<_>>())?;
    let positions: Vec<i32> = ((t - rows_kept)..t).map(|p| p as i32).collect();
    npy::write_i32(&w("pos.npy"), &[rows_kept], &positions)?;
    npy::write_f32(&w("embed.npy"), &[rows_kept, hidden], &last_rows(&cap.embed, hidden, rows_kept))?;
    for l in 0..layers {
        let lo = cap.layer_out.get(l).ok_or_else(|| format!("no capture of layer {l}"))?;
        let mid = cap.mid.get(l).ok_or_else(|| format!("no capture of layer {l}'s mixer site"))?;
        npy::write_f32(&w(&format!("layer_out_{l:02}.npy")), &[rows_kept, 4, hidden], &last_rows(lo, 4 * hidden, rows_kept))?;
        npy::write_f32(&w(&format!("mid_{l:02}.npy")), &[tail, 4, hidden], &last_rows(mid, 4 * hidden, tail))?;
    }
    // a schedule without the head mean (a mutant, see `ops::Mutant::NoHeadMean`) captures neither: write zeros, so the comparison fails on them
    // (and on the logits, which come from whatever the head read) instead of this writer panicking
    let zeros = vec![0f32; rows_kept * hidden];
    let hc_head = if cap.hc_head.is_empty() { zeros.clone() } else { last_rows(&cap.hc_head, hidden, rows_kept) };
    npy::write_f32(&w("hc_head.npy"), &[rows_kept, hidden], &hc_head)?;
    let fnorm = if cap.final_norm.is_empty() { zeros } else { last_rows(&f16_rows_to_f32(&cap.final_norm), hidden, rows_kept) };
    npy::write_f32(&w("final_norm.npy"), &[rows_kept, hidden], &fnorm)?;
    let lg = f16_rows_to_f32(&last_logits);
    npy::write_f32(&w("logits_last8.npy"), &[tail, v], &lg)?;
    // the oracle's second file is the decode-shaped GEMV over the last rows; this model's head is that GEMV for every row count
    npy::write_f32(&w("logits_rows8.npy"), &[tail, v], &lg)?;
    npy::write_i32(&w("top64_ids.npy"), &[t, TOPK], &top_ids)?;
    npy::write_f32(&w("top64_vals.npy"), &[t, TOPK], &top_vals)?;
    npy::write_f32(&w("nll.npy"), &[t], &nll)?;
    // routing: MoE layers (those with captured picks), every position
    let moe_layers: Vec<usize> = cap.route_ids.keys().copied().collect();
    if !moe_layers.is_empty() {
        let (mut ids16, mut wf) = (Vec::new(), Vec::new());
        for l in &moe_layers {
            ids16.extend(cap.route_ids[l].iter().map(|&x| x as i16));
            wf.extend(cap.route_w[l].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32));
        }
        npy::write_i16(&w("route_ids.npy"), &[moe_layers.len(), t, 8], &ids16)?;
        npy::write_f32(&w("route_w.npy"), &[moe_layers.len(), t, 8], &wf)?;
        npy::write_i32(&w("route_layers.npy"), &[moe_layers.len()], &moe_layers.iter().map(|&l| l as i32).collect::<Vec<_>>())?;
    }
    // the recurrent state at the end of the prompt, per KDA layer: the state and the conv history (this model keeps the 3 raw rows the
    // recurrence needs; the oracle's descriptor covers 4, which is why the compare tool is run with --skip kda_conv)
    let mut kda = serde_json::Map::new();
    for l in 0..layers {
        if super::ops::mixer_kind(&m.cfg, l) != super::ops::MixerKind::Kda {
            continue;
        }
        let (s, conv) = m.kda_state(l, 0).map_err(|e| e.to_string())?;
        let raw_s: Vec<u8> = s.iter().flat_map(|x| x.to_le_bytes()).collect();
        let raw_c: Vec<u8> = conv.iter().flat_map(|x| x.to_le_bytes()).collect();
        let cv: Vec<f64> = conv.iter().map(|&b| f32::from_bits((b as u32) << 16) as f64).collect();
        kda.insert(
            format!("{l:02}"),
            serde_json::json!({
                "recurrent": desc(&[64, 128, 128], "float32", &s.iter().map(|&x| x as f64).collect::<Vec<_>>(), &raw_s),
                "conv": desc(&[24576, 3], "bfloat16", &cv, &raw_c),
            }),
        );
    }
    std::fs::write(w("kda_state.json"), serde_json::to_vec_pretty(&serde_json::Value::Object(kda)).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let greedy = top_ids[(t - 1) * TOPK] as i64;
    let mut image_span = serde_json::Value::Null;
    if let Some(im) = images.first() {
        if images.len() > 1 {
            return Err("a dump holds one image (the oracle's prompts have one)".into());
        }
        let f: Vec<f32> = im.feats.buf.to_vec().map_err(|e| e.to_string())?;
        let n = im.feats.tokens;
        std::fs::create_dir_all(d.join("vision")).map_err(|e| e.to_string())?;
        npy::write_f32(&d.join("vision").join("features.npy"), &[n, hidden], &f[..n * hidden])?;
        npy::write_i64(
            &d.join("vision").join("positions.npy"),
            &[n],
            &(im.start..im.start + n).map(|p| p as i64).collect::<Vec<_>>(),
        )?;
        image_span = serde_json::json!({ "start": im.start, "rows": n });
    }
    let meta = serde_json::json!({
        "name": name, "kind": kind, "T": t, "stored_rows": if full { "all".to_string() } else { format!("last {tail}") },
        "tail_rows": tail, "image_span": image_span, "routing_mode": if forced { "forced" } else { "natural" },
        "greedy_next_token": greedy, "candidate": "rexl3 glm serial forward",
        "seconds": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
    });
    std::fs::write(w("meta.json"), serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(Dumped { tokens: t, greedy_next_token: greedy, seconds: t0.elapsed().as_secs_f64() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb_exl3::f16::f64_to_f16;

    fn row(vals: &[(usize, f32)], n: usize) -> Vec<u16> {
        let mut r = vec![f64_to_f16(-5.0); n];
        for &(i, v) in vals {
            r[i] = f64_to_f16(v as f64);
        }
        r
    }

    #[test]
    fn top64_orders_by_value_then_lower_id_and_the_stats_are_exact() {
        let n = 300;
        // three equal maxima (ties to the lowest id), a runner-up, and 70 others above the floor
        let mut vals = vec![(7usize, 9.0f32), (3, 9.0), (250, 9.0), (100, 8.0)];
        vals.extend((10..80).map(|i| (i, 1.0 + i as f32 / 1000.0)));
        let r = row(&vals, n);
        let (ids, x, lse, tl) = row_stats(&r, Some(100));
        assert_eq!(&ids[..4], &[3, 7, 250, 100]);
        assert_eq!(&x[..4], &[9.0, 9.0, 9.0, 8.0]);
        assert_eq!(ids.len(), 64);
        assert!(x.windows(2).all(|w| w[0] >= w[1]));
        assert_eq!(tl, 8.0);
        // logsumexp against a direct f64 sum
        let direct: f64 = r.iter().map(|&b| gb_exl3::f16::f16_to_f64(b).exp()).sum::<f64>().ln();
        assert!((lse - direct).abs() < 1e-9, "{lse} vs {direct}");
        // no target: NaN
        assert!(row_stats(&r, None).3.is_nan());
    }
}
