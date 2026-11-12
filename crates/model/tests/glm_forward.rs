//! GPU tests of the GLM serial forward against the ExLlamaV3 oracle's dumps, layer by layer (partial-depth models make the bisection cheap: the
//! first 4 layers are 4.6 GiB and load in a second). Env-gated and `#[ignore]`: REXL3_GLM_DIR (the pack), REXL3_GLM_ORACLE_DIR (the dumps of
//! `tools/dump_glm_reference.py`, anonymized host-d: `/srv/oracles/glm53/final4-A (example path)`), REXL3_GLM_LAYERS (default 4), REXL3_GLM_PROMPTS (default `chat_300`).
//! Without them a test prints "not run". Run under the GPU lock, one at a time (`--test-threads=1`).

mod glm_support;

use gb_model::families::glm::load::{boot, LoadOptions};
use gb_model::families::glm::model::{Capture, ForcedRouting, GlmModel, ModelOptions};
use gb_model::families::glm::npy;
use glm_support::*;
use std::path::{Path, PathBuf};

fn oracle_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) if Path::new(&d).is_dir() => Some(PathBuf::from(d)),
        Ok(d) => panic!("REXL3_GLM_ORACLE_DIR={d} is not a directory"),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        num += (x - y) * (x - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

fn ids_of(oracle: &Path, prompt: &str) -> Vec<u32> {
    npy::read(&oracle.join(prompt).join("ids.npy")).unwrap().to_i32().unwrap().iter().map(|&x| x as u32).collect()
}

fn forced_of(oracle: &Path, prompt: &str, layers: usize) -> Option<ForcedRouting> {
    let d = oracle.join(prompt);
    let ids = npy::read(&d.join("route_ids.npy")).unwrap();
    let w = npy::read(&d.join("route_w.npy")).unwrap();
    let lay = npy::read(&d.join("route_layers.npy")).unwrap().to_i32().unwrap();
    let t = ids.shape[1];
    let keep: Vec<usize> = (0..lay.len()).filter(|&i| (lay[i] as usize) < layers).collect();
    if keep.is_empty() {
        return None;
    }
    let (ids32, w32) = (ids.to_i32().unwrap(), w.to_f32().unwrap());
    let mut out = ForcedRouting { layers: vec![], t, ids: vec![], w: vec![] };
    for &i in &keep {
        out.layers.push(lay[i] as usize);
        out.ids.extend_from_slice(&ids32[i * t * 8..(i + 1) * t * 8]);
        out.w.extend(w32[i * t * 8..(i + 1) * t * 8].iter().map(|&x| gb_exl3::f16::f64_to_f16(x as f64)));
    }
    Some(out)
}

/// Run `prompt` through `m` in 128-row prefill chunks with capture on; returns the capture.
fn run_prompt(m: &mut GlmModel, ids: &[u32]) -> Capture {
    m.capture = Some(Capture::default());
    m.begin_slot(0).unwrap();
    let mut at = 0;
    while at < ids.len() {
        let end = (at + 128).min(ids.len());
        m.prefill_chunk(0, at, &ids[at..end], None).unwrap();
        at = end;
    }
    m.capture.take().unwrap()
}

#[test]
#[ignore]
fn the_first_layers_follow_the_oracle() {
    let Some(dir) = pack_dir() else { return };
    let Some(oracle) = oracle_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let prompts = std::env::var("REXL3_GLM_PROMPTS").unwrap_or_else(|_| "chat_300".into());
    require_available(40 * GIB);
    let guard = Guard::start(5 * GIB);
    let b = boot(&dir, Some(layers), &LoadOptions::from_env(8 * GIB)).unwrap();
    let mut m =
        GlmModel::new(b.pack.config.clone(), b.device, &ModelOptions { slots: 1, kv_blocks: 32, max_context: 8192, layers })
            .unwrap();
    for prompt in prompts.split(',') {
        let ids = ids_of(&oracle, prompt);
        m.forced = forced_of(&oracle, prompt, layers);
        let c = run_prompt(&mut m, &ids);
        let t = ids.len();
        let d = oracle.join(prompt);
        let eo = npy::read(&d.join("embed.npy")).unwrap().to_f32().unwrap();
        eprintln!("{prompt}: T={t}, embed rel_l2 {:.3e}", rel_l2(&c.embed, &eo));
        for l in 0..layers {
            let lo = npy::read(&d.join(format!("layer_out_{l:02}.npy"))).unwrap().to_f32().unwrap();
            let mo = npy::read(&d.join(format!("mid_{l:02}.npy"))).unwrap().to_f32().unwrap();
            let rows = lo.len() / 16384;
            let tail = (rows - 8) * 16384;
            let mid_mine = &c.mid[l][(t - 8) * 16384..];
            eprintln!(
                "  layer {l:2}: layer_out rel_l2 {:.3e}  last8 {:.3e}  post-mixer last8 {:.3e}",
                rel_l2(&c.layer_out[l], &lo),
                rel_l2(&c.layer_out[l][(t - 8) * 16384..], &lo[tail..]),
                rel_l2(mid_mine, &mo)
            );
        }
    }
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}
