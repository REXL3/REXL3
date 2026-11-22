//! Reproducer for X141: Ling-3.1's prefill sometimes leaves another slot state when a prompt is split into small chunks.
//! Engine only. Each run prefills the prompt in `RACE_GRID`-row chunks with `RACE_SLEEP_MS` of host sleep between them, then
//! digests every layer's state (KDA state and conv per KDA layer, the slot's MLA rows per MLA layer) after every chunk. The
//! digests are compared with the first run's, and the first (chunk, layer) that differs is printed.
mod ling31_support;
use gb_model::families::bailing_moe_v3::config::BailingConfig;
use gb_model::families::bailing_moe_v3::model::BailingModel;
use ling31_support::{fnv, gib, guard, ids, load, pack_dir};

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// One digest per layer, in layer order: "K<l>" for a KDA layer (current state half and conv), "M<l>" for an MLA layer (rows
/// `0..len` of the slot).
fn layer_digests(m: &BailingModel, slot: usize, len: usize) -> Vec<(String, String)> {
    m.stream.sync().unwrap();
    let mut out = Vec::new();
    let (kda, mla) = (m.cfg.kda_layers(), m.cfg.mla_layers());
    for (ord, &l) in kda.iter().enumerate() {
        let (s, parity, c) = m.pools[ord].slot_snapshot(slot).unwrap();
        let half = s.len() / 2;
        let cur = parity as usize & 1;
        out.push((l, format!("K{l}"), format!("{}{}", fnv(bytes(&s[cur * half..(cur + 1) * half])), fnv(bytes(&c)))));
    }
    for (ord, &l) in mla.iter().enumerate() {
        let cache = &m.mla_caches[ord];
        let bb = gb_cuda::mla_gated::MlaGatedCache::block_bytes(cache.dtype);
        let row = bb / gb_cuda::mla_gated::BLOCK_TOKENS;
        let mut all = Vec::new();
        for (i, &page) in m.slot_pages[slot].iter().enumerate() {
            let lo = i * gb_cuda::mla_gated::BLOCK_TOKENS;
            if lo >= len {
                break;
            }
            let rows = (len - lo).min(gb_cuda::mla_gated::BLOCK_TOKENS);
            let mut buf = vec![0u8; rows * row];
            cache.buf.download_range(page as usize * bb, &mut buf).unwrap();
            all.extend(buf);
        }
        out.push((l, format!("M{l}"), fnv(&all)));
    }
    out.sort();
    out.into_iter().map(|(_, a, b)| (a, b)).collect()
}

#[test]
#[ignore = "needs REXL3_LING_DIR and a CUDA device"]
fn prefill_state_does_not_depend_on_the_run() {
    let Some(dir) = pack_dir() else { return };
    let mtp = BailingConfig::from_dir(&dir).map(|c| c.layers.len() >= 72 && c.has_mtp()).unwrap_or(false);
    let Some(mut m) = load(1, 32 * 256, mtp) else { return };
    let (n, g, reps, sleep, every) =
        (env("RACE_N", 100), env("RACE_GRID", 16), env("RACE_REPS", 40), env("RACE_SLEEP_MS", 5), env("RACE_EVERY_CHUNK", 1));
    let p = ids(n as u64, n);
    let mut ends: Vec<usize> = (1..=n.div_ceil(g)).map(|i| (i * g).min(n)).collect();
    ends.dedup();
    let run = |m: &mut BailingModel| -> Vec<Vec<(String, String)>> {
        m.begin_slot(0).unwrap();
        let mut a = 0;
        let mut per_chunk = Vec::new();
        for &b in &ends {
            m.prefill_chunk(0, a, &p[a..b], None).unwrap();
            if every == 1 || b == n {
                per_chunk.push(layer_digests(m, 0, b));
            }
            if sleep > 0 && b < n {
                std::thread::sleep(std::time::Duration::from_millis(sleep as u64));
            }
            a = b;
        }
        m.end_slot(0);
        per_chunk
    };
    let reference = run(&mut m);
    let mut bad = 0;
    for r in 1..=reps {
        let got = run(&mut m);
        if got != reference {
            bad += 1;
            let (c, l) = got
                .iter()
                .zip(&reference)
                .enumerate()
                .find_map(|(c, (x, y))| x.iter().zip(y).position(|(a, b)| a != b).map(|l| (c, l)))
                .unwrap();
            let at = if every == 1 { ends[c] } else { n };
            let differing: Vec<&str> =
                got[c].iter().zip(&reference[c]).filter(|(a, b)| a != b).map(|(a, _)| a.0.as_str()).take(6).collect();
            println!(
                "RACE run={r} first differs after chunk {c} (rows to {at}) at layer {} ; first layers differing {differing:?}",
                got[c][l].0
            );
        }
    }
    println!("RACE n={n} grid={g} sleep_ms={sleep} runs={reps} differing_from_first={bad}");
    println!("RACE MemAvailable low {:.2} GiB", gib(guard().low()));
    assert_eq!(bad, 0, "{bad} of {reps} runs left another state");
}
