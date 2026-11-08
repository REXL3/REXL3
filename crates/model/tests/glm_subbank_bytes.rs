//! Bytes a rank loads for a mixed-width GLM pack (the RED-SNOW 4.91bpw pack: 4, 5 and 6-bit experts inside one layer), from the
//! safetensors headers alone: the replicated tensors, plus each rank's owned experts laid out as subbanks (`SubbankRegion::stacked`,
//! the GLM mapper's per-subbank layout, 256-byte aligned blocks). Needs the header table `REXL3_REDSNOW_TENSORS` (a JSON object
//! `name -> [bytes, dtype, shape, shard]`; default `~/projects/minicpm-v47-cleanroom/launch-20261014/redsnow-tensors.json`); skips
//! without it. CPU only.
//!
//!     cargo test --release -p gb-model --test glm_subbank_bytes -- --nocapture

use gb_cuda::moe::{subbank_partition, ExpertOwnership, SubbankKey, SubbankRegion};
use std::collections::BTreeMap;

const GIB: f64 = (1u64 << 30) as f64;

#[test]
fn bytes_per_rank_of_a_mixed_width_pack() {
    let path = std::env::var("REXL3_REDSNOW_TENSORS").map(std::path::PathBuf::from).unwrap_or_else(|_| {
        std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("projects/minicpm-v47-cleanroom/launch-20261014/redsnow-tensors.json")
    });
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("no header table at {} (set REXL3_REDSNOW_TENSORS): not run", path.display());
        return;
    };
    let t: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&text).unwrap();
    let (hidden, inter, experts) = (4096usize, 2048usize, 288usize);
    let mut total = 0u64;
    let mut expert_bytes = 0u64;
    // (layer, expert) -> header bytes of its groups (trellis, scales, markers), and (layer, expert, proj) -> bits
    let mut per_expert: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut bits: BTreeMap<(usize, usize, String), u8> = BTreeMap::new();
    for (name, v) in &t {
        let bytes = v[0].as_u64().unwrap();
        total += bytes;
        let Some(rest) = name.strip_prefix("model.language_model.layers.") else { continue };
        let mut it = rest.split('.');
        let layer: usize = it.next().unwrap().parse().unwrap();
        if it.next() != Some("mlp") || it.next() != Some("experts") {
            continue;
        }
        let e: usize = it.next().unwrap().parse().unwrap();
        let proj = it.next().unwrap().to_string();
        let part = it.next().unwrap();
        expert_bytes += bytes;
        *per_expert.entry((layer, e)).or_default() += bytes;
        if part == "trellis" {
            let shape = v[2].as_array().unwrap();
            bits.insert((layer, e, proj), (shape[2].as_u64().unwrap() / 16) as u8);
        }
    }
    let replicated = total - expert_bytes;
    let layers: Vec<usize> = per_expert.keys().map(|k| k.0).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    assert_eq!(layers.len(), 43);
    let key = |l: usize, e: usize| SubbankKey {
        bits_gate: bits[&(l, e, "gate_proj".into())],
        bits_up: bits[&(l, e, "up_proj".into())],
        bits_down: bits[&(l, e, "down_proj".into())],
        codebook: gb_exl3::Codebook::Mul1,
    };
    // a rank's expert regions: per layer, its subbanks stacked as the mapper lays them out
    let region_bytes = |own: &ExpertOwnership| -> (u64, u64, usize) {
        let (mut region, mut header, mut subbanks) = (0u64, 0u64, 0usize);
        for &l in &layers {
            let keys: Vec<SubbankKey> = (0..experts).map(|e| key(l, e)).collect();
            let mut cursor = 0usize;
            for (k, o) in subbank_partition(&keys, own).unwrap() {
                let (lay, end) = SubbankRegion::stacked(k, o.local(), hidden, inter, cursor, 256);
                assert_eq!(lay.gate.trellis_stride, (hidden / 16) * (inter / 16) * 16 * k.bits_gate as usize * 2);
                cursor = end;
                subbanks += 1;
                header += o.owned().iter().map(|&e| per_expert[&(l, e as usize)]).sum::<u64>();
            }
            region += cursor.div_ceil(256) as u64 * 256;
        }
        (region, header, subbanks)
    };
    let all = region_bytes(&ExpertOwnership::all(experts));
    eprintln!(
        "header: total {:.2} GiB = replicated {:.2} GiB + routed experts {:.2} GiB (markers included); single node: {} subbanks, \
         expert regions {:.2} GiB",
        total as f64 / GIB,
        replicated as f64 / GIB,
        expert_bytes as f64 / GIB,
        all.2,
        all.0 as f64 / GIB
    );
    assert_eq!(all.1, expert_bytes);
    // R = 2: the contiguous ranges, and a table balanced by bytes (longest first onto the lighter rank, over all layers at once: one
    // ownership serves every layer)
    let mut by_expert: Vec<(u64, usize)> =
        (0..experts).map(|e| (layers.iter().map(|&l| per_expert[&(l, e)]).sum::<u64>(), e)).collect();
    by_expert.sort_by(|a, b| b.cmp(a));
    let mut table = vec![0u32; experts];
    let mut load = [0u64; 2];
    for (b, e) in by_expert {
        let r = if load[0] <= load[1] { 0 } else { 1 };
        table[e] = r as u32;
        load[r] += b;
    }
    for (name, owns) in [
        ("range", (0..2).map(|r| ExpertOwnership::range(experts, r, 2).unwrap()).collect::<Vec<_>>()),
        ("balanced", (0..2).map(|r| ExpertOwnership::from_table(&table, r, 2).unwrap()).collect()),
    ] {
        let per: Vec<(u64, u64, usize)> = owns.iter().map(&region_bytes).collect();
        assert_eq!(per[0].1 + per[1].1, expert_bytes, "{name}: every expert byte on exactly one rank");
        for (r, (region, header, subbanks)) in per.iter().enumerate() {
            eprintln!(
                "R=2 {name} rank {r}: {} experts a layer, {subbanks} subbanks; header bytes {:.2} GiB (replicated {:.2} + experts {:.2}); \
                 device expert regions {:.2} GiB",
                owns[r].local(),
                (replicated + header) as f64 / GIB,
                replicated as f64 / GIB,
                *header as f64 / GIB,
                *region as f64 / GIB
            );
        }
        eprintln!(
            "R=2 {name}: mean {:.2} GiB per rank, skew {:+.2} GiB (rank 0 - rank 1)",
            (2 * replicated + expert_bytes) as f64 / 2.0 / GIB,
            (per[0].1 as f64 - per[1].1 as f64) / GIB
        );
    }
}
