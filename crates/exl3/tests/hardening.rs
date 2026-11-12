//! The checkpoint scanner reads untrusted headers and the loader reads real tensor bytes. These tests build
//! safetensors files by hand: a good one must round-trip bit for bit, and every kind of bad one must come back
//! as an error, an issue or a refused group, never a panic, a wrap or an unbounded allocation.

use gb_exl3::checkpoint::{scan, scan_indexed};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{codebook, unpack, Bits, Codebook, MARKER_MCG, MARKER_MUL1};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("hardening").join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct T {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn t(name: &str, dtype: &'static str, shape: &[usize], data: Vec<u8>) -> T {
    T { name: name.into(), dtype, shape: shape.to_vec(), data }
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A safetensors file: 8-byte length, JSON header, then the tensors back to back.
fn safetensors(tensors: &[T]) -> Vec<u8> {
    let mut hdr = serde_json::Map::new();
    let mut off = 0u64;
    for x in tensors {
        let end = off + x.data.len() as u64;
        hdr.insert(x.name.clone(), json!({"dtype": x.dtype, "shape": x.shape, "data_offsets": [off, end]}));
        off = end;
    }
    let h = serde_json::to_vec(&Value::Object(hdr)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    for x in tensors {
        out.extend_from_slice(&x.data);
    }
    out
}

struct Layer {
    trellis: Vec<i16>,
    suh: Vec<u16>,
    svh: Vec<u16>,
    kt: usize,
    nt: usize,
    bits: Bits,
}

/// One EXL3 group's tensors under `prefix`, with random trellis words and scales.
fn layer(seed: u64, kt: usize, nt: usize, twice: u8) -> Layer {
    let mut r = Rng(seed);
    let bits = Bits::from_twice(twice).unwrap();
    let trellis = (0..kt * nt * bits.tile_words()).map(|_| (r.next() & 0xFFFF) as u16 as i16).collect();
    let suh = (0..kt * 16).map(|_| f64_to_f16(0.5 + (r.next() % 1000) as f64 / 1000.0)).collect();
    let svh = (0..nt * 16).map(|_| f64_to_f16(0.001 + (r.next() % 1000) as f64 / 20000.0)).collect();
    Layer { trellis, suh, svh, kt, nt, bits }
}

fn group_tensors(prefix: &str, l: &Layer, marker: Option<(&str, u32)>) -> Vec<T> {
    let mut v = vec![
        t(
            &format!("{prefix}.trellis"),
            "I16",
            &[l.kt, l.nt, l.bits.tile_words()],
            l.trellis.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        t(&format!("{prefix}.suh"), "F16", &[l.kt * 16], le16(&l.suh)),
        t(&format!("{prefix}.svh"), "F16", &[l.nt * 16], le16(&l.svh)),
    ];
    if let Some((m, val)) = marker {
        v.push(t(&format!("{prefix}.{m}"), "I32", &[], val.to_le_bytes().to_vec()));
    }
    v
}

fn write(dir: &Path, file: &str, bytes: &[u8]) {
    std::fs::write(dir.join(file), bytes).unwrap();
}

#[test]
fn loader_round_trips_a_group_bit_for_bit() {
    let dir = tmp("roundtrip");
    let l = layer(1, 8, 8, 6); // 3 bits, K = N = 128
    let mut ts = group_tensors("m.layers.0.proj", &l, Some(("mul1", MARKER_MUL1)));
    ts.push(t("m.layers.0.proj.bias", "F32", &[128], (0..128).flat_map(|i| (i as f32 * 0.25).to_le_bytes()).collect()));
    ts.push(t("m.norm.weight", "BF16", &[4], le16(&[1, 2, 3, 4])));
    write(&dir, "model.safetensors", &safetensors(&ts));
    let ck = scan(&dir, true).unwrap();
    assert!(ck.issues.is_empty(), "{:?}", ck.issues);
    assert!(ck.bad.is_empty(), "{:?}", ck.bad);
    let g = &ck.groups["m.layers.0.proj"];
    assert_eq!((g.k, g.n, g.bits.as_f64(), g.codebook, g.bias), (128, 128, 3.0, Codebook::Mul1, true));
    let lg = ck.load_group("m.layers.0.proj").unwrap();
    assert_eq!(lg.trellis, l.trellis, "trellis words must come back exactly");
    assert_eq!(lg.suh, l.suh.iter().map(|&b| f16_to_f64(b)).collect::<Vec<_>>());
    assert_eq!(lg.svh, l.svh.iter().map(|&b| f16_to_f64(b)).collect::<Vec<_>>());
    assert_eq!(lg.bias.as_ref().unwrap()[5], 1.25);
    // decoding what was read equals decoding what was written
    let table = codebook(Codebook::Mul1);
    assert_eq!(unpack(&lg.trellis, lg.kt, lg.nt, g.bits, &table), unpack(&l.trellis, l.kt, l.nt, l.bits, &table));
    // plain tensors are not groups, and can be read raw
    assert_eq!(ck.read_tensor(&ck.plain["m.norm.weight"]).unwrap(), le16(&[1, 2, 3, 4]));
    // the 16-bit reader used by the weight loader returns the same values
    assert_eq!(ck.read_u16(&ck.plain["m.norm.weight"]).unwrap(), vec![1u16, 2, 3, 4]);
    assert!(ck.load_group("m.norm").is_err());
}

#[test]
fn packed_sign_scales_load_as_plus_minus_one() {
    let dir = tmp("signs");
    let l = layer(2, 8, 8, 4);
    let mut ts = group_tensors("p", &l, None);
    ts.retain(|x| !x.name.ends_with(".suh") && !x.name.ends_with(".svh"));
    // bit b of word w set means element 16w + b is -1
    ts.push(t("p.su", "I16", &[8], le16(&[0b1, 0, 0, 0, 0, 0, 0, 0x8000])));
    ts.push(t("p.sv", "I16", &[8], le16(&[0; 8])));
    write(&dir, "m.safetensors", &safetensors(&ts));
    let ck = scan(&dir, false).unwrap();
    assert!(ck.bad.is_empty(), "{:?}", ck.bad);
    let lg = ck.load_group("p").unwrap();
    assert_eq!((lg.group.in_scales, lg.group.out_scales), ("su", "sv"));
    assert_eq!(lg.suh[0], -1.0);
    assert_eq!(lg.suh[1], 1.0);
    assert_eq!(lg.suh[127], -1.0);
    assert!(lg.svh.iter().all(|&v| v == 1.0));
}

#[test]
fn hostile_headers_are_errors_not_allocations() {
    let dir = tmp("hostile");
    // 3 bytes: not even a length
    write(&dir, "a.safetensors", &[1, 2, 3]);
    assert!(scan(&dir, false).is_err());
    // a 2^40-byte header in a tiny file: the cap and the file size both refuse it, with no allocation
    let mut huge = (1u64 << 40).to_le_bytes().to_vec();
    huge.extend_from_slice(b"{}");
    write(&dir, "a.safetensors", &huge);
    let e = scan(&dir, false).unwrap_err();
    assert!(e.to_string().contains("exceeds"), "{e}");
    // within the cap but longer than the file
    let mut long = 1000u64.to_le_bytes().to_vec();
    long.extend_from_slice(b"{}");
    write(&dir, "a.safetensors", &long);
    let e = scan(&dir, false).unwrap_err();
    assert!(e.to_string().contains("past the end"), "{e}");
    // not an object / not JSON
    for body in [&b"[1,2]"[..], b"{not json"] {
        let mut f = (body.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(body);
        write(&dir, "a.safetensors", &f);
        assert!(scan(&dir, false).is_err());
    }
    // the error names the file
    assert!(scan(&dir, false).unwrap_err().to_string().contains("a.safetensors"));
}

fn one_group_file(dir: &Path, edit: impl Fn(&mut Value)) {
    let l = layer(3, 8, 8, 4);
    let ts = group_tensors("g", &l, Some(("mcg", MARKER_MCG)));
    let mut bytes = safetensors(&ts);
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut h: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    edit(&mut h);
    let nh = serde_json::to_vec(&h).unwrap();
    let mut out = (nh.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&nh);
    out.extend_from_slice(&bytes.split_off(8 + n));
    write(dir, "m.safetensors", &out);
}

#[test]
fn bad_offsets_and_sizes_refuse_the_group_without_panicking() {
    for (name, edit, needle) in [
        (
            "underflow",
            Box::new(|h: &mut Value| h["g.suh"]["data_offsets"] = json!([100, 4])) as Box<dyn Fn(&mut Value)>,
            "before start",
        ),
        ("past_eof", Box::new(|h: &mut Value| h["g.svh"]["data_offsets"] = json!([0, 1u64 << 40])), "past the"),
        ("size_mismatch", Box::new(|h: &mut Value| h["g.svh"]["shape"] = json!([64])), "needs"),
        ("overflow", Box::new(|h: &mut Value| h["g.svh"]["shape"] = json!([1u64 << 62, 1u64 << 62])), "overflows"),
        ("one_offset", Box::new(|h: &mut Value| h["g.suh"]["data_offsets"] = json!([0])), "expected 2"),
        (
            "no_dtype",
            Box::new(|h: &mut Value| {
                h["g.suh"].as_object_mut().unwrap().remove("dtype");
            }),
            "dtype",
        ),
        ("neg_shape", Box::new(|h: &mut Value| h["g.suh"]["shape"] = json!([-5])), "non-integer"),
    ] {
        let dir = tmp(name);
        one_group_file(&dir, edit);
        let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{name}: scan failed outright: {e}"));
        assert!(ck.groups.is_empty(), "{name}: a group with a malformed part must be refused");
        assert!(ck.bad.contains_key("g"), "{name}: {:?} {:?}", ck.bad, ck.issues);
        assert!(
            ck.bad["g"].contains(needle) || ck.issues.iter().any(|i| i.contains(needle)),
            "{name}: {:?} / {:?}",
            ck.bad,
            ck.issues
        );
        assert!(!ck.issues.is_empty(), "{name}: the malformed tensor must be reported");
    }
}

#[test]
fn group_structure_errors_are_refusals() {
    // zero-sized trellis
    let dir = tmp("zero");
    let ts = vec![
        t("z.trellis", "I16", &[0, 8, 64], vec![]),
        t("z.suh", "F16", &[0], vec![]),
        t("z.svh", "F16", &[128], le16(&[0; 128])),
    ];
    write(&dir, "m.safetensors", &safetensors(&ts));
    let ck = scan(&dir, false).unwrap();
    assert!(ck.bad["z"].contains("empty dimension"), "{:?}", ck.bad);
    // both suh and su
    let dir = tmp("both");
    let l = layer(4, 8, 8, 4);
    let mut ts = group_tensors("b", &l, Some(("mcg", MARKER_MCG)));
    ts.push(t("b.su", "I16", &[8], le16(&[0; 8])));
    write(&dir, "m.safetensors", &safetensors(&ts));
    assert!(scan(&dir, true).unwrap().bad["b"].contains("both .suh and .su"));
    // a bias of the wrong width
    let dir = tmp("bias");
    let mut ts = group_tensors("b", &l, Some(("mcg", MARKER_MCG)));
    ts.push(t("b.bias", "F16", &[7], le16(&[0; 7])));
    write(&dir, "m.safetensors", &safetensors(&ts));
    assert!(scan(&dir, true).unwrap().bad["b"].contains(".bias"));
}

#[test]
fn a_wrong_marker_value_refuses_the_group() {
    let l = layer(5, 8, 8, 4);
    // mcg tensor that holds the mul1 constant: decoding with the mcg codebook would be silently wrong
    let dir = tmp("marker");
    write(&dir, "m.safetensors", &safetensors(&group_tensors("k", &l, Some(("mcg", MARKER_MUL1)))));
    let ck = scan(&dir, true).unwrap();
    assert!(ck.groups.is_empty() && ck.bad["k"].contains("marker holds"), "{:?}", ck.bad);
    // without reading markers the group is accepted, as a header-only scan cannot know
    assert!(scan(&dir, false).unwrap().groups.contains_key("k"));
    // a marker of the wrong type
    let dir = tmp("marker_type");
    let mut ts = group_tensors("k", &l, None);
    ts.push(t("k.mul1", "F16", &[], le16(&[0])));
    write(&dir, "m.safetensors", &safetensors(&ts));
    assert!(scan(&dir, true).unwrap().bad["k"].contains("marker"));
}

#[test]
fn duplicates_across_shards() {
    let l = layer(6, 8, 8, 4);
    let ts = group_tensors("d", &l, Some(("mcg", MARKER_MCG)));
    // without an index there is no way to tell which copy is right: refuse
    let dir = tmp("dup_noindex");
    write(&dir, "a.safetensors", &safetensors(&ts));
    write(&dir, "b.safetensors", &safetensors(&ts[..1]));
    let ck = scan(&dir, true).unwrap();
    assert!(ck.bad["d"].contains("appears in both"), "{:?}", ck.bad);
    // with an index naming the owner, the owner's copy wins and the stray is reported
    let dir = tmp("dup_index");
    write(&dir, "a.safetensors", &safetensors(&ts));
    write(&dir, "b.safetensors", &safetensors(&ts[..1]));
    let wm: serde_json::Map<String, Value> = ts.iter().map(|x| (x.name.clone(), json!("a.safetensors"))).collect();
    write(&dir, "model.safetensors.index.json", json!({"weight_map": wm}).to_string().as_bytes());
    // b.safetensors is not listed, so it is not scanned at all (a stray shard cannot inject tensors)
    let ck = scan(&dir, true).unwrap();
    assert!(ck.groups.contains_key("d") && ck.bad.is_empty(), "{:?}", ck.bad);
    // list both shards, but name `a` as the owner of the trellis
    let mut wm2 = wm.clone();
    wm2.insert("d.mcg".into(), json!("b.safetensors"));
    write(&dir, "b.safetensors", &safetensors(&[t("d.trellis", "I16", &[8, 8, 32], le16(&vec![0; 8 * 8 * 32]))]));
    wm2.insert("d.extra".into(), json!("b.safetensors"));
    write(&dir, "model.safetensors.index.json", json!({"weight_map": wm2}).to_string().as_bytes());
    let ck = scan(&dir, true).unwrap();
    assert!(ck.issues.iter().any(|i| i.contains("also stored in b.safetensors; using a.safetensors")), "{:?}", ck.issues);
    assert!(ck.groups.contains_key("d"), "{:?}", ck.bad);
}

/// A published duplicate plus an overlay: the tensor is stored in two stale shards and in a third shard that sorts after both, and
/// the index names the third. This is the layout of the repaired Nemotron pack (the overlay shard `model-overlay-...` holds the
/// re-encoded matrices, whose names the published shards list as well). `scan` resolves duplicates pairwise and refuses it;
/// `scan_indexed` takes the index's copy, which is what the engine's index-driven loader does.
#[test]
fn three_copies_resolve_to_the_index_owner_only_through_scan_indexed() {
    let stale1 = group_tensors("d", &layer(10, 8, 8, 4), Some(("mcg", MARKER_MCG)));
    let stale2 = group_tensors("d", &layer(11, 8, 8, 4), Some(("mcg", MARKER_MCG)));
    let fresh_layer = layer(12, 8, 8, 4);
    let fresh = group_tensors("d", &fresh_layer, Some(("mcg", MARKER_MCG)));
    let owner = "z-overlay.safetensors"; // sorts after the two stale shards
                                         // every shard also holds a tensor of its own (the index lists a shard because it names something in it)
    let shard = |ts: &[T], own: &str| -> Vec<u8> {
        let mut v: Vec<T> = ts.iter().map(|x| t(&x.name, x.dtype, &x.shape, x.data.clone())).collect();
        v.push(t(own, "BF16", &[2], le16(&[7, 8])));
        safetensors(&v)
    };
    let index = |dir: &Path, d_in: &str, extra: &[(&str, &str)]| {
        let mut wm: serde_json::Map<String, Value> = fresh.iter().map(|x| (x.name.clone(), json!(d_in))).collect();
        for (n, s) in extra {
            wm.insert(n.to_string(), json!(s));
        }
        write(dir, "model.safetensors.index.json", json!({"weight_map": wm}).to_string().as_bytes());
    };
    let own = [("own.a", "a.safetensors"), ("own.b", "b.safetensors"), ("own.z", owner)];

    // scan: the first two copies met are both stale, so the owner is never consulted: refused, as before
    let dir = tmp("three_copies");
    write(&dir, "a.safetensors", &shard(&stale1, "own.a"));
    write(&dir, "b.safetensors", &shard(&stale2, "own.b"));
    write(&dir, owner, &shard(&fresh, "own.z"));
    index(&dir, owner, &own);
    let ck = scan(&dir, true).unwrap();
    assert!(ck.groups.is_empty(), "plain scan must keep refusing three copies: {:?}", ck.groups.keys());
    assert!(ck.bad["d"].contains("appears in both"), "{:?}", ck.bad);

    // scan_indexed: the index's copy, byte for byte, and the stale copies reported
    let ck = scan_indexed(&dir, true).unwrap();
    assert!(ck.bad.is_empty(), "{:?}", ck.bad);
    let g = ck.load_group("d").unwrap();
    assert_eq!(g.trellis, fresh_layer.trellis, "the trellis must be the index's shard's, not a stale copy's");
    assert_eq!(g.suh, fresh_layer.suh.iter().map(|&b| f16_to_f64(b)).collect::<Vec<_>>());
    assert_eq!(g.svh, fresh_layer.svh.iter().map(|&b| f16_to_f64(b)).collect::<Vec<_>>());
    assert_eq!(g.group.files, vec![owner.to_string()]);
    for stale in ["a.safetensors", "b.safetensors"] {
        let want = format!("also stored in {stale}; using {owner} as the index says");
        assert!(ck.issues.iter().any(|i| i.contains(&want)), "{want}: {:?}", ck.issues);
    }
    assert!(ck.plain.contains_key("own.a") && ck.plain.contains_key("own.b") && ck.plain.contains_key("own.z"));

    // a pack whose duplicates are all pairs: the two scans return the same entries
    let dir = tmp("pairs");
    write(&dir, "a.safetensors", &shard(&stale1, "own.a"));
    write(&dir, owner, &shard(&fresh, "own.z"));
    index(&dir, owner, &[own[0], own[2]]);
    let (p, i) = (scan(&dir, true).unwrap(), scan_indexed(&dir, true).unwrap());
    assert_eq!(p.groups, i.groups);
    assert_eq!(p.plain, i.plain);
    assert_eq!(p.bad, i.bad);
    assert_eq!(p.load_group("d").unwrap().trellis, fresh_layer.trellis);
    assert_eq!(i.load_group("d").unwrap().trellis, fresh_layer.trellis);

    // the index names a shard that does not list the tensor: an issue, and no group is assembled from a stale copy
    let dir = tmp("owner_lacks_it");
    write(&dir, "a.safetensors", &shard(&stale1, "own.a"));
    write(&dir, "b.safetensors", &shard(&stale2[1..], "own.b"));
    write(&dir, owner, &shard(&fresh, "own.z"));
    let mut wm_lacks: serde_json::Map<String, Value> = fresh.iter().map(|x| (x.name.clone(), json!(owner))).collect();
    for (n, s) in own {
        wm_lacks.insert(n.to_string(), json!(s));
    }
    wm_lacks.insert("d.trellis".into(), json!("b.safetensors"));
    write(&dir, "model.safetensors.index.json", json!({"weight_map": wm_lacks}).to_string().as_bytes());
    let ck = scan_indexed(&dir, true).unwrap();
    assert!(
        ck.issues.iter().any(|i| i.contains("d.trellis") && i.contains("b.safetensors") && i.contains("does not list it")),
        "{:?}",
        ck.issues
    );
    assert!(ck.groups.is_empty(), "no group may be assembled from a stale copy: {:?}", ck.groups.keys());
    assert!(ck.plain.contains_key("d.suh") && !ck.plain.contains_key("d.trellis"), "{:?}", ck.plain.keys());

    // a tensor the index does not name is not loaded
    let dir = tmp("unindexed");
    write(&dir, "a.safetensors", &shard(&stale1, "own.a"));
    let mut wm_one: serde_json::Map<String, Value> =
        stale1[1..].iter().map(|x| (x.name.clone(), json!("a.safetensors"))).collect();
    wm_one.insert("own.a".into(), json!("a.safetensors"));
    write(&dir, "model.safetensors.index.json", json!({"weight_map": wm_one}).to_string().as_bytes());
    let ck = scan_indexed(&dir, true).unwrap();
    assert!(ck.issues.iter().any(|i| i.contains("d.trellis") && i.contains("not in the index")), "{:?}", ck.issues);
    assert!(ck.groups.is_empty(), "{:?}", ck.groups.keys());

    // without an index there is nothing to resolve by: scan_indexed is scan
    let dir = tmp("noindex3");
    write(&dir, "a.safetensors", &shard(&stale1, "own.a"));
    write(&dir, "b.safetensors", &shard(&stale2, "own.b"));
    write(&dir, "c.safetensors", &shard(&fresh, "own.z"));
    assert!(scan_indexed(&dir, true).unwrap().bad["d"].contains("appears in both"));
}

#[test]
fn index_and_config_problems_are_reported() {
    let dir = tmp("index_missing");
    let l = layer(7, 8, 8, 4);
    write(&dir, "a.safetensors", &safetensors(&group_tensors("i", &l, Some(("mcg", MARKER_MCG)))));
    write(
        &dir,
        "model.safetensors.index.json",
        json!({"weight_map": {"i.trellis": "a.safetensors", "x": "gone.safetensors"}}).to_string().as_bytes(),
    );
    write(&dir, "config.json", b"{ this is not json");
    let ck = scan(&dir, true).unwrap();
    assert!(ck.issues.iter().any(|i| i.contains("gone.safetensors")), "{:?}", ck.issues);
    assert!(ck.issues.iter().any(|i| i.starts_with("config.json")), "{:?}", ck.issues);
    assert!(ck.config.is_null());
}

#[test]
fn truncating_or_corrupting_a_valid_file_never_panics() {
    let l = layer(8, 8, 8, 6);
    let mut ts = group_tensors("f", &l, Some(("mul1", MARKER_MUL1)));
    ts.push(t("f.norm", "BF16", &[4], le16(&[1, 2, 3, 4])));
    let good = safetensors(&ts);
    let dir = tmp("fuzz");
    let hdr_end = 8 + u64::from_le_bytes(good[..8].try_into().unwrap()) as usize;
    let mut r = Rng(99);
    let mut lens: Vec<usize> = (0..hdr_end + 16).collect();
    lens.extend((0..120).map(|_| (r.next() as usize) % good.len()));
    for len in lens {
        write(&dir, "m.safetensors", &good[..len]);
        let _ = scan(&dir, true).map(|ck| ck.groups.keys().for_each(|k| drop(ck.load_group(k))));
    }
    for _ in 0..600 {
        let mut b = good.clone();
        for _ in 0..(1 + r.next() % 3) {
            let i = (r.next() as usize) % hdr_end.min(b.len());
            b[i] ^= 1 << (r.next() % 8);
        }
        write(&dir, "m.safetensors", &b);
        let _ = scan(&dir, true).map(|ck| ck.groups.keys().for_each(|k| drop(ck.load_group(k))));
    }
}

/// Real-pack smoke test, off by default: `REXL3_PACK=/path/to/pack cargo test --release -- --ignored real_pack`.
/// Loads one group of every bit width, decodes it, and checks the layer produces finite, non-trivial output.
#[test]
#[ignore]
fn real_pack_groups_load_and_decode() {
    let Ok(root) = std::env::var("REXL3_PACK") else { panic!("set REXL3_PACK to a model folder") };
    let ck = scan(Path::new(&root), true).unwrap();
    assert!(ck.bad.is_empty(), "{:?}", ck.bad);
    let mut seen = std::collections::HashSet::new();
    for (prefix, g) in &ck.groups {
        if !seen.insert((g.bits, g.codebook)) || g.k * g.n > 4096 * 4096 {
            continue;
        }
        let lg = ck.load_group(prefix).unwrap();
        let table = codebook(g.codebook);
        let w = unpack(&lg.trellis, lg.kt, lg.nt, g.bits, &table);
        assert!(w.iter().all(|&b| f16_to_f64(b).is_finite()));
        let x: Vec<f64> = (0..g.k).map(|i| ((i * 7919) % 101) as f64 / 101.0 - 0.5).collect();
        let y = gb_exl3::forward(&x, 1, &lg.trellis, lg.kt, lg.nt, g.bits, &table, &lg.suh, &lg.svh, lg.bias.as_deref());
        assert!(y.iter().all(|v| v.is_finite()) && y.iter().any(|v| *v != 0.0), "{prefix}");
        eprintln!("{prefix}: {} bits {} K={} N={} ok", g.bits.as_f64(), g.codebook.name(), g.k, g.n);
    }
}
