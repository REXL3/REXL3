//! v0.1.2 (decision D6): the three Nemotron 3.5 Super VL packs this release is validated on, against their pins in
//! `gb_model::fingerprint` (`VALIDATED_TEXT`, `VALIDATED_SHARDS`). CPU only: the pack headers (about a second a pack), and, `--ignored`,
//! every byte of every pinned shard through `--verify-pack`'s own path (at disk speed).
//!
//! * `REXL3_NEMO_OMNI_DIR`: the gate's pack, taken by its index: d2c82b14... is the bf16-head pack (HF revision b546ce51), eb0f1ada... our
//!   4-bit-head pack (e3857098); another index is not a pinned pack and is reported, not checked.
//! * `REXL3_NEMO_H4_DIR`: our 4-bit-head pack (index eb0f1ada...).
//! * `REXL3_NEMO_MIXEDK_DIR`: the MixedK pack (vcruz305/Nemotron-Super-VL-EXL3-3.85bpw @39638152, index 75291cea...).
//!
//! Without a variable its pack's checks say what they did not run and pass. Each pinned pack that passes prints `PIN OK <name> ...`, and
//! `VERIFY OK <name> ...` under `--ignored`: the release gate counts the lines (scripts/gpu_gate_nemotron.sh, step `mixedk`).
//!
//!     REXL3_NEMO_OMNI_DIR=... REXL3_NEMO_H4_DIR=... REXL3_NEMO_MIXEDK_DIR=... \
//!       cargo test -p gb-model --release --test nemotron_omni_pins -- --include-ignored --nocapture --test-threads=1

use gb_model::fingerprint::{self, NEMOTRON_BF16_HEAD, NEMOTRON_H4, NEMOTRON_MIXEDK};
use gb_model::pack::Pack;
use gb_model::serve::{check_pack_dirs, check_packs, ServeOptions};
use gb_model::verify::{verify_pack, VerifyError};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

/// The pin a pack directory must match, by the sha256 of its `model.safetensors.index.json` (the identity the gate takes its pack by).
const BY_INDEX: &[(&str, &str)] = &[
    ("eb0f1adacd51688a978b1b3175bdf1e723dce211f594c22c21a49a8c6e822641", NEMOTRON_H4),
    ("d2c82b141efa9eca788792a5e3ed9f190ae7cfcca34ec19d8590524a9321ea6f", NEMOTRON_BF16_HEAD),
    ("75291cea51e97609ab93dc51f1ec5b6884e169fdc02baf3c367ca15db9cfc75a", NEMOTRON_MIXEDK),
];

/// The variables, and the pack each one is for (`None`: whichever pinned pack its index says).
const VARS: &[(&str, Option<&str>)] =
    &[("REXL3_NEMO_OMNI_DIR", None), ("REXL3_NEMO_H4_DIR", Some(NEMOTRON_H4)), ("REXL3_NEMO_MIXEDK_DIR", Some(NEMOTRON_MIXEDK))];

/// The boot facts are one table per process: the tests that write and read them back take turns.
static FACTS: Mutex<()> = Mutex::new(());

/// A change to a pack's headers: (what it is, the change, which returns the tensor it changed).
type Mutation = (&'static str, fn(&mut Pack) -> String);

fn facts_lock() -> MutexGuard<'static, ()> {
    FACTS.lock().unwrap_or_else(|e| e.into_inner())
}

fn fact(name: &str) -> Option<String> {
    gb_engine::boot::facts().into_iter().find(|(k, _)| *k == name).map(|(_, v)| v)
}

fn index_sha(dir: &Path) -> String {
    let p = dir.join("model.safetensors.index.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// The pinned packs the environment names: (variable, directory, the pin's name). A variable that is unset says so.
fn pinned_packs() -> Vec<(&'static str, PathBuf, &'static str)> {
    let mut out = Vec::new();
    for &(var, want) in VARS {
        let Some(dir) = std::env::var_os(var).map(PathBuf::from) else {
            eprintln!("not run: {var} is not set ({})", want.unwrap_or("the gate's pack"));
            continue;
        };
        let idx = index_sha(&dir);
        let name = BY_INDEX.iter().find(|(i, _)| *i == idx).map(|(_, n)| *n);
        if let Some(w) = want {
            assert_eq!(name, Some(w), "{var}={}: its index ({idx}) is not the one of {w}", dir.display());
        }
        match name {
            Some(n) => out.push((var, dir, n)),
            None => eprintln!(
                "{var}={}: index {} is not one of the three pinned packs; its pin is not checked",
                dir.display(),
                &idx[..16]
            ),
        }
    }
    out
}

fn open(dir: &Path) -> Pack {
    Pack::open(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

/// Each pinned pack fingerprints, by this release's code, to its pin; `--print-fingerprints` says so; the boot's own pack check passes
/// WITHOUT `--allow-unvalidated-pack` and reports the pin's name; and the pinned shards are exactly the shards the pack reads (so
/// `--verify-pack` leaves none unverified and names none the pack does not have).
#[test]
fn each_pinned_pack_fingerprints_to_its_pin_and_passes_the_boot_check_without_the_flag() {
    for (var, dir, name) in pinned_packs() {
        let pack = open(&dir);
        let fp = fingerprint::fingerprint(&pack.ck);
        assert_eq!(
            fingerprint::validated_pack(&fp, None),
            Some(name),
            "{var}: {} fingerprints to {fp}, not to {name}'s pin",
            dir.display()
        );
        assert_eq!(fingerprint::validation_line(&fp, None), format!("validated {name}"));
        {
            let _g = facts_lock();
            check_pack_dirs(&ServeOptions::new(dir.clone()))
                .unwrap_or_else(|e| panic!("{var}: the boot's pack check refuses {} without the flag: {e}", dir.display()));
            assert_eq!(fact("pack_validated").as_deref(), Some(name));
            assert_eq!(fact("target_fingerprint").as_deref(), Some(fp.as_str()));
        }
        let pinned: Vec<&str> = fingerprint::pinned_shards(&fp).iter().map(|s| s.0).collect();
        let read = pack.shard_files();
        assert_eq!(pinned, read, "{var}: the pinned shards are not the shards the pack's index references");
        for s in &pinned {
            assert!(dir.join(s).is_file(), "{var}: pinned shard {s} is not in {}", dir.display());
        }
        println!("PIN OK {name} {fp} shards={} index={} ({var})", pinned.len(), &index_sha(&dir)[..8]);
    }
}

/// The pin is the pack's form: one plain tensor removed, one plain tensor's dtype changed, or one EXL3 group removed gives another
/// fingerprint, which the boot check refuses by name without the flag and serves, reported unvalidated, with it.
#[test]
fn a_pinned_pack_with_a_tensor_removed_or_its_dtype_changed_is_refused_without_the_flag() {
    for (var, dir, name) in pinned_packs() {
        let pin = fingerprint::fingerprint(&open(&dir).ck);
        let mutations: [Mutation; 3] = [
            ("a plain tensor removed", |p| {
                let n = p.ck.plain.keys().next().expect("a plain tensor").clone();
                p.ck.plain.remove(&n);
                n
            }),
            ("a plain tensor's dtype changed", |p| {
                let (n, e) = p.ck.plain.iter_mut().next().expect("a plain tensor");
                e.dtype = if e.dtype == "F16" { "BF16".into() } else { "F16".into() };
                n.clone()
            }),
            ("an EXL3 group removed", |p| {
                let n = p.ck.groups.keys().next().expect("an EXL3 group").clone();
                p.ck.groups.remove(&n);
                n
            }),
        ];
        for (what, mutate) in mutations {
            let mut pack = open(&dir);
            let tensor = mutate(&mut pack);
            let fp = fingerprint::fingerprint(&pack.ck);
            assert_ne!(fp, pin, "{var}: {what} ({tensor}) kept {name}'s fingerprint");
            assert_eq!(fingerprint::validated_pack(&fp, None), None, "{var}: {what} ({tensor}) is still validated");
            let _g = facts_lock();
            let e = check_packs(&pack, None, false).expect_err("refused without the flag").to_string();
            assert!(e.contains("is not one this release was validated on") && e.contains(&fp), "{var}: {what}: {e}");
            assert!(e.contains("--allow-unvalidated-pack"), "{e}");
            check_packs(&pack, None, true).unwrap_or_else(|e| panic!("{var}: {what}: refused with the flag: {e}"));
            assert_eq!(fact("pack_validated").as_deref(), Some("no (--allow-unvalidated-pack)"));
            eprintln!("{var}: {what} ({tensor}): {fp}, refused without the flag");
        }
    }
}

/// `--verify-pack` on each pinned pack (the boot's own path: `check_pack_dirs` with the option on, which looks the pins up by the pack's
/// fingerprint): every pinned shard read and equal to its CRC-32, none unverified, no warning; and a pin that differs from the bytes
/// fails by the shard's name. Reads the whole pack (about 15 s warm, 60 to 70 GB).
#[test]
#[ignore = "reads every byte of each pinned pack (--verify-pack)"]
fn verify_pack_reads_every_pinned_shard_against_its_crc32() {
    for (var, dir, name) in pinned_packs() {
        let pack = open(&dir);
        let pins = fingerprint::pinned_shards(&fingerprint::fingerprint(&pack.ck));
        assert!(!dir.join("crc32.txt").exists(), "{var}: a crc32.txt the pins did not account for");
        let t0 = std::time::Instant::now();
        let verified = {
            let _g = facts_lock();
            let mut o = ServeOptions::new(dir.clone());
            o.verify_pack = true;
            check_pack_dirs(&o).unwrap_or_else(|e| panic!("{var}: --verify-pack refuses {}: {e}", dir.display()));
            fact("pack_verify").expect("the pack_verify fact")
        };
        let want = format!("ok ({} files ", pins.len());
        assert!(verified.starts_with(&want), "{var}: {verified} (expected {want}...)");
        assert!(verified.ends_with("; 0 warnings, 0 unverified shards"), "{var}: {verified}");
        // the smallest pinned shard against a pin one bit off: corrupt, by name
        let (small, _, crc) = *pins
            .iter()
            .min_by_key(|s| std::fs::metadata(dir.join(s.0)).map(|m| m.len()).unwrap_or(u64::MAX))
            .expect("a pinned shard");
        match verify_pack(&dir, &[(small, crc ^ 1)], &pack.shard_files()) {
            Err(VerifyError::Corrupt(v)) => {
                assert!(v.len() == 1 && v[0].starts_with(small) && v[0].contains("pinned for the validated pack"), "{v:?}")
            }
            other => panic!("{var}: a wrong pin for {small} gave {other:?}"),
        }
        println!("VERIFY OK {name} {verified} ({var}, {:.1} s)", t0.elapsed().as_secs_f64());
    }
}
