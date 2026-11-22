//! A pack's group markers are read ahead on several threads. A valid pack must load every group under an ordinary open-file limit
//! whatever its shard count, and a marker the read-ahead cannot open must be read again serially, never turned into a refusal.
//!
//! Each case re-runs this test binary under `sh -c 'ulimit -n N'` (the crate has no libc binding to lower the limit in-process),
//! scans a 64-shard, 5,120-group synthetic pack whose groups are interleaved across the shards, and compares the result with what
//! the pack holds: every group with a correct marker accepted, every group with a wrong one refused for that marker.
#![cfg(target_os = "linux")]

use gb_exl3::checkpoint::scan;
use gb_exl3::{MARKER_MCG, MARKER_MUL1};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const SHARDS: usize = 64;
const GROUPS: usize = 5_120;
/// Every 97th group's marker holds the other codebook's constant, so it must be refused.
const WRONG_EVERY: usize = 97;

const ENV_PACK: &str = "GB_EXL3_FD_CAP_PACK";
const ENV_CAP: &str = "GB_EXL3_FD_CAP";
const ENV_OUT: &str = "GB_EXL3_FD_CAP_OUT";
const ENV_SPARE: &str = "GB_EXL3_FD_CAP_SPARE";

fn prefix(i: usize) -> String {
    format!("model.layers.{i:05}.proj")
}

/// The codebook group `i` declares, and the marker value written for it.
fn marker_of(i: usize) -> (&'static str, u32, u32) {
    let (name, want, other) = if i % 2 == 0 { ("mcg", MARKER_MCG, MARKER_MUL1) } else { ("mul1", MARKER_MUL1, MARKER_MCG) };
    (name, want, if i % WRONG_EVERY == 0 { other } else { want })
}

/// One safetensors file holding the given groups: 2-bit, K = N = 128 (8 x 8 tiles of 32 words), zero trellis, unit scales.
fn shard(groups: &[usize]) -> Vec<u8> {
    let mut hdr = serde_json::Map::new();
    let mut data = Vec::new();
    let mut put = |name: String, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>, data: &mut Vec<u8>| {
        let off = data.len();
        data.extend_from_slice(&bytes);
        hdr.insert(name, json!({"dtype": dtype, "shape": shape, "data_offsets": [off, data.len()]}));
    };
    let one_f16: Vec<u8> = std::iter::repeat_n(0x3C00u16.to_le_bytes(), 128).flatten().collect();
    for &i in groups {
        let p = prefix(i);
        put(format!("{p}.trellis"), "I16", vec![8, 8, 32], vec![0; 8 * 8 * 32 * 2], &mut data);
        put(format!("{p}.suh"), "F16", vec![128], one_f16.clone(), &mut data);
        put(format!("{p}.svh"), "F16", vec![128], one_f16.clone(), &mut data);
        let (name, _, value) = marker_of(i);
        put(format!("{p}.{name}"), "I32", vec![], value.to_le_bytes().to_vec(), &mut data);
    }
    let h = serde_json::to_vec(&Value::Object(hdr)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    out
}

/// The pack, built once per case directory: group `i` lives in shard `i % SHARDS`, so consecutive groups are in different shards.
fn build_pack(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("marker_fd_cap").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for s in 0..SHARDS {
        let groups: Vec<usize> = (s..GROUPS).step_by(SHARDS).collect();
        std::fs::write(dir.join(format!("model-{:05}-of-{SHARDS:05}.safetensors", s + 1)), shard(&groups)).unwrap();
    }
    std::fs::write(dir.join("config.json"), b"{}").unwrap();
    dir
}

/// What a scan with markers must produce for the pack: accepted groups with their markers, and the refused ones with the serial
/// reader's message.
fn expected() -> (Vec<String>, BTreeMap<String, u32>, BTreeMap<String, String>) {
    let (mut groups, mut markers, mut bad) = (Vec::new(), BTreeMap::new(), BTreeMap::new());
    for i in 0..GROUPS {
        let (name, want, value) = marker_of(i);
        if value == want {
            groups.push(prefix(i));
            markers.insert(format!("{}.{name}", prefix(i)), value);
        } else {
            bad.insert(prefix(i), format!(".{name} marker holds {value:#x}, expected {want:#x}"));
        }
    }
    (groups, markers, bad)
}

/// The scan result in a comparable form.
fn summary(dir: &Path) -> String {
    let ck = scan(dir, true).unwrap();
    format!(
        "groups={:?}\nmarkers={:?}\nbad={:?}\nplain={:?}\nissues={:?}",
        ck.groups.keys().collect::<Vec<_>>(),
        ck.markers,
        ck.bad,
        ck.plain,
        ck.issues
    )
}

/// The soft open-file limit of this process, from `/proc/self/limits`.
fn open_file_limit() -> u64 {
    let limits = std::fs::read_to_string("/proc/self/limits").unwrap();
    let line = limits.lines().find(|l| l.starts_with("Max open files")).unwrap();
    line.split_whitespace().nth(3).unwrap().parse().unwrap()
}

/// The child half: scan the pack under the limit the parent set and write the summary for it. Does nothing unless the parent ran it.
#[test]
#[ignore = "run by the fd-cap cases below in a child process under a lowered open-file limit"]
fn fd_cap_child_scan() {
    let (Ok(pack), Ok(cap), Ok(out)) = (std::env::var(ENV_PACK), std::env::var(ENV_CAP), std::env::var(ENV_OUT)) else {
        return;
    };
    assert_eq!(open_file_limit(), cap.parse::<u64>().unwrap(), "the child must run under the limit the parent asked for");
    // optionally take every descriptor but `spare`, so the scan's threads compete for them while a serial read always has one
    let mut held = Vec::new();
    if let Ok(spare) = std::env::var(ENV_SPARE) {
        loop {
            match std::fs::File::open("/dev/null") {
                Ok(f) => held.push(f),
                Err(e) => {
                    assert_eq!(e.raw_os_error(), Some(24), "filling the descriptor table must stop at EMFILE: {e}");
                    break;
                }
            }
        }
        held.truncate(held.len() - spare.parse::<usize>().unwrap());
    }
    let got = summary(Path::new(&pack));
    drop(held);
    std::fs::write(out, got).unwrap();
}

/// Run [`fd_cap_child_scan`] on the pack under `ulimit -n cap` (and with only `spare` descriptors free, if given) and check what
/// it scanned.
fn scan_under_cap(name: &str, cap: u64, spare: Option<usize>) {
    let dir = build_pack(name);
    let out = dir.join("summary.txt");
    let exe = std::env::current_exe().unwrap();
    let run = Command::new("/bin/sh")
        .arg("-c")
        .arg(r#"ulimit -n "$1" && shift && exec "$@""#)
        .arg("sh")
        .arg(cap.to_string())
        .arg(&exe)
        .args(["--exact", "fd_cap_child_scan", "--ignored", "--test-threads=1"])
        .env(ENV_PACK, &dir)
        .env(ENV_CAP, cap.to_string())
        .env(ENV_OUT, &out)
        .envs(spare.map(|n| (ENV_SPARE, n.to_string())))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success() && stdout.contains("1 passed"),
        "child under ulimit -n {cap} failed: {}\n{stdout}\n{}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    let capped = std::fs::read_to_string(&out).unwrap();
    // every group the pack holds, accepted or refused exactly as its marker says; no refusal for an open or read error
    let (groups, markers, bad) = expected();
    let want = format!(
        "groups={groups:?}\nmarkers={markers:?}\nbad={bad:?}\nplain={:?}\nissues={:?}",
        BTreeMap::<String, ()>::new(),
        Vec::<String>::new()
    );
    assert_eq!(capped.lines().nth(2), want.lines().nth(2), "refusals under ulimit -n {cap}");
    assert_eq!(capped, want, "scan under ulimit -n {cap}");
    // and the same as this process's scan, which has the default limit
    assert_eq!(capped, summary(&dir), "scan under ulimit -n {cap} against the uncapped scan");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Above the read-ahead's thread count but far below threads x shards (32 x 64): every group loads.
#[test]
fn every_group_loads_under_a_256_file_limit() {
    scan_under_cap("cap256", 256, None);
}

/// One free descriptor for the read-ahead's 32 threads: the opens that lose the race fail with EMFILE, and those markers must be
/// read again serially, not refused.
#[test]
fn failed_read_ahead_opens_fall_back_to_the_serial_read() {
    scan_under_cap("spare1", 64, Some(1));
}
