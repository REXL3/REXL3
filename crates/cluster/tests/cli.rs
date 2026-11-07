//! The binary itself: exit statuses and the output a script can rely on. No ssh, no node: `validate`, `env` and every `--dry-run`.

use std::path::PathBuf;
use std::process::{Command, Output};

fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/examples").join(name)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rexl3-cluster")).args(args).output().expect("run rexl3-cluster")
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

#[test]
fn validate_accepts_the_shipped_examples() {
    for f in [
        "cluster.crs812-2node.json",
        "cluster.4node.json",
        "cluster.dp-fleet.json",
        "cluster.smoke-2node.json",
        "cluster.glm-tp2.json",
        "cluster.glm-redsnow-tp2.json",
        "cluster.glm-tp2-vision.json",
        "cluster.glm-redsnow-tp2-vision.json",
        "cluster.glm-tp2-mtp.json",
        "cluster.glm-redsnow-tp2-mtp.json",
        "cluster.glm-sc410-tp2.json",
        "cluster.glm-sc410-tp2-mtp.json",
        "cluster.glm-sc410-tp2-vision.json",
        "cluster.glm-sc410-tp2-vision-mtp.json",
        "cluster.glm-sc410-tp2-longctx.json",
        "cluster.glm-sc410-tp2-longctx-mtp.json",
    ] {
        let o = run(&["validate", "--cluster", example(f).to_str().unwrap()]);
        assert_eq!(o.status.code(), Some(0), "{f}: {}", text(&o));
        assert!(text(&o).starts_with("ok: cluster "), "{}", text(&o));
    }
}

#[test]
fn a_file_naming_a_disabled_node_exits_2_and_names_it_without_touching_a_node() {
    let src = std::fs::read_to_string(example("cluster.crs812-2node.json")).unwrap();
    let bad = src.replace(r#""ranks": ["N1", "N4"]"#, r#""ranks": ["N1", "N3"]"#);
    let dir = std::env::temp_dir().join(format!("rexl3-cluster-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("bad.json");
    std::fs::write(&f, bad).unwrap();
    for cmd in ["validate", "check", "up", "down", "status", "supervise", "env"] {
        // --dry-run so that even a bug cannot reach a node
        let o = run(&[cmd, "--cluster", f.to_str().unwrap(), "--group", "g0", "--dry-run"]);
        assert_eq!(o.status.code(), Some(2), "{cmd}: {}", text(&o));
        assert!(text(&o).contains("node N3, which is disabled"), "{cmd}: {}", text(&o));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn env_prints_the_assignments_the_two_node_scripts_read() {
    let o = run(&["env", "--cluster", example("cluster.crs812-2node.json").to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o));
    let t = text(&o);
    assert!(t.contains("NODE4=${NODE4:-rexl3@192.0.2.4}"), "{t}");
    assert!(t.contains("LEADER_ADDR=${LEADER_ADDR:-192.0.2.1}"), "{t}");
    assert!(t.contains("TPPORT=${TPPORT:-29600}"), "{t}");
    assert!(t.contains(r#"BIN=${BIN:-/srv/rexl3/bin/rexl3-serve}"#), "{t}");
}

#[test]
fn a_fleet_needs_a_group_for_up_and_checks_all_by_default() {
    let f = example("cluster.dp-fleet.json");
    let o = run(&["up", "--cluster", f.to_str().unwrap(), "--dry-run"]);
    assert_eq!(o.status.code(), Some(2), "{}", text(&o));
    assert!(text(&o).contains("name one with --group"), "{}", text(&o));
    let o = run(&["up", "--cluster", f.to_str().unwrap(), "--group", "g1", "--dry-run", "--skip-check"]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o));
    let t = text(&o);
    assert!(t.contains("node A3: start rank 0") && t.contains("node A4: start rank 1") && !t.contains("node A1"), "{t}");
}

/// The sc_4.10bpw image kits cap rank 0's image-preparation credit at one image's charge, 1.125 GiB (D-G3 as amended by the
/// proposed D-G6), through `leader_flags`:
/// the flag reaches rank 0's command line and never the follower's (which serves no HTTP).
#[test]
fn the_sc410_image_kits_cap_the_image_preparation_credit_on_rank_0_only() {
    for f in ["cluster.glm-sc410-tp2-vision.json", "cluster.glm-sc410-tp2-vision-mtp.json"] {
        let o = run(&["up", "--cluster", example(f).to_str().unwrap(), "--group", "g0", "--dry-run", "--skip-check"]);
        assert_eq!(o.status.code(), Some(0), "{f}: {}", text(&o));
        let t = text(&o);
        let section = |what: &str| -> String {
            let i = t.find(what).unwrap_or_else(|| panic!("{f}: no `{what}` in {t}"));
            let rest = &t[i..];
            rest[..rest.find("### end").unwrap()].to_string()
        };
        let (r0, r1) = (section("node N3: start rank 0"), section("node N2: start rank 1"));
        assert!(r0.contains("--image-preparation-bytes") && r0.contains("1207959552"), "{f}: rank 0 {r0}");
        assert!(!r1.contains("--image-preparation-bytes"), "{f}: rank 1 {r1}");
    }
}

#[test]
fn usage_errors_exit_2() {
    assert_eq!(run(&[]).status.code(), Some(2));
    assert_eq!(run(&["frobnicate", "--cluster", "x"]).status.code(), Some(2));
    assert_eq!(run(&["validate"]).status.code(), Some(2));
}
