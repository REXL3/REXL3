//! `--dry-run` goldens: the exact commands of a two-node and a four-node group, and the shipped example files parse.
//! Regenerate with `UPDATE_GOLDEN=1 cargo test -p gb-cluster --test dry_run`.

use gb_cluster::check;
use gb_cluster::ops;
use gb_cluster::schema::Cluster;

const TWO: &str = include_str!("../../../docs/examples/cluster.crs812-2node.json");
const FOUR: &str = include_str!("../../../docs/examples/cluster.4node.json");
const FLEET: &str = include_str!("../../../docs/examples/cluster.dp-fleet.json");
const SMOKE: &str = include_str!("../../../docs/examples/cluster.smoke-2node.json");
const NEMO: &str = include_str!("../../../docs/examples/cluster.nemotron-tp2.json");
const NEMO2X: &str = include_str!("../../../docs/examples/cluster.nemotron-2x-tp2.json");

fn golden(name: &str, got: &str) {
    let path = format!("{}/tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        std::fs::write(&path, got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e} (UPDATE_GOLDEN=1 writes it)"));
    assert_eq!(got, want, "dry-run output changed: review it, then UPDATE_GOLDEN=1 to accept ({path})");
}

#[test]
fn the_two_node_up_dry_run_preserves_the_launch_shape() {
    let c = Cluster::parse(TWO).unwrap();
    let g = c.select_group(None).unwrap();
    let out = ops::up_dry_run(&c, g, None, false);
    // portable command shape (docs/NEMOTRON_TP.md 5.7, 12.4), not original host metadata: rank 0 serves HTTP, both ranks get --tp-peer of rank 0, NCCL_MAX_NCHANNELS=4
    assert!(out.contains("--tp-ranks 2 --tp-rank 0 --tp-peer 192.0.2.1:29600"), "{out}");
    assert!(out.contains("--tp-ranks 2 --tp-rank 1 --tp-peer 192.0.2.1:29600"), "{out}");
    assert!(out.contains("export NCCL_MAX_NCHANNELS=4"), "{out}");
    assert!(out.contains("--host 0.0.0.0 --port 8941"), "{out}");
    // N1 before N4, and the disabled nodes never appear in a session
    let n1 = out.find("node N1: start rank 0").unwrap();
    let n4 = out.find("node N4: start rank 1").unwrap();
    assert!(n1 < n4);
    assert!(!out.contains("node N2") && !out.contains("node N3") && !out.contains("192.0.2.2") && !out.contains("192.0.2.3"));
    golden("up_2node.txt", &out);
}

#[test]
fn the_two_node_check_dry_run_is_stable() {
    let c = Cluster::parse(TWO).unwrap();
    let g = c.select_group(None).unwrap();
    golden("check_2node.txt", &check::dry_run(&c, g));
}

#[test]
fn the_four_node_up_dry_run_lists_four_sessions_in_lock_order() {
    let c = Cluster::parse(FOUR).unwrap();
    let g = c.select_group(None).unwrap();
    let out = ops::up_dry_run(&c, g, Some(600), false);
    for r in 0..4 {
        assert!(out.contains(&format!("--tp-ranks 4 --tp-rank {r} --tp-peer 203.0.113.1:29600")), "rank {r}: {out}");
    }
    let pos: Vec<usize> = (1..=4).map(|i| out.find(&format!("node A{i}: start rank {}", i - 1)).unwrap()).collect();
    assert!(pos.windows(2).all(|w| w[0] < w[1]), "the locks are taken A1, A2, A3, A4: {pos:?}");
    assert!(!out.contains("A5"), "the disabled spare is never touched");
    // the leader's flags only on rank 0; the NCCL pins and the shared env on every rank
    assert_eq!(out.matches("--served-model-name big").count(), 1);
    assert_eq!(out.matches("export NCCL_PROTO=Simple").count(), 4, "one in each rank's run script");
    assert_eq!(out.matches("export REXL3_NEMO_GRAPHS=on").count(), 4);
    // a bounded session: rank 0 stops at 600 s, each follower 60 s later so that it sees Shutdown; the watcher polls (no long sleep
    // to orphan) and only rank 0 watches the group's health
    assert!(out.contains("-ge 600 ]; then why=budget") && out.matches("-ge 660 ]; then why=budget").count() == 3, "{out}");
    assert!(!out.contains("sleep 600") && !out.contains("sleep 660"), "no long sleep: {out}");
    assert_eq!(out.matches("why=group_lost").count(), 1, "rank 0 only");
    golden("up_4node.txt", &out);
}

#[test]
fn the_shipped_examples_parse() {
    for (name, text) in [("2node", TWO), ("4node", FOUR), ("fleet", FLEET), ("smoke", SMOKE)] {
        Cluster::parse(text).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
    let c = Cluster::parse(FLEET).unwrap();
    assert_eq!(c.groups.len(), 2);
    assert!(c.select_group(None).is_err(), "two groups need --group");
    assert_eq!(c.select_group(Some("g1")).unwrap().ranks, ["A3", "A4"]);
}

#[test]
fn a_group_of_one_rank_is_a_plain_single_node_server() {
    // the aggregate wide-window mode: one rank, no --tp-* flags, no NCCL environment (docs/CLUSTER_OPERATIONS.md section 1)
    let t = TWO
        .replace(r#""ranks": ["N1", "N4"]"#, r#""ranks": ["N4"]"#)
        .replace(r#""--spec", "auto""#, r#""--spec", "none", "--queue-depth", "512""#)
        .replace(r#""--served-model-name", "m"]"#, r#""--served-model-name", "m"], "env": {"REXL3_NEMO_WINDOW_ROWS": "224"}"#);
    let c = Cluster::parse(&t).unwrap();
    let g = c.select_group(None).unwrap();
    let out = ops::up_dry_run(&c, g, None, false);
    assert!(out.contains("--spec none --queue-depth 512"), "{out}");
    assert!(out.contains("--host 0.0.0.0 --port 8941"), "{out}");
    assert!(out.contains("export REXL3_NEMO_WINDOW_ROWS=224"), "{out}");
    assert!(!out.contains("--tp-"), "no group flags for one rank: {out}");
    assert!(!out.contains("export NCCL_") && !out.contains("nccl.env"), "no NCCL environment for one rank: {out}");
    assert!(out.contains("node N4: start rank 0"), "{out}");
}

#[test]
fn naming_a_disabled_node_as_a_rank_fails_by_name_before_anything_is_run() {
    // the two-node file with rank 1 moved to N2 (a gn100)
    let t = TWO.replace(r#""ranks": ["N1", "N4"]"#, r#""ranks": ["N1", "N2"]"#);
    let e = Cluster::parse(&t).unwrap_err();
    assert!(e.iter().any(|m| m.contains("node N2, which is disabled") && m.contains("out of service")), "{e:?}");
    let t = TWO.replace(r#""ranks": ["N1", "N4"]"#, r#""ranks": ["N3", "N4"]"#);
    let e = Cluster::parse(&t).unwrap_err();
    assert!(e.iter().any(|m| m.contains("node N3, which is disabled")), "{e:?}");
    // no address is refused by a built-in list (the gn100 deny list was lifted with the four-node cluster, 2026-10-16): only `disabled`
    let t = TWO.replace(
        r#"{"name": "N4", "number": 4, "address": "192.0.2.4", "expected_hostname": "host-d"}"#,
        r#"{"name": "N4", "number": 4, "address": "192.0.2.3"}"#,
    );
    Cluster::parse(&t).unwrap_or_else(|e| panic!("{e:?}"));
}

#[test]
fn the_nemotron_tp2_example_is_the_qualified_profile_under_the_supervisor_with_one_library_and_no_home_path() {
    let c = Cluster::parse(NEMO).unwrap();
    let g = c.select_group(None).unwrap();
    // no literal home directory: every path is $HOME-relative on the node
    assert!(!NEMO.contains("/home/"), "a literal home path in the example");
    // the library is the typed nccl.lib, exported to both ranks and expanded on the node
    let out = ops::up_dry_run(&c, g, None, false);
    assert_eq!(out.matches("export REXL3_NCCL_LIB=\"$HOME\"/rexl3/nccl/lib/libnccl.so.2").count(), 2, "{out}");
    for flag in ["--max-model-len 1048576", "--slots 16", "--spec auto", "--tp-ranks 2 --tp-rank 1"] {
        assert!(out.contains(flag), "{flag}: {out}");
    }
    // the restart policy is configured (supervise reads it)
    assert!(NEMO.contains("\"supervisor\"") && NEMO.contains("rexl3-cluster supervise"));
}

#[test]
fn the_four_node_example_is_two_tp2_groups_of_the_qualified_profile_on_disjoint_nodes() {
    let c = Cluster::parse(NEMO2X).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(!NEMO2X.contains("/home/"), "a literal home path in the example");
    let names: Vec<_> = c.groups.iter().map(|g| g.name.clone()).collect();
    assert_eq!(names, ["g0", "g1"]);
    let g0 = c.group("g0").unwrap();
    let g1 = c.group("g1").unwrap();
    let ranks = |g| c.lock_order(g).into_iter().map(|(r, n)| (r, n.name.clone())).collect::<Vec<_>>();
    assert_eq!(ranks(g0), [(0, "N1".to_string()), (1, "N4".to_string())]);
    assert_eq!(ranks(g1), [(0, "N2".to_string()), (1, "N3".to_string())]);
    // each group is the qualified TP=2 profile: the same flags as the one-group example
    for g in [g0, g1] {
        let out = ops::up_dry_run(&c, g, None, false);
        for flag in ["--max-model-len 1048576", "--slots 16", "--spec auto", "--tp-ranks 2 --tp-rank 1"] {
            assert!(out.contains(flag), "{}: {flag}: {out}", g.name);
        }
    }
}
