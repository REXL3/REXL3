//! `/metrics` text and the `/health` object for a tensor-parallel group (`gb_model::tp_status`): rank 0's view of each follower (state,
//! last ack, lag in commands, last frame age, ping round trip, acks, divergences) and of the group (commands sent, settled, the loss).
//!
//! Only rank 0 serves HTTP, so this is the group's whole picture: a follower's own stderr health line is not repeated here. A process that
//! does not lead a group has no status and emits nothing, so a single-node server's `/metrics` and `/health` are byte-for-byte what
//! they were.

use gb_model::tp_status::{GroupStatus, RankStatus};
use serde_json::{json, Value};
use std::fmt::Write;

fn family(out: &mut String, name: &str, kind: &str, help: &str, samples: Vec<(String, String)>) {
    if samples.is_empty() {
        return;
    }
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
    for (labels, v) in samples {
        let _ = writeln!(out, "{name}{labels} {v}");
    }
}

fn per_rank(g: &GroupStatus, f: impl Fn(&RankStatus) -> Option<String>) -> Vec<(String, String)> {
    g.per_rank.iter().filter_map(|r| f(r).map(|v| (format!("{{rank=\"{}\"}}", r.rank), v))).collect()
}

/// Prometheus text for the group: `rexl3_tp_*`. `groups_lost` is the process's count of lost groups
/// ([`gb_model::tp_status::groups_lost`]).
pub fn render(g: &GroupStatus, groups_lost: u64) -> String {
    let mut out = String::new();
    let one = |v: String| vec![(String::new(), v)];
    family(&mut out, "rexl3_tp_ranks", "gauge", "Ranks in the tensor-parallel group, rank 0 included.", one(g.ranks.to_string()));
    family(
        &mut out,
        "rexl3_tp_group_up",
        "gauge",
        "1 while the group is whole (no rank has been lost), 0 once it is lost.",
        one(u8::from(g.lost.is_none()).to_string()),
    );
    family(
        &mut out,
        "rexl3_tp_group_lost_total",
        "counter",
        "Tensor-parallel groups this process has lost (a lost group is fatal: the process exits and is restarted whole).",
        one(groups_lost.to_string()),
    );
    family(
        &mut out,
        "rexl3_tp_commands_sent_total",
        "counter",
        "Commands rank 0 has sent to every follower (the lockstep sequence number).",
        one(g.commands_sent.to_string()),
    );
    family(&mut out, "rexl3_tp_acks_total", "counter", "Acknowledgements received from all followers.", one(g.acks.to_string()));
    family(
        &mut out,
        "rexl3_tp_settled_seq",
        "gauge",
        "The highest command every follower has acknowledged and agreed on.",
        one(g.settled.to_string()),
    );
    family(
        &mut out,
        "rexl3_tp_rank_up",
        "gauge",
        "1 while the follower's link is open, 0 once it has ended.",
        per_rank(g, |r| Some(u8::from(r.connected).to_string())),
    );
    family(
        &mut out,
        "rexl3_tp_rank_state",
        "gauge",
        "One series per follower and state (up, lost, closed), 1 for the current one.",
        g.per_rank
            .iter()
            .flat_map(|r| {
                ["up", "lost", "closed"]
                    .into_iter()
                    .map(move |s| (format!("{{rank=\"{}\",state=\"{s}\"}}", r.rank), u8::from(r.state == s).to_string()))
            })
            .collect(),
    );
    family(
        &mut out,
        "rexl3_tp_rank_acked_seq",
        "gauge",
        "The highest command the follower has acknowledged.",
        per_rank(g, |r| Some(r.acked_seq.to_string())),
    );
    family(
        &mut out,
        "rexl3_tp_rank_lag_commands",
        "gauge",
        "Commands sent to the follower and not yet acknowledged (a straggler shows here).",
        per_rank(g, |r| Some(r.lag_commands.to_string())),
    );
    family(
        &mut out,
        "rexl3_tp_rank_last_frame_age_seconds",
        "gauge",
        "Seconds since the last frame (ack or pong) from the follower; the peer timeout is the limit.",
        per_rank(g, |r| Some(format!("{:.3}", r.last_frame_age_ms as f64 / 1e3))),
    );
    family(
        &mut out,
        "rexl3_tp_rank_ping_rtt_seconds",
        "gauge",
        "The last ping's round trip to the follower, once one has completed.",
        per_rank(g, |r| r.ping_rtt_us.map(|us| format!("{:.6}", us as f64 / 1e6))),
    );
    family(
        &mut out,
        "rexl3_tp_rank_acks_total",
        "counter",
        "Acknowledgements received from the follower.",
        per_rank(g, |r| Some(r.acks.to_string())),
    );
    family(
        &mut out,
        "rexl3_tp_rank_diverged_total",
        "counter",
        "Acknowledgements whose result differed from rank 0's (the group is lost at the first).",
        per_rank(g, |r| Some(r.diverged.to_string())),
    );
    family(
        &mut out,
        "rexl3_tp_rank_queued_frames",
        "gauge",
        "Frames queued for the follower and not yet written.",
        per_rank(g, |r| Some(r.queued_frames.to_string())),
    );
    family(
        &mut out,
        "rexl3_tp_rank_sent_bytes_total",
        "counter",
        "Bytes written to the follower (pings excluded).",
        per_rank(g, |r| Some(r.bytes_sent.to_string())),
    );
    out
}

/// The `tensor_parallel` object of `/health`.
pub fn health(g: &GroupStatus) -> Value {
    json!({
        "ranks": g.ranks,
        "state": if g.lost.is_some() { "lost" } else { "up" },
        "lost": g.lost,
        "lost_rank": g.lost_rank,
        "commands_sent": g.commands_sent,
        "settled": g.settled,
        "per_rank": g.per_rank.iter().map(|r| json!({
            "rank": r.rank,
            "state": r.state,
            "connected": r.connected,
            "acked_seq": r.acked_seq,
            "applied_seq": r.applied_seq,
            "lag_commands": r.lag_commands,
            "last_frame_age_ms": r.last_frame_age_ms,
            "ping_rtt_us": r.ping_rtt_us,
            "acks": r.acks,
            "diverged": r.diverged,
            "queued_frames": r.queued_frames,
            "left": r.left,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status() -> GroupStatus {
        let rank = |rank, connected: bool, state, acked, lag| RankStatus {
            rank,
            connected,
            state,
            acked_seq: acked,
            applied_seq: acked,
            lag_commands: lag,
            last_frame_age_ms: 12,
            ping_rtt_us: (rank == 1).then_some(250),
            acks: acked,
            diverged: 0,
            queued_frames: 0,
            bytes_sent: 100 * rank as u64,
            left: (!connected).then(|| "its stream ended: connection closed by the peer".to_string()),
        };
        GroupStatus {
            ranks: 3,
            commands_sent: 40,
            settled: 38,
            acks: 78,
            tolerated: 0,
            lost: None,
            lost_rank: None,
            per_rank: vec![rank(1, true, "up", 40, 0), rank(2, true, "up", 38, 2)],
        }
    }

    #[test]
    fn the_group_renders_per_rank_state_lag_and_the_loss() {
        let g = status();
        let text = render(&g, 0);
        for needle in [
            "rexl3_tp_ranks 3",
            "rexl3_tp_group_up 1",
            "rexl3_tp_group_lost_total 0",
            "rexl3_tp_commands_sent_total 40",
            "rexl3_tp_acks_total 78",
            "rexl3_tp_rank_up{rank=\"1\"} 1",
            "rexl3_tp_rank_lag_commands{rank=\"2\"} 2",
            "rexl3_tp_rank_state{rank=\"2\",state=\"up\"} 1",
            "rexl3_tp_rank_state{rank=\"2\",state=\"lost\"} 0",
            "rexl3_tp_rank_ping_rtt_seconds{rank=\"1\"} 0.000250",
        ] {
            assert!(text.contains(needle), "{needle} missing from:\n{text}");
        }
        // no ping answered yet for rank 2: no series, not a zero
        assert!(!text.contains("rexl3_tp_rank_ping_rtt_seconds{rank=\"2\"}"), "{text}");
        // a lost group
        let mut lost = status();
        lost.lost = Some("the connection to rank 2 failed: connection closed by the peer".into());
        lost.lost_rank = Some(2);
        lost.per_rank[1].connected = false;
        lost.per_rank[1].state = "lost";
        lost.per_rank[0].connected = false;
        lost.per_rank[0].state = "closed";
        let text = render(&lost, 1);
        assert!(text.contains("rexl3_tp_group_up 0") && text.contains("rexl3_tp_group_lost_total 1"), "{text}");
        assert!(text.contains("rexl3_tp_rank_state{rank=\"2\",state=\"lost\"} 1"), "{text}");
        let h = health(&lost);
        assert_eq!(
            (h["state"].as_str(), h["lost_rank"].as_u64(), h["per_rank"][1]["state"].as_str()),
            (Some("lost"), Some(2), Some("lost"))
        );
        // every sample line is `name{labels} value` or `name value`
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            assert_eq!(line.split(' ').count(), 2, "{line}");
        }
    }
}
