//! The bound on Flash-Next's captured CUDA graphs (`gb_model::families::qwen_flash_next::graph_cache`), CPU only: the cache never
//! destroys a graph before the owner's wait for the stream, a cache with room waits for nothing, a failed wait destroys nothing, and
//! a server-like sequence of window shapes (a few hot ones among many cold ones) keeps the cache at its cap with the hot shapes
//! resident. The device side (an evicted shape captured again replays to the same bits) is the real-pack test
//! `real_pack_evicted_graphs_are_recaptured_to_the_same_bits` in `model.rs`.
use gb_model::families::qwen_flash_next::graph_cache::{GraphLru, WINDOW_GRAPH_CAP};
use std::cell::RefCell;
use std::rc::Rc;

/// A graph value that records when it is dropped, so a test can see the order of the owner's wait and the destruction.
struct Recorded(u32, Rc<RefCell<Vec<String>>>);
impl Drop for Recorded {
    fn drop(&mut self) {
        self.1.borrow_mut().push(format!("drop {}", self.0));
    }
}

/// The hazard the bound must not introduce: a victim's last replay may still be executing, so the wait comes first and every
/// graph is destroyed after it, never before. A cache with room waits for nothing; a failed wait destroys nothing.
#[test]
fn make_room_drains_the_stream_before_any_graph_is_dropped() {
    let log = Rc::new(RefCell::new(Vec::<String>::new()));
    let mut c: GraphLru<u32, Recorded> = GraphLru::new(3);
    for k in 0..3 {
        let drained = c.make_room(|| -> Result<(), ()> {
            log.borrow_mut().push("sync".into());
            Ok(())
        });
        assert_eq!(drained, Ok(0), "a cache with room drops nothing");
        c.insert(k, Recorded(k, log.clone()));
    }
    assert!(log.borrow().is_empty(), "no wait and no drop while the cache has room");

    // a failed wait leaves every resident in place
    assert_eq!(c.make_room(|| Err("stream fault")), Err("stream fault"));
    assert_eq!((c.len(), c.stats.evictions), (3, 0));
    assert!(log.borrow().is_empty(), "nothing is dropped after a failed wait");

    // one wait, then the victim (the least recently used: key 1 once key 0 has replayed)
    assert!(c.get(&0).is_some());
    let dropped = c.make_room(|| -> Result<(), ()> {
        log.borrow_mut().push("sync".into());
        Ok(())
    });
    assert_eq!(dropped, Ok(1));
    assert_eq!(*log.borrow(), vec!["sync".to_string(), "drop 1".to_string()]);
    assert_eq!((c.len(), c.stats.evictions), (2, 1));
    c.insert(3, Recorded(3, log.clone()));

    // a shrunk cap drains in one wait, every drop after it
    log.borrow_mut().clear();
    c.set_cap(1);
    let dropped = c.make_room(|| -> Result<(), ()> {
        log.borrow_mut().push("sync".into());
        Ok(())
    });
    assert_eq!(dropped, Ok(3));
    let seen = log.borrow().clone();
    assert_eq!(seen.first().map(String::as_str), Some("sync"));
    assert_eq!(seen.iter().filter(|e| *e == "sync").count(), 1, "one wait covers every victim: {seen:?}");
    assert_eq!(&seen[1..], ["drop 2", "drop 0", "drop 3"], "least recently used first: {seen:?}");
    assert!(c.is_empty() && c.stats.evictions == 4);
}

/// The owner's sequence on a running server: every lookup that misses makes room (one wait when full) and inserts. Two hot window
/// shapes recur between thousands of cold ones; the cache stays at its cap, every capture past the cap evicts exactly one graph,
/// every eviction follows a wait, and the hot shapes are never the victim.
#[test]
fn a_server_like_shape_sequence_stays_at_the_cap_and_waits_before_every_eviction() {
    let cap = WINDOW_GRAPH_CAP;
    let mut c: GraphLru<u64, u64> = GraphLru::new(cap);
    let (mut waits, mut captures) = (0u64, 0u64);
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for i in 0..20_000u64 {
        let key = if i % 3 != 2 {
            i % 2
        } else {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1_000 + x % 50_000
        };
        if !c.contains(&key) {
            let before = c.stats.evictions;
            let dropped = c
                .make_room(|| -> Result<(), ()> {
                    waits += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(c.stats.evictions - before, dropped);
            c.insert(key, key);
            captures += 1;
        }
        assert!(c.get(&key).is_some());
        assert!(c.len() <= cap);
    }
    assert_eq!(c.len(), cap);
    assert!(c.contains(&0) && c.contains(&1), "a hot shape was evicted by cold ones");
    assert_eq!(c.stats.evictions, captures - cap as u64, "every capture past the cap evicts exactly one graph");
    assert_eq!(waits, c.stats.evictions, "one wait per insert into a full cache, and only then");
}
