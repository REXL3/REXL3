//! The DSpark draft TREES (T2.5a, the DFlash2 SPEC_TREE playbook port): the round shape, the every-node-verified
//! invariant of its verify plan, the longest-verified-prefix accept, and byte-identity of the committed stream against
//! the serial computation on a synthetic round — plus (lane 2) the TRUNK EXECUTION: the path windows, the commit's
//! state surgery and its state-equality gates.
//!
//! The machinery under test is the family's `tree` module, which REUSES the Qwen engine's exact tree pieces by import
//! (`gb_spec::tree::{Lattice, DraftTree, best_first, accept_tree, Policy}` — the search, the tree and the sample walk
//! are the engine's, tuned constants and all) and adds only what is DSpark's: the markov lattice (the per-level top-k
//! of the head's logits with the `markov_head` edge terms), the confidence head's scheduling of the fanout, and (lane
//! 2) the expansion of a tree request into one serial-exact chain window per root-to-leaf path with the commit's
//! save/restore surgery over the four persistent surfaces a path window mutates.
//!
//! Everything here is host-side and synthetic: no device, no weights. The serial reference is an independent argmax
//! walk over a deterministic synthetic target, the same function the round's verify plan samples from, so
//! byte-identity is proved at the construction level the playbook claims it by.

use gb_model::families::deepseek_v41::tree::{accept_round, round, schedule_branch, TreeRound};
use gb_spec::tree::{best_first, DraftTree, Lattice, Policy};

// ------------------------------------------------------------------------------------------------------------------------ lane 2
// The trunk executes a tree round as one serial-exact chain window per root-to-leaf path
// (`tree::verify_tree` over `tree::TreeRoundHost`), and the commit reconciles the four persistent surfaces
// a path window mutates — the planes' window ring, the engram ring (its gather appends per row, so sibling
// paths contaminate each other's n-gram context), the compressor's carry and the DSpark tap — to the serial
// replay of the accepted path. The device implementation of the host trait is `Dsv41Model`'s; these tests
// drive the SAME driver (the planning, the per-path isolation, the commit surgery order) over an in-memory
// twin of the four surfaces, comparing STATE (bytes), not just tokens.

mod lane2 {
    use gb_engine::RowOutput;
    use gb_model::families::deepseek_v41::tree::{commit_round, verify_tree, TreeRoundHost, TreeWindows};
    use gb_spec::tree::accept_tree;

    /// The family's window: 8 rows (the anchor plus at most 7 nodes).
    const WINDOW_ROWS: usize = 8;
    /// The twin's sliding-window ring: small enough that a 31-token prefix wraps it (the real ring is
    /// `cfg.window` = 128; the surgery is slot arithmetic either way).
    const WIN: usize = 16;
    /// The twin's n-gram ring capacity (the real one is `engram::RING` = 128).
    const GRAM: usize = 8;

    /// The synthetic target's score, salted and mixed: `mix` carries the state the real forward READS (the
    /// n-gram rows of the context that just formed, the compressor's carried row at an odd boundary), so a
    /// contaminated surface moves a PICK, not just bytes — without the mix, isolation failures would be
    /// invisible in the outputs.
    fn score_m(ctx: &[u32], tok: u32, mix: u64, salt: u64) -> f32 {
        let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(tok) ^ mix.rotate_left(17) ^ salt.rotate_left(31);
        for &t in ctx {
            h ^= u64::from(t);
            h = h.wrapping_mul(0x1000_0019_3E4F_2B7D);
        }
        (h >> 33) as f32
    }

    /// The in-memory twin of the four persistent surfaces a path window mutates. `ring[slot]` stands in for
    /// a layer's `planes.window` row (the KV bytes a row writes at its position's slot), `(gram,
    /// gram_filled)` for the engram ring with the real `Slot` semantics (push at `filled % RING`, rewind
    /// only decrements), `carry` for the compressor's carried row and `taps` for the DSpark tap rows a
    /// window leaves behind.
    #[derive(Clone)]
    struct Twin {
        salt: u64,
        ring: Vec<u64>,
        gram: Vec<u32>,
        gram_filled: usize,
        carry: u64,
        taps: Vec<u64>,
        /// The committed prefix (the context every score reads); grown by the caller between rounds.
        prefix: Vec<u32>,
    }

    fn kv_of(ctx: &[u32], mix: u64) -> u64 {
        let mut h = 0xC0FF_EE01_D0D0_FACEu64 ^ mix;
        for &t in ctx {
            h ^= u64::from(t);
            h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        h
    }

    impl Twin {
        /// The last n-grams' signature — what the real engram gather reads back (the look-back rows of the
        /// context that just formed, this row's own token included).
        fn gram_sig(&self) -> u64 {
            let have = self.gram_filled.min(3);
            (0..have).map(|i| u64::from(self.gram[(self.gram_filled + GRAM - have + i) % GRAM]) << (i * 11)).sum()
        }

        /// One chain window of `tokens` at position `p0`: the synthetic target's greedy pick per row, plus
        /// the surface mutations the device forward performs — the row's KV into its position's ring slot,
        /// the token appended to the n-gram ring, the tap rows left behind, and (an odd end leaves the last
        /// row unpaired) the carry. An odd START folds the carried row into the first row's mix: the
        /// compressor's cached-row read, the surface whose contamination changes a path's own OUTPUT.
        fn forward(&mut self, tokens: &[u32], p0: usize, universe: &[u32]) -> Vec<u32> {
            let mut outs = Vec::with_capacity(tokens.len());
            let mut kvs = Vec::with_capacity(tokens.len());
            for (j, &t) in tokens.iter().enumerate() {
                let mut ctx = self.prefix.clone();
                ctx.extend_from_slice(&tokens[..j + 1]);
                self.gram[self.gram_filled % GRAM] = t;
                self.gram_filled += 1;
                let mix = self.gram_sig() ^ if p0 % 2 == 1 && j == 0 { self.carry } else { 0 };
                let kv = kv_of(&ctx, mix);
                self.ring[(p0 + j) % WIN] = kv;
                kvs.push(kv);
                outs.push(
                    universe
                        .iter()
                        .copied()
                        .max_by(|a, b| score_m(&ctx, *a, mix, self.salt).total_cmp(&score_m(&ctx, *b, mix, self.salt)))
                        .unwrap_or(t),
                );
            }
            if (p0 + tokens.len()) % 2 == 1 {
                self.carry = *kvs.last().unwrap();
            }
            self.taps = kvs;
            outs
        }
    }

    struct TwinBase {
        /// The ring slots of the round's touched positions (`WINDOW_ROWS` of them, position order).
        ring_region: Vec<u64>,
        gram: Vec<u32>,
        gram_filled: usize,
        carry: u64,
    }

    struct PathSave {
        /// The window's per-row KV (what the device saves of `window_bak`'s tail — and of the compressor's
        /// staged rows, which are the same rows).
        rows: Vec<u64>,
        taps: Vec<u64>,
    }

    /// `TreeRoundHost` over the twin: the same protocol the device host runs — snapshot the base surfaces
    /// on the round's first window, isolate every window to that base (the n-gram ring and the carry
    /// restored before it runs), save the window's rows/taps/carry-after for the commit, and reconcile at
    /// the commit (base ring slots restored, the accepted path's rows rewritten, the n-gram ring replayed,
    /// the carry and taps taken from the accepted path's own window).
    struct TwinHost {
        st: Twin,
        universe: Vec<u32>,
        base: Option<TwinBase>,
        saves: Vec<PathSave>,
        p0: usize,
    }

    impl TreeRoundHost for TwinHost {
        fn tree_path_window(
            &mut self,
            tokens: &[u32],
            first_position: u64,
            _candidates: usize,
        ) -> Result<Vec<RowOutput>, String> {
            let p0 = first_position as usize;
            if self.base.is_none() {
                let region = (0..WINDOW_ROWS).map(|j| self.st.ring[(p0 + j) % WIN]).collect();
                self.base = Some(TwinBase {
                    ring_region: region,
                    gram: self.st.gram.clone(),
                    gram_filled: self.st.gram_filled,
                    carry: self.st.carry,
                });
                self.p0 = p0;
            } else if p0 != self.p0 {
                return Err(format!("a path window at {p0} does not match the round's anchor at {}", self.p0));
            }
            let base = self.base.as_ref().unwrap();
            // the per-path isolation: a sibling's n-gram appends and carry must not leak into this window
            self.st.gram = base.gram.clone();
            self.st.gram_filled = base.gram_filled;
            self.st.carry = base.carry;
            let outs = self.st.forward(tokens, p0, &self.universe);
            self.saves.push(PathSave { rows: self.st.taps.clone(), taps: self.st.taps.clone() });
            Ok(outs.into_iter().map(|id| RowOutput { ids: vec![id], values: vec![1.0] }).collect())
        }

        fn tree_commit_window(&mut self, idx: usize, rows: usize, window: &[u32]) -> Result<(), String> {
            let Some(base) = self.base.take() else { return Err("no tree round is open".into()) };
            let sv = self.saves.get(idx).ok_or_else(|| format!("path window {idx} never ran"))?;
            if rows == 0 || rows > sv.rows.len() || rows > window.len() {
                return Err(format!("a commit of {rows} rows over a {}-row window", sv.rows.len()));
            }
            for j in 0..WINDOW_ROWS {
                self.st.ring[(self.p0 + j) % WIN] = base.ring_region[j];
            }
            for j in 0..rows {
                self.st.ring[(self.p0 + j) % WIN] = sv.rows[j];
            }
            self.st.gram = base.gram;
            self.st.gram_filled = base.gram_filled;
            for &t in &window[..rows] {
                self.st.gram[self.st.gram_filled % GRAM] = t;
                self.st.gram_filled += 1;
            }
            // the carry is the SERIAL replay's: an accepted prefix that ends mid-pair carries its own last
            // row (the compressor's parity restage from the window's staged rows — `commit_window_planes`'s
            // move); one that ends on a pair boundary leaves the base carry untouched
            if (self.p0 + rows) % 2 == 1 {
                self.st.carry = sv.rows[rows - 1];
            } else {
                self.st.carry = base.carry;
            }
            self.st.taps = sv.taps[..rows].to_vec();
            Ok(())
        }

        fn tree_drop_round(&mut self) -> Result<(), String> {
            if let Some(base) = self.base.take() {
                for j in 0..WINDOW_ROWS {
                    self.st.ring[(self.p0 + j) % WIN] = base.ring_region[j];
                }
                self.st.gram = base.gram;
                self.st.gram_filled = base.gram_filled;
                self.st.carry = base.carry;
            }
            Ok(())
        }
    }

    /// The fixture's fork: `anchor -> {A, B}`, `A -> A2`, `B -> B2` (rows `[anchor, A, B, A2, B2]`). Two
    /// root-to-leaf paths, `[A, A2]` and `[B, B2]`, with B's window running LAST — the sibling whose writes
    /// cover the accepted rows' ring slots and whose state would contaminate A's window without the surgery.
    fn fork() -> (Vec<u32>, Vec<i32>) {
        (vec![777, 100, 200, 110, 210], vec![-1, 0, 0, 1, 2])
    }

    /// A twin with a 31-token committed prefix: the round's anchor sits at the odd position 31, so the
    /// compressor's cached-row read fires AND the touched ring region wraps (slots 15, 0, 1, ...).
    fn twin_at_fork(universe: &[u32], salt: u64) -> TwinHost {
        let mut st = Twin {
            salt,
            ring: (0..WIN).map(|i| 0xAAA0_0000_0000_0000u64 + i as u64).collect(),
            gram: vec![0; GRAM],
            gram_filled: 0,
            carry: 0x1234_5678_9ABC_DEF0,
            taps: Vec::new(),
            prefix: Vec::new(),
        };
        let prefix: Vec<u32> = (0..31).map(|i| 500 + (i * 7) % 13).collect();
        st.forward(&prefix, 0, universe);
        TwinHost { st, universe: universe.to_vec(), base: None, saves: Vec::new(), p0: 31 }
    }

    /// The round's anchor position (odd, past the twin's ring so the touched region wraps).
    const P0: u64 = 31;

    /// The trunk expands a tree request into one chain window per root-to-leaf path: every row reached,
    /// each row's verified sample read from its designated window (the first that reaches it), the anchor
    /// riding row 0 of them all, every window within the family's row cap, and a parent that has not run
    /// yet (or a round at position 0, which no decode round can produce) refused.
    #[test]
    fn the_trunk_expands_a_tree_request_into_path_windows() {
        let (tokens, parents) = fork();
        let w = TreeWindows::plan(&tokens, &parents, P0, WINDOW_ROWS).expect("the fork plans");
        assert_eq!(w.paths.len(), 2, "the fork has two leaves");
        // root-first, each row's parent the row before it (the anchor aside)
        assert_eq!(w.paths[0].rows, vec![1, 3], "the A path");
        assert_eq!(w.paths[1].rows, vec![2, 4], "the B path");
        assert_eq!(w.window_tokens(0), vec![777, 100, 110]);
        assert_eq!(w.window_tokens(1), vec![777, 200, 210]);
        // every row has a designated window, the anchor's row 0 of the first
        assert_eq!(w.designated(0), (0, 0));
        assert_eq!(w.designated(1), (0, 1));
        assert_eq!(w.designated(3), (0, 2));
        assert_eq!(w.designated(2), (1, 1));
        assert_eq!(w.designated(4), (1, 2));
        // the caps and the refusals
        assert!(
            TreeWindows::plan(&tokens, &parents, 0, WINDOW_ROWS).is_err(),
            "a round at position 0 is not a decode round (the anchor is a committed prefix's tail)"
        );
        assert!(TreeWindows::plan(&tokens, &[-1, 2, 1, 0, 0], P0, WINDOW_ROWS).is_err(), "a parent after its child");
        assert!(TreeWindows::plan(&tokens, &[-1, 0, 0, 1], P0, WINDOW_ROWS).is_err(), "the request and the parents disagree");
        let deep = (0..=WINDOW_ROWS).map(|i| i as u32 * 3).collect::<Vec<u32>>();
        let chain = (-1..WINDOW_ROWS as i32).collect::<Vec<i32>>();
        assert!(TreeWindows::plan(&deep, &chain, P0, WINDOW_ROWS).is_err(), "a path window wider than the family's window");
        assert!(TreeWindows::plan(&tokens, &parents, 32, WINDOW_ROWS).is_ok(), "an even anchor position is fine");
    }

    /// THE state-equality gate: a two-path round commits exactly the longest verified path's tokens AND
    /// every surface (ring slots, n-gram ring, carry, taps) ends byte-equal to a serial replay of that
    /// path — not just the tokens.
    #[test]
    fn a_two_path_round_commits_the_serial_state_byte_for_byte() {
        let universe = vec![777, 100, 110, 200, 210, 999_999];
        let (tokens, parents) = fork();
        let (mut followed_a_branch, mut left_the_tree) = (0, 0);
        for salt in 0..24u64 {
            let mut host = twin_at_fork(&universe, salt);
            let w = TreeWindows::plan(&tokens, &parents, P0, WINDOW_ROWS).expect("plan");
            let before = host.st.clone();
            let rows = verify_tree(&mut host, &w, 1).expect("verify");
            // the samples are the twin's greedy picks; the accept walk is the engine's
            let samples: Vec<u32> = rows.iter().map(|r| r.ids[0]).collect();
            let path = accept_tree(&tokens, &parents, &samples);
            let committed: Vec<u32> = path.iter().map(|&r| samples[r]).collect();
            commit_round(&mut host, &w, &path).expect("commit");
            // the serial replay: the same base state, the accepted path's window run alone
            let mut serial = before;
            let accepted_window: Vec<u32> = path.iter().map(|&r| tokens[r]).collect();
            let serial_outs = serial.forward(&accepted_window, P0 as usize, &universe);
            assert_eq!(committed, serial_outs[..path.len()], "salt {salt}: the committed tokens are not the serial stream");
            // STATE equality: every surface, not the tokens
            assert_eq!(host.st.ring, serial.ring, "salt {salt}: the window ring is not the serial replay's");
            assert_eq!(
                (host.st.gram.clone(), host.st.gram_filled),
                (serial.gram.clone(), serial.gram_filled),
                "salt {salt}: the n-gram ring is not the serial replay's"
            );
            assert_eq!(host.st.carry, serial.carry, "salt {salt}: the compressor carry is not the serial replay's");
            assert_eq!(host.st.taps, serial.taps, "salt {salt}: the tap rows are not the serial replay's");
            if path.len() > 2 {
                followed_a_branch += 1;
            }
            if path.len() < tokens.len() {
                left_the_tree += 1;
            }
        }
        assert!(followed_a_branch > 0, "no salt accepted a draft node: the fork never exercised a real round");
        assert!(left_the_tree > 0, "no salt left the tree: the early-exit accept is never exercised");
    }

    /// A REJECTED branch leaves no residue: the sibling's window ran last (its writes covering the accepted
    /// rows' ring slots), and after the commit the slots past the accepted length hold the BASE bytes, the
    /// n-gram ring holds no token of the rejected branch, and the carry and taps are the accepted path's own.
    #[test]
    fn a_rejected_branch_leaves_no_plane_or_engram_residue() {
        let universe = vec![777, 100, 110, 200, 210, 999_999];
        let (tokens, parents) = fork();
        // a salt whose round follows the A branch (B's window ran last and lost)
        let mut salt = 0u64;
        let (mut host, path) = loop {
            let mut h = twin_at_fork(&universe, salt);
            let w = TreeWindows::plan(&tokens, &parents, P0, WINDOW_ROWS).expect("plan");
            let rows = verify_tree(&mut h, &w, 1).expect("verify");
            let s: Vec<u32> = rows.iter().map(|r| r.ids[0]).collect();
            let p = accept_tree(&tokens, &parents, &s);
            if p.contains(&1) || p.contains(&3) {
                break (h, p);
            }
            salt += 1;
            assert!(salt < 64, "no salt followed the A branch");
        };
        let base_state = twin_at_fork(&universe, salt).st;
        let w = TreeWindows::plan(&tokens, &parents, P0, WINDOW_ROWS).expect("plan");
        commit_round(&mut host, &w, &path).expect("commit");
        let accepted: Vec<u32> = path.iter().map(|&r| tokens[r]).collect();
        assert!(accepted.contains(&100), "this salt's round was chosen to follow the A branch");
        // (1) the ring slots past the accepted length are the BASE bytes, whoever wrote them last
        for j in path.len()..WINDOW_ROWS {
            assert_eq!(
                host.st.ring[(P0 as usize + j) % WIN],
                base_state.ring[(P0 as usize + j) % WIN],
                "slot {} still holds a rejected row's bytes",
                (P0 as usize + j) % WIN
            );
        }
        // (2) the accepted rows' slots hold the accepted path's OWN bytes (the serial replay's)
        let mut serial = base_state.clone();
        serial.forward(&accepted, P0 as usize, &universe);
        for j in 0..path.len() {
            assert_eq!(host.st.ring[(P0 as usize + j) % WIN], serial.ring[(P0 as usize + j) % WIN], "accepted slot {j}");
        }
        // (3) the n-gram ring is the base plus the accepted window's tokens — nothing else
        let mut want_gram = base_state.gram.clone();
        let mut want_filled = base_state.gram_filled;
        for &t in &accepted {
            want_gram[want_filled % GRAM] = t;
            want_filled += 1;
        }
        assert_eq!(
            (host.st.gram.clone(), host.st.gram_filled),
            (want_gram, want_filled),
            "the n-gram ring is not the serial replay's"
        );
        let rejected: Vec<u32> = tokens[1..].iter().copied().filter(|t| !accepted.contains(t)).collect();
        let have = host.st.gram_filled.min(GRAM);
        let tail: Vec<u32> = (0..have).map(|i| host.st.gram[(host.st.gram_filled + GRAM - have + i) % GRAM]).collect();
        for &r in &rejected {
            assert!(!tail.contains(&r), "the rejected token {r} is still in the n-gram context");
        }
        // (4) the carry and taps are the accepted path's
        assert_eq!(host.st.carry, serial.carry);
        assert_eq!(host.st.taps, serial.taps);
    }

    /// Per-path ISOLATION: every path window's rows are the serial computation of that path — a sibling's
    /// n-gram appends or carry must not move one pick. This is the invariant that breaks first when the
    /// between-path restore is missing (B's window runs after A's and would read A's context).
    #[test]
    fn every_path_window_is_serial_exact_despite_its_siblings() {
        let universe = vec![777, 100, 110, 200, 210, 999_999];
        let (tokens, parents) = fork();
        for salt in 0..16u64 {
            let mut host = twin_at_fork(&universe, salt);
            let base = host.st.clone();
            let w = TreeWindows::plan(&tokens, &parents, P0, WINDOW_ROWS).expect("plan");
            let rows = verify_tree(&mut host, &w, 1).expect("verify");
            for (row, out) in rows.iter().enumerate() {
                // the serial computation of this row: a fresh twin from the base, the row's own path run alone
                let (p, at) = w.designated(row);
                let mut serial = base.clone();
                let serial_outs = serial.forward(&w.window_tokens(p), P0 as usize, &universe);
                assert_eq!(
                    out.ids[0], serial_outs[at],
                    "salt {salt} row {row}: the window's pick moved (a sibling's state leaked)"
                );
            }
        }
    }

    /// The commit refuses a path that is not a root path of the window's tree (each row's parent the row
    /// before it, the anchor first) and a row the round never verified — the error paths that would
    /// otherwise corrupt the reconciled state.
    #[test]
    fn an_invalid_commit_path_is_refused() {
        let universe = vec![777, 100, 110, 200, 210, 999_999];
        let (tokens, parents) = fork();
        let w = TreeWindows::plan(&tokens, &parents, P0, WINDOW_ROWS).expect("plan");
        let mut host = twin_at_fork(&universe, 0);
        verify_tree(&mut host, &w, 1).expect("verify");
        assert!(commit_round(&mut host, &w, &[]).is_err(), "an empty path");
        assert!(commit_round(&mut host, &w, &[1]).is_err(), "the path must start at the anchor's row 0");
        assert!(commit_round(&mut host, &w, &[0, 4]).is_err(), "row 4's parent is row 2, not the anchor");
        assert!(commit_round(&mut host, &w, &[0, 1, 4]).is_err(), "a skip over row 4's parent");
        assert!(commit_round(&mut host, &w, &[0, 1, 2]).is_err(), "row 2's parent is the anchor, not row 1");
        assert!(commit_round(&mut host, &w, &[0, 9]).is_err(), "a row the round never verified");
        // the two root paths of the fork commit; a second commit is refused (the round is closed)
        assert!(commit_round(&mut host, &w, &[0, 1, 3]).is_ok());
        assert!(commit_round(&mut host, &w, &[0, 1, 3]).is_err(), "the round is closed");
    }
}

// ------------------------------------------------------------------------------------------------------- the lattice

/// A deterministic DSpark-shaped lattice: `rows` levels of `k` candidates with pseudo-random unary logits and
/// markov-style edge terms, the ids distinct within a level (the shape `crates/cuda/src/dsv4_mtp_gpu.rs` captures).
fn lattice(seed: u64, rows: usize, k: usize) -> Lattice {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut ids = Vec::new();
    let mut unary = Vec::new();
    for _ in 0..rows {
        let mut u: Vec<f32> = (0..k).map(|_| (next() * 8.0) as f32).collect();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for c in 0..k {
            ids.push(1000 + (next() * 90_000.0) as u32 * 16 + c as u32); // distinct within a level
        }
        unary.extend(u);
    }
    let edge: Vec<f32> = (0..rows * k * k).map(|_| (next() - 0.5) as f32 * 2.0).collect();
    Lattice { rows, k, ids, unary, edge, restricted_head: false }
}

/// The lattice of a certain draft: every level's first candidate dominates, the edges are flat. The markov walk's
/// picks are then exactly the first candidates, and any tree over it carries them as its spine.
fn certain_lattice(rows: usize, k: usize) -> Lattice {
    let l = lattice(5, rows, k);
    let mut l = l;
    for d in 0..rows {
        for c in 0..l.k {
            l.unary[d * l.k + c] = if c == 0 { 1000.0 } else { 0.0 };
        }
    }
    l.edge.iter_mut().for_each(|e| *e = 0.0);
    l
}

const BLOCK: usize = 5; // dspark_block_size of the pack
const WINDOW_ROWS: usize = 8; // the family's window cap (one anchor row + the nodes)

fn tree_round(l: &Lattice, conf: &[f32]) -> TreeRound {
    round(BLOCK, WINDOW_ROWS, l, conf)
}

// -------------------------------------------------------------------------------------------------------- the tests

/// The round's tree is the Qwen engine's best-first search over the DSpark lattice: parents before children, depth
/// capped at the block, nodes capped at the window's rows, siblings distinct, and every one of the markov walk's
/// picks is a node (the chain the round would have drafted without trees is inside the tree).
#[test]
fn the_tree_round_is_best_first_over_the_markov_lattice() {
    let l = lattice(7, BLOCK, 8);
    let conf = [0.42f32; BLOCK];
    let r = tree_round(&l, &conf);
    assert_eq!(
        r.policy,
        Policy { edge: 0.6, noise: 0.7, scale: 1.5, branch: 4 },
        "the engine's tuned constants, only the fanout scheduled"
    );
    assert!(!r.tree.is_empty() && r.tree.len() < WINDOW_ROWS, "{} nodes", r.tree.len());
    for i in 0..r.tree.len() {
        assert!(r.tree.parents[i] < i as i32, "node {i} sits before its parent");
        assert!(r.tree.depth[i] as usize <= BLOCK, "node {i} is deeper than the block");
        for j in i + 1..r.tree.len() {
            assert!(!(r.tree.parents[i] == r.tree.parents[j] && r.tree.tokens[i] == r.tree.tokens[j]), "duplicate sibling");
        }
    }
    // the search itself is the engine's: the same lattice, policy and budget give the same tree
    assert_eq!(r.tree, best_first(&l, None, &r.policy, WINDOW_ROWS - 1).limit_depth(BLOCK));
}

/// On a certain draft the chain is the tree's spine: the branch-any search pops the walk's picks first (their path
/// probability dwarfs every sibling's), so the tree begins with the markov chain — what the round would have drafted
/// without trees — and only then spends its leftover rows on the best siblings.
#[test]
fn the_walks_chain_is_the_confident_rounds_tree() {
    let l = certain_lattice(BLOCK, 8);
    let walk = gb_spec::tree::greedy_chain(&l);
    // a confident round: the fanout schedule keeps a sure draft narrow
    let r = tree_round(&l, &[0.9; BLOCK]);
    assert_eq!(r.policy.branch, 2);
    assert!(r.tree.len() >= walk.len(), "the tree does not even hold the chain");
    assert_eq!(&r.tree.tokens[..walk.len()], &walk[..], "a certain draft's tree starts as its chain");
    assert_eq!(&r.tree.parents[..walk.len()], &(0..walk.len() as i32).map(|i| i - 1).collect::<Vec<i32>>()[..]);
    // the leftover rows are siblings of the chain's nodes, never deeper than the block
    assert!(r.tree.depth.iter().all(|&d| d as usize <= BLOCK));
}

/// The DSpark fusion: the confidence head schedules the fanout. High confidence keeps the tree narrow (the chain is
/// right); low confidence fans out to the Qwen engine's tuned widest default.
#[test]
fn the_fanout_is_scheduled_by_the_confidence_head() {
    assert_eq!(schedule_branch(&[0.95, 0.9, 0.92]), 2, "a confident round branches 2-wide");
    assert_eq!(schedule_branch(&[0.55, 0.52]), 3, "a middling round branches 3-wide");
    assert_eq!(schedule_branch(&[0.2, 0.3, 0.1]), 4, "a lost draft branches 4-wide (the tuned default)");
    assert_eq!(schedule_branch(&[]), 4, "no confidences read: assume the worst");
    assert_eq!(schedule_branch(&[0.70]), 2, "the boundary is inclusive");
    assert_eq!(schedule_branch(&[0.50]), 3, "the boundary is inclusive");
    // and the schedule actually moves the tree's shape: an uncertain round fans the anchor out to the engine's
    // widest default while a confident round keeps the fork at two (both fill the window's rows; the width differs)
    let l = lattice(11, BLOCK, 8);
    let narrow = tree_round(&l, &[0.9; BLOCK]);
    let wide = tree_round(&l, &[0.2; BLOCK]);
    let anchor_children = |r: &TreeRound| r.tree.parents.iter().filter(|&&p| p < 0).count();
    assert_eq!(anchor_children(&narrow), 2, "the confident round forks the anchor 2-wide");
    assert_eq!(anchor_children(&wide), 4, "the uncertain round forks the anchor 4-wide (the tuned default)");
    assert!(wide.tree.len() > 4 || narrow.tree.len() > 4, "neither round built a real fork");
}

/// The verify plan: one serial-exact chain window per root-to-leaf path. Every node is reached by at least one
/// window (each node's verified sample is read from the FIRST window that reaches it — a node whose subtree holds
/// several leaves is recomputed by the deeper windows, never re-read), every window fits the family's row cap, and
/// the anchor rides row 0 of them all.
#[test]
fn every_node_is_verified_exactly_once_by_the_path_windows() {
    for seed in [3u64, 7, 11, 29] {
        let l = lattice(seed, BLOCK, 8);
        let r = tree_round(&l, &[0.35; BLOCK]); // a low-confidence round: a real fork
        if r.tree.len() <= BLOCK {
            continue; // this seed's search stayed on the chain; the invariant is trivially the chain's
        }
        let mut seen = vec![false; r.tree.len()];
        for p in &r.paths {
            assert!(
                p.nodes.len() < WINDOW_ROWS,
                "a path window of {} rows exceeds the {}-row cap",
                p.nodes.len() + 1,
                WINDOW_ROWS
            );
            // the path is root-first: each node's parent is the node before it (or the anchor)
            for (j, &n) in p.nodes.iter().enumerate() {
                let parent = r.tree.parents[n];
                let want = if j == 0 { -1 } else { p.nodes[j - 1] as i32 };
                assert_eq!(parent, want, "seed {seed}: path node {j} (tree node {n}) does not follow its parent");
                seen[n] = true;
            }
            assert_eq!(r.tree.depth[*p.nodes.last().unwrap()] as usize, p.nodes.len(), "the path ends at its leaf's depth");
        }
        assert!(seen.iter().all(|&s| s), "seed {seed}: some node no window verifies");
        // every node has exactly one designated verifying window, and it is a window that reaches it
        for n in 0..r.tree.len() {
            let w = r.verifying_path(n);
            assert!(r.paths[w].nodes.contains(&n), "seed {seed}: node {n}'s designated window does not contain it");
        }
        // every leaf terminates a path and every path ends at a leaf
        let mut is_parent = vec![false; r.tree.len()];
        for &p in &r.tree.parents {
            if p >= 0 {
                is_parent[p as usize] = true;
            }
        }
        let leaves = (0..r.tree.len()).filter(|&i| !is_parent[i]).count();
        assert_eq!(r.paths.len(), leaves, "seed {seed}: {} paths for {leaves} leaves", r.paths.len());
    }
}

/// The accept walk (the engine's `accept_tree` over the round's rows): the committed tokens are the target's keyed
/// samples down the longest verified prefix — the walk stops at the first node whose sample matches no child, and
/// that node's own sample is still committed (the one token past the drafts).
#[test]
fn acceptance_takes_the_longest_verified_prefix() {
    // the anchor's children are tok 5 (row 1) and tok 6 (row 2); tok 7 (row 3) hangs off 5, tok 8 (row 4) off 6
    let tree = DraftTree {
        tokens: vec![5, 6, 7, 8],
        parents: vec![-1, -1, 0, 1],
        depth: vec![1, 1, 2, 2],
        cand: vec![0; 4],
        neg_log_p: vec![0.1, 0.2, 0.3, 0.4],
    };
    // the anchor's sample is 6: take the second branch; row 2's sample is 8: take row 4; row 4's sample ends the walk
    let a = accept_round(&tree, 1, &[6, 0, 8, 0, 99]);
    assert_eq!(a.path, vec![0, 2, 4], "the rows are the window's (the anchor is row 0)");
    assert_eq!(a.committed, vec![6, 8, 99], "the samples at the path's rows, the last one past the drafts");
    // the anchor's sample matches no child: only the anchor's own sample commits
    let a = accept_round(&tree, 1, &[9, 0, 0, 0, 0]);
    assert_eq!(a.path, vec![0]);
    assert_eq!(a.committed, vec![9]);
    // a sample that matches a deeper token of ANOTHER branch does not count (row 1's only child is row 3)
    let a = accept_round(&tree, 1, &[5, 8, 0, 0, 0]);
    assert_eq!(a.path, vec![0, 1]);
    assert_eq!(a.committed, vec![5, 8]);
    // the full path: every node's sample matches its child
    let a = accept_round(&tree, 1, &[5, 7, 0, 11, 0]);
    assert_eq!(a.path, vec![0, 1, 3]);
    assert_eq!(a.committed, vec![5, 7, 11]);
}

/// Byte-identity: a synthetic round commits exactly the serial stream. The synthetic target scores every token of a
/// small universe as a deterministic hash of the context, the serial reference is an independent argmax walk over
/// it, and the round's verify plan samples each window row from the same function at the row's own root path — the
/// construction the trunk's per-path windows implement. Over every seed the committed stream equals the serial one,
/// including rounds where the target follows a branch the markov walk's chain did not draft (the case trees exist
/// for) and rounds where it leaves the tree at once.
#[test]
fn a_tree_round_commits_the_serial_tokens_byte_for_byte() {
    /// The synthetic target: `score(ctx, tok)` — a deterministic hash of the whole context and the token.
    fn score(ctx: &[u32], tok: u32) -> f32 {
        let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ tok as u64;
        for &t in ctx {
            h ^= t as u64;
            h = h.wrapping_mul(0x1000_0019_3E4F_2B7D);
        }
        (h >> 33) as f32
    }
    let mut followed_a_branch = 0;
    let mut left_the_tree = 0;
    for seed in 1..=60u64 {
        let l = lattice(seed * 31 + BLOCK as u64, BLOCK, 8);
        let conf = [0.3f32; BLOCK]; // a low-confidence round: the widest fanout
        let r = tree_round(&l, &conf);
        let anchor = 777u32;
        // the universe the serial reference argmaxes over: the tree's tokens, the anchor and one outsider (a token
        // no node carries, so the target can leave the tree at any depth)
        let mut universe: Vec<u32> = r.tree.tokens.clone();
        universe.push(anchor);
        universe.push(999_999);
        universe.sort_unstable();
        universe.dedup();
        // the serial reference: an independent greedy walk over the target's own scores
        let max_depth = *r.tree.depth.iter().max().unwrap() as usize;
        let mut serial = Vec::new();
        let mut ctx = vec![anchor];
        for _ in 0..=max_depth {
            let best = universe.iter().copied().max_by(|a, b| score(&ctx, *a).total_cmp(&score(&ctx, *b))).unwrap();
            ctx.push(best);
            serial.push(best);
        }
        // the round's verify: each window row is sampled at its own root path — the row's context is the anchor,
        // its ancestors and its own token (what a per-path serial window computes the row's logits from)
        let (tokens, parents) = r.tree.rows(anchor);
        let samples: Vec<u32> = (0..tokens.len())
            .map(|row| {
                let mut path = Vec::new();
                let mut j = row as i32;
                while j >= 0 {
                    path.push(tokens[j as usize]);
                    j = parents[j as usize];
                }
                path.reverse();
                universe.iter().copied().max_by(|a, b| score(&path, *a).total_cmp(&score(&path, *b))).unwrap()
            })
            .collect();
        let a = accept_round(&r.tree, anchor, &samples);
        assert!(!a.committed.is_empty(), "the anchor's own sample always commits");
        let committed = &a.committed;
        assert_eq!(committed, &serial[..committed.len()], "seed {seed}: the committed stream is not the serial one");
        // the walk left the tree (committed shorter than the deepest path could verify) or verified it all
        assert!(committed.len() <= serial.len());
        if committed.len() < serial.len() {
            left_the_tree += 1;
        }
        // a win the chain alone could not have: the accepted path holds a token the markov walk never picked
        let walk = gb_spec::tree::greedy_chain(&l);
        if a.path.len() > 1 {
            let on_path: Vec<u32> = a.path[1..].iter().map(|&row| r.tree.tokens[row - 1]).collect();
            if on_path.iter().any(|&t| !walk.contains(&t)) {
                followed_a_branch += 1;
            }
        }
    }
    assert!(followed_a_branch > 0, "no round accepted a branch node: the fixture never exercises what trees are for");
    assert!(left_the_tree > 0, "no round left the tree: the early-exit accept is never exercised");
}
