#![allow(clippy::needless_range_loop)]
//! Tree windows on the real model (docs/TREE_DRAFTING.md section 4, gates 1 and 2).
//!
//! * `tree_rows_equal_their_root_path_serial_chains`: random trees (1..=32 rows, branching 1 to 4, on two sequences at once, one of them
//!   crossing the 512-key chunk boundary, the other the 256-token page boundary) are verified in one tree window; every node's candidates
//!   (the top-28 ids and logits, bit for bit) must equal those the plain window kernels give the chain of its root path on a mirror slot
//!   that holds the same committed context. A random root path is then committed on the tree slots (conv, the owed replay of S, the KV
//!   rows, the draft is not involved) and the same path as a chain on the mirror slots; after the rounds the whole recurrent state
//!   (S and conv of all 48 layers) and every committed KV row of all 16 attention layers must be bit identical between slot and mirror.
//! * `chain_windows_through_the_tree_kernels_equal_the_plain_kernels`: the deferred-commit session of `deferred_commit.rs` (windows of 1 to
//!   8 rows, keeps from 1 to the whole window, sequences joining, leaving and reordering) run once on the plain kernels and once with every
//!   window forced through the tree kernels (`Model::enable_trees(true)`): candidates of every round and the state afterwards identical.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test tree_exact -- --ignored --nocapture

use gb_cuda::gdn::{CONV_SLOT_ELEMS, S_SLOT_ELEMS};
use gb_cuda::tree::{random_parents, Tree};
use gb_engine::{RowRequest, TreeRequest};
use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;

const COUNT: usize = 28;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn tokens(rng: &mut Lcg, n: usize) -> Vec<u32> {
    (0..n).map(|_| 1000 + rng.below(99_000) as u32).collect()
}

fn load(slots: usize, kv_blocks: usize, max_context: usize) -> Option<Model> {
    load_window(slots, kv_blocks, max_context, gb_model::forward::WINDOW_ROWS)
}

/// [`load`] with a window of `window` rows (`--window-rows`: up to 128).
fn load_window(slots: usize, kv_blocks: usize, max_context: usize, window: usize) -> Option<Model> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    if !gb_cuda::device_available() {
        return None;
    }
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    Some(Model::with_window_rows(cfg, w, slots, kv_blocks, max_context, 4096, window).unwrap())
}

fn rows_equal(what: &str, a: &Candidates, ar: usize, b: &Candidates, br: usize) {
    let (ai, bi) = (&a.ids[ar * a.count..(ar + 1) * a.count], &b.ids[br * b.count..(br + 1) * b.count]);
    assert_eq!(ai, bi, "{what}: candidate ids");
    let abits: Vec<u32> = a.vals[ar * a.count..(ar + 1) * a.count].iter().map(|v| v.to_bits()).collect();
    let bbits: Vec<u32> = b.vals[br * b.count..(br + 1) * b.count].iter().map(|v| v.to_bits()).collect();
    assert_eq!(abits, bbits, "{what}: candidate logits (bits)");
}

/// One slot's recurrent state as (S, conv) per linear layer.
fn slot_state(m: &Model, slot: usize) -> Vec<(Vec<u32>, Vec<u16>)> {
    (0..m.pools.s.len())
        .map(|li| {
            let s = m.pools.s[li].to_vec::<f32>().unwrap();
            let c = m.pools.conv[li].to_vec::<u16>().unwrap();
            (
                s[slot * S_SLOT_ELEMS..(slot + 1) * S_SLOT_ELEMS].iter().map(|x| x.to_bits()).collect(),
                c[slot * CONV_SLOT_ELEMS..(slot + 1) * CONV_SLOT_ELEMS].to_vec(),
            )
        })
        .collect()
}

/// A digest of every byte of the paged KV cache (all 16 attention layers, K and V, every page).
fn kv_digest(m: &Model) -> Vec<u64> {
    let digest = |v: &[u16]| {
        let mut h = 0xcbf29ce484222325u64;
        for c in v.chunks(4) {
            let mut w = 0u64;
            for (i, &x) in c.iter().enumerate() {
                w |= (x as u64) << (16 * i);
            }
            h = (h ^ w).wrapping_mul(0x100000001b3);
        }
        h
    };
    m.kv.iter().flat_map(|c| [digest(&c.k.to_vec::<u16>().unwrap()), digest(&c.v.to_vec::<u16>().unwrap())]).collect()
}

/// The committed K and V rows (positions `0..upto`) of a slot, per full-attention layer.
fn slot_kv(m: &Model, slot: usize, upto: usize) -> Vec<(Vec<u16>, Vec<u16>)> {
    let blocks = m.kv_tab.blocks_of(slot).to_vec();
    (0..m.kv.len())
        .map(|fi| {
            let (k, v) = (m.kv[fi].k.to_vec::<u16>().unwrap(), m.kv[fi].v.to_vec::<u16>().unwrap());
            let (mut ko, mut vo) = (Vec::new(), Vec::new());
            for t in 0..upto {
                let off = (blocks[t / 256] as usize * 256 + t % 256) * 1024;
                ko.extend_from_slice(&k[off..off + 1024]);
                vo.extend_from_slice(&v[off..off + 1024]);
            }
            (ko, vo)
        })
        .collect()
}

/// Rounds of the tree sessions: `REXL3_TREE_ROUNDS` (default 10; the full gate runs more).
fn tree_rounds() -> usize {
    std::env::var("REXL3_TREE_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(10)
}

#[test]
#[ignore]
fn tree_rows_equal_their_root_path_serial_chains() {
    tree_session(2, tree_rounds());
}

/// One sequence with a window of up to 64 rows (the whole window: `WINDOW_ROWS`).
#[test]
#[ignore]
fn one_sequence_of_up_to_64_tree_rows_equals_its_serial_chains() {
    tree_session(1, tree_rounds());
}

/// Tree windows of up to 128 rows, the window `--window-rows 128` gives a model (c32): the tree buffers are sized by the model's window, so a
/// window past the served 64 rows must run and give every node the bits of its serial chain. Two sequences of 48 to 64 rows (96 to 128 in all).
#[test]
#[ignore]
fn tree_windows_of_up_to_128_rows_equal_their_root_path_serial_chains() {
    tree_session_in(2, tree_rounds(), 128);
}

/// One sequence with a tree of 96 to 128 rows (the kernels' tables hold 128).
#[test]
#[ignore]
fn one_sequence_of_up_to_128_tree_rows_equals_its_serial_chains() {
    tree_session_in(1, tree_rounds(), 128);
}

/// `t_n` tree slots `0..t_n`, mirror slots `t_n..2 * t_n`, a window of the served 64 rows.
fn tree_session(t_n: usize, rounds: usize) {
    tree_session_in(t_n, rounds, gb_model::forward::WINDOW_ROWS)
}

/// [`tree_session`] on a model whose window is `window` rows. Up to the served 64 rows a tree has a random size; wider windows are filled (every
/// tree takes its share of the window less a few rows) so that most rounds run past 64 rows.
fn tree_session_in(t_n: usize, rounds: usize, window: usize) {
    let Some(mut m) = load_window(2 * t_n, 48, 2048, window) else {
        return eprintln!("not run: needs REXL3_QWEN_DIR and a GPU");
    };
    m.enable_trees(false).unwrap();
    // the plan counts exactly what is allocated (families::qwen: `window_record_bytes_per_row`), at the window the model was built with
    assert_eq!(
        m.tree_bytes() as u64,
        gb_model::forward::tree_buffer_bytes_for(&m.cfg, window),
        "the tree buffers are not what the memory plan counts at a {window}-row window"
    );
    let mut widest = 0usize;
    let mut rng = Lcg(0x5DEECE66D);
    // slot 0 starts just below the 512-key chunk boundary, slot 1 just below a 256-token page boundary
    let mut pos = [506usize, 249];
    for s in 0..t_n {
        let prompt = tokens(&mut rng, pos[s]);
        for slot in [s, t_n + s] {
            m.begin_slot(slot).unwrap();
            m.prefill_chunk(slot, 0, &prompt, prompt.len(), None, 0).unwrap();
        }
    }
    let (mut nodes, mut committed_rows, mut deep) = (0usize, 0usize, 0usize);
    for round in 0..rounds {
        // random trees, one per tree slot
        let share = window / t_n;
        let trees: Vec<Tree> = (0..t_n)
            .map(|s| {
                let size = if window > 64 { share - rng.below(share / 4) } else { 1 + rng.below(share) };
                Tree::new(&random_parents(round as u64 * 7 + s as u64 + 1, size, 1 + rng.below(4))).unwrap()
            })
            .collect();
        widest = widest.max(trees.iter().map(Tree::len).sum());
        let toks: Vec<Vec<u32>> = trees.iter().map(|t| tokens(&mut rng, t.len())).collect();
        // (a) the chain of every leaf path, on the mirror slots, read only
        let mut want: Vec<Vec<Option<(Candidates, usize)>>> = Vec::new();
        for s in 0..t_n {
            let t = &trees[s];
            let mut per_node: Vec<Option<(Candidates, usize)>> = (0..t.len()).map(|_| None).collect();
            let leaves: Vec<usize> = (0..t.len()).filter(|&i| !t.parents().contains(&(i as i32))).collect();
            for leaf in leaves {
                let path = t.path(leaf);
                let req = RowRequest {
                    slot: t_n + s,
                    tokens: path.iter().map(|&i| toks[s][i]).collect(),
                    first_position: pos[s] as u64,
                };
                let c = m.window_candidates(&RowBatch::from_requests(&[req]), COUNT).unwrap();
                for (j, &node) in path.iter().enumerate() {
                    if per_node[node].is_none() {
                        per_node[node] = Some((c.clone(), j));
                    }
                }
            }
            want.push(per_node);
        }
        // (b) the tree window of both sequences at once
        let reqs: Vec<TreeRequest> = (0..t_n)
            .map(|s| TreeRequest {
                slot: s,
                tokens: toks[s].clone(),
                parents: trees[s].parents().to_vec(),
                first_position: pos[s] as u64,
            })
            .collect();
        let b = RowBatch::from_trees(&reqs);
        let cache_before = kv_digest(&m);
        let got = m.window_candidates(&b, COUNT).unwrap();
        // committed state is read only during a window: siblings share logical positions, so nothing of a tree window may enter the pages
        assert!(cache_before == kv_digest(&m), "round {round}: a tree window wrote into the paged KV cache");
        for s in 0..t_n {
            for node in 0..trees[s].len() {
                let (c, j) = want[s][node].as_ref().expect("every node is on a leaf path");
                rows_equal(
                    &format!("round {round} seq {s} node {node} (depth {}) of {} rows", trees[s].depth(node), trees[s].len()),
                    &got,
                    b.seq_start[s] as usize + node,
                    c,
                    *j,
                );
                nodes += 1;
            }
        }
        // (c) commit a random root path on the tree slots, the same path as a chain on the mirrors
        let paths: Vec<Vec<usize>> = (0..t_n).map(|s| trees[s].path(rng.below(trees[s].len()))).collect();
        for s in 0..t_n {
            m.commit_path(s, &paths[s]).unwrap();
        }
        for s in 0..t_n {
            // the mirror's window is the accepted path as a chain; its commit applies to that window
            let req = RowRequest {
                slot: t_n + s,
                tokens: paths[s].iter().map(|&i| toks[s][i]).collect(),
                first_position: pos[s] as u64,
            };
            m.window_candidates(&RowBatch::from_requests(&[req]), COUNT).unwrap();
            m.commit(t_n + s, paths[s].len());
            pos[s] += paths[s].len();
            committed_rows += paths[s].len();
            deep += (paths[s].len() >= 4) as usize;
        }
    }
    m.settle().unwrap();
    assert!(rounds < 10 || (nodes >= 100 && deep >= 4), "the session must cover many nodes and deep paths ({nodes}, {deep})");
    assert!(window <= 64 || widest > 64, "a {window}-row window never ran past 64 rows (widest {widest})");
    for s in 0..t_n {
        let (a, b) = (slot_state(&m, s), slot_state(&m, t_n + s));
        for (li, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.0, y.0, "slot {s}: S of linear layer {li} differs from its serial mirror");
            assert_eq!(x.1, y.1, "slot {s}: conv history of linear layer {li} differs from its serial mirror");
        }
        let (ka, kb) = (slot_kv(&m, s, pos[s]), slot_kv(&m, t_n + s, pos[s]));
        for (fi, (x, y)) in ka.iter().zip(&kb).enumerate() {
            assert_eq!(x.0, y.0, "slot {s}: committed K rows of attention layer {fi} differ from the serial mirror");
            assert_eq!(x.1, y.1, "slot {s}: committed V rows of attention layer {fi} differ from the serial mirror");
        }
    }
    eprintln!("tree windows == per-path serial chains ({t_n} sequence(s), window {window}, widest {widest} rows): {nodes} nodes in {rounds} rounds, {committed_rows} committed rows; S, conv and KV bit identical");
}

// ---------------------------------------------------------------------------------------------------------------------------
// the draft's context after a path commit

fn load_with_draft(slots: usize, kv_blocks: usize, max_context: usize) -> Option<Model> {
    let dd = std::path::PathBuf::from(std::env::var("REXL3_DRAFT_DIR").ok()?);
    let mut m = load(slots, kv_blocks, max_context)?;
    let vocab = m.cfg.vocab;
    m.draft = Some(gb_model::draft::Draft::load(&Pack::open(&dd).unwrap(), vocab, slots, 256).unwrap().0);
    Some(m)
}

/// The draft reads the target's hidden states of the committed rows (its context), which a commit hands it row by row. After a TREE
/// window the accepted rows are not a prefix of the window, so the rows the draft absorbs must be the path's: the drafts that follow have to
/// be those of the same history committed as a chain. (A wrong draft context never changes a reply, only how many drafts are accepted, so
/// no reply-level test sees it.)
#[test]
#[ignore]
fn a_path_commit_gives_the_draft_the_context_a_serial_chain_does() {
    let Some(mut m) = load_with_draft(2, 64, 2048) else {
        return eprintln!("not run: needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR and a GPU");
    };
    m.enable_trees(false).unwrap();
    let mut rng = Lcg(0xC0FFEE);
    let mut committed = tokens(&mut rng, 150);
    for slot in 0..2 {
        m.begin_slot(slot).unwrap();
        m.prefill_chunk(slot, 0, &committed, committed.len(), Some(COUNT), 0).unwrap();
    }
    let samplings = [
        gb_sampling::Sampling { seed: 5, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
        gb_sampling::Sampling { seed: 6, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
    ];
    let mut anchor = 1000 + rng.below(99_000) as u32;
    let (mut non_prefix, mut compared, mut longest) = (0, 0, 0);
    for round in 0..8u64 {
        // round 3 commits a copy-branch-shaped path: a fork under the anchor (row 1 is a dead end) and a 31-row chain under its sibling
        let parents: Vec<i32> = if round == 3 {
            let mut p = vec![-1, 0, 0];
            p.extend((3..34).map(|i| if i == 3 { 2 } else { i - 1 }));
            p
        } else {
            random_parents(round * 5 + 3, 12 + rng.below(12), 2 + rng.below(3))
        };
        let tree = Tree::new(&parents).unwrap();
        let mut toks = tokens(&mut rng, tree.len());
        toks[0] = anchor;
        // the deepest node's path: the longest, and a prefix of the window only if the tree is a chain
        let deepest = (0..tree.len()).max_by_key(|&i| (tree.depth(i), i)).unwrap();
        let path = tree.path(deepest);
        non_prefix += usize::from(path.iter().enumerate().any(|(i, &r)| i != r));
        longest = longest.max(path.len());
        let first = committed.len() as u64;
        let req = TreeRequest { slot: 0, tokens: toks.clone(), parents: tree.parents().to_vec(), first_position: first };
        m.window_candidates(&RowBatch::from_trees(&[req]), COUNT).unwrap();
        m.commit_path(0, &path).unwrap();
        let chain = RowRequest { slot: 1, tokens: path.iter().map(|&i| toks[i]).collect(), first_position: first };
        m.window_candidates(&RowBatch::from_requests(&[chain]), COUNT).unwrap();
        m.commit(1, path.len());
        committed.extend(path.iter().map(|&i| toks[i]));
        anchor = 1000 + rng.below(99_000) as u32;
        let mut seq = committed.clone();
        seq.push(anchor);
        for (k, sampling) in samplings.iter().enumerate() {
            let a = m.propose_drafts(0, &seq, sampling).unwrap().expect("a draft");
            let b = m.propose_drafts(1, &seq, sampling).unwrap().expect("a draft");
            assert_eq!(
                a, b,
                "round {round} sampling {k}: the drafts after a path commit differ from those after the same path as a chain"
            );
            compared += 1;
        }
    }
    assert!(non_prefix >= 4, "the session must commit paths that are not a prefix of their window ({non_prefix})");
    assert!(longest >= 32, "the session must commit a path as long as a copy branch makes ({longest})");
    eprintln!("drafts after {compared} path commits ({non_prefix} of them not a prefix of their window) == drafts after the same history as chains");
}

// ---------------------------------------------------------------------------------------------------------------------------
// chain == tree

const SLOTS: usize = 3;
const ROUNDS: usize = 40;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn hash(c: &Candidates) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for (i, v) in c.ids.iter().zip(&c.vals) {
        fnv(&mut h, &i.to_le_bytes());
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    h
}

/// What a session returns: the state hashes and, per window, the tokens and their candidates' ids.
type Replay = (Vec<u64>, Vec<(Vec<u32>, Vec<u16>)>);

/// The deferred-commit session (see `deferred_commit.rs`): random windows of 1 to 8 rows over a random subset of the slots, partial keeps.
fn session(m: &mut Model) -> Replay {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let mut pos = [0usize; SLOTS];
    for (s, p) in pos.iter_mut().enumerate() {
        let prompt = tokens(&mut rng, 140 + 33 * s);
        m.begin_slot(s).unwrap();
        m.prefill_chunk(s, 0, &prompt, prompt.len(), None, 0).unwrap();
        *p = prompt.len();
    }
    let mut hashes = Vec::new();
    for _ in 0..ROUNDS {
        let mut order: Vec<usize> = (0..SLOTS).collect();
        for i in (1..SLOTS).rev() {
            order.swap(i, rng.below(i + 1));
        }
        order.truncate(1 + rng.below(SLOTS));
        let reqs: Vec<RowRequest> = order
            .iter()
            .map(|&slot| {
                let n = 1 + rng.below(8);
                RowRequest { slot, tokens: tokens(&mut rng, n), first_position: pos[slot] as u64 }
            })
            .collect();
        let keeps: Vec<usize> = reqs.iter().map(|r| 1 + rng.below(r.tokens.len())).collect();
        let c = m.window_candidates(&RowBatch::from_requests(&reqs), COUNT).unwrap();
        hashes.push(hash(&c));
        for (r, &k) in reqs.iter().zip(&keeps) {
            m.commit(r.slot, k);
            pos[r.slot] += k;
        }
    }
    m.settle().unwrap();
    let mut state = Vec::new();
    for s in 0..SLOTS {
        state.extend(slot_state(m, s));
    }
    (hashes, state)
}

#[test]
#[ignore]
fn chain_windows_through_the_tree_kernels_equal_the_plain_kernels() {
    let Some(mut m) = load(SLOTS, 96, 16384) else {
        return eprintln!("not run: needs REXL3_QWEN_DIR and a GPU");
    };
    let (h_plain, s_plain) = session(&mut m);
    m.enable_trees(true).unwrap();
    let (h_tree, s_tree) = session(&mut m);
    let first = h_plain.iter().zip(&h_tree).position(|(a, b)| a != b);
    assert_eq!(
        first, None,
        "candidates differ at round {first:?}: a chain through the tree kernels is not the plain kernels' chain"
    );
    assert!(s_plain == s_tree, "the recurrent state after {ROUNDS} rounds differs between the plain and the tree kernels");
    eprintln!("{ROUNDS} rounds of chain windows: candidates and the whole recurrent state identical through the tree kernels");
}
