//! The DSpark draft TREES (T2.5a): the DFlash2 `SPEC_TREE` playbook (the Qwen engine's `docs/SERVING.md`, "Speculation")
//! ported to this family — per round, the markov chain branches into several candidate continuations, every node is
//! verified against the serial computation, and the longest verified prefix is accepted, so replies stay byte-identical
//! to serial by construction: the committed tokens are always the target's own samples.
//!
//! # What is reused and what is mirrored
//!
//! The tree machinery is the engine's, imported unchanged ([`gb_spec::tree`]): the best-first search over a candidate
//! lattice with TensorFold's tuned constants (`Policy::default`: edge 0.6, noise 0.7, scale 1.5, branch 4 — the S0
//! study's grid winner), the `DraftTree` in pop order, and the target-samples walk [`gb_spec::tree::accept_tree`].
//! What is DSpark's own — and therefore lives here — is:
//!
//! * the **lattice** the search runs on: per level, the top-k of the head's unary logits (plus the markov walk's own
//!   pick, which the full-vocabulary bias can rank outside the unary top-k) and the `markov_head(p) . candidate`
//!   edge terms — the same `unary + edge` score the walk itself maximises (`spec.rs`'s "the block is a Markov walk
//!   through a rank-256 token embedding"). The capture is `Dsv41Model::propose_mtp`'s `lattice_host` round.
//! * the **confidence-scheduled fanout** ([`schedule_branch`]): DSpark's confidence head, which already schedules the
//!   chain's verify length, here schedules how wide the tree branches. This is the fusion of the two playbooks: a
//!   confident draft keeps the tree at the chain (branch 2), an uncertain one fans out to the engine's widest tuned
//!   default (branch 4).
//! * the **verify plan** ([`TreeRound::paths`]): the family's window is chain-causal (`dsv4_attn::forward_slot`), so
//!   a tree is verified as one serial-exact chain window per root-to-leaf path — every node sits on exactly one such
//!   path, and a path's rows attend exactly their committed prefix plus their ancestors, which is the serial
//!   computation of that path bit for bit. [`accept_round`] then walks the target's samples down the whole tree.
//!
//! # The shape (the pack's constants)
//!
//! `dspark_block_size` is 5 and the family's window holds [`WINDOW_ROWS`](super::model::WINDOW_ROWS) = 8 rows, so a
//! round's tree carries at most 7 nodes at most 5 deep: the markov chain (5 nodes) plus up to 2 branch nodes. The
//! engine's own trees run to 63 nodes over 16-candidate lattices; the DSpark lattice is capped at 8 candidates a
//! level (the widest fanout the search uses), which also keeps the capture's device read-out small.
//!
//! Nothing here can change what is committed (the exactness contract, `docs/SERVING.md`: "Replies are byte-identical
//! for every setting of the speculation, draft-tree, draft-head, window, prefill-policy and prefix-cache knobs").

use gb_engine::RowOutput;
use gb_spec::tree::{accept_tree, best_first, DraftTree, Lattice, Policy};

/// How wide one round's tree branches, from the confidence head's read of its own chain.
///
/// The chain's per-position acceptance runs 45% (prose) to 83% (counting) against the same head's confidences, whose
/// accept threshold is 0.38 (`REXL3_DSV41_MTP_THRESHOLD`): confidences at or above the n-gram heuristic's "sure"
/// constant (0.70, `propose_mtp`) mean the chain is right and branching only spends window rows, while confidences
/// down at the threshold mean the draft is guessing and the extra candidates are worth the rows. The widest setting
/// is the Qwen engine's tuned default branch (4).
pub fn schedule_branch(conf: &[f32]) -> usize {
    let mean = if conf.is_empty() { 0.0 } else { conf.iter().sum::<f32>() / conf.len() as f32 };
    if mean >= 0.70 {
        2
    } else if mean >= 0.50 {
        3
    } else {
        4
    }
}

/// One root-to-leaf path of the round's tree, as the verify window that checks it: the tree's node indices from the
/// root's child to the leaf, parents before children. The window itself is `[anchor, path nodes' tokens]` — the
/// anchor rides row 0 of every path's window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreePath {
    pub nodes: Vec<usize>,
}

impl TreeRound {
    /// The window that verifies `node`: the first path (in pop order of the leaves) that reaches it. A node with
    /// several leaf descendants rides several windows — the deeper ones recompute it — but exactly one window is
    /// designated to supply its verified sample, so no node's output is read twice or from a stale window.
    pub fn verifying_path(&self, node: usize) -> usize {
        self.paths.iter().position(|p| p.nodes.contains(&node)).expect("every node is on some root-to-leaf path")
    }
}

/// One DSpark tree round: the tree (the engine's search over this family's lattice) and the verify plan that checks
/// every node of it through the family's chain-causal windows.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeRound {
    /// The tree in pop order, parents before children (`gb_spec::tree::DraftTree`).
    pub tree: DraftTree,
    /// The policy the round searched with — the engine's tuned constants with [`schedule_branch`]'s fanout.
    pub policy: Policy,
    /// One [`TreePath`] per leaf: between them the paths reach every node (a shared trunk is recomputed by each
    /// window that rides it; [`TreeRound::verifying_path`] designates the one window a node's sample is read from).
    pub paths: Vec<TreePath>,
}

/// The round over `l` (a `block`-level lattice): the engine's best-first search, its fanout scheduled by the
/// confidence head's `conf` (the walked chain's confidences, truncated to whatever the round actually walked), its
/// depth capped at the block and its nodes at what one window holds (`window_rows`, anchor included).
///
/// The search is greedy (`keyed = None`): the DSpark walk is an argmax walk whatever the request's sampling, and its
/// confidence head — not the tree policy — carries the acceptance model. A smaller window or block is a prefix of
/// this round's tree (the search's first `n` pops are the `n`-node tree), so the row budget composes.
pub fn round(block: usize, window_rows: usize, l: &Lattice, conf: &[f32]) -> TreeRound {
    let policy = Policy { branch: schedule_branch(conf), ..Policy::default() };
    let nodes = window_rows.saturating_sub(1); // the anchor rides row 0 of the same window
    let tree = best_first(l, None, &policy, nodes).limit_depth(block);
    TreeRound { paths: paths_of(&tree), policy, tree }
}

/// The root-to-leaf paths of `tree`, in pop order of their leaves: every node is on at least one path (a node with
/// several leaf descendants rides several — the price of verifying a tree through chain-causal windows), and each
/// node's verified sample is read from the first path that reaches it.
pub fn paths_of(tree: &DraftTree) -> Vec<TreePath> {
    let mut is_parent = vec![false; tree.len()];
    for &p in &tree.parents {
        if p >= 0 {
            is_parent[p as usize] = true;
        }
    }
    (0..tree.len())
        .filter(|&i| !is_parent[i])
        .map(|leaf| {
            let mut nodes = Vec::with_capacity(tree.depth[leaf] as usize);
            let mut cur = leaf as i32;
            while cur >= 0 {
                nodes.push(cur as usize);
                cur = tree.parents[cur as usize];
            }
            nodes.reverse();
            TreePath { nodes }
        })
        .collect()
}

/// What a round's verify produced: the accepted rows and the tokens they commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundAccept {
    /// The accepted rows of the verify window (the anchor's row 0 first, each the parent of the next).
    pub path: Vec<usize>,
    /// The committed tokens: the target's samples at the path's rows — the accepted drafts, then the sample past
    /// them (the anchor row's sample is the first committed token).
    pub committed: Vec<u32>,
}

/// Accept a round: walk the target's samples down the tree (the engine's `accept_tree`, unchanged) and read off the
/// longest verified prefix. `samples[r]` is the sample the target drew at window row `r`'s own position (row 0 is the
/// anchor); from each row on the path take the child whose token equals the sample drawn there, and stop at the first
/// row whose sample matches no child — its own sample is still committed, the one token past the drafts.
pub fn accept_round(tree: &DraftTree, anchor: u32, samples: &[u32]) -> RoundAccept {
    let (tokens, parents) = tree.rows(anchor);
    let path = accept_tree(&tokens, &parents, samples);
    let committed = path.iter().map(|&r| samples[r]).collect();
    RoundAccept { path, committed }
}

// ---------------------------------------------------------------------------------------------------------------------- lane 2
// The TRUNK EXECUTION: a tree request verified as one serial-exact chain window per root-to-leaf path, and the
// commit's save/restore surgery over the persistent surfaces those windows mutate. The planning and the protocol
// live here, host-side, so their invariants are testable without a device; the surface operations live behind
// [`TreeRoundHost`] — `Dsv41Model` implements them over the real planes/engram/compressor/tap state, and the tests
// over an in-memory twin of the same contract.

/// One planned verify window of a tree request: the root-to-leaf path as ROW indices of the request (root-first,
/// the anchor's row 0 excluded — it rides row 0 of every window).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowPath {
    pub rows: Vec<usize>,
}

/// A tree request expanded into its path windows: the request's rows (`tokens[0]` is the anchor), their parents,
/// and one [`WindowPath`] per leaf, in leaf order — between them the paths reach every row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeWindows {
    /// The request's rows, the anchor first (the layout of `gb_engine::TreeRequest` / `DraftTree::rows`).
    pub tokens: Vec<u32>,
    /// `parents[i]` is row i's parent row; `parents[0] == -1`.
    pub parents: Vec<i32>,
    /// The anchor's absolute position (every path window starts there).
    pub first_position: u64,
    /// One path per leaf, root-first, parents before children.
    pub paths: Vec<WindowPath>,
}

impl TreeWindows {
    /// Expand a tree request (a `gb_engine::TreeRequest`'s `tokens`/`parents` at `first_position`). Refused by
    /// name: an empty request, a parent that has not run before its child, a request or a path window wider
    /// than the family's `window_rows`, and a round at position 0 (no decode round starts there — the anchor
    /// is always a committed prefix's tail, which the per-path isolation and the ring surgery both assume).
    pub fn plan(tokens: &[u32], parents: &[i32], first_position: u64, window_rows: usize) -> Result<Self, String> {
        if tokens.is_empty() || parents.len() != tokens.len() {
            return Err(format!("a tree window of {} rows and {} parents", tokens.len(), parents.len()));
        }
        if tokens.len() > window_rows {
            return Err(format!("a tree window of {} rows exceeds the family's {}-row window", tokens.len(), window_rows));
        }
        if first_position == 0 {
            return Err("a tree round at position 0 has no committed prefix to verify against".into());
        }
        if parents[0] != -1 {
            return Err(format!("the anchor's parent is {} (row 0 must be the root)", parents[0]));
        }
        for (i, &p) in parents.iter().enumerate().skip(1) {
            if p < 0 || p as usize >= i {
                return Err(format!("row {i}'s parent is {p}: parents run before their children"));
            }
        }
        if tokens.len() < 2 {
            return Err("a one-row tree is a chain, not a tree round".into());
        }
        let mut is_parent = vec![false; tokens.len()];
        for &p in &parents[1..] {
            is_parent[p as usize] = true;
        }
        let mut paths = Vec::new();
        for leaf in 1..tokens.len() {
            if is_parent[leaf] {
                continue;
            }
            let mut rows = Vec::with_capacity(parents[leaf].max(0) as usize + 1);
            let mut cur = leaf as i32;
            while cur > 0 {
                rows.push(cur as usize);
                cur = parents[cur as usize];
            }
            rows.reverse();
            if rows.len() + 1 > window_rows {
                return Err(format!(
                    "a root-to-leaf path of {} rows exceeds the family's {}-row window",
                    rows.len() + 1,
                    window_rows
                ));
            }
            paths.push(WindowPath { rows });
        }
        Ok(Self { tokens: tokens.to_vec(), parents: parents.to_vec(), first_position, paths })
    }

    /// Path `p`'s window: `[anchor, path rows' tokens]` — the chain a serial decode of that path would run.
    pub fn window_tokens(&self, p: usize) -> Vec<u32> {
        let mut t = Vec::with_capacity(self.paths[p].rows.len() + 1);
        t.push(self.tokens[0]);
        t.extend(self.paths[p].rows.iter().map(|&r| self.tokens[r]));
        t
    }

    /// The window row `r`'s verified sample is read from: `(path, row within that path's window)`. The FIRST
    /// path that reaches a row designates it (a row with several leaf descendants is recomputed by the deeper
    /// windows, never re-read); the anchor rides row 0 of the first.
    pub fn designated(&self, r: usize) -> (usize, usize) {
        if r == 0 {
            return (0, 0);
        }
        let (p, at) = self
            .paths
            .iter()
            .enumerate()
            .find_map(|(p, w)| w.rows.iter().position(|&x| x == r).map(|at| (p, at)))
            .unwrap_or_else(|| panic!("row {r} is on no root-to-leaf path"));
        (p, at + 1)
    }

    /// The engine's accepted path (request row indices, the anchor's row 0 first) as the window that commits
    /// it: `(path index, rows of that window the commit keeps, the anchor included)`. Refused when the path
    /// is not a connected root path of the planned tree — committing anything else would leave the model's
    /// state out of step with the committed tokens.
    pub fn accept_window(&self, path: &[usize]) -> Result<(usize, usize), String> {
        let Some((&root, rest)) = path.split_first() else { return Err("an empty path commits nothing".into()) };
        if root != 0 {
            return Err(format!("the path starts at row {root}, not the anchor's row 0"));
        }
        let Some((&last, _)) = path.split_last() else { unreachable!() };
        for (k, &r) in rest.iter().enumerate() {
            if r == 0 || r >= self.tokens.len() {
                return Err(format!("path row {r} is outside the window's {} rows", self.tokens.len()));
            }
            let want = path[k] as i32; // the row before it (root-first)
            if self.parents[r] != want {
                return Err(format!("row {r}'s parent is row {}, but the path reaches it from row {}", self.parents[r], want));
            }
        }
        let (p, at) = self.designated(last);
        Ok((p, at + 1))
    }
}

/// The persistent surfaces a tree round's path windows mutate, as the operations the round's protocol needs
/// (T2.5a lane 2). The DEVICE implementation is `Dsv41Model`'s — the planes' window rings, the engram rings,
/// the compressors' carry and the DSpark taps; a test twin implements the same contract in memory, so the
/// driver's invariants (the per-path isolation, the commit surgery) are checked without a device.
pub trait TreeRoundHost {
    /// One path's chain window `tokens` at `first_position`, ISOLATED to the round's base state: restore the
    /// base engram ring and compressor carry before the window runs (the gather appends per row and the
    /// carry folds into the first pair, so sibling paths contaminate each other without the restore), then
    /// snapshot the round's base surfaces on the FIRST window and, after every window, save its per-row
    /// state for the commit (the KV rows it wrote, the carry it left, its tap rows). One output per row.
    fn tree_path_window(&mut self, tokens: &[u32], first_position: u64, candidates: usize) -> Result<Vec<RowOutput>, String>;

    /// Reconcile every surface to the serial replay of path window `idx`'s first `rows` rows (the anchor is
    /// row 0 of `window`): the touched ring slots back to the round's base bytes, the accepted path's own KV
    /// rows written over them, the engram ring replayed (`window[..rows]` appended to the base ring), the
    /// compressor carry taken from the accepted path's own window, and the DSpark tap reseeded from its
    /// saved tap rows. After a commit the round is closed.
    fn tree_commit_window(&mut self, idx: usize, rows: usize, window: &[u32]) -> Result<(), String>;

    /// Abandon the round (no commit will come — the sequence ended in-draft, or the round failed): drop the
    /// saves and restore the base surfaces.
    fn tree_drop_round(&mut self) -> Result<(), String>;
}

/// Verify one planned tree request over `host`: every path window in leaf order, then the per-row outputs
/// assembled in the REQUEST's row order (row `r`'s output read from its designated window's row) — the shape
/// `Backend::step_tree` answers with, whatever order the windows ran in.
pub fn verify_tree<H: TreeRoundHost>(host: &mut H, w: &TreeWindows, candidates: usize) -> Result<Vec<RowOutput>, String> {
    let mut per_path = Vec::with_capacity(w.paths.len());
    for p in 0..w.paths.len() {
        per_path.push(host.tree_path_window(&w.window_tokens(p), w.first_position, candidates)?);
    }
    let mut out = Vec::with_capacity(w.tokens.len());
    out.push(per_path[0][0].clone());
    for r in 1..w.tokens.len() {
        let (p, at) = w.designated(r);
        out.push(per_path[p][at].clone());
    }
    Ok(out)
}

/// Commit the engine's accepted `path` (request row indices, the anchor first): map it to the window that
/// verified it and hand the host the surgery. The committed TOKENS stay the engine's business — nothing here
/// can move a byte of a reply.
pub fn commit_round<H: TreeRoundHost>(host: &mut H, w: &TreeWindows, path: &[usize]) -> Result<(), String> {
    let (idx, rows) = w.accept_window(path)?;
    host.tree_commit_window(idx, rows, &w.window_tokens(idx))
}
