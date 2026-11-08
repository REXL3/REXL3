//! Draft trees (docs/TREE_DRAFTING.md, stages S1 and S2): the best-first search over the DFlash2 selector's lattice, the calibrated
//! probability of each node, and the walk of the target's samples down a tree.
//!
//! The selector's block runs once on `[anchor, MASK x 7]`, so each level's top-16 candidates and their unary scores do not depend on which
//! tokens are picked; only the edge term between a node's token and each candidate of the next level does (the chain walk picks one path
//! through that lattice). [`best_first`] searches the whole lattice instead: a heap over summed sibling log-probabilities pops nodes best
//! first and expands each popped node's `branch` best children, so the first `n` pops are the `n`-node tree (nested prefixes: a smaller
//! budget is a prefix). The policy is TensorFold's (edge 0.6, the target's keyed Gumbel noise at weight 0.7 on sampled requests, scale 1.5,
//! branch 4), which won the parameter grid of the S0 study (`tools/tree_ceiling.py`).
//!
//! Nothing here can change what is committed: the committed tokens are the target's keyed samples, and a tree only adds chances for one of
//! a node's children to equal the sample drawn from the node's own logits.

use crate::stats::Mode;
use gb_sampling::{uniform, Sampling};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// The selector's candidate lattice of one draft round (one sequence): per draft row (level `1..=rows`) the head's top-`k` candidates with
/// their unary scores, and the dot terms between a node's token and each candidate of the level (`gb_dflash_selector_edges`). The chain
/// walk's score of candidate `c` of level `i` after pick `parent` is `unary[i][c] + edge[i][parent][c]`, bit for bit, where `parent` indexes
/// the previous level's candidates (level 0: the anchor, index 0).
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    pub rows: usize,
    pub k: usize,
    /// u32 `[rows][k]`, token ids, best unary first (ties lowest id)
    pub ids: Vec<u32>,
    /// f32 `[rows][k]`, the head's logits of those candidates
    pub unary: Vec<f32>,
    /// f32 `[rows][k parents][k candidates]`
    pub edge: Vec<f32>,
    /// the round ran the restricted (`true`) or the full (`false`) draft head
    pub restricted_head: bool,
}

/// The tree policy's constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    /// Weight of the edge term next to the unary score.
    pub edge: f64,
    /// Weight of the target's keyed Gumbel noise at the candidate's position and token (sampled requests only).
    pub noise: f64,
    /// Divides the scores before the per-level softmax (a temperature on the draft's own confidence).
    pub scale: f64,
    /// Children expanded under each popped node.
    pub branch: usize,
}

impl Default for Policy {
    /// TensorFold's constants: the best of the 24 grid points of the S0 study on the lattices of 72 real generations.
    fn default() -> Self {
        Self { edge: 0.6, noise: 0.7, scale: 1.5, branch: 4 }
    }
}

/// A draft tree in pop order (parents before children). Row 0 of the verify window is the anchor; node `i` is row `i + 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftTree {
    /// Token of each node.
    pub tokens: Vec<u32>,
    /// Index of each node's parent among the nodes, -1 for a child of the anchor.
    pub parents: Vec<i32>,
    /// Depth of each node: 1 for a child of the anchor.
    pub depth: Vec<u8>,
    /// Index of the node's token among its level's candidates (the next level's edge row to read).
    pub cand: Vec<u8>,
    /// The path's summed -log probability: `exp(-neg_log_p)` is the raw number the search ranks by.
    pub neg_log_p: Vec<f64>,
}

impl DraftTree {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    /// The raw path number of node `i` (what the search ranks by).
    pub fn raw(&self, i: usize) -> f64 {
        (-self.neg_log_p[i]).exp()
    }
    /// The first `n` pops: the smaller tree of a smaller budget.
    pub fn prefix(&self, n: usize) -> DraftTree {
        let n = n.min(self.len());
        DraftTree {
            tokens: self.tokens[..n].to_vec(),
            parents: self.parents[..n].to_vec(),
            depth: self.depth[..n].to_vec(),
            cand: self.cand[..n].to_vec(),
            neg_log_p: self.neg_log_p[..n].to_vec(),
        }
    }
    /// The tree without the nodes deeper than `cap` (a request that can take only `cap` more tokens past the sample gains nothing from them).
    /// A node's parent is shallower than it, so what is kept is a tree and the order is the pop order.
    pub fn limit_depth(&self, cap: usize) -> DraftTree {
        let mut map = vec![-1i32; self.len()];
        let mut out = DraftTree { tokens: vec![], parents: vec![], depth: vec![], cand: vec![], neg_log_p: vec![] };
        for i in 0..self.len() {
            if self.depth[i] as usize > cap {
                continue;
            }
            map[i] = out.len() as i32;
            out.tokens.push(self.tokens[i]);
            out.parents.push(if self.parents[i] < 0 { -1 } else { map[self.parents[i] as usize] });
            out.depth.push(self.depth[i]);
            out.cand.push(self.cand[i]);
            out.neg_log_p.push(self.neg_log_p[i]);
        }
        out
    }

    /// A chain of `tokens` (parent of node i is i - 1): the shape of a plain window's drafts.
    pub fn chain(tokens: &[u32]) -> DraftTree {
        let n = tokens.len();
        DraftTree {
            tokens: tokens.to_vec(),
            parents: (0..n as i32).map(|i| i - 1).collect(),
            depth: (1..=n as u8).collect(),
            cand: vec![0; n],
            neg_log_p: vec![0.0; n],
        }
    }
    /// The verify window's rows: the anchor first, then the nodes in pop order. `(tokens, parents)` with parents as row indices
    /// (the anchor's parent is -1, a child of the anchor has parent 0).
    pub fn rows(&self, anchor: u32) -> (Vec<u32>, Vec<i32>) {
        let mut tokens = Vec::with_capacity(self.len() + 1);
        let mut parents = Vec::with_capacity(self.len() + 1);
        tokens.push(anchor);
        parents.push(-1);
        for i in 0..self.len() {
            tokens.push(self.tokens[i]);
            parents.push(self.parents[i] + 1);
        }
        (tokens, parents)
    }
}

/// A heap entry: smallest summed -log p first, ties by parent, token, depth (TensorFold's tuple order).
#[derive(Clone, Copy, Debug)]
struct Entry {
    score: f64,
    parent: i32,
    token: u32,
    depth: u8,
    cand: u8,
}

impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .total_cmp(&o.score)
            .then(self.parent.cmp(&o.parent))
            .then(self.token.cmp(&o.token))
            .then(self.depth.cmp(&o.depth))
    }
}

/// The target's keyed Gumbel noise `-ln(-ln u)` at (seed, absolute position, token id), the draw its sampler uses.
pub fn gumbel(seed: u64, position: u64, id: u64) -> f64 {
    -(-uniform(seed, position, id).ln()).ln()
}

/// One level's per-candidate log probabilities given the parent's index among the previous level's candidates.
fn level_logp(l: &Lattice, p: &Policy, keyed: Option<(&Sampling, u64)>, d: usize, parent: usize) -> Vec<f64> {
    let k = l.k;
    let t = keyed.map_or(1.0, |(s, _)| s.temperature.max(1e-6));
    let mut z: Vec<f64> = (0..k)
        .map(|c| {
            let mut v = (l.unary[d * k + c] as f64 + p.edge * l.edge[(d * k + parent) * k + c] as f64) / t;
            if let Some((s, pos0)) = keyed {
                v += p.noise * gumbel(s.seed, pos0 + d as u64, l.ids[d * k + c] as u64);
            }
            v / p.scale
        })
        .collect();
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = z.iter().map(|&x| (x - m).exp()).sum::<f64>().ln();
    for x in z.iter_mut() {
        *x = *x - m - lse;
    }
    z
}

/// The best-first tree over `l` with at most `max_nodes` nodes. `keyed`: for a sampled request, its sampling and the absolute position of
/// the first draft row (the anchor's position + 1); `None` for greedy. The first `n` nodes are the `n`-node tree, so a smaller budget is
/// [`DraftTree::prefix`].
pub fn best_first(l: &Lattice, keyed: Option<(&Sampling, u64)>, p: &Policy, max_nodes: usize) -> DraftTree {
    let mut tree = DraftTree { tokens: vec![], parents: vec![], depth: vec![], cand: vec![], neg_log_p: vec![] };
    let mut heap: BinaryHeap<Reverse<Entry>> = BinaryHeap::new();
    let push = |heap: &mut BinaryHeap<Reverse<Entry>>, d: usize, parent_cand: usize, parent_node: i32, base: f64| {
        let lp = level_logp(l, p, keyed, d, parent_cand);
        let mut order: Vec<usize> = (0..l.k).collect();
        order.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]).then(a.cmp(&b)));
        for &c in order.iter().take(p.branch) {
            if !lp[c].is_finite() {
                break;
            }
            heap.push(Reverse(Entry {
                score: base - lp[c],
                parent: parent_node,
                token: l.ids[d * l.k + c],
                depth: (d + 1) as u8,
                cand: c as u8,
            }));
        }
    };
    if l.rows == 0 || max_nodes == 0 {
        return tree;
    }
    push(&mut heap, 0, 0, -1, 0.0);
    while tree.len() < max_nodes {
        let Some(Reverse(e)) = heap.pop() else { break };
        let me = tree.len() as i32;
        tree.tokens.push(e.token);
        tree.parents.push(e.parent);
        tree.depth.push(e.depth);
        tree.cand.push(e.cand);
        tree.neg_log_p.push(e.score);
        if (e.depth as usize) < l.rows {
            push(&mut heap, e.depth as usize, e.cand as usize, me, e.score);
        }
    }
    tree
}

/// The most tokens one copy branch carries (the width of `measured_copy::COPY_Q`).
pub const MAX_COPY: usize = crate::measured_copy::COPY_Q[0].len();

/// The probability that the `k`-th (0-based) token of a copy chain, produced by a match of `match_len` tokens, equals the target's, given the
/// chain's tokens before it did (`measured_copy`).
pub fn copy_q(match_len: usize, k: usize) -> f64 {
    let b = crate::measured_copy::MATCH_BUCKETS.iter().position(|&(lo, hi)| (lo..=hi).contains(&match_len)).unwrap_or(0);
    crate::measured_copy::COPY_Q[b][k.min(MAX_COPY - 1)]
}

/// A chain of tokens copied from the stream's own text (`CopyDrafter::propose_after`) to hang off the end of the draft's picks.
#[derive(Clone, Copy, Debug)]
pub struct CopyBranch<'a> {
    /// The draft's own picks, one per level (the walk's chain): the copy continues past the last of them. Every pick must be one of its
    /// level's candidates or no copy branch is made.
    pub picks: &'a [u32],
    /// The copied tokens, the continuation after the longest earlier match of the context's suffix.
    pub copy: &'a [u32],
    /// How many tokens that match spans (picks included in the context).
    pub match_len: usize,
}

/// The tree of [`best_first`] plus the draft's own picks path and a copy chain off the end of it, every node with its calibrated probability
/// of lying on the accepted path, in order of decreasing probability (a node's probability never exceeds its parent's, so parents come
/// first and a prefix of the order is a tree: the nested allocation of `gb_sched::plan_trees` works on it unchanged). A lattice node's
/// probability is `calibrated(raw)` as for a plain tree; a node of the picks path that the search had not reached has the same, from the path's
/// own summed log probability; the `k`-th copied token's is its parent's times `copy_q(match_len, k)`. Returns the tree, the probabilities
/// (index = node, in the tree's order) and which nodes are copied tokens. Without a usable copy branch this is the plain tree.
///
/// Nothing here changes what is committed: the copied tokens are only more chances for a child to equal the sample drawn at its parent.
pub fn with_copy(
    tree: &DraftTree,
    l: &Lattice,
    keyed: Option<(&Sampling, u64)>,
    p: &Policy,
    mode: Mode,
    branch: Option<CopyBranch>,
) -> (DraftTree, Vec<f64>, Vec<bool>) {
    struct Node {
        parent: i32,
        token: u32,
        depth: u8,
        cand: u8,
        prob: f64,
        copied: bool,
    }
    let mut nodes: Vec<Node> = (0..tree.len())
        .map(|i| Node {
            parent: tree.parents[i],
            token: tree.tokens[i],
            depth: tree.depth[i],
            cand: tree.cand[i],
            prob: calibrated(tree.raw(i), mode),
            copied: false,
        })
        .collect();
    let base_probs: Vec<f64> = nodes.iter().map(|n| n.prob).collect();
    let plain = || (tree.clone(), base_probs.clone(), vec![false; tree.len()]);
    let Some(b) = branch.filter(|b| !b.copy.is_empty() && !b.picks.is_empty() && b.picks.len() <= l.rows) else {
        return plain();
    };
    let mut at: std::collections::HashMap<(i32, u32), usize> =
        nodes.iter().enumerate().map(|(i, n)| ((n.parent, n.token), i)).collect();
    // the picks path: every pick must be a candidate of its level; each node of it that the search has not reached joins with its own probability
    let (mut parent, mut parent_cand, mut neg) = (-1i32, 0usize, 0.0f64);
    for (d, &tok) in b.picks.iter().enumerate() {
        let Some(c) = (0..l.k).find(|&c| l.ids[d * l.k + c] == tok) else { return plain() };
        neg -= level_logp(l, p, keyed, d, parent_cand)[c];
        let idx = match at.get(&(parent, tok)) {
            Some(&i) => i,
            None => {
                nodes.push(Node {
                    parent,
                    token: tok,
                    depth: (d + 1) as u8,
                    cand: c as u8,
                    prob: calibrated((-neg).exp(), mode),
                    copied: false,
                });
                at.insert((parent, tok), nodes.len() - 1);
                nodes.len() - 1
            }
        };
        parent = idx as i32;
        parent_cand = c;
    }
    // a node is never likelier than its parent, EXACTLY: the probabilities are floating point (the picks path's sums, the calibration's knots)
    // and the allocation's contract (a prefix of the order is a tree) needs parents first, which a one-ulp inversion between a certain pick and
    // its parent would break in a release build (`nodes` is in parent-first arrival order, so one forward pass is enough)
    let parent_of: Vec<i32> = nodes.iter().map(|n| n.parent).collect();
    let mut probs_now: Vec<f64> = nodes.iter().map(|n| n.prob).collect();
    cap_at_parent(&mut probs_now, &parent_of);
    for (n, &p) in nodes.iter_mut().zip(&probs_now) {
        n.prob = p;
    }
    let mut prob = nodes[parent as usize].prob;
    let mut depth = b.picks.len();
    for (k, &tok) in b.copy.iter().take(MAX_COPY).enumerate() {
        prob *= copy_q(b.match_len, k);
        depth += 1;
        let idx = match at.get(&(parent, tok)) {
            Some(&i) => i,
            None => {
                nodes.push(Node { parent, token: tok, depth: depth.min(255) as u8, cand: 0, prob, copied: true });
                at.insert((parent, tok), nodes.len() - 1);
                nodes.len() - 1
            }
        };
        parent = idx as i32;
        // a node that was already there keeps its own probability; the chain goes on below it
        prob = nodes[idx].prob;
    }
    // decreasing probability, parents first: ties by depth, then by arrival
    let mut order: Vec<usize> = (0..nodes.len()).collect();
    order.sort_by(|&a, &b| nodes[b].prob.total_cmp(&nodes[a].prob).then(nodes[a].depth.cmp(&nodes[b].depth)).then(a.cmp(&b)));
    let mut place = vec![0usize; nodes.len()];
    for (new, &old) in order.iter().enumerate() {
        place[old] = new;
    }
    let mut out = DraftTree { tokens: vec![], parents: vec![], depth: vec![], cand: vec![], neg_log_p: vec![] };
    let (mut probs, mut copied) = (Vec::with_capacity(nodes.len()), Vec::with_capacity(nodes.len()));
    for &old in &order {
        let n = &nodes[old];
        // a parent has a probability at least its child's and a smaller depth, so it was placed first
        debug_assert!(n.parent < 0 || place[n.parent as usize] < place[old], "node {old} placed before its parent");
        out.tokens.push(n.token);
        out.parents.push(if n.parent < 0 { -1 } else { place[n.parent as usize] as i32 });
        out.depth.push(n.depth);
        out.cand.push(n.cand);
        out.neg_log_p.push(-n.prob.max(f64::MIN_POSITIVE).ln());
        probs.push(n.prob);
        copied.push(n.copied);
    }
    (out, probs, copied)
}

/// What [`with_copy`] returns, cut to the nodes no deeper than `cap`: the tree, and the probabilities and copied flags of exactly the nodes that
/// stay, in the tree's order (a node's parent is shallower, so what stays is a tree). The three vectors are indexed by node: a cut that dropped
/// a node from one and not from the others would price every later node by the wrong probability.
pub fn limit_depth_with(tree: DraftTree, model: Vec<f64>, copied: Vec<bool>, cap: usize) -> (DraftTree, Vec<f64>, Vec<bool>) {
    assert!(model.len() == tree.len() && copied.len() == tree.len());
    let keep: Vec<usize> = (0..tree.len()).filter(|&k| tree.depth[k] as usize <= cap).collect();
    if keep.len() == tree.len() {
        return (tree, model, copied);
    }
    (tree.limit_depth(cap), keep.iter().map(|&k| model[k]).collect(), keep.iter().map(|&k| copied[k]).collect())
}

/// Make no node likelier than its parent (`parent[i] < i`, -1 for a child of the anchor), in one forward pass. Probabilities that come from
/// floating-point sums and a calibration table with knots can invert by an ulp between a near-certain node and its parent; the allocation's
/// contract (a prefix of the order is a tree, a node costs a row and is worth its probability) needs the inversion gone, and a sort by
/// probability would otherwise place the child first.
pub fn cap_at_parent(prob: &mut [f64], parent: &[i32]) {
    for i in 0..prob.len() {
        if parent[i] >= 0 {
            prob[i] = prob[i].min(prob[parent[i] as usize]);
        }
    }
}

/// The selector's own walk through the lattice (greedy: first maximum of `unary + edge`), as a chain: what a plain window would draft.
/// Used to prove the lattice reproduces the walk (the tree's chain through the best candidates is this).
pub fn greedy_chain(l: &Lattice) -> Vec<u32> {
    let k = l.k;
    let (mut parent, mut out) = (0usize, Vec::with_capacity(l.rows));
    for d in 0..l.rows {
        let mut best = 0usize;
        let score = |c: usize| l.unary[d * k + c] + l.edge[(d * k + parent) * k + c];
        for c in 1..k {
            if score(c) > score(best) {
                best = c;
            }
        }
        out.push(l.ids[d * k + best]);
        parent = best;
    }
    out
}

/// Walk the target's samples down a window's tree. `tokens[r]` and `parents[r]` describe row `r` (row 0 is the anchor, parent -1; parents
/// come first), `samples[r]` is the keyed sample drawn from row `r`'s logits at row `r`'s position. From the anchor take the child whose token
/// equals the sample drawn at the node (the first such row if two had the same token), and repeat; stop at a node whose sample matches none
/// of its children. Returns the accepted rows root first: `path[0] == 0`, each the child of the one before it. The committed tokens are
/// `samples` at the path's rows: the accepted drafts, then the sample past them (the first row's sample is the first committed token).
pub fn accept_tree(tokens: &[u32], parents: &[i32], samples: &[u32]) -> Vec<usize> {
    assert!(tokens.len() == parents.len() && tokens.len() == samples.len());
    let mut path = vec![0usize];
    let mut cur = 0usize;
    loop {
        let s = samples[cur];
        let next = (cur + 1..tokens.len()).find(|&r| parents[r] == cur as i32 && tokens[r] == s);
        match next {
            Some(r) => {
                path.push(r);
                cur = r;
            }
            None => return path,
        }
    }
}

/// The calibrated probability that a node whose raw path number is `raw` lies on the accepted path (`measured_tree`, monotone, linear in
/// the raw number between the table's points, flat past its ends, 0 at 0).
pub fn calibrated(raw: f64, mode: Mode) -> f64 {
    let t = match mode {
        Mode::Greedy => crate::measured_tree::GREEDY,
        Mode::Sampled => crate::measured_tree::SAMPLED,
    };
    let raw = raw.clamp(0.0, 1.0);
    if raw <= t[0].0 {
        return t[0].1 * raw / t[0].0;
    }
    for w in t.windows(2) {
        if raw <= w[1].0 {
            let f = (raw - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + f * (w[1].1 - w[0].1);
        }
    }
    t[t.len() - 1].1 + (1.0 - t[t.len() - 1].1) * (raw - t[t.len() - 1].0) / (1.0 - t[t.len() - 1].0).max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic lattice: `rows` levels of `k` candidates with decreasing unary scores and a pseudo-random edge term.
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
        Lattice { rows, k, ids, unary, edge, restricted_head: true }
    }

    #[test]
    fn a_tree_is_best_first_with_parents_before_children_and_nested_prefixes() {
        let l = lattice(7, 7, 16);
        let t = best_first(&l, None, &Policy::default(), 63);
        assert_eq!(t.len(), 63);
        for i in 0..t.len() {
            assert!(t.parents[i] < i as i32, "parents come first");
            let want = if t.parents[i] < 0 { 1 } else { t.depth[t.parents[i] as usize] + 1 };
            assert_eq!(t.depth[i], want);
            // a node is never likelier than its parent (probabilities of siblings are at most 1)
            if t.parents[i] >= 0 {
                assert!(t.neg_log_p[i] >= t.neg_log_p[t.parents[i] as usize] - 1e-12);
            }
        }
        // pops come in non-decreasing -log p, so the first n are the n best nodes
        assert!(t.neg_log_p.windows(2).all(|w| w[0] <= w[1] + 1e-12));
        for n in [1usize, 7, 15, 31] {
            assert_eq!(best_first(&l, None, &Policy::default(), n), t.prefix(n), "a smaller budget is a prefix");
        }
        // no depth past the lattice, siblings are distinct tokens
        assert!(t.depth.iter().all(|&d| d as usize <= l.rows));
        for i in 0..t.len() {
            for j in i + 1..t.len() {
                assert!(!(t.parents[i] == t.parents[j] && t.tokens[i] == t.tokens[j]));
            }
        }
    }

    /// Without the draft's noise or scale, with edge weight 1 and one child per node, the search is the selector's walk.
    #[test]
    fn branch_one_unit_edge_is_the_selectors_chain() {
        for seed in 1..40u64 {
            let l = lattice(seed, 7, 16);
            let p = Policy { edge: 1.0, noise: 0.0, scale: 1.0, branch: 1 };
            let t = best_first(&l, None, &p, 7);
            assert_eq!(t.tokens, greedy_chain(&l), "seed {seed}");
            assert_eq!(t.parents, (0..7).map(|i| i - 1).collect::<Vec<i32>>());
        }
    }

    #[test]
    fn the_keyed_noise_is_the_targets_draw_at_the_candidates_own_position() {
        // a sampled request: the noise of level d is drawn at first_position + d, whatever the path above it
        let l = lattice(3, 7, 16);
        let s = Sampling { seed: 42, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };
        let a = best_first(&l, Some((&s, 100)), &Policy::default(), 20);
        let b = best_first(&l, Some((&s, 100)), &Policy::default(), 20);
        assert_eq!(a, b, "deterministic");
        let moved = best_first(&l, Some((&s, 101)), &Policy::default(), 20);
        assert_ne!(a, moved, "another position draws other noise");
        let greedy = best_first(&l, None, &Policy::default(), 20);
        assert_ne!(a.tokens, greedy.tokens, "noise reorders the candidates");
    }

    /// The shipped search is the study's: real lattices of the S0 dump (greedy and keyed, T=0.7 and 1.0) with the 31-node trees
    /// `tools/tree_ceiling.py` builds on them (`tools/gen_tree_fixtures.py`): same tokens, same parents, same path scores.
    #[test]
    fn best_first_matches_the_python_study_on_real_lattices() {
        let doc: serde_json::Value = serde_json::from_str(include_str!("../../../testdata/tree/best_first.json")).unwrap();
        let p = &doc["policy"];
        let policy = Policy {
            edge: p["edge"].as_f64().unwrap(),
            noise: p["noise"].as_f64().unwrap(),
            scale: p["scale"].as_f64().unwrap(),
            branch: p["branch"].as_u64().unwrap() as usize,
        };
        assert_eq!(policy, Policy::default(), "the fixtures are TensorFold's policy, which is the shipped default");
        let nodes = doc["nodes"].as_u64().unwrap() as usize;
        let cases = doc["cases"].as_array().unwrap();
        assert!(cases.len() >= 6);
        let f = |v: &serde_json::Value| -> Vec<f64> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() };
        let mut sampled = 0;
        for c in cases {
            let l = Lattice {
                rows: 7,
                k: 16,
                ids: c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(),
                unary: f(&c["unary"]).iter().map(|&x| x as f32).collect(),
                edge: f(&c["edge"]).iter().map(|&x| x as f32).collect(),
                restricted_head: true,
            };
            let temperature = c["temperature"].as_f64().unwrap();
            let s = Sampling {
                seed: c["seed"].as_u64().unwrap(),
                temperature,
                top_k: 20,
                top_p: c["top_p"].as_f64().unwrap(),
                min_p: 0.0,
            };
            let keyed = (temperature > 0.0).then_some((&s, c["first_position"].as_u64().unwrap()));
            sampled += keyed.is_some() as usize;
            let t = best_first(&l, keyed, &policy, nodes);
            let name = c["name"].as_str().unwrap();
            let want_tokens: Vec<u32> = c["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
            let want_parents: Vec<i32> = c["parents"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
            assert_eq!(t.tokens, want_tokens, "{name}: tokens");
            assert_eq!(t.parents, want_parents, "{name}: parents");
            for (i, (a, b)) in t.neg_log_p.iter().zip(f(&c["neg_log_p"])).enumerate() {
                assert!((a - b).abs() < 1e-9, "{name}: node {i} path score {a} vs {b}");
            }
        }
        assert!(sampled >= 3, "the fixtures must include keyed (sampled) cases");
    }

    #[test]
    fn limiting_the_depth_keeps_a_valid_tree_in_pop_order() {
        let l = lattice(11, 7, 16);
        let t = best_first(&l, None, &Policy::default(), 40);
        for cap in 0..=7 {
            let c = t.limit_depth(cap);
            assert!(c.depth.iter().all(|&d| d as usize <= cap));
            assert_eq!(c.len(), t.depth.iter().filter(|&&d| d as usize <= cap).count());
            for i in 0..c.len() {
                assert!(c.parents[i] < i as i32);
                if c.parents[i] >= 0 {
                    assert_eq!(c.depth[i], c.depth[c.parents[i] as usize] + 1);
                }
            }
            assert!(c.neg_log_p.windows(2).all(|w| w[0] <= w[1] + 1e-12));
        }
        assert!(t.limit_depth(0).is_empty());
        assert_eq!(t.limit_depth(7), t);
    }

    #[test]
    fn rows_put_the_anchor_first_and_shift_the_parents() {
        let t = DraftTree {
            tokens: vec![11, 12, 13],
            parents: vec![-1, 0, -1],
            depth: vec![1, 2, 1],
            cand: vec![0; 3],
            neg_log_p: vec![0.1, 0.2, 0.3],
        };
        let (tokens, parents) = t.rows(9);
        assert_eq!(tokens, vec![9, 11, 12, 13]);
        assert_eq!(parents, vec![-1, 0, 1, 0]);
        assert_eq!(DraftTree::chain(&[5, 6, 7]).rows(1).1, vec![-1, 0, 1, 2]);
    }

    /// The walk is the chain walk on a chain, and takes the matching child of a fork.
    #[test]
    fn the_accept_walk_follows_the_samples_down_the_tree() {
        // 0 -> 1 (tok 5) -> 3 (tok 7);   0 -> 2 (tok 6) -> 4 (tok 8)
        let tokens = [1u32, 5, 6, 7, 8];
        let parents = [-1i32, 0, 0, 1, 2];
        // the anchor's sample is 6: take row 2; row 2's sample is 8: take row 4; row 4's sample (any) ends the walk
        assert_eq!(accept_tree(&tokens, &parents, &[6, 0, 8, 0, 99]), vec![0, 2, 4]);
        // the anchor's sample matches no child: only the anchor
        assert_eq!(accept_tree(&tokens, &parents, &[9, 0, 0, 0, 0]), vec![0]);
        // a sample that matches a deeper token of ANOTHER branch does not count: row 1's children are row 3 only
        assert_eq!(accept_tree(&tokens, &parents, &[5, 8, 0, 0, 0]), vec![0, 1]);
        // a chain is the plain accept rule
        let chain = [1u32, 2, 3, 4, 5];
        let cp = [-1i32, 0, 1, 2, 3];
        assert_eq!(accept_tree(&chain, &cp, &[2, 3, 9, 0, 0]), vec![0, 1, 2]);
        assert_eq!(accept_tree(&chain, &cp, &[2, 3, 4, 5, 6]), vec![0, 1, 2, 3, 4]);
        // the committed tokens are the samples at the path's rows
        let samples = [2u32, 3, 9, 0, 0];
        let path = accept_tree(&chain, &cp, &samples);
        let committed: Vec<u32> = path.iter().map(|&r| samples[r]).collect();
        assert_eq!(committed, vec![2, 3, 9]);
        // duplicate siblings: the first row wins
        assert_eq!(accept_tree(&[1, 5, 5, 6], &[-1, 0, 0, 2], &[5, 0, 6, 0]), vec![0, 1]);
    }

    #[test]
    fn the_calibration_is_monotone_bounded_and_close_to_the_identity_where_the_raw_number_was_right() {
        for mode in Mode::ALL {
            let mut prev = 0.0;
            for i in 0..=1000 {
                let q = calibrated(i as f64 / 1000.0, mode);
                assert!((0.0..=1.0).contains(&q) && q + 1e-12 >= prev, "{mode:?} at {i}: {q} after {prev}");
                prev = q;
            }
            assert!(calibrated(0.0, mode) < 1e-9 && calibrated(1.0, mode) > 0.97);
        }
        // greedy requests accept more than the raw number says in the middle of its range than sampled ones do: the
        // matched-dump tables (docs/TREE_DRAFTING.md section 13) sit either side of the identity there (0.52 against
        // 0.41 at 0.4). The wide-margin pins that the depth block's own knots past the seventh are in use are in
        // `levels_8_to_15_carry_their_own_calibration_not_level_7s`.
        assert!(calibrated(0.4, Mode::Greedy) > calibrated(0.4, Mode::Sampled) + 0.05);
        assert!(calibrated(0.4, Mode::Greedy) > 0.45 && calibrated(0.4, Mode::Sampled) < 0.45);
    }

    /// The depth block's levels 8..15 are scored by the table's knots past the seventh, not by a ramp extrapolated
    /// from level 7's: on the matched dump's greedy text a deep node's raw path number sits in and above the table's
    /// middle (the conditional acceptance holds 0.93+ to level 15), where the full table reads far above the linear
    /// ramp a `GREEDY[..7]` truncation would draw (0.80 against 0.65 at 0.65; 0.95 against 0.90 at 0.9). This is the
    /// test that kills the `levels 8..15 calibrated from level 7` mutant of `tools/mutation_check.py`.
    #[test]
    fn levels_8_to_15_carry_their_own_calibration_not_level_7s() {
        assert!(calibrated(0.65, Mode::Greedy) > 0.72, "0.65 -> {}", calibrated(0.65, Mode::Greedy));
        assert!(calibrated(0.9, Mode::Greedy) > 0.93, "0.9 -> {}", calibrated(0.9, Mode::Greedy));
        assert!(calibrated(0.65, Mode::Sampled) > 0.55, "0.65 -> {}", calibrated(0.65, Mode::Sampled));
    }

    fn root_path(t: &DraftTree, mut i: usize) -> Vec<u32> {
        let mut v = vec![t.tokens[i]];
        while t.parents[i] >= 0 {
            i = t.parents[i] as usize;
            v.push(t.tokens[i]);
        }
        v.reverse();
        v
    }

    #[test]
    fn without_a_usable_copy_branch_the_tree_is_the_plain_one() {
        let l = lattice(11, 7, 16);
        let t = best_first(&l, None, &Policy::default(), 20);
        let plain: Vec<f64> = (0..t.len()).map(|i| calibrated(t.raw(i), Mode::Greedy)).collect();
        let picks = greedy_chain(&l);
        let not_candidates = [999_999_999u32; 7];
        for b in [
            None,
            Some(CopyBranch { picks: &picks, copy: &[], match_len: 8 }),
            Some(CopyBranch { picks: &[], copy: &[1, 2], match_len: 8 }),
            Some(CopyBranch { picks: &not_candidates, copy: &[1, 2], match_len: 8 }),
        ] {
            let (m, probs, copied) = with_copy(&t, &l, None, &Policy::default(), Mode::Greedy, b);
            assert_eq!(m, t);
            assert_eq!(probs, plain);
            assert!(copied.iter().all(|&c| !c));
        }
    }

    #[test]
    fn a_copy_branch_hangs_off_the_picks_path_and_its_probabilities_only_fall_down_a_path() {
        let l = lattice(11, 7, 16);
        let pol = Policy::default();
        let t = best_first(&l, None, &pol, 20);
        let picks = greedy_chain(&l);
        let copy: Vec<u32> = (0..12).map(|i| 500_000 + i).collect();
        let (m, probs, copied) =
            with_copy(&t, &l, None, &pol, Mode::Greedy, Some(CopyBranch { picks: &picks, copy: &copy, match_len: 12 }));
        assert_eq!((m.len(), copied.len()), (probs.len(), probs.len()));
        // every node of the plain tree is still there, with its probability
        for i in 0..t.len() {
            let path = root_path(&t, i);
            let j = (0..m.len()).find(|&j| root_path(&m, j) == path).expect("a lattice node was lost");
            assert_eq!(probs[j], calibrated(t.raw(i), Mode::Greedy));
            assert!(!copied[j]);
        }
        // parents first, depths one more than the parent's, probabilities never growing down a path, and decreasing overall
        for i in 0..m.len() {
            let par = m.parents[i];
            assert_eq!(m.depth[i] as usize, if par < 0 { 1 } else { m.depth[par as usize] as usize + 1 });
            if par >= 0 {
                assert!((par as usize) < i && probs[par as usize] >= probs[i], "node {i} outweighs its parent");
            }
            if i > 0 {
                assert!(probs[i - 1] >= probs[i], "not in order of decreasing probability at {i}");
            }
        }
        // the copied tokens form one chain whose root path is the picks and then the copy; each probability is the parent's times copy_q
        let n_copied = copied.iter().filter(|&&c| c).count();
        assert_eq!(n_copied, copy.len());
        for k in 0..copy.len() {
            let j = (0..m.len()).find(|&j| copied[j] && m.tokens[j] == copy[k]).unwrap();
            let mut want: Vec<u32> = picks.clone();
            want.extend_from_slice(&copy[..=k]);
            assert_eq!(root_path(&m, j), want);
            assert_eq!(m.depth[j] as usize, picks.len() + k + 1);
            let par = m.parents[j] as usize;
            assert!((probs[j] - probs[par] * copy_q(12, k)).abs() < 1e-12);
        }
        // a smaller budget is a prefix that is still a tree
        for n in [1usize, 5, 17, m.len()] {
            let pre = m.prefix(n);
            for i in 0..pre.len() {
                assert!(pre.parents[i] < i as i32);
            }
        }
    }

    #[test]
    fn a_short_picks_path_takes_the_copy_at_its_own_end_and_the_copy_table_is_by_match_length() {
        let l = lattice(3, 7, 16);
        let pol = Policy::default();
        let t = best_first(&l, None, &pol, 8);
        let full = greedy_chain(&l);
        let picks = &full[..3];
        let (m, _, copied) =
            with_copy(&t, &l, None, &pol, Mode::Sampled, Some(CopyBranch { picks, copy: &[7, 8, 9], match_len: 4 }));
        let j = (0..m.len()).find(|&j| copied[j] && m.tokens[j] == 7).unwrap();
        assert_eq!(m.depth[j], 4, "after the three picks");
        assert_eq!(&root_path(&m, j)[..3], picks);
        // longer matches are more trustworthy; the index past the table clamps
        assert!(copy_q(4, 0) < copy_q(8, 0) && copy_q(8, 0) < copy_q(12, 0) && copy_q(12, 0) < copy_q(40, 0));
        assert_eq!(copy_q(40, MAX_COPY + 5), copy_q(40, MAX_COPY - 1));
        assert!((0.0..=1.0).contains(&copy_q(300, 3)));
    }

    #[test]
    fn no_node_outweighs_its_parent_after_the_cap_even_by_an_ulp() {
        // a child one ulp above its parent, and one far above: both come down to the parent's, a smaller child stays
        let mut p = vec![0.5, 0.5f64.next_up(), 0.9, 0.2, 0.7];
        cap_at_parent(&mut p, &[-1, 0, 1, 1, -1]);
        assert_eq!(p, vec![0.5, 0.5, 0.5, 0.2, 0.7]);
    }

    /// Every level's first candidate overwhelms the others (unary 1000 against 0), so the picks path is certain: log probability exactly 0, and a pick
    /// the search never reached ties its parent's probability to the bit.
    fn certain_lattice(rows: usize) -> Lattice {
        let mut l = lattice(5, rows, 16);
        for d in 0..rows {
            for c in 0..l.k {
                l.unary[d * l.k + c] = if c == 0 { 1000.0 } else { 0.0 };
            }
        }
        l.edge.iter_mut().for_each(|e| *e = 0.0);
        l
    }

    #[test]
    fn certain_picks_tie_with_their_parents_and_every_prefix_of_the_order_is_still_a_tree() {
        let l = certain_lattice(7);
        let pol = Policy::default();
        let s = Sampling { seed: 9, temperature: 0.8, top_k: 20, top_p: 0.95, min_p: 0.0 };
        for (keyed, mode) in [(None, Mode::Greedy), (Some((&s, 100u64)), Mode::Sampled)] {
            // only the first three nodes of the picks path are reached by the search; the other four join as picks
            let t = best_first(&l, keyed, &pol, 3);
            let picks = greedy_chain(&l);
            assert_eq!(picks.len(), 7);
            let copy: Vec<u32> = (0..10).map(|i| 600_000 + i).collect();
            let (m, probs, copied) =
                with_copy(&t, &l, keyed, &pol, mode, Some(CopyBranch { picks: &picks, copy: &copy, match_len: 30 }));
            assert_eq!(copied.iter().filter(|&&c| c).count(), 10);
            for i in 0..m.len() {
                let par = m.parents[i];
                assert!(par < i as i32, "node {i} sits before its parent {par}");
                if par >= 0 {
                    assert!(probs[par as usize] >= probs[i], "node {i} outweighs its parent");
                    assert_eq!(m.depth[i], m.depth[par as usize] + 1);
                }
                if i > 0 {
                    assert!(probs[i - 1] >= probs[i], "not in order of decreasing probability at {i}");
                }
            }
            // the picks path ties down its length: the first seven nodes of the order are the picks, depth ascending
            let tie = calibrated(1.0, mode);
            for d in 0..7 {
                assert_eq!(probs[d], tie, "pick {d}");
                assert_eq!((m.depth[d] as usize, m.tokens[d]), (d + 1, picks[d]));
            }
            for n in 1..=m.len() {
                let pre = m.prefix(n);
                assert!((0..pre.len()).all(|i| pre.parents[i] < i as i32), "prefix {n} is not a tree");
            }
        }
    }

    #[test]
    fn a_depth_cap_keeps_each_node_its_own_probability_and_copied_flag() {
        let l = lattice(11, 7, 16);
        let pol = Policy::default();
        let t = best_first(&l, None, &pol, 30);
        let picks = greedy_chain(&l);
        let copy: Vec<u32> = (0..9).map(|i| 700_000 + i).collect();
        let (m, probs, copied) =
            with_copy(&t, &l, None, &pol, Mode::Greedy, Some(CopyBranch { picks: &picks, copy: &copy, match_len: 8 }));
        assert!(m.depth.iter().any(|&d| d > 7), "the copy goes past the draft's levels");
        for cap in [0usize, 1, 3, 7, 9, 12, 16, 255] {
            let (c, cp, cc) = limit_depth_with(m.clone(), probs.clone(), copied.clone(), cap);
            let want = (0..m.len()).filter(|&i| m.depth[i] as usize <= cap).count();
            assert_eq!((c.len(), cp.len(), cc.len()), (want, want, want), "cap {cap}");
            assert!(c.depth.iter().all(|&d| d as usize <= cap));
            for i in 0..c.len() {
                // the same node: the same root path, probability and flag
                let j = (0..m.len()).find(|&j| root_path(&m, j) == root_path(&c, i)).unwrap();
                assert_eq!((cp[i], cc[i]), (probs[j], copied[j]), "cap {cap}, node {i}");
                assert!(c.parents[i] < i as i32);
            }
        }
    }
}
