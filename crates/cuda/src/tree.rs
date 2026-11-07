//! Draft trees in a verify window: the host side of the kernels' contract (no CUDA here).
//!
//! A window of one sequence is a tree of `n` rows: row 0 is the anchor (the token not yet fed), every other row is a drafted token
//! whose parent is an earlier row (`parents[0] = -1`, `0 <= parents[i] < i`: parents come first, the drafter's pop order). A row's
//! context is the committed prefix followed by its ancestors in depth order, so its logical position is `L + depth` and its bits
//! must be those serial decoding gives the token at the end of its root path (docs/TREE_DRAFTING.md). A chain is `parents[i] = i - 1`.
//!
//! What the kernels read, per sequence (rows are LOCAL to the sequence here; the Model adds the sequence's first row):
//!
//! * `depth[i]`: the row's depth (`position = L + depth[i]`; the conv history and the attention keys follow `path(i)`).
//! * the **path table**: row `i`'s root-to-`i` rows, padded with -1 to [`PATH_STRIDE`] (the attention's tail reads its ancestors'
//!   window K/V through it; the GDN's conv gather reads three levels up through the parent array).
//! * the **GDN plan** ([`Plan`]): the order in which the recurrence visits the rows and where each row's input state comes from.
//!   Entries `(node, source, dest)`: `source` is [`SRC_COMMITTED`] for the root (the committed state, the previous round's owed
//!   replay already folded in), [`SRC_PREV`] for "the state the previous entry left in registers" (the row's parent ran just
//!   before it) and `>= 0` for a state saved in that slot; `dest >= 0` saves the state this row leaves for a later sibling. A
//!   depth-first order with the heaviest subtree last needs few slots (a Strahler number: at most `log2(n) + 1`); a slot is
//!   released as soon as its last reader has loaded it.

/// Rows of one tree window. The plan, path and slot tables are sized for it, which is also the widest window `--window-rows` allows (128): the Model sizes
/// its tree buffers by its own window (`forward::tree_buffer_bytes_for`), and the planner caps a tree window at 64 rows within the tree cost table's streams (`gb_sched::RoundCost::tree_window_rows`), at the window past them.
pub const MAX_NODES: usize = 128;
/// i32 entries of one sequence's plan (three per row).
pub const PLAN_STRIDE: usize = 3 * MAX_NODES;
/// i32 entries of one row's path (depth + 1 rows, then -1).
pub const PATH_STRIDE: usize = MAX_NODES;
/// Most recurrent states a plan may keep alive besides the one in registers.
pub const MAX_SLOTS: usize = 32;
/// `source` of the root's entry: the committed state.
pub const SRC_COMMITTED: i32 = -1;
/// `source` of an entry whose parent ran just before it: the state in registers.
pub const SRC_PREV: i32 = -2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeError {
    Empty,
    TooManyNodes(usize),
    /// `parents[0]` must be -1 (the anchor has no parent) and no other row may be a root.
    BadRoot(usize),
    /// `parents[node]` is not an earlier row.
    BadParent {
        node: usize,
        parent: i32,
    },
    /// The plan would keep more than [`MAX_SLOTS`] states alive.
    TooManySlots(usize),
}

impl std::fmt::Display for TreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TreeError::Empty => write!(f, "a tree needs at least its root row"),
            TreeError::TooManyNodes(n) => write!(f, "a tree of {n} rows exceeds the {MAX_NODES} the kernels' tables hold"),
            TreeError::BadRoot(i) => write!(f, "row {i}: only row 0 is the root (parent -1)"),
            TreeError::BadParent { node, parent } => write!(f, "row {node} names parent {parent}, which is not an earlier row"),
            TreeError::TooManySlots(n) => write!(f, "the plan needs {n} saved states; at most {MAX_SLOTS}"),
        }
    }
}

impl std::error::Error for TreeError {}

/// The shape of one sequence's window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tree {
    parents: Vec<i32>,
    depth: Vec<u32>,
    children: Vec<Vec<usize>>,
}

/// The recurrence's visiting order (see the module docs) and the saved states it needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// `(node, source, dest)` per row, in visiting order, flattened (`3 * n` entries).
    pub entries: Vec<i32>,
    /// Slots the plan uses (0 for a chain).
    pub slots: usize,
}

impl Tree {
    pub fn new(parents: &[i32]) -> Result<Self, TreeError> {
        let n = parents.len();
        if n == 0 {
            return Err(TreeError::Empty);
        }
        if n > MAX_NODES {
            return Err(TreeError::TooManyNodes(n));
        }
        let mut depth = vec![0u32; n];
        let mut children = vec![Vec::new(); n];
        for (i, &p) in parents.iter().enumerate() {
            if i == 0 {
                if p != -1 {
                    return Err(TreeError::BadRoot(0));
                }
                continue;
            }
            if p < 0 {
                return Err(TreeError::BadRoot(i));
            }
            if p as usize >= i {
                return Err(TreeError::BadParent { node: i, parent: p });
            }
            depth[i] = depth[p as usize] + 1;
            children[p as usize].push(i);
        }
        Ok(Self { parents: parents.to_vec(), depth, children })
    }

    /// A chain of `n` rows: `parents[i] = i - 1`.
    pub fn chain(n: usize) -> Result<Self, TreeError> {
        let parents: Vec<i32> = (0..n as i32).map(|i| i - 1).collect();
        Self::new(&parents)
    }

    pub fn len(&self) -> usize {
        self.parents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    pub fn parents(&self) -> &[i32] {
        &self.parents
    }

    pub fn depth(&self, i: usize) -> usize {
        self.depth[i] as usize
    }

    pub fn max_depth(&self) -> usize {
        self.depth.iter().copied().max().unwrap_or(0) as usize
    }

    /// Whether every row's parent is the row before it.
    pub fn is_chain(&self) -> bool {
        self.parents.iter().enumerate().all(|(i, &p)| p == i as i32 - 1)
    }

    /// Rows from the root to `i`, inclusive.
    pub fn path(&self, i: usize) -> Vec<usize> {
        let mut p = Vec::with_capacity(self.depth(i) + 1);
        let mut cur = i as i32;
        while cur >= 0 {
            p.push(cur as usize);
            cur = self.parents[cur as usize];
        }
        p.reverse();
        p
    }

    /// The rows a commit keeps when `leaf` is the last accepted row: its path (root first).
    pub fn accepted_path(&self, leaf: usize) -> Vec<usize> {
        self.path(leaf)
    }

    /// The path table of the tree: `len * PATH_STRIDE` entries, row `i`'s root-to-`i` rows then -1.
    pub fn path_table(&self) -> Vec<i32> {
        let mut t = vec![-1i32; self.len() * PATH_STRIDE];
        for i in 0..self.len() {
            for (j, r) in self.path(i).into_iter().enumerate() {
                t[i * PATH_STRIDE + j] = r as i32;
            }
        }
        t
    }

    /// Saved states needed to explore the subtree of `u` depth first when its state is in registers: the heaviest child goes last
    /// (it needs no slot of `u`'s), the others first with `u`'s slot held.
    fn need(&self, u: usize, memo: &mut [Option<usize>]) -> usize {
        if let Some(v) = memo[u] {
            return v;
        }
        let mut needs: Vec<usize> = self.children[u].iter().map(|&c| self.need(c, memo)).collect();
        needs.sort_unstable();
        let v = match needs.split_last() {
            None => 0,
            Some((&last, rest)) => rest.iter().map(|&x| x + 1).max().unwrap_or(0).max(last),
        };
        // a node with several children also holds its own slot while exploring all but the last
        let v = if self.children[u].len() > 1 { v.max(1) } else { v };
        memo[u] = Some(v);
        v
    }

    /// The depth-first plan with the heaviest subtree last (fewest saved states).
    pub fn plan(&self) -> Result<Plan, TreeError> {
        let n = self.len();
        let mut memo = vec![None; n];
        let need: Vec<usize> = (0..n).map(|u| self.need(u, &mut memo)).collect();
        let mut entries = Vec::with_capacity(3 * n);
        let mut live = [false; MAX_SLOTS + 1];
        let mut peak = 0usize;
        // explicit stack: (node, source, slot to release once this entry has been emitted)
        let mut stack: Vec<(usize, i32, i32)> = vec![(0, SRC_COMMITTED, -1)];
        while let Some((u, source, release)) = stack.pop() {
            let mut kids = self.children[u].clone();
            kids.sort_by_key(|&c| need[c]); // lightest first, the heaviest last; stable: ties keep the row order
                                            // the entry loads its source state before it saves its own, so its dest may reuse the slot it reads
            if release >= 0 {
                live[release as usize] = false;
            }
            let dest = if kids.len() > 1 {
                let s = live.iter().position(|&l| !l).unwrap();
                if s >= MAX_SLOTS {
                    return Err(TreeError::TooManySlots(s + 1));
                }
                live[s] = true;
                peak = peak.max(s + 1);
                s as i32
            } else {
                -1
            };
            entries.extend_from_slice(&[u as i32, source, dest]);
            // push in reverse: the first child pops first and runs from registers; later ones read the slot
            let last = kids.len().wrapping_sub(1);
            for (i, &c) in kids.iter().enumerate().rev() {
                let (src, rel) = if i == 0 { (SRC_PREV, -1) } else { (dest, if i == last { dest } else { -1 }) };
                stack.push((c, src, rel));
            }
        }
        debug_assert_eq!(entries.len(), 3 * n);
        Ok(Plan { entries, slots: peak })
    }
}

impl Plan {
    /// Visiting order's `(node, source, dest)` triples.
    pub fn iter(&self) -> impl Iterator<Item = (usize, i32, i32)> + '_ {
        self.entries.chunks_exact(3).map(|e| (e[0] as usize, e[1], e[2]))
    }

    /// The kernel's slot tier for `slots` saved states: the smallest of 0, 1, 2, 4, 8, 16, 32 that holds them.
    pub fn tier(slots: usize) -> usize {
        [0usize, 1, 2, 4, 8, 16, 32].into_iter().find(|&t| t >= slots).unwrap_or(32)
    }

    /// Check the plan against its tree: every row once, the root first from the committed state, and every other row's input state
    /// is its parent's output (in registers right after the parent, or in a slot the parent saved and nothing has overwritten since).
    pub fn validate(&self, t: &Tree) -> Result<(), String> {
        let n = t.len();
        if self.entries.len() != 3 * n {
            return Err(format!("{} entries for {n} rows", self.entries.len() / 3));
        }
        let mut seen = vec![false; n];
        let mut regs: Option<usize> = None; // the node whose output the registers hold
        let mut slot_of: Vec<Option<usize>> = vec![None; MAX_SLOTS + 1];
        for (e, (node, source, dest)) in self.iter().enumerate() {
            if node >= n || seen[node] {
                return Err(format!("entry {e}: row {node} is out of range or visited twice"));
            }
            seen[node] = true;
            let parent = t.parents()[node];
            let input: Option<usize> = match source {
                SRC_COMMITTED => None,
                SRC_PREV => regs,
                s if s >= 0 => slot_of[s as usize],
                _ => return Err(format!("entry {e}: bad source {source}")),
            };
            if node == 0 {
                if source != SRC_COMMITTED || e != 0 {
                    return Err("the root must be the first entry, from the committed state".into());
                }
            } else if input != Some(parent as usize) || source == SRC_COMMITTED {
                return Err(format!("entry {e}: row {node} starts from {input:?}, its parent is {parent}"));
            }
            regs = Some(node);
            if dest >= 0 {
                if dest as usize >= MAX_SLOTS {
                    return Err(format!("entry {e}: slot {dest} out of range"));
                }
                slot_of[dest as usize] = Some(node);
            }
        }
        if seen.iter().any(|&s| !s) {
            return Err("a row is never visited".into());
        }
        Ok(())
    }
}

/// A deterministic random tree of `n` rows (parents first, each row has at most `max_branch` children; deeper rows are likelier parents,
/// like a best-first tree's). For the tests and benches of the tree kernels, which share it. `max_branch = 1` is a chain.
pub fn random_parents(seed: u64, n: usize, max_branch: usize) -> Vec<i32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state >> 33
    };
    let mut kids = vec![0usize; n];
    let mut parents = vec![-1i32];
    for i in 1..n {
        loop {
            let p = (next() % i as u64) as usize;
            let p = if next() % 3 != 0 { i - 1 - (p % i.min(3)) } else { p };
            if kids[p] < max_branch {
                kids[p] += 1;
                parents.push(p as i32);
                break;
            }
        }
    }
    parents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_needs_no_slots_and_runs_in_order() {
        let t = Tree::chain(9).unwrap();
        assert!(t.is_chain());
        let p = t.plan().unwrap();
        assert_eq!(p.slots, 0);
        let want: Vec<(usize, i32, i32)> = (0..9).map(|i| (i, if i == 0 { SRC_COMMITTED } else { SRC_PREV }, -1)).collect();
        assert_eq!(p.iter().collect::<Vec<_>>(), want);
        p.validate(&t).unwrap();
        assert_eq!(t.path(5), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(t.depth(5), 5);
    }

    #[test]
    fn malformed_trees_are_refused() {
        assert_eq!(Tree::new(&[]), Err(TreeError::Empty));
        assert_eq!(Tree::new(&[0]), Err(TreeError::BadRoot(0)));
        assert_eq!(Tree::new(&[-1, -1]), Err(TreeError::BadRoot(1)));
        assert_eq!(Tree::new(&[-1, 1]), Err(TreeError::BadParent { node: 1, parent: 1 }));
        assert_eq!(Tree::new(&[-1, 0, 2]), Err(TreeError::BadParent { node: 2, parent: 2 }));
        assert_eq!(Tree::new(&[-1, 0, 5]), Err(TreeError::BadParent { node: 2, parent: 5 }));
        assert!(matches!(Tree::chain(MAX_NODES + 1), Err(TreeError::TooManyNodes(_))));
        assert!(Tree::chain(MAX_NODES).is_ok());
    }

    #[test]
    fn a_fork_saves_one_state_and_the_heavy_branch_goes_last() {
        // 0 -> {1, 2}, 1 -> {3, 4}, 2 a leaf: the leaf is the light child and goes first, from registers; row 1 (which forks too) reads the
        // root's slot and reuses it for its own save (it loads before it saves), so one slot is enough
        let t = Tree::new(&[-1, 0, 0, 1, 1]).unwrap();
        let p = t.plan().unwrap();
        assert_eq!(p.slots, 1);
        let e: Vec<_> = p.iter().collect();
        assert_eq!(e[0], (0, SRC_COMMITTED, 0), "the root saves its state for the second child");
        assert_eq!(e[1], (2, SRC_PREV, -1), "the light child first, straight from registers");
        assert_eq!(e[2], (1, 0, 0), "the heavy child reads the slot and saves into it");
        assert_eq!(e[3], (3, SRC_PREV, -1));
        assert_eq!(e[4], (4, 0, -1), "the last child reads row 1's save");
        p.validate(&t).unwrap();
    }

    #[test]
    fn random_trees_validate_and_keep_slots_logarithmic() {
        for seed in 0..300u64 {
            for &(n, b) in &[(2usize, 4usize), (7, 2), (8, 4), (16, 4), (31, 3), (64, 4), (64, 2), (128, 4), (100, 8)] {
                let parents = random_parents(seed * 31 + n as u64, n, b);
                let t = Tree::new(&parents).unwrap();
                let p = t.plan().unwrap();
                p.validate(&t).unwrap_or_else(|e| panic!("seed {seed} n {n}: {e}: {parents:?}"));
                // a Strahler bound: at most log2(n) + 1 saved states for a DFS with the heaviest subtree last
                let bound = (n as f64).log2().floor() as usize + 1;
                assert!(p.slots <= bound, "seed {seed} n {n}: {} slots > {bound}: {parents:?}", p.slots);
            }
        }
    }

    #[test]
    fn the_plan_runs_a_toy_recurrence_to_the_per_path_serial_result() {
        // state' = state * 3 + x (mod 2^61 - 1): order sensitive, so a wrong parent or a stale slot changes the value
        let m: u64 = (1 << 61) - 1;
        for seed in 0..200u64 {
            let parents = random_parents(seed, 40, 4);
            let t = Tree::new(&parents).unwrap();
            let p = t.plan().unwrap();
            let x: Vec<u64> = (0..t.len()).map(|i| (i as u64 * 2654435761 + seed * 97 + 11) % m).collect();
            let step = |s: u64, i: usize| (s.wrapping_mul(3) % m + x[i]) % m;
            let mut out = vec![0u64; t.len()];
            let (mut regs, mut slots) = (7u64, vec![0u64; MAX_SLOTS]); // 7: the committed state
            for (node, source, dest) in p.iter() {
                let s = match source {
                    SRC_COMMITTED => 7,
                    SRC_PREV => regs,
                    k => slots[k as usize],
                };
                let s = step(s, node);
                out[node] = s;
                if dest >= 0 {
                    slots[dest as usize] = s;
                }
                regs = s;
            }
            for (i, &got) in out.iter().enumerate().take(t.len()) {
                let serial = t.path(i).into_iter().fold(7u64, &step);
                assert_eq!(got, serial, "seed {seed} row {i}");
            }
        }
    }

    #[test]
    fn the_path_table_lists_root_to_row() {
        let t = Tree::new(&[-1, 0, 0, 1, 1, 3]).unwrap();
        let tab = t.path_table();
        let row =
            |i: usize| tab[i * PATH_STRIDE..(i + 1) * PATH_STRIDE].iter().copied().take_while(|&x| x >= 0).collect::<Vec<_>>();
        assert_eq!(row(0), vec![0]);
        assert_eq!(row(2), vec![0, 2]);
        assert_eq!(row(5), vec![0, 1, 3, 5]);
        assert_eq!(t.max_depth(), 3);
        assert_eq!(Plan::tier(0), 0);
        assert_eq!(Plan::tier(3), 4);
        assert_eq!(Plan::tier(17), 32);
    }

    #[test]
    fn validate_catches_a_stale_slot_a_wrong_source_and_a_wrong_order() {
        // 0 -> {1, 2, 3}: the root saves its state for the second and third child
        let t = Tree::new(&[-1, 0, 0, 0]).unwrap();
        let p = t.plan().unwrap();
        p.validate(&t).unwrap();
        assert_eq!(p.slots, 1);
        // the root does not save: the second child reads an empty slot
        let mut no_save = p.clone();
        no_save.entries[2] = -1;
        assert!(no_save.validate(&t).is_err());
        // the second child starts from registers, which hold its sibling's state, not its parent's
        let mut from_regs = p.clone();
        let k = p.iter().position(|(n, s, _)| n != 0 && s >= 0).unwrap();
        from_regs.entries[3 * k + 1] = SRC_PREV;
        assert!(from_regs.validate(&t).is_err());
        // a slot overwritten before its last reader: the second child saves into the root's slot while the third still needs it
        let t2 = Tree::new(&[-1, 0, 0, 0, 1, 1]).unwrap(); // row 1 forks too
        let p2 = t2.plan().unwrap();
        p2.validate(&t2).unwrap();
        assert!(p2.slots >= 1);
        let mut clobber = p2.clone();
        let root_slot = p2.entries[2];
        let sibling = p2.iter().position(|(n, s, d)| n != 0 && s >= 0 && d < 0).expect("a sibling that reads the root's slot");
        clobber.entries[3 * sibling + 2] = root_slot; // it saves its own state over the root's, which a later sibling still reads
        assert!(clobber.validate(&t2).is_err(), "{:?}", p2.iter().collect::<Vec<_>>());
        // rows visited in the wrong order (a child before its parent)
        let mut wrong = p.clone();
        wrong.entries.swap(0, 3);
        wrong.entries.swap(1, 4);
        wrong.entries.swap(2, 5);
        assert!(wrong.validate(&t).is_err());
    }
}
