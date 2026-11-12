//! Input statistics of the quantized linears: H = sum of x^T x per qmap (an input Hessian shared by the linears fed
//! the same activation), the row counts, and the routing counts of each MoE layer.
//!
//! H is accumulated as the fp64 pairwise tree of its per-batch fp32 partials x^T x over batch index, rounded to fp32
//! once. The tree is fixed by the batch count alone ([`Tree`]): node [lo, hi) is node [lo, lo + p) + node [lo + p, hi)
//! with p the largest power of two below hi - lo. A capture streams its batches through a binary counter ([`Stream`]),
//! which builds that tree without knowing the count in advance. A row-sharded capture gives each shard a contiguous
//! batch range; the shard evaluates the tree nodes inside its range ([`Tree::cover`]) and [`Tree::merge`] completes the
//! tree from them, so the merged H has the same bits as one capture over every batch, for any shard count.
//! Row and routing counts are integers and merge by summing.

use std::collections::BTreeMap;

/// Elementwise fp64 addition of partials (host vectors, device buffers).
pub trait Partial: Sized {
    /// self + other, elementwise.
    fn add(self, other: Self) -> Self;
}

impl Partial for Vec<f64> {
    fn add(mut self, other: Self) -> Self {
        assert_eq!(self.len(), other.len(), "partials of different sizes");
        for (a, b) in self.iter_mut().zip(other) {
            *a += b;
        }
        self
    }
}

/// A binary counter over leaves pushed in batch order: the pairwise tree of [`Tree`] over however many were pushed.
pub struct Stream<T> {
    stack: Vec<(usize, T)>,
    leaves: usize,
}

impl<T: Partial> Default for Stream<T> {
    fn default() -> Self {
        Stream { stack: Vec::new(), leaves: 0 }
    }
}

impl<T: Partial> Stream<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, leaf: T) {
        let mut cur = (1usize, leaf);
        while let Some((n, _)) = self.stack.last() {
            if *n != cur.0 {
                break;
            }
            let (n, a) = self.stack.pop().expect("checked");
            cur = (2 * n, a.add(cur.1));
        }
        self.stack.push(cur);
        self.leaves += 1;
    }

    pub fn leaves(&self) -> usize {
        self.leaves
    }

    /// The tree's root (None without leaves).
    pub fn finish(mut self) -> Option<T> {
        let mut tot = self.stack.pop()?.1;
        while let Some((_, a)) = self.stack.pop() {
            tot = a.add(tot);
        }
        Some(tot)
    }
}

/// The fixed tree over `n` batch leaves.
#[derive(Clone, Copy, Debug)]
pub struct Tree {
    pub n: usize,
}

/// A node of the tree: the batch range it sums.
pub type Node = (usize, usize);

fn split(lo: usize, hi: usize) -> usize {
    let len = hi - lo;
    debug_assert!(len >= 2);
    lo + (1usize << (usize::BITS - 1 - (len - 1).leading_zeros()))
}

impl Tree {
    /// The maximal tree nodes inside [a, b), in batch order (the nodes a shard owning those batches evaluates).
    pub fn cover(&self, a: usize, b: usize) -> Vec<Node> {
        assert!(a <= b && b <= self.n, "shard range {a}..{b} outside 0..{}", self.n);
        let mut out = Vec::new();
        if a < b {
            Self::cover_rec(0, self.n, a, b, &mut out);
        }
        out
    }

    fn cover_rec(lo: usize, hi: usize, a: usize, b: usize, out: &mut Vec<Node>) {
        if b <= lo || hi <= a {
            return;
        }
        if a <= lo && hi <= b {
            out.push((lo, hi));
            return;
        }
        let m = split(lo, hi);
        Self::cover_rec(lo, m, a, b, out);
        Self::cover_rec(m, hi, a, b, out);
    }

    /// Evaluates node (lo, hi) from its leaves in order (each covered node is itself a tree of the same rule).
    pub fn eval<T: Partial>(leaves: impl IntoIterator<Item = T>) -> Option<T> {
        let mut s = Stream::new();
        for l in leaves {
            s.push(l);
        }
        s.finish()
    }

    /// The root from the nodes of every shard (exactly a cover of 0..n).
    pub fn merge<T: Partial>(&self, nodes: Vec<(Node, T)>) -> Result<T, String> {
        let mut map: BTreeMap<Node, T> = BTreeMap::new();
        for (k, v) in nodes {
            if map.insert(k, v).is_some() {
                return Err(format!("node {k:?} given twice"));
            }
        }
        if self.n == 0 {
            return Err("an empty tree".into());
        }
        let root = Self::merge_rec(0, self.n, &mut map)?;
        if let Some(k) = map.keys().next() {
            return Err(format!("node {k:?} is not part of the tree over {} batches", self.n));
        }
        Ok(root)
    }

    fn merge_rec<T: Partial>(lo: usize, hi: usize, map: &mut BTreeMap<Node, T>) -> Result<T, String> {
        if let Some(v) = map.remove(&(lo, hi)) {
            return Ok(v);
        }
        if hi - lo == 1 {
            return Err(format!("batch {lo} is in no shard"));
        }
        let m = split(lo, hi);
        let a = Self::merge_rec(lo, m, map)?;
        let b = Self::merge_rec(m, hi, map)?;
        Ok(a.add(b))
    }
}

/// Per-qmap counts and the routing counts of a capture (or a shard of one); merge by summing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// qmap -> (rows accumulated, rows dropped as non-finite)
    pub rows: BTreeMap<String, (u64, u64)>,
    /// tokens routed to each expert (top-k hits) of the captured stream, per layer
    pub routing: BTreeMap<usize, Vec<u64>>,
}

impl Counts {
    pub fn add_rows(&mut self, qmap: &str, rows: u64, dropped: u64) {
        let e = self.rows.entry(qmap.to_string()).or_default();
        e.0 += rows;
        e.1 += dropped;
    }

    pub fn add_routing(&mut self, layer: usize, hits: &[u64]) {
        let e = self.routing.entry(layer).or_insert_with(|| vec![0; hits.len()]);
        assert_eq!(e.len(), hits.len(), "routing counts of different expert counts");
        for (a, b) in e.iter_mut().zip(hits) {
            *a += b;
        }
    }

    pub fn merge(&mut self, other: &Counts) {
        for (q, (r, d)) in &other.rows {
            self.add_rows(q, *r, *d);
        }
        for (l, h) in &other.routing {
            self.add_routing(*l, h);
        }
    }
}

mod device {
    use super::Partial;
    use crate::core::dev::Result;
    use gb_cuda::rt::{DevBuf, Stream as CuStream};

    mod ffi {
        use core::ffi::c_void;
        crate::gpu::foreign! { 100;
            pub fn tq_f32_to_f64(out: *mut f64, x: *const f32, n: i64, stream: *mut c_void) -> i32;
            pub fn tq_add_f64(a: *mut f64, b: *const f64, n: i64, stream: *mut c_void) -> i32;
            pub fn tq_f64_to_f32(out: *mut f32, x: *const f64, n: i64, stream: *mut c_void) -> i32;
        }
    }

    fn ck(rc: i32) {
        assert_eq!(rc, 0, "capture kernel launch failed: CUDA error {rc}");
    }

    /// An fp64 partial on the device (n values), added on `stream`.
    pub struct DevPartial {
        pub buf: DevBuf,
        pub n: usize,
        stream: *mut core::ffi::c_void,
    }

    unsafe impl Send for DevPartial {}

    impl DevPartial {
        /// The fp64 copy of an fp32 device array (a leaf).
        pub(crate) fn from_f32(x: *const f32, n: usize, stream: &CuStream) -> Result<DevPartial> {
            let buf = DevBuf::alloc(n.max(1) * 8)?;
            ck(unsafe { ffi::tq_f32_to_f64(buf.ptr() as *mut f64, x, n as i64, stream.raw()) });
            Ok(DevPartial { buf, n, stream: stream.raw() })
        }

        /// The fp32 rounding into `out` (n values).
        pub(crate) fn round_into(&self, out: *mut f32) {
            ck(unsafe { ffi::tq_f64_to_f32(out, self.buf.ptr() as *const f64, self.n as i64, self.stream) });
        }
    }

    impl Partial for DevPartial {
        fn add(self, other: Self) -> Self {
            assert_eq!(self.n, other.n, "partials of different sizes");
            ck(unsafe { ffi::tq_add_f64(self.buf.ptr() as *mut f64, other.buf.ptr() as *const f64, self.n as i64, self.stream) });
            // `other` is freed after the stream reaches the add: DevBuf frees synchronize the device
            drop(other);
            self
        }
    }
}

pub use device::DevPartial;

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fp32 "partials" with a wide exponent range, as fp64 leaves.
    fn leaves(n: usize, len: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                (0..len)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let m = ((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
                        let e = ((s >> 20) & 31) as i32 - 12;
                        (m * 2f32.powi(e)) as f64
                    })
                    .collect()
            })
            .collect()
    }

    fn bits(v: &[f64]) -> Vec<u32> {
        v.iter().map(|x| (*x as f32).to_bits()).collect()
    }

    fn sharded(tree: Tree, ls: &[Vec<f64>], bounds: &[usize]) -> Vec<f64> {
        let mut nodes = Vec::new();
        for w in bounds.windows(2) {
            for (lo, hi) in tree.cover(w[0], w[1]) {
                nodes.push(((lo, hi), Tree::eval(ls[lo..hi].iter().cloned()).unwrap()));
            }
        }
        tree.merge(nodes).unwrap()
    }

    #[test]
    fn stream_is_the_recursive_tree() {
        fn rec(ls: &[Vec<f64>], lo: usize, hi: usize) -> Vec<f64> {
            if hi - lo == 1 {
                return ls[lo].clone();
            }
            let m = split(lo, hi);
            rec(ls, lo, m).add(rec(ls, m, hi))
        }
        for n in 1..40 {
            let ls = leaves(n, 64, n as u64);
            let s = Tree::eval(ls.iter().cloned()).unwrap();
            let r = rec(&ls, 0, n);
            assert_eq!(s.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), r.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn row_shards_merge_exactly() {
        for n in [1usize, 2, 3, 7, 8, 13, 32, 37] {
            let ls = leaves(n, 4096, 11 + n as u64);
            let tree = Tree { n };
            let one = Tree::eval(ls.iter().cloned()).unwrap();
            for shards in [1usize, 2, 3, 4] {
                if shards > n {
                    continue;
                }
                // even split, then an uneven one
                let even: Vec<usize> = (0..=shards).map(|i| i * n / shards).collect();
                let mut uneven: Vec<usize> = (0..=shards).map(|i| (i * i * n).div_ceil(shards * shards)).collect();
                uneven.dedup();
                for b in [even, uneven] {
                    let m = sharded(tree, &ls, &b);
                    assert_eq!(
                        m.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        one.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "n {n} bounds {b:?}"
                    );
                    assert_eq!(bits(&m), bits(&one));
                }
            }
        }
    }

    #[test]
    fn naive_shard_sums_differ() {
        // the test's sensitivity: per-shard sequential sums merged in shard order do not reproduce the tree's fp64 bits
        let n = 32;
        let ls = leaves(n, 1 << 16, 5);
        let one = Tree::eval(ls.iter().cloned()).unwrap();
        let mut naive = vec![0f64; ls[0].len()];
        for sh in ls.chunks(n / 4) {
            let mut part = vec![0f64; ls[0].len()];
            for l in sh {
                part = part.add(l.clone());
            }
            naive = naive.add(part);
        }
        let b64 = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_ne!(b64(&naive), b64(&one));
    }

    #[test]
    fn merge_rejects_gaps_and_overlaps() {
        let tree = Tree { n: 6 };
        let ls = leaves(6, 4, 1);
        let node = |lo: usize, hi: usize| ((lo, hi), Tree::eval(ls[lo..hi].iter().cloned()).unwrap());
        assert!(tree.merge(vec![node(0, 4)]).unwrap_err().contains("in no shard"));
        assert!(tree.merge(vec![node(0, 4), node(4, 6), node(1, 2)]).unwrap_err().contains("not part of the tree"));
        assert!(tree.merge(vec![node(0, 4), node(4, 6)]).is_ok());
        assert_eq!(tree.cover(1, 6), vec![(1, 2), (2, 4), (4, 6)]);
    }

    #[test]
    fn counts_merge_by_sum() {
        let mut a = Counts::default();
        a.add_rows("q", 10, 1);
        a.add_routing(0, &[1, 2, 3]);
        let mut b = Counts::default();
        b.add_rows("q", 5, 0);
        b.add_rows("r", 2, 0);
        b.add_routing(0, &[3, 2, 1]);
        a.merge(&b);
        assert_eq!(a.rows["q"], (15, 1));
        assert_eq!(a.rows["r"], (2, 0));
        assert_eq!(a.routing[&0], vec![4, 4, 4]);
    }
}
