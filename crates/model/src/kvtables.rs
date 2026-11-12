//! Host-side bookkeeping for the paged KV cache: which pages each sequence slot owns.
//!
//! Pages are 256 tokens. A slot's table lists its pages in position order. Pages are taken from a shared pool as the
//! sequence grows and returned when it ends. The device sees a flat `[slots][max_blocks]` i32 table, -1 for unused.

use gb_sched::{BlockId, BlockPool};

pub const BLOCK_TOKENS: usize = 256;

pub struct KvTables {
    pool: BlockPool,
    tables: Vec<Vec<BlockId>>,
    pub max_blocks: usize,
    /// Per page, how many times it has been handed out ([`KvTables::generation`]).
    gens: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum KvError {
    /// The pool has no free pages for this growth (callers pre-reserve via admission; this is the backstop).
    OutOfPages {
        slot: usize,
        needed: usize,
        free: usize,
    },
    ContextTooLong {
        slot: usize,
        tokens: usize,
        max: usize,
    },
}

impl KvTables {
    pub fn new(n_blocks: usize, slots: usize, max_blocks: usize) -> Self {
        Self {
            pool: BlockPool::new(n_blocks, BLOCK_TOKENS),
            tables: vec![Vec::new(); slots],
            max_blocks,
            gens: vec![0; n_blocks],
        }
    }

    /// How many times `block` has been taken from the pool. A page a prefix-cache entry holds is never freed, so its generation does not move
    /// while the entry lives; a page freed and handed out again under an entry that still names it (a reference-count bug) shows a new
    /// generation, which the entry's content digest catches.
    pub fn generation(&self, block: BlockId) -> u32 {
        self.gens.get(block as usize).copied().unwrap_or(0)
    }

    fn handed_out(&mut self, blocks: &[BlockId]) {
        for &b in blocks {
            if let Some(g) = self.gens.get_mut(b as usize) {
                *g = g.wrapping_add(1);
            }
        }
    }

    pub fn free_blocks(&self) -> usize {
        self.pool.free_blocks()
    }

    pub fn blocks_of(&self, slot: usize) -> &[BlockId] {
        &self.tables[slot]
    }

    /// Make sure `slot` has pages for positions `0..tokens`. Atomic: on failure nothing changes.
    pub fn ensure(&mut self, slot: usize, tokens: usize) -> Result<(), KvError> {
        let need = tokens.div_ceil(BLOCK_TOKENS);
        if need > self.max_blocks {
            return Err(KvError::ContextTooLong { slot, tokens, max: self.max_blocks * BLOCK_TOKENS });
        }
        let have = self.tables[slot].len();
        if need <= have {
            return Ok(());
        }
        match self.pool.alloc(need - have) {
            Some(b) => {
                self.handed_out(&b);
                self.tables[slot].extend(b);
                Ok(())
            }
            None => Err(KvError::OutOfPages { slot, needed: need - have, free: self.pool.free_blocks() }),
        }
    }

    /// Take a reference to the first `pages` pages of `slot`'s table (a prefix-cache entry keeps them alive after the slot is gone) and return
    /// them. The slot keeps its own references; a page is returned to the pool when the last holder lets go.
    pub fn retain_prefix(&mut self, slot: usize, pages: usize) -> Vec<BlockId> {
        let blocks: Vec<BlockId> = self.tables[slot][..pages].to_vec();
        self.pool.retain(&blocks);
        blocks
    }

    /// Drop references taken by [`KvTables::retain_prefix`] (an entry was evicted, a capture was discarded).
    pub fn release_pages(&mut self, blocks: &[BlockId]) {
        self.pool.release(blocks);
    }

    /// Start `slot`'s table with pages another holder owns: one more reference to each, in position order. The slot must be empty. The caller
    /// guarantees nothing writes to those pages through this slot (a prefix's full pages are never written again).
    pub fn adopt_shared(&mut self, slot: usize, blocks: &[BlockId]) {
        assert!(self.tables[slot].is_empty(), "adopting pages into a slot that holds some");
        self.pool.retain(blocks);
        self.tables[slot].extend_from_slice(blocks);
    }

    /// Append one fresh page to `slot`'s table and return it (the copy-on-write page of an adopted prefix). Fails when the pool is empty.
    pub fn alloc_block(&mut self, slot: usize) -> Result<BlockId, KvError> {
        if self.tables[slot].len() >= self.max_blocks {
            return Err(KvError::ContextTooLong {
                slot,
                tokens: (self.tables[slot].len() + 1) * BLOCK_TOKENS,
                max: self.max_blocks * BLOCK_TOKENS,
            });
        }
        match self.pool.alloc(1) {
            Some(b) => {
                self.handed_out(&b);
                self.tables[slot].push(b[0]);
                Ok(b[0])
            }
            None => Err(KvError::OutOfPages { slot, needed: 1, free: 0 }),
        }
    }

    /// References held on `block` (all slots and entries).
    pub fn refcount(&self, block: BlockId) -> u32 {
        self.pool.refcount(block)
    }

    /// Give back every page of `slot`.
    pub fn release(&mut self, slot: usize) {
        let blocks = std::mem::take(&mut self.tables[slot]);
        self.pool.release(&blocks);
    }

    /// The device table: `[slots][max_blocks]`, unused entries -1.
    pub fn flat(&self) -> Vec<i32> {
        let mut v = vec![-1i32; self.tables.len() * self.max_blocks];
        for (s, t) in self.tables.iter().enumerate() {
            for (i, &b) in t.iter().enumerate() {
                v[s * self.max_blocks + i] = b as i32;
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_grow_and_return() {
        let mut t = KvTables::new(10, 3, 8);
        t.ensure(0, 1).unwrap();
        assert_eq!(t.blocks_of(0).len(), 1);
        t.ensure(0, 256).unwrap();
        assert_eq!(t.blocks_of(0).len(), 1, "256 tokens still fit one page");
        t.ensure(0, 257).unwrap();
        assert_eq!(t.blocks_of(0).len(), 2);
        t.ensure(1, 1000).unwrap(); // 4 pages
        assert_eq!(t.free_blocks(), 10 - 2 - 4);
        let flat = t.flat();
        assert_eq!(flat.len(), 3 * 8);
        assert_eq!(flat[2..8], [-1; 6]);
        assert_eq!(flat[8 + 4..16], [-1; 4]);
        // pages are distinct across slots
        let mut seen: Vec<i32> = flat.iter().copied().filter(|&b| b >= 0).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 6);
        t.release(0);
        assert_eq!(t.free_blocks(), 10 - 4);
        assert!(t.blocks_of(0).is_empty());
    }

    #[test]
    fn shared_pages_live_until_their_last_holder_lets_go() {
        let mut t = KvTables::new(8, 3, 8);
        t.ensure(0, 600).unwrap(); // 3 pages
        let kept = t.retain_prefix(0, 3); // an entry's references
        assert_eq!(kept.iter().map(|&b| t.refcount(b)).collect::<Vec<_>>(), vec![2, 2, 2]);
        // another sequence adopts the first two pages and gets a page of its own where the prefix's last page was partial
        t.adopt_shared(1, &kept[..2]);
        let fresh = t.alloc_block(1).unwrap();
        assert!(!kept.contains(&fresh));
        assert_eq!(t.blocks_of(1).len(), 3);
        assert_eq!((t.refcount(kept[0]), t.refcount(kept[1]), t.refcount(kept[2])), (3, 3, 2));
        // the donor goes away: nothing is freed, the entry and the other sequence still hold the pages
        t.release(0);
        assert_eq!((t.refcount(kept[0]), t.refcount(kept[1]), t.refcount(kept[2])), (2, 2, 1));
        assert_eq!(t.free_blocks(), 8 - 4);
        // the entry is evicted while the other sequence still runs: the page only the entry held goes back, the pages the sequence uses stay
        t.release_pages(&kept);
        assert_eq!((t.refcount(kept[0]), t.refcount(kept[1]), t.refcount(kept[2])), (1, 1, 0));
        assert_eq!(t.free_blocks(), 8 - 3);
        // the sequence ends: everything is back
        t.release(1);
        assert_eq!(t.free_blocks(), 8);
    }

    #[test]
    fn failures_change_nothing() {
        let mut t = KvTables::new(3, 2, 4);
        t.ensure(0, 512).unwrap(); // 2 pages
        assert_eq!(t.ensure(1, 512), Err(KvError::OutOfPages { slot: 1, needed: 2, free: 1 }));
        assert!(t.blocks_of(1).is_empty());
        assert_eq!(t.free_blocks(), 1);
        assert_eq!(t.ensure(1, 4 * 256 + 1), Err(KvError::ContextTooLong { slot: 1, tokens: 1025, max: 1024 }));
    }

    /// A page's generation moves each time the pool hands it out, never while a holder keeps it: a page an entry retained keeps its
    /// generation as slots come and go, and one freed and handed out again shows a new one (what a prefix-cache digest reads).
    #[test]
    fn a_pages_generation_moves_only_when_it_is_handed_out_again() {
        let mut t = KvTables::new(2, 2, 4);
        t.ensure(0, 512).unwrap();
        let pages = t.blocks_of(0).to_vec();
        let g: Vec<u32> = pages.iter().map(|&b| t.generation(b)).collect();
        assert_eq!(g, vec![1, 1]);
        let kept = t.retain_prefix(0, 1);
        t.release(0);
        t.adopt_shared(1, &kept);
        assert_eq!(t.generation(kept[0]), 1, "retained and adopted: the same generation");
        t.release(1);
        t.release_pages(&kept);
        t.ensure(0, 512).unwrap();
        assert!(t.blocks_of(0).iter().all(|&b| t.generation(b) == 2), "freed and handed out again: a new generation");
        assert_eq!(t.alloc_block(1), Err(KvError::OutOfPages { slot: 1, needed: 1, free: 0 }));
        assert_eq!(t.generation(99), 0, "a page that does not exist");
    }
}
