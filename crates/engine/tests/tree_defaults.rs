//! The `Backend` defaults of tree windows: a backend that does not verify trees runs chains through `step` and refuses a real tree and a
//! path that is not a prefix, instead of quietly verifying the wrong thing. `Plain` implements only the required methods, so these are
//! the trait's own defaults (the mock overrides them when it is built `with_trees`).

use gb_engine::mock::{Fault, MockBackend};
use gb_engine::{Backend, RowOutput, RowRequest, TreeRequest};

/// The required methods only: `supports_trees`, `step_tree` and `commit_path` keep the trait's defaults.
struct Plain(MockBackend);

impl Backend for Plain {
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.0.prefill(slot, start, tokens, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.0.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.0.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.0.release(slot)
    }
}

/// A backend whose slot 0 holds a 5-token prompt (the windows below start at position 5).
fn ready() -> Plain {
    let mut b = Plain(MockBackend::new(64, 1, Fault::None));
    b.prefill(0, 0, &[1, 2, 3, 4, 5], 4);
    b
}

fn chain(slot: usize, n: usize) -> TreeRequest {
    TreeRequest {
        slot,
        tokens: (0..n as u32).map(|i| 10 + i).collect(),
        parents: (0..n as i32).map(|i| i - 1).collect(),
        first_position: 5,
    }
}

fn fork() -> TreeRequest {
    TreeRequest { slot: 0, tokens: vec![1, 2, 3], parents: vec![-1, 0, 0], first_position: 5 }
}

#[test]
fn a_chain_goes_through_step_and_a_tree_is_refused_by_a_backend_without_trees() {
    let mut b = ready();
    assert!(!b.supports_trees());
    let out = b.step_tree(&[chain(0, 3)], 4);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 3, "one output row per token of the chain");
    assert!(chain(0, 3).is_chain() && !fork().is_chain());
    assert_eq!(fork().depths(), vec![0, 1, 1]);
}

#[test]
#[should_panic(expected = "does not verify trees")]
fn a_real_tree_is_refused() {
    let mut b = ready();
    b.step_tree(&[fork()], 4);
}

#[test]
fn a_prefix_path_commits_as_a_plain_commit() {
    let mut b = ready();
    let _ = b.step_tree(&[chain(0, 4)], 4);
    b.commit_path(0, &[0, 1, 2]);
}

#[test]
#[should_panic(expected = "does not commit a tree path")]
fn a_path_that_skips_rows_is_refused() {
    let mut b = ready();
    let _ = b.step_tree(&[chain(0, 4)], 4);
    b.commit_path(0, &[0, 2]);
}

#[test]
#[should_panic(expected = "does not verify trees")]
fn the_mock_without_trees_refuses_a_real_tree_too() {
    let mut b = MockBackend::new(64, 1, Fault::None);
    b.prefill(0, 0, &[1, 2, 3, 4, 5], 4);
    b.step_tree(&[fork()], 4);
}

#[test]
#[should_panic(expected = "does not commit a tree path")]
fn the_mock_without_trees_refuses_a_path_that_skips_rows() {
    let mut b = MockBackend::new(64, 1, Fault::None);
    b.prefill(0, 0, &[1, 2, 3, 4, 5], 4);
    let _ = b.step_tree(&[chain(0, 4)], 4);
    b.commit_path(0, &[0, 2]);
}
