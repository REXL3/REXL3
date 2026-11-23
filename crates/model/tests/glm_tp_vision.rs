//! GLM-5.3-Flash IMAGE prompts on a two-rank group ON ONE GPU (`docs/GLM_TP.md` section 10), against one model with no group.
//!
//! * `an_image_prompt_on_two_ranks_equals_one_model_byte_for_byte`: synthetic images through the REAL preprocessing (the pack's
//!   `processor_config.json`, the server's `image_preprocessor`) and the REAL encoder (`GlmEncoder` over the tower of rank 0's map of the
//!   pack); a layer prefix as one model behind the single-node `GlmBackend`, against two share-only ranks (`ExpertOwnership::range(288, r,
//!   2)`, `LoopbackGroup`) behind rank 0's `GlmTpBackend` over a `TpLeader` and a follower thread (`run_follower` over a socket pair). The
//!   follower holds NO tower: it is handed the rows. Each rank's runtime digests the candidates and every fp16 logit of every answering
//!   call; the single node digests the same from its backend's answers and its model's logits. Script, at prefill chunks of 512 rows and of
//!   128 (the deadline row limit): a prompt with two images (one straddling chunk boundaries at 128 rows, both inside one chunk at 512) and
//!   text between and after them, a text prompt in another slot, a mixed window with partial commits and greedy windows. One model with
//!   graphs on == graphs off. Negative controls: (a) one spliced row perturbed by one bf16 ulp in every value (f32 bit 16) on BOTH ranks: the answers differ from
//!   the single node; (b) perturbed on the follower ONLY: rank 0 loses the group at the call whose digest differs.
//! * `the_redsnow_tower_equals_the_225bpw_tower_byte_for_byte` (V0): the same images through the tower of each pack: every feature byte
//!   equal (the 347 tower tensors of the two packs are byte-identical), and the two packs' preprocessing equal.
//!
//! Env: `REXL3_GLM_DIR` (the 2.25bpw pack), `REXL3_REDSNOW_DIR` (V0), `REXL3_GLM_TP_LAYERS` (default 8). Under the GPU lock, alone:
//!
//!     flock -x /tmp/rexl3-gpu.lock env REXL3_GLM_DIR=... cargo test --release -p gb-model --test glm_tp_vision -- --ignored --nocapture --test-threads=1

mod glm_support;

use gb_cuda::moe::ExpertOwnership;
use gb_engine::{Backend, ImageSpan, RowOutput, RowRequest};
use gb_model::error::ModelError;
use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::load::{boot_mapped, LoadOptions};
use gb_model::families::glm::model::{DraftAsk, Drafts, GlmModel, ModelOptions, TapBook};
use gb_model::families::glm::tp::{self, GlmCall, GlmRank, PrefixRuntime, SnapHandles};
use gb_model::families::glm::tp_backend::GlmTpBackend;
use gb_model::families::glm::tp_mtp::GlmMtpRuntime;
use gb_model::families::glm::tp_vision::{self, ImageRuntime, PrefillImage, TpVision};
use gb_model::families::glm::vision::{GlmVisionTower, TowerOptions};
use gb_model::families::glm::vision_serve::{FeatureBudget, GlmEncoder, GlmImageFeatures};
use gb_model::families::glm::GlmFamily;
use gb_model::families::nemotron_h_omni::tp_comm::LoopbackGroup;
use gb_model::forward::Candidates;
use gb_model::lockstep::{run_follower, CallReply, FollowerConfig, FollowerExit, Link, LockstepRuntime, TpLeader, TpTimeouts};
use gb_model::rows::RowBatch;
use gb_model::serve::{ImageEncoder, ServeOptions};
use glm_support::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const CHUNK: usize = 512;
const SLOTS: usize = 3;
const COUNT: usize = 8;
const E: usize = 288;
const OUT: usize = 4096;

type Log = Arc<Mutex<Vec<u64>>>;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
}

/// The digest of an answering call: every candidate (ids, value bits) of its rows in order, then every fp16 logit of those rows.
fn digest(ids: &[i32], vals: &[f32], logits: &[u16]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &i in ids {
        fnv(&mut h, &i.to_le_bytes());
    }
    for &v in vals {
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    for &x in logits {
        fnv(&mut h, &x.to_le_bytes());
    }
    h
}

/// A rank's runtime that digests every answering call (and, for the controls, perturbs the rows it splices).
struct Probe {
    r: GlmRank,
    log: Log,
    /// Flip f32 bit 16 (one bf16 ulp) of every value of the first row of the first splice of every image call.
    perturb: bool,
}

impl Probe {
    fn note(&self, c: &Candidates) {
        let logits = self.r.m.logits_host(0, c.rows).unwrap();
        self.log.lock().unwrap().push(digest(&c.ids, &c.vals, &logits));
    }
}

impl LockstepRuntime for Probe {
    type Call = GlmCall;
    fn vocab_size(&self) -> usize {
        self.r.vocab_size()
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.r.begin_slot(slot)
    }
    fn end_slot(&mut self, slot: usize) {
        self.r.end_slot(slot)
    }
    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        let c = self.r.prefill_chunk(b, want)?;
        if let Some(c) = &c {
            self.note(c);
        }
        Ok(c)
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        let c = self.r.window_candidates(b, count)?;
        self.note(&c);
        Ok(c)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.r.commit(slot, keep)
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        self.r.flush()
    }
    fn apply_call(&mut self, _seq: u64, call: GlmCall) -> Result<CallReply, ModelError> {
        tp::apply_family(self, call)
    }
    fn has_mtp(&self) -> bool {
        self.r.has_mtp()
    }
}

// every runtime of a GLM backend can draft (`GlmTpBackend` is bound on both): this file's model has no NextN layer, so it never does
impl GlmMtpRuntime for Probe {
    fn mtp_book(&self, slot: usize) -> Option<TapBook> {
        self.r.mtp_book(slot)
    }
    fn mtp_draft_keyed(
        &mut self,
        asks: &[DraftAsk<'_>],
        keyed: &[Option<gb_sampling::Sampling>],
    ) -> Result<Vec<Option<Drafts>>, ModelError> {
        self.r.mtp_draft_keyed(asks, keyed)
    }
    fn mtp_disable_slot(&mut self, slot: usize) {
        self.r.mtp_disable_slot(slot)
    }
}

impl ImageRuntime for Probe {
    fn prefill_image(&mut self, call: &PrefillImage) -> Result<Option<Candidates>, ModelError> {
        let c = if self.perturb {
            let mut p = call.clone();
            // one bf16 ulp (f32 bit 16) of every value of the first row: the model reads the f32 stream at half precision, so a single
            // f32 ulp of one value moved none of the 14 answers (measured: the first run of this test)
            for v in 0..OUT {
                p.splice[0].data[4 * v + 2] ^= 1;
            }
            self.r.prefill_image(&p)?
        } else {
            self.r.prefill_image(call)?
        };
        if let Some(c) = &c {
            self.note(c);
        }
        Ok(c)
    }
}

// every runtime of a GLM backend can keep snapshots (`GlmTpBackend` is bound on it): this file's backends run without a prefix cache
impl PrefixRuntime for Probe {
    fn enable_snapshots(&mut self, slots: usize) -> Result<u64, ModelError> {
        self.r.enable_snapshots(slots)
    }
    fn free_snapshots(&self) -> usize {
        self.r.free_snapshots()
    }
    fn p2_anchor(&self, slot: usize) -> Option<usize> {
        self.r.p2_anchor(slot)
    }
    fn alloc_snapshot(&mut self) -> Option<u32> {
        self.r.alloc_snapshot()
    }
    fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<bool, ModelError> {
        self.r.prefix_snapshot(slot, snap, len)
    }
    fn retain_prefix_pages(&mut self, slot: usize, len: usize) -> Vec<u32> {
        self.r.retain_prefix_pages(slot, len)
    }
    fn restore_prefix(&mut self, slot: usize, snap: u32, pages: &[u32], len: usize) -> Result<(), ModelError> {
        self.r.restore_prefix(slot, snap, pages, len)
    }
    fn release_prefix(&mut self, snap: u32, pages: &[u32]) {
        self.r.release_prefix(snap, pages)
    }
    fn seal_snapshot(&mut self, snap: u32, pages: &[u32]) -> Result<u64, ModelError> {
        self.r.seal_snapshot(snap, pages)
    }
    fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, ModelError> {
        self.r.prefix_state_digest(slot, len)
    }
    fn snapshot_handles(&mut self) -> &mut SnapHandles {
        self.r.snapshot_handles()
    }
}

/// A deterministic synthetic RGB image: gradients, a checkerboard and a disc, so the tower sees structure, not noise.
fn synth(width: usize, height: usize, seed: u32) -> gb_vision::Rgb {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f64 / width as f64, y as f64 / height as f64);
            let disc = ((fx - 0.5).powi(2) + (fy - 0.45).powi(2)).sqrt() < 0.25;
            let check = ((x / 32) + (y / 32) + seed as usize) % 2 == 0;
            let r = if disc { 230 } else { (fx * 200.0) as u8 };
            let g = if check { 180 } else { (fy * 160.0) as u8 };
            let b = ((fx + fy) * 100.0) as u8 ^ (seed as u8).wrapping_mul(37);
            data.extend_from_slice(&[r, g, b]);
        }
    }
    gb_vision::Rgb { width, height, data }
}

fn options(dir: &Path) -> ServeOptions {
    let mut o = ServeOptions::new(dir.to_path_buf());
    o.max_image_tokens = 1024;
    o.vision = Some(Default::default());
    o
}

/// The encoder of rank 0's tower (`GlmEncoder` over the tower of `pack`) and the server's preprocessing of `dir`.
fn encoder(
    dir: &Path,
    pack: &gb_model::families::glm::GlmPack,
) -> (GlmEncoder, Arc<dyn gb_vision::preprocessor::ImagePreprocessor>) {
    let o = options(dir);
    let pp = gb_model::families::glm::image_preprocessor(dir, &o).unwrap().expect("the pack has a vision tower");
    let tower = GlmVisionTower::load(pack, dir, &TowerOptions::new(1024 * 4, 8 * GIB)).unwrap();
    (GlmEncoder::new(tower, FeatureBudget::new(4 * GIB), &pp.id()).unwrap(), pp)
}

fn features_bytes(f: &Arc<dyn std::any::Any + Send + Sync>) -> Vec<u8> {
    let g = f.downcast_ref::<GlmImageFeatures>().unwrap();
    let mut out = vec![0u8; g.tokens * tp_vision::ROW_BYTES];
    g.buf.download_range(0, &mut out).unwrap();
    out
}

/// A layer prefix holding `own`'s experts (`None`: every expert).
fn load(dir: &Path, layers: usize, own: Option<&ExpertOwnership>) -> GlmModel {
    load_with(dir, layers, own, CHUNK, 64, 8192)
}

/// [`load`] for prefill chunks of `chunk` rows and prompts of up to 64K tokens.
fn load_chunked(dir: &Path, layers: usize, own: Option<&ExpertOwnership>, chunk: usize) -> GlmModel {
    load_with(dir, layers, own, chunk, 320, 65_536)
}

fn load_with(
    dir: &Path,
    layers: usize,
    own: Option<&ExpertOwnership>,
    chunk: usize,
    kv_blocks: usize,
    max_context: usize,
) -> GlmModel {
    gb_cuda::rt::init_context().unwrap();
    let pack = match own {
        Some(o) => GlmFamily::map_pack_owned(dir, o).unwrap(),
        None => GlmFamily::map_pack(dir).unwrap(),
    };
    let b = boot_mapped(dir, pack, Some(layers), &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    GlmModel::new_chunked(b.pack.config.clone(), b.device, &ModelOptions { slots: SLOTS, kv_blocks, max_context, layers }, chunk)
        .unwrap()
}

struct Prompt {
    tokens: Vec<u32>,
    images: Vec<ImageSpan>,
}

/// text (5) + image A + text (40) + image B + text (30).
fn image_prompt(image_token: u32, a: &gb_model::serve::EncodedImage, b: &gb_model::serve::EncodedImage) -> Prompt {
    let mut t: Vec<u32> = (0..5).map(|i| 1000 + i).collect();
    let mut images = Vec::new();
    for (k, im) in [a, b].into_iter().enumerate() {
        images.push(ImageSpan {
            start: t.len(),
            tokens: im.tokens,
            merged: im.merged,
            features: im.features.clone(),
            key: k as u64,
        });
        t.extend(std::iter::repeat_n(image_token, im.tokens));
        t.extend((0..if k == 0 { 40 } else { 30 }).map(|i| 2000 + 37 * i + k as u32));
    }
    Prompt { tokens: t, images }
}

/// The engine calls of the script on a backend; `answer` sees every answering call (a prefill's last row, a window's rows).
fn script<B: Backend>(be: &mut B, p: &Prompt, rows_limit: Option<usize>, answer: &mut dyn FnMut(&mut B, &[RowOutput])) {
    be.set_prefill_row_limit(rows_limit);
    let mut done = 0;
    let first = loop {
        let s = be.prefill_slice(0, &p.tokens, &p.images, done, COUNT);
        done = s.done;
        if let Some(r) = s.row {
            break r;
        }
    };
    answer(be, std::slice::from_ref(&first));
    let text: Vec<u32> = (0..37).map(|i| 3000 + 11 * i).collect();
    let mut done = 0;
    let second = loop {
        let s = be.prefill_slice(1, &text, &[], done, COUNT);
        done = s.done;
        if let Some(r) = s.row {
            break r;
        }
    };
    answer(be, std::slice::from_ref(&second));
    let (mut pos, mut last) = ([p.tokens.len(), text.len()], [first.ids[0], second.ids[0]]);
    let w = vec![
        RowRequest { slot: 0, tokens: vec![last[0], 11, 12], first_position: pos[0] as u64 },
        RowRequest { slot: 1, tokens: vec![last[1]], first_position: pos[1] as u64 },
    ];
    let out = be.step(&w, COUNT);
    let flat: Vec<RowOutput> = out.iter().flatten().cloned().collect();
    answer(be, &flat);
    for (s, keep) in [(0usize, 2usize), (1, 1)] {
        be.commit(s, keep);
        pos[s] += keep;
        last[s] = out[s][keep - 1].ids[0];
    }
    for _ in 0..4 {
        let w: Vec<RowRequest> =
            (0..2).map(|s| RowRequest { slot: s, tokens: vec![last[s]], first_position: pos[s] as u64 }).collect();
        let out = be.step(&w, COUNT);
        let flat: Vec<RowOutput> = out.iter().flatten().cloned().collect();
        answer(be, &flat);
        for s in 0..2 {
            be.commit(s, 1);
            pos[s] += 1;
            last[s] = out[s][0].ids[0];
        }
    }
    be.release(0);
    be.release(1);
}

fn timeouts() -> TpTimeouts {
    let d = Duration::from_secs(600);
    TpTimeouts { connect: d, ready: d, peer: d, write: d, barrier: d, hang_exit: None, ..TpTimeouts::default() }
}

/// The single node's digests of the script at both chunk grids (the backend's answers and its model's logits).
fn single(be: &mut GlmBackend, p: &Prompt) -> Vec<u64> {
    let mut d = Vec::new();
    for limit in [None, Some(128)] {
        script(be, p, limit, &mut |b: &mut GlmBackend, rows: &[RowOutput]| {
            let ids: Vec<i32> = rows.iter().flat_map(|r| r.ids.iter().map(|&x| x as i32)).collect();
            let vals: Vec<f32> = rows.iter().flat_map(|r| r.values.iter().copied()).collect();
            let logits = b.m.logits_host(0, rows.len()).unwrap();
            d.push(digest(&ids, &vals, &logits));
        });
        assert!(b_ok(be), "the single node failed: {:?}", be.take_error());
    }
    d
}

fn b_ok(be: &mut dyn Backend) -> bool {
    be.take_error().is_none()
}

/// The script at both chunk grids on a group of the two ranks (`perturb`: per rank), with the follower on its own thread. Returns each
/// rank's digests and the leader's failure (`None`: the group served everything).
fn group_run(ranks: [GlmModel; 2], perturb: [bool; 2], image_token: u32, p: &Prompt) -> (Vec<u64>, Vec<u64>, Option<String>) {
    group_run_with(ranks, perturb, image_token, &mut |be| {
        for limit in [None, Some(128)] {
            script(be, p, limit, &mut |_, _| {});
            let e = be.take_error();
            if e.is_some() || be.leader.failure().is_some() {
                return Err(be.leader.failure().unwrap_or_else(|| format!("{e:?}")));
            }
        }
        Ok(())
    })
}

type GroupBackend = GlmTpBackend<Probe>;

/// [`group_run`] with the leader's calls made by `run` (an error: the group failed, why).
fn group_run_with(
    ranks: [GlmModel; 2],
    perturb: [bool; 2],
    image_token: u32,
    run: &mut dyn FnMut(&mut GroupBackend) -> Result<(), String>,
) -> (Vec<u64>, Vec<u64>, Option<String>) {
    let [m0, m1] = ranks;
    let (l0, l1): (Log, Log) = (Arc::default(), Arc::default());
    let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(600)).unwrap();
    let f = Probe { r: GlmRank::new(m1), log: l1.clone(), perturb: perturb[1] };
    let h = std::thread::spawn(move || run_follower(f, b, FollowerConfig::new(timeouts())));
    let leader = TpLeader::lead(Probe { r: GlmRank::new(m0), log: l0.clone(), perturb: perturb[0] }, a, timeouts());
    let mut be = GlmTpBackend::new(leader, CHUNK, SLOTS).with_vision(TpVision::device(Some(image_token)));
    let failure = run(&mut be).err();
    if failure.is_none() {
        be.quiesce().unwrap();
    }
    drop(be);
    match h.join().expect("the follower thread") {
        Ok(FollowerExit::Shutdown(_)) if failure.is_none() => {}
        x => eprintln!("follower exit: {x:?}"),
    }
    let (d0, d1) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
    (d0, d1, failure)
}

fn ranks(dir: &Path, layers: usize, owns: &[ExpertOwnership; 2]) -> [GlmModel; 2] {
    let r = [load(dir, layers, Some(&owns[0])), load(dir, layers, Some(&owns[1]))];
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(600)).into_iter();
    let [mut a, mut b] = r;
    a.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), owns[1].clone()).unwrap();
    assert!(!a.graphs_on() && !b.graphs_on(), "a rank runs eagerly");
    [a, b]
}

#[test]
#[ignore]
fn an_image_prompt_on_two_ranks_equals_one_model_byte_for_byte() {
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];

    // rank 0's tower, from rank 0's map of the pack (the serving path's), and the server's preprocessing
    let pack0 = GlmFamily::map_pack_owned(&dir, &owns[0]).unwrap();
    let image_token = pack0.config.vision.as_ref().unwrap().image_token_id;
    let (mut enc, pp) = encoder(&dir, &pack0);
    let (ia, ib) = (synth(448, 448, 1), synth(336, 224, 2));
    let ea = enc.encode(&pp.preprocess(&ia).unwrap()).unwrap();
    let eb = enc.encode(&pp.preprocess(&ib).unwrap()).unwrap();
    // an image is a function of itself (the tower is deterministic run to run)
    let ea2 = enc.encode(&pp.preprocess(&ia).unwrap()).unwrap();
    assert_eq!(features_bytes(&ea.features), features_bytes(&ea2.features), "the tower is not deterministic");
    drop(ea2);
    drop(enc); // the tower's job is done: the group splices the rows it was given
    let p = image_prompt(image_token, &ea, &eb);
    eprintln!(
        "image prompt: {} tokens; image A {} rows ({}x{}) at {}, image B {} rows ({}x{}) at {}",
        p.tokens.len(),
        ea.tokens,
        ea.merged.0,
        ea.merged.1,
        p.images[0].start,
        eb.tokens,
        eb.merged.0,
        eb.merged.1,
        p.images[1].start
    );
    assert!(
        p.images[0].start < 128 && p.images[0].start + p.images[0].tokens > 256,
        "image A straddles two 128-row chunk boundaries"
    );
    assert!(p.tokens.len() <= CHUNK, "both images inside one 512-row chunk");

    // one model, every expert, behind the single-node backend: graphs on, then off
    let one = load(&dir, layers, None);
    let mut be = GlmBackend::new(one).with_chunk(CHUNK);
    be.m.set_graphs(true);
    let reference = single(&mut be, &p);
    be.m.set_graphs(false);
    let eager = single(&mut be, &p);
    assert_eq!(reference, eager, "one model: graphs and eager differ");
    // the control of the control: text in place of the images (the placeholders' own embeddings) differs
    let text_only = Prompt { tokens: p.tokens.clone(), images: Vec::new() };
    be.m.set_graphs(true);
    let plain = single(&mut be, &text_only);
    assert_ne!(plain[0], reference[0], "the image rows changed nothing: the splice is not seen");
    drop(be);
    eprintln!("one model: {} answering calls, graphs == eager; without the splice the first answer differs", reference.len());

    // two ranks, the follower with no tower, behind rank 0's backend
    let t = std::time::Instant::now();
    let (d0, d1, failure) = group_run(ranks(&dir, layers, &owns), [false, false], image_token, &p);
    assert!(failure.is_none(), "the group failed: {failure:?}");
    eprintln!("two ranks ran the script in {:.1?}", t.elapsed());
    assert_eq!(d0, d1, "the two ranks' answers differ");
    assert_eq!(d0, reference, "the group's answers differ from the one model's");
    eprintln!("BYTE-IDENTICAL: {} answering calls (candidates + every fp16 logit), chunks of 512 and 128 rows", d0.len());

    // control (a): the same one-bf16-ulp change of one spliced row on both ranks: the group agrees with itself and differs from one model
    let (n0, n1, failure) = group_run(ranks(&dir, layers, &owns), [true, true], image_token, &p);
    assert!(failure.is_none(), "{failure:?}");
    assert_eq!(n0, n1);
    let differ = n0.iter().zip(&reference).filter(|(x, y)| x != y).count();
    assert!(differ > 0, "a one-bf16-ulp change of a spliced row went unnoticed");
    eprintln!(
        "control (a), one bf16 ulp of every value of one spliced row, on both ranks: {differ} of {} answering calls differ",
        n0.len()
    );

    // control (b): the change on the follower only: rank 0 loses the group, naming the call
    let (_, _, failure) = group_run(ranks(&dir, layers, &owns), [false, true], image_token, &p);
    let why = failure.expect("a follower that spliced other bytes must lose the group");
    assert!(why.contains("prefill_image") || why.contains("rank 1"), "{why}");
    eprintln!("control (b), the follower alone splices other bytes: the group is lost: {why}");
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

#[test]
#[ignore]
fn the_redsnow_tower_equals_the_225bpw_tower_byte_for_byte() {
    let Some(a_dir) = pack_dir() else { return };
    let Some(b_dir) = std::env::var("REXL3_REDSNOW_DIR").ok().map(PathBuf::from).filter(|p| p.join("config.json").exists())
    else {
        eprintln!("REXL3_REDSNOW_DIR not set: not run");
        return;
    };
    require_available(20 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let pc = |d: &Path| std::fs::read(d.join("processor_config.json")).unwrap();
    eprintln!("processor_config.json byte-identical: {}", pc(&a_dir) == pc(&b_dir));
    let demo = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision/demo_scene.png")).unwrap();
    let run = |dir: &Path| -> (Vec<Vec<u16>>, Vec<Vec<u8>>) {
        let pack = GlmFamily::map_pack(dir).unwrap();
        let (mut enc, pp) = encoder(dir, &pack);
        let imgs = [synth(448, 448, 1), synth(336, 224, 2), pp.decode(&demo).unwrap()];
        let mut patches = Vec::new();
        let mut feats = Vec::new();
        for im in &imgs {
            let pre = pp.preprocess(im).unwrap();
            patches.push(pre.patches.clone());
            feats.push(features_bytes(&enc.encode(&pre).unwrap().features));
        }
        (patches, feats)
    };
    let (pa, fa) = run(&a_dir);
    let (pb, fb) = run(&b_dir);
    assert_eq!(pa, pb, "the two packs' preprocessing gives other patches");
    for (i, (x, y)) in fa.iter().zip(&fb).enumerate() {
        let same = x == y;
        eprintln!("image {i}: {} feature bytes, byte-identical: {same}", x.len());
        assert!(same, "image {i}: the RED-SNOW tower's features differ from the 2.25bpw tower's");
    }
    eprintln!(
        "V0: the RED-SNOW tower's features equal the 2.25bpw tower's for {} images; MemAvailable min {:.1} GiB",
        fa.len(),
        gib(guard.finish())
    );
}

/// Rank 0's boot path for the tower (`tp_vision::load_leader_tower`, then `publish_encoder`, what `tp_boot::build_leader` runs): rank 0
/// serving images loads the tower from its own map of the pack and publishes it as the server's encoder (whose features equal the
/// tower's, encoded directly); a follower, and rank 0 with `--no-vision`, load none.
#[test]
#[ignore]
fn rank_zeros_boot_loads_and_publishes_the_tower_and_a_follower_loads_none() {
    use gb_model::families::glm::tp_vision::{holds_tower, load_leader_tower, publish_encoder};
    use gb_model::serve::TpOptions;
    let Some(dir) = pack_dir() else { return };
    require_available(20 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let d = Duration::from_secs(1);
    let mut o = options(&dir);
    o.tp =
        Some(TpOptions { ranks: 2, rank: 0, peer: "127.0.0.1:1".into(), connect_timeout: d, ready_timeout: d, peer_timeout: d });
    let packs = [0, 1].map(|r| GlmFamily::map_pack_owned(&dir, &ExpertOwnership::range(E, r, 2).unwrap()).unwrap());
    // a follower (with a vision slot it would never get from the server) and rank 0 with --no-vision: no tower
    let mut f = o.clone();
    f.tp.as_mut().unwrap().rank = 1;
    assert!(!holds_tower(&f, &packs[1]) && load_leader_tower(&f, &packs[1], None).unwrap().is_none());
    let mut nv = o.clone();
    nv.vision = None;
    assert!(!holds_tower(&nv, &packs[0]) && load_leader_tower(&nv, &packs[0], None).unwrap().is_none());
    // rank 0: the tower, published
    assert!(holds_tower(&o, &packs[0]));
    let before = mem_available();
    let t = load_leader_tower(&o, &packs[0], None).unwrap().expect("rank 0 loads the tower");
    eprintln!(
        "rank 0's tower: weights {:.3} GiB, scratch {:.3} GiB; MemAvailable fell {:.2} GiB",
        gib(t.weight_bytes()),
        gib(t.scratch_bytes()),
        gib(before.saturating_sub(mem_available()))
    );
    publish_encoder(&o, t, GIB).unwrap();
    let svc = o.vision.as_ref().unwrap().get().expect("the encoder is published").clone();
    let pp = gb_model::families::glm::image_preprocessor(&dir, &o).unwrap().unwrap();
    let pre = pp.preprocess(&synth(336, 224, 2)).unwrap();
    let via_service = features_bytes(&svc.encode(&pre).unwrap().features);
    let (mut enc, _) = encoder(&dir, &packs[0]);
    let direct = features_bytes(&enc.encode(&pre).unwrap().features);
    assert_eq!(via_service, direct, "the published encoder's features are the tower's");
    eprintln!(
        "published encoder == the tower encoded directly ({} bytes); MemAvailable min {:.1} GiB",
        direct.len(),
        gib(guard.finish())
    );
}

/// PASS_RULES_GLM R-G-vision (a) on one GPU: the features of the tower that rank 0's boot path loads from ITS map of the pack (its share
/// of the experts; `load_leader_tower` then `publish_encoder`, what `tp_boot::build_leader` runs) at the image kit's
/// `--max-image-tokens 4096`, against the same binary's tower alone on one node (the whole pack mapped, `GlmVisionTower::load` with the
/// same token limit), byte for byte, on the rule's images: the two synthetic ones, testdata/vision/demo_scene.png and the kit's large
/// image (grad_6000x4000_cap.png, scaled to the limit by the server's preprocessing). `REXL3_GLM_DIR` (the pack).
#[test]
#[ignore]
fn rank_zeros_published_tower_equals_the_single_node_tower_on_the_kit_images() {
    use gb_model::families::glm::tp_vision::{load_leader_tower, publish_encoder};
    use gb_model::serve::TpOptions;
    let Some(dir) = pack_dir() else { return };
    require_available(20 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let d = Duration::from_secs(1);
    let mut o = options(&dir);
    o.max_image_tokens = 4096;
    o.tp =
        Some(TpOptions { ranks: 2, rank: 0, peer: "127.0.0.1:1".into(), connect_timeout: d, ready_timeout: d, peer_timeout: d });
    let pack0 = GlmFamily::map_pack_owned(&dir, &ExpertOwnership::range(E, 0, 2).unwrap()).unwrap();
    let t = load_leader_tower(&o, &pack0, None).unwrap().expect("rank 0 loads the tower");
    let leader_patches = t.max_patches;
    publish_encoder(&o, t, GIB).unwrap();
    let svc = o.vision.as_ref().unwrap().get().expect("the encoder is published").clone();
    let pp = gb_model::families::glm::image_preprocessor(&dir, &o).unwrap().unwrap();
    // one node: the whole pack mapped, a tower for the same token limit
    let whole = GlmFamily::map_pack(&dir).unwrap();
    let tower = GlmVisionTower::load(&whole, &dir, &TowerOptions::new(4096 * 4, 8 * GIB)).unwrap();
    assert_eq!(tower.max_patches, leader_patches, "the two towers are sized for different images");
    let mut single = GlmEncoder::new(tower, FeatureBudget::new(4 * GIB), &pp.id()).unwrap();
    let read =
        |name: &str| std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision").join(name)).unwrap();
    let images = [
        ("synthetic 448x448", synth(448, 448, 1)),
        ("synthetic 336x224", synth(336, 224, 2)),
        ("demo_scene.png", pp.decode(&read("demo_scene.png")).unwrap()),
        ("grad_6000x4000_cap.png", pp.decode(&read("grad_6000x4000_cap.png")).unwrap()),
    ];
    for (name, im) in &images {
        let pre = pp.preprocess(im).unwrap();
        let a = features_bytes(&svc.encode(&pre).unwrap().features);
        let b = features_bytes(&single.encode(&pre).unwrap().features);
        let tokens = a.len() / tp_vision::ROW_BYTES;
        eprintln!(
            "{name}: {}x{} -> grid {:?}, {tokens} tokens, {} feature bytes: rank 0's published tower == the single node's: {}",
            im.width,
            im.height,
            pre.grid,
            a.len(),
            a == b
        );
        assert_eq!(a, b, "{name}: rank 0's tower differs from the single node's");
    }
    eprintln!("R-G-vision (a): {} images byte-identical; MemAvailable min {:.1} GiB", images.len(), gib(guard.finish()));
}

// ------------------------------------------------------------------------------------------- V3: the tower's scratch lent (`lend.rs`)

const MIB: u64 = 1 << 20;

fn text(n: usize, seed: u32) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i.wrapping_mul(7919) ^ seed.wrapping_mul(104_729)) % 60_000).collect()
}

fn kit_image(pp: &dyn gb_vision::preprocessor::ImagePreprocessor, name: &str) -> gb_vision::Rgb {
    pp.decode(&std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision").join(name)).unwrap()).unwrap()
}

fn median_ms(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// V3 (`REXL3_GLM_TOWER_LEND`): with the tower's scratch lent from a model's prefill scratch (`GlmModel::lend_prefill_scratch`), every
/// image's features equal the tower's own scratch's, byte for byte, at the RED-SNOW profile's `--max-image-tokens 4096` and `--prefill-chunk
/// 2048` (all six lendable buffers lent: 512 MiB): the kit's images, one at the cap and one above it (scaled to the cap by the server's
/// preprocessing), each encoded with every lent byte set to 0xFF first and again after a 2,048-row prompt chunk ran through the model on the
/// same bytes. Then at a 512-row chunk, where only part of the scratch fits (the rest stays the tower's own). Prints the scratch the tower
/// allocated, the MemAvailable each tower took and the encode times, own against lent. `REXL3_GLM_DIR` (the pack), one GPU.
#[test]
#[ignore]
fn a_lent_towers_features_equal_its_own_byte_for_byte() {
    use gb_model::families::glm::lend::{LEND_ALIGN, LENT_ITEMS};
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    const CAP: usize = 4096;
    let pack = GlmFamily::map_pack(&dir).unwrap();
    let pp = glm_preprocessor(&dir, CAP);
    let images = [
        ("synthetic 448x448", synth(448, 448, 1)),
        ("synthetic 336x224", synth(336, 224, 2)),
        ("demo_scene.png", kit_image(&*pp, "demo_scene.png")),
        ("grad_6000x4000_cap.png (above the cap)", kit_image(&*pp, "grad_6000x4000_cap.png")),
        ("synthetic 1792x1792 (at the cap)", synth(1792, 1792, 3)),
    ];
    let pres: Vec<gb_vision::Preprocessed> = images.iter().map(|(_, im)| pp.preprocess(im).unwrap()).collect();
    let tokens: Vec<usize> = pres.iter().map(|p| p.n_patches() / 4).collect();
    assert_eq!(tokens.iter().max(), Some(&CAP), "an image at the cap: {tokens:?}");

    // the tower's own scratch: the reference features and times
    let a0 = mem_available();
    let (mut own, own_bytes, none) = glm_encoder(&dir, &pack, CAP, None, false);
    let own_fell = a0.saturating_sub(mem_available());
    assert_eq!(none, 0);
    let want: Vec<Vec<u8>> = pres.iter().map(|p| features_bytes(&own.encode(p).unwrap().features)).collect();
    let time = |enc: &mut GlmEncoder, p: &gb_vision::Preprocessed| {
        median_ms(
            (0..3)
                .map(|_| {
                    let t = std::time::Instant::now();
                    let _ = enc.encode(p).unwrap();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        )
    };
    let own_ms: Vec<f64> = pres.iter().map(|p| time(&mut own, p)).collect();

    // the model at the profile's chunk lends its prefill scratch; the tower borrows from it
    let mut m = load_chunked(&dir, layers, None, 2048);
    let lease = m.lend_prefill_scratch().unwrap().expect("a 2,048-row chunk has a prefill scratch");
    eprintln!("lent regions at 2,048 rows (MiB): {:?}", lease.region_bytes().iter().map(|b| b / MIB).collect::<Vec<_>>());
    let a1 = mem_available();
    let (mut lent, lent_own, lent_bytes) = glm_encoder(&dir, &pack, CAP, Some(lease.clone()), false);
    let lent_fell = a1.saturating_sub(mem_available());
    eprintln!(
        "tower at {CAP} tokens: own scratch {:.1} MiB (MemAvailable fell {:.3} GiB with its weights); lent tower: own scratch {:.1} MiB + lent \
         {:.1} MiB (MemAvailable fell {:.3} GiB): {:.3} GiB less",
        own_bytes as f64 / MIB as f64,
        gib(own_fell),
        lent_own as f64 / MIB as f64,
        lent_bytes as f64 / MIB as f64,
        gib(lent_fell),
        gib(own_fell.saturating_sub(lent_fell))
    );
    assert_eq!(lent_bytes, 512 * MIB, "all six lendable buffers at the cap");
    assert_eq!(own_bytes - lent_own, lent_bytes);
    let st = gb_cuda::Stream::new().unwrap();
    let prompt = text(2048, 7);
    for (i, ((name, im), pre)) in images.iter().zip(&pres).enumerate() {
        lease.fill(0xFF, &st).unwrap();
        st.sync().unwrap();
        let a = features_bytes(&lent.encode(pre).unwrap().features);
        assert!(a == want[i], "{name}: the lent tower's features (lent bytes 0xFF before) differ from its own scratch's");
        m.begin_slot(0).unwrap();
        m.prefill_chunk(0, 0, &prompt, Some(COUNT)).unwrap();
        m.end_slot(0);
        let b = features_bytes(&lent.encode(pre).unwrap().features);
        assert!(b == want[i], "{name}: the lent tower's features after a prompt chunk on the same bytes differ");
        eprintln!(
            "{name}: {}x{} -> grid {:?}, {} tokens, {} feature bytes: lent == own (0xFF before; after a 2,048-row chunk); encode {:.1} ms own, {:.1} ms lent",
            im.width,
            im.height,
            pre.grid,
            tokens[i],
            a.len(),
            own_ms[i],
            time(&mut lent, pre)
        );
    }
    let (taken, waited) = lease.counts();
    eprintln!("lease: taken {taken:?} (model, tower), waited on the device {waited:?}");
    assert!(lease.is_free() && taken[0] > 0 && taken[1] > 0);
    drop(lent);
    drop(m);

    // a 512-row chunk: only part of the scratch fits the prefill scratch; the rest stays the tower's own
    let mut m = load_chunked(&dir, layers, None, 512);
    let lease = m.lend_prefill_scratch().unwrap().unwrap();
    let (mut part, part_own, part_lent) = glm_encoder(&dir, &pack, CAP, Some(lease.clone()), false);
    assert!(part_lent > 0 && part_lent < 512 * MIB, "a partial placement at 512 rows: {part_lent}");
    for (i, pre) in pres.iter().enumerate() {
        lease.fill(0xFF, &st).unwrap();
        st.sync().unwrap();
        assert!(features_bytes(&part.encode(pre).unwrap().features) == want[i], "{}: partial placement", images[i].0);
    }
    eprintln!(
        "512-row chunk: {:.1} MiB of the scratch lent (regions {:?} MiB), {:.1} MiB the tower's own: every image's features == own (some of \
         the {} lendable items fit; offsets on {} MiB boundaries)",
        part_lent as f64 / MIB as f64,
        lease.region_bytes().iter().map(|b| b / MIB).collect::<Vec<_>>(),
        part_own as f64 / MIB as f64,
        LENT_ITEMS.len(),
        LEND_ALIGN / MIB
    );
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// Turn one: `t1` in slot 0, answered, four greedy steps, released. Turn two, in the same slot from scratch (no cache): turn one's prompt,
/// the four tokens it decoded, text, image `c` (a later image of the conversation, at a position past turn one's chunks) and text, answered,
/// four greedy steps. `answer` sees every answering call, as [`script`]'s.
fn turns<B: Backend>(
    be: &mut B,
    t1: &Prompt,
    image_token: u32,
    c: &gb_model::serve::EncodedImage,
    answer: &mut dyn FnMut(&mut B, &[RowOutput]),
) {
    be.set_prefill_row_limit(None);
    let mut run = |be: &mut B, p: &Prompt| -> Vec<u32> {
        let mut done = 0;
        let first = loop {
            let s = be.prefill_slice(0, &p.tokens, &p.images, done, COUNT);
            done = s.done;
            if let Some(r) = s.row {
                break r;
            }
        };
        answer(be, std::slice::from_ref(&first));
        let (mut pos, mut last, mut out) = (p.tokens.len(), first.ids[0], vec![first.ids[0]]);
        for _ in 0..4 {
            let o = be.step(&[RowRequest { slot: 0, tokens: vec![last], first_position: pos as u64 }], COUNT);
            answer(be, &o[0]);
            be.commit(0, 1);
            pos += 1;
            last = o[0][0].ids[0];
            out.push(last);
        }
        be.release(0);
        out
    };
    let decoded = run(be, t1);
    let mut tokens = t1.tokens.clone();
    tokens.extend(&decoded[..4]);
    tokens.extend(text(300, 11));
    let start = tokens.len();
    tokens.extend(std::iter::repeat_n(image_token, c.tokens));
    tokens.extend(text(15, 12));
    let mut images = t1.images.clone();
    images.push(ImageSpan { start, tokens: c.tokens, merged: c.merged, features: c.features.clone(), key: 9 });
    run(be, &Prompt { tokens, images });
}

/// V3 on the group: rank 0's tower borrows its scratch from rank 0's model, and while the group runs the image script (both chunk grids)
/// and a two-turn conversation (an image in the second turn, after a long first turn), another thread encodes images on that tower without
/// pause (what image requests waiting for the slot do beside the engine thread's forwards). Every answering call (candidates and every
/// fp16 logit) of both ranks equals the one model's with the tower's OWN scratch, every concurrent encode gives the own scratch's bytes,
/// and both parties took the lease and waited for each other on the device.
#[test]
#[ignore]
fn an_image_prompt_with_a_lent_tower_on_two_ranks_equals_one_model_byte_for_byte() {
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];
    let pack0 = GlmFamily::map_pack_owned(&dir, &owns[0]).unwrap();
    let image_token = pack0.config.vision.as_ref().unwrap().image_token_id;
    // the reference: the tower's own scratch
    let (mut enc, pp) = encoder(&dir, &pack0);
    let imgs = [synth(448, 448, 1), synth(336, 224, 2), synth(560, 392, 4)];
    let pres: Vec<gb_vision::Preprocessed> = imgs.iter().map(|im| pp.preprocess(im).unwrap()).collect();
    let own: Vec<gb_model::serve::EncodedImage> = pres.iter().map(|p| enc.encode(p).unwrap()).collect();
    drop(enc);
    let p = image_prompt(image_token, &own[0], &own[1]);
    let t1 = Prompt { tokens: p.tokens.clone(), images: p.images.clone() };
    let one = load(&dir, layers, None);
    let mut be = GlmBackend::new(one).with_chunk(CHUNK);
    be.m.set_graphs(false);
    let mut reference = single(&mut be, &p);
    turns(&mut be, &t1, image_token, &own[2], &mut |b: &mut GlmBackend, rows: &[RowOutput]| {
        let ids: Vec<i32> = rows.iter().flat_map(|r| r.ids.iter().map(|&x| x as i32)).collect();
        let vals: Vec<f32> = rows.iter().flat_map(|r| r.values.iter().copied()).collect();
        let logits = b.m.logits_host(0, rows.len()).unwrap();
        reference.push(digest(&ids, &vals, &logits));
    });
    assert!(b_ok(&mut be));
    drop(be);
    eprintln!("one model (the tower's own scratch): {} answering calls", reference.len());

    // the group: rank 0's model lends, the tower borrows; the prompt's images are the lent tower's own encodes
    let mut r = ranks(&dir, layers, &owns);
    let lease = r[0].lend_prefill_scratch().unwrap().expect("rank 0 has a prefill scratch");
    let (mut lent, _, lent_bytes) = glm_encoder(&dir, &pack0, 1024, Some(lease.clone()), false);
    eprintln!("rank 0's tower: {:.1} MiB of its scratch lent", lent_bytes as f64 / MIB as f64);
    assert!(lent_bytes > 0);
    let mine: Vec<gb_model::serve::EncodedImage> = pres.iter().map(|p| lent.encode(p).unwrap()).collect();
    for (a, b) in mine.iter().zip(&own) {
        assert!(features_bytes(&a.features) == features_bytes(&b.features), "a lent encode differs from the own scratch's");
    }
    let lp = image_prompt(image_token, &mine[0], &mine[1]);
    let lt1 = Prompt { tokens: lp.tokens.clone(), images: lp.images.clone() };
    let jobs = pres.iter().cloned().zip(own.iter().map(|e| features_bytes(&e.features))).collect();
    let hammer = Hammer::start(lent, jobs);
    let t = std::time::Instant::now();
    let (d0, d1, failure) = group_run_with(r, [false, false], image_token, &mut |be| {
        for limit in [None, Some(128)] {
            script(be, &lp, limit, &mut |_, _| {});
        }
        turns(be, &lt1, image_token, &mine[2], &mut |_, _| {});
        match (be.take_error(), be.leader.failure()) {
            (None, None) => Ok(()),
            (e, f) => Err(f.unwrap_or_else(|| format!("{e:?}"))),
        }
    });
    let (encodes, bad, _enc) = hammer.finish();
    assert!(failure.is_none(), "the group failed: {failure:?}");
    let (taken, waited) = lease.counts();
    eprintln!(
        "two ranks with a lent tower, {:.1?}: {} answering calls; {encodes} concurrent encodes ({bad} differ); lease taken {taken:?}, waited on \
         the device {waited:?} (model, tower)",
        t.elapsed(),
        d0.len()
    );
    assert_eq!(bad, 0, "a concurrent encode on the lent scratch differs from the own scratch's");
    assert!(encodes > 0 && taken[0] > 0 && taken[1] > 0 && lease.is_free());
    assert_eq!(d0, d1, "the two ranks' answers differ");
    assert_eq!(d0, reference, "the group's answers with a lent tower differ from the one model's");
    eprintln!(
        "BYTE-IDENTICAL with a lent tower and encodes running: {} answering calls (script at 512 and 128 rows, two turns)",
        d0.len()
    );
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The lease is what keeps them apart: a model at the profile's 2,048-row chunk lends its prefill scratch, and a 4,096-token prompt (two
/// chunks) runs again and again while another thread encodes an image at the cap on the borrowing tower. With the lease, every prompt
/// answers as it does alone and every encode gives the own scratch's bytes; with a tower that skips the lease (`unleased`, the bug) on the
/// same bytes, some prompt or encode differs. Then a holder dropped without giving the lease back poisons it: the next encode fails by name.
/// Prints the prompt's time with the lease alone, under contention, and without lending.
#[test]
#[ignore]
fn the_lease_keeps_a_prompt_chunk_and_an_encode_apart_and_skipping_it_corrupts() {
    use gb_model::families::glm::lend::Party;
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    const CAP: usize = 4096;
    let pack = GlmFamily::map_pack(&dir).unwrap();
    let pp = glm_preprocessor(&dir, CAP);
    let pre = pp.preprocess(&synth(1792, 1792, 3)).unwrap();
    let (mut own, _, _) = glm_encoder(&dir, &pack, CAP, None, false);
    let want = features_bytes(&own.encode(&pre).unwrap().features);
    drop(own);
    let prompt = text(4096, 21);
    let mut m = load_chunked(&dir, layers, None, 2048);
    let run = |m: &mut GlmModel| -> (u64, f64) {
        let t = std::time::Instant::now();
        m.begin_slot(0).unwrap();
        m.prefill_chunk(0, 0, &prompt[..2048], None).unwrap();
        let c = m.prefill_chunk(0, 2048, &prompt[2048..], Some(COUNT)).unwrap().unwrap();
        let d = digest(&c.ids, &c.vals, &m.logits_host(0, 1).unwrap());
        m.end_slot(0);
        (d, t.elapsed().as_secs_f64() * 1e3)
    };
    let alone: Vec<(u64, f64)> = (0..3).map(|_| run(&mut m)).collect();
    let reference = alone[0].0;
    assert!(alone.iter().all(|x| x.0 == reference), "the prompt is not deterministic");
    let lease = m.lend_prefill_scratch().unwrap().unwrap();
    let leased_alone: Vec<(u64, f64)> = (0..3).map(|_| run(&mut m)).collect();
    assert!(leased_alone.iter().all(|x| x.0 == reference));

    // with the lease: the prompt and the encodes side by side
    let (lent, _, lent_bytes) = glm_encoder(&dir, &pack, CAP, Some(lease.clone()), false);
    assert_eq!(lent_bytes, 512 * MIB);
    let hammer = Hammer::start(lent, vec![(pre.clone(), want.clone())]);
    let contended: Vec<(u64, f64)> = (0..6).map(|_| run(&mut m)).collect();
    let (encodes, bad, lent) = hammer.finish();
    let (taken, waited) = lease.counts();
    eprintln!(
        "with the lease: {} prompts beside {encodes} encodes: {} prompts and {bad} encodes differ; lease taken {taken:?}, waited {waited:?}",
        contended.len(),
        contended.iter().filter(|x| x.0 != reference).count()
    );
    assert!(encodes > 0 && waited[0] > 0 && waited[1] > 0, "the two parties never met");
    assert_eq!(bad, 0, "an encode under the lease differs");
    assert!(contended.iter().all(|x| x.0 == reference), "a prompt under the lease differs");
    drop(lent);
    eprintln!(
        "prompt (4,096 tokens, two 2,048-row chunks, {layers} layers) median ms: not lent {:.1}, lent alone {:.1}, lent beside encodes {:.1}",
        median_ms(alone.iter().map(|x| x.1).collect()),
        median_ms(leased_alone.iter().map(|x| x.1).collect()),
        median_ms(contended.iter().map(|x| x.1).collect())
    );

    // the bug: a tower on the same bytes that skips the lease
    let (bypass, _, _) = glm_encoder(&dir, &pack, CAP, Some(lease.clone()), true);
    let hammer = Hammer::start(bypass, vec![(pre.clone(), want.clone())]);
    let mut differ = 0;
    let mut tries = 0;
    while tries < 20 && differ == 0 {
        differ += usize::from(run(&mut m).0 != reference);
        tries += 1;
    }
    let (encodes, bad, bypass) = hammer.finish();
    eprintln!("WITHOUT the lease: {tries} prompts beside {encodes} encodes: {differ} prompts and {bad} encodes differ");
    assert!(differ + bad > 0, "skipping the lease corrupted nothing: this test cannot see a missing lease");
    drop(bypass);

    // a holder dropped without giving the lease back: the next encode is refused by name, not run on bytes of unknown state
    let (mut lent, _, _) = glm_encoder(&dir, &pack, CAP, Some(lease.clone()), false);
    let st = gb_cuda::Stream::new().unwrap();
    drop(lease.acquire(Party::Model, &st).unwrap());
    let e = lent.encode(&pre).err().expect("an encode on a poisoned lease").to_string();
    assert!(e.contains("lease"), "{e}");
    eprintln!("a holder dropped without releasing: the next encode fails: {e}");
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// Fairness: a 32,768-token prompt (sixteen 2,048-row chunks, the RED-SNOW profile's chunk) on a model lending its prefill scratch, and an
/// encode at the cap asked for while the prompt's third chunk runs. The model takes the lease per chunk and the lease is first come, first
/// served, so the encode is served at the next chunk boundary: it ends within one chunk's time plus its own encode time of being asked for,
/// with chunks of the prompt still to run, not after the whole prompt. The prompt's answer and the features equal their runs alone. Prints
/// the prompt's chunk times not lent, lent alone and lent with the encode (what the lease costs a chunk).
#[test]
#[ignore]
fn an_encode_asked_for_during_a_long_prompt_is_served_at_the_next_chunk() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    const CAP: usize = 4096;
    const ROWS: usize = 2048;
    const CHUNKS: usize = 16;
    let pack = GlmFamily::map_pack(&dir).unwrap();
    let pre = glm_preprocessor(&dir, CAP).preprocess(&synth(1792, 1792, 3)).unwrap();
    let prompt = text(ROWS * CHUNKS, 33);
    let mut m = load_chunked(&dir, layers, None, ROWS);
    // the prompt chunk by chunk: its answer's digest and each chunk's milliseconds; `done` counts the chunks queued
    let run = |m: &mut GlmModel, done: &AtomicUsize| -> (u64, Vec<f64>) {
        m.begin_slot(0).unwrap();
        let mut ms = Vec::new();
        let mut d = 0;
        for k in 0..CHUNKS {
            let t = std::time::Instant::now();
            let want = (k + 1 == CHUNKS).then_some(COUNT);
            if let Some(c) = m.prefill_chunk(0, k * ROWS, &prompt[k * ROWS..(k + 1) * ROWS], want).unwrap() {
                d = digest(&c.ids, &c.vals, &m.logits_host(0, 1).unwrap());
            }
            m.sync().unwrap();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            done.store(k + 1, Ordering::SeqCst);
        }
        m.end_slot(0);
        (d, ms)
    };
    let (reference, not_lent) = run(&mut m, &AtomicUsize::new(0));
    let lease = m.lend_prefill_scratch().unwrap().unwrap();
    let (d, lent_alone) = run(&mut m, &AtomicUsize::new(0));
    assert_eq!(d, reference);
    let (mut enc, _, lent) = glm_encoder(&dir, &pack, CAP, Some(lease.clone()), false);
    assert_eq!(lent, 512 * MIB);
    let want = features_bytes(&enc.encode(&pre).unwrap().features);
    let alone_ms = median_ms(
        (0..3)
            .map(|_| {
                let t = std::time::Instant::now();
                let _ = enc.encode(&pre).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    );
    let done = AtomicUsize::new(0);
    let ((d, with_encode), (wait_ms, at, got)) = std::thread::scope(|sc| {
        let h = sc.spawn(|| run(&mut m, &done));
        while done.load(Ordering::SeqCst) < 2 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let t = std::time::Instant::now();
        let f = features_bytes(&enc.encode(&pre).unwrap().features);
        let wait_ms = t.elapsed().as_secs_f64() * 1e3;
        let at = done.load(Ordering::SeqCst);
        (h.join().unwrap(), (wait_ms, at, f))
    });
    let chunk_max = not_lent.iter().chain(&lent_alone).cloned().fold(0.0, f64::max);
    eprintln!(
        "{CHUNKS} chunks of {ROWS} rows, {layers} layers: chunk median ms not lent {:.1}, lent alone {:.1}, lent with the encode {:.1} (max {:.1}); \
         the encode asked for after chunk 2: answered in {wait_ms:.1} ms (alone {alone_ms:.1} ms), with {at} of {CHUNKS} chunks queued",
        median_ms(not_lent.clone()),
        median_ms(lent_alone.clone()),
        median_ms(with_encode.clone()),
        chunk_max
    );
    assert_eq!(d, reference, "the prompt's answer with an encode served between its chunks differs");
    assert!(got == want, "the encode served between the prompt's chunks differs");
    assert!(
        wait_ms <= chunk_max + 2.0 * alone_ms + 250.0,
        "the encode waited {wait_ms:.1} ms: longer than one chunk and its own time"
    );
    assert!(at < CHUNKS, "the encode was served only after the whole prompt");
    assert!(lease.is_free());
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// V3, rank 0's boot path with `REXL3_GLM_TOWER_LEND=1` (`tp_vision::load_leader_tower` then `publish_encoder`, what `tp_boot::build_leader`
/// runs) at the RED-SNOW argv's `--max-image-tokens 4096` and `--prefill-chunk 2048`: the plan lends 512 MiB; the boot refuses by name a
/// lend without a lender, and a lender the plan did not ask for (the setting off); with rank 0's model lending, the tower loads with the
/// plan's placement and 512 MiB lent, and the published encoder's features equal the tower's own scratch's. `REXL3_GLM_DIR` (the pack).
/// Sets the process's `REXL3_GLM_TOWER_LEND` while it runs: run this binary with `--test-threads=1`, as every test here.
#[test]
#[ignore]
fn rank_zeros_boot_lends_the_tower_its_scratch_from_its_model_with_the_setting_on() {
    use gb_model::families::glm::lend::TOWER_LEND_ENV;
    use gb_model::families::glm::tp_vision::{load_leader_tower, planned_lend, publish_encoder};
    use gb_model::serve::TpOptions;
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(40 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let d = Duration::from_secs(1);
    let rank0 = || {
        Some(TpOptions { ranks: 2, rank: 0, peer: "127.0.0.1:1".into(), connect_timeout: d, ready_timeout: d, peer_timeout: d })
    };
    let opts = || {
        let mut o = options(&dir);
        (o.max_image_tokens, o.prefill_chunk, o.tp) = (4096, 2048, rank0());
        o
    };
    let own0 = ExpertOwnership::range(E, 0, 2).unwrap();
    let pack0 = GlmFamily::map_pack_owned(&dir, &own0).unwrap();
    std::env::set_var(TOWER_LEND_ENV, "1");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let o = opts();
        let planned = planned_lend(&o, &pack0).unwrap();
        assert_eq!(planned.lent_bytes(), 512 * MIB, "{planned:?}");
        let e = load_leader_tower(&o, &pack0, None).err().expect("a lend without a lender").to_string();
        assert!(e.contains("no prefill scratch to lend"), "{e}");
        let chunk = gb_model::families::glm::alloc::prefill_chunk_for(o.prefill_chunk);
        let mut m = load_chunked(&dir, layers, Some(&own0), chunk);
        let lease = m.lend_prefill_scratch().unwrap().expect("a 2,048-row chunk has a prefill scratch");
        let t = load_leader_tower(&o, &pack0, Some(lease.clone())).unwrap().expect("rank 0 loads the tower");
        assert_eq!(t.lent_scratch_bytes(), 512 * MIB);
        assert_eq!(t.placement(), &planned, "the boot's placement is the plan's");
        publish_encoder(&o, t, GIB).unwrap();
        let svc = o.vision.as_ref().unwrap().get().expect("the encoder is published").clone();
        let pp = glm_preprocessor(&dir, 4096);
        let (mut own, _, _) = glm_encoder(&dir, &pack0, 4096, None, false);
        for im in [synth(1792, 1792, 3), synth(448, 448, 1)] {
            let pre = pp.preprocess(&im).unwrap();
            let a = features_bytes(&svc.encode(&pre).unwrap().features);
            assert!(
                a == features_bytes(&own.encode(&pre).unwrap().features),
                "the published lent encoder differs from the own scratch"
            );
        }
        // the setting off: a lender the plan did not ask for is refused by name
        std::env::set_var(TOWER_LEND_ENV, "0");
        let e = load_leader_tower(&opts(), &pack0, Some(lease.clone())).err().expect("an unplanned lender").to_string();
        assert!(e.contains("the plan did not lend"), "{e}");
        eprintln!("rank 0's boot with the tower lent: 512 MiB lent at the plan's placement, published features == own; both refusals by name");
    }));
    std::env::remove_var(TOWER_LEND_ENV);
    r.unwrap();
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// A layer prefix holding `own`'s experts (`None`: every expert) in prompt chunks of `chunk` rows, under P2, long enough for `context`.
fn load_p2(dir: &Path, layers: usize, own: Option<&ExpertOwnership>, chunk: usize, context: usize) -> GlmModel {
    gb_cuda::rt::init_context().unwrap();
    let pack = match own {
        Some(o) => GlmFamily::map_pack_owned(dir, o).unwrap(),
        None => GlmFamily::map_pack(dir).unwrap(),
    };
    let b = boot_mapped(dir, pack, Some(layers), &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    let mut m = GlmModel::new_chunked(
        b.pack.config.clone(),
        b.device,
        &ModelOptions { slots: SLOTS, kv_blocks: context.div_ceil(256), max_context: context, layers },
        chunk,
    )
    .unwrap();
    m.set_prefill_contract(gb_model::families::glm::contract::PrefillContract::P2).unwrap();
    m
}

/// Under P2 at the shipped 4,096-row chunk, a prompt whose FINAL slice is 4,095 rows, 4,094 of them image rows (90 text tokens, a synthetic
/// image of 90 x 90 = 8,100 rows, 1 text token: 8,191 tokens), does not fit one lockstep frame (4,094 x 16 KiB of features and the batch's
/// ids, positions and slots). Rank 0's backend ends that chunk early. It must end it on the segment's 16-row grid: half of 4,095 is 2,047
/// rows, off the grid, and the chunk after it was refused by name (`GlmModel::p2_segment_of`: "must start a whole number of 16-row KDA
/// chunks") on the builds before this test, failing the request. Now the chunk ends at 2,032 rows (4,096 + 2,032 = 6,128) and the rest
/// follows: three chunks on the group, two on the single node (4,096 + 4,095, no frame). P2 is chunk-invariant, so the group's answers (the prompt's last row and three
/// greedy windows: candidates and every fp16 logit) equal the single node's byte for byte on both ranks.
#[test]
#[ignore]
fn p2_a_final_image_slice_over_one_frame_ends_on_the_grid_and_equals_the_single_node() {
    let Some(dir) = pack_dir() else { return };
    let layers: usize = std::env::var("REXL3_GLM_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    const BIG: usize = 4096;
    const CONTEXT: usize = 12288;
    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];
    let cfg = gb_model::families::glm::GlmConfig::from_dir(&dir).unwrap();
    let image_token = cfg.vision.as_ref().unwrap().image_token_id;
    // a synthetic image's feature rows (f32 [8100][4096], finite, small): the backends splice them as given
    let (grid, before, after) = ((90usize, 90usize), 90usize, 1usize);
    let tokens = grid.0 * grid.1;
    let mut x = 0x9E37_79B9u32;
    let rows: Vec<f32> = (0..tokens * OUT)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 8) as f32 / (1u32 << 24) as f32 * 0.2 - 0.1
        })
        .collect();
    let features: Arc<dyn std::any::Any + Send + Sync> = Arc::new(GlmImageFeatures {
        buf: gb_cuda::DevBuf::from_slice(&rows).unwrap(),
        tokens,
        merged: grid,
        key: 1,
        permit: None,
    });
    drop(rows);
    let mut t: Vec<u32> = (0..before as u32).map(|i| 1000 + 7 * i).collect();
    let span = ImageSpan { start: t.len(), tokens, merged: grid, features, key: 1 };
    t.extend(std::iter::repeat_n(image_token, tokens));
    t.extend((0..after as u32).map(|i| 2000 + i));
    let p = Prompt { tokens: t, images: vec![span] };
    let n = p.tokens.len();
    let last = n - BIG;
    assert!((4091..=4095).contains(&last) && n <= 2 * BIG, "the final slice is {last} rows");
    let image_rows_last = before + tokens - BIG;
    // the call rank 0 would send for the whole final slice, measured as `TpVision::chunk_call` measures it
    let probe = PrefillImage {
        batch: RowBatch::from_requests(&[RowRequest { slot: 0, tokens: p.tokens[BIG..].to_vec(), first_position: BIG as u64 }]),
        want: Some(COUNT as u32),
        splice: vec![tp_vision::SpliceWire { dst_row: 0, n: image_rows_last as u64, data: Vec::new() }],
    };
    let len = probe.encoded_len() + image_rows_last * tp_vision::ROW_BYTES;
    assert!(len > tp_vision::MAX_IMAGE_BODY, "the final slice must not fit one frame ({len} bytes)");
    eprintln!("prompt {n} tokens: final slice {last} rows, {image_rows_last} of them image rows");

    // one model, every expert, no group (no frame: two slices), then the group (rank 0's backend over a leader, the follower on a thread)
    let mut be = GlmBackend::new(load_p2(&dir, layers, None, BIG, CONTEXT)).with_chunk(BIG);
    be.m.set_graphs(false);
    let mut reference = Vec::new();
    let ends = run_big(&mut be, &p, &mut |b: &mut GlmBackend, r: &RowOutput| {
        let ids: Vec<i32> = r.ids.iter().map(|&x| x as i32).collect();
        reference.push(digest(&ids, &r.values, &b.m.logits_host(0, 1).unwrap()));
    });
    assert!(be.take_error().is_none(), "the single node failed");
    assert_eq!(ends, vec![BIG, n], "the single node's slices");
    drop(be);
    eprintln!("one model: slices end at {ends:?}; {} answering calls", reference.len());

    let [mut m0, mut m1] =
        [load_p2(&dir, layers, Some(&owns[0]), BIG, CONTEXT), load_p2(&dir, layers, Some(&owns[1]), BIG, CONTEXT)];
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(600)).into_iter();
    m0.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    m1.enable_tp(Arc::new(comms.next().unwrap()), owns[1].clone()).unwrap();
    let (l0, l1): (Log, Log) = (Arc::default(), Arc::default());
    let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(600)).unwrap();
    let f = Probe { r: GlmRank::new(m1), log: l1.clone(), perturb: false };
    let h = std::thread::spawn(move || run_follower(f, b, FollowerConfig::new(timeouts())));
    let leader = TpLeader::lead(Probe { r: GlmRank::new(m0), log: l0.clone(), perturb: false }, a, timeouts());
    let mut be = GlmTpBackend::new(leader, BIG, SLOTS)
        .with_vision(TpVision::device(Some(image_token)))
        .with_contract(gb_model::families::glm::contract::PrefillContract::P2);
    let ends = run_big(&mut be, &p, &mut |_, _| {});
    let err = be.take_error();
    let failure = be.leader.failure();
    assert!(
        err.is_none() && failure.is_none(),
        "the group failed the request after slices ending at {ends:?}: {err:?} {failure:?}"
    );
    be.quiesce().unwrap();
    drop(be);
    assert!(matches!(h.join().expect("the follower thread"), Ok(FollowerExit::Shutdown(_))), "the follower ended cleanly");
    // the final slice ended early on the grid (half of 4,093 rows rounded down to 16), then the rest
    assert_eq!(ends, vec![BIG, BIG + 2032, n], "the group's slices");
    assert!(ends.iter().take(ends.len() - 1).all(|e| e % 16 == 0), "{ends:?}");
    let (d0, d1) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
    assert_eq!(d0, d1, "the two ranks' answers differ");
    assert_eq!(d0, reference, "P2: the group's three chunks differ from the single node's two");
    eprintln!(
        "BYTE-IDENTICAL: the group's slices {ends:?} (the final image slice ended on the 16-row grid) == the single node's [{BIG}, {n}]: {} answering calls",
        d0.len()
    );
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The prompt `p` in slot 0 and three greedy windows on `be`; `answer` sees every answering call's row. Returns where each slice ended
/// (stops at a slice that made no progress: the backend failed the request).
fn run_big<B: Backend>(be: &mut B, p: &Prompt, answer: &mut dyn FnMut(&mut B, &RowOutput)) -> Vec<usize> {
    let n = p.tokens.len();
    let mut ends = Vec::new();
    let mut done = 0;
    let first = loop {
        let s = be.prefill_slice(0, &p.tokens, &p.images, done, COUNT);
        if s.done <= done {
            break None;
        }
        done = s.done;
        ends.push(done);
        if let Some(r) = s.row {
            break Some(r);
        }
    };
    let Some(first) = first else { return ends };
    answer(be, &first);
    let (mut pos, mut tok) = (n, first.ids[0]);
    for _ in 0..3 {
        let out = be.step(&[RowRequest { slot: 0, tokens: vec![tok], first_position: pos as u64 }], COUNT);
        answer(be, &out[0][0]);
        be.commit(0, 1);
        pos += 1;
        tok = out[0][0].ids[0];
    }
    be.release(0);
    ends
}
