//! B7 of `docs/CLUSTER_N_NODE_PLAN_2026-10-11.md`: balanced, exact expert ownership for tensor parallelism.
//!
//! Which rank holds a routed expert's weights is a pure lookup: every other rank holds [`Proj::Absent`](gb_cuda::nemotron_moe::Proj)
//! for it and the MoE kernels write exact zeros for its pairs, the disjoint support the group's sum needs. So ANY assignment of the
//! experts to the ranks is bit-exact, and the contiguous ranges `[E r / R, E (r + 1) / R)` the engine started with are only a default.
//! They are not a good one: the routing is skewed (rank 1 of two receives 52 to 53 percent of the routed pairs) and every collective
//! waits for the slower rank, a straggler that grows with the rank count.
//!
//! This module holds the three pieces of the balanced map:
//!
//! 1. [`ExpertMap`]: the ownership table (per trunk layer, expert -> rank), its versioned text file and its 64-bit content hash, and
//!    the process's configured map (`REXL3_TP_EXPERT_MAP`, [`configured_map`]). Absent means today's contiguous ownership, bit for bit.
//!    The shard plan consults the table instead of the range when one is present (`tp_shard::ShardPlan::with_expert_map`).
//!    [`expert_map_fingerprint`] is what the rendezvous compares: both ranks must hold the same map or one would wait for pairs the
//!    other computed.
//! 2. The routing trace ([`RoutingTrace`]): the routed expert ids of every row of every MoE layer, recorded from the real model by the
//!    gated test `tests/nemotron_omni_routing.rs` through [`tap_enable`] and read back by [`tap_take`].
//! 3. The solver ([`solve`]): from traces, per layer, an assignment with the contiguous plan's counts per rank (so every memory plan
//!    number stays) that minimizes the sum over sampled steps of the busiest rank's load, by greedy LPT and a swap local search, and
//!    [`evaluate`], which scores any assignment (busiest over mean) on any set of steps.
//!
//! **What is balanced.** A step is one forward window of `rows` rows. A rank's cost in a MoE layer is the weights it reads: the
//! routed GEMMs run a grid over the compacted list of the experts that have rows, one group each, and a group takes its rows in
//! passes of 8 (`docs/NEMOTRON_MOE_PERF.md`). So at 8 rows or fewer the rank's time follows the number of DISTINCT owned experts that
//! are picked ([`Cost::Passes`]), and at one row that equals the picked pairs ([`Cost::Pairs`], the proxy: `rows * 22` of them, the
//! quantity the cluster plan names). Both are scored; the shipped map is solved for the one `expert_balance solve --cost` says.
//!
//! **Byte weights (v0.1.2, mixed-bit-rate packs).** Every count above treats the experts as equal, which they are in a pack whose routed
//! experts share one EXL3 bit width. A MixedK pack stores the experts of one layer at 2 to 6 bits, so a pass over one expert reads up to
//! three times another's bytes, and an even split of the experts is not an even split of the bytes. [`ExpertBytes`] carries each
//! expert's resident bytes (the plan's tensors, `tp_shard::expert_bytes`) and the trellis widths of its layer, and two things use it:
//! the solver weighs a pick by its expert's bytes ([`solve_weighted`], [`LayerProblem::build_weighted`]), and [`byte_balanced_map`]
//! balances the ranks' bytes of every layer with no traces at all (the memory plan's fullest rank and the read time of a wide window,
//! where every expert is picked). Both act ONLY on a layer whose routed experts mix trellis widths ([`ExpertBytes::mixed`]): a layer of
//! one width is today's equal-cost problem bit for bit (its bf16 experts included, which the published packs keep at three places),
//! and a pack with no mixed layer gets no map at all (`None`: the contiguous ranges, today's plan).

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::config::OmniConfig;

/// The environment variable that names the ownership table file. Unset, empty, `contiguous` or `off`: the contiguous ranges.
pub const MAP_ENV: &str = "REXL3_TP_EXPERT_MAP";
/// First line of a map file.
pub const MAP_MAGIC: &str = "rexl3-expert-map";
/// Version of the map file format.
pub const MAP_VERSION: u32 = 1;
/// A rank id is one base-36 character of the file, so at most this many ranks.
pub const MAX_RANKS: usize = 36;
/// The value [`expert_map_fingerprint`] returns when a map was configured and cannot be used (unreadable, damaged): a boot then refuses
/// by name at the weights' load, and a peer without a map or with another one is refused at the rendezvous.
pub const FINGERPRINT_UNUSABLE: u64 = u64::MAX;

// ---------------------------------------------------------------------------------------------------------------------------------
// The contiguous rule and the hash.

/// The experts of `experts` that rank `rank` of `ranks` holds under the contiguous rule: `[E r / R, E (r + 1) / R)` (at two ranks and
/// 512 experts `[256 r, 256 (r + 1))`). The one definition: `ShardPlan::experts` is this.
pub fn contiguous_range(experts: usize, rank: usize, ranks: usize) -> Range<usize> {
    experts * rank / ranks..experts * (rank + 1) / ranks
}

/// How many experts rank `rank` holds under the contiguous rule (and under every map: a map keeps the counts).
pub fn contiguous_count(experts: usize, rank: usize, ranks: usize) -> usize {
    contiguous_range(experts, rank, ranks).len()
}

/// FNV-1a 64 over `bytes` from `h`, with a splitmix finalizer in [`Hasher::finish`] (a change detector for a map file, not a
/// cryptographic hash).
struct Hasher(u64);

impl Hasher {
    fn new() -> Self {
        Hasher(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn finish(self) -> u64 {
        let mut z = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The ownership table.

/// Which rank holds each routed expert of each MoE layer of the trunk (and, optionally, of the MTP head's block).
///
/// Invariants (held by [`ExpertMap::new`], so by every value): every row has `experts` entries, every entry is a rank below `ranks`,
/// and every rank holds exactly [`contiguous_count`] experts of every row: the weights a rank holds per layer are in number what the
/// contiguous plan holds, so the memory plan's slot and page numbers stay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertMap {
    ranks: usize,
    experts: usize,
    /// Trunk layer index -> the owning rank of each expert.
    layers: BTreeMap<usize, Vec<u8>>,
    /// The MTP head's block, when the map carries one (absent: the head's experts are contiguous).
    head: Option<Vec<u8>>,
}

impl ExpertMap {
    /// A table from rows. Refused by name: a rank count outside `2..=36`, no experts, a row of the wrong length, an id that is not a
    /// rank, or a row whose per-rank counts are not the contiguous plan's.
    pub fn new(ranks: usize, experts: usize, layers: BTreeMap<usize, Vec<u8>>, head: Option<Vec<u8>>) -> Result<Self, String> {
        if !(2..=MAX_RANKS).contains(&ranks) {
            return Err(format!("expert map: {ranks} ranks (a map is for 2 to {MAX_RANKS})"));
        }
        if experts == 0 {
            return Err("expert map: no experts".into());
        }
        let check = |what: String, row: &[u8]| -> Result<(), String> {
            if row.len() != experts {
                return Err(format!("expert map: {what} has {} entries, the map is for {experts} experts", row.len()));
            }
            let mut count = vec![0usize; ranks];
            for (e, &r) in row.iter().enumerate() {
                match count.get_mut(r as usize) {
                    Some(c) => *c += 1,
                    None => return Err(format!("expert map: {what}: expert {e} is on rank {r}, the map has {ranks} ranks")),
                }
            }
            for (r, &c) in count.iter().enumerate() {
                let want = contiguous_count(experts, r, ranks);
                if c != want {
                    return Err(format!(
                        "expert map: {what}: rank {r} holds {c} experts, the plan holds {want} a rank (equal counts keep the memory plan)"
                    ));
                }
            }
            Ok(())
        };
        for (l, row) in &layers {
            check(format!("layer {l}"), row)?;
        }
        if let Some(h) = &head {
            check("the MTP head".into(), h)?;
        }
        Ok(Self { ranks, experts, layers, head })
    }

    /// The contiguous ownership as a table, for the trunk layers `layers` (what the engine does with no map: the reference of the tests).
    pub fn contiguous(ranks: usize, experts: usize, layers: impl IntoIterator<Item = usize>) -> Result<Self, String> {
        let row: Vec<u8> = (0..experts).map(|e| contiguous_owner(experts, e, ranks) as u8).collect();
        Self::new(ranks, experts, layers.into_iter().map(|l| (l, row.clone())).collect(), None)
    }

    pub fn ranks(&self) -> usize {
        self.ranks
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    /// The trunk layers the table covers, ascending.
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.keys().copied()
    }

    /// Whether the table has its own row for the MTP head's block.
    pub fn has_head(&self) -> bool {
        self.head.is_some()
    }

    /// The rank that holds expert `e` of trunk layer `layer`; `None` when the table has no row for the layer or `e` is out of range.
    pub fn owner(&self, layer: usize, e: usize) -> Option<usize> {
        self.layers.get(&layer)?.get(e).map(|&r| r as usize)
    }

    /// The rank that holds expert `e` of the MTP head's block; `None` when the table has no head row.
    pub fn head_owner(&self, e: usize) -> Option<usize> {
        self.head.as_ref()?.get(e).map(|&r| r as usize)
    }

    /// The experts of `layer` that `rank` holds, ascending (empty for a layer the table does not cover).
    pub fn owned(&self, layer: usize, rank: usize) -> Vec<usize> {
        match self.layers.get(&layer) {
            Some(row) => (0..self.experts).filter(|&e| row[e] as usize == rank).collect(),
            None => Vec::new(),
        }
    }

    /// The row of a trunk layer (rank id per expert).
    pub fn row(&self, layer: usize) -> Option<&[u8]> {
        self.layers.get(&layer).map(Vec::as_slice)
    }

    /// The 64-bit content hash: the rank count, the expert count, every row in layer order and the head row. What the file's header
    /// carries and [`expert_map_fingerprint`] returns; two ranks hold the same map exactly when their hashes are equal.
    pub fn hash(&self) -> u64 {
        let mut h = Hasher::new();
        h.bytes(MAP_MAGIC.as_bytes());
        h.u32(MAP_VERSION);
        h.u32(self.ranks as u32);
        h.u32(self.experts as u32);
        h.u32(self.layers.len() as u32);
        for (l, row) in &self.layers {
            h.u32(*l as u32);
            h.bytes(row);
        }
        match &self.head {
            Some(row) => {
                h.u32(1);
                h.bytes(row);
            }
            None => h.u32(0),
        }
        h.finish()
    }

    /// The map fits the model the plan is built for: the rank count, the expert count, and a row for exactly the MoE layers of
    /// `cfg.pattern`. Refused by name (a map made for another pack or group size must never be applied half).
    pub fn check_for(&self, cfg: &OmniConfig, ranks: usize) -> Result<(), String> {
        if self.ranks != ranks {
            return Err(format!("{}: the map is for {} ranks, this group has {ranks}", self.describe(), self.ranks));
        }
        if self.experts != cfg.experts {
            return Err(format!("{}: the map is for {} experts, the model has {}", self.describe(), self.experts, cfg.experts));
        }
        let moe: Vec<usize> = cfg.pattern.chars().enumerate().filter(|&(_, c)| c == 'E').map(|(i, _)| i).collect();
        for &l in &moe {
            if !self.layers.contains_key(&l) {
                return Err(format!("{}: the map has no row for MoE layer {l}", self.describe()));
            }
        }
        if let Some(l) = self.layers.keys().find(|l| !moe.contains(l)) {
            return Err(format!("{}: the map has a row for layer {l}, which is not a MoE layer of this model", self.describe()));
        }
        Ok(())
    }

    /// `expert map <hash>` for messages.
    pub fn describe(&self) -> String {
        format!("expert map {:016x}", self.hash())
    }

    /// The load of each rank over the rows of the map under uniform popularity is by construction equal; this is the number of
    /// experts a rank holds of a row (the same on every layer, kept by [`ExpertMap::new`]).
    pub fn experts_on(&self, rank: usize) -> usize {
        contiguous_count(self.experts, rank, self.ranks)
    }

    // -- the file

    /// The versioned text file: a header (`rexl3-expert-map v1`, `ranks`, `experts`, `head`, `layers`, `hash`) and one line per row,
    /// `L <layer> <one base-36 character per expert>` (`H <...>` for the head).
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("{MAP_MAGIC} v{MAP_VERSION}\n"));
        s.push_str(&format!(
            "ranks {}\nexperts {}\nhead {}\nlayers {}\nhash {:016x}\n",
            self.ranks,
            self.experts,
            u8::from(self.head.is_some()),
            self.layers.len(),
            self.hash()
        ));
        for (l, row) in &self.layers {
            s.push_str(&format!("L {l} {}\n", row_text(row)));
        }
        if let Some(h) = &self.head {
            s.push_str(&format!("H {}\n", row_text(h)));
        }
        s
    }

    /// Reads a text map. The header's hash must equal the content's: a hand-edited or truncated file is refused by name.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
        let first = lines.next().ok_or("expert map: empty file")?;
        let want = format!("{MAP_MAGIC} v{MAP_VERSION}");
        if first != want {
            return Err(format!("expert map: first line is `{first}`, this engine reads `{want}`"));
        }
        let mut field = |name: &str| -> Result<String, String> {
            let l = lines.next().ok_or_else(|| format!("expert map: the header ends before `{name}`"))?;
            match l.split_once(' ') {
                Some((k, v)) if k == name => Ok(v.trim().to_string()),
                _ => Err(format!("expert map: expected `{name} <value>`, found `{l}`")),
            }
        };
        let num = |name: &str, v: String| -> Result<usize, String> {
            v.parse::<usize>().map_err(|_| format!("expert map: `{name}` is `{v}`, not a number"))
        };
        let ranks = num("ranks", field("ranks")?)?;
        let experts = num("experts", field("experts")?)?;
        let head = num("head", field("head")?)?;
        let nlayers = num("layers", field("layers")?)?;
        let hash_text = field("hash")?;
        let hash =
            u64::from_str_radix(&hash_text, 16).map_err(|_| format!("expert map: `hash` is `{hash_text}`, not 16 hex digits"))?;
        let mut layers = BTreeMap::new();
        let mut head_row = None;
        for l in lines {
            let parts: Vec<&str> = l.split(' ').collect();
            match parts.as_slice() {
                ["L", layer, row] => {
                    let layer: usize = layer.parse().map_err(|_| format!("expert map: layer `{layer}` is not a number"))?;
                    let row = parse_row(row, ranks).map_err(|e| format!("expert map: layer {layer}: {e}"))?;
                    if layers.insert(layer, row).is_some() {
                        return Err(format!("expert map: layer {layer} appears twice"));
                    }
                }
                ["H", row] => {
                    if head_row.replace(parse_row(row, ranks).map_err(|e| format!("expert map: head: {e}"))?).is_some() {
                        return Err("expert map: the head row appears twice".into());
                    }
                }
                _ => return Err(format!("expert map: unreadable line `{l}`")),
            }
        }
        if layers.len() != nlayers {
            return Err(format!("expert map: the header says {nlayers} layers, the file has {}", layers.len()));
        }
        if head_row.is_some() != (head == 1) {
            return Err(format!(
                "expert map: the header says head {head}, the file {} a head row",
                if head_row.is_some() { "has" } else { "has no" }
            ));
        }
        let m = Self::new(ranks, experts, layers, head_row)?;
        if m.hash() != hash {
            return Err(format!(
                "expert map: the header's hash {hash:016x} is not the content's {:016x} (the file was edited or truncated)",
                m.hash()
            ));
        }
        Ok(m)
    }

    /// Reads and parses a map file.
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("expert map {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("{} ({})", e, path.display()))
    }
}

fn contiguous_owner(experts: usize, e: usize, ranks: usize) -> usize {
    (0..ranks).find(|&r| contiguous_range(experts, r, ranks).contains(&e)).unwrap_or(ranks - 1)
}

fn row_text(row: &[u8]) -> String {
    row.iter().map(|&r| std::char::from_digit(r as u32, 36).unwrap_or('?')).collect()
}

fn parse_row(text: &str, ranks: usize) -> Result<Vec<u8>, String> {
    text.chars()
        .enumerate()
        .map(|(e, c)| match c.to_digit(36) {
            Some(r) if (r as usize) < ranks => Ok(r as u8),
            _ => Err(format!("expert {e} is `{c}`, not a rank below {ranks}")),
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The configured map.

static CONFIGURED: OnceLock<Result<Option<Arc<ExpertMap>>, String>> = OnceLock::new();

/// The map `REXL3_TP_EXPERT_MAP` names, read once per process: `Ok(None)` when unset (the contiguous ranges), `Err` when the file
/// cannot be used (named; the weights' load refuses by it).
pub fn configured_map() -> Result<Option<Arc<ExpertMap>>, String> {
    CONFIGURED
        .get_or_init(|| match std::env::var(MAP_ENV) {
            Ok(p) if !p.trim().is_empty() && !matches!(p.trim(), "contiguous" | "off") => {
                ExpertMap::read(Path::new(p.trim())).map(|m| Some(Arc::new(m))).map_err(|e| format!("{MAP_ENV}: {e}"))
            }
            _ => Ok(None),
        })
        .clone()
}

/// What the two ranks of a group compare so that both hold the same ownership (a rank that computed pairs its peer did not expect
/// would wait in a collective for a sum that never completes): 0 for the contiguous ranges (no map), the map's content hash (never 0)
/// otherwise, [`FINGERPRINT_UNUSABLE`] when a map is configured and cannot be used.
///
/// Both ranks must also set `tp_arm::plan_flags` bit 4 (it does, when a map is configured), so a rank with a map and a peer without
/// one differ in the byte the rendezvous already compares; the 64 bits make two different maps a refusal by name.
pub fn expert_map_fingerprint() -> u64 {
    match configured_map() {
        Ok(None) => 0,
        Ok(Some(m)) => m.hash().max(1),
        Err(_) => FINGERPRINT_UNUSABLE,
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The route tap (tooling).

static TAP_ON: AtomicBool = AtomicBool::new(false);
static TAP: Mutex<Vec<(usize, usize, Vec<i32>)>> = Mutex::new(Vec::new());

/// Turns the route tap on or off. Off (the default): `arm_moe::run` costs one atomic load per MoE layer and nothing else. On: after
/// every decode-path MoE layer the picks are downloaded (a stream synchronisation: never with CUDA graphs on) and pushed here.
pub fn tap_enable(on: bool) {
    TAP_ON.store(on, Ordering::Relaxed);
    if !on {
        TAP.lock().unwrap_or_else(|p| p.into_inner()).clear();
    }
}

/// Whether the tap is on.
#[inline]
pub fn tap_enabled() -> bool {
    TAP_ON.load(Ordering::Relaxed)
}

/// Records one layer's route: `(layer, rows, picks [rows, 22])`.
pub fn tap_push(layer: usize, rows: usize, picks: Vec<i32>) {
    TAP.lock().unwrap_or_else(|p| p.into_inner()).push((layer, rows, picks));
}

/// Takes what the tap recorded since the last take, in layer order of the calls.
pub fn tap_take() -> Vec<(usize, usize, Vec<i32>)> {
    std::mem::take(&mut *TAP.lock().unwrap_or_else(|p| p.into_inner()))
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The routing trace.

/// The kind of a recorded row.
pub const KIND_PREFILL: u8 = 0;
/// A decode row whose token is the text's (teacher forced, as the verify window of a draft that is accepted).
pub const KIND_TEXT: u8 = 1;
/// A decode row whose token is the model's own greedy continuation.
pub const KIND_GENERATED: u8 = 2;

const TRACE_MAGIC: &[u8; 8] = b"RXRT0001";

/// The routed expert ids of every row of one prompt's run, per MoE layer, and the windows the run was executed in.
///
/// A row's routing does not depend on which rows share its window (the engine is row-independent bit for bit), so a trace can be
/// regrouped into any window shape afterwards ([`Grouping`]); the windows as run are kept for what they are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingTrace {
    pub name: String,
    pub experts: usize,
    pub topk: usize,
    /// The MoE layers the picks cover, ascending.
    pub layers: Vec<usize>,
    /// Per row, in sequence order: [`KIND_PREFILL`], [`KIND_TEXT`] or [`KIND_GENERATED`].
    pub kind: Vec<u8>,
    /// The windows as run, in order; they sum to the row count.
    pub windows: Vec<u8>,
    /// Per layer (the order of `layers`): `rows * topk` expert ids, row-major, in the router's rank order.
    pub picks: Vec<Vec<u16>>,
}

impl RoutingTrace {
    pub fn new(name: &str, experts: usize, topk: usize, layers: Vec<usize>) -> Self {
        let n = layers.len();
        Self { name: name.to_string(), experts, topk, layers, kind: Vec::new(), windows: Vec::new(), picks: vec![Vec::new(); n] }
    }

    pub fn rows(&self) -> usize {
        self.kind.len()
    }

    /// Appends one window of `rows` rows of `kind`, the layers' picks given as the tap took them (`(layer, rows, ids)`, each layer of
    /// `self.layers` exactly once).
    pub fn push_window(&mut self, rows: usize, kind: u8, tap: &[(usize, usize, Vec<i32>)]) -> Result<(), String> {
        if tap.len() != self.layers.len() {
            return Err(format!("the tap holds {} layers, the trace covers {}", tap.len(), self.layers.len()));
        }
        for (pos, (layer, r, ids)) in tap.iter().enumerate() {
            if *layer != self.layers[pos] || *r != rows || ids.len() != rows * self.topk {
                return Err(format!(
                    "tap entry {pos} is layer {layer} with {r} rows and {} ids; expected layer {} with {rows} rows and {}",
                    ids.len(),
                    self.layers[pos],
                    rows * self.topk
                ));
            }
        }
        for (pos, (_, _, ids)) in tap.iter().enumerate() {
            for &id in ids {
                if id < 0 || id as usize >= self.experts {
                    return Err(format!("layer {}: expert id {id} out of range", self.layers[pos]));
                }
                self.picks[pos].push(id as u16);
            }
        }
        self.kind.extend(std::iter::repeat_n(kind, rows));
        self.windows.push(rows as u8);
        Ok(())
    }

    /// The binary file (little endian): magic, experts, topk, layer list, name, row count, window count, kinds, windows, picks.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(TRACE_MAGIC);
        for v in [self.experts as u32, self.topk as u32, self.layers.len() as u32] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for &l in &self.layers {
            b.extend_from_slice(&(l as u32).to_le_bytes());
        }
        b.extend_from_slice(&(self.name.len() as u32).to_le_bytes());
        b.extend_from_slice(self.name.as_bytes());
        b.extend_from_slice(&(self.kind.len() as u64).to_le_bytes());
        b.extend_from_slice(&(self.windows.len() as u32).to_le_bytes());
        b.extend_from_slice(&self.kind);
        b.extend_from_slice(&self.windows);
        for p in &self.picks {
            for &id in p {
                b.extend_from_slice(&id.to_le_bytes());
            }
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let s = b.get(at..at + n).ok_or_else(|| "routing trace: truncated".to_string())?;
            at += n;
            Ok(s)
        };
        if take(8)? != TRACE_MAGIC {
            return Err("routing trace: not an RXRT0001 file".into());
        }
        let u32le = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap()) as usize;
        let experts = u32le(take(4)?);
        let topk = u32le(take(4)?);
        let nl = u32le(take(4)?);
        let mut layers = Vec::with_capacity(nl);
        for _ in 0..nl {
            layers.push(u32le(take(4)?));
        }
        let nlen = u32le(take(4)?);
        let name = String::from_utf8_lossy(take(nlen)?).into_owned();
        let rows = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
        let nw = u32le(take(4)?);
        let kind = take(rows)?.to_vec();
        let windows = take(nw)?.to_vec();
        if windows.iter().map(|&w| w as usize).sum::<usize>() != rows {
            return Err("routing trace: the windows do not sum to the row count".into());
        }
        let mut picks = Vec::with_capacity(nl);
        for _ in 0..nl {
            let raw = take(rows * topk * 2)?;
            picks.push(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect());
        }
        Ok(Self { name, experts, topk, layers, kind, windows, picks })
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&b).map_err(|e| format!("{}: {e}", path.display()))
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Byte weights (mixed-bit-rate packs).

/// What a routed expert costs in bytes, per MoE layer of the trunk: each expert's resident bytes (its up and down projections, every
/// part: the trellis, the scales and the codebook marker, or the bf16 weights) and the EXL3 trellis widths of the layer's experts. Built
/// from the plan by `tp_shard::expert_bytes` (the numbers `ShardPlan::rank_bytes` counts) or by hand in the tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpertBytes {
    /// Routed experts per layer.
    pub experts: usize,
    /// Trunk layer -> bytes of each expert (`experts` entries).
    pub bytes: BTreeMap<usize, Vec<u64>>,
    /// Trunk layer -> the distinct trellis bit widths (`shape[2] / 16`) of its routed experts' EXL3 projections, ascending. A bf16 expert
    /// has no trellis and adds nothing here.
    pub widths: BTreeMap<usize, Vec<usize>>,
}

/// The integer unit of [`ExpertBytes::weights`]: the lightest expert of a mixed layer weighs this much, every other its bytes in
/// proportion (rounded). 64 resolves the 2-to-6-bit steps of a MixedK pack to about 1 percent.
pub const WEIGHT_UNIT: u32 = 64;

impl ExpertBytes {
    /// Whether the routed experts of `layer` mix trellis widths (two or more distinct K): the only layers the byte weights touch. A
    /// layer of one width, with or without bf16 experts, is the equal-cost problem exactly as before.
    pub fn mixed(&self, layer: usize) -> bool {
        self.widths.get(&layer).is_some_and(|w| w.len() >= 2)
    }

    /// The layers that mix widths, ascending.
    pub fn mixed_layers(&self) -> Vec<usize> {
        self.widths.keys().copied().filter(|&l| self.mixed(l)).collect()
    }

    /// The solver's per-expert weights of `layer`: `None` for a layer that does not mix widths (or that the table does not cover), else
    /// each expert's bytes over the lightest's, in units of [`WEIGHT_UNIT`], rounded (at least 1).
    pub fn weights(&self, layer: usize) -> Option<Vec<u32>> {
        if !self.mixed(layer) {
            return None;
        }
        let row = self.bytes.get(&layer)?;
        let min = row.iter().copied().filter(|&b| b > 0).min()?;
        Some(
            row.iter()
                .map(|&b| ((b as u128 * WEIGHT_UNIT as u128 + min as u128 / 2) / min as u128).clamp(1, u32::MAX as u128) as u32)
                .collect(),
        )
    }

    /// Bytes each rank holds of `layer`'s experts under `owner` (rank per expert).
    pub fn rank_bytes(&self, layer: usize, owner: &[u8], ranks: usize) -> Vec<u64> {
        let mut out = vec![0u64; ranks];
        if let Some(row) = self.bytes.get(&layer) {
            for (e, &b) in row.iter().enumerate() {
                out[owner[e] as usize] += b;
            }
        }
        out
    }

    /// Bytes each rank holds of every layer's experts under `map` (`None`: the contiguous ranges).
    pub fn rank_totals(&self, map: Option<&ExpertMap>, ranks: usize) -> Vec<u64> {
        let mut out = vec![0u64; ranks];
        for &l in self.bytes.keys() {
            let owner: Vec<u8> = match map.and_then(|m| m.row(l)) {
                Some(row) => row.to_vec(),
                None => (0..self.experts).map(|e| contiguous_owner(self.experts, e, ranks) as u8).collect(),
            };
            for (r, b) in self.rank_bytes(l, &owner, ranks).into_iter().enumerate() {
                out[r] += b;
            }
        }
        out
    }
}

/// One layer's experts assigned to `ranks` ranks by their bytes, with the contiguous plan's count on every rank: the contiguous ranges,
/// then, while some swap lowers the fullest rank's bytes, the swap that lowers it most (the fullest rank gives an expert of `a` bytes
/// to another rank for one of `b < a` bytes, and the other rank ends below the old fullest). Deterministic: ties go to the lowest rank,
/// then to the lightest pair of byte values, then to the lowest expert ids. A layer whose experts all weigh the same is returned as
/// the contiguous ranges (no swap moves a byte); so is a layer whose only heavy experts cannot be moved without making another rank the
/// fullest (one bf16 expert in a layer of 4-bit ones). Terminates: every swap lowers the fullest bytes or the number of ranks at them.
pub fn balance_layer_bytes(bytes: &[u64], ranks: usize) -> Vec<u8> {
    let experts = bytes.len();
    let mut owner: Vec<u8> = (0..experts).map(|e| contiguous_owner(experts, e, ranks) as u8).collect();
    if ranks < 2 || experts == 0 {
        return owner;
    }
    let mut classes: Vec<u64> = bytes.to_vec();
    classes.sort_unstable();
    classes.dedup();
    let class_of = |b: u64| classes.binary_search(&b).expect("a byte value of the layer");
    // per rank: its bytes and how many of its experts weigh each class
    let mut sum = vec![0u64; ranks];
    let mut count = vec![vec![0usize; classes.len()]; ranks];
    for (e, &b) in bytes.iter().enumerate() {
        sum[owner[e] as usize] += b;
        count[owner[e] as usize][class_of(b)] += 1;
    }
    for _ in 0..experts * ranks {
        let hi = (0..ranks).fold(0, |m, r| if sum[r] > sum[m] { r } else { m });
        // (the fuller of the two ranks after the swap, the other rank, class given, class taken)
        let mut best: Option<(u64, usize, usize, usize)> = None;
        for lo in (0..ranks).filter(|&r| r != hi) {
            for ca in (0..classes.len()).filter(|&c| count[hi][c] > 0) {
                for cb in (0..ca).filter(|&c| count[lo][c] > 0) {
                    let d = classes[ca] - classes[cb];
                    if sum[lo] + d >= sum[hi] {
                        continue;
                    }
                    let fuller = (sum[hi] - d).max(sum[lo] + d);
                    if best.is_none_or(|(f, ..)| fuller < f) {
                        best = Some((fuller, lo, ca, cb));
                    }
                }
            }
        }
        let Some((_, lo, ca, cb)) = best else { break };
        let a = (0..experts).find(|&e| owner[e] as usize == hi && bytes[e] == classes[ca]).expect("counted");
        let b = (0..experts).find(|&e| owner[e] as usize == lo && bytes[e] == classes[cb]).expect("counted");
        owner[a] = lo as u8;
        owner[b] = hi as u8;
        let d = classes[ca] - classes[cb];
        sum[hi] -= d;
        sum[lo] += d;
        count[hi][ca] -= 1;
        count[lo][ca] += 1;
        count[lo][cb] -= 1;
        count[hi][cb] += 1;
    }
    owner
}

/// The byte-balanced ownership of the MoE layers `moe` for a group of `ranks`: [`balance_layer_bytes`] on every layer that mixes
/// trellis widths ([`ExpertBytes::mixed`]), the contiguous ranges on every other. `Ok(None)` when no layer moves an expert (every pack
/// of one bit width, ours included: the plan stays today's, `ShardPlan::expert_map == None`, bit for bit); `Ok(Some(map))` otherwise, a
/// table with the contiguous counts on every row (so the memory plan's slots stay) and no head row (the head's experts stay contiguous).
///
/// The gate is load-bearing, not only a shortcut: at a rank count that does not divide the experts (three ranks: 170 / 171 / 171) the
/// layer balance WOULD move a lone bf16 expert of a one-width layer onto the rank of 170. And the balance is per layer (each layer's
/// collective waits for its own fullest rank): at such a rank count it can raise the fullest rank's total over all layers, because the
/// ranks of 171 experts are the fuller ones on every layer of one width. At two ranks (the only group run on a fabric) and at any rank
/// count that divides 512 the totals only fall.
///
/// Serving does not apply this by itself: `tp_arm::serving_shard_plan` takes a map only from `REXL3_TP_EXPERT_MAP` (the file
/// `expert_balance bytes-balance` writes), because the boot gate compares that file's hash and the P2 pair gather
/// (`tp_arm::moe_pair_gather`) assumes the contiguous ranges whenever no file is configured.
pub fn byte_balanced_map(bytes: &ExpertBytes, ranks: usize, moe: &[usize]) -> Result<Option<ExpertMap>, String> {
    if ranks < 2 {
        return Ok(None);
    }
    let experts = bytes.experts;
    let contiguous: Vec<u8> = (0..experts).map(|e| contiguous_owner(experts, e, ranks) as u8).collect();
    let mut rows = BTreeMap::new();
    let mut moved = false;
    for &l in moe {
        let row = match bytes.bytes.get(&l) {
            Some(b) if bytes.mixed(l) => {
                if b.len() != experts {
                    return Err(format!("expert bytes: layer {l} has {} experts, the table is for {experts}", b.len()));
                }
                balance_layer_bytes(b, ranks)
            }
            _ => contiguous.clone(),
        };
        moved |= row != contiguous;
        rows.insert(l, row);
    }
    if !moved {
        return Ok(None);
    }
    ExpertMap::new(ranks, experts, rows, None).map(Some)
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Steps, costs, the problem.

/// How rows are grouped into steps (a step is one forward window: the unit whose busiest rank everyone waits for).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grouping {
    /// The decode windows exactly as the run executed them (1 to 8 rows).
    Recorded,
    /// Consecutive decode rows of one prompt in windows of `k` rows (the draft depth of a verify window plus one; 1 is the serial step).
    Window(usize),
    /// The prefill rows in slices of `k` rows (64 is the engine's slice).
    Prefill(usize),
}

/// What a rank's cost of a step is, per expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cost {
    /// The (row, expert) pairs the rank computes: `c` rows picked the expert, it costs `c`.
    Pairs,
    /// The weight passes: a picked expert's weights are read once per 8 rows, `ceil(c / 8)` (the distinct experts at 8 rows or fewer).
    Passes,
}

impl Cost {
    fn weight(self, c: usize) -> u32 {
        match self {
            Cost::Pairs => c as u32,
            Cost::Passes => c.div_ceil(8) as u32,
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "pairs" => Ok(Cost::Pairs),
            "passes" | "distinct" => Ok(Cost::Passes),
            other => Err(format!("cost `{other}` is not `pairs` or `passes`")),
        }
    }
}

/// One step: the row ranges `(trace index, first row, one past the last row)` whose rows share the window (several traces when
/// streams are merged).
pub type StepRef = Vec<(usize, usize, usize)>;

/// The steps of `traces` under `grouping`, with `streams` prompts merged into each window (traces taken in groups of `streams`, step
/// `i` of a group is the union of step `i` of its members; `streams = 1` is a single stream). A trace's rows of another kind than the
/// grouping asks for are not in its steps.
pub fn steps_of(traces: &[&RoutingTrace], grouping: Grouping, streams: usize) -> Vec<StepRef> {
    let streams = streams.max(1);
    let per_trace: Vec<Vec<(usize, usize)>> = traces.iter().map(|t| trace_steps(t, grouping)).collect();
    let mut out = Vec::new();
    for group in (0..traces.len()).collect::<Vec<_>>().chunks(streams) {
        if group.len() < streams {
            break;
        }
        let n = group.iter().map(|&t| per_trace[t].len()).min().unwrap_or(0);
        for i in 0..n {
            out.push(group.iter().map(|&t| (t, per_trace[t][i].0, per_trace[t][i].1)).collect());
        }
    }
    out
}

fn trace_steps(t: &RoutingTrace, grouping: Grouping) -> Vec<(usize, usize)> {
    let decode = |i: usize| t.kind[i] != KIND_PREFILL;
    match grouping {
        Grouping::Recorded => {
            let mut out = Vec::new();
            let mut at = 0usize;
            for &w in &t.windows {
                let (lo, hi) = (at, at + w as usize);
                at = hi;
                if (lo..hi).all(decode) {
                    out.push((lo, hi));
                }
            }
            out
        }
        Grouping::Window(k) => runs(t, decode, k.max(1)),
        Grouping::Prefill(k) => runs(t, |i| t.kind[i] == KIND_PREFILL, k.max(1)),
    }
}

/// Maximal runs of rows the predicate accepts, cut into pieces of `k` rows (a short last piece of a run is dropped: every step of
/// the grouping has exactly `k` rows).
fn runs(t: &RoutingTrace, keep: impl Fn(usize) -> bool, k: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let n = t.rows();
    let mut i = 0;
    while i < n {
        if !keep(i) {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && keep(j) {
            j += 1;
        }
        let mut s = i;
        while s + k <= j {
            out.push((s, s + k));
            s += k;
        }
        i = j;
    }
    out
}

/// One layer's balancing problem: for each expert the steps it is picked in and its cost there.
pub struct LayerProblem {
    pub experts: usize,
    pub steps: usize,
    /// Per expert, `(step, cost)` ascending by step.
    act: Vec<Vec<(u32, u32)>>,
    /// The cost of the lightest expert's pick: 1 for the equal-cost problem, [`WEIGHT_UNIT`] for a byte-weighted one (the solver's
    /// `min_gain` is in units of it, so a weighted layer accepts the swaps of the same size).
    pub unit: u32,
}

impl LayerProblem {
    /// The problem of the layer at position `layer_pos` of the traces' layer lists (all traces must cover the same layers).
    pub fn build(traces: &[&RoutingTrace], layer_pos: usize, steps: &[StepRef], cost: Cost) -> Self {
        Self::build_weighted(traces, layer_pos, steps, cost, None)
    }

    /// [`LayerProblem::build`] with each expert's pick weighed by `weights[e]` (its bytes over the layer's lightest expert's, in units of
    /// [`WEIGHT_UNIT`]: [`ExpertBytes::weights`]): what a rank reads, not how many experts it reads. `None` is `build`, bit for bit.
    pub fn build_weighted(
        traces: &[&RoutingTrace],
        layer_pos: usize,
        steps: &[StepRef],
        cost: Cost,
        weights: Option<&[u32]>,
    ) -> Self {
        let experts = traces.first().map_or(0, |t| t.experts);
        let w = |e: usize| weights.map_or(1, |w| w[e]);
        let mut act: Vec<Vec<(u32, u32)>> = vec![Vec::new(); experts];
        let mut count = vec![0usize; experts];
        let mut touched: Vec<usize> = Vec::new();
        for (s, step) in steps.iter().enumerate() {
            for &(t, lo, hi) in step {
                let tr = traces[t];
                let ids = &tr.picks[layer_pos][lo * tr.topk..hi * tr.topk];
                for &id in ids {
                    let e = id as usize;
                    if count[e] == 0 {
                        touched.push(e);
                    }
                    count[e] += 1;
                }
            }
            for &e in &touched {
                act[e].push((s as u32, cost.weight(count[e]).saturating_mul(w(e))));
                count[e] = 0;
            }
            touched.clear();
        }
        Self { experts, steps: steps.len(), act, unit: if weights.is_some() { WEIGHT_UNIT } else { 1 } }
    }

    /// Total cost of expert `e` over the steps (its marginal frequency).
    pub fn total(&self, e: usize) -> u64 {
        self.act[e].iter().map(|&(_, w)| w as u64).sum()
    }
}

/// A score: the sum over steps of the busiest rank's cost, and of the mean rank's cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Score {
    pub busiest: f64,
    pub mean: f64,
    pub steps: usize,
}

impl Score {
    /// Busiest over mean: 1.0 is perfectly even; every collective waits for the busiest rank.
    pub fn ratio(&self) -> f64 {
        if self.mean > 0.0 {
            self.busiest / self.mean
        } else {
            1.0
        }
    }

    pub fn add(&mut self, o: &Score) {
        self.busiest += o.busiest;
        self.mean += o.mean;
        self.steps += o.steps;
    }
}

/// Scores one layer's assignment on a problem.
pub fn evaluate(p: &LayerProblem, owner: &[u8], ranks: usize) -> Score {
    let mut load = vec![0u32; p.steps * ranks];
    let mut total = 0u64;
    for (e, list) in p.act.iter().enumerate() {
        let r = owner[e] as usize;
        for &(s, w) in list {
            load[s as usize * ranks + r] += w;
            total += w as u64;
        }
    }
    let mut busiest = 0u64;
    for s in 0..p.steps {
        busiest += *load[s * ranks..(s + 1) * ranks].iter().max().unwrap_or(&0) as u64;
    }
    Score { busiest: busiest as f64, mean: total as f64 / ranks as f64, steps: p.steps }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The solver.

/// Solver settings.
#[derive(Clone, Copy, Debug)]
pub struct SolveOpts {
    /// Wall time for the local search of one layer.
    pub budget: Duration,
    pub seed: u64,
    /// Stop after this many full sweeps without improvement (a sweep tries, for every expert, its swap with every expert of the other
    /// ranks).
    pub max_sweeps: usize,
    /// A swap is applied only when it lowers the training objective by at least this much (units of the cost summed over steps).
    /// Small gains on thousands of steps are mostly the training sample's noise, which a held-out half punishes: raising this trades
    /// training fit for generalisation.
    pub min_gain: i64,
    /// Start from the contiguous ranges instead of the greedy LPT assignment (fewest changes from today's ownership).
    pub from_contiguous: bool,
}

impl Default for SolveOpts {
    fn default() -> Self {
        Self { budget: Duration::from_secs(20), seed: 0x5eed, max_sweeps: 40, min_gain: 1, from_contiguous: false }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The greedy LPT start: experts by total cost descending (ties by id), each to the rank of lowest current total with room. The
/// result has exactly [`contiguous_count`] experts a rank.
pub fn lpt(p: &LayerProblem, ranks: usize) -> Vec<u8> {
    let mut order: Vec<usize> = (0..p.experts).collect();
    order.sort_by(|&a, &b| p.total(b).cmp(&p.total(a)).then(a.cmp(&b)));
    let cap: Vec<usize> = (0..ranks).map(|r| contiguous_count(p.experts, r, ranks)).collect();
    let mut used = vec![0usize; ranks];
    let mut sum = vec![0u64; ranks];
    let mut owner = vec![0u8; p.experts];
    for e in order {
        let r = (0..ranks)
            .filter(|&r| used[r] < cap[r])
            .min_by(|&a, &b| sum[a].cmp(&sum[b]).then(used[a].cmp(&used[b])).then(a.cmp(&b)))
            .expect("capacities sum to the expert count");
        owner[e] = r as u8;
        used[r] += 1;
        sum[r] += p.total(e);
    }
    owner
}

/// The change of the objective if experts `a` and `b` (on different ranks) swapped ranks.
#[inline]
fn swap_delta(p: &LayerProblem, load: &[u32], smax: &[u32], owner: &[u8], ranks: usize, a: usize, b: usize) -> i64 {
    let (ra, rb) = (owner[a] as usize, owner[b] as usize);
    let (la, lb) = (&p.act[a], &p.act[b]);
    let (mut i, mut j) = (0usize, 0usize);
    let mut delta = 0i64;
    while i < la.len() || j < lb.len() {
        let (s, wa, wb) = match (la.get(i), lb.get(j)) {
            (Some(&(sa, wa)), Some(&(sb, wb))) if sa == sb => {
                i += 1;
                j += 1;
                (sa, wa as i64, wb as i64)
            }
            (Some(&(sa, wa)), Some(&(sb, _))) if sa < sb => {
                i += 1;
                (sa, wa as i64, 0)
            }
            (Some(&(sa, wa)), None) => {
                i += 1;
                (sa, wa as i64, 0)
            }
            (_, Some(&(sb, wb))) => {
                j += 1;
                (sb, 0, wb as i64)
            }
            (None, None) => break,
        };
        let row = &load[s as usize * ranks..(s as usize + 1) * ranks];
        let na = row[ra] as i64 - wa + wb;
        let nb = row[rb] as i64 - wb + wa;
        let mut m = na.max(nb);
        for (r, &v) in row.iter().enumerate() {
            if r != ra && r != rb {
                m = m.max(v as i64);
            }
        }
        delta += m - smax[s as usize] as i64;
    }
    delta
}

/// Greedy LPT, then the swap local search, on one layer. Returns the assignment and its objective (the sum over steps of the busiest
/// rank's cost on the training steps). Deterministic for a seed and a budget that is not the limit; the budget bounds the wall time.
pub fn solve_layer(p: &LayerProblem, ranks: usize, opts: &SolveOpts) -> (Vec<u8>, u64) {
    let mut owner = if opts.from_contiguous {
        (0..p.experts).map(|e| contiguous_owner(p.experts, e, ranks) as u8).collect()
    } else {
        lpt(p, ranks)
    };
    let mut load = vec![0u32; p.steps * ranks];
    for (e, list) in p.act.iter().enumerate() {
        for &(s, w) in list {
            load[s as usize * ranks + owner[e] as usize] += w;
        }
    }
    let mut smax: Vec<u32> = (0..p.steps).map(|s| *load[s * ranks..(s + 1) * ranks].iter().max().unwrap_or(&0)).collect();
    let deadline = Instant::now() + opts.budget;
    let mut rng = Rng(opts.seed | 1);
    let mut order: Vec<usize> = (0..p.experts).collect();
    let mut stale = 0usize;
    'sweeps: for _ in 0..opts.max_sweeps {
        // a random order of the first expert of the pair, so the sweep does not favour low ids
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut improved = false;
        for &a in &order {
            if Instant::now() >= deadline {
                break 'sweeps;
            }
            let mut best: Option<(i64, usize)> = None;
            for b in 0..p.experts {
                if owner[b] == owner[a] {
                    continue;
                }
                let d = swap_delta(p, &load, &smax, &owner, ranks, a, b);
                if d <= -(opts.min_gain.max(1) * p.unit as i64) && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, b));
                }
            }
            if let Some((_, b)) = best {
                apply_swap(p, &mut load, &mut smax, &mut owner, ranks, a, b);
                improved = true;
            }
        }
        if improved {
            stale = 0;
        } else {
            stale += 1;
            if stale >= 1 {
                break;
            }
        }
    }
    let objective: u64 = smax.iter().map(|&m| m as u64).sum();
    (owner, objective)
}

fn apply_swap(p: &LayerProblem, load: &mut [u32], smax: &mut [u32], owner: &mut [u8], ranks: usize, a: usize, b: usize) {
    let (ra, rb) = (owner[a] as usize, owner[b] as usize);
    for &(s, w) in &p.act[a] {
        load[s as usize * ranks + ra] -= w;
        load[s as usize * ranks + rb] += w;
    }
    for &(s, w) in &p.act[b] {
        load[s as usize * ranks + rb] -= w;
        load[s as usize * ranks + ra] += w;
    }
    owner.swap(a, b);
    for &(s, _) in p.act[a].iter().chain(p.act[b].iter()) {
        let s = s as usize;
        smax[s] = *load[s * ranks..(s + 1) * ranks].iter().max().unwrap_or(&0);
    }
}

/// One layer's solution: the assignment and its training objective.
type LayerResult = (Vec<u8>, u64);

/// The result of [`solve`]: the table and per layer the training objective.
pub struct Solved {
    pub map: ExpertMap,
    pub objective: BTreeMap<usize, u64>,
}

/// Solves every layer of `traces` (which must all cover the same MoE layers, `experts` and `topk`) for `ranks` ranks on the steps
/// `grouping` and `streams` define, in parallel over layers. `lpt_only` stops at the greedy start (the marginal-frequency baseline).
pub fn solve(
    traces: &[&RoutingTrace],
    ranks: usize,
    grouping: Grouping,
    streams: usize,
    cost: Cost,
    opts: &SolveOpts,
    lpt_only: bool,
) -> Result<Solved, String> {
    solve_weighted(traces, ranks, grouping, streams, cost, opts, lpt_only, None)
}

/// [`solve`] with each pick weighed by its expert's bytes (`bytes`, [`ExpertBytes::weights`]) on the layers that mix trellis widths;
/// every other layer, and every layer when `bytes` is `None`, is solved exactly as [`solve`] solves it (the same problem, the same seed,
/// the same assignment).
#[allow(clippy::too_many_arguments)]
pub fn solve_weighted(
    traces: &[&RoutingTrace],
    ranks: usize,
    grouping: Grouping,
    streams: usize,
    cost: Cost,
    opts: &SolveOpts,
    lpt_only: bool,
    bytes: Option<&ExpertBytes>,
) -> Result<Solved, String> {
    let first = traces.first().ok_or("solve: no traces")?;
    if let Some(b) = bytes {
        if b.experts != first.experts {
            return Err(format!("solve: the byte table is for {} experts, the traces for {}", b.experts, first.experts));
        }
    }
    for t in traces {
        if t.layers != first.layers || t.experts != first.experts || t.topk != first.topk {
            return Err(format!("solve: trace `{}` does not cover the layers and experts of `{}`", t.name, first.name));
        }
    }
    let steps = steps_of(traces, grouping, streams);
    if steps.is_empty() {
        return Err("solve: the grouping selects no steps".into());
    }
    let nl = first.layers.len();
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<LayerResult>>> = Mutex::new(vec![None; nl]);
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(nl.max(1));
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= nl {
                    break;
                }
                let weights = bytes.and_then(|b| b.weights(first.layers[i]));
                let p = LayerProblem::build_weighted(traces, i, &steps, cost, weights.as_deref());
                let r = if lpt_only {
                    let o = lpt(&p, ranks);
                    let obj = evaluate(&p, &o, ranks).busiest as u64;
                    (o, obj)
                } else {
                    solve_layer(&p, ranks, &SolveOpts { seed: opts.seed ^ (i as u64).wrapping_mul(0x9E37_79B9), ..*opts })
                };
                results.lock().unwrap_or_else(|p| p.into_inner())[i] = Some(r);
            });
        }
    });
    let results = results.into_inner().unwrap_or_else(|p| p.into_inner());
    let mut rows = BTreeMap::new();
    let mut objective = BTreeMap::new();
    for (i, r) in results.into_iter().enumerate() {
        let (owner, obj) = r.ok_or("solve: a worker did not finish")?;
        rows.insert(first.layers[i], owner);
        objective.insert(first.layers[i], obj);
    }
    Ok(Solved { map: ExpertMap::new(ranks, first.experts, rows, None)?, objective })
}

/// The score of `map` (all its layers that the traces cover) on the steps, summed over layers.
pub fn evaluate_map(map: &ExpertMap, traces: &[&RoutingTrace], grouping: Grouping, streams: usize, cost: Cost) -> Score {
    evaluate_map_weighted(map, traces, grouping, streams, cost, None)
}

/// [`evaluate_map`] with the picks of the layers that mix trellis widths weighed by their experts' bytes ([`ExpertBytes::weights`]),
/// in units of the lightest expert's pick ([`WEIGHT_UNIT`] is divided out, so a weighted layer scores in the same units as an equal-cost
/// one). `None` is [`evaluate_map`].
pub fn evaluate_map_weighted(
    map: &ExpertMap,
    traces: &[&RoutingTrace],
    grouping: Grouping,
    streams: usize,
    cost: Cost,
    bytes: Option<&ExpertBytes>,
) -> Score {
    let mut total = Score::default();
    let steps = steps_of(traces, grouping, streams);
    if steps.is_empty() {
        return total;
    }
    let Some(first) = traces.first() else { return total };
    for (pos, &layer) in first.layers.iter().enumerate() {
        let Some(row) = map.row(layer) else { continue };
        let weights = bytes.and_then(|b| b.weights(layer));
        let p = LayerProblem::build_weighted(traces, pos, &steps, cost, weights.as_deref());
        let mut s = evaluate(&p, row, map.ranks());
        if p.unit != 1 {
            s.busiest /= p.unit as f64;
            s.mean /= p.unit as f64;
        }
        total.add(&s);
    }
    total
}

/// The share of all routed (row, expert) pairs of `traces` (every row, every layer of the map) that each rank computes under `map`.
pub fn rank_pair_shares(map: &ExpertMap, traces: &[&RoutingTrace]) -> Vec<f64> {
    let mut count = vec![0u64; map.ranks()];
    for t in traces {
        for (pos, &layer) in t.layers.iter().enumerate() {
            let Some(row) = map.row(layer) else { continue };
            for &id in &t.picks[pos] {
                count[row[id as usize] as usize] += 1;
            }
        }
    }
    let total: u64 = count.iter().sum();
    count.iter().map(|&c| if total > 0 { c as f64 / total as f64 } else { 0.0 }).collect()
}

/// A seeded random assignment with the contiguous plan's counts (the worst case for exactness: nothing about it is structured).
pub fn shuffled_map(ranks: usize, experts: usize, layers: &[usize], seed: u64) -> Result<ExpertMap, String> {
    shuffled_map_with_head(ranks, experts, layers, seed, false)
}

/// [`shuffled_map`], with a shuffled row for the MTP head's block too when `head` is set.
pub fn shuffled_map_with_head(
    ranks: usize,
    experts: usize,
    layers: &[usize],
    seed: u64,
    head: bool,
) -> Result<ExpertMap, String> {
    let mut rng = Rng(seed | 1);
    let base: Vec<u8> = (0..experts).map(|e| contiguous_owner(experts, e, ranks) as u8).collect();
    let shuffled = |rng: &mut Rng| {
        let mut row = base.clone();
        for i in (1..row.len()).rev() {
            row.swap(i, rng.below(i + 1));
        }
        row
    };
    let mut rows = BTreeMap::new();
    for &l in layers {
        rows.insert(l, shuffled(&mut rng));
    }
    let head_row = head.then(|| shuffled(&mut rng));
    ExpertMap::new(ranks, experts, rows, head_row)
}

/// The MoE layers of a model (`E` in the pattern).
pub fn moe_layers(cfg: &OmniConfig) -> Vec<usize> {
    cfg.pattern.chars().enumerate().filter(|&(_, c)| c == 'E').map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");

    fn cfg() -> OmniConfig {
        OmniConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    /// A synthetic trace: `rows` rows of `topk` distinct experts per layer, drawn from a skewed popularity (the low experts are hot) with
    /// planted co-activation (a row that picks expert `2k` often picks `2k + 1` too).
    fn synthetic(name: &str, experts: usize, layers: &[usize], rows: usize, seed: u64) -> RoutingTrace {
        let mut rng = Rng(seed | 1);
        let topk = 22;
        let mut t = RoutingTrace::new(name, experts, topk, layers.to_vec());
        let mut w = 0;
        while t.rows() < rows {
            let win = 1 + (w % 8);
            w += 1;
            let mut tap = Vec::new();
            for &l in layers {
                let mut ids = Vec::new();
                for _ in 0..win {
                    let mut row: Vec<i32> = Vec::new();
                    while row.len() < topk {
                        // skew: the square of a uniform draw favours low ids
                        let u = rng.below(1 << 20) as f64 / (1 << 20) as f64;
                        let e = ((u * u) * experts as f64) as usize % experts;
                        for c in [e, e ^ 1] {
                            if row.len() < topk && !row.contains(&(c as i32)) {
                                row.push(c as i32);
                            }
                        }
                    }
                    ids.extend(row);
                }
                tap.push((l, win, ids));
            }
            t.push_window(win, KIND_TEXT, &tap).unwrap();
        }
        t
    }

    #[test]
    fn the_contiguous_rule_is_the_engines() {
        for ranks in 1..=8 {
            let mut seen = vec![0; 512];
            for r in 0..ranks {
                for e in contiguous_range(512, r, ranks) {
                    seen[e] += 1;
                    assert_eq!(contiguous_owner(512, e, ranks), r);
                }
            }
            assert!(seen.iter().all(|&n| n == 1));
        }
        assert_eq!(contiguous_range(512, 1, 2), 256..512);
        assert_eq!(contiguous_count(512, 0, 3), 170);
        assert_eq!(contiguous_count(512, 1, 3), 171);
    }

    #[test]
    fn a_map_round_trips_through_its_file_and_its_hash_holds() {
        let c = cfg();
        let layers = moe_layers(&c);
        assert_eq!(layers.len(), 40);
        for ranks in [2usize, 3, 5, 8] {
            let m = shuffled_map(ranks, c.experts, &layers, 7).unwrap();
            let text = m.to_text();
            assert!(text.starts_with("rexl3-expert-map v1\n"));
            let back = ExpertMap::parse(&text).unwrap();
            assert_eq!(back, m);
            assert_eq!(back.hash(), m.hash());
            m.check_for(&c, ranks).unwrap();
            // the same content under another seed is another map
            assert_ne!(shuffled_map(ranks, c.experts, &layers, 8).unwrap().hash(), m.hash());
        }
        // the contiguous table is the default rule
        let m = ExpertMap::contiguous(2, 512, [1usize, 3]).unwrap();
        assert_eq!(m.owner(1, 255), Some(0));
        assert_eq!(m.owner(1, 256), Some(1));
        assert_eq!(m.owner(2, 0), None);
        assert_eq!(m.owned(3, 1), (256..512).collect::<Vec<_>>());
    }

    #[test]
    fn a_damaged_or_foreign_map_is_refused_by_name() {
        let c = cfg();
        let layers = moe_layers(&c);
        let m = shuffled_map(2, c.experts, &layers, 1).unwrap();
        let text = m.to_text();
        // one expert moved without fixing the hash: the content hash differs
        let lines: Vec<String> = text.lines().map(String::from).collect();
        let li = lines.iter().position(|l| l.starts_with("L ")).unwrap();
        let mut edited = lines.clone();
        let mut row: Vec<char> = edited[li].chars().collect();
        let n = row.len();
        // swap the first two characters that differ, a count-preserving edit
        let a = 4;
        let b = (a + 1..n).find(|&i| row[i] != row[a]).unwrap();
        row.swap(a, b);
        edited[li] = row.into_iter().collect();
        let err = ExpertMap::parse(&edited.join("\n")).unwrap_err();
        assert!(err.contains("hash"), "{err}");
        // a count-breaking edit is refused by the counts
        let mut skew = lines.clone();
        skew[li] = format!("L {} {}", skew[li].split(' ').nth(1).unwrap(), "0".repeat(512));
        let err = ExpertMap::parse(&skew.join("\n")).unwrap_err();
        assert!(err.contains("holds") && err.contains("equal counts"), "{err}");
        // a truncated row, a rank outside the group, a wrong header, a wrong group size
        assert!(ExpertMap::parse("rexl3-expert-map v2\n").unwrap_err().contains("v1"));
        assert!(ExpertMap::parse(&text.replace("ranks 2", "ranks 3")).is_err());
        assert!(ExpertMap::parse(&text.lines().take(10).collect::<Vec<_>>().join("\n")).unwrap_err().contains("layers"));
        let err = m.check_for(&c, 3).unwrap_err();
        assert!(err.contains("2 ranks") && err.contains("3"), "{err}");
        let mut partial = BTreeMap::new();
        partial.insert(1usize, m.row(1).unwrap().to_vec());
        let p = ExpertMap::new(2, 512, partial, None).unwrap();
        assert!(p.check_for(&c, 2).unwrap_err().contains("no row for MoE layer"));
        let mut odd: BTreeMap<usize, Vec<u8>> = layers.iter().map(|&l| (l, m.row(l).unwrap().to_vec())).collect();
        odd.insert(0usize, m.row(1).unwrap().to_vec());
        assert!(ExpertMap::new(2, 512, odd, None).unwrap().check_for(&c, 2).unwrap_err().contains("not a MoE layer"));
    }

    #[test]
    fn a_head_row_is_carried_and_hashed() {
        let row: Vec<u8> = (0..512).map(|e| (e % 2) as u8).collect();
        let mut layers = BTreeMap::new();
        layers.insert(1usize, row.clone());
        let without = ExpertMap::new(2, 512, layers.clone(), None).unwrap();
        let with = ExpertMap::new(2, 512, layers, Some(row)).unwrap();
        assert!(with.has_head() && !without.has_head());
        assert_ne!(with.hash(), without.hash());
        assert_eq!(ExpertMap::parse(&with.to_text()).unwrap(), with);
        assert_eq!(with.head_owner(3), Some(1));
        assert_eq!(without.head_owner(3), None);
    }

    #[test]
    fn a_trace_round_trips_and_regroups_into_windows() {
        let layers = [1usize, 3];
        let t = synthetic("syn", 512, &layers, 100, 1);
        assert_eq!(RoutingTrace::from_bytes(&t.to_bytes()).unwrap(), t);
        assert!(RoutingTrace::from_bytes(&t.to_bytes()[..40]).is_err());
        let rec = steps_of(&[&t], Grouping::Recorded, 1);
        assert_eq!(rec.len(), t.windows.len());
        let w4 = steps_of(&[&t], Grouping::Window(4), 1);
        assert!(w4.iter().all(|s| s.len() == 1 && s[0].2 - s[0].1 == 4));
        assert_eq!(w4.len(), t.rows() / 4);
        // merging two prompts doubles the rows of a step
        let u = synthetic("syn2", 512, &layers, 100, 2);
        let m = steps_of(&[&t, &u], Grouping::Window(1), 2);
        assert_eq!(m.len(), 100);
        assert!(m.iter().all(|s| s.len() == 2));
    }

    #[test]
    fn the_solver_beats_contiguous_and_keeps_the_counts_and_the_objective_is_what_evaluate_says() {
        let layers = [1usize, 3, 7];
        let train = synthetic("a", 512, &layers, 600, 11);
        let test = synthetic("b", 512, &layers, 600, 12);
        for ranks in [2usize, 3, 4, 8] {
            let opts = SolveOpts { budget: Duration::from_secs(5), ..SolveOpts::default() };
            for cost in [Cost::Pairs, Cost::Passes] {
                let s = solve(&[&train], ranks, Grouping::Recorded, 1, cost, &opts, false).unwrap();
                s.map.check_for_layers(&layers, ranks, 512);
                let contiguous = ExpertMap::contiguous(ranks, 512, layers).unwrap();
                let ct = evaluate_map(&contiguous, &[&train], Grouping::Recorded, 1, cost);
                let bt = evaluate_map(&s.map, &[&train], Grouping::Recorded, 1, cost);
                let bh = evaluate_map(&s.map, &[&test], Grouping::Recorded, 1, cost);
                let ch = evaluate_map(&contiguous, &[&test], Grouping::Recorded, 1, cost);
                assert!(bt.busiest <= ct.busiest, "R={ranks} {cost:?}: train {} vs contiguous {}", bt.busiest, ct.busiest);
                // the skew is structural (hot low experts), so the held-out half improves too
                assert!(bh.busiest < ch.busiest, "R={ranks} {cost:?}: held-out {} vs contiguous {}", bh.busiest, ch.busiest);
                // the objective the solver reports is the evaluator's number
                let total: u64 = s.objective.values().sum();
                assert_eq!(total as f64, bt.busiest);
                // equal counts per rank, every layer
                for &l in &layers {
                    for r in 0..ranks {
                        assert_eq!(s.map.owned(l, r).len(), contiguous_count(512, r, ranks));
                    }
                }
            }
        }
    }

    #[test]
    fn the_solver_is_deterministic_and_lpt_is_a_valid_start() {
        let layers = [1usize];
        let t = synthetic("a", 512, &layers, 300, 5);
        let opts = SolveOpts { budget: Duration::from_secs(30), seed: 9, max_sweeps: 3, ..SolveOpts::default() };
        let a = solve(&[&t], 3, Grouping::Recorded, 1, Cost::Passes, &opts, false).unwrap();
        let b = solve(&[&t], 3, Grouping::Recorded, 1, Cost::Passes, &opts, false).unwrap();
        assert_eq!(a.map, b.map);
        let l = solve(&[&t], 3, Grouping::Recorded, 1, Cost::Passes, &opts, true).unwrap();
        assert!(l.objective[&1] >= a.objective[&1]);
    }

    #[test]
    fn without_a_configured_map_the_fingerprint_is_zero_and_the_plan_flag_is_clear() {
        // the unit tests run with REXL3_TP_EXPERT_MAP unset: the contiguous ranges, no table, nothing for the rendezvous to compare
        assert!(std::env::var_os(MAP_ENV).is_none());
        assert_eq!(expert_map_fingerprint(), 0);
        assert!(configured_map().unwrap().is_none());
        assert_eq!(super::super::tp_arm::plan_flags() & 0x10, 0);
        // a table's fingerprint is its hash and never 0 or the unusable marker
        let m = ExpertMap::contiguous(2, 512, [1usize]).unwrap();
        assert!(m.hash().max(1) != 0 && m.hash() != FINGERPRINT_UNUSABLE);
    }

    #[test]
    fn the_route_tap_is_off_by_default_and_collects_what_is_pushed() {
        assert!(!tap_enabled());
        tap_enable(true);
        tap_push(3, 1, vec![1; 22]);
        tap_push(7, 2, vec![2; 44]);
        let got = tap_take();
        assert_eq!(got.len(), 2);
        assert_eq!((got[1].0, got[1].1, got[1].2.len()), (7, 2, 44));
        assert!(tap_take().is_empty());
        tap_enable(false);
        assert!(!tap_enabled());
    }

    // ---- byte weights (v0.1.2 MixedK)

    /// Bytes of a routed expert pair of the pack's shapes (latent 1,024, intermediate 2,688) at `up` and `down` bits: the two trellises,
    /// the fp16 scales and the two codebook markers. `None` bits: a bf16 projection.
    fn pair_bytes(up: Option<usize>, down: Option<usize>) -> u64 {
        let proj = |bits: Option<usize>| -> u64 {
            match bits {
                Some(k) => (1024 * 2688 * k / 8 + 2 * (1024 + 2688) + 4) as u64,
                None => 1024 * 2688 * 2,
            }
        };
        proj(up) + proj(down)
    }

    /// A byte table: every layer of `layers` at 4 bits, the given experts overridden.
    fn table(layers: &[usize], over: &[(usize, usize, Option<usize>, Option<usize>)]) -> ExpertBytes {
        let mut t = ExpertBytes { experts: 512, ..Default::default() };
        for &l in layers {
            t.bytes.insert(l, vec![pair_bytes(Some(4), Some(4)); 512]);
            t.widths.insert(l, vec![4]);
        }
        for &(l, e, up, down) in over {
            t.bytes.get_mut(&l).unwrap()[e] = pair_bytes(up, down);
            let w = t.widths.get_mut(&l).unwrap();
            for k in [up, down].into_iter().flatten() {
                if !w.contains(&k) {
                    w.push(k);
                    w.sort_unstable();
                }
            }
        }
        t
    }

    /// The published packs' form: one width everywhere, a bf16 pair in layers 26, 39 and 81 (experts 111, 304, 461).
    fn one_width_with_bf16_pairs(layers: &[usize]) -> ExpertBytes {
        table(layers, &[(26, 111, None, None), (39, 304, None, None), (81, 461, None, None)])
    }

    #[test]
    fn a_pack_of_one_width_gets_no_map_its_bf16_pairs_included() {
        let c = cfg();
        let moe = moe_layers(&c);
        let t = one_width_with_bf16_pairs(&moe);
        assert!(t.mixed_layers().is_empty());
        assert!(moe.iter().all(|&l| t.weights(l).is_none()), "no layer is weighted");
        for ranks in [1usize, 2, 3, 4, 5, 8] {
            assert_eq!(byte_balanced_map(&t, ranks, &moe).unwrap(), None, "R={ranks}: the contiguous ranges, today's plan");
        }
        // where the ranks hold equal counts the balance itself would not move them either: one heavy expert cannot leave the fullest
        // rank without making another one the fullest
        let contiguous = |ranks: usize| -> Vec<u8> { (0..512).map(|e| contiguous_owner(512, e, ranks) as u8).collect() };
        for ranks in [2usize, 4, 8] {
            for l in [26usize, 39, 81, 1] {
                assert_eq!(balance_layer_bytes(&t.bytes[&l], ranks), contiguous(ranks), "R={ranks} layer {l}");
            }
        }
        // at three ranks (170 / 171 / 171 experts) it WOULD: the bf16 pair of layer 39 sits on a rank of 171 and moves to the rank of
        // 170. The gate on mixed widths is what keeps a pack of one width on today's plan at every rank count
        assert_ne!(balance_layer_bytes(&t.bytes[&39], 3), contiguous(3));
        assert_eq!(byte_balanced_map(&t, 3, &moe).unwrap(), None);
    }

    #[test]
    fn a_mixed_layer_is_balanced_by_bytes_with_the_contiguous_counts_and_the_others_stay_contiguous() {
        let c = cfg();
        let moe = moe_layers(&c);
        // layer 1: experts 0..64 at 6 bits and 64..96 at 2 / 3 bits (all in rank 0's range at two ranks), the rest at 4; every other layer
        // one width; layer 26 its bf16 pair too
        let mut over: Vec<(usize, usize, Option<usize>, Option<usize>)> = Vec::new();
        for e in 0..64 {
            over.push((1, e, Some(6), Some(6)));
        }
        for e in 64..96 {
            over.push((1, e, Some(2), Some(3)));
        }
        over.push((26, 111, None, None));
        let t = table(&moe, &over);
        assert_eq!(t.mixed_layers(), vec![1]);
        for ranks in [2usize, 3, 4, 8] {
            let m = byte_balanced_map(&t, ranks, &moe).unwrap().expect("a mixed layer moves experts");
            m.check_for(&c, ranks).unwrap();
            assert!(!m.has_head());
            let contiguous: Vec<u8> = (0..512).map(|e| contiguous_owner(512, e, ranks) as u8).collect();
            for &l in &moe {
                if l != 1 {
                    assert_eq!(m.row(l).unwrap(), contiguous.as_slice(), "R={ranks}: layer {l} stays contiguous");
                }
                for r in 0..ranks {
                    assert_eq!(m.owned(l, r).len(), contiguous_count(512, r, ranks));
                }
            }
            let before = t.rank_bytes(1, &contiguous, ranks);
            let after = t.rank_bytes(1, m.row(1).unwrap(), ranks);
            assert_eq!(before.iter().sum::<u64>(), after.iter().sum::<u64>(), "every byte once");
            let spread = |v: &[u64]| v.iter().max().unwrap() - v.iter().min().unwrap();
            assert!(after.iter().max() < before.iter().max(), "R={ranks}: {before:?} -> {after:?}");
            // the balance stops when no swap of a heavier expert for a lighter one fits under the fullest rank: within the widest gap of
            // the layer's byte values (6/6 against 2/3 bits), and within one expert where the counts differ by one (R = 3: 170 / 171 / 171)
            let slack = if 512 % ranks == 0 {
                pair_bytes(Some(6), Some(6)) - pair_bytes(Some(2), Some(3))
            } else {
                pair_bytes(Some(6), Some(6))
            };
            assert!(spread(&after) <= slack, "R={ranks}: {after:?}");
            assert!(spread(&after) < spread(&before), "R={ranks}: {before:?} -> {after:?}");
            // deterministic
            assert_eq!(byte_balanced_map(&t, ranks, &moe).unwrap().unwrap(), m);
            // the totals over every layer: where the counts are equal (the layers of one width weigh the same on every rank) the fullest
            // rank is lower; at three ranks the ranks of 171 experts are the fuller ones on every other layer, and evening out layer 1
            // moves bytes onto them (the balance is per layer, for the layer's read; no group of three ranks runs on a fabric)
            let (tb, ta) = (t.rank_totals(None, ranks), t.rank_totals(Some(&m), ranks));
            assert_eq!(tb.iter().sum::<u64>(), ta.iter().sum::<u64>());
            if 512 % ranks == 0 {
                assert!(ta.iter().max() < tb.iter().max(), "R={ranks}: {tb:?} -> {ta:?}");
            }
        }
    }

    #[test]
    fn the_layer_balance_gets_within_one_step_on_a_random_mixed_layer_and_terminates() {
        let mut rng = Rng(77);
        for ranks in [2usize, 3, 4, 8] {
            for _ in 0..8 {
                let bytes: Vec<u64> = (0..512).map(|_| pair_bytes(Some(2 + rng.below(5)), Some(2 + rng.below(5)))).collect();
                let row = balance_layer_bytes(&bytes, ranks);
                let mut sums = vec![0u64; ranks];
                let mut counts = vec![0usize; ranks];
                for (e, &r) in row.iter().enumerate() {
                    sums[r as usize] += bytes[e];
                    counts[r as usize] += 1;
                }
                for (r, &n) in counts.iter().enumerate() {
                    assert_eq!(n, contiguous_count(512, r, ranks));
                }
                let max = *sums.iter().max().unwrap();
                let min = *sums.iter().min().unwrap();
                let step = (1024 * 2688 / 8) as u64;
                if 512 % ranks == 0 {
                    assert!(max - min <= step, "R={ranks}: {sums:?}");
                } else {
                    assert!(max - min <= pair_bytes(Some(6), Some(6)), "R={ranks}: {sums:?}");
                }
            }
        }
    }

    #[test]
    fn the_weighted_solver_is_the_equal_cost_solver_on_layers_of_one_width() {
        let layers = [1usize, 26, 39];
        let train = synthetic("a", 512, &layers, 400, 21);
        // one width everywhere, with heavy bf16 experts among the hot ones: still the equal-cost problem, bit for bit
        let t = table(&layers, &[(26, 3, None, None), (39, 1, None, None), (1, 0, None, None)]);
        assert!(t.mixed_layers().is_empty());
        // the sweeps bound the work; the wall-clock budget must never be what stops a solve (the result would depend on the machine)
        let opts = SolveOpts { budget: Duration::from_secs(600), seed: 4, max_sweeps: 4, ..SolveOpts::default() };
        for ranks in [2usize, 3] {
            for cost in [Cost::Pairs, Cost::Passes] {
                for lpt_only in [false, true] {
                    let plain = solve(&[&train], ranks, Grouping::Recorded, 1, cost, &opts, lpt_only).unwrap();
                    let wt = solve_weighted(&[&train], ranks, Grouping::Recorded, 1, cost, &opts, lpt_only, Some(&t)).unwrap();
                    assert_eq!(plain.map, wt.map, "R={ranks} {cost:?} lpt_only={lpt_only}");
                    assert_eq!(plain.objective, wt.objective);
                    let (a, b) = (
                        evaluate_map(&plain.map, &[&train], Grouping::Window(4), 1, cost),
                        evaluate_map_weighted(&plain.map, &[&train], Grouping::Window(4), 1, cost, Some(&t)),
                    );
                    assert_eq!(a, b);
                }
            }
        }
    }

    #[test]
    fn the_weighted_solver_balances_what_a_mixed_layer_reads() {
        let layers = [1usize];
        let train = synthetic("a", 512, &layers, 600, 31);
        // the hot low experts at 6 bits, the rest at 2 to 3: an equal count of picks is not an equal read
        let mut over = Vec::new();
        for e in 0..512 {
            over.push(if e < 96 { (1, e, Some(6), Some(6)) } else { (1, e, Some(2), Some(3)) });
        }
        let t = table(&layers, &over);
        assert_eq!(t.mixed_layers(), vec![1]);
        let w = t.weights(1).unwrap();
        assert_eq!(w.iter().min(), Some(&WEIGHT_UNIT));
        assert!(w[0] > 2 * WEIGHT_UNIT, "a 6-bit pair weighs about 2.4 of a 2/3-bit one: {}", w[0]);
        // as above: the sweeps (stopping at the first without a gain) bound the work, never the wall clock
        let opts = SolveOpts { budget: Duration::from_secs(600), seed: 4, max_sweeps: 12, ..SolveOpts::default() };
        for ranks in [2usize, 4] {
            for cost in [Cost::Pairs, Cost::Passes] {
                let plain = solve(&[&train], ranks, Grouping::Recorded, 1, cost, &opts, false).unwrap();
                let wt = solve_weighted(&[&train], ranks, Grouping::Recorded, 1, cost, &opts, false, Some(&t)).unwrap();
                let score = |m: &ExpertMap| evaluate_map_weighted(m, &[&train], Grouping::Recorded, 1, cost, Some(&t));
                let (sp, sw) = (score(&plain.map), score(&wt.map));
                assert!(sw.busiest <= sp.busiest, "R={ranks} {cost:?}: weighted {} vs equal-cost {}", sw.busiest, sp.busiest);
                // the reported objective is the weighted evaluator's, in WEIGHT_UNIT units
                let total: u64 = wt.objective.values().sum();
                assert_eq!(total as f64, sw.busiest * WEIGHT_UNIT as f64);
                for r in 0..ranks {
                    assert_eq!(wt.map.owned(1, r).len(), contiguous_count(512, r, ranks));
                }
            }
        }
    }

    impl ExpertMap {
        /// Test helper: the table covers exactly `layers` with the contiguous counts.
        fn check_for_layers(&self, layers: &[usize], ranks: usize, experts: usize) {
            assert_eq!(self.ranks(), ranks);
            assert_eq!(self.experts(), experts);
            assert_eq!(self.layers().collect::<Vec<_>>(), layers.to_vec());
        }
    }
}
