//! An env-gated, host-only log of the candidates a GLM backend hands the engine (`REXL3_GLM_DIGEST_LOG=PATH`): one line per answered row,
//! `GEN POS TOKEN IDS VALS` (the sequence's number in begin order, the row's position and token, FNV-1a 64 of the candidate ids and of the
//! value bits). The comparison of a tensor-parallel group with one node (`tools/glm_tp_compare.py`, `docs/GLM_TP.md` 6) diffs the logs
//! of the same requests sent one at a time. Unset (the default) it is one branch per call and writes nothing: it reads what the engine is
//! given and changes no bit of it.

use gb_engine::RowOutput;
use std::io::Write;

pub const ENV: &str = "REXL3_GLM_DIGEST_LOG";

pub struct DigestLog {
    f: Option<std::fs::File>,
    next: u64,
    gen: Vec<u64>,
}

fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

impl DigestLog {
    /// The log of `REXL3_GLM_DIGEST_LOG`, appended to; off when unset (or when the file cannot be opened, said once).
    pub fn from_env(slots: usize) -> DigestLog {
        let f = std::env::var(ENV).ok().filter(|p| !p.is_empty()).and_then(|p| {
            match std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                Ok(f) => {
                    eprintln!("rexl3 event=glm_digest_log path={p}");
                    Some(f)
                }
                Err(e) => {
                    eprintln!("rexl3 level=warn event=glm_digest_log_off path={p} error=\"{e}\"");
                    None
                }
            }
        });
        DigestLog { f, next: 0, gen: vec![0; slots] }
    }

    /// A log appended to `path` (the tests' seam; `from_env` is this with the variable's value).
    pub fn to_path(path: &std::path::Path, slots: usize) -> std::io::Result<DigestLog> {
        let f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        Ok(DigestLog { f: Some(f), next: 0, gen: vec![0; slots] })
    }

    pub fn on(&self) -> bool {
        self.f.is_some()
    }

    /// A sequence starts in `slot` (the next number).
    pub fn begin(&mut self, slot: usize) {
        if self.f.is_none() {
            return;
        }
        self.next += 1;
        if let Some(g) = self.gen.get_mut(slot) {
            *g = self.next;
        }
    }

    /// The row at `pos` of `slot`'s sequence (its token `token`) was answered with `out`.
    pub fn row(&mut self, slot: usize, pos: usize, token: u32, out: &RowOutput) {
        let Some(f) = self.f.as_mut() else { return };
        let ids = out.ids.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &i| fnv(h, &i.to_le_bytes()));
        let vals = out.values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| fnv(h, &v.to_bits().to_le_bytes()));
        let g = self.gen.get(slot).copied().unwrap_or(0);
        let _ = writeln!(f, "{g} {pos} {token} {ids:016x} {vals:016x}");
    }
}
