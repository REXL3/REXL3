//! The MTP tap dump, a calibration hook: with `REXL3_NEMO_TAP_DUMP=<dir>` the model appends, per sequence, the rows the MTP head is fed
//! (the trunk's post-`norm_f` rows, f32) and the token id at each row's position, for every prefill slice (`mtp_prefill_hook`) and every
//! committed decode or verify window (`mtp_commit_taps`, the kept rows). The 4-bit head's quantizer reads them to build each matrix's input
//! Hessian from the head's own forward (`tools/nemotron_omni/h4_head_calib.py`). Off by default: unset, [`enabled`] is one load of a
//! once-initialised flag and nothing else runs. On, each record costs a device sync and a download: a calibration run, never serving.
//!
//! One file per process, `<dir>/taps-<pid>.bin`, a sequence of records, little-endian:
//!
//! ```text
//! magic  b"RXTD"         4 bytes
//! kind   u32             0 = prefill slice, 1 = committed window rows
//! slot   u32
//! first  u32             the position of the first row (the sequence's length before the rows)
//! rows   u32
//! width  u32             4096
//! ids    i32 [rows]      the token at each row's position (row r is position first + r)
//! taps   f32 [rows][width]  post-norm_f rows: the head pairs the tap of position p with the token at p + 1
//! ```
//!
//! A slot's sequence restarts with a record whose `first` is 0. Nothing in the file is an evaluation set: what goes through the server is
//! the caller's choice.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// The variable naming the dump directory.
pub const TAP_DUMP_ENV: &str = "REXL3_NEMO_TAP_DUMP";
const MAGIC: &[u8; 4] = b"RXTD";

/// What produced a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DumpKind {
    Prefill = 0,
    Window = 1,
}

static DUMP: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();

fn dump() -> &'static Option<Mutex<BufWriter<File>>> {
    DUMP.get_or_init(|| {
        let dir = PathBuf::from(std::env::var_os(TAP_DUMP_ENV)?);
        if dir.as_os_str().is_empty() {
            return None;
        }
        let open = || -> std::io::Result<File> {
            std::fs::create_dir_all(&dir)?;
            File::create(dir.join(format!("taps-{}.bin", std::process::id())))
        };
        match open() {
            Ok(f) => {
                eprintln!(
                    "rexl3 level=warn event=nemo_tap_dump dir={} (calibration only: every record syncs the stream)",
                    dir.display()
                );
                Some(Mutex::new(BufWriter::with_capacity(8 << 20, f)))
            }
            Err(e) => {
                eprintln!("rexl3 level=warn event=nemo_tap_dump_failed dir={} why={e} (no dump)", dir.display());
                None
            }
        }
    })
}

/// Whether the dump is on (`REXL3_NEMO_TAP_DUMP` names a directory the process could open its file in). Read once per process.
pub fn enabled() -> bool {
    dump().is_some()
}

/// One record's bytes (the layout in the module documentation).
pub fn encode(kind: DumpKind, slot: usize, first: usize, ids: &[i32], taps: &[f32], width: usize) -> Vec<u8> {
    assert_eq!(taps.len(), ids.len() * width, "one tap row per id");
    let mut v = Vec::with_capacity(24 + ids.len() * 4 + taps.len() * 4);
    v.extend_from_slice(MAGIC);
    for x in [kind as u32, slot as u32, first as u32, ids.len() as u32, width as u32] {
        v.extend_from_slice(&x.to_le_bytes());
    }
    ids.iter().for_each(|i| v.extend_from_slice(&i.to_le_bytes()));
    taps.iter().for_each(|t| v.extend_from_slice(&t.to_le_bytes()));
    v
}

/// A decoded record (the reader the tests and tools mirror).
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub kind: u32,
    pub slot: u32,
    pub first: u32,
    pub ids: Vec<i32>,
    pub width: usize,
    pub taps: Vec<f32>,
}

/// Every record of a dump file's bytes, or the offset of the first malformed one.
pub fn decode(mut b: &[u8]) -> Result<Vec<Record>, usize> {
    let total = b.len();
    let mut out = Vec::new();
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
    while !b.is_empty() {
        let at = total - b.len();
        if b.len() < 24 || &b[..4] != MAGIC {
            return Err(at);
        }
        let (kind, slot, first, rows, width) =
            (u32_at(b, 1), u32_at(b, 2), u32_at(b, 3), u32_at(b, 4) as usize, u32_at(b, 5) as usize);
        let need = 24 + rows * 4 + rows * width * 4;
        if b.len() < need {
            return Err(at);
        }
        let ids = b[24..24 + rows * 4].chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let taps = b[24 + rows * 4..need].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        out.push(Record { kind, slot, first, ids, width, taps });
        b = &b[need..];
    }
    Ok(out)
}

/// Appends one record when the dump is on (a write error is logged once per call and drops the record: calibration, not serving).
pub fn write(kind: DumpKind, slot: usize, first: usize, ids: &[i32], taps: &[f32], width: usize) {
    let Some(m) = dump() else { return };
    let bytes = encode(kind, slot, first, ids, taps, width);
    let mut f = m.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = f.write_all(&bytes).and_then(|_| f.flush()) {
        eprintln!("rexl3 level=warn event=nemo_tap_dump_write_failed why={e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_round_trips_and_a_truncated_or_foreign_one_is_refused_at_its_offset() {
        let (w, ids) = (4usize, [11, -1, 7]);
        let taps: Vec<f32> = (0..ids.len() * w).map(|i| i as f32 * 0.5 - 1.0).collect();
        let a = encode(DumpKind::Prefill, 3, 0, &ids, &taps, w);
        assert_eq!(a.len(), 24 + 3 * 4 + 12 * 4);
        let b = encode(DumpKind::Window, 3, 3, &ids[..1], &taps[..w], w);
        let both = [a.clone(), b.clone()].concat();
        let r = decode(&both).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].kind, r[0].slot, r[0].first, r[0].ids.clone(), r[0].width), (0, 3, 0, ids.to_vec(), w));
        assert_eq!(r[0].taps, taps);
        assert_eq!((r[1].kind, r[1].first, r[1].ids.clone(), r[1].taps.clone()), (1, 3, vec![11], taps[..w].to_vec()));
        assert_eq!(decode(&both[..both.len() - 1]), Err(a.len()), "a truncated record");
        let mut bad = both.clone();
        bad[a.len()] = b'X';
        assert_eq!(decode(&bad), Err(a.len()), "a foreign magic");
        assert_eq!(decode(&[]), Ok(vec![]));
    }

    #[test]
    fn the_dump_is_off_unless_the_variable_names_a_directory() {
        // the test processes never set it: nothing is written, nothing is synced
        if std::env::var_os(TAP_DUMP_ENV).is_none() {
            assert!(!enabled());
            write(DumpKind::Prefill, 0, 0, &[1], &[0.0; 4], 4);
        }
    }
}
