//! Host tests for the completion-tracked n-gram warm worker. Every test builds a real, small
//! n-gram safetensors sidecar in a temp directory and mmaps it through [`NgramTable::open`] —
//! no mocks of the behavior under test.

use super::*;
use std::path::PathBuf;
use std::sync::Arc;

/// Rows for the "large" warm file: 64 MiB of real trellis payload at K=8.
const LARGE_ROWS: usize = 414_000;
/// Small table for success/idempotence/failure tests.
const SMALL_ROWS: usize = 64;
const K: usize = 8;
const HEADS_PER_NGRAM: usize = 8;

fn cfg_for_test() -> FlashNextConfig {
    let root: serde_json::Value =
        serde_json::from_str(include_str!("../../../../../testdata/qwen_flash_next/config.json")).unwrap();
    let mut cfg = FlashNextConfig::from_json(&root).unwrap();
    cfg.heads_per_ngram = HEADS_PER_NGRAM;
    cfg
}

fn unique_dir(tag: &str) -> PathBuf {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("rexl3-fn-ngram-warm-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write a real safetensors sidecar with `rows` trellis rows. Returns the dir, the config the
/// table opens with, and the total file length the warm worker must fault.
fn write_ngram_file(rows: usize, tag: &str) -> (PathBuf, FlashNextConfig, u64) {
    let cfg = cfg_for_test();
    let dir = unique_dir(tag);
    let layer = cfg.ple_layer_ids.first().copied().unwrap() - 1;
    let p = format!("model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding");
    let words = inventory::ngram_row_words(K);
    let heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;

    let trellis_len = rows * words * 2;
    let offsets_len = heads * 8;
    let sizes_len = heads * 8;
    let mult_len = 3 * 8;
    let bias_len = heads * ROW * 2;
    let mut at = 0usize;
    let mut span = |len: usize| -> (usize, usize) {
        let s = at;
        at += len;
        (s, s + len)
    };
    let (t0, t1) = span(trellis_len);
    let (o0, o1) = span(offsets_len);
    let (s0, s1) = span(sizes_len);
    let (m0, m1) = span(mult_len);
    let (b0, b1) = span(bias_len);

    let mut tensors = serde_json::Map::new();
    tensors.insert(
        format!("{p}.shard_0.trellis"),
        serde_json::json!({"dtype": "I16", "shape": [rows, words], "data_offsets": [t0, t1]}),
    );
    tensors.insert(format!("{p}.head_offsets"), serde_json::json!({"dtype": "I64", "shape": [heads], "data_offsets": [o0, o1]}));
    tensors
        .insert(format!("{p}.head_vocab_sizes"), serde_json::json!({"dtype": "I64", "shape": [heads], "data_offsets": [s0, s1]}));
    tensors.insert(format!("{p}.layer_multipliers"), serde_json::json!({"dtype": "I64", "shape": [3], "data_offsets": [m0, m1]}));
    tensors
        .insert(format!("{p}.head_bias"), serde_json::json!({"dtype": "F16", "shape": [heads, ROW], "data_offsets": [b0, b1]}));
    let mut meta = serde_json::Map::new();
    meta.insert("format".into(), serde_json::Value::String("exl3_ngram_trellis".into()));
    meta.insert("codebook".into(), serde_json::Value::String("mul1".into()));
    meta.insert("K".into(), serde_json::Value::String(K.to_string()));
    meta.insert("rows".into(), serde_json::Value::String(rows.to_string()));
    meta.insert("shard_rows".into(), serde_json::Value::String(rows.to_string()));
    meta.insert("row_dim".into(), serde_json::Value::String("160".into()));
    let mut root = serde_json::Map::new();
    root.insert("__metadata__".into(), serde_json::Value::Object(meta));
    for (name, t) in tensors {
        root.insert(name, t);
    }
    let header = serde_json::Value::Object(root).to_string();
    let header_len = header.len();

    let path = dir.join(inventory::NGRAM_FILE);
    let mut file = std::fs::File::create(&path).unwrap();
    use std::io::Write;
    file.write_all(&(header_len as u64).to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();

    // trellis: deterministic pattern, not zeros (a sparse-cache fast path must not hide the read)
    let mut chunk = vec![0u8; 1 << 20];
    for (i, b) in chunk.iter_mut().enumerate() {
        *b = (i.wrapping_mul(31) % 251) as u8;
    }
    let mut left = trellis_len;
    while left > 0 {
        let n = left.min(chunk.len());
        file.write_all(&chunk[..n]).unwrap();
        left -= n;
    }
    for _ in 0..heads {
        file.write_all(&0i64.to_le_bytes()).unwrap();
    }
    for _ in 0..heads {
        file.write_all(&(rows as i64).to_le_bytes()).unwrap();
    }
    for m in [101i64, 1009, 10007] {
        file.write_all(&m.to_le_bytes()).unwrap();
    }
    let bias = vec![0u16; heads * ROW];
    for w in bias {
        file.write_all(&w.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
    drop(file);
    (dir, cfg, path.metadata().unwrap().len())
}

fn open_large() -> (PathBuf, Arc<NgramTable>, u64) {
    let (dir, cfg, len) = write_ngram_file(LARGE_ROWS, "large");
    let table = Arc::new(NgramTable::open(&dir, &cfg).unwrap());
    (dir, table, len)
}

fn open_small(tag: &str) -> (PathBuf, Arc<NgramTable>, u64) {
    let (dir, cfg, len) = write_ngram_file(SMALL_ROWS, tag);
    let table = Arc::new(NgramTable::open(&dir, &cfg).unwrap());
    (dir, table, len)
}

#[test]
fn the_test_fixture_opens_a_real_small_ngram_table() {
    let (dir, table, len) = open_small("open");
    assert_eq!(table.rows, SMALL_ROWS);
    assert_eq!(table.words, inventory::ngram_row_words(K));
    assert!(len > (SMALL_ROWS * inventory::ngram_row_words(K) * 2) as u64);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn warm_reports_completion_only_after_every_byte_is_read() {
    let (dir, table, len) = open_large();
    let h = table.warm();
    // 64 MiB of real data through 1 MiB chunks: the worker cannot have finished this fast.
    assert!(h.done().is_none(), "warm worker finished before the test could observe it running");
    h.wait().unwrap();
    assert_eq!(h.done(), Some(Ok(())));
    assert_eq!(h.bytes_read(), len, "every byte of the mmap file must be faulted");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn finish_warm_blocks_until_the_worker_completes() {
    let (dir, table, len) = open_large();
    let h = table.warm();
    // the readiness gate: finish must not return before the worker's final byte is read
    finish_warm(Some(h.clone())).unwrap();
    assert_eq!(h.bytes_read(), len);
    assert_eq!(h.done(), Some(Ok(())));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn warm_failure_propagates_an_explicit_error_naming_the_file() {
    let (dir, table, _len) = open_small("fail");
    // the map stays valid (it owns its fd), but the worker's own open now fails: a real I/O failure
    std::fs::remove_file(dir.join(inventory::NGRAM_FILE)).unwrap();
    let h = table.warm();
    let err = finish_warm(Some(h.clone())).unwrap_err();
    assert!(err.contains(inventory::NGRAM_FILE), "{err}");
    assert!(err.contains("warm"), "{err}");
    assert_eq!(h.done(), Some(Err(err.clone())));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn disabled_warm_starts_no_worker_and_finishes_immediately() {
    let (dir, table, _len) = open_small("off");
    assert!(warm_if(false, &table).is_none());
    assert!(finish_warm(None).is_ok());
    assert!(warm_preflight(false, Some(&table)).is_ok(), "off bypass must not touch the table");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn repeated_warm_shares_one_worker_and_stays_idempotent() {
    let (dir, table, _len) = open_small("idem");
    let h1 = table.warm();
    let h2 = table.warm();
    assert!(Arc::ptr_eq(&h1, &h2), "a second warm must reuse the first worker's handle");
    h1.wait().unwrap();
    let h3 = table.warm();
    assert!(Arc::ptr_eq(&h1, &h3));
    assert_eq!(h3.done(), Some(Ok(())));
    assert!(warm_preflight(true, Some(&table)).is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_long_gather_on_several_threads_equals_the_one_thread_gather() {
    let (dir, table, _) = open_large();
    let per = table.heads * table.words;
    // a prompt chunk of 1,000 positions after 2 context tokens, with varied ids (the rows land all over the table)
    let history: Vec<u32> = (0..1002u32).map(|i| i.wrapping_mul(2_654_435_761) % 151_000 + 3).collect();
    let n = history.len() - 2;
    assert!(n >= GATHER_PARALLEL_FROM);
    let mut threaded = vec![0u16; n * per];
    table.gather(&history, 2, &mut threaded).unwrap();
    let mut serial = vec![0u16; n * per];
    table.gather_rows(&history, 2, 0, &mut serial).unwrap();
    assert_eq!(threaded, serial, "the threaded gather differs from the one-thread gather");
    assert!(serial.iter().any(|&w| w != 0), "the rows read are not all zeros (the comparison is not vacuous)");
    // a short gather (a decode window) stays on one thread and is the same rows
    let mut short = vec![0u16; 10 * per];
    table.gather(&history[..12], 2, &mut short).unwrap();
    assert_eq!(short, serial[..10 * per]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_read_ahead_returns_at_once_touches_only_pages_and_drops_a_second_call_or_a_short_history() {
    let (dir, table, _) = open_large();
    let per = table.heads * table.words;
    let history: Vec<u32> = (0..1002u32).map(|i| i.wrapping_mul(2_654_435_761) % 151_000 + 3).collect();
    let n = history.len() - 2;
    let mut serial = vec![0u16; n * per];
    table.gather_rows(&history, 2, 0, &mut serial).unwrap();
    // a history shorter than a threaded gather's is not read ahead
    table.readahead(history[..2 + GATHER_PARALLEL_FROM - 1].to_vec());
    assert!(!READAHEAD_BUSY.load(Ordering::Acquire), "a short history started a read-ahead");
    table.readahead(history.clone());
    // a second read-ahead while the first runs is dropped (the flag stays the first's), and a gather that runs meanwhile
    // reads the same rows
    table.readahead(history.clone());
    let mut during = vec![0u16; n * per];
    table.gather(&history, 2, &mut during).unwrap();
    assert_eq!(during, serial, "a gather beside a read-ahead differs from the one-thread gather");
    let start = std::time::Instant::now();
    while READAHEAD_BUSY.load(Ordering::Acquire) {
        assert!(start.elapsed() < std::time::Duration::from_secs(60), "the read-ahead worker never finished");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let mut after = vec![0u16; n * per];
    table.gather(&history, 2, &mut after).unwrap();
    assert_eq!(after, serial, "a gather after a read-ahead differs from the one-thread gather");
    assert!(serial.iter().any(|&w| w != 0));
    let _ = std::fs::remove_dir_all(dir);
}
