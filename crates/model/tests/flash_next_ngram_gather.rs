//! The n-gram table of the real Flash-Next pack (`REXL3_FLASH_NEXT_DIR`; CPU only, the table is mmap'd): a prompt chunk's gather, which
//! runs on several threads, equals the gather of each position on its own (one position, the caller's thread), row for row. Prints
//! "not run" without the pack.
use gb_model::families::qwen_flash_next::config::FlashNextConfig;
use gb_model::families::qwen_flash_next::ngram::NgramTable;
use std::path::PathBuf;
use std::sync::Arc;

#[test]
fn a_chunks_threaded_gather_equals_its_positions_gathered_one_at_a_time() {
    let Some(dir) = std::env::var_os("REXL3_FLASH_NEXT_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_FLASH_NEXT_DIR is unset");
        return;
    };
    let cfg = FlashNextConfig::from_dir(&dir).unwrap();
    let t = NgramTable::open(&dir, &cfg).unwrap();
    let per = t.heads * t.words;
    // 1,024 positions after two context tokens, ids spread over the vocabulary (rows all over the table)
    let history: Vec<u32> = (0..1026u32).map(|i| i.wrapping_mul(2_654_435_761) % (cfg.vocab as u32 - 300) + 3).collect();
    let n = history.len() - 2;
    let mut chunk = vec![0u16; n * per];
    t.gather(&history, 2, &mut chunk).unwrap();
    for p in 0..n {
        let mut one = vec![0u16; per];
        t.gather(&history[p..p + 3], 2, &mut one).unwrap();
        assert_eq!(&chunk[p * per..(p + 1) * per], &one[..], "position {p}");
    }
    assert!(chunk.iter().any(|&w| w != 0));
}

/// A read-ahead of the next chunk's pages (started before, and running beside, that chunk's gather) changes no row.
#[test]
fn a_gather_beside_a_read_ahead_of_its_own_pages_equals_its_positions_gathered_one_at_a_time() {
    let Some(dir) = std::env::var_os("REXL3_FLASH_NEXT_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_FLASH_NEXT_DIR is unset");
        return;
    };
    let cfg = FlashNextConfig::from_dir(&dir).unwrap();
    let t = Arc::new(NgramTable::open(&dir, &cfg).unwrap());
    let per = t.heads * t.words;
    let history: Vec<u32> = (0..2050u32).map(|i| i.wrapping_mul(2_246_822_519) % (cfg.vocab as u32 - 300) + 3).collect();
    let n = history.len() - 2;
    t.readahead(history.clone());
    let mut chunk = vec![0u16; n * per];
    t.gather(&history, 2, &mut chunk).unwrap();
    for p in 0..n {
        let mut one = vec![0u16; per];
        t.gather(&history[p..p + 3], 2, &mut one).unwrap();
        assert_eq!(&chunk[p * per..(p + 1) * per], &one[..], "position {p}");
    }
    assert!(chunk.iter().any(|&w| w != 0));
}
