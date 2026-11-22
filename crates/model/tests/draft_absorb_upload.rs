//! `Draft::absorb` and `Draft::propose_batch` must use reusable `HostStage`s, and their staged uploads must be ordered on
//! the model stream used by GLM and Flash-Next (and accepted by Generic Qwen's stream too). Every array must reach the next
//! kernel in order, and recycling the previous round's buffers must keep each stage bounded over a long sequence.
//!
//! The wiring regressions run in host-only CI. The race check needs CUDA; `REXL3_DRAFT_ABSORB_ITERS` changes its round
//! count (512 by default). No target or draft pack is needed.

/// Source assertions ignore formatting so rustfmt cannot make the wiring tests fail.
fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Keep this host-only regression beside the CUDA test: constructing a useful `Draft` requires its device buffers, but
/// replacing a staged upload with `DevBuf::upload` must still fail a named test in stub CI.
#[test]
fn draft_absorb_routes_both_index_arrays_through_its_reusable_stage() {
    let source = include_str!("../src/draft.rs");
    let absorb = source
        .split("pub fn absorb(")
        .nth(1)
        .and_then(|s| s.split("\n    /// Absorb several sequences").next())
        .expect("Draft::absorb body");
    assert!(
        compact(absorb).contains("self.absorb_batch(taps,&[Absorb{row0,n,slot,pos0}],stream)"),
        "Draft::absorb no longer uses the shared absorb_batch path"
    );

    let batch = source
        .split("pub fn absorb_batch(")
        .nth(1)
        .and_then(|s| s.split("\n    /// One draft round").next())
        .expect("Draft::absorb_batch body");
    let index_upload = batch.split("let Draft {").next().expect("index upload precedes the model work");
    assert!(
        compact(index_upload).contains(
            "stage_absorb_indices(&mutself.absorb_stage,&mutself.absorb_stage_stream,&mutself.ctx_pos,&mutself.ctx_slot,&pos,&slots,stream,)?"
        ),
        "Draft::absorb_batch must stage both index arrays through self.absorb_stage"
    );
    assert!(
        !index_upload.contains("ctx_pos.upload") && !index_upload.contains("ctx_slot.upload"),
        "direct DevBuf uploads are not retired by the draft's reusable HostStage"
    );
}

#[test]
fn draft_propose_routes_every_small_upload_through_its_reusable_stage() {
    let source = include_str!("../src/draft.rs");
    let propose = source.split("pub fn propose_batch_scored(").nth(1).expect("Draft::propose_batch_scored body");
    let uploads = propose.split("let Draft {").next().expect("small uploads precede the model work");
    let uploads = compact(uploads);
    for (buffer, values) in [
        ("ids", "ids"),
        ("pos", "pos"),
        ("anchor", "anchors"),
        ("slot_a", "slots"),
        ("p_a", "ps"),
        ("walk", "walks"),
        ("row_off", "row_off"),
    ] {
        let staged = format!("self.propose_stage.upload(&mutself.{buffer},&{values},stream)?");
        assert!(uploads.contains(&staged), "Draft::propose_batch_scored must stage {buffer} on the model stream");
        assert!(!uploads.contains(&format!("self.{buffer}.upload")), "{buffer} bypasses the reusable HostStage");
    }
    let propose = compact(propose);
    assert!(
        propose.contains("bind_stage_stream(&mutself.propose_stage_stream,stream);"),
        "the propose stage must retain and check its stream"
    );
    assert!(propose.contains("self.propose_stage.sync(stream)?;"), "the existing round sync must retire the propose stage");
}

#[cfg(not(rexl3_no_cuda))]
mod cuda {
    use core::ffi::c_void;
    use gb_cuda::glue;
    use gb_cuda::rt::HostStage;
    use gb_cuda::{DevBuf, Stream};
    use gb_model::draft::stage_absorb_indices;

    const ROWS: usize = 96;
    const DIM: usize = 32;
    const POS_VOCAB: usize = 8192;
    const SLOT_VOCAB: usize = 32;

    fn table(vocab: usize, salt: usize) -> Vec<u16> {
        (0..vocab * DIM).map(|i| 0x3f80 ^ (((i / DIM) * 31 + (i % DIM) + salt) & 0x3ff) as u16).collect()
    }

    fn indices(round: usize) -> (Vec<i32>, Vec<i32>) {
        let pos = (0..ROWS).map(|j| ((round * 977 + j * 131 + 1) % POS_VOCAB) as i32).collect();
        let slots = (0..ROWS).map(|j| ((round * 17 + j * 7 + 3) % SLOT_VOCAB) as i32).collect();
        (pos, slots)
    }

    fn gather(table: &[u16], ids: &[i32]) -> Vec<u16> {
        ids.iter().flat_map(|&id| table[id as usize * DIM..(id as usize + 1) * DIM].iter().copied()).collect()
    }

    #[test]
    fn absorb_indices_are_stream_ordered_and_the_stage_stays_bounded() {
        let iters = std::env::var("REXL3_DRAFT_ABSORB_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        let pos_table = table(POS_VOCAB, 0);
        let slot_table = table(SLOT_VOCAB, 19);
        let pos_table_d = DevBuf::from_slice(&pos_table).unwrap();
        let slot_table_d = DevBuf::from_slice(&slot_table).unwrap();
        let mut pos_d = DevBuf::zeroed(ROWS * 4).unwrap();
        let mut slot_d = DevBuf::zeroed(ROWS * 4).unwrap();
        let pos_out = DevBuf::zeroed(ROWS * DIM * 2).unwrap();
        let slot_out = DevBuf::zeroed(ROWS * DIM * 2).unwrap();
        let stream = Stream::new_nonblocking().unwrap();
        gb_cuda::rt::device_sync().unwrap();
        let mut stage = HostStage::new();
        let mut stage_stream = None;
        let mut got_pos = vec![0u16; ROWS * DIM];
        let mut got_slots = vec![0u16; ROWS * DIM];

        for round in 0..iters {
            let (pos, slots) = indices(round);
            stage_absorb_indices(&mut stage, &mut stage_stream, &mut pos_d, &mut slot_d, &pos, &slots, &stream).unwrap();
            assert_eq!(stage.queued(), 2, "round {round}: the previous pair was not retired and reused");
            // SAFETY: both tables are `[vocab, DIM]` bf16, both index arrays are i32 `[ROWS]`, and the outputs are bf16
            // `[ROWS, DIM]`. These are the first consumers' stream-ordering contract, without loading a multi-GB model pack.
            unsafe {
                glue::embed(
                    pos_table_d.ptr() as *const c_void,
                    pos_d.ptr() as *const c_void,
                    pos_out.ptr(),
                    ROWS,
                    DIM,
                    POS_VOCAB,
                    &stream,
                )
                .unwrap();
                glue::embed(
                    slot_table_d.ptr() as *const c_void,
                    slot_d.ptr() as *const c_void,
                    slot_out.ptr(),
                    ROWS,
                    DIM,
                    SLOT_VOCAB,
                    &stream,
                )
                .unwrap();
            }
            // Sync the stream, not the stage: the next absorb must perform the stage's retirement itself.
            stream.sync().unwrap();
            pos_out.download(&mut got_pos).unwrap();
            slot_out.download(&mut got_slots).unwrap();
            assert_eq!(got_pos, gather(&pos_table, &pos), "round {round}: positions arrived out of order");
            assert_eq!(got_slots, gather(&slot_table, &slots), "round {round}: slots arrived out of order");
        }
        stage.sync(&stream).unwrap();
        assert_eq!(stage.queued(), 0);
        assert_eq!(stage_stream, Some(stream.raw() as usize));
    }
}
