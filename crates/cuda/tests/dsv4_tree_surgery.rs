//! The tree round's device surface moves (T2.5a lane 2): the byte surgery a draft-tree round's commit runs
//! over the planes' window ring and the compressors' carry — the moves `gb_model`'s tree host makes, stated
//! in `gb_cuda` so both the model and this test exercise the same code.
//!
//! GPU-only in practice (`DevBuf` needs a device); the battery's CPU runs print `not run`:
//!
//! ```text
//! REXL3_REQUIRE_GPU=1 cargo test --release -p gb-cuda --test dsv4_tree_surgery -- --nocapture --test-threads=1
//! ```

use gb_cuda::dsv4_attn::{
    compressor_carry_restore, compressor_carry_save, window_ring_restore, window_ring_save, window_ring_write, Dsv4Config,
    Dsv4Planes,
};
use gb_cuda::rt::{device_available, DevBuf, Stream};

fn dev() -> Option<Stream> {
    if !device_available() {
        println!("not run: no device");
        return None;
    }
    Stream::new().ok()
}

fn require_gpu() -> bool {
    std::env::var("REXL3_REQUIRE_GPU").is_ok_and(|v| v == "1")
}

/// Distinct u16 patterns per ring slot, written through the same ring kernel the forward uses.
fn seed_ring(planes: &mut Dsv4Planes, d: usize, st: &Stream, base: u16) {
    let rows: Vec<u16> = (0..planes.window_slots * d).map(|i| base.wrapping_add(i as u16)).collect();
    let src = DevBuf::from_slice(&rows).expect("src");
    gb_cuda::dsv4_attn::ring_write_dev(&src, 0, &planes.window, planes.window_slots, d, 0, planes.window_slots, st)
        .expect("ring write");
}

fn ring_of(planes: &Dsv4Planes) -> Vec<u16> {
    planes.window.to_vec::<u16>().expect("d2h")
}

/// The ring moves round-trip, including a region that WRAPS the ring: the base slots saved before the path
/// windows, a sibling's rows written over them, then the commit's restore + the accepted rows' rewrite leave
/// exactly the serial replay's bytes.
#[test]
fn the_ring_surgery_round_trips_across_the_wrap() {
    let Some(st) = dev() else {
        assert!(!require_gpu(), "REXL3_REQUIRE_GPU=1 but the ring surgery test found no CUDA device");
        return;
    };
    let cfg = Dsv4Config::v41();
    let mut planes = Dsv4Planes::new(&cfg, 512, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).expect("planes");
    let d = cfg.head_dim;
    let rb = d * 2;
    seed_ring(&mut planes, d, &st, 0x1000);
    let base = ring_of(&planes);
    // an anchor at 125 of a 128-slot ring: the 8 touched positions wrap (slots 125, 126, 127, 0, 1, ...)
    let p0 = planes.window_slots - 3;
    let mut save = DevBuf::zeroed_view(8 * rb).expect("save");
    window_ring_save(&planes, p0, 8, &mut save, &st).expect("save");
    st.sync().expect("sync");
    // a sibling path's rows clobber the touched slots (written as a linear block the way window_bak's tail is)
    let sibling: Vec<u16> = (0..8 * d).map(|i| 0xA000u16.wrapping_add(i as u16)).collect();
    let sib = DevBuf::from_slice(&sibling).expect("sibling");
    window_ring_write(&mut planes, d, p0, 8, &sib, &st).expect("sibling write");
    st.sync().expect("sync");
    for j in 0..8 {
        let slot = (p0 + j) % planes.window_slots;
        assert_eq!(&ring_of(&planes)[slot * d..slot * d + d], &sibling[j * d..j * d + d], "the sibling's row {j} is in");
    }
    // the commit: base back, then the ACCEPTED path's own first 3 rows over it
    let accepted: Vec<u16> = (0..3 * d).map(|i| 0xB000u16.wrapping_add(i as u16)).collect();
    let acc = DevBuf::from_slice(&accepted).expect("accepted");
    window_ring_restore(&mut planes, p0, 8, &save, &st).expect("restore");
    window_ring_write(&mut planes, d, p0, 3, &acc, &st).expect("accepted write");
    st.sync().expect("sync");
    let after = ring_of(&planes);
    for j in 0..3 {
        let slot = (p0 + j) % planes.window_slots;
        assert_eq!(&after[slot * d..slot * d + d], &accepted[j * d..j * d + d], "accepted row {j}");
    }
    for j in 3..8 {
        let slot = (p0 + j) % planes.window_slots;
        assert_eq!(&after[slot * d..slot * d + d], &base[slot * d..slot * d + d], "rejected row {j}'s slot is the BASE bytes");
    }
    // and every untouched slot is the base
    for slot in 0..planes.window_slots {
        if (p0..p0 + 8).any(|p| p % planes.window_slots == slot) {
            continue;
        }
        assert_eq!(&after[slot * d..slot * d + d], &base[slot * d..slot * d + d], "untouched slot {slot}");
    }
}

/// The carry moves: slot 0's carried rows saved out of the state buffers and restored to the byte, at the
/// offsets `MAX_SLOTS` slots apart.
#[test]
fn the_carry_surgery_saves_and_restores_the_slot_rows() {
    let Some(st) = dev() else {
        assert!(!require_gpu(), "REXL3_REQUIRE_GPU=1 but the carry surgery test found no CUDA device");
        return;
    };
    let cfg = Dsv4Config::v41();
    let d = cfg.head_dim;
    let ratio = cfg.ratio(2).max(2) as usize;
    let bytes = ratio * d * 4;
    // the state buffers sized as the compressor allocates them (MAX_SLOTS slots of ratio rows)
    let mut kv_state = DevBuf::zeroed(gb_cuda::dsv4_attn::MAX_SLOTS * ratio * d * 4).expect("kv");
    let mut score_state = DevBuf::zeroed(gb_cuda::dsv4_attn::MAX_SLOTS * ratio * d * 4).expect("score");
    let kv_rows: Vec<f32> = (0..gb_cuda::dsv4_attn::MAX_SLOTS * ratio * d).map(|i| i as f32 * 0.5).collect();
    let score_rows: Vec<f32> = (0..gb_cuda::dsv4_attn::MAX_SLOTS * ratio * d).map(|i| -(i as f32)).collect();
    kv_state.upload(&kv_rows).expect("upload");
    score_state.upload(&score_rows).expect("upload");
    let mut kv_save = DevBuf::zeroed_view(bytes).expect("kv save");
    let mut score_save = DevBuf::zeroed_view(bytes).expect("score save");
    for slot in [0usize, 1, 5] {
        compressor_carry_save(&kv_state, &score_state, ratio, d, slot, &mut kv_save, &mut score_save, &st).expect("save");
        // clobber the slot's rows, then restore and compare to the byte
        let junk = DevBuf::from_slice(&vec![42.0f32; ratio * d]).expect("junk");
        kv_state.copy_from_at(slot * ratio * d * 4, &junk, 0, junk.bytes(), &st).expect("clobber");
        compressor_carry_restore(&mut kv_state, &mut score_state, ratio, d, slot, &kv_save, &score_save, &st).expect("restore");
        st.sync().expect("sync");
        let mut back = vec![0f32; ratio * d];
        kv_state.download_range(slot * ratio * d * 4, &mut back).expect("d2h");
        assert_eq!(back, kv_rows[slot * ratio * d..slot * ratio * d + ratio * d], "slot {slot}'s carry rows restored");
        let mut sback = vec![0f32; ratio * d];
        score_state.download_range(slot * ratio * d * 4, &mut sback).expect("d2h");
        assert_eq!(sback, score_rows[slot * ratio * d..slot * ratio * d + ratio * d], "slot {slot}'s score rows restored");
    }
}
