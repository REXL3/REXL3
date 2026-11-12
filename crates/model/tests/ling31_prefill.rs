//! The Ling-3.1 prefill on the whole model (the real pack, ~106 GiB resident, MTP on): what a prompt leaves in a slot, compared at the bit
//! level. `#[ignore]`, env-gated (`REXL3_LING_DIR`; synthetic token ids: row invariance does not depend on them), one model per process
//! (`--test-threads=1`), with a 1 GiB `MemAvailable` guard that aborts the process.
//!
//! * `a_prompt_leaves_the_same_bits_every_time`: the same prompt prefilled twice (another prompt in between, so the second run starts
//!   from dirty scratch, records and pages) leaves the same KDA states and conv histories, MLA rows, last-row logits, MTP absorb and
//!   drafts, and the same greedy continuation; the digests it prints are the record a build is compared on (`DIGEST <name> <hex>` lines:
//!   a change that claims to keep every bit prints the same). One slot: the profile's, and what fits beside the weights on a GB10.
//! * `embed_gather::host_gathered_rows_are_the_rows_the_kernel_gathers`: a synthetic test (no pack) that the host gather of the embedding rows equals the
//!   `gb_embed` kernel's, row for row.

mod ling31_support;

use ling31_support::*;

const SLOTS: usize = 1;
const KV_TOKENS: usize = 32 * 256;
/// Five chunks of 512 rows and a tail: the trunk, the MTP absorb and the tap ring all run more than one chunk.
const N: usize = 2600;
const STEPS: usize = 32;

fn model() -> Option<std::sync::MutexGuard<'static, gb_model::families::bailing_moe_v3::model::BailingModel>> {
    shared(SLOTS, KV_TOKENS, true)
}

/// Everything the prompt `toks` leaves in `slot`, prefilled cold in calls of `piece` tokens, as named digests.
fn run(
    m: &mut gb_model::families::bailing_moe_v3::model::BailingModel,
    slot: usize,
    toks: &[u32],
    piece: usize,
) -> Vec<(String, String)> {
    let n = toks.len();
    m.begin_slot(slot).unwrap();
    let first = prefill(m, slot, toks, 0, n, piece);
    let mut out = vec![("logits".to_string(), logits_digest(m))];
    let d = slot_digest(m, slot, n, 0);
    out.extend([("kda_state".into(), d.kda_state), ("kda_conv".into(), d.kda_conv), ("mla".into(), d.mla)]);
    out.push(("first".into(), format!("{:?} {:?}", first.ids, first.vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>())));
    // the MTP layer absorbs the prompt (rows 0..n-1, in chunks), then drafts from the first sampled token
    m.mtp_absorb_up_to(slot, toks).unwrap();
    out.push(("mtp_absorb".into(), slot_digest(m, slot, n, n - 1).mtp));
    let mut seq = toks.to_vec();
    seq.push(first.ids[0] as u32);
    let drafts = m.mtp_propose(slot, &seq, 3).unwrap();
    out.push(("drafts".into(), format!("{drafts:?}")));
    let (tokens, trace) = greedy(m, slot, toks, &first, STEPS);
    out.push(("greedy_tokens".into(), format!("{tokens:?}")));
    out.push(("greedy_trace".into(), trace));
    m.end_slot(slot);
    out
}

#[test]
#[ignore]
fn a_prompt_leaves_the_same_bits_every_time() {
    let Some(mut m) = model() else { return };
    let m = &mut *m;
    let toks = ids(1, N);
    let a = run(m, 0, &toks, N);
    let _ = run(m, 0, &ids(2, 1100), 1100);
    let b = run(m, 0, &toks, N);
    for (name, value) in &a {
        println!("DIGEST {name} {value}");
    }
    assert_eq!(a, b, "the same prompt left other bits the second time");
    eprintln!("MemAvailable low {:.2} GiB", gib(guard().low()));
}

/// The embedding rows a step needs, gathered on the host and copied through `HostStage::upload` (what `BailingModel` does for a table read
/// in place), against the `gb_embed` kernel that gathered them on the device before: the same bytes for every id, a zero row for an id
/// outside the table. The table is a file laid out like the real pack's (it starts 12 bytes past a 16-byte boundary, so the kernel takes its
/// scalar path), read through the mapping `EmbeddingSidecar::map_in_place` makes. Synthetic: needs a CUDA device, not the pack; skips
/// (and passes) without one.
mod embed_gather {
    use gb_cuda::rt::{DevBuf, HostStage, Stream};
    use gb_exl3::budget::Category;
    use gb_model::families::bailing_moe_v3::load::{gather_rows, EmbeddingSidecar};
    use gb_model::families::bailing_moe_v3::weights::{Dst, PlainDtype, PlainTensor, Src, Xform};

    const OFFSET: u64 = 7 * 4096 + 2060;
    const VOCAB: usize = 1500;
    const HIDDEN: usize = 64;

    fn table() -> Vec<u16> {
        (0..VOCAB * HIDDEN).map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_left(11) as u16).collect()
    }

    #[test]
    fn host_gathered_rows_are_the_rows_the_kernel_gathers() {
        if !gb_cuda::device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("ling31_embed_gather");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes: Vec<u8> = (0..OFFSET).map(|i| (i % 251) as u8).collect();
        let tab = table();
        for v in &tab {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend((0..333).map(|i| (i % 13) as u8));
        let files = vec!["model-00001-of-00001.safetensors".to_string()];
        std::fs::write(dir.join(&files[0]), &bytes).unwrap();
        let t = PlainTensor {
            name: "model.word_embeddings.weight".into(),
            dtype: PlainDtype::Bf16,
            shape: vec![VOCAB, HIDDEN],
            bytes: (VOCAB * HIDDEN * 2) as u64,
            xform: Xform::None,
            dst_bytes: (VOCAB * HIDDEN * 2) as u64,
            src: Src { file: 0, offset: OFFSET },
            dst: Dst { region: u32::MAX, offset: 0 },
            category: Category::Embedding,
        };
        let sidecar = EmbeddingSidecar::open(&dir, &files, &t, VOCAB, HIDDEN).unwrap();
        let mapped = sidecar.map_in_place().unwrap();
        let base = mapped.block().expect("a view of the mapping").host_ptr_at(0).cast_const();
        assert_eq!(base as usize % 16, 12, "the table is not 16-byte aligned, as in the real pack");
        // the device copy of the same bytes, which the kernel gathers from on any platform
        let st = Stream::new_nonblocking().unwrap();
        let mut stage = HostStage::new();
        let mut device_table = DevBuf::alloc(tab.len() * 2).unwrap();
        stage.upload(&mut device_table, &tab, &st).unwrap();
        stage.sync(&st).unwrap();
        // a decode round, a verify window, a chunk of a prefill (a batch over the WILLNEED threshold, repeats and bad ids among the rows)
        let mut seed = 12345u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        let long: Vec<i32> = (0..700)
            .map(|i| match i % 97 {
                5 => -1,
                6 => VOCAB as i32,
                7 => i32::MAX,
                _ => (next() % VOCAB) as i32,
            })
            .collect();
        for ids in [vec![0], vec![VOCAB as i32 - 1], vec![3, 3, 9, 1499], (0..8).map(|_| (next() % VOCAB) as i32).collect(), long]
        {
            let rows = ids.len();
            // the kernel, as the model ran it: ids on the device, rows written to a device buffer
            let mut ids_dev = DevBuf::alloc(rows * 4).unwrap();
            stage.upload(&mut ids_dev, &ids, &st).unwrap();
            let out_kernel = DevBuf::alloc(rows * HIDDEN * 2).unwrap();
            unsafe {
                gb_cuda::glue::embed(
                    device_table.ptr().cast_const(),
                    ids_dev.ptr().cast_const(),
                    out_kernel.ptr(),
                    rows,
                    HIDDEN,
                    VOCAB,
                    &st,
                )
                .unwrap();
            }
            // the host path, as `embed_rows_in_place` runs it: gathered into host scratch, one async copy through the stage
            let mut out_host = DevBuf::alloc(rows * HIDDEN * 2).unwrap();
            let mut scratch = vec![0u8; rows * HIDDEN * 2];
            // SAFETY: the mapping holds VOCAB * HIDDEN bf16 values
            unsafe { gather_rows(base, VOCAB, HIDDEN * 2, &ids, &mut scratch) };
            stage.upload(&mut out_host, &scratch, &st).unwrap();
            stage.sync(&st).unwrap();
            let (mut a, mut b) = (vec![0u16; rows * HIDDEN], vec![0u16; rows * HIDDEN]);
            out_kernel.download(&mut a).unwrap();
            out_host.download(&mut b).unwrap();
            assert!(a == b, "{rows} rows: the host gather differs from the kernel");
            for (r, &id) in ids.iter().enumerate() {
                let want: &[u16] =
                    if (0..VOCAB as i32).contains(&id) { &tab[id as usize * HIDDEN..][..HIDDEN] } else { &[0u16; HIDDEN] };
                assert_eq!(&b[r * HIDDEN..(r + 1) * HIDDEN], want, "row {r} of id {id}");
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// The bit-identity battery: writes the digest file named by `REXL3_BITS_OUT` for the prompts of `REXL3_BITS_PROMPTS` (see
/// `docs/LING_BIT_IDENTITY.md`; `tools/ling_bits.sh run`). It asserts nothing about the bits: two files are compared by `tools/ling_bits.py`.
#[test]
#[ignore]
fn ling_bits_battery() {
    bits::battery();
}

/// Layer localization of a difference the battery found: the residual stream after every layer, for the first layers `d = 1..=72` only.
#[test]
#[ignore]
fn ling_bits_localize() {
    bits::localize();
}
