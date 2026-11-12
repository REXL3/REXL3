//! The DeepSeek-V4.1-Flash image serve path, CPU-verifiable parts (P3 T3.1-T3.2): the server's vision gate opens
//! once the tower is a built component, the preprocessing plans the reference's token counts, and an image span in
//! a prompt produces exactly the splice rows and the engram DEAD mask the model's op walk consumes.
//!
//! What this file cannot check without a device: the tower's numerics (the kernels of
//! `csrc/dsv4_vision.cu`, gated by `crates/cuda/tests/dsv4_vision.rs` on a GPU box) and the engine's boot of them
//! (`engine::build_engine` needs a CUDA context). The gate, the plan and the span are the parts a CPU can hold.

use gb_model::families::deepseek_v41::vision::{SpanPlan, TokenType, IMAGE_TOKEN_ID, OUT_HIDDEN};
use gb_model::families::deepseek_v41::vision_serve::{
    dead_ids, image_key, preprocessing_key, Dsv41ImageFeatures, Dsv41Splice, SpanEmbeddings,
};
use gb_model::families::deepseek_v41::DeepseekV41Family;
use gb_model::family::{ModelFamily, Registry};
use gb_model::serve::ServeOptions;
use std::sync::Arc;

/// The pack's `config.json`, committed as the family's fixture.
const CONFIG: &str = include_str!("../../../testdata/deepseek_v41/config.json");

fn pack_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("rexl3-dsv41-serve-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    dir
}

/// The gate: the tower is a built component, so a serve asked for images (`o.vision` set, what the server does
/// unless `--no-vision`) passes preflight instead of being refused with "image input needs vision_deepseek".
#[test]
fn an_image_request_is_no_longer_refused_once_the_tower_is_bound() {
    let dir = pack_dir("gate");
    let plan = DeepseekV41Family.probe(&dir).unwrap();
    assert_eq!(plan.unbuilt_vision(), None, "vision_deepseek is a built component");
    // the gate the server's preflight runs, with the tower asked for
    let mut o = ServeOptions::new(dir.clone());
    o.vision = Some(Default::default());
    let format = gb_model::serve::preflight_in(Registry::global(), &o)
        .unwrap_or_else(|e| panic!("an image serve must not be refused once the tower is bound: {e}"));
    assert_eq!(format, "deepseek_v41");
    // and the component gate is still closed on the rest of the family
    assert!(plan.missing_components().is_empty());
    // the VisionSpec the server plans the slot from
    let v = plan.vision.as_ref().unwrap();
    assert_eq!(
        (v.tower, v.patch_size, v.merge_size, v.out_hidden, v.image_token_id),
        ("vision_deepseek", 14, 3, 5120, IMAGE_TOKEN_ID)
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// A text-only serve (`--no-vision`) was never refused and still is not.
#[test]
fn a_text_only_serve_passes_the_same_gate() {
    let dir = pack_dir("text");
    let o = ServeOptions::new(dir.clone());
    assert!(o.vision.is_none());
    assert_eq!(gb_model::serve::preflight_in(Registry::global(), &o).unwrap(), "deepseek_v41");
    std::fs::remove_dir_all(dir).unwrap();
}

/// The preprocessing the family hands the server plans the reference's own numbers (`ds41ref/image_processor.py`'s
/// `plan_image_grid`, transcribed and run on the same inputs): the canvas, the patch grid and — what the prompt
/// expands to — the token count, which is exactly the `SpanPlan` the model splices with.
#[test]
fn the_preprocessor_and_the_span_plan_agree_on_the_token_counts() {
    let dir = pack_dir("plan");
    let o = ServeOptions::new(dir.clone());
    let pre = DeepseekV41Family
        .image_preprocessor(&dir, &o)
        .unwrap()
        .expect("the pack has a vision_config, so the family has a preprocessor");
    // (w, h) -> (canvas, vit grid, tokens), from the reference
    let cases = [
        (800, 600, ((812, 602), (43, 58), 317), "pads to the patch grid"),
        (100, 80, ((616, 490), (35, 44), 194), "a small image scales up to min_pixels"),
        (640, 640, ((644, 644), (46, 46), 274), "a square"),
        (4096, 3072, ((1456, 1092), (78, 104), 938), "a big image shrinks inside the budget"),
        (1920, 1080, ((1708, 966), (69, 122), 968), "a wide one"),
        (200, 1000, ((252, 1218), (87, 18), 205), "a tall one"),
    ];
    for (w, h, (canvas, vit, tokens), why) in cases {
        let p = pre.plan(w, h).unwrap();
        assert_eq!((p.canvas, (p.grid.1, p.grid.2), p.tokens), (canvas, vit, tokens), "{why}: {w}x{h}");
        // the SpanPlan the model splices with, over the same grid, costs the same positions
        let span = SpanPlan::new(0, vit.0, vit.1).unwrap();
        assert_eq!(span.len(), p.tokens, "{why}: the span and the plan agree");
        assert_eq!(span.n_llm_h(), (p.grid.1).div_ceil(3), "{why}");
        assert_eq!(span.n_llm_w(), (p.grid.2).div_ceil(3), "{why}");
        assert_eq!(span.types().len(), p.tokens, "{why}: every position has a type");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// The full prompt with an image: the span's types compose with the text around them, every span position carries
/// the placeholder id, the engram's ids are DEAD across the span and alive outside it, and the splice block writes
/// the aligner's rows and the delimiter embeddings in reading order across every hc stream — the layout
/// `model.py`'s `merge_image_embeddings` produces before the hc expansion repeats it.
#[test]
fn a_prompt_with_an_image_splices_and_masks_exactly_the_span() {
    const HC: usize = 4;
    let d = OUT_HIDDEN;
    // "look <image> speak": text, a 2 x 3 merged grid (6 patches of 3 x 3), text
    let span = SpanPlan::new(4, 6, 9).unwrap();
    assert_eq!((span.n_llm_h(), span.n_llm_w()), (2, 3));
    let types = span.types();
    // the prompt's token types and ids
    let mut prompt_types = vec![TokenType::Text as i8; 4];
    prompt_types.extend(&types);
    prompt_types.extend([TokenType::Text as i8; 2]);
    let ids: Vec<u32> = (0..4u32).chain(std::iter::repeat_n(IMAGE_TOKEN_ID, span.len())).chain([100, 101]).collect();
    assert_eq!(ids.len(), prompt_types.len());

    // 1. the engram's mask: `None` on a text-only prompt, `false` across the span otherwise
    let mask = gb_cuda::dsv4_vision::engram_mask(Some(&prompt_types)).expect("an image is a mask");
    assert_eq!(mask[..4], [true, true, true, true]);
    assert_eq!(mask[4..span.end()], vec![false; span.len()]);
    assert_eq!(mask[span.end()..], [true, true]);
    assert!(gb_cuda::dsv4_vision::engram_mask(None).is_none(), "the text-only fast path");
    // the image mask the text attention reads is its complement
    assert_eq!(gb_cuda::dsv4_vision::image_mask(&prompt_types), mask.iter().map(|b| !b).collect::<Vec<_>>());

    // 2. the ids the engram gathers over a chunk that holds the whole prompt: DEAD across the span
    let feats = Arc::new(Dsv41ImageFeatures {
        rows: (0..6).flat_map(|r| std::iter::repeat_n((r + 1) as f32, d)).collect(),
        tokens: 6,
        merged: (2, 3),
        key: image_key(7, (812, 602), preprocessing_key("t")),
        permit: None,
    });
    let whole = Dsv41Splice::whole(feats.clone(), span.start).unwrap();
    assert_eq!(whole.n, span.len());
    let dead = dead_ids(
        &(ids.iter().map(|&t| t as i32).collect::<Vec<i32>>()),
        &[Dsv41Splice::whole(feats.clone(), span.start).unwrap()],
    );
    assert_eq!(dead[..4].to_vec(), vec![0, 1, 2, 3]);
    assert_eq!(dead[4..span.end()], vec![gb_model::families::deepseek_v41::engram::DEAD as i32; span.len()]);
    assert_eq!(dead[span.end()..], vec![100, 101]);

    // 3. the splice block: `[START] [1 2 3] [NL] [4 5 6] [NL] [END]`, every row in every hc stream
    let emb = SpanEmbeddings { start: vec![10f32; d], end: vec![20f32; d], newline: vec![30f32; d] };
    let block = whole.block(HC, &emb).unwrap();
    assert_eq!(block.len(), whole.n * HC * d);
    let head = |i: usize, c: usize| block[(i * HC + c) * d];
    for c in 0..HC {
        assert_eq!(head(0, c), 10.0, "IMAGE_START");
        assert_eq!([head(1, c), head(2, c), head(3, c)], [1.0, 2.0, 3.0], "aligner row 0");
        assert_eq!(head(4, c), 30.0, "IMAGE_NEW_LINE");
        assert_eq!([head(5, c), head(6, c), head(7, c)], [4.0, 5.0, 6.0], "aligner row 1");
        assert_eq!(head(8, c), 30.0);
        assert_eq!(head(9, c), 20.0, "IMAGE_END");
    }
    // and the host twin of the same splice writes the same rows into a plain [L, d] stream
    let mut h = vec![f32::NAN; (span.end() + 1) * d];
    gb_model::families::deepseek_v41::vision::splice_into(
        &mut h,
        &span,
        &gb_model::families::deepseek_v41::vision::SpanFeatures {
            rows: &feats.rows,
            image_start: &emb.start,
            image_end: &emb.end,
            image_newline: &emb.newline,
        },
    )
    .unwrap();
    // 4:START 5:A0 6:A1 7:A2 8:NL 9:A3 10:A4 11:A5 12:NL 13:END
    assert!(
        h[4 * d] == 10.0
            && h[5 * d] == 1.0
            && h[6 * d] == 2.0
            && h[7 * d] == 3.0
            && h[8 * d] == 30.0
            && h[9 * d] == 4.0
            && h[12 * d] == 30.0
            && h[13 * d] == 20.0,
        "the span itself"
    );
    assert!(h[3 * d].is_nan() && h[14 * d].is_nan(), "the text rows are untouched");
}

/// A chunk that starts inside a span keeps its place: the features' rows continue in reading order and the
/// delimiters still land where the span's layout says (the backend clips every span to the chunk it runs).
#[test]
fn a_chunk_that_cuts_a_span_splices_from_where_it_starts() {
    let d = OUT_HIDDEN;
    let feats = Arc::new(Dsv41ImageFeatures {
        rows: (0..6).flat_map(|r| std::iter::repeat_n((r + 1) as f32, d)).collect(),
        tokens: 6,
        merged: (2, 3),
        key: 0,
        permit: None,
    });
    let whole = Dsv41Splice::whole(feats, 2).unwrap();
    // the chunk holds prompt rows 6..11: span positions 4..8 — [NL] [4 5 6] [NL]; the span's END (row 11) is
    // the next chunk's
    let cut = whole.clip(6, 11).unwrap();
    assert_eq!((cut.dst, cut.src, cut.n), (0, 4, 5));
    let emb = SpanEmbeddings { start: vec![0f32; d], end: vec![20f32; d], newline: vec![30f32; d] };
    let b = cut.block(1, &emb).unwrap();
    let head = |i: usize| b[i * d];
    assert_eq!([head(0), head(1), head(2), head(3), head(4)], [30.0, 4.0, 5.0, 6.0, 30.0]);
    // the END is where the next chunk picks up
    let rest = whole.clip(11, 12).unwrap();
    assert_eq!((rest.src, rest.n), (9, 1));
    assert_eq!(rest.block(1, &emb).unwrap()[0], 20.0);
    // a chunk entirely past the span holds none of it
    assert!(whole.clip(30, 40).is_none());
    // and a span that would run off its chunk is refused by the model, not truncated (checked in the model's
    // `prefill_chunk_mm`; here: the clipped splice never claims rows outside its n)
    assert!(cut.dst + cut.n <= 5);
}
