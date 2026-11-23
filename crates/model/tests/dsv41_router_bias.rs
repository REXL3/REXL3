use gb_model::families::deepseek_v41::vision_serve::{image_mask_rows, Dsv41ImageFeatures, Dsv41Splice};
use std::sync::Arc;

fn splice(dst: usize) -> Dsv41Splice {
    let features = Arc::new(Dsv41ImageFeatures {
        rows: vec![0.0; 2 * gb_model::families::deepseek_v41::vision::OUT_HIDDEN],
        tokens: 2,
        merged: (2, 1),
        key: 0,
        permit: None,
    });
    Dsv41Splice::whole(features, dst).unwrap()
}

#[test]
fn router_bias_mask_marks_only_image_rows() {
    let span = splice(3);
    let span_len = span.n;
    let mask = image_mask_rows(16, &[span]).unwrap();
    assert_eq!(&mask[..3], &[0, 0, 0]);
    assert_eq!(&mask[3..3 + span_len], vec![1; span_len]);
    assert_eq!(&mask[3 + span_len..], vec![0; 13 - span_len]);
    assert!(image_mask_rows(16, &[]).is_none());
}

#[test]
fn clipped_router_bias_mask_tracks_the_chunk_relative_span() {
    let whole = splice(2);
    let clipped = whole.clip(4, 8).unwrap();
    let mask = image_mask_rows(6, &[clipped]).unwrap();
    assert_eq!(mask, vec![1, 1, 1, 1, 0, 0]);
}
