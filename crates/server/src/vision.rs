//! The request-side half of image input: find the images in a chat request, check them against the limits, and expand the chat
//! template's single `<|image_pad|>` per image into the right number of pad tokens (one per merged 2x2 block of patches).
//! Decoding and encoding happen elsewhere (`gb-vision`, `gb-model`); everything here is pure and tested without a GPU.

use crate::api::ApiError;
use gb_vision::{decode, VisionError};
use serde_json::Value;

/// `<|image_pad|>` in the Qwen3.8 vocabulary.
pub const IMAGE_PAD: u32 = 248056;

/// What a request may carry (TensorFold's policy, `docs/VISION.md`): images per request, encoded bytes per image, and the
/// language-model tokens one image may occupy (the encoder's attention is quadratic in patches).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_images: usize,
    pub max_bytes: usize,
    pub max_tokens_per_image: usize,
    pub max_total_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self { max_images: 4, max_bytes: 10 << 20, max_tokens_per_image: 4096, max_total_bytes: 40 << 20 }
    }
}

/// Where an image sits in the request, for error messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRef {
    pub message: usize,
    pub part: usize,
    /// The `data:` URI (or whatever the client sent; non-data URLs are refused when decoded).
    pub url: String,
}

pub fn is_image_part(part: &Value) -> bool {
    // the chat template's own test, so the pads it renders and the images found here are always the same set
    part.get("image").is_some() || part.get("image_url").is_some() || part.get("type").and_then(|t| t.as_str()) == Some("image")
}

fn url_str(part: &Value) -> Option<&str> {
    let v = part.get("image_url").or_else(|| part.get("image"))?;
    match v {
        Value::String(s) => Some(s.as_str()),
        Value::Object(o) => o.get("url").and_then(|u| u.as_str()),
        _ => None,
    }
}
fn url_of(part: &Value) -> Option<String> {
    url_str(part).map(str::to_string)
}

/// Size the retained copies before allocating ImageRefs or entering the wait queue.
pub fn image_input_reserve(
    messages: &Value,
    limits: &Limits,
    is_image: impl Fn(&Value) -> bool,
) -> Result<Option<usize>, ApiError> {
    let mut retained = 0usize;
    let mut count = 0usize;
    for (i, m) in messages.as_array().into_iter().flatten().enumerate() {
        for (j, part) in m.get("content").and_then(|c| c.as_array()).into_iter().flatten().enumerate() {
            if !is_image(part) {
                continue;
            }
            count += 1;
            if count > limits.max_images {
                return Err(ApiError::bad(format!("at most {} images per request", limits.max_images)));
            }
            let url =
                url_str(part).ok_or_else(|| ApiError::bad(format!("messages[{i}].content[{j}]: an image part needs a url")))?;
            let max_url = if url.starts_with("data:") {
                limits
                    .max_bytes
                    .checked_add(2)
                    .and_then(|n| n.checked_div(3))
                    .and_then(|n| n.checked_mul(4))
                    .and_then(|n| n.checked_add(64))
            } else {
                Some(2048)
            };
            if max_url.is_none_or(|max| url.len() > max) {
                return Err(ApiError::status(
                    413,
                    "invalid_request_error",
                    format!("messages[{i}].content[{j}]: image URL exceeds size limit"),
                ));
            }
            retained = retained
                .checked_add(url.len())
                .and_then(|n| n.checked_add(limits.max_bytes))
                .and_then(|n| n.checked_add(64))
                .ok_or_else(|| ApiError::bad("image retained size overflow"))?;
        }
    }
    Ok((count > 0).then_some(retained))
}

/// Every image part of `messages`, in order of appearance (the order the template renders its pads in), by the Qwen template's own
/// test of what an image part is.
pub fn extract_images(messages: &Value, limits: &Limits) -> Result<Vec<ImageRef>, ApiError> {
    extract_images_with(messages, limits, is_image_part)
}

/// [`extract_images`] with the model family's own test of what its template renders as an image (`ChatFormat::is_image_part`).
pub fn extract_images_with(
    messages: &Value,
    limits: &Limits,
    is_image: impl Fn(&Value) -> bool,
) -> Result<Vec<ImageRef>, ApiError> {
    let mut out = Vec::new();
    for (i, m) in messages.as_array().into_iter().flatten().enumerate() {
        for (j, part) in m.get("content").and_then(|c| c.as_array()).into_iter().flatten().enumerate() {
            if !is_image(part) {
                continue;
            }
            let url = url_of(part)
                .ok_or_else(|| ApiError::bad(format!("messages[{i}].content[{j}]: an image part needs a `url` (a data: URI)")))?;
            if out.len() == limits.max_images {
                return Err(ApiError::bad(format!("at most {} images per request", limits.max_images)));
            }
            out.push(ImageRef { message: i, part: j, url });
        }
    }
    Ok(out)
}

fn vision_error(r: &ImageRef, e: VisionError) -> ApiError {
    ApiError::bad(format!("messages[{}].content[{}]: {e}", r.message, r.part))
}

/// Decode and preprocess one image, enforcing the size limits before the encoder sees it. `pp` is the family's preprocessing
/// (`gb_vision::preprocessor::ImagePreprocessor`): its decode (alpha and orientation rules), its `plan` (the canvas and the number of placeholder tokens,
/// from the size alone, so an image over the token limit is refused before the resize) and its `preprocess`.
pub fn prepare(
    r: &ImageRef,
    limits: &Limits,
    pp: &dyn gb_vision::preprocessor::ImagePreprocessor,
) -> Result<gb_vision::Preprocessed, ApiError> {
    if r.url.len() > limits.max_bytes / 3 * 4 + 64 {
        return Err(ApiError::bad(format!(
            "messages[{}].content[{}]: image is larger than {} bytes",
            r.message, r.part, limits.max_bytes
        )));
    }
    let bytes = decode::data_uri_bytes(&r.url).map_err(|e| vision_error(r, e))?;
    if bytes.len() > limits.max_bytes {
        return Err(ApiError::bad(format!(
            "messages[{}].content[{}]: image is larger than {} bytes",
            r.message, r.part, limits.max_bytes
        )));
    }
    prepare_bytes(r, &bytes, limits, pp, None)
}

/// Remote and data images use exactly the same decoder and preprocessing pipeline.
pub fn prepare_bytes(
    r: &ImageRef,
    bytes: &[u8],
    limits: &Limits,
    pp: &dyn gb_vision::preprocessor::ImagePreprocessor,
    work: Option<&crate::preparation::RequestWork>,
) -> Result<gb_vision::Preprocessed, ApiError> {
    let check = || work.map_or(Ok(()), |w| w.check());
    check()?;
    if bytes.len() > limits.max_bytes {
        return Err(ApiError::status(
            413,
            "invalid_request_error",
            format!("messages[{}].content[{}]: image is larger than {} bytes", r.message, r.part, limits.max_bytes),
        ));
    }
    let img = pp.decode(bytes).map_err(|e| vision_error(r, e))?;
    check()?;
    // refuse before the (expensive) resize when the canvas would exceed the token budget
    let plan = pp.plan(img.width, img.height).map_err(|e| vision_error(r, e))?;
    if plan.tokens > limits.max_tokens_per_image {
        return Err(ApiError::bad(format!(
            "messages[{}].content[{}]: the image would occupy {} tokens ({}x{} after resizing); the limit is {}",
            r.message, r.part, plan.tokens, plan.canvas.0, plan.canvas.1, limits.max_tokens_per_image
        )));
    }
    check()?;
    let pre = pp.preprocess(&img).map_err(|e| vision_error(r, e))?;
    check()?;
    Ok(pre)
}

/// One image's placeholder run in the expanded prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// Index of the first pad token in the prompt.
    pub start: usize,
    pub tokens: usize,
}

/// Replace the i-th `<|image_pad|>` of `ids` with `counts[i]` copies and report where each run landed. The rendered prompt has exactly
/// one pad per image; any other count (a user typing the literal token, a template change) is refused.
pub fn expand_pads(ids: &[u32], counts: &[usize]) -> Result<(Vec<u32>, Vec<Span>), ApiError> {
    expand_placeholders(ids, IMAGE_PAD, "<|image_pad|>", counts)
}

/// [`expand_pads`] for any family: `token` (spelled `literal` in text) is the placeholder the family's template renders once per image.
pub fn expand_placeholders(ids: &[u32], token: u32, literal: &str, counts: &[usize]) -> Result<(Vec<u32>, Vec<Span>), ApiError> {
    let pads = ids.iter().filter(|&&t| t == token).count();
    if pads != counts.len() {
        return Err(ApiError::bad(format!(
            "the prompt has {pads} image placeholders but {} images were sent (the literal text {literal} is not allowed in messages)",
            counts.len()
        )));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut spans = Vec::with_capacity(counts.len());
    let mut k = 0;
    for &t in ids {
        if t == token {
            spans.push(Span { start: out.len(), tokens: counts[k] });
            out.extend(std::iter::repeat_n(token, counts[k]));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok((out, spans))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn finds_images_in_template_order_across_url_shapes() {
        let m = json!([
            {"role": "user", "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                {"type": "input_image", "image_url": "data:image/png;base64,BB"},
                {"type": "image", "image": "data:image/png;base64,CC"},
            ]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,DD"}}]},
        ]);
        let v = extract_images(&m, &Limits { max_images: 9, ..Limits::default() }).unwrap();
        assert_eq!(v.iter().map(|r| (r.message, r.part)).collect::<Vec<_>>(), vec![(0, 1), (0, 2), (0, 3), (2, 0)]);
        assert_eq!(v[1].url, "data:image/png;base64,BB");
        assert!(extract_images(&m, &Limits::default()).is_ok(), "four images is exactly the default limit");
        assert!(extract_images(&m, &Limits { max_images: 3, ..Limits::default() }).is_err());
    }

    #[test]
    fn an_image_part_without_a_url_is_an_error() {
        let m = json!([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]);
        let e = extract_images(&m, &Limits::default()).unwrap_err();
        assert!(e.message.contains("messages[0].content[0]"), "{}", e.message);
    }

    #[test]
    fn pads_expand_per_image_and_spans_are_reported() {
        let ids = vec![1, 2, IMAGE_PAD, 3, IMAGE_PAD, 4];
        let (out, spans) = expand_pads(&ids, &[3, 2]).unwrap();
        assert_eq!(out, vec![1, 2, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, 3, IMAGE_PAD, IMAGE_PAD, 4]);
        assert_eq!(spans, vec![Span { start: 2, tokens: 3 }, Span { start: 6, tokens: 2 }]);
        assert!(expand_pads(&ids, &[3]).is_err(), "more placeholders than images");
        assert!(expand_pads(&[1, 2], &[3]).is_err(), "more images than placeholders");
        assert_eq!(expand_pads(&[1, 2], &[]).unwrap(), (vec![1, 2], vec![]));
    }

    #[test]
    fn limits_and_bad_data_are_clear_400s() {
        let pp = gb_vision::Preprocessing::qwen38();
        let r = |url: &str| ImageRef { message: 0, part: 1, url: url.to_string() };
        let e = prepare(&r("https://example.com/a.png"), &Limits::default(), &pp).unwrap_err();
        assert!(e.message.contains("data: URIs") && e.message.contains("messages[0].content[1]"), "{}", e.message);
        let e = prepare(&r("data:image/png;base64,AAAA"), &Limits::default(), &pp).unwrap_err();
        assert_eq!(e.status, 400);
        let tiny = Limits { max_bytes: 100, ..Limits::default() };
        let big = format!("data:image/png;base64,{}", "A".repeat(400));
        assert!(prepare(&r(&big), &tiny, &pp).unwrap_err().message.contains("larger than"));
    }

    #[test]
    fn a_real_png_is_preprocessed_and_the_token_budget_is_enforced() {
        let pp = gb_vision::Preprocessing::qwen38();
        let bytes =
            std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision/check_512x512.png"))
                .unwrap();
        let uri = format!("data:image/png;base64,{}", b64(&bytes));
        let r = ImageRef { message: 0, part: 0, url: uri };
        let p = prepare(&r, &Limits::default(), &pp).unwrap();
        assert_eq!((p.grid.1, p.grid.2, p.n_tokens(&pp)), (32, 32, 256));
        let tight = Limits { max_tokens_per_image: 255, ..Limits::default() };
        assert!(prepare(&r, &tight, &pp).unwrap_err().message.contains("256 tokens"));
    }

    #[test]
    fn the_glm_preprocessing_goes_through_the_same_prepare_and_its_own_canvas_decides_the_tokens() {
        // 512x512: a canvas of 532x532 (aligned up to 28), 38x38 patches, 361 tokens; the fork's profile, padded on the right and bottom
        let pp = gb_vision::glm::GlmPreprocessing::default();
        let bytes =
            std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision/check_512x512.png"))
                .unwrap();
        let r = ImageRef { message: 0, part: 0, url: format!("data:image/png;base64,{}", b64(&bytes)) };
        let limits = Limits { max_tokens_per_image: 8000, ..Limits::default() };
        let p = prepare(&r, &limits, &pp).unwrap();
        assert_eq!((p.grid, p.size, p.patch_dim, p.n_patches() / 4), ((1, 38, 38), (532, 532), 1176, 361));
        // the placeholder count the prompt is expanded with is the plan's, and is what the preprocessing made
        assert_eq!(gb_vision::preprocessor::ImagePreprocessor::plan(&pp, 512, 512).unwrap().tokens, 361);
        // a limit under it refuses before the resize, naming the numbers
        let tight = Limits { max_tokens_per_image: 360, ..limits };
        let e = prepare(&r, &tight, &pp).unwrap_err();
        assert!(
            e.message.contains("361 tokens") && e.message.contains("532x532") && e.message.contains("limit is 360"),
            "{}",
            e.message
        );
        // the same image under the Qwen preprocessing is still 256 tokens through the same function
        let q = gb_vision::Preprocessing::qwen38();
        assert_eq!(prepare(&r, &Limits::default(), &q).unwrap().n_tokens(&q), 256);
    }

    fn b64(b: &[u8]) -> String {
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::new();
        for c in b.chunks(3) {
            let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
            s.push(T[(n >> 18) as usize & 63] as char);
            s.push(T[(n >> 12) as usize & 63] as char);
            s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
            s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
        }
        s
    }
}
