//! Serving settings the family memory models are evaluated at.
//!
//! Each family answers the planner's questions through `gb_sched::MemoryModel` (see `families::qwen` and `families::glm`); what
//! they are evaluated at (window and prefill rows, context length, whether the pack's own MTP head drafts, whether images are
//! served) is the same for every family and lives here. The activation reserve of a family is `max(its blanket, modelled
//! buffers)`: the blanket (5 GiB, what the Qwen engine has always held back) is never lowered, and a model whose scratch plus
//! window records outgrow it raises the reserve instead of overcommitting.

/// Vision encodes are serialised under one lock, so one encoder's scratch is in use at a time.
pub const ENCODERS_IN_FLIGHT: u64 = 1;

/// Serving settings the memory model depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryOptions {
    /// Rows of the widest decode or verify window (all sequences together).
    pub window_rows: usize,
    /// Rows of a prefill chunk.
    pub prefill_rows: usize,
    /// Longest context the server accepts.
    pub max_context: usize,
    /// Draft with the pack's own MTP head (GLM): its cache is charged per token. Refused for a pack with no MTP layer.
    pub mtp: bool,
    /// Serve images. `None`: text only, the tower is not loaded and nothing is reserved for it.
    pub vision: Option<VisionOptions>,
    /// Qwen: tree windows are enabled (`--spec-tree`), so their buffers (node K/V per attention layer, the plan, path and commit tables)
    /// are counted with the window records. Off: nothing is counted and nothing is allocated.
    pub trees: bool,
}

/// Limits on images, from the server (`--max-image-tokens`, and its per-request image count).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionOptions {
    /// Most language-model tokens one image may take.
    pub max_image_tokens: usize,
    /// Most images in one request.
    pub max_images_per_request: usize,
}

impl VisionOptions {
    /// The server's defaults: 4,096 tokens per image, 4 images per request (`gb_server::vision::Limits`).
    pub const SERVER_DEFAULT: VisionOptions = VisionOptions { max_image_tokens: 4096, max_images_per_request: 4 };

    /// Feature buffers that can be alive at once: every slot may hold a request of `max_images_per_request` images.
    pub fn concurrent_images(&self, slots: usize) -> u64 {
        slots as u64 * self.max_images_per_request as u64
    }
}

impl MemoryOptions {
    /// The shipped serving defaults: a 64-row window (`forward::WINDOW_ROWS`), 4,096-row prefill chunks, a 262,144-token context.
    pub fn serving_defaults() -> Self {
        Self {
            window_rows: crate::forward::WINDOW_ROWS,
            prefill_rows: 4096,
            max_context: 262_144,
            mtp: false,
            vision: Some(VisionOptions::SERVER_DEFAULT),
            trees: false,
        }
    }
}
