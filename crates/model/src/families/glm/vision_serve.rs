//! The vision tower's side of serving: what an encoded image is on the device ([`GlmImageFeatures`]), the bound on how many of them may be alive at once
//! ([`FeatureBudget`]), the rows an image writes over a prefill chunk's embeddings ([`SpliceOp`]) and the tower as the server's image encoder
//! ([`GlmEncoder`], `serve::ImageEncoder`). The tower itself, which knows nothing of requests, is `vision.rs`.

use super::vision::{GlmVisionTower, MERGE, OUT_HIDDEN, PATCHES_PER_TOKEN, PATCH_K};
use crate::error::ModelError;
use gb_cuda::{DevBuf, Stream};

/// An image's encoder output on the device: f32 `[tokens, 4096]` (the fork splices the merger's fp32 output as it is), one row per merged block of 2x2
/// patches in row-major order of the merged grid. Written once by the encoder and only read afterwards (by the engine thread's prefill), so sharing it
/// between the encoding thread and the engine thread is sound.
pub struct GlmImageFeatures {
    pub buf: DevBuf,
    pub tokens: usize,
    /// The merged grid (rows, columns); `tokens == merged.0 * merged.1`.
    pub merged: (usize, usize),
    /// The image's cache-key material: `sched::image_content_hash(pixel_hash, canvas width, canvas height, hash of the preprocessing id)`. The server hands
    /// the same value to the engine as `ImageSpan::key` (`http.rs`, from the same pixel hash and preprocessing id), which keys a prefix that contains
    /// these rows in the prefix cache (`docs/GLM53.md`, "Prefix cache"); this copy is for tests and tools.
    pub key: u64,
    /// The share of the image-feature budget this image holds until it is dropped (`None` outside the server's encoder).
    pub permit: Option<FeaturePermit>,
}

// SAFETY: the buffer is immutable after encoding and device memory is not tied to a host thread.
unsafe impl Sync for GlmImageFeatures {}

/// The bound on the image features alive at once and a claim on part of it: the shared types (`crate::feature_budget`, where they moved
/// verbatim), re-exported under this module's path.
pub use crate::feature_budget::{FeatureBudget, FeaturePermit};

/// One image's rows of features to write over the embedding rows of a prefill chunk: rows `src_row..src_row + n` of `feats` onto chunk rows
/// `dst_row..dst_row + n`. Holds the features so they outlive the forward that reads them.
#[derive(Clone)]
pub struct SpliceOp {
    pub feats: std::sync::Arc<GlmImageFeatures>,
    pub dst_row: usize,
    pub src_row: usize,
    pub n: usize,
}

/// The preprocessing's identity as a key: FNV-1a of `ImagePreprocessor::id()` (the family, the profile and every numeric parameter), only ever compared for
/// equality.
pub fn preprocessing_key(id: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in id.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// The cache-key material of an image: its resized canvas (`pixel_hash`, FNV-1a of the canvas pixels), the canvas size and the preprocessing that made it.
/// Two images that differ in any of these must never share cached state (the server keys the prefix cache with the same ingredients: `ImageSpan::key`).
pub fn image_key(pixel_hash: u64, canvas: (usize, usize), preprocessing_key: u64) -> u64 {
    gb_sched::image_content_hash(pixel_hash, canvas.0 as u32, canvas.1 as u32, preprocessing_key)
}

/// One encode handed to the worker thread: the image's still patches and grid, and where the features go.
struct Job {
    still: Vec<u16>,
    grid: (usize, usize),
    bytes: usize,
    reply: std::sync::mpsc::Sender<Result<DevBuf, ModelError>>,
}

/// The tower as the server's image encoder (`serve::ImageEncoder`): the tower lives on ONE thread of its own, with its stream and its cuBLAS handle
/// (a cuBLAS handle is per host thread and is never destroyed, so an encode on whichever thread of the server's blocking pool happens to be free would
/// create a handle, with its workspace, for every new pool thread: a slow leak of device memory for a server that runs for days). The caller takes a permit
/// for the features' bytes before any work is done, hands the patches over and waits; the features come back in a fresh f32 buffer.
pub struct GlmEncoder {
    jobs: Option<std::sync::mpsc::Sender<Job>>,
    worker: Option<std::thread::JoinHandle<()>>,
    budget: std::sync::Arc<FeatureBudget>,
    max_patches: usize,
    /// Hash of the preprocessing's `id()`: part of every image's cache key.
    preprocessing_key: u64,
}

impl GlmEncoder {
    pub fn new(tower: GlmVisionTower, budget: std::sync::Arc<FeatureBudget>, preprocessing_id: &str) -> Result<Self, ModelError> {
        let max_patches = tower.max_patches;
        let st = Stream::new()?;
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let worker = std::thread::Builder::new()
            .name("rexl3-glm-vision".into())
            .spawn(move || {
                let mut tower = tower;
                for job in rx {
                    let r = (|| -> Result<DevBuf, ModelError> {
                        let mut buf = DevBuf::alloc(job.bytes)?;
                        tower.encode(&job.still, job.grid, &mut buf, &st)?;
                        st.sync()?;
                        Ok(buf)
                    })();
                    let _ = job.reply.send(r);
                }
            })
            .map_err(|e| ModelError::Config(format!("cannot start the vision worker thread: {e}")))?;
        Ok(Self {
            jobs: Some(tx),
            worker: Some(worker),
            budget,
            max_patches,
            preprocessing_key: preprocessing_key(preprocessing_id),
        })
    }
}

impl Drop for GlmEncoder {
    fn drop(&mut self) {
        self.jobs.take(); // the worker's loop ends and the tower is freed on its own thread
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

impl crate::serve::ImageEncoder for GlmEncoder {
    fn encode(&mut self, pre: &gb_vision::Preprocessed) -> Result<crate::serve::EncodedImage, ModelError> {
        let (gh, gw) = (pre.grid.1, pre.grid.2);
        let p = pre.n_patches();
        if pre.grid.0 != 1 || p == 0 || p > self.max_patches || gh % MERGE != 0 || gw % MERGE != 0 || pre.patch_dim != 2 * PATCH_K
        {
            return Err(ModelError::Request(format!(
                "image of {p} patches ({gh}x{gw}, patch width {}); the encoder takes still images of up to {} patches",
                pre.patch_dim, self.max_patches
            )));
        }
        let tokens = p / PATCHES_PER_TOKEN;
        let bytes = tokens * OUT_HIDDEN * 4;
        let permit = self.budget.acquire(bytes as u64).ok_or_else(|| {
            ModelError::Busy(format!(
                "the memory for image features is in use ({:.0} MiB free of {:.0} MiB, this image needs {:.0} MiB); retry shortly",
                self.budget.free() as f64 / 1048576.0,
                self.budget.total() as f64 / 1048576.0,
                bytes as f64 / 1048576.0
            ))
        })?;
        // both temporal slots of a still image hold the same pixels: the tower folds the conv over them and reads one
        let still = gb_vision::glm::still_patches(pre, 2);
        let (reply, rx) = std::sync::mpsc::channel();
        let gone = || ModelError::Request("the vision worker thread has stopped".into());
        self.jobs.as_ref().ok_or_else(gone)?.send(Job { still, grid: (gh, gw), bytes, reply }).map_err(|_| gone())?;
        let buf = rx.recv().map_err(|_| gone())??;
        let key = image_key(pre.pixel_hash, pre.size, self.preprocessing_key);
        let merged = (gh / MERGE, gw / MERGE);
        Ok(crate::serve::EncodedImage {
            tokens,
            merged,
            features: std::sync::Arc::new(GlmImageFeatures { buf, tokens, merged, key, permit: Some(permit) }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn the_budget_hands_out_what_it_has_and_takes_it_back_on_drop() {
        let b = FeatureBudget::new(100);
        let p1 = b.acquire(60).unwrap();
        assert_eq!(b.free(), 40);
        assert!(b.acquire(41).is_none(), "more than is free is refused, whole");
        assert_eq!(b.free(), 40, "a refused acquire holds nothing");
        let p2 = b.acquire(40).unwrap();
        assert_eq!(b.free(), 0);
        assert!(b.acquire(1).is_none());
        drop(p1);
        assert_eq!(b.free(), 60);
        drop(p2);
        assert_eq!((b.free(), b.total()), (100, 100));
        assert!(b.acquire(0).is_some(), "an empty permit is allowed and free");
    }

    #[test]
    fn the_budget_is_never_oversubscribed_under_contention() {
        let b = FeatureBudget::new(1000);
        let held = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let max = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let threads: Vec<_> = (0..8u64)
            .map(|t| {
                let (b, held, max) = (b.clone(), held.clone(), max.clone());
                std::thread::spawn(move || {
                    for i in 0..2000u64 {
                        let want = 50 + (t * 37 + i * 11) % 200;
                        if let Some(p) = b.acquire(want) {
                            let now = held.fetch_add(want, std::sync::atomic::Ordering::SeqCst) + want;
                            max.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                            std::thread::yield_now();
                            held.fetch_sub(want, std::sync::atomic::Ordering::SeqCst);
                            drop(p);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert!(max.load(std::sync::atomic::Ordering::SeqCst) <= 1000, "the permits held at once exceeded the budget");
        assert_eq!(b.free(), 1000, "every permit came back");
    }

    #[test]
    fn an_images_key_changes_with_its_pixels_its_size_and_its_preprocessing() {
        let id = "glm5-next-vision-v2:fork:p14m2t2:tok16-8000:mean0,0,0:std0,0,0";
        let k = preprocessing_key(id);
        let base = image_key(0x1234, (644, 448), k);
        assert_eq!(base, image_key(0x1234, (644, 448), k), "a key is a function of its inputs");
        assert_ne!(base, image_key(0x1235, (644, 448), k), "different pixels (one bit)");
        assert_ne!(base, image_key(0x1234, (448, 644), k), "the same pixels in another shape");
        assert_ne!(base, image_key(0x1234, (644, 448), preprocessing_key(&id.replace("8000", "4096"))), "another token budget");
        assert_ne!(preprocessing_key(id), preprocessing_key(&id.replace("fork", "hf")), "another profile");
        assert_eq!(base, gb_sched::image_content_hash(0x1234, 644, 448, k), "it is the scheduler's image hash");
    }
}
