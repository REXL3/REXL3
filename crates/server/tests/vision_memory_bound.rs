//! Complete decoder + metadata + resize + normalization + patch tensor heap measurement, CPU only.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let n = LIVE.fetch_add(l.size(), Ordering::SeqCst) + l.size();
            PEAK.fetch_max(n, Ordering::SeqCst);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            let n = LIVE.fetch_add(l.size(), Ordering::SeqCst) + l.size();
            PEAK.fetch_max(n, Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        LIVE.fetch_sub(l.size(), Ordering::SeqCst);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = System.realloc(p, l, n);
        if !q.is_null() {
            if n >= l.size() {
                let v = LIVE.fetch_add(n - l.size(), Ordering::SeqCst) + n - l.size();
                PEAK.fetch_max(v, Ordering::SeqCst);
            } else {
                LIVE.fetch_sub(l.size() - n, Ordering::SeqCst);
            }
        }
        q
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;
#[test]
fn maximum_decoded_rgba_pipeline_fits_weighted_reserve_for_both_families() {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 8192, 8192);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&vec![127; 8192 * 8192 * 4]).unwrap();
    }
    assert!(bytes.len() <= 10 << 20);
    let image = gb_server::vision::ImageRef { message: 0, part: 0, url: "data:test".into() };
    let mut qwen = gb_vision::Preprocessing::qwen38();
    qwen.max_pixels = 4096 * 32 * 32;
    let glm = gb_vision::glm::GlmPreprocessing::default();
    for (tokens, pp) in [(4096, &qwen as &dyn gb_vision::preprocessor::ImagePreprocessor), (8000, &glm)] {
        let baseline = LIVE.load(Ordering::SeqCst);
        PEAK.store(baseline, Ordering::SeqCst);
        let pre = gb_server::vision::prepare_bytes(
            &image,
            &bytes,
            &gb_server::vision::Limits { max_tokens_per_image: tokens, ..Default::default() },
            pp,
            None,
        )
        .unwrap();
        let peak = PEAK.load(Ordering::SeqCst).saturating_sub(baseline);
        let charged = gb_server::preparation::image_work_bytes(tokens).unwrap();
        eprintln!("family={} decoded_pixels={} encoded_bytes={} measured_pipeline_peak_bytes={} weighted_reserve_bytes={} output_patch_bytes={}",pp.family(),8192*8192,bytes.len(),peak,charged,pre.patches.len()*2);
        assert!(peak <= charged, "complete preprocessing peak {peak} exceeds byte lease {charged}");
        drop(pre);
    }
}
