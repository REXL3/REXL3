//! Decode bombs: images that are small on the wire and large in memory. The decoders must refuse (or bound) them BEFORE they allocate what the
//! header, or the compressed stream, promises. The test owns a counting allocator (`#[global_allocator]`) and measures the peak heap of each
//! decode, so it can tell "refused at the header" from "decoded 800 MB and then refused", which the error value alone cannot.
//!
//! One `#[test]` runs every case in sequence: the allocator is process wide, and a second test thread would add its own allocations to the
//! peak. The bounds are the contract the server relies on (`gb_server::vision::prepare` decodes up to four images of a request on the
//! blocking pool, with no global limit on how many requests do so at once): a decoded image costs at most `BYTES_PER_PIXEL` bytes of peak heap
//! per pixel of the limit (`decode::MAX_DECODED_PIXELS`), and an image over the limit costs nothing but its header. PNG, JPEG and WebP.

use gb_vision::decode::{self, MAX_DECODED_PIXELS};
use gb_vision::rgb_decode::{decode_rgb, AlphaPolicy};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

struct Counting;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = CUR.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            let now = CUR.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(now, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        CUR.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() {
            if new >= l.size() {
                let now = CUR.fetch_add(new - l.size(), Relaxed) + new - l.size();
                PEAK.fetch_max(now, Relaxed);
            } else {
                CUR.fetch_sub(l.size() - new, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Peak heap bytes above the level before `f`, and the time it took.
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize, f64) {
    let base = CUR.load(Relaxed);
    PEAK.store(base, Relaxed);
    let t = Instant::now();
    let r = f();
    let secs = t.elapsed().as_secs_f64();
    (r, PEAK.load(Relaxed).saturating_sub(base), secs)
}

// ---- hand-built PNG and zlib, so no compressor is needed ----

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xedb8_8320 } else { c >> 1 };
        }
    }
    !c
}

fn chunk(out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let from = out.len();
    out.extend_from_slice(ty);
    out.extend_from_slice(data);
    let crc = crc32(&out[from..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    /// `count` bits of `v`, least significant first (DEFLATE's order for everything but Huffman codes).
    fn put(&mut self, v: u32, count: u32) {
        self.acc |= (v as u64) << self.n;
        self.n += count;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    /// A Huffman code: most significant bit first.
    fn code(&mut self, v: u32, len: u32) {
        for i in (0..len).rev() {
            self.put((v >> i) & 1, 1);
        }
    }
}

/// A zlib stream that inflates to `n` zero bytes (`n >= 1`), a few bits per 258 bytes: the classic bomb, 1 KB of input per 160 KB of output.
fn zlib_zeros(n: usize) -> Vec<u8> {
    let mut b = Bits { out: vec![0x78, 0x01], acc: 0, n: 0 };
    b.put(1, 1); // final block
    b.put(1, 2); // fixed Huffman codes
    b.code(0x30, 8); // the literal 0x00
    let mut left = n - 1;
    while left >= 258 {
        b.code(0xc5, 8); // length 258 (symbol 285)
        b.code(0, 5); // distance 1
        left -= 258;
    }
    for _ in 0..left {
        b.code(0x30, 8);
    }
    b.code(0, 7); // end of block
    if b.n > 0 {
        b.put(0, 8 - b.n);
    }
    let mut out = b.out;
    let adler = (((n % 65521) as u32) << 16) | 1;
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

/// A PNG of `w x h` 8-bit gray whose IDAT inflates to `idat_bytes` zeros (the right number is `h * (w + 1)`), with `extra` chunks before IDAT.
fn png(w: u32, h: u32, idat_bytes: usize, extra: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
    let mut o = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 0, 0, 0, 0]);
    chunk(&mut o, b"IHDR", &ihdr);
    for (ty, data) in extra {
        chunk(&mut o, ty, data);
    }
    chunk(&mut o, b"IDAT", &zlib_zeros(idat_bytes));
    chunk(&mut o, b"IEND", &[]);
    o
}

/// One JPEG fixture with its start-of-frame size replaced.
fn jpeg_with_size(w: u16, h: u16) -> Vec<u8> {
    let mut b =
        std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision/jpeg_q90_64x48.jpg")).unwrap();
    let at = b.windows(2).position(|p| p == [0xff, 0xc0] || p == [0xff, 0xc2]).expect("a start-of-frame marker");
    b[at + 5..at + 7].copy_from_slice(&h.to_be_bytes());
    b[at + 7..at + 9].copy_from_slice(&w.to_be_bytes());
    b
}

const MIB: usize = 1 << 20;

/// Peak heap per pixel of the limit that a decode that succeeds may use: the decoder's raw rows (1 to 4 bytes a pixel) and the RGB output (3).
const BYTES_PER_PIXEL: usize = 8;

fn show(name: &str, peak: usize, secs: f64, what: &str) {
    println!("{name:<44} peak {:>8.1} MiB  {secs:>6.2} s  {what}", peak as f64 / MIB as f64);
}

// ---- WebP: committed small files with their headers rewritten, hand-built RIFF wrappers, and image-webp's own (lossless) encoder ----

fn webp_fixture(name: &str) -> Vec<u8> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision/webp").join(name)).unwrap()
}

/// A RIFF/WEBP file of these chunks (padded to even sizes, the RIFF size honest).
fn riff(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let mut body = b"WEBP".to_vec();
    for (c, p) in chunks {
        body.extend_from_slice(*c);
        body.extend_from_slice(&(p.len() as u32).to_le_bytes());
        body.extend_from_slice(p);
        if p.len() % 2 == 1 {
            body.push(0);
        }
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// A simple lossy fixture's VP8 payload with its frame size replaced (the 14-bit width and height after the start code).
fn vp8_payload_with_size(w: u16, h: u16) -> Vec<u8> {
    let b = webp_fixture("flat_lossy_2048.webp");
    assert_eq!(&b[12..16], b"VP8 ");
    let mut p = b[20..].to_vec();
    p[6..8].copy_from_slice(&w.to_le_bytes());
    p[8..10].copy_from_slice(&h.to_le_bytes());
    p
}

/// A VP8X header chunk: flags, canvas size (24-bit minus one).
fn vp8x(flags: u8, w: u32, h: u32) -> Vec<u8> {
    let mut p = vec![flags, 0, 0, 0];
    p.extend_from_slice(&(w - 1).to_le_bytes()[..3]);
    p.extend_from_slice(&(h - 1).to_le_bytes()[..3]);
    p
}

/// A lossless WebP of a flat `w x h` image made by image-webp's encoder (RGB, or RGBA with half alpha).
fn lossless(w: u32, h: u32, alpha: bool) -> Vec<u8> {
    let (bpp, ct) = if alpha { (4, image_webp::ColorType::Rgba8) } else { (3, image_webp::ColorType::Rgb8) };
    let px: Vec<u8> = [90u8, 120, 200, 128].iter().copied().take(bpp).cycle().take(w as usize * h as usize * bpp).collect();
    let mut out = Vec::new();
    image_webp::WebPEncoder::new(&mut out).encode(&px, w, h, ct).unwrap();
    out
}

#[test]
fn decode_bombs_are_refused_before_they_are_allocated_and_the_rest_is_bounded() {
    // 1. a PNG whose header promises 4 billion pixels: refused at the header
    for (w, h) in [(65536u32, 65536u32), (0x7fff_ffff, 0x7fff_ffff), (MAX_DECODED_PIXELS as u32 / 8192 + 1, 8192)] {
        let b = png(w, h, 1, &[]);
        let (r, peak, secs) = peak_of(|| decode::decode(&b));
        show(&format!("png header {w}x{h}"), peak, secs, "refused");
        // either our own size check (Unsupported) or the png crate's (a Decode error naming its limits): both are a refusal at the header
        assert!(r.is_err(), "{w}x{h}: {:?}", r.map(|i| (i.width, i.height)));
        assert!(peak < 4 * MIB, "{w}x{h}: {} MiB were allocated for a header that is over the limit", peak / MIB);
        let (r, peak, _) = peak_of(|| decode_rgb(&b, AlphaPolicy::CompositeWhite));
        assert!(r.is_err() && peak < 4 * MIB, "decode_rgb {w}x{h}: {:?}, {} MiB", r.map(|i| (i.width, i.height)), peak / MIB);
    }

    // 2. exactly the limit: decodes, inside the stated bound (8,192 x 8,192, a 70 KB file)
    let side = (MAX_DECODED_PIXELS as f64).sqrt() as usize;
    assert_eq!(side * side, MAX_DECODED_PIXELS);
    let at_limit = png(side as u32, side as u32, side * (side + 1), &[]);
    let (r, peak, secs) = peak_of(|| decode::decode(&at_limit));
    show("png at the pixel limit (8192 x 8192 gray)", peak, secs, "decoded");
    assert_eq!(r.map(|i| (i.width, i.height)), Ok((side, side)));
    assert!(peak <= BYTES_PER_PIXEL * MAX_DECODED_PIXELS, "peak {} MiB over {} bytes a pixel", peak / MIB, BYTES_PER_PIXEL);
    assert!(secs < 30.0, "{secs} s");

    // 3. an IDAT that inflates to far more than the image holds (a 64 x 64 image, 512 MiB of zeros): bounded by the image, not by the stream
    let bomb = png(64, 64, 512 * MIB, &[]);
    assert!(bomb.len() < 4 * MIB, "the bomb is {} bytes on the wire", bomb.len());
    let (r, peak, secs) = peak_of(|| decode::decode(&bomb));
    show("png IDAT inflating to 512 MiB (64x64)", peak, secs, if r.is_ok() { "decoded" } else { "refused" });
    assert!(peak < 8 * MIB, "{} MiB for a 64 x 64 image", peak / MIB);

    // 4. ancillary chunks that inflate: iCCP and zTXt (a 512 MiB profile and a 512 MiB text)
    let mut iccp = b"p\0\0".to_vec();
    iccp.extend_from_slice(&zlib_zeros(512 * MIB));
    let mut ztxt = b"k\0\0".to_vec();
    ztxt.extend_from_slice(&zlib_zeros(512 * MIB));
    for (name, extra) in [("iCCP", ([*b"iCCP"], iccp)), ("zTXt", ([*b"zTXt"], ztxt))] {
        let b = png(8, 8, 8 * 9, &[(extra.0[0], extra.1)]);
        let (r, peak, secs) = peak_of(|| decode::decode(&b));
        show(&format!("png {name} inflating to 512 MiB"), peak, secs, if r.is_ok() { "decoded" } else { "refused" });
        assert!(peak < 128 * MIB, "{name}: {} MiB for an 8 x 8 image with a 512 MiB chunk", peak / MIB);
        let (_, peak, _) = peak_of(|| decode_rgb(&b, AlphaPolicy::Drop));
        assert!(peak < 128 * MIB, "{name} (decode_rgb): {} MiB", peak / MIB);
    }

    // 5. JPEG: the start-of-frame size decides the allocation, so over the limit must be refused from the header
    for (w, h) in [(16384u16, 16384u16), (16384, 8193), (16383, 16383), (65535, 65535)] {
        let b = jpeg_with_size(w, h);
        let (r, peak, secs) = peak_of(|| decode::decode(&b));
        show(&format!("jpeg header {w}x{h}"), peak, secs, if r.is_ok() { "decoded" } else { "refused" });
        assert!(r.is_err(), "{w}x{h} is over the limit: {:?}", r.map(|i| (i.width, i.height)));
        assert!(peak < 4 * MIB, "{w}x{h}: {} MiB were allocated before the size limit was applied", peak / MIB);
    }
    // 6. a JPEG header at the limit (a truncated file: it may decode as flat gray or fail) stays inside the bound
    let b = jpeg_with_size(8192, 8192);
    let (r, peak, secs) = peak_of(|| decode::decode(&b));
    show("jpeg header 8192x8192 (truncated data)", peak, secs, if r.is_ok() { "decoded" } else { "refused" });
    assert!(peak <= 2 * BYTES_PER_PIXEL * MAX_DECODED_PIXELS, "{} MiB", peak / MIB);

    // 7. WebP: the size comes from the bitstream header and is refused there. A simple lossy file of 16,383 x 16,383, the same frame
    // inside a VP8X declaring a 1 x 1 canvas (image-webp sizes its planes from the frame header and compares with the canvas only after: 400
    // MB), canvases of 8,193 x 8,192 and 2^24 x 2^24, a lossless header of 16,384 x 16,384, and chunks whose sizes run past the file
    let big = vp8_payload_with_size(16383, 16383);
    let mut ll = lossless(8, 8, false);
    let v = (16383u32) | (16383 << 14);
    ll[21..25].copy_from_slice(&v.to_le_bytes());
    let mut lying_riff = webp_fixture("lossy_q75_97x61.webp");
    lying_riff[4..8].copy_from_slice(&0xffff_fff0u32.to_le_bytes());
    let mut lying_chunk = riff(&[(b"VP8X", &vp8x(0x08, 64, 64)), (b"EXIF", &[0; 16]), (b"VP8 ", &vp8_payload_with_size(64, 64))]);
    lying_chunk[34..38].copy_from_slice(&0x7fff_fff0u32.to_le_bytes());
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("webp VP8 16383x16383", riff(&[(b"VP8 ", &big)])),
        ("webp VP8X 1x1 around VP8 16383x16383", riff(&[(b"VP8X", &vp8x(0, 1, 1)), (b"VP8 ", &big)])),
        ("webp VP8X 8193x8192", riff(&[(b"VP8X", &vp8x(0, 8193, 8192)), (b"VP8 ", &vp8_payload_with_size(8193, 8192))])),
        ("webp VP8X 2^24 x 2^24", riff(&[(b"VP8X", &vp8x(0, 1 << 24, 1 << 24)), (b"VP8 ", &big)])),
        ("webp VP8L 16384x16384", ll),
        ("webp RIFF size 4 GiB", lying_riff),
        ("webp EXIF chunk of 2 GiB", lying_chunk),
    ];
    for (name, b) in &cases {
        for (dec, r, peak, secs) in
            [("decode", peak_of(|| decode::decode(b))), ("decode_rgb", peak_of(|| decode_rgb(b, AlphaPolicy::Drop)))]
                .map(|(n, (r, p, s))| (n, r, p, s))
        {
            show(&format!("{name} ({dec})"), peak, secs, "refused");
            assert!(r.is_err(), "{name}: {:?}", r.map(|i| (i.width, i.height)));
            assert!(peak < MIB, "{name} ({dec}): {} KiB before the refusal", peak >> 10);
        }
    }

    // 8. WebP decodes that succeed stay inside the per-pixel bound: lossy (Y, U, V planes and the output) with and without alpha at
    // 2,048 x 2,048 (committed files), lossless at the pixel limit with and without alpha (image-webp's encoder, outside the measurement)
    for name in ["flat_lossy_2048.webp", "flat_lossy_alpha_2048.webp"] {
        let b = webp_fixture(name);
        let (r, peak, secs) = peak_of(|| decode_rgb(&b, AlphaPolicy::Drop));
        show(&format!("webp {name}"), peak, secs, "decoded");
        assert_eq!(r.map(|i| (i.width, i.height)), Ok((2048, 2048)));
        assert!(peak <= BYTES_PER_PIXEL * 2048 * 2048, "{name}: {:.2} bytes a pixel", peak as f64 / (2048.0 * 2048.0));
    }
    for alpha in [false, true] {
        let b = lossless(side as u32, side as u32, alpha);
        let (r, peak, secs) = peak_of(|| decode::decode(&b));
        show(&format!("webp lossless at the pixel limit (alpha {alpha})"), peak, secs, "decoded");
        assert_eq!(r.map(|i| (i.width, i.height)), Ok((side, side)));
        assert!(
            peak <= BYTES_PER_PIXEL * MAX_DECODED_PIXELS,
            "alpha {alpha}: {:.2} bytes a pixel",
            peak as f64 / MAX_DECODED_PIXELS as f64
        );
        assert!(secs < 30.0, "{secs} s");
    }
    // 9. a lossy frame header at the limit over truncated data (a 7 KB file claiming 8,192 x 8,192): bounded like a decode, in bounded time
    let b = riff(&[(b"VP8 ", &vp8_payload_with_size(8192, 8192))]);
    let (r, peak, secs) = peak_of(|| decode::decode(&b));
    show("webp VP8 8192x8192 (truncated data)", peak, secs, if r.is_ok() { "decoded" } else { "refused" });
    assert!(peak <= BYTES_PER_PIXEL * MAX_DECODED_PIXELS, "{} MiB", peak / MIB);
    assert!(secs < 30.0, "{secs} s");
}
