//! The image decoders against hostile input: a decoder that panics, hangs or allocates without bound is a way to take a server down with one
//! request (an image arrives as a base64 data URI in a chat message, from whoever can reach the port).
//!
//! No fuzzing engine is needed or available offline, so the cases are generated here, deterministically, from the committed fixtures: every
//! prefix of the small images, a sweep of single-byte corruptions, and seeded random mutations that are format aware (PNG chunk lengths, IHDR
//! fields and CRCs repaired so the damage reaches the decoder's own logic; JPEG start-of-frame sizes, segment lengths and EXIF blocks; WebP
//! RIFF and chunk sizes, fourcc swaps, VP8 / VP8L / VP8X size fields and flags, inserted ANIM, ANMF, ICCP, EXIF and ALPH chunks). The
//! contract each case checks, for all three decoders the server can reach (`decode::decode`, `rgb_decode::decode_rgb` under both alpha
//! policies): the call returns, in bounded time, with an image or an error, and never panics.
//!
//! `REXL3_FUZZ_CASES=N` sets the random mutations per seed file (default 400; a few minutes of CPU at 100,000). A failure prints the seed file,
//! the case number and the first bytes of the input, which is enough to rebuild it (the generator is deterministic).
//! Memory bounds (limits enforced before the allocation) are in `alloc_bounds.rs`, which owns a counting allocator.

use gb_vision::decode;
use gb_vision::glm::GlmPreprocessing;
use gb_vision::preprocessor::ImagePreprocessor;
use gb_vision::rgb_decode::{decode_rgb, AlphaPolicy};
use gb_vision::{Preprocessing, VisionError};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision")
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// SplitMix64: a seeded generator, so a failing case can be rebuilt.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
    fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[self.below(v.len())]
    }
}

/// Values that are edges of a length or a size field.
const EDGES32: [u32; 12] = [0, 1, 2, 3, 255, 256, 65535, 65536, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 16384];

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

/// The chunks of a PNG as (offset of the length field, data length), as far as the file is well formed.
fn png_chunks(b: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = 8;
    while at + 12 <= b.len() {
        let len = u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize;
        if at + 12 + len > b.len() {
            break;
        }
        out.push((at, len));
        at += 12 + len;
    }
    out
}

/// Recompute every chunk's CRC, so a change inside a chunk is not stopped at the checksum.
fn fix_crcs(b: &mut [u8]) {
    for (at, len) in png_chunks(b) {
        let crc = crc32(&b[at + 4..at + 8 + len]);
        b[at + 8 + len..at + 12 + len].copy_from_slice(&crc.to_be_bytes());
    }
}

fn mutate_png(seed: &[u8], r: &mut Rng) -> Vec<u8> {
    let mut b = seed.to_vec();
    let chunks = png_chunks(&b);
    match r.below(8) {
        0 => b.truncate(r.below(b.len() + 1)),
        1 => {
            for _ in 0..1 + r.below(8) {
                let i = r.below(b.len());
                b[i] ^= 1 << r.below(8);
            }
        }
        2 if !chunks.is_empty() => {
            // bytes inside one chunk's data, checksums repaired
            let (at, len) = chunks[r.below(chunks.len())];
            for _ in 0..1 + r.below(4) {
                if len > 0 {
                    b[at + 8 + r.below(len)] = r.next() as u8;
                }
            }
            fix_crcs(&mut b);
        }
        3 if !chunks.is_empty() => {
            // IHDR: width, height, bit depth, colour type, compression, filter, interlace
            let (at, len) = chunks[0];
            if len >= 13 {
                match r.below(5) {
                    0 => b[at + 8..at + 12].copy_from_slice(&r.pick(&EDGES32).to_be_bytes()),
                    1 => b[at + 12..at + 16].copy_from_slice(&r.pick(&EDGES32).to_be_bytes()),
                    2 => b[at + 16] = r.pick(&[0u8, 1, 2, 4, 8, 16, 3, 7, 255]),
                    3 => b[at + 17] = r.pick(&[0u8, 1, 2, 3, 4, 5, 6, 7, 255]),
                    _ => b[at + 20] = r.pick(&[0u8, 1, 2, 255]),
                }
                fix_crcs(&mut b);
            }
        }
        4 if !chunks.is_empty() => {
            // a length field at an edge (no checksum repair: the length is what is wrong)
            let (at, len) = chunks[r.below(chunks.len())];
            let v = r.pick(&[0u32, 1, len as u32 + 1, (len as u32).wrapping_sub(1), 0x7fff_ffff, 0xffff_ffff]);
            b[at..at + 4].copy_from_slice(&v.to_be_bytes());
        }
        5 if chunks.len() > 2 => {
            // drop a chunk (IHDR and IEND included) or repeat one
            let (at, len) = chunks[r.below(chunks.len())];
            if r.below(2) == 0 {
                b.drain(at..at + 12 + len);
            } else {
                let copy = b[at..at + 12 + len].to_vec();
                let to = chunks[r.below(chunks.len())].0;
                b.splice(to..to, copy);
            }
        }
        6 if !chunks.is_empty() => {
            // a well-formed chunk of a random type and content, before IDAT or before IEND
            let n = r.below(64);
            let mut chunk = (n as u32).to_be_bytes().to_vec();
            let ty = r.pick(&[
                *b"tRNS", *b"PLTE", *b"gAMA", *b"iCCP", *b"zTXt", *b"tEXt", *b"iTXt", *b"acTL", *b"fcTL", *b"fdAT", *b"sBIT",
                *b"bKGD", *b"pHYs", *b"eXIf", *b"xxxx", *b"XXXX",
            ]);
            chunk.extend_from_slice(&ty);
            chunk.extend((0..n).map(|_| r.next() as u8));
            let crc = crc32(&chunk[4..]);
            chunk.extend_from_slice(&crc.to_be_bytes());
            let to = chunks[r.below(chunks.len())].0;
            b.splice(to..to, chunk);
        }
        _ => {
            let i = r.below(b.len());
            let n = (1 + r.below(16)).min(b.len() - i);
            for x in &mut b[i..i + n] {
                *x = r.pick(&[0u8, 255, 0x80, 0x7f]);
            }
        }
    }
    b
}

/// The marker segments of a JPEG up to the start of scan, as (offset of the 0xFF, marker, length field).
fn jpeg_segments(b: &[u8]) -> Vec<(usize, u8, usize)> {
    let mut out = Vec::new();
    let mut i = 2;
    while i + 4 <= b.len() && b[i] == 0xff {
        let m = b[i + 1];
        let len = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        out.push((i, m, len));
        if m == 0xda || len < 2 {
            break;
        }
        i += 2 + len;
    }
    out
}

/// An APP1 segment that starts like EXIF and carries a random TIFF directory: the code that reads the orientation is ours, not the decoder's.
fn exif_segment(r: &mut Rng) -> Vec<u8> {
    let le = r.below(2) == 0;
    let u16b = |v: u16| if le { v.to_le_bytes() } else { v.to_be_bytes() };
    let u32b = |v: u32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
    let mut tiff = if le { b"II".to_vec() } else { b"MM".to_vec() };
    tiff.extend_from_slice(&u16b(42));
    tiff.extend_from_slice(&u32b(r.pick(&[8u32, 8, 8, 0, 4, 10, 100, 0xffff_fff0, 0xffff_ffff])));
    let entries = r.pick(&[0u16, 1, 1, 2, 5, 100, 65535]);
    tiff.extend_from_slice(&u16b(entries));
    for _ in 0..r.below(6) {
        tiff.extend_from_slice(&u16b(r.pick(&[0x0112u16, 0x0112, 0x0100, 0x8769])));
        tiff.extend_from_slice(&u16b(3));
        tiff.extend_from_slice(&u32b(1));
        tiff.extend_from_slice(&u16b(r.pick(&[1u16, 2, 3, 4, 5, 6, 7, 8, 0, 9, 255, 65535])));
        tiff.extend_from_slice(&[0, 0]);
    }
    tiff.truncate(r.below(tiff.len() + 1).max(r.below(tiff.len() + 1)));
    let mut seg = vec![0xff, 0xe1];
    let body_len = 6 + tiff.len();
    // the length field is honest, too small or too large
    let len = r.pick(&[body_len as u32 + 2, body_len as u32 + 2, 0, 1, 2, 3, 8, 14, 15, 65535]) as u16;
    seg.extend_from_slice(&len.to_be_bytes());
    seg.extend_from_slice(b"Exif\0\0");
    seg.extend_from_slice(&tiff);
    seg
}

fn mutate_jpeg(seed: &[u8], r: &mut Rng) -> Vec<u8> {
    let mut b = seed.to_vec();
    let segs = jpeg_segments(&b);
    match r.below(7) {
        0 => b.truncate(r.below(b.len() + 1)),
        1 => {
            for _ in 0..1 + r.below(8) {
                let i = r.below(b.len());
                b[i] ^= 1 << r.below(8);
            }
        }
        2 => {
            // the start-of-frame size (the allocation is decided by it)
            if let Some(&(at, _, _)) = segs.iter().find(|s| matches!(s.1, 0xc0..=0xc2)) {
                let h = r.pick(&[0u16, 1, 2, 8, 16, 255, 4096, 8192, 16383, 16384, 16385, 32767, 65535]);
                let w = r.pick(&[0u16, 1, 2, 8, 16, 255, 4096, 8192, 16383, 16384, 16385, 32767, 65535]);
                if at + 9 <= b.len() {
                    b[at + 5..at + 7].copy_from_slice(&h.to_be_bytes());
                    b[at + 7..at + 9].copy_from_slice(&w.to_be_bytes());
                }
            }
        }
        3 if !segs.is_empty() => {
            let (at, _, _) = segs[r.below(segs.len())];
            let v = r.pick(&[0u16, 1, 2, 3, 4, 17, 255, 65535]);
            b[at + 2..at + 4].copy_from_slice(&v.to_be_bytes());
        }
        4 | 5 => {
            // an EXIF block right after SOI (the orientation reader runs after a successful decode)
            let seg = exif_segment(r);
            b.splice(2..2, seg);
        }
        _ => {
            let i = r.below(b.len());
            let n = (1 + r.below(16)).min(b.len() - i);
            for x in &mut b[i..i + n] {
                *x = r.pick(&[0u8, 255, 0xd9, 0xda, 0xc0]);
            }
        }
    }
    b
}

/// The chunks of a WebP after the 12-byte RIFF header, as (offset of the fourcc, payload length), as far as the file is well formed.
fn webp_chunks(b: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = 12;
    while at + 8 <= b.len() {
        let len = u32::from_le_bytes([b[at + 4], b[at + 5], b[at + 6], b[at + 7]]) as usize;
        if at + 8 + len > b.len() {
            break;
        }
        out.push((at, len));
        at += 8 + len + (len & 1);
    }
    out
}

/// Make the RIFF size honest again, so a change inside the file reaches the chunk walker and the decoder.
fn fix_riff(b: &mut [u8]) {
    if b.len() >= 8 {
        let n = (b.len() - 8) as u32;
        b[4..8].copy_from_slice(&n.to_le_bytes());
    }
}

const FOURCCS: [[u8; 4]; 12] =
    [*b"VP8 ", *b"VP8L", *b"VP8X", *b"ALPH", *b"ANIM", *b"ANMF", *b"ICCP", *b"EXIF", *b"XMP ", *b"xxxx", *b"WEBP", *b"RIFF"];

fn mutate_webp(seed: &[u8], r: &mut Rng) -> Vec<u8> {
    let mut b = seed.to_vec();
    let chunks = webp_chunks(&b);
    match r.below(10) {
        0 => b.truncate(r.below(b.len() + 1)),
        1 => {
            for _ in 0..1 + r.below(8) {
                let i = r.below(b.len());
                b[i] ^= 1 << r.below(8);
            }
        }
        2 => {
            // the RIFF size at an edge, or the form type replaced (WAVE, AVI, a near miss)
            if r.below(2) == 0 {
                let v = r.pick(&EDGES32).wrapping_add(r.pick(&[0u32, 1, 4, 12]));
                b[4..8].copy_from_slice(&v.to_le_bytes());
            } else {
                b[8..12].copy_from_slice(&r.pick(&[*b"WAVE", *b"AVI ", *b"WEBp", *b"WEB ", *b"VP8 "]));
            }
        }
        3 if !chunks.is_empty() => {
            // a chunk length at an edge (the RIFF size left alone, or repaired)
            let (at, len) = chunks[r.below(chunks.len())];
            let v = r.pick(&[0u32, 1, 2, 9, len as u32 + 1, (len as u32).wrapping_sub(1), 0x7fff_ffff, 0xffff_fff0, 0xffff_ffff]);
            b[at + 4..at + 8].copy_from_slice(&v.to_le_bytes());
            if r.below(2) == 0 {
                fix_riff(&mut b);
            }
        }
        4 if !chunks.is_empty() => {
            // a fourcc swapped for another
            let (at, _) = chunks[r.below(chunks.len())];
            b[at..at + 4].copy_from_slice(&r.pick(&FOURCCS));
        }
        5 if !chunks.is_empty() => {
            // the image sizes and flags: a VP8 frame's 14-bit sizes (and scale bits), a VP8L header, a VP8X canvas and flags
            let (at, len) = chunks[r.below(chunks.len())];
            let p = at + 8;
            let edges16 = [0u16, 1, 2, 8, 255, 4096, 8192, 8193, 16383, 0x3fff | 0xc000, 0xffff];
            let fourcc = [b[at], b[at + 1], b[at + 2], b[at + 3]];
            match &fourcc {
                b"VP8 " if len >= 10 => {
                    let o = p + 6 + 2 * r.below(2);
                    b[o..o + 2].copy_from_slice(&r.pick(&edges16).to_le_bytes());
                }
                b"VP8L" if len >= 5 => {
                    let v = (r.below(1 << 14) as u32) | ((r.pick(&edges16) as u32 & 0x3fff) << 14) | ((r.below(16) as u32) << 28);
                    b[p + 1..p + 5].copy_from_slice(&v.to_le_bytes());
                }
                b"VP8X" if len >= 10 => {
                    if r.below(2) == 0 {
                        b[p] = r.pick(&[0u8, 0x02, 0x04, 0x08, 0x10, 0x20, 0x3e, 0xff]);
                    } else {
                        let o = p + 4 + 3 * r.below(2);
                        let v = r.pick(&[0u32, 1, 8191, 8192, 16382, 16383, 0xff_ffff]);
                        b[o..o + 3].copy_from_slice(&v.to_le_bytes()[..3]);
                    }
                }
                b"ALPH" if len >= 1 => b[p] = r.next() as u8,
                _ => {
                    if len > 0 {
                        b[p + r.below(len)] = r.next() as u8;
                    }
                }
            }
        }
        6 | 7 if b.len() >= 12 => {
            // a well-formed chunk inserted at a chunk boundary (or the end), the RIFF size repaired: animation chunks, metadata
            // (an EXIF block with a random TIFF directory), an ALPH plane, junk
            let ty = r.pick(&FOURCCS);
            let payload: Vec<u8> = if &ty == b"EXIF" && r.below(2) == 0 {
                exif_segment(r)[4..].to_vec()
            } else {
                (0..r.pick(&[0usize, 1, 6, 10, 16, 24, 64])).map(|_| r.next() as u8).collect()
            };
            let mut c = ty.to_vec();
            c.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            c.extend_from_slice(&payload);
            if payload.len() % 2 == 1 {
                c.push(0);
            }
            let to = if chunks.is_empty() || r.below(4) == 0 { b.len() } else { chunks[r.below(chunks.len())].0 };
            b.splice(to..to, c);
            fix_riff(&mut b);
        }
        8 if chunks.len() > 1 => {
            // drop a chunk, or repeat one
            let (at, len) = chunks[r.below(chunks.len())];
            let end = (at + 8 + len + (len & 1)).min(b.len());
            if r.below(2) == 0 {
                b.drain(at..end);
            } else {
                let copy = b[at..end].to_vec();
                b.splice(at..at, copy);
            }
            fix_riff(&mut b);
        }
        _ => {
            // a run of bytes inside the compressed data
            let i = 12 + r.below(b.len().saturating_sub(12));
            let n = (1 + r.below(16)).min(b.len() - i);
            for x in &mut b[i..i + n] {
                *x = r.pick(&[0u8, 255, 0x80, 0x7f, 0x2f, 0x9d]);
            }
        }
    }
    b
}

/// All the decoders the server can reach, on `bytes`. Fails the test (naming `what`) on a panic or on a call that takes longer than `budget`.
fn run(bytes: &[u8], what: &str, budget: Duration) -> Vec<Result<(usize, usize), VisionError>> {
    let q = Preprocessing::qwen38();
    let g = GlmPreprocessing::default();
    let mut out = Vec::new();
    type Dec = Box<dyn Fn(&[u8]) -> Result<gb_vision::Rgb, VisionError>>;
    let decoders: [(&str, Dec); 3] = [
        ("decode::decode", Box::new(decode::decode)),
        ("decode_rgb(Drop)", Box::new(|b| decode_rgb(b, AlphaPolicy::Drop))),
        ("decode_rgb(CompositeWhite)", Box::new(|b| decode_rgb(b, AlphaPolicy::CompositeWhite))),
    ];
    for (name, f) in decoders {
        let t = Instant::now();
        let r = catch_unwind(AssertUnwindSafe(|| f(bytes)));
        let took = t.elapsed();
        let head: String = bytes.iter().take(64).map(|b| format!("{b:02x}")).collect();
        let r = r.unwrap_or_else(|_| panic!("{name} panicked on {what} ({} bytes, starts {head})", bytes.len()));
        assert!(took < budget, "{name} took {took:?} on {what} ({} bytes, starts {head})", bytes.len());
        match r {
            Ok(img) => {
                assert_eq!(img.data.len(), img.width * img.height * 3, "{name}: the buffer does not match the size, on {what}");
                // what the server does next: plan from the size, and preprocess when the image is small enough to be cheap
                let _ = catch_unwind(AssertUnwindSafe(|| q.plan(img.width, img.height)))
                    .unwrap_or_else(|_| panic!("Qwen plan panicked on {what} ({}x{})", img.width, img.height));
                let _ = catch_unwind(AssertUnwindSafe(|| g.plan(img.width, img.height)))
                    .unwrap_or_else(|_| panic!("GLM plan panicked on {what} ({}x{})", img.width, img.height));
                if img.width * img.height <= 1 << 18 {
                    catch_unwind(AssertUnwindSafe(|| {
                        let _ = q.preprocess(&img);
                        let _ = g.preprocess(&img);
                    }))
                    .unwrap_or_else(|_| panic!("preprocess panicked on {what} ({}x{})", img.width, img.height));
                }
                out.push(Ok((img.width, img.height)));
            }
            Err(e) => out.push(Err(e)),
        }
    }
    out
}

fn cases() -> usize {
    std::env::var("REXL3_FUZZ_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(400)
}

const PNG_SEEDS: [&str; 7] = [
    "grad_64x64.png",
    "check_512x512.png",
    "gray_220x140.png",
    "la_120x120.png",
    "palette_180x180.png",
    "rgba_200x150.png",
    "noise_96x96_down.png",
];
const JPEG_SEEDS: [&str; 4] = ["jpeg_q90_64x48.jpg", "jpeg_q50_75x61_sub420.jpg", "jpeg_q50_sub420.jpg", "jpeg_q90_320x240.jpg"];
/// Simple lossy, lossy with an ALPH plane, simple lossless (with and without alpha, a palette), extended with EXIF, the smallest file.
const WEBP_SEEDS: [&str; 8] = [
    "webp/lossy_q75_97x61.webp",
    "webp/lossy_q0_m0_17x13.webp",
    "webp/lossy_alpha_q80_63x65.webp",
    "webp/lossless_63x65.webp",
    "webp/lossless_alpha_exact_63x65.webp",
    "webp/lossless_pal4_97x61.webp",
    "webp/lossless_exif6_37x21.webp",
    "webp/lossy_exif8_37x21.webp",
];

#[test]
fn the_seeds_decode() {
    // the mutations below start from images that are good
    for n in PNG_SEEDS.iter().chain(JPEG_SEEDS.iter()).chain(WEBP_SEEDS.iter()) {
        let r = run(&read(n), n, Duration::from_secs(20));
        assert!(r.iter().all(|x| x.is_ok()), "{n}: {r:?}");
    }
}

#[test]
fn every_prefix_of_the_small_images_returns_an_error_or_an_image() {
    for n in [
        "grad_64x64.png",
        "palette_180x180.png",
        "jpeg_q90_64x48.jpg",
        "jpeg_q50_75x61_sub420.jpg",
        "webp/lossy_q0_m0_17x13.webp",
        "webp/lossless_pal4_97x61.webp",
        "webp/lossy_alpha_q80_63x65.webp",
        "webp/lossy_exif8_37x21.webp",
    ] {
        let b = read(n);
        let step = if b.len() > 2000 { 7 } else { 1 };
        for cut in (0..b.len()).step_by(step) {
            run(&b[..cut], &format!("{n} cut at {cut}"), Duration::from_secs(5));
        }
    }
}

#[test]
fn a_corrupted_byte_anywhere_in_a_tiny_image_never_panics() {
    for n in ["grad_64x64.png", "jpeg_q50_75x61_sub420.jpg", "webp/lossy_q0_m0_17x13.webp", "webp/lossless_pal4_97x61.webp"] {
        let b = read(n);
        for at in 0..b.len() {
            for v in [0x00u8, 0xff, b[at] ^ 0x01, b[at] ^ 0x80] {
                let mut c = b.clone();
                c[at] = v;
                run(&c, &format!("{n} byte {at} = {v:#04x}"), Duration::from_secs(5));
            }
        }
    }
}

#[test]
fn seeded_mutations_of_png_files_never_panic_or_hang() {
    let mut r = Rng(0x0070_6e67);
    for n in PNG_SEEDS {
        let seed = read(n);
        for case in 0..cases() {
            let m = mutate_png(&seed, &mut r);
            run(&m, &format!("{n} PNG mutation {case}"), Duration::from_secs(10));
        }
    }
}

#[test]
fn seeded_mutations_of_jpeg_files_never_panic_or_hang() {
    let mut r = Rng(0x6a70_6567);
    for n in JPEG_SEEDS {
        let seed = read(n);
        for case in 0..cases() {
            let m = mutate_jpeg(&seed, &mut r);
            run(&m, &format!("{n} JPEG mutation {case}"), Duration::from_secs(10));
        }
    }
}

#[test]
fn seeded_mutations_of_webp_files_never_panic_or_hang() {
    let mut r = Rng(0x7765_6270);
    for n in WEBP_SEEDS.iter().chain(["webp/animated_2f_8x8.webp"].iter()) {
        let seed = read(n);
        for case in 0..cases() {
            let m = mutate_webp(&seed, &mut r);
            run(&m, &format!("{n} WebP mutation {case}"), Duration::from_secs(10));
        }
    }
}

#[test]
fn an_animated_webp_and_its_prefixes_are_refused_not_decoded() {
    let b = read("webp/animated_2f_8x8.webp");
    for r in run(&b, "animated WebP", Duration::from_secs(5)) {
        assert_eq!(r, Err(VisionError::Unsupported("animated images are not supported".into())));
    }
    for cut in 0..b.len() {
        for r in run(&b[..cut], &format!("animated WebP cut at {cut}"), Duration::from_secs(5)) {
            assert!(r.is_err(), "cut at {cut}: {r:?}");
        }
    }
}

#[test]
fn a_jpeg_with_a_short_or_odd_exif_segment_is_an_error_or_an_image() {
    let seed = read("jpeg_q90_64x48.jpg");
    for len in [0u16, 1, 2, 3, 4, 7, 8, 13, 14, 15, 16, 64, 65535] {
        for body in [
            &b""[..],
            b"Exif\0\0",
            b"Exif\0\0II*\0",
            b"Exif\0\0MM\0*\0\0\0\x08",
            b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0",
        ] {
            let mut b = seed[..2].to_vec();
            b.extend_from_slice(&[0xff, 0xe1]);
            b.extend_from_slice(&len.to_be_bytes());
            b.extend_from_slice(body);
            b.extend_from_slice(&seed[2..]);
            run(&b, &format!("APP1 length {len}, body {body:?}"), Duration::from_secs(5));
        }
    }
}

#[test]
fn files_that_are_not_png_jpeg_or_webp_are_refused_by_name_not_decoded() {
    let others: [(&str, &[u8]); 10] = [
        ("empty", b""),
        ("one byte", b"\x89"),
        ("gif", b"GIF89a\x01\0\x01\0\0\0\0;"),
        ("riff wave", b"RIFF\x1a\0\0\0WAVEfmt \x0e\0\0\0"),
        ("bmp", b"BM\x1e\0\0\0\0\0\0\0\x1a\0\0\0\x0c\0\0\0\x01\0\x01\0\x01\0\x18\0"),
        ("tiff", b"II*\0\x08\0\0\0\0\0"),
        ("svg", b"<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'/>"),
        ("heic", b"\0\0\0\x18ftypheic\0\0\0\0mif1heic"),
        ("pdf", b"%PDF-1.4\n"),
        ("text", b"hello, I am not an image"),
    ];
    for (name, b) in others {
        for r in run(b, name, Duration::from_secs(1)) {
            assert!(matches!(r, Err(VisionError::Unsupported(_))) || b.len() < 2 && r.is_err(), "{name}: {r:?}");
        }
    }
}
