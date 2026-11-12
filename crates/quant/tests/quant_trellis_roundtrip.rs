//! Trellis round trip: tiles encoded by our own quantizer (`tools/rexl3_quant/trellis`) and packed by its packer must
//! decode in gb-exl3's CPU decoder (the serving contract) exactly as the quantizer decodes them.
//!
//! `testdata/fixtures/quant_trellis_roundtrip.json` (from `tools/rexl3_quant/trellis/gen_fixture.py`) holds, for every
//! width the format has (1, 1.5, 2, 2.5, 3, 3.5, 4..8 bits), a [32, 48] mul1 trellis (2 x 3 tiles), the encoder's
//! 16-bit states and the quantizer's decode of the packed words. This requires `gb_exl3::states` to give back the
//! encoder's states and `gb_exl3::unpack` to give W_r bit for bit, plus the three codebook tables' FNV-1a.
//!
//! The large version (GPU-encoded tensors written by `gen_fixture.py --gate-dir DIR`) is off by default:
//! `REXL3_QUANT_TRELLIS_DIR=DIR cargo test --release -p gb-quant --test quant_trellis_roundtrip -- --ignored --nocapture`

use gb_exl3::{codebook, fnv1a, states, tile_positions, unpack, Bits, Codebook};
use gb_quant::exl3_trellis::states_seeded;
use serde_json::Value;
use std::path::Path;

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn hex_u16(s: &str) -> Vec<u16> {
    let b: Vec<u8> = (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect();
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn u16_fnv(v: &[u16]) -> u64 {
    fnv1a(v.iter().flat_map(|x| x.to_le_bytes()))
}

fn parse_fnv(v: &Value) -> u64 {
    u64::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
}

/// Checks one packed tensor: every tile's states, and W_r = `unpack` bit for bit. Returns the number of values checked.
fn check_tensor(label: &str, bits: Bits, kt: usize, nt: usize, trellis: &[i16], want_states: &[u16], want_wq: &[u16]) -> usize {
    let words = bits.tile_words();
    assert_eq!(trellis.len(), kt * nt * words, "{label}: trellis size");
    assert_eq!(want_states.len(), kt * nt * 256, "{label}: states size");
    assert_eq!(want_wq.len(), kt * nt * 256, "{label}: W_r size");
    let ends = bits.stream_ends();
    for t in 0..kt * nt {
        let got = states(&trellis[t * words..(t + 1) * words], bits, &ends);
        let want = &want_states[t * 256..(t + 1) * 256];
        if got[..] != *want {
            let p = (0..256).find(|&p| got[p] != want[p]).unwrap();
            panic!("{label}: tile {t} position {p}: decoder state {:#06x} != encoder state {:#06x}", got[p], want[p]);
        }
    }
    let wq = unpack(trellis, kt, nt, bits, &codebook(Codebook::Mul1));
    if wq != want_wq {
        let i = (0..wq.len()).find(|&i| wq[i] != want_wq[i]).unwrap();
        panic!("{label}: W_r[{}, {}] = {:#06x}, the quantizer decoded {:#06x}", i / (16 * nt), i % (16 * nt), wq[i], want_wq[i]);
    }
    wq.len()
}

#[test]
fn our_encoder_s_packed_tiles_decode_bit_for_bit_in_the_engine() {
    let p = root().join("testdata/fixtures/quant_trellis_roundtrip.json");
    let fx: Value =
        serde_json::from_str(&std::fs::read_to_string(p).expect("run tools/rexl3_quant/trellis/gen_fixture.py")).unwrap();
    for cb in Codebook::ALL {
        let want = parse_fnv(&fx["codebook_fnv"][cb.name()]);
        assert_eq!(u16_fnv(&codebook(cb)), want, "codebook {} table differs from the quantizer's", cb.name());
    }
    let (kt, nt) = (fx["kt"].as_u64().unwrap() as usize, fx["nt"].as_u64().unwrap() as usize);
    let widths = fx["widths"].as_array().unwrap();
    let mut seen = Vec::new();
    for w in widths {
        let twice = w["twice"].as_u64().unwrap() as u8;
        let bits = Bits::from_twice(twice).unwrap();
        assert_eq!(w["words"].as_u64().unwrap() as usize, bits.tile_words(), "{twice}/2 bits: words a tile");
        let trellis: Vec<i16> = hex_u16(w["trellis_hex"].as_str().unwrap()).into_iter().map(|x| x as i16).collect();
        let st = hex_u16(w["states_hex"].as_str().unwrap());
        let wq = hex_u16(w["wq_hex"].as_str().unwrap());
        assert_eq!(u16_fnv(&wq), parse_fnv(&w["wq_fnv"]), "{twice}/2 bits: fixture W_r hash");
        check_tensor(&format!("{} bits", bits.as_f64()), bits, kt, nt, &trellis, &st, &wq);
        seen.push(twice);
    }
    // every width the format has, the half widths included
    assert_eq!(seen, Bits::ALL_TWICE.to_vec(), "fixture widths");
    // the fixture tensor is not square, so the [kt, nt] tile order is exercised; the position order is the decoder's
    assert_ne!(kt, nt);
    let (r, c) = tile_positions();
    assert_eq!((r[1], c[1], r[2], c[4]), (1, 0, 8, 8), "trellis position order");
}

#[test]
#[ignore]
fn gpu_encoded_tensors_decode_bit_for_bit_in_the_engine() {
    let dir = std::env::var("REXL3_QUANT_TRELLIS_DIR").expect("set REXL3_QUANT_TRELLIS_DIR to gen_fixture.py --gate-dir output");
    let dir = Path::new(&dir);
    let mut total = 0usize;
    let mut widths = 0usize;
    for twice in Bits::ALL_TWICE {
        let d = dir.join(format!("twice{twice:02}"));
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(d.join("meta.json")).unwrap()).unwrap();
        let (kt, nt) = (meta["kt"].as_u64().unwrap() as usize, meta["nt"].as_u64().unwrap() as usize);
        let bits = Bits::from_twice(twice).unwrap();
        let rd = |name: &str| -> Vec<u16> {
            std::fs::read(d.join(name)).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        };
        let trellis: Vec<i16> = rd("trellis.bin").into_iter().map(|x| x as i16).collect();
        let (st, wq) = (rd("states.bin"), rd("wq.bin"));
        assert_eq!(u16_fnv(&wq), parse_fnv(&meta["wq_fnv"]), "{twice}/2 bits: W_r hash");
        let n = check_tensor(&format!("{} bits", bits.as_f64()), bits, kt, nt, &trellis, &st, &wq);
        eprintln!("{:>4} bits: {kt}x{nt} tiles, {n} values, states and W_r equal (fnv {:#018x})", bits.as_f64(), u16_fnv(&wq));
        total += n;
        widths += 1;
    }
    eprintln!("{widths} widths, {total} values: all equal");
}

/// Free-start tiles (encoder tail="free", no tail-biting) with their start states: `states_seeded` must give back the
/// encoder's states and the codebook its values, bit for bit. Written by `gen_fixture.py --free-dir DIR`:
/// `REXL3_QUANT_TRELLIS_FREE_DIR=DIR cargo test --release -p gb-quant --test quant_trellis_roundtrip -- --ignored`
#[test]
#[ignore]
fn gpu_free_start_tiles_decode_bit_for_bit_with_their_start_states() {
    let dir = std::env::var("REXL3_QUANT_TRELLIS_FREE_DIR").expect("set REXL3_QUANT_TRELLIS_FREE_DIR");
    let dir = Path::new(&dir);
    let table = codebook(Codebook::Mul1);
    let (mut total, mut widths, mut differ) = (0usize, 0usize, 0usize);
    for twice in Bits::ALL_TWICE {
        let d = dir.join(format!("twice{twice:02}"));
        let bits = Bits::from_twice(twice).unwrap();
        let rd = |name: &str| -> Vec<u16> {
            std::fs::read(d.join(name)).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        };
        let trellis: Vec<i16> = rd("trellis.bin").into_iter().map(|x| x as i16).collect();
        let (st, start, vals) = (rd("states.bin"), rd("start.bin"), rd("values.bin"));
        let words = bits.tile_words();
        let tiles = trellis.len() / words;
        assert_eq!(st.len(), tiles * 256);
        assert_eq!(start.len(), tiles);
        let ends = bits.stream_ends();
        for t in 0..tiles {
            let tw = &trellis[t * words..(t + 1) * words];
            let got = states_seeded(tw, bits, &ends, start[t]);
            let want = &st[t * 256..(t + 1) * 256];
            if got[..] != *want {
                let p = (0..256).find(|&p| got[p] != want[p]).unwrap();
                panic!("{} bits: tile {t} position {p}: decoder {:#06x} != encoder {:#06x}", bits.as_f64(), got[p], want[p]);
            }
            for p in 0..256 {
                assert_eq!(table[got[p] as usize], vals[t * 256 + p], "{} bits: tile {t} value {p}", bits.as_f64());
            }
            // the tail-biting read differs where the start is not the stream's tail (the point of storing it)
            if states(tw, bits, &ends)[0] != got[0] {
                differ += 1;
            }
        }
        eprintln!("{:>4} bits: {tiles} free-start tiles, states and values equal", bits.as_f64());
        total += tiles;
        widths += 1;
    }
    eprintln!("{widths} widths, {total} tiles: all equal; {differ} tiles start off their own tail");
    assert!(differ > 0, "no tile used the free start");
}
