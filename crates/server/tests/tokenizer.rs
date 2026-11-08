//! The Rust tokenizer against `tokenizers` in Python (tools/gen_tokenizer_fixtures.py), and streaming decode.
//! Needs the model folders: set REXL3_QWEN_DIR and REXL3_GLM_DIR (each test skips without its folder).

use gb_server::chat_format::ChatFormat;
use gb_server::formats::glm::GlmFormat;
use gb_server::tokenizer::{StreamDecoder, Tok};
use serde_json::{json, Value};
use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    std::env::var("REXL3_QWEN_DIR").ok().map(PathBuf::from).filter(|p| p.join("tokenizer.json").exists())
}

fn cases() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tokenizer_cases.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn ids_and_decodes_match_python() {
    let Some(d) = dir() else { return eprintln!("REXL3_QWEN_DIR not set: skipping") };
    let tok = Tok::from_dir(&d).unwrap();
    let fx = cases();
    assert_eq!(tok.vocab as u64, fx["meta"]["vocab_size"].as_u64().unwrap());
    for (name, id) in fx["meta"]["special"].as_object().unwrap() {
        assert_eq!(tok.token_to_id(name).map(u64::from), id.as_u64(), "{name}");
    }
    assert_eq!(tok.eos, vec![248046, 248044]);
    for c in fx["cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(tok.encode(text).unwrap(), want, "encode {text:?}");
        assert_eq!(tok.decode(&want, false), c["decoded"].as_str().unwrap(), "decode {text:?}");
        assert_eq!(tok.decode(&want, true), c["decoded_skip"].as_str().unwrap(), "decode skip {text:?}");
    }
}

#[test]
fn streaming_decode_equals_a_full_decode_for_any_chunking() {
    let Some(d) = dir() else { return eprintln!("REXL3_QWEN_DIR not set: skipping") };
    let tok = Tok::from_dir(&d).unwrap();
    for c in cases()["cases"].as_array().unwrap() {
        let ids: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let full = tok.decode(&ids, true);
        for chunk in [1usize, 2, 3, 7] {
            let mut sd = StreamDecoder::new(true);
            let mut got = String::new();
            for part in ids.chunks(chunk) {
                let piece = sd.push(&tok, part);
                assert!(!piece.contains('\u{FFFD}') || full.contains('\u{FFFD}'), "a partial character leaked: {piece:?}");
                got.push_str(&piece);
            }
            got.push_str(&sd.finish(&tok));
            assert_eq!(got, full, "chunk size {chunk}, text {:?}", c["text"]);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// GLM-5.3-Flash: needs the model folder, set REXL3_GLM_DIR (the HF folder or the EXL3 pack: tokenizer.json, generation_config.json)

fn glm_dir() -> Option<PathBuf> {
    std::env::var("REXL3_GLM_DIR").ok().map(PathBuf::from).filter(|p| p.join("tokenizer.json").exists())
}

fn glm_fixture(name: &str) -> Value {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn ids_of(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

#[test]
fn glm_ids_and_decodes_match_python() {
    let Some(d) = glm_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let tok = Tok::from_dir(&d).unwrap();
    let fx = glm_fixture("glm53_tokenizer_cases.json");
    assert_eq!(tok.vocab as u64, fx["meta"]["vocab_size"].as_u64().unwrap());
    assert_eq!(tok.eos, vec![154820, 154827, 154829], "<|endoftext|>, <|user|>, <|observation|>");
    assert_eq!(tok.eos, ids_of(&fx["meta"]["eos"]));
    let added = fx["meta"]["added_tokens"].as_object().unwrap();
    assert_eq!(added.len(), 36);
    for (content, info) in added {
        assert_eq!(tok.token_to_id(content).map(u64::from), info["id"].as_u64(), "{content}");
    }
    // the ids the chat format and the engine rely on
    for (t, id) in [
        ("[gMASK]", 154822),
        ("<sop>", 154824),
        ("<|system|>", 154826),
        ("<|user|>", 154827),
        ("<|assistant|>", 154828),
        ("<|observation|>", 154829),
        ("<|begin_of_image|>", 154830),
        ("<|end_of_image|>", 154831),
        ("<|begin_of_video|>", 154832),
        ("<|end_of_video|>", 154833),
        ("<think>", 154841),
        ("</think>", 154842),
        ("<tool_call>", 154843),
        ("</tool_call>", 154844),
        ("<tool_response>", 154845),
        ("</tool_response>", 154846),
        ("<arg_key>", 154847),
        ("</arg_key>", 154848),
        ("<arg_value>", 154849),
        ("</arg_value>", 154850),
        ("<|image|>", 154854),
        ("<|video|>", 154855),
    ] {
        assert_eq!(tok.token_to_id(t), Some(id), "{t}");
    }
    assert_eq!(tok.think_close, Some(154842));
    for c in fx["cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want = ids_of(&c["ids"]);
        assert_eq!(tok.encode(text).unwrap(), want, "encode {text:?}");
        assert_eq!(tok.decode(&want, false), c["decoded"].as_str().unwrap(), "decode {text:?}");
        assert_eq!(tok.decode(&want, true), c["decoded_skip"].as_str().unwrap(), "decode skip {text:?}");
    }
}

/// `skip_special_tokens` (the server's default) drops the role tokens but must keep the reasoning and tool-call markers: they are ordinary
/// added tokens, and the reasoning splitter and the tool-call splitter read them from the decoded text.
#[test]
fn glm_decode_keeps_the_reasoning_and_tool_call_markers_and_drops_the_role_tokens() {
    let Some(d) = glm_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let tok = Tok::from_dir(&d).unwrap();
    let text = "<|user|>a<think>b</think>c<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call><|observation|>";
    let ids = tok.encode(text).unwrap();
    assert_eq!(tok.decode(&ids, false), text);
    assert_eq!(tok.decode(&ids, true), "a<think>b</think>c<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>");
    for marker in ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"]
    {
        assert_eq!(tok.encode(marker).unwrap().len(), 1, "{marker} is a single token");
    }
}

#[test]
fn glm_streaming_decode_equals_a_full_decode_for_any_chunking() {
    let Some(d) = glm_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let tok = Tok::from_dir(&d).unwrap();
    for c in glm_fixture("glm53_tokenizer_cases.json")["cases"].as_array().unwrap() {
        let ids = ids_of(&c["ids"]);
        let full = tok.decode(&ids, true);
        for chunk in [1usize, 2, 3, 7] {
            let mut sd = StreamDecoder::new(true);
            let mut got = String::new();
            for part in ids.chunks(chunk) {
                let piece = sd.push(&tok, part);
                assert!(!piece.contains('\u{FFFD}') || full.contains('\u{FFFD}'), "a partial character leaked: {piece:?}");
                got.push_str(&piece);
            }
            got.push_str(&sd.finish(&tok));
            assert_eq!(got, full, "chunk size {chunk}, text {:?}", c["text"]);
        }
    }
}

/// messages -> chat format -> tokenizer: the ids HF's `apply_chat_template(tokenize=True)` produces, for every fixture conversation.
#[test]
fn glm_conversations_tokenize_to_the_ids_hf_produces() {
    let Some(d) = glm_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let tok = Tok::from_dir(&d).unwrap();
    let f = GlmFormat::new(
        &std::fs::read_to_string(format!("{}/tests/fixtures/glm53_chat_template.jinja", env!("CARGO_MANIFEST_DIR"))).unwrap(),
    )
    .unwrap();
    let fx = glm_fixture("glm53_chat_template_cases.json");
    let (mut checked, mut longest) = (0, 0);
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let Some(want) = c["ids"].as_array() else { continue };
        let mut kwargs = c["kwargs"].as_object().unwrap().clone();
        if kwargs.remove("add_generation_prompt").is_some() {
            // generation prompt off: only the tokenizer half applies
            assert_eq!(tok.encode(c["rendered"].as_str().unwrap()).unwrap(), ids_of(&c["ids"]), "{name}");
            continue;
        }
        if c["messages"].as_array().unwrap().is_empty() {
            continue; // HF refuses the empty conversation (tests/chat_format.rs)
        }
        let r = f.render(&c["messages"], &kwargs).unwrap_or_else(|e| panic!("{name}: {e}"));
        let got = tok.encode(&r.text).unwrap();
        assert_eq!(got, ids_of(&c["ids"]), "{name}");
        longest = longest.max(want.len());
        checked += 1;
    }
    assert!(checked >= 90 && longest > 2500, "{checked} conversations, longest {longest} tokens");
}

/// The image path end to end, with the real tokenizer: the template's one `<|image|>` per image, expanded with the token count the
/// format computes from the image size, is the sequence `Glm5NextProcessor` produces for synthetic images of those sizes.
#[test]
fn glm_image_prompts_expand_to_the_ids_the_hf_processor_produces() {
    let Some(d) = glm_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let tok = Tok::from_dir(&d).unwrap();
    let f = GlmFormat::new(
        &std::fs::read_to_string(format!("{}/tests/fixtures/glm53_chat_template.jinja", env!("CARGO_MANIFEST_DIR"))).unwrap(),
    )
    .unwrap();
    let fx = glm_fixture("glm53_chat_template_cases.json");
    assert_eq!(fx["meta"]["image_token_id"].as_u64(), f.image_placeholder(&tok).map(|(id, _)| id as u64).ok());
    let (mut cases, mut images, mut biggest) = (0, 0, 0);
    for c in fx["cases"].as_array().unwrap() {
        let Some(sizes) = c.get("images").and_then(|i| i.as_array()) else { continue };
        let name = c["name"].as_str().unwrap();
        let ids = tok.encode(c["rendered"].as_str().unwrap()).unwrap();
        let (mut counts, mut grids) = (Vec::new(), Vec::new());
        for s in sizes {
            let (h, w) = (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize);
            counts.push(f.image_tokens(h, w).unwrap().unwrap());
            let (gh, gw) = f.image.grid(h, w).unwrap();
            grids.push(json!([1, gh, gw]));
        }
        assert_eq!(Value::Array(grids), c["image_grids"], "{name}: the grid the vision tower will see");
        let (expanded, spans) = f.expand_images(&tok, &ids, &counts).unwrap_or_else(|e| panic!("{name}: {}", e.message));
        assert_eq!(expanded, ids_of(&c["expanded_ids"]), "{name}");
        assert_eq!(spans.iter().map(|s| s.tokens).collect::<Vec<_>>(), counts, "{name}");
        // every run sits between the begin and end markers
        for sp in &spans {
            assert_eq!(expanded[sp.start - 1], 154830, "{name}: <|begin_of_image|>");
            assert_eq!(expanded[sp.start + sp.tokens], 154831, "{name}: <|end_of_image|>");
        }
        cases += 1;
        images += counts.len();
        biggest = biggest.max(counts.iter().copied().max().unwrap());
    }
    assert!(cases >= 15 && images >= 20 && biggest >= 400, "{cases} cases, {images} images, biggest run {biggest}");
}
