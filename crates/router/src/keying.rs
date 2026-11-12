//! The default affinity key: a hash of the part of a request that shares a KV/prefix cache on a backend.
//!
//! For a chat completion that is the system prompt(s) plus the first user message (every later turn of the same
//! conversation starts with exactly these, so its cache lives on the backend that served turn one); for a text
//! completion it is the prompt. The JSON body is walked with a streaming visitor that skips everything it does not need
//! (no `Value` tree, no copy of image data URIs) and stops at the first user message, and the text it hashes is capped
//! at `cap` bytes, so a 2 MiB body costs a scan of the bytes before the end of the first user message at most.
//!
//! The hash is FNV-1a 64 finished with a splitmix64 mix: deterministic across runs and Rust versions (router restarts
//! and several routers in front of the same fleet agree), not a security boundary.
use serde::de::{DeserializeSeed, Error as _, IgnoredAny, MapAccess, SeqAccess, Visitor};
use std::cell::RefCell;
use std::fmt;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 over `bytes`, avalanched.
pub fn fnv64(bytes: &[u8]) -> u64 {
    mix64(fnv_update(FNV_OFFSET, bytes))
}

fn fnv_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// splitmix64 finalizer: spreads the low-entropy FNV tail over all 64 bits (the rendezvous comparison below depends on it).
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Rendezvous (highest-random-weight) score of `key` against a backend hash.
pub fn rendezvous(key: u64, backend_hash: u64) -> u64 {
    mix64(key ^ backend_hash.rotate_left(17))
}

/// Where an affinity key came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    /// The client's `x-affinity-key` header.
    Header,
    /// The hash of the request's cacheable prefix.
    Prompt,
}

pub fn header_key(value: &[u8]) -> u64 {
    fnv64(&[b"h:", value].concat())
}

struct Capture {
    hash: u64,
    /// Bytes of text committed to the hash.
    committed: usize,
    cap: usize,
    role: String,
    content: String,
    stop: bool,
}

impl Capture {
    fn room(&self) -> usize {
        self.cap.saturating_sub(self.committed + self.content.len())
    }

    fn push(&mut self, s: &str) {
        let room = self.room();
        if room == 0 {
            return;
        }
        let mut n = s.len().min(room);
        while !s.is_char_boundary(n) {
            n -= 1;
        }
        self.content.push_str(&s[..n]);
    }

    fn commit_message(&mut self) {
        if self.role.is_empty() && self.content.is_empty() {
            return;
        }
        let h = fnv_update(self.hash, self.role.as_bytes());
        let h = fnv_update(h, b"\x00");
        let h = fnv_update(h, self.content.as_bytes());
        self.hash = fnv_update(h, b"\x01");
        self.committed += self.content.len() + self.role.len() + 2;
        if self.role == "user" {
            self.stop = true;
        }
        if self.committed >= self.cap {
            self.stop = true;
        }
        self.role.clear();
        self.content.clear();
    }
}

type Cell = RefCell<Capture>;

/// Derive the prefix key from a (possibly truncated) request body, `None` when it has no cacheable prefix (not JSON, no
/// `messages`/`prompt`, empty). `cap` bounds the hashed text in bytes.
pub fn prefix_key(body: &[u8], cap: usize) -> Option<u64> {
    if cap == 0 || body.iter().find(|b| !b.is_ascii_whitespace()) != Some(&b'{') {
        return None;
    }
    let cell: Cell =
        RefCell::new(Capture { hash: FNV_OFFSET, committed: 0, cap, role: String::new(), content: String::new(), stop: false });
    let mut de = serde_json::Deserializer::from_slice(body);
    // The error is expected: the visitors abort after the first user message, and a truncated body ends mid-token. What
    // was captured before that is what counts.
    let _ = serde::Deserializer::deserialize_map(&mut de, Top(&cell));
    let mut c = cell.into_inner();
    c.commit_message(); // a message cut off by the end of a truncated body still contributes its text so far
    (c.committed > 0).then(|| mix64(c.hash))
}

struct Top<'a>(&'a Cell);

impl<'de> Visitor<'de> for Top<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a request object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "messages" => map.next_value_seed(Messages(self.0))?,
                "prompt" => map.next_value_seed(PromptText(self.0))?,
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
            if self.0.borrow().stop {
                return Err(A::Error::custom("done"));
            }
        }
        Ok(())
    }
}

struct Messages<'a>(&'a Cell);

impl<'de> DeserializeSeed<'de> for Messages<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Messages<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a message list")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while seq.next_element_seed(Message(self.0))?.is_some() {
            if self.0.borrow().stop {
                return Err(A::Error::custom("done"));
            }
        }
        Ok(())
    }
    // `messages` of the wrong type is the backend's error to report.
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(())
    }
    fn visit_str<E>(self, _: &str) -> Result<(), E> {
        Ok(())
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }
    fn visit_bool<E>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E>(self, _: f64) -> Result<(), E> {
        Ok(())
    }
}

struct Message<'a>(&'a Cell);

impl<'de> DeserializeSeed<'de> for Message<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Message<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a message object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "role" => {
                    if let Ok(r) = map.next_value::<String>() {
                        self.0.borrow_mut().role = r;
                    }
                }
                "content" => map.next_value_seed(Content(self.0))?,
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        self.0.borrow_mut().commit_message();
        Ok(())
    }
    fn visit_str<E>(self, _: &str) -> Result<(), E> {
        Ok(())
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }
}

/// `content`: a string, or a list of parts (`{"type":"text","text":..}`, `{"type":"image_url",..}`, ...).
struct Content<'a>(&'a Cell);

impl<'de> DeserializeSeed<'de> for Content<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Content<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("message content")
    }
    fn visit_str<E>(self, s: &str) -> Result<(), E> {
        self.0.borrow_mut().push(s);
        Ok(())
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while seq.next_element_seed(Part(self.0))?.is_some() {}
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(())
    }
}

struct Part<'a>(&'a Cell);

impl<'de> DeserializeSeed<'de> for Part<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Part<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a content part")
    }
    fn visit_str<E>(self, s: &str) -> Result<(), E> {
        self.0.borrow_mut().push(s);
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "text" => map.next_value_seed(PromptText(self.0))?,
                "type" => {
                    if let Ok(t) = map.next_value::<String>() {
                        if t != "text" {
                            // A non-text part (image, audio, video): its type is part of the prefix, its bytes are not read.
                            let mut c = self.0.borrow_mut();
                            c.push("\u{1}");
                            c.push(&t);
                        }
                    }
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(())
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }
}

/// A completion `prompt` (string or list of strings; token-id lists are ignored) or a text part's `text`.
struct PromptText<'a>(&'a Cell);

impl<'de> DeserializeSeed<'de> for PromptText<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for PromptText<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a prompt")
    }
    fn visit_str<E>(self, s: &str) -> Result<(), E> {
        self.0.borrow_mut().push(s);
        Ok(())
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while let Some(()) = seq.next_element_seed(PromptText(self.0))? {}
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(())
    }
    fn visit_bool<E>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E>(self, _: f64) -> Result<(), E> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(body: &str) -> Option<u64> {
        prefix_key(body.as_bytes(), 32 * 1024)
    }

    #[test]
    fn same_system_and_first_user_share_a_key_whatever_follows() {
        let a = r#"{"model":"m","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hello"}],"max_tokens":8}"#;
        let b = r#"{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"and then?"}],"stream":true,"temperature":0.2}"#;
        let c = r#"{"messages":[{"content":"hello","role":"user"}]}"#;
        assert!(key(a).is_some());
        assert_eq!(key(a), key(b), "later turns and sampling fields do not change the key");
        assert_ne!(key(a), key(c), "a different system prompt is a different prefix");
        assert_ne!(key(a), key(r#"{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"goodbye"}]}"#));
    }

    #[test]
    fn content_parts_and_key_order_are_handled() {
        let a = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}"#;
        let b = r#"{"messages":[{"content":[{"text":"describe","type":"text"},{"type":"image_url","image_url":{"url":"data:image/png;base64,BBBB"}}],"role":"user"}]}"#;
        assert!(key(a).is_some());
        assert_eq!(key(a), key(b));
        let c = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"other"}]}]}"#;
        assert_ne!(key(a), key(c));
    }

    #[test]
    fn completions_prompt_is_the_key() {
        let p = prefix_key(br#"{"model":"m","prompt":"once upon a time","stream":true}"#, 1024);
        assert!(p.is_some());
        let q = prefix_key(br#"{"prompt":["once upon a time"]}"#, 1024);
        assert!(q.is_some());
        let r = prefix_key(br#"{"prompt":"another one"}"#, 1024);
        assert_ne!(p, r);
    }

    #[test]
    fn no_prefix_for_bodies_without_one() {
        assert_eq!(key(""), None);
        assert_eq!(key("not json"), None);
        assert_eq!(key("{}"), None);
        assert_eq!(key(r#"{"messages":[]}"#), None);
        assert_eq!(key(r#"{"messages":"oops"}"#), None);
        assert_eq!(key(r#"{"prompt":[1,2,3]}"#), None);
    }

    #[test]
    fn text_is_bounded_and_a_truncated_body_still_keys() {
        let long = "x".repeat(100_000);
        let body = format!(r#"{{"messages":[{{"role":"system","content":"{long}"}},{{"role":"user","content":"a"}}]}}"#);
        let body2 = format!(r#"{{"messages":[{{"role":"system","content":"{long}"}},{{"role":"user","content":"b"}}]}}"#);
        // The cap falls inside the shared system prompt: both bodies key alike (and the walk stopped there).
        assert!(prefix_key(body.as_bytes(), 4096).is_some());
        assert_eq!(prefix_key(body.as_bytes(), 4096), prefix_key(body2.as_bytes(), 4096));
        // Cut the body mid-user-message: the text before the cut is still a key.
        let cut = &br#"{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"a long docu"#[..];
        assert!(prefix_key(cut, 4096).is_some());
    }

    #[test]
    fn rendezvous_prefers_one_backend_per_key_and_spreads() {
        let (a, b) = (fnv64(b"192.0.2.1:8001"), fnv64(b"192.0.2.2:8001"));
        let mut to_a = 0;
        for i in 0..1000u64 {
            let k = fnv64(&i.to_le_bytes());
            let first = rendezvous(k, a) > rendezvous(k, b);
            assert_eq!(first, rendezvous(k, a) > rendezvous(k, b));
            to_a += usize::from(first);
        }
        assert!((400..600).contains(&to_a), "roughly even split: {to_a}");
    }
}
