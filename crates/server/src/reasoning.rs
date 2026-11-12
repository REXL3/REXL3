//! Splits a thinking model's streamed text into `reasoning_content` and `content`.
//!
//! A thinking template ends the generation prompt with the open marker (`<think>\n`), so the reply starts inside the reasoning and the
//! close marker (`</think>`; the model's `ReasoningCapabilities` markers) closes it. (With thinking switched off the prompt already holds
//! the closed block and the whole reply is content.) The splitter holds back text that might be the start of the close marker until it
//! knows, so the tag never leaks into either stream, and it drops the newlines the template puts between the closing tag and the answer.

const CLOSE: &str = "</think>";

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Split {
    pub reasoning: String,
    pub content: String,
}

pub struct ReasoningSplitter {
    in_reasoning: bool,
    pending: String,
    /// The text that closes the reasoning (the model's `ReasoningCapabilities` markers).
    close: String,
    /// Skip newlines/spaces at the start of the content that follows `</think>`.
    trim_content_start: bool,
}

impl ReasoningSplitter {
    /// `starts_in_reasoning`: the prompt ended inside an open `<think>`.
    pub fn new(starts_in_reasoning: bool) -> Self {
        Self::with_close(starts_in_reasoning, CLOSE)
    }

    /// [`ReasoningSplitter::new`] for a model whose reasoning ends with `close` (`ReasoningSupport::close`).
    pub fn with_close(starts_in_reasoning: bool, close: &str) -> Self {
        Self {
            in_reasoning: starts_in_reasoning,
            pending: String::new(),
            close: close.to_string(),
            trim_content_start: starts_in_reasoning,
        }
    }

    /// Feed a piece of decoded text; returns the parts that can be shown now.
    pub fn push(&mut self, text: &str) -> Split {
        self.pending.push_str(text);
        let mut out = Split::default();
        loop {
            if self.in_reasoning {
                if let Some(i) = self.pending.find(self.close.as_str()) {
                    out.reasoning.push_str(&self.pending[..i]);
                    self.pending = self.pending[i + self.close.len()..].to_string();
                    self.in_reasoning = false;
                    continue;
                }
                let keep = partial_tag_len(&self.pending, &self.close);
                let cut = self.pending.len() - keep;
                out.reasoning.push_str(&self.pending[..cut]);
                self.pending = self.pending[cut..].to_string();
            } else {
                let mut s = std::mem::take(&mut self.pending);
                if self.trim_content_start {
                    let t = s.trim_start_matches(['\n', ' ']);
                    s = t.to_string();
                    if !s.is_empty() {
                        self.trim_content_start = false;
                    }
                }
                out.content.push_str(&s);
            }
            return out;
        }
    }

    /// End of stream: anything held back (a lone `</thi`) is real text after all.
    pub fn finish(&mut self) -> Split {
        let rest = std::mem::take(&mut self.pending);
        if self.in_reasoning {
            Split { reasoning: rest, content: String::new() }
        } else {
            Split { reasoning: String::new(), content: rest }
        }
    }

    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }
}

/// Counts the tokens of a reply that were thinking, for `usage.completion_tokens_details.reasoning_tokens`: every token up to and
/// including the one that closes the thought (TensorFold counts through the marker too; OpenAI's own field is the tokens spent
/// reasoning). A reply that never closes its thought is all reasoning; one that started outside a thought has none. It works on token
/// ids, so it does not depend on how the text was chunked or on what the splitter held back.
#[derive(Debug, Clone)]
pub struct ReasoningCount {
    open: bool,
    close: Option<u32>,
    n: usize,
}

impl ReasoningCount {
    /// `starts_in_reasoning`: the prompt ended inside an open `<think>`. `close`: the id of the token that closes the thought (`None`
    /// when the closing marker is not a single token: then the whole reply of a thinking prompt counts, an upper bound).
    pub fn new(starts_in_reasoning: bool, close: Option<u32>) -> Self {
        Self { open: starts_in_reasoning, close, n: 0 }
    }

    /// The next committed tokens of the reply, in order.
    pub fn push(&mut self, ids: &[u32]) {
        if !self.open {
            return;
        }
        match self.close.and_then(|c| ids.iter().position(|&t| t == c)) {
            Some(i) => {
                self.n += i + 1;
                self.open = false;
            }
            None => self.n += ids.len(),
        }
    }

    pub fn tokens(&self) -> usize {
        self.n
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `</think>`.
fn partial_tag_len(s: &str, close: &str) -> usize {
    (1..close.len())
        .rev()
        .find(|&n| s.len() >= n && s.is_char_boundary(s.len() - n) && close.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(parts: &[&str], starts: bool) -> Split {
        let mut sp = ReasoningSplitter::new(starts);
        let mut all = Split::default();
        for p in parts {
            let s = sp.push(p);
            all.reasoning += &s.reasoning;
            all.content += &s.content;
        }
        let s = sp.finish();
        all.reasoning += &s.reasoning;
        all.content += &s.content;
        all
    }

    #[test]
    fn splits_at_the_closing_tag_however_it_is_chunked() {
        let want = Split { reasoning: "I think. ".into(), content: "The answer is 4.".into() };
        let whole = "I think. </think>\n\nThe answer is 4.";
        assert_eq!(run(&[whole], true), want);
        // one character at a time, so the tag is split everywhere
        let chars: Vec<String> = whole.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(|s| s.as_str()).collect();
        assert_eq!(run(&refs, true), want);
        assert_eq!(run(&["I think. </th", "ink>\n", "\nThe answer is 4."], true), want);
    }

    #[test]
    fn thinking_off_is_all_content() {
        assert_eq!(run(&["Just ", "the answer."], false), Split { reasoning: "".into(), content: "Just the answer.".into() });
    }

    #[test]
    fn a_lookalike_that_is_not_the_tag_is_not_lost() {
        assert_eq!(run(&["a </th", "e cat"], true), Split { reasoning: "a </the cat".into(), content: "".into() });
        // truncated while holding a partial tag: it comes back out at the end
        assert_eq!(run(&["still thinking </thi"], true).reasoning, "still thinking </thi");
    }

    #[test]
    fn never_finishing_the_thought_is_all_reasoning() {
        let r = run(&["no close tag here"], true);
        assert_eq!((r.reasoning.as_str(), r.content.as_str()), ("no close tag here", ""));
    }

    #[test]
    fn reasoning_tokens_count_through_the_closing_marker_however_the_tokens_arrive() {
        const CLOSE: u32 = 9;
        let reply = [1u32, 2, 3, CLOSE, 4, 5, 6];
        // one batch, one token at a time, and a close split off at the start of its own batch: the same count
        for chunks in [vec![&reply[..]], reply.iter().map(std::slice::from_ref).collect(), vec![&reply[..3], &reply[3..]]] {
            let mut c = ReasoningCount::new(true, Some(CLOSE));
            for ch in chunks {
                c.push(ch);
            }
            assert_eq!(c.tokens(), 4, "the three thinking tokens and the marker");
        }
        // a second close later in the reply (quoted text) changes nothing
        let mut c = ReasoningCount::new(true, Some(CLOSE));
        c.push(&[1, CLOSE, 2, CLOSE, 3]);
        assert_eq!(c.tokens(), 2);
        // never closed: everything counts. Started outside a thought: nothing does, a stray marker included
        let mut c = ReasoningCount::new(true, Some(CLOSE));
        c.push(&[1, 2]);
        c.push(&[3]);
        assert_eq!(c.tokens(), 3);
        let mut c = ReasoningCount::new(false, Some(CLOSE));
        c.push(&[1, CLOSE, 2]);
        assert_eq!(c.tokens(), 0);
        // a close that is only the first token of its batch
        let mut c = ReasoningCount::new(true, Some(CLOSE));
        c.push(&[CLOSE, 7, 8]);
        assert_eq!(c.tokens(), 1);
    }

    #[test]
    fn multibyte_text_next_to_the_tag() {
        let r = run(&["caf\u{e9} \u{2603}</think>", "\n\n\u{4e2d}\u{6587}"], true);
        assert_eq!((r.reasoning.as_str(), r.content.as_str()), ("caf\u{e9} \u{2603}", "\u{4e2d}\u{6587}"));
    }
}
