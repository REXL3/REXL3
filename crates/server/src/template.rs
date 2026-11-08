//! Chat template rendering, compatible with HuggingFace's Jinja environment.
//!
//! HF compiles chat templates in a sandboxed Jinja environment with `trim_blocks`, `lstrip_blocks`, the `loopcontrols`
//! extension (`{% break %}`, `{% continue %}`; the cargo feature `loop_controls` here), a `raise_exception` global, and a
//! `tojson` filter that is `json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`: Python's
//! default `", "` and `": "` separators, insertion-ordered keys, non-ASCII left as is, floats as `repr(float)`. A different
//! `tojson` renders tool definitions differently and so changes the prompt's tokens. `tests/template.rs` holds renders from
//! the real Python environment and requires byte equality.

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Serde, Value};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{Map, Value as Json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// The template itself called `raise_exception` (bad role, no messages, ...): the request is invalid.
    Raised(String),
    /// Construction stopped before appending beyond the request limit.
    Limit,
    /// Anything else: a broken template, or input the template cannot handle.
    Render(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::Raised(m) => write!(f, "{m}"),
            TemplateError::Limit => write!(f, "rendered prompt byte limit exceeded"),
            TemplateError::Render(m) => write!(f, "chat template failed: {m}"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// The `json.dumps` options a template can pass through `tojson(...)`, as in HuggingFace's override:
/// `tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JsonOpts {
    pub ensure_ascii: bool,
    /// `indent=N` is N spaces, `indent="\t"` the string itself; `None` is the single-line form.
    pub indent: Option<String>,
    /// `(item_separator, key_separator)`; `None` is Python's default, `(", ", ": ")`, or `(",", ": ")` when `indent` is set.
    pub separators: Option<(String, String)>,
    pub sort_keys: bool,
}

/// Python's `json.dumps(v, ensure_ascii=False)`: `", "` and `": "` separators, keys in insertion order, non-ASCII as is.
pub fn python_json(v: &Json) -> String {
    python_json_with(v, &JsonOpts::default())
}

/// Python's `json.dumps(v, **opts)`, byte for byte (floats as `repr(float)`, strings escaped as `json.encoder` does).
pub fn python_json_with(v: &Json, opts: &JsonOpts) -> String {
    let (item, key) = match (&opts.separators, &opts.indent) {
        (Some((i, k)), _) => (i.as_str(), k.as_str()),
        (None, Some(_)) => (",", ": "),
        (None, None) => (", ", ": "),
    };
    let mut w = JsonWriter { opts, item, key, out: LimitedText::new(current_render_limit().unwrap_or(usize::MAX)) };
    w.value(v, 0);
    w.out.text
}

struct JsonWriter<'a> {
    opts: &'a JsonOpts,
    item: &'a str,
    key: &'a str,
    out: LimitedText,
}

impl JsonWriter<'_> {
    fn newline(&mut self, level: usize) {
        if self.out.exceeded {
            return;
        }
        if let Some(ind) = &self.opts.indent {
            self.out.push('\n');
            for _ in 0..level {
                if self.out.exceeded {
                    break;
                }
                self.out.push_str(ind);
            }
        }
    }

    fn value(&mut self, v: &Json, level: usize) {
        if self.out.exceeded {
            return;
        }
        match v {
            Json::Null => self.out.push_str("null"),
            Json::Bool(b) => self.out.push_str(if *b { "true" } else { "false" }),
            Json::Number(n) => match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => self.out.push_str(&i.to_string()),
                (None, Some(u)) => self.out.push_str(&u.to_string()),
                _ => self.out.push_str(&python_float_repr(n.as_f64().unwrap_or(f64::NAN))),
            },
            Json::String(s) => self.string(s),
            Json::Array(a) => {
                if a.is_empty() {
                    self.out.push_str("[]");
                    return;
                }
                self.out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(self.item);
                    }
                    self.newline(level + 1);
                    self.value(x, level + 1);
                }
                self.newline(level);
                self.out.push(']');
            }
            Json::Object(m) => {
                if m.is_empty() {
                    self.out.push_str("{}");
                    return;
                }
                let mut entries: Vec<(&String, &Json)> = m.iter().collect();
                if self.opts.sort_keys {
                    entries.sort_by(|a, b| a.0.cmp(b.0)); // UTF-8 byte order is code point order, as Python sorts `str`
                }
                self.out.push('{');
                for (i, (k, x)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(self.item);
                    }
                    self.newline(level + 1);
                    self.string(k);
                    self.out.push_str(self.key);
                    self.value(x, level + 1);
                }
                self.newline(level);
                self.out.push('}');
            }
        }
    }

    /// `json.encoder`'s `encode_basestring` (and `encode_basestring_ascii` when `ensure_ascii`).
    fn string(&mut self, s: &str) {
        self.out.push('"');
        for c in s.chars() {
            if self.out.exceeded {
                break;
            }
            match c {
                '"' => self.out.push_str("\\\""),
                '\\' => self.out.push_str("\\\\"),
                '\n' => self.out.push_str("\\n"),
                '\r' => self.out.push_str("\\r"),
                '\t' => self.out.push_str("\\t"),
                '\u{8}' => self.out.push_str("\\b"),
                '\u{c}' => self.out.push_str("\\f"),
                c if (c as u32) < 0x20 || (self.opts.ensure_ascii && c as u32 > 0x7e) => {
                    let mut units = [0u16; 2];
                    for u in c.encode_utf16(&mut units) {
                        self.out.push_str(&format!("\\u{u:04x}"));
                    }
                }
                c => self.out.push(c),
            }
        }
        self.out.push('"');
    }
}

/// Python's `repr(float)` (what `json.dumps` writes for a float): shortest round-trip digits, positional notation for
/// decimal exponents in `-4..16`, else `d.ddde+XX` with at least two exponent digits. serde_json's own formatting differs
/// (`1e16`, `1e-7`), which would change a tool-call argument's tokens.
pub fn python_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let sci = format!("{f:e}"); // shortest round-trip digits: "-1.5e-5", "1e16", "0e0"
    let (neg, sci) = match sci.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, sci.as_str()),
    };
    let (mant, exp) = sci.split_once('e').unwrap_or((sci, "0"));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp.parse::<i32>().unwrap_or(0) + 1; // value = 0.DIGITS x 10^decpt
    let n = digits.len() as i32;
    let body = if decpt <= -4 || decpt > 16 {
        let e = decpt - 1;
        let m = if n > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt >= n {
        format!("{digits}{}.0", "0".repeat((decpt - n) as usize))
    } else {
        format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    };
    if neg {
        format!("-{body}")
    } else {
        body
    }
}

/// The keyword arguments of the `tojson` filter, read as HuggingFace's Python function reads them. An argument this function
/// does not know is an error, never ignored: a template that asked for a rendering that cannot be given must not render quietly.
fn json_opts(kwargs: &Kwargs) -> Result<JsonOpts, Error> {
    let bad = |what: &str| Error::new(ErrorKind::InvalidOperation, format!("tojson: {what}"));
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let indent = match kwargs.get::<Option<Value>>("indent")? {
        Some(v) if !v.is_none() && !v.is_undefined() => match (v.as_i64(), v.as_str()) {
            (Some(n), _) => {
                let n = usize::try_from(n.max(0)).map_err(|_| bad("indent is too large"))?;
                if n > current_render_limit().unwrap_or(8 << 20) {
                    return Err(bad("indent exceeds render limit"));
                }
                Some(" ".repeat(n))
            }
            (None, Some(s)) => Some(s.to_string()),
            _ => return Err(bad("indent must be an integer or a string")),
        },
        _ => None,
    };
    let separators = match kwargs.get::<Option<Value>>("separators")? {
        Some(v) if !v.is_none() && !v.is_undefined() => {
            let parts: Option<Vec<String>> = v
                .try_iter()
                .map_err(|_| bad("separators must be a pair of strings"))?
                .map(|x| x.as_str().map(str::to_string))
                .collect();
            match parts.and_then(|p| <[String; 2]>::try_from(p).ok()) {
                Some([i, k]) => Some((i, k)),
                None => return Err(bad("separators must be a pair of strings")),
            }
        }
        _ => None,
    };
    kwargs.assert_all_used()?;
    Ok(JsonOpts { ensure_ascii: ensure_ascii.unwrap_or(false), indent, separators, sort_keys: sort_keys.unwrap_or(false) })
}

thread_local! {
    static RENDER_LIMIT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static RENDER_EXCEEDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
fn current_render_limit() -> Option<usize> {
    RENDER_LIMIT.with(|v| v.get())
}
/// Synchronous, worker-local scope; RAII restores limits even on template panic. Never crosses an await.
pub fn with_render_limit<R>(limit: usize, f: impl FnOnce() -> R) -> R {
    struct Reset(Option<usize>, bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            RENDER_LIMIT.with(|v| v.set(self.0));
            RENDER_EXCEEDED.with(|v| v.set(self.1));
        }
    }
    let old = RENDER_LIMIT.with(|v| v.replace(Some(limit)));
    let exceeded = RENDER_EXCEEDED.with(|v| v.replace(false));
    let _reset = Reset(old, exceeded);
    f()
}
struct LimitedText {
    text: String,
    limit: usize,
    exceeded: bool,
}
impl LimitedText {
    fn new(limit: usize) -> Self {
        Self { text: String::new(), limit, exceeded: false }
    }
    fn push_str(&mut self, s: &str) {
        if self.exceeded {
            return;
        }
        if self.text.len().checked_add(s.len()).is_none_or(|n| n > self.limit) {
            self.exceeded = true;
            RENDER_EXCEEDED.with(|v| v.set(true));
        } else {
            self.text.push_str(s);
        }
    }
    fn push(&mut self, c: char) {
        let mut buf = [0; 4];
        self.push_str(c.encode_utf8(&mut buf));
    }
}

pub struct ChatTemplate {
    env: Environment<'static>,
    source: String,
}

impl ChatTemplate {
    /// Closed one-user-turn decision rendering; cap bytes WHILE Jinja writes, not afterwards.
    pub fn render_decision_bounded(
        &self,
        messages: &Json,
        kwargs: &Map<String, Json>,
        limit: usize,
    ) -> Result<String, TemplateError> {
        struct Bounded {
            out: Vec<u8>,
            limit: usize,
        }
        impl std::io::Write for Bounded {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                if b.len() > self.limit.saturating_sub(self.out.len()) {
                    return Err(std::io::Error::other("decision rendered prompt exceeds byte limit"));
                }
                self.out.extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut ctx = kwargs.clone();
        ctx.insert("messages".into(), messages.clone());
        ctx.insert("add_generation_prompt".into(), Json::Bool(true));
        let tpl = self.env.get_template("chat").map_err(|e| TemplateError::Render(e.to_string()))?;
        let mut out = Bounded { out: Vec::new(), limit };
        tpl.render_captured_to(Serde(&ctx), &mut out)
            .map_err(|_| TemplateError::Raised("decision rendered prompt exceeds byte limit or template failed".into()))?;
        String::from_utf8(out.out).map_err(|e| TemplateError::Render(e.to_string()))
    }
    pub fn new(source: &str) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .keep_trailing_newline(false)
                .build()
                .expect("valid syntax config"),
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_filter("tojson", |v: Value, kwargs: Kwargs| -> Result<Value, Error> {
            let opts = json_opts(&kwargs)?;
            let j: Json = serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
            let text = python_json_with(&j, &opts);
            if RENDER_EXCEEDED.with(|v| v.get()) {
                return Err(Error::new(ErrorKind::InvalidOperation, "rendered prompt byte limit exceeded"));
            }
            Ok(Value::from_safe_string(text))
        });
        env.add_template_owned("chat", source.to_string()).map_err(|e| TemplateError::Render(e.to_string()))?;
        Ok(Self { env, source: source.to_string() })
    }

    /// Render `messages` (OpenAI-style JSON) with the template. `kwargs` become top-level template variables
    /// (`tools`, `enable_thinking`, `reasoning_effort`, `preserve_thinking`, ...).
    /// The top-level variables the template reads from its context: minijinja's scan of the parsed template, plus a lexical scan for what
    /// that scan misses (a variable read in its own top-level reassignment, `{% set x = x if x is defined else ... %}`, which the parser
    /// counts as declared). Used to check a family's reasoning capabilities against the template (`reasoning_caps::derive`).
    pub fn variables(&self) -> std::collections::BTreeSet<String> {
        let mut out: std::collections::BTreeSet<String> =
            self.env.get_template("chat").map(|t| t.undeclared_variables(false).into_iter().collect()).unwrap_or_default();
        out.extend(lexical_reads(&self.source));
        out
    }

    /// The template's source text.
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn render(
        &self,
        messages: &Json,
        add_generation_prompt: bool,
        kwargs: &Map<String, Json>,
    ) -> Result<String, TemplateError> {
        let mut ctx = Map::new();
        for (k, v) in kwargs {
            ctx.insert(k.clone(), v.clone());
        }
        ctx.insert("messages".into(), messages.clone());
        ctx.insert("add_generation_prompt".into(), Json::Bool(add_generation_prompt));
        let tpl = self.env.get_template("chat").map_err(|e| TemplateError::Render(e.to_string()))?;
        let result = if let Some(limit) = current_render_limit() {
            let mut writer = crate::budget::BoundedWriter::new(limit);
            let result = tpl.render_captured_to(Serde(&ctx), &mut writer);
            if writer.exceeded || RENDER_EXCEEDED.with(|v| v.get()) {
                return Err(TemplateError::Limit);
            }
            result.map(|_| String::from_utf8(writer.bytes).expect("Jinja output is UTF-8"))
        } else {
            tpl.render(Serde(&ctx))
        };
        result.map_err(|e| match e.kind() {
            ErrorKind::InvalidOperation if e.detail().is_some() && !format!("{e}").contains("unknown") => {
                TemplateError::Raised(e.detail().unwrap_or_default().to_string())
            }
            _ => TemplateError::Render(format!("{e:#}")),
        })
    }
}

/// Identifiers a Jinja template reads before (or without) assigning them, by a lexical scan of its `{{ }}` and `{% %}` tags: string
/// literals, attributes (`.x`), filters (`|x`), tests (`is x`) and keyword arguments (`f(x=1)`) are not reads; `set` and `for` targets are
/// assignments that take effect at the end of their tag (so the right-hand side of `set x = x ...` is a read of the context's `x`).
pub fn lexical_reads(src: &str) -> std::collections::BTreeSet<String> {
    const KEYWORDS: &[&str] = &[
        "and",
        "or",
        "not",
        "in",
        "is",
        "if",
        "else",
        "elif",
        "endif",
        "for",
        "endfor",
        "set",
        "endset",
        "macro",
        "endmacro",
        "call",
        "endcall",
        "filter",
        "endfilter",
        "true",
        "false",
        "none",
        "True",
        "False",
        "None",
        "with",
        "endwith",
        "block",
        "endblock",
        "extends",
        "include",
        "import",
        "from",
        "as",
        "recursive",
        "break",
        "continue",
        "do",
        "raw",
        "endraw",
        "generation",
        "endgeneration",
    ];
    let mut assigned: std::collections::HashMap<String, usize> = Default::default();
    let mut reads: Vec<(String, usize)> = Vec::new();
    let b = src.as_bytes();
    let mut i = 0;
    while let Some(off) = src[i..].find('{') {
        let start = i + off;
        let kind = b.get(start + 1).copied();
        let close = match kind {
            Some(b'{') => "}}",
            Some(b'%') => "%}",
            Some(b'#') => "#}",
            _ => {
                i = start + 1;
                continue;
            }
        };
        let Some(len) = src[start + 2..].find(close) else { break };
        let end = start + 2 + len;
        let tag_end = end + 2;
        i = tag_end;
        if kind == Some(b'#') {
            continue;
        }
        // tokens: identifiers with their position, and single punctuation characters; string literals dropped
        let code = &src[start + 2..end];
        let cb = code.as_bytes();
        let mut toks: Vec<(String, usize)> = Vec::new();
        let mut j = 0;
        while j < cb.len() {
            let c = cb[j];
            if c == b'\'' || c == b'"' {
                j += 1;
                while j < cb.len() && cb[j] != c {
                    j += if cb[j] == b'\\' { 2 } else { 1 };
                }
                toks.push(("\"".into(), start + 2 + j));
                j += 1;
            } else if c.is_ascii_alphabetic() || c == b'_' {
                let k = j;
                while j < cb.len() && (cb[j].is_ascii_alphanumeric() || cb[j] == b'_') {
                    j += 1;
                }
                toks.push((code[k..j].to_string(), start + 2 + k));
            } else if c.is_ascii_whitespace()
                || c == b'-' && (j == 0 || j + 1 == cb.len())
                || c == b'+' && (j == 0 || j + 1 == cb.len())
            {
                j += 1;
            } else if c.is_ascii_digit() {
                while j < cb.len() && (cb[j].is_ascii_alphanumeric() || cb[j] == b'.') {
                    j += 1;
                }
            } else {
                let two = code.get(j..j + 2).unwrap_or("");
                if ["==", "!=", "<=", ">="].contains(&two) {
                    toks.push((two.to_string(), start + 2 + j));
                    j += 2;
                } else {
                    toks.push(((c as char).to_string(), start + 2 + j));
                    j += 1;
                }
            }
        }
        let is_ident = |t: &str| t.as_bytes().first().is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_');
        let mut first = 0;
        if kind == Some(b'%') {
            match toks.first().map(|t| t.0.as_str()) {
                Some("set") => {
                    let eq = toks.iter().position(|t| t.0 == "=").unwrap_or(toks.len());
                    for t in &toks[1..eq] {
                        if is_ident(&t.0) {
                            assigned.entry(t.0.clone()).or_insert(tag_end);
                        }
                    }
                    first = eq;
                }
                Some("for") => {
                    let inn = toks.iter().position(|t| t.0 == "in").unwrap_or(toks.len());
                    for t in &toks[1..inn] {
                        if is_ident(&t.0) {
                            assigned.entry(t.0.clone()).or_insert(tag_end);
                        }
                    }
                    first = inn;
                }
                Some("macro") => first = toks.len(),
                _ => {}
            }
        }
        for n in first..toks.len() {
            let (t, pos) = &toks[n];
            if !is_ident(t) || KEYWORDS.contains(&t.as_str()) {
                continue;
            }
            let prev = n.checked_sub(1).map(|p| toks[p].0.as_str());
            let prev2 = n.checked_sub(2).map(|p| toks[p].0.as_str());
            let next = toks.get(n + 1).map(|t| t.0.as_str());
            let attribute_filter_or_test = matches!(prev, Some("." | "|" | "is")) || (prev == Some("not") && prev2 == Some("is"));
            let keyword_argument = next == Some("=") && matches!(prev, Some("(" | ","));
            if !attribute_filter_or_test && !keyword_argument {
                reads.push((t.clone(), *pos));
            }
        }
    }
    reads.into_iter().filter(|(n, pos)| assigned.get(n).is_none_or(|a| pos < a)).map(|(n, _)| n).collect()
}
