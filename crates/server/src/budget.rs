//! Admission before retention or unrestricted JSON allocation. Limits are active by default.
use crate::api::ApiError;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// What the tokenizer stage holds in resident memory, per byte of the rendered text it is given. Measured on the real Qwen tokenizer
/// (`docs/REQUEST_CONTRACT.md`, "Request memory"): 371 to 416 bytes of RSS per rendered byte for prompts of one token per byte (ASCII digits,
/// digits between whitespace; the high end is the 3.99 MiB prompt, where the fixed part of a job weighs more), against 116 for lorem-ipsum
/// text. The figure is the worst measured plus 8 %, a multiple of 32. The processing reserve used to price this stage at 128.
pub const TOKENIZER_BYTES_PER_RENDERED_BYTE: usize = 448;
/// A rendered prompt may have this many bytes per token of the context window (`--max-rendered-prompt-mib` default): 4 MiB for a
/// 262,144-token window. Natural text is 3 to 5 bytes a token, so this is three to five times the longest prompt a window admits.
pub const RENDERED_BYTES_PER_WINDOW_TOKEN: usize = 16;
/// The smallest default rendered-prompt cap, whatever the window.
pub const RENDERED_FLOOR_BYTES: usize = 1 << 20;
/// The context window the `Default` limits are derived for.
pub const DEFAULT_WINDOW_TOKENS: usize = 262_144;
/// The default processing reserve, unless one prompt at the rendered cap needs more (see [`Limits::derived_reserve`]).
pub const DEFAULT_PROCESSING_RESERVE_BYTES: usize = 2 << 30;
/// How long a tokenizer call may wait for the reserve before it is answered 429.
pub const DEFAULT_TOKENIZER_WAIT: Duration = Duration::from_secs(60);
/// A decision label (at most 32 bytes) is appended to a prompt that is at the cap before the pair is encoded again.
const TOKENIZER_SLACK_BYTES: usize = 64;

/// The default rendered-prompt cap for a context window of `window_tokens`: 16 bytes a token, at least 1 MiB.
pub fn default_rendered_bytes(window_tokens: usize) -> usize {
    window_tokens.saturating_mul(RENDERED_BYTES_PER_WINDOW_TOKEN).max(RENDERED_FLOOR_BYTES)
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub body_bytes: usize,
    pub total_bytes: usize,
    pub json_nodes: usize,
    pub json_depth: usize,
    pub rendered_bytes: usize,
    pub preparation_concurrency: usize,
    /// Bounded async waiters; zero means fail-fast.
    pub preparation_queue: usize,
    /// The memory the tokenizer stage may price at once (see [`TokenizerGate`]) and the startup check of the parse stage.
    pub processing_reserve_bytes: usize,
    /// How long a tokenizer call waits for the reserve before it is answered 429 (not a flag).
    pub tokenizer_wait: Duration,
}
impl Default for Limits {
    fn default() -> Self {
        Self::for_window(DEFAULT_WINDOW_TOKENS)
    }
}
impl Limits {
    /// The defaults for a context window of `window_tokens`: the rendered cap follows the window ([`default_rendered_bytes`]) and the
    /// reserve is the default unless one prompt at that cap needs more.
    pub fn for_window(window_tokens: usize) -> Self {
        let mut l = Self {
            body_bytes: 64 << 20,
            total_bytes: 128 << 20,
            json_nodes: 300_000,
            json_depth: 128,
            rendered_bytes: default_rendered_bytes(window_tokens),
            preparation_concurrency: 2,
            preparation_queue: 64,
            processing_reserve_bytes: DEFAULT_PROCESSING_RESERVE_BYTES,
            tokenizer_wait: DEFAULT_TOKENIZER_WAIT,
        };
        l.processing_reserve_bytes = l.derived_reserve();
        l
    }
    /// What the tokenizer stage prices the largest prompt these limits admit at: the cap plus the slack of a decision label.
    pub fn tokenizer_price(&self) -> Result<usize, String> {
        self.rendered_bytes
            .checked_add(TOKENIZER_SLACK_BYTES)
            .and_then(|n| n.checked_mul(TOKENIZER_BYTES_PER_RENDERED_BYTE))
            .ok_or_else(|| "render preparation reserve arithmetic overflow".to_string())
    }
    /// The processing reserve a server started without `--processing-reserve-mib` gets: the default, or the price of one prompt at the
    /// rendered cap when that is more (a window past about 300K tokens), so that the largest admissible prompt can always be tokenized.
    pub fn derived_reserve(&self) -> usize {
        self.tokenizer_price().map_or(usize::MAX, |p| p.max(DEFAULT_PROCESSING_RESERVE_BYTES))
    }
    pub fn validate(&self) -> Result<(), String> {
        if [
            self.body_bytes,
            self.total_bytes,
            self.json_nodes,
            self.json_depth,
            self.rendered_bytes,
            self.preparation_concurrency,
        ]
        .contains(&0)
        {
            return Err("request limits must be positive".into());
        }
        if self.json_depth > 128 {
            return Err("max-json-depth must not exceed 128".into());
        }
        // Bound the configured raw/Value expansion as well as rendered/tokenizer expansion. These are conservative logical
        // reservations, not an allocator-level RSS guarantee; changing any input ceiling must not silently weaken the reserve.
        // The parse stage is bounded by count: `preparation_concurrency` jobs, each at most the larger of its raw and node price. The
        // tokenizer stage is bounded by price (`TokenizerGate`): jobs wait until their priced bytes fit, so the reserve has to hold the
        // largest single prompt, not `preparation_concurrency` of them.
        let raw = self.body_bytes.min(self.total_bytes).checked_mul(16).ok_or("raw preparation reserve arithmetic overflow")?;
        let nodes = self.json_nodes.checked_mul(128).ok_or("JSON preparation reserve arithmetic overflow")?;
        let parse = raw.max(nodes).checked_mul(self.preparation_concurrency).ok_or("processing reserve arithmetic overflow")?;
        let tokenize = self.tokenizer_price()?;
        let required = parse.max(tokenize);
        if self.processing_reserve_bytes < required {
            return Err(format!(
                "processing reserve must be at least {required} bytes for these body/node/render/concurrency limits \
                 (the largest rendered prompt is priced at {tokenize} bytes: {TOKENIZER_BYTES_PER_RENDERED_BYTE} per rendered byte)"
            ));
        }
        Ok(())
    }
}

/// The tokenizer stage's share of the processing reserve. Tokenizing a prompt costs [`TOKENIZER_BYTES_PER_RENDERED_BYTE`] bytes of memory per
/// byte of its text, far more than the prompt itself and more for one token per byte than for prose, so the number of prompts that may be
/// tokenized at once has to follow their size and not a worker count: a call takes its priced bytes from this gate before it allocates
/// and gives them back when the encoding has been copied out. A prompt that does not fit waits (first come, first served: a large one
/// is not starved by small ones); one that could never fit is refused 413; one that waited past `wait` is refused 429.
pub struct TokenizerGate {
    capacity: usize,
    wait: Duration,
    state: Mutex<GateState>,
    freed: Condvar,
    waited: AtomicU64,
    refused: AtomicU64,
}
struct GateState {
    free: usize,
    queue: VecDeque<u64>,
    next: u64,
}
/// Priced bytes held by one tokenizer call; given back on drop.
pub struct TokenizerCharge {
    gate: Arc<TokenizerGate>,
    bytes: usize,
}
impl TokenizerGate {
    pub fn new(capacity: usize, wait: Duration) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            wait,
            state: Mutex::new(GateState { free: capacity, queue: VecDeque::new(), next: 0 }),
            freed: Condvar::new(),
            waited: AtomicU64::new(0),
            refused: AtomicU64::new(0),
        })
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Priced bytes held right now.
    pub fn in_use(&self) -> usize {
        self.capacity - self.state.lock().unwrap_or_else(|e| e.into_inner()).free
    }
    /// Calls that had to wait, and calls refused (413 or 429), since the gate was made.
    pub fn counts(&self) -> (u64, u64) {
        (self.waited.load(Ordering::Relaxed), self.refused.load(Ordering::Relaxed))
    }
    /// The price of tokenizing `text_bytes` bytes, or `None` on overflow.
    pub fn price(text_bytes: usize) -> Option<usize> {
        text_bytes.checked_mul(TOKENIZER_BYTES_PER_RENDERED_BYTE)
    }
    /// Take the priced bytes of a text of `text_bytes` bytes, waiting while they are not free. 413 when the price is more than the whole
    /// reserve (it can never fit), 429 with `Retry-After` when the wait passed `wait`.
    pub fn charge(self: &Arc<Self>, text_bytes: usize) -> Result<TokenizerCharge, ApiError> {
        let never = |price: Option<usize>| {
            self.refused.fetch_add(1, Ordering::Relaxed);
            too_large(match price {
                Some(p) => format!(
                    "the rendered prompt of {text_bytes} bytes is priced at {p} bytes of tokenizer memory \
                     ({TOKENIZER_BYTES_PER_RENDERED_BYTE} per byte), more than the processing reserve of {} bytes \
                     (--processing-reserve-mib)",
                    self.capacity
                ),
                None => "the rendered prompt size overflows the tokenizer memory price".to_string(),
            })
        };
        let Some(price) = Self::price(text_bytes) else { return Err(never(None)) };
        if price > self.capacity {
            return Err(never(Some(price)));
        }
        let give_up = Instant::now() + self.wait;
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.queue.is_empty() && st.free >= price {
            st.free -= price;
            return Ok(TokenizerCharge { gate: self.clone(), bytes: price });
        }
        self.waited.fetch_add(1, Ordering::Relaxed);
        let ticket = st.next;
        st.next += 1;
        st.queue.push_back(ticket);
        loop {
            if st.queue.front() == Some(&ticket) && st.free >= price {
                st.queue.pop_front();
                st.free -= price;
                if !st.queue.is_empty() {
                    self.freed.notify_all();
                }
                return Ok(TokenizerCharge { gate: self.clone(), bytes: price });
            }
            let left = give_up.saturating_duration_since(Instant::now());
            if left.is_zero() {
                st.queue.retain(|t| *t != ticket);
                self.freed.notify_all();
                self.refused.fetch_add(1, Ordering::Relaxed);
                return Err(overloaded("the tokenizer's share of the processing reserve is in use; retry shortly"));
            }
            st = self.freed.wait_timeout(st, left).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
}
impl Drop for TokenizerCharge {
    fn drop(&mut self) {
        self.gate.state.lock().unwrap_or_else(|e| e.into_inner()).free += self.bytes;
        self.gate.freed.notify_all();
    }
}

pub struct RequestBudget {
    pub limits: Limits,
    used: AtomicUsize,
    preparation: Arc<Semaphore>,
    waiters: Arc<Semaphore>,
    decisions: Arc<Semaphore>,
    tokenizer: Arc<TokenizerGate>,
}
impl RequestBudget {
    pub fn new(limits: Limits) -> Result<Arc<Self>, String> {
        limits.validate()?;
        if limits.preparation_queue > Semaphore::MAX_PERMITS || limits.preparation_concurrency > Semaphore::MAX_PERMITS {
            return Err("preparation semaphore limit is too large".into());
        }
        Ok(Arc::new(Self {
            waiters: Arc::new(Semaphore::new(limits.preparation_queue)),
            decisions: Arc::new(Semaphore::new(4)),
            preparation: Arc::new(Semaphore::new(limits.preparation_concurrency)),
            used: AtomicUsize::new(0),
            tokenizer: TokenizerGate::new(limits.processing_reserve_bytes, limits.tokenizer_wait),
            limits,
        }))
    }
    /// The tokenizer stage's gate: `http::router` hands it to the server's `Tok`, which charges every encode against it.
    pub fn tokenizer_gate(&self) -> &Arc<TokenizerGate> {
        &self.tokenizer
    }
    /// Keep prepared prompts/full score rows bounded independently of the shared CPU worker gate.
    pub fn decision_admitted(&self) -> Result<OwnedSemaphorePermit, ApiError> {
        self.decisions.clone().try_acquire_owned().map_err(|_| overloaded("active decisions request capacity is exhausted"))
    }
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
    pub fn lease(self: &Arc<Self>) -> ByteLease {
        ByteLease { budget: self.clone(), bytes: 0 }
    }
    /// Wait asynchronously BEFORE Value allocation. Raw bytes and waiting job count are both bounded.
    /// Lifecycle owns the encompassing HTTP deadline; dropping this future releases its waiting place.
    pub async fn prepare_admitted(&self) -> Result<OwnedSemaphorePermit, ApiError> {
        if let Ok(permit) = self.prepare() {
            return Ok(permit);
        }
        let _waiting = self.waiters.clone().try_acquire_owned().map_err(|_| overloaded("request preparation queue is full"))?;
        self.preparation.clone().acquire_owned().await.map_err(|_| overloaded("request preparation is unavailable"))
    }
    pub fn prepare(&self) -> Result<OwnedSemaphorePermit, ApiError> {
        self.preparation.clone().try_acquire_owned().map_err(|_| overloaded("request preparation capacity is exhausted"))
    }
}
impl Default for RequestBudget {
    fn default() -> Self {
        let limits = Limits::default();
        Self {
            used: AtomicUsize::new(0),
            preparation: Arc::new(Semaphore::new(2)),
            waiters: Arc::new(Semaphore::new(64)),
            decisions: Arc::new(Semaphore::new(4)),
            tokenizer: TokenizerGate::new(limits.processing_reserve_bytes, limits.tokenizer_wait),
            limits,
        }
    }
}
pub struct ByteLease {
    budget: Arc<RequestBudget>,
    bytes: usize,
}
impl ByteLease {
    /// Charge capacity before reserving/retaining bytes. Never truncate a reservation to remaining capacity.
    pub fn grow_to(&mut self, bytes: usize) -> Result<(), ApiError> {
        if bytes <= self.bytes {
            return Ok(());
        }
        let extra = bytes.checked_sub(self.bytes).ok_or_else(|| too_large("request byte arithmetic overflow"))?;
        self.budget
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(extra).filter(|n| *n <= self.budget.limits.total_bytes)
            })
            .map_err(|_| overloaded("raw request byte budget is exhausted"))?;
        self.bytes = bytes;
        Ok(())
    }
}
impl Drop for ByteLease {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
pub struct RequestLease {
    pub raw: ByteLease,
    preparation: std::sync::Mutex<Option<OwnedSemaphorePermit>>,
}
impl RequestLease {
    pub fn new(raw: ByteLease, preparation: OwnedSemaphorePermit) -> Self {
        Self { raw, preparation: std::sync::Mutex::new(Some(preparation)) }
    }
    /// Only call after the actual preparation worker has joined. Raw/body credits remain owned
    /// by this lease; native score waits must not monopolize unrelated text preparation slots.
    pub fn finish_preparation(&self) {
        self.preparation.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
}
pub fn overloaded(message: &str) -> ApiError {
    ApiError::status(429, "server_overloaded", message).retry_after(2)
}
pub fn too_large(message: impl Into<String>) -> ApiError {
    ApiError::status(413, "invalid_request_error", message)
}

/// Allocation-free JSON grammar admission. Counts every value AND object key (including duplicate keys), not punctuation in strings.
/// This is a complexity gate; serde_json remains authoritative for Unicode surrogates, UTF-8 and number representation.
pub fn validate_json(bytes: &[u8], max_nodes: usize, max_depth: usize) -> Result<usize, ApiError> {
    if max_depth > 128 {
        return Err(too_large("JSON depth limit exceeds the supported ceiling"));
    }
    let mut s = Scanner { b: bytes, at: 0, nodes: 0, max_nodes, max_depth };
    s.value(0)?;
    s.ws();
    if s.at != bytes.len() {
        return Err(s.bad());
    }
    Ok(s.nodes)
}
struct Scanner<'a> {
    b: &'a [u8],
    at: usize,
    nodes: usize,
    max_nodes: usize,
    max_depth: usize,
}
impl Scanner<'_> {
    fn bad(&self) -> ApiError {
        ApiError::bad("the request body is not valid JSON")
    }
    fn ws(&mut self) {
        while self.b.get(self.at).is_some_and(|c| matches!(c, b' ' | b'\n' | b'\r' | b'\t')) {
            self.at += 1;
        }
    }
    fn node(&mut self) -> Result<(), ApiError> {
        self.nodes = self.nodes.checked_add(1).ok_or_else(|| too_large("JSON node arithmetic overflow"))?;
        if self.nodes > self.max_nodes {
            return Err(too_large(format!("the request has more than {} JSON nodes", self.max_nodes)));
        }
        Ok(())
    }
    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.b.get(self.at) == Some(&c) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn string(&mut self) -> Result<(), ApiError> {
        if !self.eat(b'"') {
            return Err(self.bad());
        }
        while let Some(&c) = self.b.get(self.at) {
            self.at += 1;
            match c {
                b'"' => return Ok(()),
                0..=31 => return Err(self.bad()),
                b'\\' => {
                    let c = *self.b.get(self.at).ok_or_else(|| self.bad())?;
                    self.at += 1;
                    match c {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
                        b'u' => {
                            for _ in 0..4 {
                                if !self.b.get(self.at).is_some_and(u8::is_ascii_hexdigit) {
                                    return Err(self.bad());
                                }
                                self.at += 1;
                            }
                        }
                        _ => return Err(self.bad()),
                    }
                }
                _ => {}
            }
        }
        Err(self.bad())
    }
    fn value(&mut self, depth: usize) -> Result<(), ApiError> {
        self.ws();
        self.node()?;
        match self.b.get(self.at).copied() {
            Some(b'"') => self.string(),
            Some(open @ (b'{' | b'[')) => {
                if depth >= self.max_depth {
                    return Err(too_large(format!("JSON nesting exceeds {} containers", self.max_depth)));
                }
                self.at += 1;
                let close = if open == b'{' { b'}' } else { b']' };
                if self.eat(close) {
                    return Ok(());
                }
                loop {
                    if open == b'{' {
                        self.node()?;
                        self.string()?;
                        if !self.eat(b':') {
                            return Err(self.bad());
                        }
                    }
                    self.value(depth + 1)?;
                    if self.eat(close) {
                        return Ok(());
                    }
                    if !self.eat(b',') {
                        return Err(self.bad());
                    }
                }
            }
            Some(b't') => self.literal(b"true"),
            Some(b'f') => self.literal(b"false"),
            Some(b'n') => self.literal(b"null"),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.bad()),
        }
    }
    fn literal(&mut self, lit: &[u8]) -> Result<(), ApiError> {
        if self.b.get(self.at..self.at + lit.len()) != Some(lit) {
            return Err(self.bad());
        }
        self.at += lit.len();
        Ok(())
    }
    fn digits(&mut self) -> bool {
        let start = self.at;
        while self.b.get(self.at).is_some_and(u8::is_ascii_digit) {
            self.at += 1;
        }
        self.at > start
    }
    fn number(&mut self) -> Result<(), ApiError> {
        if self.b.get(self.at) == Some(&b'-') {
            self.at += 1;
        }
        match self.b.get(self.at) {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.bad()),
        }
        if self.b.get(self.at) == Some(&b'.') {
            self.at += 1;
            if !self.digits() {
                return Err(self.bad());
            }
        }
        if matches!(self.b.get(self.at), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.b.get(self.at), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if !self.digits() {
                return Err(self.bad());
            }
        }
        Ok(())
    }
}

/// A writer which refuses BEFORE appending beyond the cap, usable for templates and serialized tool schemas.
pub struct BoundedWriter {
    pub bytes: Vec<u8>,
    pub limit: usize,
    pub exceeded: bool,
}
impl BoundedWriter {
    pub fn new(limit: usize) -> Self {
        Self { bytes: Vec::new(), limit, exceeded: false }
    }
}
impl std::io::Write for BoundedWriter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        if self.bytes.len().checked_add(b.len()).is_none_or(|n| n > self.limit) {
            self.exceeded = true;
            return Err(std::io::Error::other("rendered prompt byte limit exceeded"));
        }
        self.bytes.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
