mod common;
use base64::Engine;
use common::{start, Opts};
use gb_model::{
    error::ModelError,
    serve::{EncodedImage, ImageEncoder, VisionService},
};
use gb_server::{
    http::VisionHandle,
    preparation::{PreparationGate, PreparationLimits},
    vision_ingress::{HttpsIngress, IngressPolicy},
};
use serde_json::{json, Value};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};
fn post(port: u16, path: &str, body: &Value) -> common::Reply {
    common::http_timeout(port, "POST", path, Some(body), Duration::from_secs(10))
}
// Only the CPU fixture's token ids differ from the real pack. Keep the actual Qwen template/media behavior.
struct FixtureFormat(Box<dyn gb_server::chat_format::ChatFormat>);
impl gb_server::chat_format::ChatFormat for FixtureFormat {
    fn id(&self) -> &'static str {
        self.0.id()
    }
    fn render(
        &self,
        m: &Value,
        k: &serde_json::Map<String, Value>,
    ) -> Result<gb_server::chat_format::RenderedChat, gb_server::template::TemplateError> {
        self.0.render(m, k)
    }
    fn render_bounded(
        &self,
        m: &Value,
        k: &serde_json::Map<String, Value>,
        max_bytes: usize,
    ) -> Result<gb_server::chat_format::RenderedChat, gb_server::template::TemplateError> {
        self.0.render_bounded(m, k, max_bytes)
    }
    fn starts_in_reasoning(&self, p: &str) -> bool {
        self.0.starts_in_reasoning(p)
    }
    fn tool_syntax(&self) -> gb_server::tools::ToolSyntax {
        self.0.tool_syntax()
    }
    fn check_media(&self, m: &Value, allow: bool) -> Result<(), gb_server::api::ApiError> {
        self.0.check_media(m, allow)
    }
    fn is_image_part(&self, p: &Value) -> bool {
        self.0.is_image_part(p)
    }
    fn image_placeholder(&self, t: &gb_server::tokenizer::Tok) -> Result<(u32, &'static str), String> {
        t.token_to_id("<|image_pad|>").map(|id| (id, "<|image_pad|>")).ok_or_else(|| "fixture missing image pad".into())
    }
    fn serves_images(&self) -> bool {
        true
    }
}
struct Encoder {
    calls: Arc<AtomicUsize>,
    mode: &'static str,
    barrier: Option<Arc<std::sync::atomic::AtomicBool>>,
}
impl ImageEncoder for Encoder {
    fn encode(&mut self, pre: &gb_vision::Preprocessed) -> Result<EncodedImage, ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(b) = &self.barrier {
            let until = std::time::Instant::now() + Duration::from_secs(5);
            while !b.load(Ordering::SeqCst) && std::time::Instant::now() < until {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        match self.mode {
            "panic" => panic!("mock encoder panic"),
            "busy" => return Err(ModelError::Busy("mock capacity".into())),
            "error" => return Err(ModelError::Request("mock bad image".into())),
            _ => {}
        }
        Ok(EncodedImage { tokens: pre.n_patches() / 4, merged: (pre.grid.1 / 2, pre.grid.2 / 2), features: Arc::new(()) })
    }
}
fn setup(
    mode: &'static str,
    deadline: Duration,
    barrier: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> (common::Server, Arc<AtomicUsize>, Arc<PreparationGate>) {
    setup_in(&common::fixture_dir(), mode, deadline, barrier)
}
/// [`setup`] on the model folder `dir` (its chat format and tokenizer): the Qwen image-flow cases run on `common::qwen_fixture_dir()`,
/// whose template expands an image into `<|image_pad|>` rows, as they did before `fixture_dir` moved to the Nemotron template.
fn setup_in(
    dir: &std::path::Path,
    mode: &'static str,
    deadline: Duration,
    barrier: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> (common::Server, Arc<AtomicUsize>, Arc<PreparationGate>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let slot = Arc::new(OnceLock::new());
    slot.set(Arc::new(VisionService::new(Box::new(Encoder { calls: calls.clone(), mode, barrier }), 4096))).unwrap();
    let gate = PreparationGate::new(PreparationLimits { deadline, ..Default::default() }).unwrap();
    let vision = VisionHandle {
        slot,
        limits: Default::default(),
        pp: Arc::new(gb_vision::Preprocessing::qwen38()),
        gate: gate.clone(),
        ingress: HttpsIngress::new(IngressPolicy::default()).unwrap(),
    };
    let format = Arc::new(FixtureFormat(gb_server::chat_format::FormatRegistry::builtin().open(dir, None).unwrap()));
    (start(Opts { vision: Some(vision), format: Some(format), dir: Some(dir.to_path_buf()), ..Default::default() }), calls, gate)
}
fn body(n: usize) -> Value {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/vision/check_512x512.png")).unwrap();
    let uri = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes));
    json!({"messages":[{"role":"user", "content": (0..n).map(|_| json!({"type":"image_url","image_url":{"url":uri}})).collect::<Vec<_>>()}],"max_tokens":2,"chat_template_kwargs":{"enable_thinking":false}})
}
#[test]
fn mock_encoder_positive_four_images_and_errors() {
    let (s, calls, gate) = setup_in(&common::qwen_fixture_dir(), "ok", Duration::from_secs(5), None);
    let r = post(s.port, "/v1/chat/completions", &body(4));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(gate.active_bytes(), 0);
    for (mode, status, ty) in
        [("panic", 500, "server_error"), ("busy", 503, "server_overloaded"), ("error", 400, "invalid_request_error")]
    {
        let (s, _, gate) = setup_in(&common::qwen_fixture_dir(), mode, Duration::from_secs(5), None);
        let r = post(s.port, "/v1/chat/completions", &body(1));
        assert_eq!(r.status, status, "{}", r.body);
        assert_eq!(r.json()["error"]["type"], ty);
        assert_eq!(gate.active_bytes(), 0);
    }
}
#[test]
fn deadline_does_not_release_blocking_worker_credits_or_encode_next_image() {
    let barrier = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (s, calls, gate) = setup("ok", Duration::from_millis(150), Some(barrier.clone()));
    let port = s.port;
    let t = std::thread::spawn(move || post(port, "/v1/chat/completions", &body(2)));
    s.wait(Duration::from_secs(3), "encoder entered", |_| calls.load(Ordering::SeqCst) == 1);
    let r = t.join().unwrap();
    assert_eq!(r.status, 504, "{}", r.body);
    assert!(gate.active_bytes() > 0, "spawn_blocking still holds decoded input/output");
    barrier.store(true, Ordering::SeqCst);
    s.wait(Duration::from_secs(2), "worker credit return", |_| gate.active_bytes() == 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(s.in_flight(), 0);
}
#[test]
fn real_tcp_drop_pre_response_is_observed_without_assuming_hyper_cancels() {
    use std::io::Write;
    let barrier = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (s, calls, gate) = setup("ok", Duration::from_secs(3), Some(barrier.clone()));
    let mut client = std::net::TcpStream::connect(("127.0.0.1", s.port)).unwrap();
    let bytes = body(2).to_string();
    write!(
        client,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        bytes.len(),
        bytes
    )
    .unwrap();
    s.wait(Duration::from_secs(3), "encoder entered", |_| calls.load(Ordering::SeqCst) == 1);
    client.shutdown(std::net::Shutdown::Both).unwrap();
    drop(client);
    std::thread::sleep(Duration::from_millis(100));
    assert!(gate.active_bytes() > 0);
    let timed_out = gate.timed_out.load(Ordering::SeqCst);
    assert_eq!(timed_out, 0, "the 3-second deadline has not expired during this transport observation");
    barrier.store(true, Ordering::SeqCst);
    s.wait(Duration::from_secs(2), "dropped-client cleanup", |_| gate.active_bytes() == 0);
    let cancelled = gate.cancelled.load(Ordering::SeqCst);
    eprintln!("real TCP pre-response drop: deadline_notifications={timed_out}, cooperatively_cancelled_workers={cancelled}; pinned Hyper notified before deadline");
    assert_eq!(cancelled, 1, "the worker must observe the actual disconnect flag, not merely an expired deadline");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(s.in_flight(), 0);
}
#[test]
fn flag_off_and_unsafe_ingress_never_reach_encoder() {
    let (s, calls, _) = setup("ok", Duration::from_secs(5), None);
    for url in ["https://example.com/a", "http://127.0.0.1/a", "file:///a", "gopher://127.0.0.1/a"] {
        let mut b = body(1);
        b["messages"][0]["content"][0]["image_url"]["url"] = json!(url);
        let r = post(s.port, "/v1/chat/completions", &b);
        assert_eq!(r.status, 400, "{}", r.body);
        assert!(r.body.contains("messages[0].content[0]"));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn four_complete_valid_ten_mib_pngs_remain_accepted() {
    let (s, calls, gate) = setup_in(&common::qwen_fixture_dir(), "ok", Duration::from_secs(60), None);
    let original = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/vision/check_512x512.png")).unwrap();
    let cap = 10usize << 20;
    let payload_len = cap - original.len() - 12;
    let mut png = original[..original.len() - 12].to_vec();
    png.extend_from_slice(&(payload_len as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(b"vpAg"); // valid private ancillary chunk, safely ignored by the decoder
    png.resize(png.len() + payload_len, 0);
    let crc = crc32fast::hash(&png[start..]);
    png.extend_from_slice(&crc.to_be_bytes());
    png.extend_from_slice(&original[original.len() - 12..]);
    assert_eq!(png.len(), cap);
    let uri = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(png));
    let mut b = body(4);
    for part in b["messages"][0]["content"].as_array_mut().unwrap() {
        part["image_url"]["url"] = json!(uri);
    }
    let r = common::http_timeout(s.port, "POST", "/v1/chat/completions", Some(&b), Duration::from_secs(90));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(gate.retained_bytes(), 0);
}
#[test]
fn an_aggregate_overrun_is_413_and_names_the_image() {
    let calls = Arc::new(AtomicUsize::new(0));
    let slot = Arc::new(OnceLock::new());
    slot.set(Arc::new(VisionService::new(Box::new(Encoder { calls: calls.clone(), mode: "ok", barrier: None }), 4096))).unwrap();
    let png = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/vision/check_512x512.png")).unwrap();
    let limits = gb_server::vision::Limits { max_total_bytes: png.len(), ..Default::default() };
    let gate = PreparationGate::new(Default::default()).unwrap();
    let vision = VisionHandle {
        slot,
        limits,
        pp: Arc::new(gb_vision::Preprocessing::qwen38()),
        gate: gate.clone(),
        ingress: HttpsIngress::new(Default::default()).unwrap(),
    };
    let format =
        Arc::new(FixtureFormat(gb_server::chat_format::FormatRegistry::builtin().open(&common::fixture_dir(), None).unwrap()));
    let s = start(Opts { vision: Some(vision), format: Some(format), ..Default::default() });
    let r = post(s.port, "/v1/chat/completions", &body(2));
    assert_eq!(r.status, 413, "{}", r.body);
    assert!(r.body.contains("messages[0].content[1]"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.retained_bytes(), 0);
}
#[test]
fn image_metrics_are_present_and_low_cardinality() {
    let (s, _, _) = setup_in(&common::qwen_fixture_dir(), "ok", Duration::from_secs(5), None);
    assert_eq!(post(s.port, "/v1/chat/completions", &body(1)).status, 200);
    let (status, metrics) = common::http(s.port, "GET", "/metrics", None);
    assert_eq!(status, 200);
    for metric in [
        "rexl3_image_retained_bytes 0",
        "rexl3_images_processed_total 1",
        "rexl3_image_preparation_bytes 0",
        "rexl3_image_fetch_total{result=\"denied\"} 0",
    ] {
        assert!(metrics.contains(metric), "{metric}: {metrics}");
    }
}

#[test]
fn sixty_tcp_drops_during_image_work_balance_backlog_and_byte_credits() {
    use std::io::Write;
    let barrier = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (s, calls, gate) = setup("ok", Duration::from_secs(3), Some(barrier.clone()));
    let request = body(2).to_string();
    let send = || {
        let mut c = std::net::TcpStream::connect(("127.0.0.1", s.port)).unwrap();
        write!(c,"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",request.len(),request).unwrap();
        c
    };
    let c = send();
    s.wait(Duration::from_secs(2), "first encoder parked", |_| calls.load(Ordering::SeqCst) == 1);
    for _ in 0..60 {
        let c = send();
        c.shutdown(std::net::Shutdown::Both).unwrap();
    }
    assert!(gate.active_bytes() <= gate.limits.bytes as u64);
    assert!(gate.retained_bytes() <= gate.limits.retained_bytes);
    assert!(gate.jobs() <= gate.limits.backlog);
    c.shutdown(std::net::Shutdown::Both).unwrap();
    drop(c);
    // Allow the transport to notify handlers before releasing the running safe boundary.
    std::thread::sleep(Duration::from_millis(100));
    barrier.store(true, Ordering::SeqCst);
    s.wait(Duration::from_secs(4), "all dropped jobs returned credits", |_| gate.jobs() == 0);
    assert_eq!(gate.active_bytes(), 0);
    assert_eq!(gate.retained_bytes(), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(s.in_flight(), 0);
}

#[derive(Debug, Default)]
struct StagePark {
    entered: AtomicUsize,
    release: std::sync::atomic::AtomicBool,
}
impl StagePark {
    fn park(&self) {
        self.entered.fetch_add(1, Ordering::SeqCst);
        let until = std::time::Instant::now() + Duration::from_secs(5);
        while !self.release.load(Ordering::SeqCst) && std::time::Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
struct ParkedFormat {
    inner: FixtureFormat,
    park: Arc<StagePark>,
}
impl gb_server::chat_format::ChatFormat for ParkedFormat {
    fn id(&self) -> &'static str {
        self.inner.id()
    }
    fn render(
        &self,
        m: &Value,
        k: &serde_json::Map<String, Value>,
    ) -> Result<gb_server::chat_format::RenderedChat, gb_server::template::TemplateError> {
        self.park.park();
        self.inner.render(m, k)
    }
    fn render_bounded(
        &self,
        m: &Value,
        k: &serde_json::Map<String, Value>,
        max_bytes: usize,
    ) -> Result<gb_server::chat_format::RenderedChat, gb_server::template::TemplateError> {
        self.park.park();
        self.inner.render_bounded(m, k, max_bytes)
    }
    fn starts_in_reasoning(&self, p: &str) -> bool {
        self.inner.starts_in_reasoning(p)
    }
    fn tool_syntax(&self) -> gb_server::tools::ToolSyntax {
        self.inner.tool_syntax()
    }
    fn check_media(&self, m: &Value, allow: bool) -> Result<(), gb_server::api::ApiError> {
        self.inner.check_media(m, allow)
    }
    fn is_image_part(&self, p: &Value) -> bool {
        self.inner.is_image_part(p)
    }
    fn image_placeholder(&self, t: &gb_server::tokenizer::Tok) -> Result<(u32, &'static str), String> {
        self.inner.image_placeholder(t)
    }
    fn serves_images(&self) -> bool {
        true
    }
}
#[derive(Debug)]
struct ParkedDecode {
    inner: gb_vision::Preprocessing,
    park: Arc<StagePark>,
}
impl gb_vision::preprocessor::ImagePreprocessor for ParkedDecode {
    fn family(&self) -> &'static str {
        self.inner.family()
    }
    fn id(&self) -> String {
        self.inner.id()
    }
    fn decode(&self, b: &[u8]) -> Result<gb_vision::Rgb, gb_vision::VisionError> {
        self.park.park();
        self.inner.decode(b)
    }
    fn plan(&self, w: usize, h: usize) -> Result<gb_vision::preprocessor::ImagePlan, gb_vision::VisionError> {
        self.inner.plan(w, h)
    }
    fn preprocess(&self, img: &gb_vision::Rgb) -> Result<gb_vision::Preprocessed, gb_vision::VisionError> {
        self.inner.preprocess(img)
    }
}
#[test]
fn real_tcp_disconnect_notifies_render_and_decode_workers_before_deadline() {
    use gb_server::chat_format::ChatFormat;
    use std::io::Write;
    for stage in ["render", "decode"] {
        let park = Arc::new(StagePark::default());
        let calls = Arc::new(AtomicUsize::new(0));
        let slot = Arc::new(OnceLock::new());
        slot.set(Arc::new(VisionService::new(Box::new(Encoder { calls: calls.clone(), mode: "ok", barrier: None }), 4096)))
            .unwrap();
        let gate = PreparationGate::new(PreparationLimits { deadline: Duration::from_secs(3), ..Default::default() }).unwrap();
        let pp: Arc<dyn gb_vision::preprocessor::ImagePreprocessor> = if stage == "decode" {
            Arc::new(ParkedDecode { inner: gb_vision::Preprocessing::qwen38(), park: park.clone() })
        } else {
            Arc::new(gb_vision::Preprocessing::qwen38())
        };
        let inner = FixtureFormat(gb_server::chat_format::FormatRegistry::builtin().open(&common::fixture_dir(), None).unwrap());
        let format: Arc<dyn ChatFormat> =
            if stage == "render" { Arc::new(ParkedFormat { inner, park: park.clone() }) } else { Arc::new(inner) };
        let vision = VisionHandle { slot, pp, gate: gate.clone(), ..Default::default() };
        let s = start(Opts { vision: Some(vision), format: Some(format), ..Default::default() });
        let mut client = std::net::TcpStream::connect(("127.0.0.1", s.port)).unwrap();
        let request = body(2).to_string();
        write!(client,"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",request.len(),request).unwrap();
        s.wait(Duration::from_secs(2), "preparation stage parked", |_| park.entered.load(Ordering::SeqCst) == 1);
        client.shutdown(std::net::Shutdown::Both).unwrap();
        drop(client);
        std::thread::sleep(Duration::from_millis(100));
        assert!(gate.active_bytes() > 0, "{stage} worker must retain credits after HTTP future drop");
        park.release.store(true, Ordering::SeqCst);
        s.wait(Duration::from_secs(2), "cancelled preparation worker released", |_| gate.jobs() == 0);
        assert_eq!(gate.cancelled.load(Ordering::SeqCst), 1, "{stage} must see actual cancellation before deadline");
        assert_eq!(gate.timed_out.load(Ordering::SeqCst), 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no encoder launch after {stage} disconnect");
        assert_eq!(s.in_flight(), 0);
        eprintln!(
            "real TCP {stage}: cancelled_workers=1 deadline_notifications=0 encoder_calls=0 bytes_after_drop={}",
            gate.active_bytes()
        );
    }
}
