//! Optional HTTPS image ingress. No client is created until literal/DNS targets are validated.
//! Each fetch has a fresh client, one validated address set, no fallback DNS, proxy, redirects, retries or decompression.
use crate::{api::ApiError, preparation::RequestWork};
use reqwest::{Client, Url};
use std::{
    net::{IpAddr, SocketAddr},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
const MAX_URL_BYTES: usize = 2048;
const MAX_DNS_ANSWERS: usize = 32;
#[derive(Clone, Copy, Debug)]
pub struct IngressPolicy {
    pub enabled: bool,
    pub max_bytes: usize,
    pub connect_timeout: Duration,
    pub total_timeout: Duration,
}
impl Default for IngressPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_bytes: 10 << 20,
            connect_timeout: Duration::from_secs(10),
            total_timeout: Duration::from_secs(30),
        }
    }
}
#[derive(Debug)]
pub struct HttpsIngress {
    pub policy: IngressPolicy,
    pub ok: AtomicU64,
    pub denied: AtomicU64,
    pub timeout: AtomicU64,
    pub oversize: AtomicU64,
    pub bad_type: AtomicU64,
}
impl HttpsIngress {
    pub fn new(policy: IngressPolicy) -> Result<Arc<Self>, String> {
        if policy.max_bytes == 0
            || policy.max_bytes > 10 << 20
            || policy.connect_timeout.is_zero()
            || policy.total_timeout.is_zero()
            || policy.connect_timeout > policy.total_timeout
            || policy.total_timeout > Duration::from_secs(1800)
        {
            return Err("invalid image URL limits".into());
        }
        Ok(Arc::new(Self {
            policy,
            ok: AtomicU64::new(0),
            denied: AtomicU64::new(0),
            timeout: AtomicU64::new(0),
            oversize: AtomicU64::new(0),
            bad_type: AtomicU64::new(0),
        }))
    }
    pub async fn fetch(&self, raw: &str, work: &RequestWork) -> Result<Vec<u8>, ApiError> {
        work.check()?;
        let result = if !self.policy.enabled {
            Err(denied("remote image fetch is disabled; --image-urls off"))
        } else {
            tokio::time::timeout(self.policy.total_timeout.min(work.remaining()), async {
                fetch_using(raw, self.policy, work, resolve_target, |url, addrs, policy| {
                    client_builder(url, addrs, policy).build().map_err(|_| denied("cannot initialize image URL client"))
                })
                .await
            })
            .await
            .unwrap_or_else(|_| Err(fetch_timeout()))
        };
        match &result {
            Ok(_) => {
                self.ok.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) if e.status == 504 => {
                self.timeout.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) if e.status == 413 => {
                self.oversize.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) if e.message.contains("content type") || e.message.contains("compression") => {
                self.bad_type.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.denied.fetch_add(1, Ordering::Relaxed);
            }
        }
        result
    }
}
// Private injection seam used by CPU canaries, never configurable by an HTTP caller/deployment.
async fn fetch_using<R, F, B>(
    raw: &str,
    policy: IngressPolicy,
    work: &RequestWork,
    resolve: R,
    build: B,
) -> Result<Vec<u8>, ApiError>
where
    R: FnOnce(String) -> F,
    F: std::future::Future<Output = Result<Vec<SocketAddr>, ApiError>>,
    B: FnOnce(&Url, &[SocketAddr], IngressPolicy) -> Result<Client, ApiError>,
{
    work.check()?;
    let url = validate_url(raw)?;
    let addrs = validate_addresses(resolve(url.host_str().unwrap().to_owned()).await?)?;
    work.check()?;
    let client = build(&url, &addrs, policy)?;
    fetch_response(client, url, policy.max_bytes, work).await
}
async fn resolve_target(host: String) -> Result<Vec<SocketAddr>, ApiError> {
    if let Ok(ip) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
        return Ok(vec![SocketAddr::new(ip, 443)]);
    }
    let job = BoundedDnsJob::new(host)?;
    tokio::task::spawn_blocking(move || job.resolve())
        .await
        .map_err(|_| ApiError::status(500, "server_error", "internal image DNS worker failed"))?
}
// OS getaddrinfo cannot be forcibly cancelled. Its own worker retains its slot even when the fetch deadline drops its future.
fn dns_slots() -> &'static Arc<tokio::sync::Semaphore> {
    static SLOTS: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> = std::sync::OnceLock::new();
    SLOTS.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(2)))
}
struct BoundedDnsJob {
    host: String,
    _permit: tokio::sync::OwnedSemaphorePermit,
}
impl BoundedDnsJob {
    fn new(host: String) -> Result<Self, ApiError> {
        Self::with_slots(host, dns_slots().clone())
    }
    fn with_slots(host: String, slots: Arc<tokio::sync::Semaphore>) -> Result<Self, ApiError> {
        let permit = slots
            .try_acquire_owned()
            .map_err(|_| ApiError::status(503, "server_overloaded", "image DNS capacity exhausted").retry_after(5))?;
        Ok(Self { host, _permit: permit })
    }
    fn resolve(&self) -> Result<Vec<SocketAddr>, ApiError> {
        use std::net::ToSocketAddrs;
        let answers = (self.host.as_str(), 443).to_socket_addrs().map_err(|_| denied("image URL DNS lookup failed"))?;
        Ok(answers.take(MAX_DNS_ANSWERS + 1).collect())
    }
}
fn denied(message: &str) -> ApiError {
    ApiError::bad(message)
}
fn fetch_timeout() -> ApiError {
    ApiError::status(504, "request_timeout", "image URL fetch timed out")
}
fn oversize() -> ApiError {
    ApiError::status(413, "invalid_request_error", "image URL exceeds encoded byte limit")
}
/// Conservative public-unicast allowlist. Reject all special-use IPv4 and non-global IPv6, including transition mechanisms.
pub fn is_public_unicast(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, _] = ip.octets();
            !(a == 0
                || a == 10
                || a == 127
                || a >= 224
                || (a == 100 && (64..=127).contains(&b))
                || (a == 169 && b == 254)
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && (b == 168 || (b == 0 && (c == 0 || c == 2)) || (b == 88 && c == 99)))
                || (a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100)))
                || (a == 203 && b == 0 && c == 113))
        }
        IpAddr::V6(ip) => {
            let s = ip.segments();
            // Only native global unicast 2000::/3, excluding IETF assignments, docs, 6to4 and new doc/SID ranges.
            (s[0] & 0xe000) == 0x2000
                && !(s[0] == 0x2001 && (s[1] < 0x0200 || s[1] == 0x0db8))
                && s[0] != 0x2002
                && !(s[0] == 0x3fff && (s[1] & 0xf000) == 0)
        }
    }
}
pub fn validate_url(raw: &str) -> Result<Url, ApiError> {
    if raw.len() > MAX_URL_BYTES || raw.chars().any(|c| c.is_control() || c == '\\') {
        return Err(denied("invalid or overlong image URL"));
    }
    let url = Url::parse(raw).map_err(|_| denied("invalid image URL"))?;
    if url.scheme() != "https"
        || url.port_or_known_default() != Some(443)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(denied("image URLs require HTTPS port 443, without credentials or fragments"));
    }
    let host = url.host_str().ok_or_else(|| denied("image URL requires a hostname"))?;
    if let Ok(ip) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
        if !is_public_unicast(ip) {
            return Err(denied("image URL target is not public unicast"));
        }
    }
    Ok(url)
}
pub fn validate_addresses(addrs: Vec<SocketAddr>) -> Result<Vec<SocketAddr>, ApiError> {
    if addrs.is_empty() || addrs.len() > MAX_DNS_ANSWERS || addrs.iter().any(|a| a.port() != 443 || !is_public_unicast(a.ip())) {
        return Err(denied("image URL DNS target is not public unicast"));
    }
    Ok(addrs)
}
#[derive(Debug)]
struct NoSecondaryDns;
impl reqwest::dns::Resolve for NoSecondaryDns {
    fn resolve(&self, _: reqwest::dns::Name) -> reqwest::dns::Resolving {
        Box::pin(async { Err(std::io::Error::other("secondary DNS forbidden").into()) })
    }
}
fn client_builder(url: &Url, addrs: &[SocketAddr], policy: IngressPolicy) -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .https_only(true)
        .http1_only()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .pool_max_idle_per_host(0)
        .connect_timeout(policy.connect_timeout)
        .timeout(policy.total_timeout)
        .dns_resolver(Arc::new(NoSecondaryDns))
        .resolve_to_addrs(url.host_str().unwrap(), addrs)
}
async fn fetch_response(client: Client, url: Url, cap: usize, work: &RequestWork) -> Result<Vec<u8>, ApiError> {
    // Do not include reqwest errors: they may contain credentials or the full URL query.
    let mut response = client.get(url).header("Accept-Encoding", "identity").send().await.map_err(|e| {
        if e.is_timeout() {
            fetch_timeout()
        } else {
            denied("image URL connection failed")
        }
    })?;
    if response.status().is_redirection() {
        return Err(denied("image URL redirects are disabled"));
    }
    if !response.status().is_success() {
        return Err(denied("image URL did not return a successful response"));
    }
    if response.headers().get("content-encoding").is_some_and(|v| v != "identity") {
        return Err(denied("image URL compression is disabled"));
    }
    let content_type =
        response.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").split(';').next().unwrap_or("").trim();
    // an early filter only: the decoder sniffs the bytes, whatever the type says
    if !["image/png", "image/jpeg", "image/webp", "application/octet-stream"].contains(&content_type) {
        return Err(denied("unsupported image URL content type"));
    }
    if response.content_length().is_some_and(|n| n > cap as u64) {
        return Err(oversize());
    }
    // Fixed capacity: no Vec geometric over-allocation beyond the accounted cap.
    let mut bytes = Vec::with_capacity(cap);
    while let Some(chunk) =
        response.chunk().await.map_err(|e| if e.is_timeout() { fetch_timeout() } else { denied("image URL transfer failed") })?
    {
        work.check()?;
        if chunk.len() > cap.saturating_sub(bytes.len()) {
            return Err(oversize());
        }
        bytes.extend_from_slice(&chunk);
    }
    work.check()?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustls::pki_types::pem::PemObject;
    use std::io::{Read, Write};
    fn fixture(response: Vec<u8>, stall: Duration) -> (SocketAddr, Arc<AtomicU64>, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicU64::new(0));
        let counter = connections.clone();
        let certs = rustls::pki_types::CertificateDer::pem_slice_iter(include_bytes!("../tests/fixtures/ingress_tls/cert.pem"))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let key =
            rustls::pki_types::PrivateKeyDer::from_pem_slice(include_bytes!("../tests/fixtures/ingress_tls/key.pem")).unwrap();
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .unwrap();
        let thread = std::thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            counter.fetch_add(1, Ordering::SeqCst);
            socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            socket.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
            let conn = rustls::ServerConnection::new(Arc::new(config)).unwrap();
            let mut tls = rustls::StreamOwned::new(conn, socket);
            let mut request = [0; 8192];
            if tls.read(&mut request).is_ok() {
                std::thread::sleep(stall);
                let _ = tls.write_all(&response);
                let _ = tls.flush();
            }
        });
        (address, connections, thread)
    }
    fn test_client(url: &Url, address: SocketAddr, timeout: Duration, trust: bool) -> Client {
        // Test-only explicit socket injection; NOT a production policy or certificate bypass.
        let policy = IngressPolicy { enabled: true, total_timeout: timeout, ..Default::default() };
        let builder = client_builder(url, &[address], policy);
        let builder = if trust {
            builder.add_root_certificate(
                reqwest::Certificate::from_pem(include_bytes!("../tests/fixtures/ingress_tls/cert.pem")).unwrap(),
            )
        } else {
            builder
        };
        builder.build().unwrap()
    }
    async fn get_fixture(
        response: Vec<u8>,
        stall: Duration,
        cap: usize,
        timeout: Duration,
        trust: bool,
    ) -> Result<Vec<u8>, ApiError> {
        let (address, connections, thread) = fixture(response, stall);
        let url = validate_url("https://ingress.test/image").unwrap();
        let client = test_client(&url, address, timeout, trust);
        let result = fetch_response(client, url, cap, &RequestWork::new(Duration::from_secs(5))).await;
        thread.join().unwrap();
        assert_eq!(connections.load(Ordering::SeqCst), 1, "one forced socket connection, no retries or second resolution");
        result
    }
    #[tokio::test]
    async fn local_tls_positive_same_decoder_and_hostname_verification() {
        let png = include_bytes!("../../../testdata/vision/check_512x512.png");
        let mut response =
            format!("HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", png.len())
                .into_bytes();
        response.extend_from_slice(png);
        let bytes = get_fixture(response.clone(), Duration::ZERO, 10 << 20, Duration::from_secs(2), true).await.unwrap();
        let image = crate::vision::ImageRef { message: 0, part: 0, url: "https://ingress.test/image".into() };
        let pre =
            crate::vision::prepare_bytes(&image, &bytes, &Default::default(), &gb_vision::Preprocessing::qwen38(), None).unwrap();
        assert_eq!(pre.n_patches() / 4, 256);
        // A self-signed fixture is NOT trusted in the default production client.
        assert!(get_fixture(response, Duration::ZERO, 10 << 20, Duration::from_secs(2), false).await.is_err());
        let (address, _, thread) = fixture(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n".to_vec(), Duration::ZERO);
        let wrong_host = validate_url("https://wrong-host.test/image").unwrap();
        let result = fetch_response(
            test_client(&wrong_host, address, Duration::from_secs(2), true),
            wrong_host,
            100,
            &RequestWork::new(Duration::from_secs(5)),
        )
        .await;
        assert!(result.is_err(), "TLS must authenticate original hostname even with explicitly mapped socket");
        thread.join().unwrap();
    }
    #[tokio::test]
    async fn a_webp_response_is_accepted_and_decoded_and_other_image_types_are_not() {
        let webp = include_bytes!("../../../testdata/vision/webp/lossy_q75_97x61.webp");
        let mut response =
            format!("HTTP/1.1 200 OK\r\nContent-Type: image/webp\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", webp.len())
                .into_bytes();
        response.extend_from_slice(webp);
        let bytes = get_fixture(response, Duration::ZERO, 10 << 20, Duration::from_secs(2), true).await.unwrap();
        assert_eq!(bytes, webp);
        let image = crate::vision::ImageRef { message: 0, part: 0, url: "https://ingress.test/image.webp".into() };
        let pre =
            crate::vision::prepare_bytes(&image, &bytes, &Default::default(), &gb_vision::Preprocessing::qwen38(), None).unwrap();
        assert!(pre.n_patches() > 0);
        for ty in ["image/gif", "image/avif", "video/webm"] {
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: {ty}\r\nContent-Length: 4\r\nConnection: close\r\n\r\nRIFF");
            let e = get_fixture(response.into_bytes(), Duration::ZERO, 100, Duration::from_secs(2), true).await.unwrap_err();
            assert!(e.status == 400 && e.message.contains("content type"), "{ty}: {}", e.message);
        }
    }
    #[tokio::test]
    async fn tls_redirect_compression_type_oversize_and_timeout_canaries() {
        let redirected_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        redirected_listener.set_nonblocking(true).unwrap();
        let target = redirected_listener.local_addr().unwrap();
        let response = format!(
            "HTTP/1.1 302 Found\r\nLocation: https://127.0.0.1:{}/secret\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            target.port()
        );
        assert!(get_fixture(response.into_bytes(), Duration::ZERO, 100, Duration::from_secs(2), true)
            .await
            .unwrap_err()
            .message
            .contains("redirect"));
        assert!(redirected_listener.accept().is_err(), "redirect target got zero connections");
        for (headers, body, status) in [
            ("Content-Type: text/html\r\n", "html", 400),
            ("Content-Type: image/png\r\nContent-Encoding: gzip\r\n", "gzip", 400),
            ("Content-Type: image/png\r\n", "0123456789012345", 413),
        ] {
            let response =
                format!("HTTP/1.1 200 OK\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            assert_eq!(
                get_fixture(response.into_bytes(), Duration::ZERO, 8, Duration::from_secs(2), true).await.unwrap_err().status,
                status
            );
        }
        // Absent length/chunked overrun is refused while receiving, not just based on headers.
        let response=b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n8\r\n12345678\r\n8\r\nabcdefgh\r\n0\r\n\r\n";
        assert_eq!(
            get_fixture(response.to_vec(), Duration::ZERO, 10, Duration::from_secs(2), true).await.unwrap_err().status,
            413
        );
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 4\r\nConnection: close\r\n\r\n1234";
        assert_eq!(
            get_fixture(response.to_vec(), Duration::from_millis(150), 10, Duration::from_millis(50), true)
                .await
                .unwrap_err()
                .status,
            504
        );
    }
    #[tokio::test]
    async fn private_dns_and_literal_targets_never_dial() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let ingress = HttpsIngress::new(IngressPolicy { enabled: true, ..Default::default() }).unwrap();
        let work = RequestWork::new(Duration::from_secs(2));
        for url in [
            "https://127.0.0.1/a",
            "https://[::1]/a",
            "https://localhost/a",
            "http://localhost/a",
            "https://[::ffff:127.0.0.1]/a",
        ] {
            assert!(ingress.fetch(url, &work).await.is_err(), "{url}");
        }
        assert!(listener.accept().is_err());
        // Rebinding answers are revalidated on each fetch; mixed responses are denied as a whole, not filtered.
        assert!(validate_addresses(vec!["8.8.8.8:443".parse().unwrap()]).is_ok());
        assert!(validate_addresses(vec!["127.0.0.1:443".parse().unwrap()]).is_err());
        assert!(validate_addresses(vec!["8.8.8.8:443".parse().unwrap(), "127.0.0.1:443".parse().unwrap()]).is_err());
        assert!(listener.accept().is_err());
    }
    #[tokio::test]
    async fn rebinding_and_unsafe_targets_have_zero_connector_calls_and_connections() {
        let response =
            b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 4\r\nConnection: close\r\n\r\n1234".to_vec();
        let (socket, connections, thread) = fixture(response, Duration::ZERO);
        let resolves = Arc::new(AtomicU64::new(0));
        let builds = Arc::new(AtomicU64::new(0));
        let policy = IngressPolicy { enabled: true, ..Default::default() };
        let work = RequestWork::new(Duration::from_secs(5));
        for visit in 0..2 {
            let dns = resolves.clone();
            let connector = builds.clone();
            let result = fetch_using(
                "https://ingress.test/image",
                policy,
                &work,
                move |_| async move {
                    let visit = dns.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![if visit == 0 { "8.8.8.8:443" } else { "127.0.0.1:443" }.parse().unwrap()])
                },
                move |url, _, _| {
                    connector.fetch_add(1, Ordering::SeqCst);
                    // ONLY this test client injects fixture socket/CA after the production target policy.
                    Ok(test_client(url, socket, Duration::from_secs(2), true))
                },
            )
            .await;
            if visit == 0 {
                assert_eq!(result.unwrap(), b"1234");
            } else {
                assert!(result.is_err());
            }
        }
        thread.join().unwrap();
        assert_eq!(resolves.load(Ordering::SeqCst), 2, "exactly one resolver visit per fetch; no secondary lookup");
        assert_eq!(builds.load(Ordering::SeqCst), 1, "rebinding to private never constructs a connector");
        assert_eq!(connections.load(Ordering::SeqCst), 1, "second unsafe fetch made zero connections");
        for raw in [
            "https://127.0.0.1/image",
            "https://[::1]/image",
            "https://[::ffff:8.8.8.8]/image",
            "https://ingress.test:444/image",
            "https://user:password@ingress.test/image",
            "http://ingress.test/image",
        ] {
            let connector = builds.clone();
            let result = fetch_using(
                raw,
                policy,
                &work,
                |_| async { Ok(vec!["8.8.8.8:443".parse().unwrap()]) },
                move |url, _, _| {
                    connector.fetch_add(1, Ordering::SeqCst);
                    Ok(test_client(url, socket, Duration::from_secs(2), true))
                },
            )
            .await;
            assert!(result.is_err(), "{raw}");
        }
        for answers in
            [vec!["10.0.0.1:443"], vec!["8.8.8.8:443", "127.0.0.1:443"], vec!["[fc00::1]:443"], vec!["[64:ff9b::808:808]:443"]]
        {
            let addresses = answers.iter().map(|a| a.parse().unwrap()).collect::<Vec<_>>();
            let connector = builds.clone();
            assert!(fetch_using(
                "https://ingress.test/image",
                policy,
                &work,
                move |_| async move { Ok(addresses) },
                move |url, _, _| {
                    connector.fetch_add(1, Ordering::SeqCst);
                    Ok(test_client(url, socket, Duration::from_secs(2), true))
                }
            )
            .await
            .is_err());
        }
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "every unsafe literal/DNS/scheme/credential target stopped before connection construction"
        );
        assert_eq!(connections.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn detached_dns_job_retains_its_slot_until_actual_drop() {
        let slots = Arc::new(tokio::sync::Semaphore::new(2));
        let first = BoundedDnsJob::with_slots("first.invalid".into(), slots.clone()).unwrap();
        let second = BoundedDnsJob::with_slots("second.invalid".into(), slots.clone()).unwrap();
        assert_eq!(slots.available_permits(), 0);
        let (tx, rx) = std::sync::mpsc::channel();
        let (entered, started) = std::sync::mpsc::channel();
        let handle = tokio::task::spawn_blocking(move || {
            let _held = &first; // whole owner, not a disjoint permit field capture
            entered.send(()).unwrap();
            rx.recv_timeout(Duration::from_secs(2)).unwrap();
        });
        started.recv_timeout(Duration::from_secs(1)).unwrap();
        drop(handle);
        assert_eq!(BoundedDnsJob::with_slots("third.invalid".into(), slots.clone()).err().unwrap().status, 503);
        assert_eq!(slots.available_permits(), 0);
        tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while slots.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(second);
        assert_eq!(slots.available_permits(), 2);
    }
    #[tokio::test]
    async fn secondary_dns_is_fail_closed() {
        use reqwest::dns::Resolve;
        let name = "unapproved.test".parse().unwrap();
        assert!(NoSecondaryDns.resolve(name).await.is_err());
    }
}
