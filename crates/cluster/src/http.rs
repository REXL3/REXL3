//! The one HTTP request the launcher makes: `GET /health` and `GET /health/ready` of rank 0, over std's `TcpStream`.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub body: String,
}

/// `GET http://host:port/path` with a connect and a read timeout. `Err` is a transport failure (refused, timed out, malformed).
pub fn get(host: &str, port: u16, path: &str, timeout: Duration) -> Result<Response, String> {
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("{host}:{port}: {e}"))?
        .next()
        .ok_or_else(|| format!("{host}:{port}: no address"))?;
    let mut s = TcpStream::connect_timeout(&addr, timeout).map_err(|e| format!("{host}:{port}: {e}"))?;
    s.set_read_timeout(Some(timeout)).map_err(|e| e.to_string())?;
    s.set_write_timeout(Some(timeout)).map_err(|e| e.to_string())?;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUser-Agent: rexl3-cluster\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    );
    s.write_all(req.as_bytes()).map_err(|e| format!("{host}:{port}: {e}"))?;
    let mut buf = Vec::new();
    // at most 256 KiB: a health body is a few hundred bytes
    let mut chunk = [0u8; 8192];
    while buf.len() < 256 * 1024 {
        match s.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if buf.is_empty() => return Err(format!("{host}:{port}: {e}")),
            Err(_) => break,
        }
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((text.as_str(), ""));
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| format!("{host}:{port}: not an HTTP response"))?;
    Ok(Response { status, body: body.trim().to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn serve_once(reply: &'static str) -> u16 {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = l.accept() {
                let mut b = [0u8; 1024];
                let _ = s.read(&mut b);
                let _ = s.write_all(reply.as_bytes());
            }
        });
        port
    }

    #[test]
    fn reads_status_and_body() {
        let p = serve_once("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok");
        let r = get("127.0.0.1", p, "/health/ready", Duration::from_secs(2)).unwrap();
        assert_eq!(r, Response { status: 200, body: "ok".into() });
        let p = serve_once("HTTP/1.1 503 Service Unavailable\r\n\r\nloading");
        let r = get("127.0.0.1", p, "/health/ready", Duration::from_secs(2)).unwrap();
        assert_eq!((r.status, r.body.as_str()), (503, "loading"));
    }

    #[test]
    fn a_closed_port_is_an_error() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let p = l.local_addr().unwrap().port();
        drop(l);
        assert!(get("127.0.0.1", p, "/", Duration::from_millis(500)).is_err());
    }
}
