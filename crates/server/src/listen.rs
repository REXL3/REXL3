//! The HTTP/1.1 accept loop of `rexl3-serve`: `axum::serve` with the connection timeouts it cannot set.
//!
//! `axum::serve` builds its hyper connections without a timer, so hyper's header-read timeout (30 s by default) is silently off: a
//! client that opens a connection and sends half a request line, or nothing, holds a socket and a task for as long as it likes, and so
//! does an idle keep-alive connection. [`serve`] is the same loop (one task a connection, the router as the service, upgrades on,
//! graceful shutdown that stops accepting and lets every open connection finish its in-flight response) with hyper's
//! [`TokioTimer`] installed and `header_read_timeout` set:
//!
//! - a request's head (request line and headers) must arrive within the timeout of the connection becoming ready to read it, or the
//!   connection is closed;
//! - the same clock runs while a keep-alive connection waits idle for its next request (hyper 1.x starts it whenever the connection
//!   waits for a request head), so an idle keep-alive connection is closed after the timeout too;
//! - a request body and a response (a long SSE stream included) are not bounded by it: those are the request deadline's and the write
//!   stall's (`--request-timeout`, `--write-stall`).
use futures_util::FutureExt;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use std::future::Future;
use std::time::Duration;

/// Serve `app` on `listener` until `shutdown` resolves; then stop accepting, let every open connection finish the response it is
/// writing (an idle one closes at once) and return once all of them are gone. `header_read_timeout` bounds the wait for a request head,
/// on a new connection and on an idle keep-alive one.
pub async fn serve(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    header_read_timeout: Duration,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    // the graceful shutdown of `axum::serve`: dropping the receiver tells the accept loop and every connection task to stop
    let (signal_tx, signal_rx) = tokio::sync::watch::channel(());
    tokio::spawn(async move {
        shutdown.await;
        drop(signal_rx);
    });
    // every connection task holds a receiver; the loop waits for them all once it stops accepting
    let (close_tx, close_rx) = tokio::sync::watch::channel(());
    loop {
        let io = tokio::select! {
            r = listener.accept() => match r {
                Ok((io, _)) => io,
                Err(e) => {
                    accept_error(e).await;
                    continue;
                }
            },
            _ = signal_tx.closed() => break,
        };
        let service = TowerToHyperService::new(app.clone());
        let signal_tx = signal_tx.clone();
        let close_rx = close_rx.clone();
        tokio::spawn(async move {
            let mut builder = hyper::server::conn::http1::Builder::new();
            builder.timer(TokioTimer::new()).header_read_timeout(header_read_timeout);
            let mut conn = std::pin::pin!(builder.serve_connection(TokioIo::new(io), service).with_upgrades());
            let mut signal = std::pin::pin!(signal_tx.closed().fuse());
            loop {
                tokio::select! {
                    // a header timeout, a reset or a protocol error ends this connection only
                    _ = conn.as_mut() => break,
                    _ = &mut signal => conn.as_mut().graceful_shutdown(),
                }
            }
            drop(close_rx);
        });
    }
    drop(close_rx);
    drop(listener);
    close_tx.closed().await;
}

/// What `axum::serve` does with an accept error: a connection-level one (the peer gave up before we took it) is skipped; anything else
/// (out of file descriptors) is logged and the loop backs off for a second rather than spinning.
async fn accept_error(e: std::io::Error) {
    use std::io::ErrorKind::*;
    if matches!(e.kind(), ConnectionRefused | ConnectionAborted | ConnectionReset) {
        return;
    }
    eprintln!("rexl3 level=warn event=accept_error error={e:?}");
    tokio::time::sleep(Duration::from_secs(1)).await;
}
