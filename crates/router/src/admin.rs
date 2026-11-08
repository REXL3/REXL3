//! The admin API: list, add, remove, drain, disable and enable backends at runtime.
//!
//! It is served on its own listener (`--admin-bind`), never on the public port (which answers 404 for `/admin/*`), and
//! when an admin token is configured every request must carry it (`Authorization: Bearer <token>` or `x-admin-token`).
//! Changes are in memory: a restarted router starts from its `--endpoint` list again.
use crate::app::App;
use crate::backend::Admin;
use crate::handlers::backend_json;
use axum::extract::{Path, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

#[derive(Clone)]
struct Auth {
    token: Option<String>,
}

pub fn build_admin_router(app: App, token: Option<String>) -> Router {
    Router::new()
        .route("/admin/backends", get(list).post(add))
        .route("/admin/backends/{id}", get(show).delete(remove))
        .route("/admin/backends/{id}/drain", post(drain))
        .route("/admin/backends/{id}/disable", post(disable))
        .route("/admin/backends/{id}/enable", post(enable))
        .layer(axum::middleware::from_fn_with_state(Auth { token }, auth))
        .with_state(app)
}

/// Constant-time equality (the token is a shared secret).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
    }
    diff == 0
}

async fn auth(State(a): State<Auth>, req: Request, next: Next) -> Response {
    if let Some(token) = &a.token {
        let bearer = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(str::trim);
        let custom = req.headers().get("x-admin-token").and_then(|v| v.to_str().ok()).map(str::trim);
        let ok = [bearer, custom].into_iter().flatten().any(|p| ct_eq(p.as_bytes(), token.as_bytes()));
        if !ok {
            let body = json!({ "error": "admin token required" });
            let mut r = (StatusCode::UNAUTHORIZED, Json(body)).into_response();
            r.headers_mut().insert(header::WWW_AUTHENTICATE, header::HeaderValue::from_static("Bearer"));
            return r;
        }
    }
    next.run(req).await
}

fn not_found(id: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({ "error": "no such backend", "backend": id }))).into_response()
}

async fn list(State(app): State<App>) -> Response {
    let backends = app.backends();
    let routable = backends.iter().filter(|b| b.routable()).count();
    let doc = json!({
        "backends": backends.iter().map(backend_json).collect::<Vec<_>>(),
        "routable": routable,
        "total": backends.len(),
    });
    Json(doc).into_response()
}

async fn show(State(app): State<App>, Path(id): Path<String>) -> Response {
    match app.find(&id) {
        Some(b) => Json(backend_json(&b)).into_response(),
        None => not_found(&id),
    }
}

#[derive(Deserialize)]
struct AddBody {
    url: String,
}

/// `http://host:port` with nothing after it (TLS to backends is not compiled in: see `docs/CLUSTER_ROUTER.md`).
fn valid_backend_url(u: &str) -> bool {
    let u = u.trim_end_matches('/');
    let rest = match u.strip_prefix("http://") {
        Some(r) => r,
        None => return false,
    };
    !rest.is_empty() && !rest.contains(['/', '?', '#', ' ', '@', '\\']) && rest.chars().all(|c| !c.is_control())
}

async fn add(State(app): State<App>, Json(body): Json<AddBody>) -> Response {
    let url = body.url.trim().to_string();
    if !valid_backend_url(&url) {
        let err =
            json!({ "error": "url must be http://host:port (no path, query or credentials; TLS to backends is not supported)" });
        return (StatusCode::BAD_REQUEST, Json(err)).into_response();
    }
    match app.add_backend(&url) {
        Err(existing) => {
            let doc = json!({ "error": "backend already present", "backend": backend_json(&existing) });
            (StatusCode::CONFLICT, Json(doc)).into_response()
        }
        Ok(b) => {
            // First probe now, so the answer says whether it is up; a backend that is still loading is added anyway and
            // the prober brings it up when it is ready.
            let up = app.probe_one(&b, true).await;
            eprintln!("rexl3-router event=admin action=add backend={} up={up}", b.id());
            (StatusCode::CREATED, Json(backend_json(&b))).into_response()
        }
    }
}

async fn remove(State(app): State<App>, Path(id): Path<String>) -> Response {
    match app.remove_backend(&id) {
        Some(b) => {
            eprintln!(
                "rexl3-router event=admin action=remove backend={} inflight={}",
                b.id(),
                b.inflight.load(std::sync::atomic::Ordering::Relaxed)
            );
            Json(json!({ "removed": backend_json(&b) })).into_response()
        }
        None => not_found(&id),
    }
}

fn set(app: &App, id: &str, to: Admin) -> Response {
    match app.find(id) {
        Some(b) => {
            let from = b.admin();
            b.set_admin(to);
            eprintln!("rexl3-router event=admin action={} backend={} from={}", to.name(), b.id(), from.name());
            Json(backend_json(&b)).into_response()
        }
        None => not_found(id),
    }
}

async fn drain(State(app): State<App>, Path(id): Path<String>) -> Response {
    set(&app, &id, Admin::Draining)
}

async fn disable(State(app): State<App>, Path(id): Path<String>) -> Response {
    set(&app, &id, Admin::Disabled)
}

async fn enable(State(app): State<App>, Path(id): Path<String>) -> Response {
    // A disabled backend is not probed, so its health is stale: probe once before it takes traffic again.
    if let Some(b) = app.find(&id) {
        if b.admin() == Admin::Disabled {
            app.probe_one(&b, true).await;
        }
    }
    set(&app, &id, Admin::Enabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_validation() {
        assert!(valid_backend_url("http://192.0.2.1:8001"));
        assert!(valid_backend_url("http://host-d.example:8001/"));
        assert!(!valid_backend_url("https://host-d.example:443"), "TLS to backends is not compiled in");
        assert!(!valid_backend_url("192.0.2.1:8001"));
        assert!(!valid_backend_url("http://"));
        assert!(!valid_backend_url("http://u:p@host:1"));
        assert!(!valid_backend_url("http://host:1/v1"));
        assert!(!valid_backend_url("http://host:1?x=1"));
        assert!(!valid_backend_url("ftp://host:1"));
    }

    #[test]
    fn constant_time_compare() {
        assert!(ct_eq(b"secret", b"secret"));
        assert!(!ct_eq(b"secret", b"secreT"));
        assert!(!ct_eq(b"secret", b"secret2"));
        assert!(!ct_eq(b"", b"x"));
    }
}
