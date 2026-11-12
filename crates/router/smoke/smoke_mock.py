#!/usr/bin/env python3
"""Live router smoke against real rexl3-serve processes on the CPU mock engine (no GPU).

Three mock backends behind the router: spread, conversation stickiness, SSE equality vs direct, kill mid-stream, failover,
restart + recovery, drain, runtime add, real "engine failed" 503 failover (mock fatal_step), metrics, SIGTERM.
"""
import http.client, json, os, signal, subprocess, sys, tempfile, time, urllib.request, urllib.error

# usage: python3 -I smoke_mock.py PATH_TO_rexl3-router
# env: REXL3_SERVE_BIN (rexl3-serve), REXL3_PACK (model dir, for the tokenizer), REXL3_SMOKE_OUT (logs, evidence json)
HERE = os.environ.get("REXL3_SMOKE_OUT") or tempfile.mkdtemp(prefix="router-smoke-mock-")
os.makedirs(HERE, exist_ok=True)
SERVE = os.environ["REXL3_SERVE_BIN"]
PACK = os.environ["REXL3_PACK"]
ROUTER = sys.argv[1]
RP, AP = 8990, 8991
PORTS = {"A": 8961, "B": 8962, "C": 8963}
procs = {}


def start_backend(name, port, mock="step_ms=20"):
    log = open(f"{HERE}/mock_{name}.log", "ab")
    p = subprocess.Popen([SERVE, "--model", PACK, "--backend", "mock", "--slots", "4", "--no-vision", "--allow-unvalidated-pack",
                          "--max-model-len", "8192", "--served-model-name", "m", "--host", "127.0.0.1", "--port", str(port),
                          "--spec", "none", "--mock", mock], stdout=log, stderr=log)
    procs[name] = p
    return p


def wait_ready(port, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready", timeout=1) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            time.sleep(0.05)
    raise SystemExit(f"backend on {port} never became ready")


def req(port, path, body=None, method=None, headers=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    h = {"content-type": "application/json"} if body is not None else {}
    h.update(headers or {})
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def chat(port, user, system="You are a test.", max_tokens=8, stream=False, seed=7, headers=None, timeout=30):
    body = {"model": "m", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": 0, "seed": seed, "stream": stream}
    return req(port, "/v1/chat/completions", body, headers=headers, timeout=timeout)


def router_health():
    st, _, b = req(RP, "/health")
    return st, json.loads(b)


def backend_state(doc, name):
    for b in doc["backends"]:
        if b["backend"] == f"127.0.0.1:{PORTS[name]}":
            return b
    return None


def wait_state(name, up, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, doc = router_health()
        b = backend_state(doc, name)
        if b is not None and b["up"] == up:
            return time.time() - t0
        time.sleep(0.05)
    raise SystemExit(f"backend {name} never reached up={up}")


def sse_events(raw):
    out = []
    for line in raw.decode().splitlines():
        if line.startswith("data: "):
            out.append(line[6:])
    return out


def sse_summary(raw):
    ev = sse_events(raw)
    text, finish, usage = "", None, None
    for e in ev:
        if e == "[DONE]":
            continue
        d = json.loads(e)
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices", []):
            text += (c.get("delta") or {}).get("content") or ""
            text += (c.get("delta") or {}).get("reasoning_content") or ""
            finish = c.get("finish_reason") or finish
    return {"events": len(ev), "text": text, "finish": finish, "done": ev[-1:] == ["[DONE]"], "usage": usage}


def sse_through(port, user, max_tokens, on_first=None, timeout=20):
    """Stream a chat; return (status, backend header, raw bytes, error or None, seconds)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = {"model": "m", "messages": [{"role": "user", "content": user}], "max_tokens": max_tokens, "temperature": 0,
            "seed": 3, "stream": True}
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    resp = conn.getresponse()
    backend = resp.getheader("x-rexl3-backend")
    raw, err, t_err = b"", None, None
    fired = False
    try:
        while True:
            chunk = resp.read1(4096)
            if not chunk:
                break
            raw += chunk
            if on_first and not fired:
                fired = True
                on_first(backend)
    except Exception as e:  # incomplete read, reset, timeout
        err = f"{type(e).__name__}: {e}"
        t_err = time.time()
    conn.close()
    return resp.status, backend, raw, err, t_err


def main():
    evidence = {}
    for n, p in PORTS.items():
        start_backend(n, p)
    for n, p in PORTS.items():
        wait_ready(p)
    rlog = open(f"{HERE}/router.log", "ab")
    endpoints = [a for p in PORTS.values() for a in ("--endpoint", f"http://127.0.0.1:{p}")]
    router = subprocess.Popen([ROUTER, *endpoints, "--bind", f"127.0.0.1:{RP}", "--admin-bind", f"127.0.0.1:{AP}",
                               "--probe-interval-ms", "300", "--probe-timeout-ms", "1000", "--probe-fail-threshold", "2",
                               "--probe-recover-threshold", "2"], stdout=rlog, stderr=rlog)
    t0 = time.time()
    while time.time() - t0 < 10:
        try:
            if router_health()[0] == 200:
                break
        except Exception:
            time.sleep(0.05)
    st, doc = router_health()
    assert st == 200 and doc["routable"] == 3, doc
    print("router up: 3 routable")

    # 1. distinct prompts spread (no affinity header)
    by = {}
    for i in range(30):
        st, h, _ = chat(RP, f"distinct question number {i} about topic {i * 7}")
        assert st == 200, st
        by[h["x-rexl3-backend"]] = by.get(h["x-rexl3-backend"], 0) + 1
    print("spread of 30 distinct prompts:", by)
    assert len(by) == 3 and min(by.values()) >= 4, by
    evidence["spread"] = by

    # 2. one conversation stays put
    homes = set()
    for t in range(6):
        body = {"model": "m", "max_tokens": 4, "messages": [
            {"role": "system", "content": "You are a test."}, {"role": "user", "content": "the conversation opener"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": f"turn {t}"}]}
        st, h, _ = req(RP, "/v1/chat/completions", body)
        homes.add(h["x-rexl3-backend"])
    print("conversation backends:", homes)
    assert len(homes) == 1
    evidence["conversation_backend"] = sorted(homes)

    # 3. SSE through the router equals SSE direct (same seeded request, mock engine is deterministic)
    st, h, via = chat(RP, "stream equality probe", max_tokens=24, stream=True)
    home = h["x-rexl3-backend"]
    direct_port = int(home.split(":")[1])
    st2, h2, direct = chat(direct_port, "stream equality probe", max_tokens=24, stream=True)
    a, b = sse_summary(via), sse_summary(direct)
    print("sse via router:", {k: a[k] for k in ("events", "finish", "done", "usage")})
    print("sse direct    :", {k: b[k] for k in ("events", "finish", "done", "usage")})
    assert a == b and a["done"], (a, b)
    evidence["sse_equal"] = {"events": a["events"], "text_sha_len": len(a["text"])}

    # 4. kill the backend mid-stream: the client gets an error, not a hang
    killed = {}

    def kill(backend):
        port = int(backend.split(":")[1])
        name = [n for n, p in PORTS.items() if p == port][0]
        procs[name].send_signal(signal.SIGKILL)
        killed["name"], killed["t"] = name, time.time()

    st, backend, raw, err, t_err = sse_through(RP, "kill me mid stream please", 300, on_first=kill, timeout=15)
    summ = sse_summary(raw)
    took = (t_err or time.time()) - killed["t"]
    print(f"mid-stream kill of {killed['name']}: status={st} err={err!r} done={summ['done']} events={summ['events']} "
          f"client_saw_end_in={took:.2f}s")
    assert err is not None or not summ["done"], "the stream must not look complete"
    assert took < 10, "client must not hang"
    evidence["kill_midstream"] = {"backend": killed["name"], "client_error": err, "seconds_to_client_error": round(took, 2),
                                  "events_before_cut": summ["events"], "done_marker": summ["done"]}
    procs[killed["name"]].wait()

    # 5. traffic continues on the others (retry covers the first request), health shows the dead one down
    codes = [chat(RP, f"after the kill {i}")[0] for i in range(12)]
    print("requests after kill:", codes)
    assert all(c == 200 for c in codes), codes
    t_down = wait_state(killed["name"], False)
    print(f"{killed['name']} marked down (waited {t_down:.2f}s after the requests)")
    evidence["after_kill_codes"] = codes

    # 6. restart the same port: router brings it back with no router restart
    t_restart = time.time()
    start_backend(killed["name"], PORTS[killed["name"]])
    boot = wait_ready(PORTS[killed["name"]])
    t_up = wait_state(killed["name"], True)
    total = time.time() - t_restart
    print(f"{killed['name']} restarted: ready in {boot:.2f}s, router up again {t_up:.2f}s later (restart->up {total:.2f}s)")
    seen = False
    for i in range(60):
        _, h, _ = chat(RP, f"post recovery probe {i}")
        if h["x-rexl3-backend"] == f"127.0.0.1:{PORTS[killed['name']]}":
            seen = True
            break
    assert seen, "recovered backend takes traffic again"
    evidence["restart_recovery"] = {"backend": killed["name"], "boot_s": round(boot, 2), "router_up_after_ready_s": round(t_up, 2),
                                    "restart_to_up_s": round(total, 2), "took_traffic": seen}

    # 7. drain B while it streams: the stream finishes, new requests avoid it
    out = {}

    def drain(backend):
        st, _, b = req(AP, f"/admin/backends/{backend}/drain", method="POST")
        out["drain"] = json.loads(b)
        out["backend"] = backend

    st, backend, raw, err, _ = sse_through(RP, "drain while streaming", 60, on_first=drain)
    s = sse_summary(raw)
    print(f"drain during stream: status={st} err={err} done={s['done']} events={s['events']} "
          f"drain_state={out['drain']['admin']} inflight_at_drain={out['drain']['inflight']}")
    assert err is None and s["done"], "in-flight stream must finish"
    landed = set()
    for i in range(20):
        _, h, _ = chat(RP, f"while drained {i}")
        landed.add(h["x-rexl3-backend"])
    assert out["backend"] not in landed, landed
    _, doc = router_health()
    drained_entry = [b for b in doc["backends"] if b["backend"] == out["backend"]][0]
    print("drained entry:", {k: drained_entry[k] for k in ("admin", "routable", "drained", "inflight")})
    assert drained_entry["drained"] is True
    req(AP, f"/admin/backends/{out['backend']}/enable", method="POST")
    evidence["drain"] = {"backend": out["backend"], "stream_completed": s["done"], "events": s["events"],
                         "new_requests_avoided_it": True, "entry": {k: drained_entry[k] for k in ("admin", "routable", "drained")}}

    # 8. a real "engine failed" 503: a mock engine with fatal_step=3 dies on its first generation. Add it at runtime,
    # make it the preferred backend, and check that clients never see the 503.
    start_backend("D", 8964, mock="step_ms=20,fatal_step=3")
    wait_ready(8964)
    st, _, b = req(AP, "/admin/backends", {"url": "http://127.0.0.1:8964"}, method="POST")
    d = json.loads(b)
    print("add D:", st, d["up"])
    PORTS["D"] = 8964
    assert st == 201 and d["up"]
    codes, served = [], []
    for i in range(40):
        st, h, body = chat(RP, f"fatal probe {i}", max_tokens=12)
        codes.append(st)
        served.append(h.get("x-rexl3-backend"))
        _, doc = router_health()
        db = backend_state(doc, "D")
        if db and not db["up"]:
            break
    print("codes while D's engine dies:", codes)
    assert all(c == 200 for c in codes), codes
    _, doc = router_health()
    db = backend_state(doc, "D")
    print("D after engine failure:", {k: db[k] for k in ("up", "status", "consecutive_failures")})
    assert not db["up"]
    st, _, m = req(RP, "/metrics?scope=router")
    metrics = m.decode()
    keep = [l for l in metrics.splitlines() if l.startswith(("rexl3_router_requests_total", "rexl3_router_failovers_total",
            "rexl3_router_no_backend_total", "rexl3_router_affinity_requests_total", "rexl3_router_health_transitions_total",
            "rexl3_router_client_disconnects_total")) and not l.endswith(" 0") or l.startswith("rexl3_router_failovers_total")]
    evidence["metrics_nonzero"] = keep
    evidence["engine_failed_failover"] = {"codes_all_200": True, "D_up_after": db["up"]}
    full = req(RP, "/metrics")[2].decode()
    assert 'rexl3_router_rexl3_ready{backend="127.0.0.1:8961"}' in full or "rexl3_router_rexl3_ready" in full
    print("metrics lines:\n  " + "\n  ".join(keep))

    # 9. SIGTERM: graceful exit
    t = time.time()
    router.send_signal(signal.SIGTERM)
    rc = router.wait(timeout=40)
    print(f"router exited rc={rc} {time.time() - t:.2f}s after SIGTERM")
    evidence["sigterm"] = {"rc": rc, "seconds": round(time.time() - t, 2)}

    for p in procs.values():
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    for p in procs.values():
        try:
            p.wait(timeout=20)
        except Exception:
            p.kill()
    json.dump(evidence, open(f"{HERE}/mock_evidence.json", "w"), indent=1)
    print("SMOKE_MOCK_OK")


if __name__ == "__main__":
    try:
        main()
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
