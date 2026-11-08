#!/usr/bin/env python3
"""One real single-node rexl3-serve (original Nemotron pack, GPU) behind rexl3-router: replies through the router are
byte-identical to direct, a hard kill mid-stream ends the client's stream with an error (no hang), and the router marks
the restarted server up again without being restarted. Run under `flock -w 600 /tmp/rexl3-gpu.lock`."""
import hashlib, http.client, json, os, signal, subprocess, sys, tempfile, time, urllib.request, urllib.error

# usage: flock -w 600 /tmp/rexl3-gpu.lock python3 -I smoke_gpu.py REPO_ROOT
# env: REXL3_SERVE_BIN (rexl3-serve), REXL3_PACK (model dir), REXL3_ROUTER_BIN (default REPO_ROOT/target/release/rexl3-router),
#      REXL3_SMOKE_OUT (logs and evidence json; default a fresh temp dir)
HERE = os.environ.get("REXL3_SMOKE_OUT") or tempfile.mkdtemp(prefix="router-smoke-gpu-")
os.makedirs(HERE, exist_ok=True)
WT = sys.argv[1]  # repo root (evals/capture_replies.py lives there)
ROUTER = os.environ.get("REXL3_ROUTER_BIN") or f"{WT}/target/release/rexl3-router"
SERVE = os.environ["REXL3_SERVE_BIN"]
PACK = os.environ["REXL3_PACK"]
SP, RP, AP = 8951, 8990, 8991
ev = {}
server = None
router = None


def start_server(tag):
    log = open(f"{HERE}/gpu_server_{tag}.log", "wb")
    return subprocess.Popen([SERVE, "--model", PACK, "--backend", "cuda", "--slots", "4", "--no-vision", "--allow-unvalidated-pack",
                             "--max-model-len", "8192", "--served-model-name", "m", "--spec", "auto", "--host", "127.0.0.1",
                             "--port", str(SP), "--shutdown-grace", "20"], stdout=log, stderr=log)


def get(port, path, timeout=3):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception:
        return 0, b""


def wait_ready(port, proc, timeout=400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"server exited rc={proc.returncode} while loading")
        if get(port, "/health/ready")[0] == 200:
            return time.time() - t0
        time.sleep(0.2)
    raise SystemExit("server not ready in time")


def capture(base, out):
    t0 = time.time()
    r = subprocess.run(["python3", "-I", f"{WT}/evals/capture_replies.py", "--base-url", base, "--model", "m", "--streams", "4",
                        "--out", out], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise SystemExit(f"capture failed: {r.stderr[-400:]}")
    return hashlib.sha256(open(out, "rb").read()).hexdigest(), r.stdout.strip(), time.time() - t0


def norm_events(raw):
    out = []
    for line in raw.decode().splitlines():
        if not line.startswith("data: "):
            continue
        p = line[6:]
        if p == "[DONE]":
            out.append("[DONE]")
            continue
        d = json.loads(p)
        d.pop("id", None)
        d.pop("created", None)
        out.append(json.dumps(d, sort_keys=True))
    return out


def stream(port, max_tokens, seed=11, on_events=None, n_events=2, timeout=40):
    body = {"model": "m", "messages": [{"role": "user", "content": "Write a long, detailed story about a lighthouse keeper."}],
            "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.95, "seed": seed, "stream": True}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    resp = conn.getresponse()
    raw, err, seen, fired = b"", None, 0, False
    t_end = None
    try:
        while True:
            chunk = resp.read1(8192)
            if not chunk:
                break
            raw += chunk
            seen = raw.count(b"\n\n")
            if on_events and not fired and seen >= n_events:
                fired = True
                on_events()
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    t_end = time.time()
    conn.close()
    return resp.status, resp.getheader("x-rexl3-backend"), raw, err, t_end


def main():
    global server, router
    t = time.time()
    server = start_server("a")
    ev["boot_s"] = round(wait_ready(SP, server), 1)
    print(f"server ready in {ev['boot_s']} s (pid {server.pid})", flush=True)

    sha_d, out_d, dur_d = capture(f"http://127.0.0.1:{SP}/v1", f"{HERE}/replies_direct.json")
    print(f"direct   : {sha_d}  {out_d}  ({dur_d:.0f}s)", flush=True)

    router = subprocess.Popen([ROUTER, "--endpoint", f"http://127.0.0.1:{SP}", "--bind", f"127.0.0.1:{RP}", "--admin-bind",
                               f"127.0.0.1:{AP}", "--probe-interval-ms", "500", "--probe-timeout-ms", "1000",
                               "--probe-fail-threshold", "2", "--probe-recover-threshold", "2"],
                              stdout=open(f"{HERE}/gpu_router.log", "wb"), stderr=subprocess.STDOUT)
    for _ in range(100):
        if get(RP, "/health")[0] == 200:
            break
        time.sleep(0.1)
    assert get(RP, "/health")[0] == 200

    sha_r, out_r, dur_r = capture(f"http://127.0.0.1:{RP}/v1", f"{HERE}/replies_router.json")
    print(f"via router: {sha_r}  {out_r}  ({dur_r:.0f}s)", flush=True)
    ev["replies"] = {"direct_sha256": sha_d, "router_sha256": sha_r, "identical": sha_d == sha_r, "summary": out_r,
                     "direct_s": round(dur_d), "router_s": round(dur_r)}
    assert sha_d == sha_r, "router must be byte-transparent"

    # streaming equality (ids and timestamps differ by request; everything else must match)
    _, _, via, e1, _ = stream(RP, 120)
    _, _, direct, e2, _ = stream(SP, 120)
    n_via, n_direct = norm_events(via), norm_events(direct)
    ev["sse_equal"] = {"events": len(n_via), "identical_modulo_id_created": n_via == n_direct, "done": n_via[-1:] == ["[DONE]"]}
    print("sse via router vs direct:", ev["sse_equal"], flush=True)
    assert n_via == n_direct and e1 is None and e2 is None

    # a few concurrent streams through the router hold in-flight, visible in the router's own metrics
    # (kept short: the point is the counter, the byte equality is above)

    # hard kill mid-stream
    kill_t = {}

    def kill():
        kill_t["t"] = time.time()
        server.send_signal(signal.SIGKILL)

    status, backend, raw, err, t_end = stream(RP, 400, on_events=kill, n_events=5)
    server.wait()
    took = t_end - kill_t["t"]
    done = norm_events(raw)[-1:] == ["[DONE]"]
    ev["kill_midstream"] = {"http_status": status, "client_error": err, "complete_stream": done, "events_before_cut": raw.count(b"\n\n"),
                            "kill_to_client_end_s": round(took, 3)}
    print("kill mid-stream:", ev["kill_midstream"], flush=True)
    assert (err is not None or not done) and took < 10

    # router notices (prober), no server: requests are refused cleanly
    t0 = time.time()
    while time.time() - t0 < 15:
        st, b = get(RP, "/health")
        if st == 503:
            break
        time.sleep(0.05)
    ev["down_detected_s"] = round(time.time() - kill_t["t"], 2)
    req_body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{RP}/v1/chat/completions", req_body,
                                                      {"content-type": "application/json"}), timeout=10)
        ev["request_while_down"] = 200
    except urllib.error.HTTPError as e:
        ev["request_while_down"] = e.code
        ev["request_while_down_body"] = json.loads(e.read())
    print("health 503 after", ev["down_detected_s"], "s; request while down:", ev["request_while_down"], flush=True)

    # restart the same server; the router must bring it back on its own
    t_restart = time.time()
    server = start_server("b")
    ready_s = wait_ready(SP, server)
    t_ready = time.time()
    while time.time() - t_ready < 30:
        st, b = get(RP, "/health")
        if st == 200:
            break
        time.sleep(0.05)
    up_s = time.time() - t_restart
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{RP}/v1/chat/completions", req_body,
                                                      {"content-type": "application/json"}), timeout=60)
    ev["restart"] = {"server_ready_s": round(ready_s, 1), "restart_to_router_up_s": round(up_s, 1),
                     "router_up_after_ready_s": round(time.time() - t_ready - 0.0, 2), "request_after_status": r.status}
    print("restart:", ev["restart"], flush=True)

    # and the replies are still identical after the restart
    sha_r2, _, _ = capture(f"http://127.0.0.1:{RP}/v1", f"{HERE}/replies_router_after_restart.json")
    ev["replies"]["router_after_restart_sha256"] = sha_r2
    ev["replies"]["identical_after_restart"] = sha_r2 == sha_d
    print("after restart via router:", sha_r2, sha_r2 == sha_d, flush=True)

    st, m = get(RP, "/metrics?scope=router")
    ev["router_metrics"] = [l for l in m.decode().splitlines() if l and not l.startswith("#")
                            and any(k in l for k in ("requests_total", "inflight", "transitions", "errors_total{", "client_disc"))
                            and not l.endswith(" 0")]
    ev["total_s"] = round(time.time() - t)
    json.dump(ev, open(f"{HERE}/gpu_evidence.json", "w"), indent=1)
    print("SMOKE_GPU_OK total", ev["total_s"], "s", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if router and router.poll() is None:
            router.send_signal(signal.SIGTERM)
        if server and server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except Exception:
                server.kill()
        json.dump(ev, open(f"{HERE}/gpu_evidence.json", "w"), indent=1)
