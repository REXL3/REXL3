# REXL3 runtime — qwen38-27b-dflash2-exl3-4.00

A prebuilt native runtime with pinned model files, reviewed serving profiles and
an OpenAI-compatible API. Use the **exact production release**, not a development
checkout or GitHub's automatic source-code ZIP, to run inference. The operator
path needs no private development-repository access, Rust/CUDA compilation,
container image, or floating `latest` dependency.

## 1. Know what you are running

| Identity | Locked value |
| --- | --- |
| Runtime | `qwen38-27b-dflash2-exl3-4.00` |
| Default profile | `qwen38-single-262k` |
| Target repository | `r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw` |
| Profile source repository | `REXL3/REXL3-dev` |
| Profile source commit | `6b56c6bad8dfb041d39b431355e7d9e5bcfe6e66` |
| Profile source path / SHA-256 | `docs/examples/rexl3-serve.service` / `dc9052ea7211b90feddce5caca8ac8319ef2c72368c64b68392fe880a5c39d4e` |

| Role | Repository | Revision | Files | Bytes | GiB | Fingerprint | Weight license |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `draft` | `r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw` | `265b5240592907d2d55ff0dc4d5f66569692604d` | 3 | 1254702082 | 1.17 | `sha256:3b47e03a4d1e15615e5340baf0fa5e5ffbb0ab15f2b25b069dc25132a18c687d` | Apache-2.0 (model card metadata of the pack repository) ([upstream at pin](https://huggingface.co/r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw/tree/265b5240592907d2d55ff0dc4d5f66569692604d)) |
| `target` | `r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw` | `3f1771b8c21f83cbb8e82169559ced9f38ca04e5` | 14 | 16533404065 | 15.40 | `sha256:967c3be941ce25284d4ddaaaed02cd3cda5e940ffa622400ed792ffc7a367940` | Apache-2.0 (model card metadata of the pack repository) ([upstream at pin](https://huggingface.co/r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw/tree/3f1771b8c21f83cbb8e82169559ced9f38ca04e5)) |

The profile source above is provenance for the serving configuration, **not** the
binary build identity. `release.lock.json` records the production commit/tree,
native build/version fields and each bundled file's hash and mode. The sibling
release asset `approval.json` binds the archive and qualification evidence. It is
**not assumed to be inside the tarball**. A generated table, `state: final`,
`DOCTOR_PASS`, or passing CPU fixture is not hardware qualification; consult that
exact release's approval and [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### Profiles and maximum context

| Profile | Kind | Context tokens (configured) | Slots | KV dtype | Speculation | Prefix cache | Served model | Readiness deadline (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen38-single-262k` (default) | `single` | `262144` | `6` | not declared | `auto` | `on` | `qwen38-27b-exl3` | 600 |

These are declared settings, not measured throughput or RAM promises. Select the
profile for your topology and required context; do not edit sealed flags to make a
smaller profile into a larger one. `max_model_len` from `/v1/models` is the effective
limit; `configured_max_model_len` is the configured ceiling. Prompt/template/image
and requested output tokens must fit the effective budget. A maximum-context
profile does not mean every prompt leaves room for an equally large completion.
Keep the declared slots baseline; concurrency or context changes require separate
memory/quality qualification. Speculation and prefix caching follow the profile,
not arbitrary environment overrides.

### Capability matrix

| Capability | `qwen38-single-262k` |
| --- | --- |
| `audio` | unsupported |
| `images` | enabled |
| `json_schema` | unsupported |
| `prefix_cache` | enabled |
| `reasoning` | enabled |
| `sse` | enabled |
| `text` | enabled |
| `tools` | enabled |
| `video` | unsupported |

`json_schema` means **constrained decoding**, not asking the model to write JSON.
Ordinary JSON prompting is text generation and must be parsed/validated by your
application. The current API rejects `response_format` values `json_object` and
`json_schema`; only text format is implemented. Do not infer video/audio support
from images, or forced tool calls from tool support. Unsupported capabilities are
not enabled by adding a request field. Image input uses local bytes in a data URI,
not a remote image URL. Only `tool_choice: "auto"` or `"none"` is supported.

## 2. Download and verify before extracting

Use [production releases](https://github.com/REXL3/REXL3/releases). Select an exact
tag and copy the download URL of its `approval.json` asset. Set `RELEASE_BASE` to
that URL **without** the final `/approval.json` (preserve the tag's URL encoding).
Do not use a latest-release redirect or guess the archive name. The following
requires Bash, curl and Python 3.11+ on the downloading machine. If the target host
lacks Python, do this on a trusted workstation and transfer the complete verified
directory; the target launcher can bootstrap Python as described in SETUP.

<!-- runtime-example: release-download -->
```bash
set -euo pipefail
: "${RELEASE_BASE:?Set the exact production release download URL prefix first}"
case "$RELEASE_BASE" in
  https://github.com/REXL3/REXL3/releases/download/*) ;;
  *) printf '%s\n' 'Expected the selected production release URL' >&2; exit 2 ;;
esac
DOWNLOAD_DIR=$(mktemp -d "${TMPDIR:-$HOME}/rexl3-release.XXXXXXXX")
export DOWNLOAD_DIR
cd "$DOWNLOAD_DIR"
curl --fail --location --proto '=https' --proto-redir '=https' \
  --connect-timeout 15 --max-time 120 --output approval.json "$RELEASE_BASE/approval.json"
ARCHIVE=$(python3 - <<'PY'
import json, re
from pathlib import Path
if not __debug__:
    raise SystemExit('Verification requires assertions; rerun without Python optimization')
approval = json.loads(Path('approval.json').read_text())
assert approval['schema'] == 'rexl3.profile-approval.v1', 'Unknown approval schema'
assert approval['runtime_id'] == 'qwen38-27b-dflash2-exl3-4.00', 'Wrong runtime'
bundle = approval['bundle']
assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz', bundle['archive']), 'Unsafe archive name'
assert re.fullmatch(r'[0-9a-f]{64}', bundle['sha256']), 'Invalid checksum'
assert type(bundle['bytes']) is int and bundle['bytes'] > 0, 'Invalid size'
print(bundle['archive'])
PY
)
export ARCHIVE
curl --fail --location --proto '=https' --proto-redir '=https' \
  --connect-timeout 15 --max-time 3600 --output "$ARCHIVE" "$RELEASE_BASE/$ARCHIVE"
```

Run this verification/extraction block in the same download directory. It hashes
before creating the new `bundle/` directory and rejects links, traversal,
duplicates and special files. A mismatch is a hard failure, not a warning.
Checksums establish byte identity with the approval; obtain that approval from
the trusted exact release, not an unrelated message or a modified local archive.

<!-- runtime-example: release-verify -->
```python
import hashlib, json, os, re, shutil, tarfile
from pathlib import Path, PurePosixPath
if not __debug__:
    raise SystemExit('Verification requires assertions; rerun without Python optimization')
approval = json.loads(Path('approval.json').read_text())
assert approval['schema'] == 'rexl3.profile-approval.v1'
assert approval['runtime_id'] == 'qwen38-27b-dflash2-exl3-4.00'
record = approval['bundle']
assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz', record['archive'])
assert re.fullmatch(r'[0-9a-f]{64}', record['sha256'])
assert type(record['bytes']) is int and record['bytes'] > 0
archive = Path(record['archive'])
assert not archive.is_symlink() and archive.is_file()
assert archive.stat().st_size == record['bytes'], 'Archive size mismatch'
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256'], 'Archive checksum mismatch'
destination = Path('bundle')
assert not destination.exists() and not destination.is_symlink(), 'Use a fresh directory'
with tarfile.open(archive, 'r:gz') as tar:
    members, names, total = [], set(), 0
    for member in tar:
        path = PurePosixPath(member.name)
        assert path.parts and not path.is_absolute() and str(path) == member.name
        assert '..' not in path.parts and '.git' not in path.parts and '\\' not in member.name
        assert not any(ord(c) < 32 or ord(c) == 127 for c in member.name)
        assert member.isfile() and not member.issparse() and not member.pax_headers
        assert member.mode in (0o644, 0o755) and member.name not in names
        total += member.size
        assert 0 <= member.size <= 1024**3 and total <= 8 * 1024**3 and len(members) < 10000
        names.add(member.name)
        members.append(member)
    assert {'launch.sh', 'release.lock.json', 'profile.json', 'model.lock.json'} <= names
    assert not any(parent.as_posix() in names for name in names for parent in PurePosixPath(name).parents)
    destination.mkdir(mode=0o755)
    for member in members:
        target = destination / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as source, target.open('xb') as output:
            shutil.copyfileobj(source, output)
        target.chmod(member.mode)
print('ARCHIVE_VERIFIED_AND_EXTRACTED')
```

The Python blocks here are programs: save them to a local file and run with
`python3`, or paste into a `python3 -` heredoc. Do not run with `python3 -O` (these
verification examples deliberately assert their contracts). Keep `approval.json`
and the archive beside `bundle/` for updates, rollback and issue identity.

## 3. Prepare a supported host and choose a profile

The installer adapter supports **Ubuntu 24.04 ARM64 on NVIDIA GB10**. It requires
working signed Ubuntu/NVIDIA repositories, Bash/coreutils, apt/dpkg, systemd 255+
with cgroup v2 and a reachable login user manager. Sudo is used only for approved
system-package operations; do **not** run `start` or the whole launcher with sudo.
Driver, CUDA runtime and cuBLAS versions are not interchangeable simply because
they are newer. See the generated [dependency table and bootstrap boundary](docs/SETUP.md).
Exact GPU/memory/library requirements and native version comparisons come from
`release.lock.json`; the archive does not contain model weights.

Locked payload total: **17788106147 bytes** (16.57 GiB). For an empty single-node managed cache, the current downloader requires at least **18861847971 free bytes** (payload plus its 1073741824-byte / 1 GiB reserve). For a partial download it checks missing-file bytes plus the same reserve; existing fragments already consume disk. This is a download floor, not a RAM requirement or a promise of peak disk use. Allow additional space for the archive, extraction, Python wheels/environment, logs and rollback copies. Attested two-node profiles instead require the full payload pre-staged on each rank; this managed-download floor does not enable remote acquisition. Full hash verification rereads every locked file; neither setup nor startup has a promised duration.

In `bundle/`, select one profile from the table. Keep these variables in **each**
operator terminal, and keep the same site file for every command on an instance.
Single-node sites are optional. An `attested-v1` TP profile has a **required
two-node site**: prepare it using [SETUP](docs/SETUP.md) and export `SITE` as its
absolute filename **before** this block. The shipped example contains placeholder
addresses/paths, not discovered hosts. Do not pass that cluster example to a
single-node profile. Export `PROFILE` to select a different profile before this
block; otherwise it uses the generated default. Leave `SITE` empty only when that
profile permits it.

<!-- runtime-example: select-profile -->
```bash
set -euo pipefail
cd "$DOWNLOAD_DIR/bundle"
export RUNTIME_ROOT="$PWD"
if [[ -z "${PROFILE:-}" ]]; then
    PROFILE=qwen38-single-262k
fi
export PROFILE
export SITE="${SITE:-}"
SITE_ARGS=()
if [[ -n "$SITE" ]]; then SITE_ARGS=(--site "$SITE"); fi
umask 077
WORK=$(mktemp -d "${TMPDIR:-$HOME}/rexl3-client.XXXXXXXX")
export WORK
./launch.sh --help
./launch.sh print-command --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh --yes install-deps
./launch.sh doctor
```

`doctor` is read-only, takes no profile/site options, and is **not** proof of model
bytes, free disk, GPU idleness, occupied ports or inference. `print-command`
validates configuration and prints the resolved argv/environment without starting
inference, contacting ranks or downloading payloads. The selection block resolves
first, so a missing required site fails before installation. Resolution is not
remote identity or readiness verification. It may expose local paths; review
before sharing. Dependency installation here prepares the controller host only;
TP ranks must already satisfy the per-rank prerequisites below.

### Model preparation: single-node reuse/download versus TP verification

| Profile | Declared site fields | Site requirement | Model preparation |
| --- | --- | --- | --- |
| `qwen38-single-262k` | `binary_dir`, `http_host`, `http_port`, `model_dir` | optional; omitted site uses profile defaults / managed cache | typed `model_dir` binding; read-only verification |

- **Single-node:** a complete already-managed payload is reverified and reused
  without download. If your selected profile declares `model_dir`, use its
  **typed site binding**
  for an existing absolute model directory. It is a read-only source: every
  locked file/sidecar, size, hash, index and fingerprint must verify before
  launch. Never download into or repair that external directory automatically.
  See [site setup](docs/SETUP.md); this is not an arbitrary argv override.
- A single-node profile/launcher without that binding rejects it. Historical v1
  launchers do not gain external-directory or remote-rank support from a
  documentation change.
- **Attested two-node TP:** `nodes.leader.model_dir` and
  `nodes.follower.model_dir` are mandatory read-only external directories, even
  though no top-level `model_dir` is declared. Before setup, each rank must have
  pre-staged exact matching `rexl3-serve` and `rexl3-cluster` binaries, the complete
  locked model payload (all shards and sidecars), and the locked dependencies.
  `setup`/`verify` attest those ranks; there is **no automatic remote download**,
  remote dependency installation or repair. `--accept-model-download` does not
  provision TP ranks. See [the required two-node site](docs/SETUP.md).
- **Single-node managed download only:** `setup` shows the remaining byte
  count/destination and acquires only the locked revision/files. Routine `--yes` is **not** model-download
  consent. No separate draft is downloaded unless the model lock declares one.

```bash
# Complete single-node payload or pre-staged TP ranks: verify without download consent.
./launch.sh --yes setup --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh verify --profile "$PROFILE" "${SITE_ARGS[@]}"
```

Only for a **single-node managed cache** with missing files (no external
`model_dir`), explicitly authorize the pinned download instead:

```bash
./launch.sh --yes setup --profile "$PROFILE" "${SITE_ARGS[@]}" --accept-model-download
./launch.sh verify --profile "$PROFILE" "${SITE_ARGS[@]}"
```

`setup` and `start` both bootstrap/check dependencies before model preparation;
`install-deps` alone never downloads weights or starts inference. For unattended
work, authorize the required sudo operations separately with your administrator;
`--yes` cannot supply a password. Download credentials, if needed, use the official
HF client environment (`HF_TOKEN`); never put tokens in argv, URLs, site JSON,
logs or issue reports. The inference child does not inherit them.

### Start and wait for owned readiness

```bash
./launch.sh --yes start --profile "$PROFILE" "${SITE_ARGS[@]}"
# Single-node managed download only: add --accept-model-download after reviewing the lock/size.
```

`start` is foreground: keep the terminal open. Ctrl-C, SIGTERM and SIGHUP (a
dropped terminal or SSH session) request cleanup of this invocation's owned
process/group. It must verify the lock/process identity,
listener, `/health/ready`, `/v1/models`, non-streaming chat and (when enabled) SSE
smoke before reporting `READY`. An unrelated healthy server cannot satisfy this
check. The per-profile readiness deadline starts at native launch, **after** model
verification/acquisition; do not promise a fixed cold-start time.

If the launcher itself was killed (for example SIGKILL) while its server survived,
run `stop` (or `reset-state`) with the same profile and site: they terminate only a
recorded server whose PID and process-start identity still match, run the native
cluster `down` for an orphaned TP2 run, and never signal any other PID.

In another terminal with the same `RUNTIME_ROOT`, `PROFILE` and selected `SITE`
absolute filename, run this bounded readiness wait. Set `WAIT_READY_S` to your
explicit operator budget (including any preparation still in progress). Timeout
exits **24**; it never falls through to success and does not kill unrelated work.
The foreground launcher retains its own deadline/cleanup responsibility.

<!-- runtime-example: owned-readiness -->
```python
import json, math, os, subprocess, sys, time
from pathlib import Path
budget = float(os.environ['WAIT_READY_S'])
if not math.isfinite(budget) or budget <= 0:
    raise SystemExit('WAIT_READY_S must be finite and positive')
command = [str(Path(os.environ['RUNTIME_ROOT']) / 'launch.sh'), 'status', '--profile', os.environ['PROFILE']]
if os.environ.get('SITE'):
    command += ['--site', os.environ['SITE']]
deadline = time.monotonic() + budget
while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        break
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=min(10, remaining))
    except subprocess.TimeoutExpired:
        continue
    if time.monotonic() >= deadline:
        break
    if result.returncode == 0:
        if json.loads(result.stdout).get('phase') != 'ready':
            raise SystemExit(24)
        print('OWNED_READY')
        raise SystemExit(0)
    if result.returncode != 24:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    time.sleep(min(1, max(0, deadline - time.monotonic())))
print('READINESS_WAIT_TIMEOUT: inspect status/logs; not ready', file=sys.stderr)
raise SystemExit(24)
```

## 4. Use the API: text, streaming, thinking, image and tools

Use an OpenAI-compatible client's **base URL ending in `/v1`**, not the server
root. Discover the model ID from `GET /v1/models`; do not use the HF repository ID
as a substitute. The following dependency-free client discovers that ID, checks
it against the selected profile and refuses examples not enabled by that profile.
It runs the real read-only `launch.sh print-command --profile ... --site ...`
resolution using `RUNTIME_ROOT`, `PROFILE` and exported `SITE` (omitting `--site`
only when `SITE` is empty). Both topologies use the resolved HTTP host/port,
including the TP leader address and site overrides; IPv6 is bracketed. A failed
or timed-out resolution stops before any HTTP request. Use the owned readiness
check first. Keep single-node endpoints on loopback and TP endpoints on a trusted
private network; this is not an authenticated public Internet gateway.

Normally leave `API_BASE` unset. For an operator-managed tunnel to the **same
selected instance**, explicitly set its reachable `http://HOST:PORT/v1` URL
(bracket IPv6). This changes only the client destination, never the launch
binding or site requirement; even an override must pass runtime resolution.

| Client case | Required capability | Enabled profiles |
| --- | --- | --- |
| `text` | `text` | `qwen38-single-262k` |
| `sse` | `sse` | `qwen38-single-262k` |
| `thinking-off / thinking-on` | `reasoning` | `qwen38-single-262k` |
| `image` | `images` | `qwen38-single-262k` |
| `tools` | `tools` | `qwen38-single-262k` |

Save this block as `$WORK/client.py` (outside the sealed bundle). Responses printed
by this client are private application data, not sanitized diagnostics. The image
is generated locally; no image download or extra Python library is needed. The
tool is a deliberately **local example fixture**, not a real device query.

<!-- runtime-example: api-client -->
```python
import base64, json, os, struct, subprocess, sys, urllib.request, zlib
from pathlib import Path


def image_uri():
    width, height = 256, 256
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG row filter
        for x in range(width):
            color = (255, 255, 255)
            if 20 <= x < 110 and 25 <= y < 145:
                color = (255, 0, 0)
            if (x - 180)**2 + (y - 85)**2 < 45**2:
                color = (0, 0, 255)
            if 20 <= x < 236 and 200 <= y < 230:
                color = (0, 160, 0)
            rows.extend(color)
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')


def run(case, base, choice):
    required = {'text': 'text', 'sse': 'sse', 'thinking-off': 'reasoning',
                'thinking-on': 'reasoning', 'image': 'images', 'tools': 'tools'}
    if case not in required or not choice['capabilities'][required[case]]:
        raise ValueError('Example not enabled by selected profile: ' + case)
    if not base.endswith('/v1'):
        raise ValueError('API_BASE must end in /v1 (without a trailing slash)')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def open_request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
        return opener.open(request, timeout=300)
    def get_json(path, payload=None):
        with open_request(path, payload) as response:
            return json.load(response)
    models = get_json('/models')['data']
    if len(models) != 1 or models[0]['id'] != choice['smoke_request']['model']:
        raise ValueError('Discovered model differs from selected profile')
    model = models[0]['id']
    print('model:', model, 'effective context:', models[0]['max_model_len'],
          'configured context:', models[0]['configured_max_model_len'])
    payload = {'model': model, 'messages': [{'role': 'user', 'content': 'Write one short sentence about a lighthouse.'}],
               'max_tokens': 256, 'temperature': 0}
    if choice['capabilities']['reasoning']:
        payload['chat_template_kwargs'] = {'enable_thinking': case == 'thinking-on'}
    if case in ('thinking-on', 'thinking-off'):
        payload['messages'][0]['content'] = 'Explain why every square is a rectangle but not every rectangle is a square.'
        payload['max_tokens'] = 1024
    if case == 'image':
        payload['messages'][0]['content'] = [
            {'type': 'text', 'text': 'Return only a JSON object mapping the rectangle, circle and bottom_bar to their colors; use lowercase color names and no markdown.'},
            {'type': 'image_url', 'image_url': {'url': image_uri()}}]
    if case == 'sse':
        payload.update(stream=True, stream_options={'include_usage': True})
        done, finished, content = False, False, ''
        with open_request('/chat/completions', payload) as response:
            if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                raise ValueError('Expected SSE')
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                for item in event['choices']:
                    delta = item.get('delta', {})
                    if delta.get('reasoning_content'):
                        print('reasoning:', delta['reasoning_content'])
                    piece = delta.get('content') or ''
                    content += piece
                    print(piece, end='', flush=True)
                    if item.get('finish_reason') is not None:
                        if item['finish_reason'] != 'stop':
                            raise ValueError('Stream truncated or did not complete')
                        finished = True
                if event.get('usage'):
                    print('\nusage:', event['usage'])
        if not (done and finished and content.strip()):
            raise ValueError('Incomplete SSE: need content, stop and [DONE]')
        print()
        return
    if case == 'tools':
        payload['messages'][0]['content'] = 'Call get_status for sample-unit, then report the returned status and source.'
        payload['tools'] = [{'type': 'function', 'function': {
            'name': 'get_status', 'description': 'Read the local example fixture status (no external effects).',
            'parameters': {'type': 'object', 'properties': {'unit': {'type': 'string', 'enum': ['sample-unit']}},
                           'required': ['unit'], 'additionalProperties': False}}}]
        payload['tool_choice'] = 'auto'  # Not "required" or a forced function object.
        first = get_json('/chat/completions', payload)['choices'][0]
        assistant = first['message']
        calls = assistant.get('tool_calls') or []
        if first['finish_reason'] != 'tool_calls' or not calls or len(calls) > 4:
            raise ValueError('Expected a bounded tool-call turn; auto does not force one')
        # Preserve the complete assistant turn, including reasoning and call IDs.
        payload['messages'].append(assistant)
        seen = set()
        for call in calls:
            function = call['function']
            if call['type'] != 'function' or function['name'] != 'get_status':
                raise ValueError('Tool not allowlisted; never eval/execute model output')
            arguments = json.loads(function['arguments'])
            if arguments != {'unit': 'sample-unit'} or not call['id'] or call['id'] in seen:
                raise ValueError('Invalid tool arguments or call ID')
            seen.add(call['id'])
            result = {'unit': 'sample-unit', 'status': 'idle', 'source': 'local example fixture'}
            payload['messages'].append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)})
        payload['tool_choice'] = 'none'  # One bounded round trip, then a final answer.
    reply = get_json('/chat/completions', payload)
    answer = reply['choices'][0]
    if answer['finish_reason'] != 'stop' or not (answer['message'].get('content') or '').strip():
        raise ValueError('No complete final answer; inspect token budget/finish_reason')
    message = answer['message']
    print('reasoning:', message.get('reasoning_content') or '')
    print('answer:', message['content'])
    print('usage:', reply.get('usage'))
    if case == 'image':
        try:
            colors = json.loads(message['content'])
        except ValueError:
            raise ValueError('Grounded image smoke failed: expected a JSON color map') from None
        if colors != {'rectangle': 'red', 'circle': 'blue', 'bottom_bar': 'green'}:
            raise ValueError('Grounded image smoke failed: incorrect shape/color association')


if __name__ == '__main__':
    root = Path(os.environ['RUNTIME_ROOT'])
    command = [str(root / 'launch.sh'), 'print-command', '--profile', os.environ['PROFILE']]
    if os.environ.get('SITE'):
        command += ['--site', os.environ['SITE']]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True, timeout=30)
    resolved = json.loads(result.stdout)
    profile = json.loads((root / 'profile.json').read_text())
    choice = profile['profiles'][resolved['profile']]
    host = resolved['http_host']
    if ':' in host:
        host = '[' + host + ']'
    base = os.environ.get('API_BASE', 'http://' + host + ':' + str(resolved['http_port']) + '/v1')
    run(sys.argv[1] if len(sys.argv) > 1 else 'text', base, choice)
```

```bash
python3 "$WORK/client.py" text
python3 "$WORK/client.py" sse
python3 "$WORK/client.py" thinking-off
python3 "$WORK/client.py" thinking-on
python3 "$WORK/client.py" image
python3 "$WORK/client.py" tools
```

The image check validates each shape/color association, not merely the presence
of color words. Its JSON request is **ordinary prompting**, with no constrained
decoder or `response_format` guarantee; invalid JSON is a failed smoke.
Run only cases enabled by your chosen profile. `/v1/models` also exposes the
model's `reasoning` controls; do not transplant another model's effort/budget
aliases. Reasoning is returned separately in `reasoning_content` (streaming:
`delta.reasoning_content`), not necessarily inside `content`. The thinking examples
use `chat_template_kwargs.enable_thinking` explicitly. The client checks protocol
completion, not factual quality or throughput. A single grounded image/tool smoke
is not a replacement for the exact release's qualification suite.

## 5. Operate, recover, update

```bash
./launch.sh status --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh logs --profile "$PROFILE" "${SITE_ARGS[@]}" --lines 80 --no-follow
./launch.sh diagnostics --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh stop --profile "$PROFILE" "${SITE_ARGS[@]}"
```

`status` returns 0 only for verified ready; **24 means not ready**, including
stopped/preparing/failed. `stop` can return 0 when already stopped. A successful
stop is not a successful inference test. Raw `logs` may contain private data;
`diagnostics` writes/prints an allowlisted JSON report, not raw logs, prompts,
responses, environment values, credentials or SSH keys. Review even that report
before sharing. State is under
`${XDG_DATA_HOME:-$HOME/.local/share}/rexl3/<runtime-id>/`; model cache, receipts,
Python environment and private logs persist across launches.

For recovery and the full exit-code table, use [TROUBLESHOOTING](docs/TROUBLESHOOTING.md).
Update/rollback by stopping with the **old** bundle/profile/site first, preserving
its approval/archive and state, and verifying/extracting the new (or previous)
release into a **different empty directory**. Rerun install-deps, doctor, setup,
print-command, owned readiness and enabled examples. Never overwrite a running
bundle, rewrite hashes to silence an integrity failure, or use `git pull` as a
binary updater. Identity-mismatched state is a refusal, not permission to delete
locks or kill a PID. Guarded `reset-state` may be used only once all previous owned
processes are proved absent; it preserves model data, logs and environment.

## Source, provenance and licenses

REXL3's own engine, launcher, profiles, evaluation code and documentation are
**AGPL-3.0-only**, copyright **the REXL3 authors**; retain [LICENSE](LICENSE),
[NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `sbom/`.
Weights are separate and not relicensed by the runtime. The upstream pack's
license is recorded/linked in the model table; an MIT-licensed upstream pack
remains MIT, not AGPL. If the lock points to the upstream card instead of an SPDX
identifier, consult that pinned card/license before redistribution.

For corresponding source, follow the **accessible source artifact/link in the
exact production release notes**, matched to `release.lock.json`'s production
commit/tree. This must work without private dev access. GitHub's automatic source
archive of a bundle-only repository is not necessarily the engine's corresponding
source. If the release has no accessible matching source, ask the publisher to
supply it before redistribution; this README does not fabricate an asset URL or
claim one was published. [SETUP](docs/SETUP.md) includes source verification and
platform inspection. Agents must preserve sealed inputs and report exact evidence
rather than copying historical performance claims.

## Agent execution checklist

1. Read this README, SETUP, TROUBLESHOOTING and KNOWN_ISSUES before acting. Pin the
   exact archive, approval, profile, model revision and native build identity.
2. Keep installation, model-download, driver-maintenance and GPU/remote-execution
   authority separate. A CPU/documentation task does not authorize any of them.
3. Use only the selected profile and typed site bindings. Never change sealed
   knobs, model pins, inventory hashes or environment to turn a failure green.
4. Preserve the foreground owner and its locks; wait with a hard deadline. Only
   owned status 0 is ready. A stopped service or unrelated HTTP 200 is not a pass.
5. Exercise enabled examples using discovered model identity. Require real tool
   calls/results and final answers; never fabricate responses, execute arbitrary
   model output, or treat truncated SSE as completion.
6. Keep raw logs, model payloads, credentials and conversations private. Review
   diagnostics JSON before sharing; do not upload the full state directory.
7. Stop only owned work, verify cleanup, and retain archives/model caches for
   rollback. Never broad-kill, signal a reused PID, unlink locks or force-kill a
   dependency controller whose cgroup drain is unverified.
8. Report commands, exit codes, exact artifact identities, tested outcomes and
   blockers. CPU fixtures are not GPU/TP/context/performance qualification; do
   not copy historical metrics or assert an inaccessible source artifact exists.
