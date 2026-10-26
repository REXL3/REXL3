# REXL3 user guide

This guide takes you from an empty directory to a running, OpenAI-compatible model server on your own
NVIDIA GB10 (DGX Spark class) host, shows how to use it, and tells you what to do when something goes
wrong. It is written for testers and operators who did not build REXL3.

It is runtime-neutral. Everything that depends on the pack you chose (its context window, port, model
size, which request switches it honours, its measured behaviour) is in that pack's **pack sheet** under
[`docs/packs/`](packs/README.md). Read this guide once, then keep the pack sheet beside you.

## 1. How the documentation fits together

| Document | What it is for | Authority |
| --- | --- | --- |
| [README](../README.md) | Front page: which packs are released, the shortest path | The released-packs table |
| This guide | The whole journey, the API cookbook, operations, troubleshooting | How to use any pack |
| [Pack sheets](packs/README.md) | One page per released runtime: sizes, window, port, switches, expectations | That runtime's facts |
| The bundle's own `README.md`, `docs/SETUP.md`, `docs/TROUBLESHOOTING.md` | The exact command reference inside each release archive | **Wins over this guide** for that release |
| `KNOWN_ISSUES.md` in each release | What was and was not qualified, behaviour to expect | That release |
| `approval.json` (a release asset) | Who approved which bytes | Identity of the release |
| [SECURITY](../SECURITY.md) | Reporting a vulnerability, the threat model | Security |

If this guide and the bundle disagree, follow the bundle and tell the maintainers (section 11). One exception
while this repository is private: the bundle's download example uses `curl` against the public release URL,
which fails without GitHub access. Download with `gh release download` (section 4.2), then continue with the
bundle's verification and extraction steps.

Conventions: commands are for Bash on the GB10 host unless said otherwise. `$RUNTIME` is a runtime id
such as `qwen38-27b-dflash2-exl3-4.00`; `$VER` is a release version such as `v1`. A line starting with `#`
inside a code block is a comment.

## 2. Before you start

### 2.1 What you need

| Need | Detail |
| --- | --- |
| Host | An NVIDIA GB10 machine (DGX Spark class), **Ubuntu 24.04 on ARM64**. Another GPU, OS or architecture is not supported and `doctor` will say so. |
| GPU | Idle. One server per machine; the launcher refuses to start on a busy GPU (exit code 22). Close other GPU workloads first. |
| Memory and disk | See the pack sheet: model size, and the free space the download needs. A pack can need most of the machine's memory, so do not run other large workloads beside it. |
| Privileges | A normal user with `sudo` available **for system packages only**. Never run `./launch.sh start` or the whole launcher as root; it refuses. |
| System services | systemd 255 or newer with cgroup v2 and a working login user manager (the installer checks). |
| Network | Outbound HTTPS to GitHub (the release) and Hugging Face (the model), and to the Ubuntu and NVIDIA package repositories. |
| GitHub access | While the repository is private you need an invitation and the GitHub CLI, `gh`, logged in (`gh auth login`, then `gh auth status`). Without access the release cannot be downloaded. |
| Time | Plan for the model download (size is on the pack sheet) and a first start that verifies every model file. The launcher gives no fixed start-up time because it depends on disk and network. |

### 2.2 What the launcher will and will not do

- It installs only the pinned system packages and a private Python environment, after you pass `--yes`.
  `--yes` is **not** consent to download a model.
- It downloads the model only when you pass `--accept-model-download`, shows how much it will fetch and
  where it goes, and verifies every file against the release's lock before use.
- It never reboots the machine and never changes the GPU driver unless you pass the separate
  `--allow-driver-change` flag after agreeing it with your administrator.
- It starts the server as your user, bound to `127.0.0.1` only.
- It stops only the process it started. It never uses a broad kill.
- It stays attached to your terminal while serving (`start` is a foreground command).

## 3. Quick start

For a pack whose sheet says "single node". Replace the two variables; everything else is as written.

```bash
RUNTIME=<runtime-id>      # from the README's released-packs table
VER=v1                                  # or the contents of runtimes/$RUNTIME/LATEST

gh release download "prod/$RUNTIME/$VER" -R REXL3/REXL3 -D /tmp/pack
ARCHIVE=$(python3 -c "import json;print(json.load(open('/tmp/pack/approval.json'))['bundle']['archive'])")   # the name approval.json records
sha256sum "/tmp/pack/$ARCHIVE"        # must equal bundle.sha256 in /tmp/pack/approval.json
mkdir -p ~/rexl3 && tar -xzf "/tmp/pack/$ARCHIVE" -C ~/rexl3 && cd ~/rexl3

./launch.sh --yes install-deps          # system packages and a private Python environment
./launch.sh doctor                      # read-only checks; must exit 0
./launch.sh --yes setup --accept-model-download   # downloads and verifies the model
./launch.sh --yes start                 # foreground; leave this terminal open
```

In a second terminal:

```bash
curl -s http://127.0.0.1:8940/v1/models            # use the port on the pack sheet
```

The rest of this guide explains each step, what success looks like, and what to do when it does not.
If a pack has more than one profile, add `--profile <name>` to `setup`, `start`, `status`, `logs`, `stop`
and `verify`, and read section 4.6.

## 4. Step by step

### 4.1 Choose a pack and a version

The [README](../README.md) lists the released packs. Open the pack's sheet to see its model, context
window, slots, port, and what it supports. Pick the version in `runtimes/<id>/LATEST` unless a maintainer
told you to test a specific one. Releases are immutable: a fix is a new version, never an edit.

### 4.2 Download the release

```bash
gh auth status                                     # confirm you are logged in
gh release download "prod/$RUNTIME/$VER" -R REXL3/REXL3 -D /tmp/pack
ls -l /tmp/pack                                    # the archive, approval.json, and any source archives
```

You should see the runnable bundle (a `*.tar.gz`) and `approval.json`; newer releases attach more than one archive, so always take the bundle's name from `approval.json` (next step), never a wildcard. Newer releases also attach source
archives (the Corresponding Source for the binaries); the first release has none. Download into a fresh directory each time.

### 4.3 Verify before you extract

`approval.json` records the archive's SHA-256 and size. Compare them yourself:

```bash
ARCHIVE=$(python3 -c "import json;print(json.load(open('/tmp/pack/approval.json'))['bundle']['archive'])")   # the name approval.json records
sha256sum "/tmp/pack/$ARCHIVE"
grep -A4 '"bundle"' /tmp/pack/approval.json
```

The two hashes must be identical. If they differ, **stop**: do not extract or run the archive, download it
again from the release, and if it still differs report it (section 11). For the stricter verify-and-extract
program (it rejects links, path traversal and duplicates before writing anything), use the one in the
bundle's `README.md`, section "Download and verify before extracting".

### 4.4 Extract

```bash
mkdir -p ~/rexl3 && tar -xzf "/tmp/pack/$ARCHIVE" -C ~/rexl3 && cd ~/rexl3
```

Use a new directory for every version. Never extract over a bundle that is running. Keep the archive and
`approval.json` next to it: you need them for updates, rollback and bug reports.

### 4.5 Install the dependencies, then run `doctor`

```bash
./launch.sh --yes install-deps
./launch.sh doctor
```

`install-deps` installs the pinned system packages (this is where `sudo` is used) and builds a private,
hash-pinned Python environment. It does not download a model and does not start anything.

`doctor` is read-only. Exit 0 means the platform, packages and bundle integrity checks passed. It does
**not** prove there is enough disk space, that the GPU is idle, that the port is free, or that inference
works; those are checked later with their own error tokens (section 8).

### 4.6 Profiles and print-command

A release can offer several profiles (for example different context windows or topologies). The pack
sheet lists them. To see exactly what a profile will run, without starting anything or downloading:

```bash
./launch.sh print-command --profile <profile-name>
```

Omit `--profile` to use the release's default profile. A **two-node** profile needs a site file that
names both machines and cannot be started by a single-node command; the pack sheet says if that applies,
and the bundle's `docs/SETUP.md` has the site format. Do not edit a sealed profile to change its flags.

### 4.7 Download and verify the model

```bash
./launch.sh --yes setup --accept-model-download    # add --profile <name> if the pack has several
./launch.sh verify                                 # re-check without downloading
```

`--yes` (before the command) approves routine package checks only; it is never consent to download a model,
which is why `--accept-model-download` is separate. `setup` shows the remaining bytes and the destination, fetches only the locked repository revision and
files, and verifies every file's size and hash. An interrupted download resumes at the same pin. Models
live under `${XDG_DATA_HOME:-$HOME/.local/share}/rexl3/<runtime-id>/models/`; nothing is stored in the
bundle. If you already have the model files, a single-node profile may allow an existing directory through
a site file (see the bundle's `docs/SETUP.md`); the launcher then verifies it read-only and never repairs
or downloads into it.

If a download needs credentials, set `HF_TOKEN` in your environment. Never put a token in a command line,
a URL or a site file, and never paste one into a bug report.

### 4.8 Start

```bash
./launch.sh --yes start                            # --profile <name> if needed
```

`start` stays in the foreground. It verifies the files, launches the server, and reports `READY` only after
the server answers its health check, lists the expected model, and completes a real chat request (and a
streaming one when the profile enables it). An unrelated server that happens to be healthy cannot satisfy
this check. Ctrl-C, a terminal hang-up or `SIGTERM` stops the server it started.

To wait for readiness from a script, poll `./launch.sh status`: exit 0 means verified ready, exit 24 means
not ready (stopped, preparing, starting or failed). Do not treat a timeout as success.

### 4.9 Check that it works

From a second terminal on the same host (the server listens on `127.0.0.1` only):

```bash
BASE=http://127.0.0.1:8940/v1                      # port from the pack sheet
curl -s $BASE/models | python3 -m json.tool
curl -s http://127.0.0.1:8940/health/ready         # 200 when ready, 503 while loading or draining
```

`/v1/models` shows the model **id to use in requests**, `max_model_len` (the effective limit you must stay
within) and `configured_max_model_len` (the ceiling the profile asked for). Use the id it prints; do not use
the Hugging Face repository name.

## 5. Using the server

The server speaks the OpenAI chat-completions protocol. Point any OpenAI-compatible client at
`http://127.0.0.1:<port>/v1` with any non-empty API key (the server does not check keys).

### 5.1 A first request

```bash
curl -s $BASE/chat/completions -H 'content-type: application/json' -d '{
  "model": "<id from /v1/models>",
  "messages": [{"role": "user", "content": "In two sentences, what is a hash table?"}],
  "max_tokens": 512
}' | python3 -m json.tool
```

Read `choices[0].finish_reason`: `stop` means the model finished; `length` means it was cut off by
`max_tokens` or by the context window, so the answer is incomplete. Reasoning models can spend their whole
budget thinking: if `content` is empty and `finish_reason` is `length`, raise `max_tokens` or lower the
reasoning effort (section 5.3). Always set `max_tokens` yourself.

### 5.2 Streaming

Add `"stream": true` for server-sent events: each `data:` line holds a JSON chunk and the stream ends with
`data: [DONE]`. Add `"stream_options": {"include_usage": true}` to receive a final chunk with token counts.
Treat a stream that ends without `[DONE]` as incomplete.

### 5.3 Thinking (reasoning)

Packs that reason return their thinking separately: `message.reasoning_content` (streaming:
`delta.reasoning_content`) holds the thinking and `message.content` holds the answer. Two request fields
control it:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
{"reasoning_effort": "low"}
```

Whether a request thinks when it names neither depends on the pack and server build; `GET /v1/models`
reports the reasoning controls the pack exposes, and the pack sheet states the default. Thinking uses
context and time: a long thought plus a long answer must fit `max_model_len`.

### 5.4 Sampling, seeds and repeatable output

`temperature`, `top_p`, `top_k` and `min_p` are honoured. If you send none, the server uses the values in
the pack's `generation_config.json` when it has one, otherwise temperature 1.0, top_p 0.95 and top_k 20.
`top_k` of 0 or -1 means "no cutoff" but the engine considers at most 56 candidates, so any larger value
is capped at 56. `stop` sequences are honoured. `max_tokens` and `max_completion_tokens` both set the
output limit.

**Identical requests give identical replies unless you send a `seed`.** With no `seed` the server derives
its random numbers from the conversation, so the same prompt at a non-zero temperature returns the same
text every time. To get varied replies, send a different integer `seed` per request; to reproduce one,
resend its seed. For strictly repeatable output set `temperature` to 0.

### 5.5 Long prompts, context and the prefix cache

The prompt, the thinking and the answer together must fit `max_model_len`. A request that cannot fit is
refused with HTTP 400 and error code `context_length_exceeded`; it is not truncated. A very long prompt is
processed in one go before the first token appears, so expect a long wait before the first token on a
large input (the pack sheet gives the measured prompt speed).

The server keeps the state of earlier prompts and reuses a shared beginning, so a follow-up turn that
repeats earlier messages is much cheaper than the first. The response's
`usage.prompt_tokens_details.cached_tokens` shows how many prompt tokens were reused. Keep the start of
your conversation byte-identical between turns to benefit. Send `"prefix_cache": false` on a request to
bypass the cache (for example to compare). Speculative decoding is automatic; when it ran, `usage.spec`
reports drafted and accepted token counts.

### 5.6 Tools (function calling)

Send OpenAI-style `tools`, with `tool_choice` either `"auto"` (default) or `"none"`. The model returns
`tool_calls`; run them yourself and send the results back as `tool` messages. Limits: at most 64 tools per
request (more returns HTTP 413), and `tool_choice` of `"required"` or a named function is refused with
HTTP 400. Some client libraries add `parallel_tool_calls` automatically; the server refuses it, so turn it
off in the client. A tool turn that returns no calls is a normal answer, not an error; do not synthesise a
call.

### 5.7 Images

Packs that accept images take them inline, as base64 in a data URI, in a message content part:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What colour is the large shape?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}}]}
```

Remote image URLs are not fetched. The bundle's own examples use a 256 by 256 pixel image; much smaller test
images have been misread, so do not judge image support from a tiny picture. The pack sheet says whether
images are supported. A pack started without vision refuses image
parts. Video and audio are refused by every pack today.

### 5.8 Concurrency

A pack runs a fixed number of **slots** (the pack sheet lists them). Requests beyond the slots wait in a
queue and complete in turn; a full queue is answered with HTTP 429, and a request that waits too long with
HTTP 503 `server_overloaded` and a `Retry-After` header. More slots does not mean more speed: the GPU
is shared, so aggregate throughput rises slowly while each stream slows down. For the fastest single
answer use one request at a time.

### 5.9 Health, metrics and errors

| Endpoint | Meaning |
| --- | --- |
| `GET /health/ready` | 200 only while the engine is up and healthy; 503 while loading, draining or failed |
| `GET /health/live` | 200 while the process is up |
| `GET /v1/models` | Model id, effective and configured window, reasoning controls |
| `GET /metrics` | Prometheus text: request counts, queue, prefix-cache and speculation counters |

| HTTP | Typical cause | What to do |
| --- | --- | --- |
| 400 | An unsupported field, or `context_length_exceeded` | Read the JSON `error.message`; remove the field or shorten the request |
| 404 | Unknown model id or route | Use the id from `/v1/models` and a path under `/v1` |
| 413 | More than 64 tools | Send fewer tools |
| 429 | The queue is full | Back off and retry |
| 503 | Not ready, shutting down, or `server_overloaded` | Wait for `/health/ready`; retry after the `Retry-After` delay |

### 5.10 Not supported

Refused by name with a 400, so you can find them quickly: `response_format` of `json_object` or
`json_schema` (only plain text), `n` other than 1, `logit_bias`, non-zero `presence_penalty`,
`frequency_penalty` or `repetition_penalty`, `tool_choice` `required` or a named function,
`parallel_tool_calls`, per-token output `logprobs`, video and audio input, remote image URLs, and
fine-tuned or alternative model ids. There is no authentication, no TLS and no multi-model routing.

## 6. Day-to-day operation

```bash
./launch.sh status                         # JSON; exit 0 = ready, 24 = not ready
./launch.sh logs --lines 80 --no-follow    # a bounded tail (1 to 1000 lines)
./launch.sh stop                           # stops only the server this bundle started
./launch.sh diagnostics                    # a sanitized report you can share (review it first)
```

Use the **same bundle, profile and site** for `status`, `logs` and `stop` as for `start`. `start` must
stay running: close the terminal and the server stops (a normal drain), so use `tmux`, `screen` or your
own service manager if you need it to survive a logout. The launcher does not install a service of its own.

**Where files are.** The bundle is the directory you extracted. Models and runtime state are under
`${XDG_DATA_HOME:-$HOME/.local/share}/rexl3/<runtime-id>/`. Do not delete the state tree or the model
cache as a shortcut. If a launcher was killed and `start` refuses with an ownership error, run `stop` with
the original bundle and profile first, and `reset-state` only after you have proved no server process
remains.

**Updating and rolling back.** Treat a new version as a new installation: stop the old bundle, extract
the new archive into a **new** directory after verifying it (section 4.3), repeat 4.5 to 4.9, and keep the
old archive and `approval.json` so you can roll back by launching the old exact artifact the same way. Never
extract over a running bundle and never copy verification records between versions.

## 7. Performance expectations

The pack sheet gives measured speed for that release on a GB10: prompt speed (tokens per second while
reading your prompt) and generation speed. Remember what moves them:

- The first token of a long prompt waits for the whole prompt; a cached beginning skips that work.
- Generation speed is for one stream; concurrent streams share it (section 5.8).
- Thinking models spend generation time before they answer.
- Another process on the GPU or a low-memory host slows everything and can make the launcher refuse.
- Measured numbers describe the release's own test machine and conditions; yours can differ.

## 8. Troubleshooting

Start with `./launch.sh doctor`, then `./launch.sh status` and `./launch.sh logs --no-follow`. The launcher
prints an error **token** in capitals; the table lists the common ones and the bundle's
`docs/TROUBLESHOOTING.md` has the full list.

| Exit code | Meaning |
| --- | --- |
| 0 | Succeeded (`status`: verified ready) |
| 2 | Bad usage, an unsupported option, profile or site, or an incomplete profile |
| 20 | A dependency or privilege is missing, a reboot is required, or model-download consent is missing |
| 21 | Unsupported platform or GPU, or an unapproved environment override |
| 22 | A conflict: busy GPU or port, too little model disk, an unsafe path, or a stale process identity |
| 23 | Integrity failure: the bundle, native files or model do not match their locks |
| 24 | Readiness or runtime failure, or `status` reporting not ready |
| 130 | Interrupted |

| What you see | Likely cause | What to do |
| --- | --- | --- |
| `UNSUPPORTED_OS`, `UNSUPPORTED_ARCH`, `UNSUPPORTED_GPU` | Not Ubuntu 24.04 ARM64 on a GB10 | Use a supported host |
| `RUN_AS_NORMAL_USER` | You ran `start` as root or with `sudo` | Run as your normal user; only the installer uses `sudo` |
| `MODEL_DOWNLOAD_APPROVAL_REQUIRED` | The model is missing and you did not consent | Review the size, then add `--accept-model-download` |
| `INSUFFICIENT_MODEL_DISK` | Not enough free disk for the remaining download | Free space, then rerun `setup`; partial files already count |
| `MODEL_DOWNLOAD_FAILED` | Network or access problem | Restore access, rerun `setup`: it resumes at the same pin |
| `MODEL_INTEGRITY`, hash or fingerprint mismatch | Corrupt or changed model files | Stop. Do not edit hashes. Keep the evidence and report it |
| `PORT_OCCUPIED_OR_UNAVAILABLE`, `GPU_LOCK_BUSY`, `GPU_BUSY_OR_IDLE_PROBE_FAILED` | Another process owns the port or GPU | Stop that workload yourself; the launcher never kills others |
| `READINESS_TIMEOUT` or a smoke failure | The server did not become healthy in time | Run `logs --no-follow` and `diagnostics`; let cleanup finish; report |
| `RUNTIME_BUSY`, `STALE_OR_REUSED_PID_REFUSED` | A previous launch was killed without cleanup | `stop` with the original bundle and profile; `reset-state` only after nothing remains |
| `finish_reason: length` or empty `content` | Output budget used up, often by thinking | Raise `max_tokens` or lower reasoning effort (5.3) |
| HTTP 400 on an unfamiliar field | An unsupported request field (5.10) | Read `error.message` and remove the field |
| The client cannot connect | Wrong host or port, or the server is not ready | Check `status`, `/health/ready`, and the pack sheet's port; the server is loopback-only |

## 9. Security and privacy

- The server has **no authentication and no TLS**. It listens on `127.0.0.1` and the launcher refuses a
  non-loopback address for a single-node profile. Keep it that way. To reach it from another machine use an
  SSH tunnel or put your own reverse proxy in front of it that terminates TLS, authenticates clients and
  limits request size and time.
- Prompts and answers are held in memory and written to the server log only as the log shows; do not paste
  raw logs, prompts, environment dumps or credentials into a report. `diagnostics` is an allow-list for this
  reason, but read it before you share it.
- Model files are fetched only from the repository and revision pinned in the release lock and are checked
  against recorded hashes. Do not change a lock to load a model you do not trust.
- Report a vulnerability privately as described in [SECURITY](../SECURITY.md). Do not open a public issue.

## 10. Licences and source

The repository and the launcher are AGPL-3.0-only. Model weights keep their own licence, named in each pack
sheet and in the release's `model.lock.json`; weights are never stored in this repository. If you receive a
runtime archive, its binaries are built from the `source_commit` in `approval.json`; the Corresponding
Source is attached to newer releases as an engine source archive, and you can ask the maintainers for it
for any release.

## 11. Reporting a problem or feedback

Open **one issue per problem** in this repository: Issues, New issue, **Tester report**. The form asks for
the facts below. Issues are visible to the maintainers and to every invited tester, so write nothing in one
that you want kept private. For a quick question, or to compare notes with other testers, use the testers'
Discord server (your invitation names it); anything you want looked at and tracked belongs in an issue.
A report needs:

1. The release tag and the archive SHA-256 (from `approval.json`), and the profile you ran.
2. The exact command and its exit code and error token.
3. The output of `./launch.sh diagnostics`, after you have read it.
4. What you expected and what happened, and the steps to repeat it. For a bad answer, the request you sent
   (remove anything private), the `seed` if you set one, and the `finish_reason`.
5. What you checked and what you could not check on your machine.

Do not include raw logs, prompts or responses you do not want shared, credentials, or SSH keys. For a
security problem follow [SECURITY](../SECURITY.md) instead: never an issue and never a chat channel.

## 12. Glossary

| Term | Meaning |
| --- | --- |
| Pack | A sealed, hash-pinned runtime for one model: profile, launcher, locks, approval |
| Bundle | The runnable archive of a pack, including the binaries (a release asset) |
| Runtime id | The pack's name, for example `qwen38-27b-dflash2-exl3-4.00` |
| Profile | One approved way to run the pack: window, slots, topology, flags |
| Slot | One request being processed at the same time |
| Context window | The most tokens (prompt plus thinking plus answer) a request may use: `max_model_len` |
| Prefix cache | Saved state of earlier prompts reused for a shared beginning |
| Speculative decoding | The engine proposes several tokens and checks them at once; output is identical to normal decoding |
| Site file | A JSON file naming your machines and paths, needed for a two-node profile |
| Approval | `approval.json`: the record binding an archive to its source commit and hashes |

## 13. Quick reference

```text
./launch.sh --yes install-deps                       install system packages and a private Python environment
./launch.sh doctor                                   read-only host and bundle checks
./launch.sh print-command [--profile P]              show what would run, change nothing
./launch.sh --yes setup [--profile P] --accept-model-download   download and verify the model
./launch.sh verify [--profile P]                     re-verify the model without downloading
./launch.sh --yes start [--profile P]                run in the foreground until READY, then serve
./launch.sh status [--profile P]                     0 = ready, 24 = not ready
./launch.sh logs [--profile P] --lines N --no-follow bounded log tail
./launch.sh stop [--profile P]                       stop the server this bundle started
./launch.sh diagnostics [--profile P]                sanitized report to share
./launch.sh reset-state [--profile P]                only after proving nothing is running
```

`--yes` and `--allow-driver-change` go **before** the command. Options such as `--profile` go after it.
