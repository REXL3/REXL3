# Troubleshooting and exit codes

Start with `./launch.sh doctor` for read-only software/platform/integrity diagnosis.
It cannot establish runtime readiness or validate a selected site's device/port
ownership. Use the **same bundle, profile and site** as the foreground start for
status/logs/stop. Never bypass a refusal to produce a green screenshot.

## Exit semantics

These are launcher/adapter exit classes; read the accompanying error token.
An external shell/tool failure can additionally preserve its native exit code.

| Exit | Meaning and next action |
| --- | --- |
| 0 | The requested operation succeeded. Only `status` 0 means verified ready; `stop` may return 0 when already stopped. `doctor` 0 is static diagnosis, not inference. |
| 2 | Usage, unsupported option/profile/site, incomplete profile or invalid data contract. Check `--help` and the exact release inputs. |
| 20 | Missing/unusable dependencies, approval/privilege/installer/supervision failure or reboot required; also missing/declined model-download consent. Follow the error token, not a broad reinstall. |
| 21 | Unsupported platform/GPU or incompatible inherited environment. Use the supported host and remove unapproved tuning/loader overrides. |
| 22 | Ownership/resource conflict: busy GPU/port/lock, insufficient model disk, unsafe state/path or stale/reused process identity. Coordinate with the owner. |
| 23 | Bundle/native/model integrity or model acquisition/verification failed. Preserve evidence; never rewrite hashes to accept it. |
| 24 | Runtime operation/readiness/smoke failure, no owned log, stop timeout, or `status`: not ready (including stopped, preparing, starting or failed). |
| 130 | Interrupted operation where the adapter reports interruption. A normally drained foreground start may instead return 0. Inspect cleanup/state. |

`start` remains foreground while serving; do not wait for it to exit to decide
readiness. The bounded README wait uses status, rejects errors other than 24 and
fails with 24 at its deadline. A status poll timing out is not evidence of READY.
Never turn a failing wait into success with `|| true` or a successful final echo.

## Recover without changing identity

| Failure/token | Required action |
| --- | --- |
| Archive checksum/size mismatch | Do not extract or execute. Download the exact asset again from the trusted release; compare the sibling approval. |
| `PROFILE_NOT_READY` | The bundle/handoff is incomplete. Obtain the finalized reviewed archive and real native metadata; do not invent a lock. |
| `UNSUPPORTED_OS`, `UNSUPPORTED_ARCH`, `UNSUPPORTED_GPU` | Use the release's Ubuntu 24.04 ARM64 GB10 platform and memory requirements; another host needs a reviewed adapter/profile. |
| `NATIVE_VERSION_MISMATCH` | Compare `release.lock.json` with native output locally. Exact driver/runtime/cuBLAS fields are intentional; newer is not automatically qualified. Do not edit the lock. |
| Missing/unavailable package or wheel | Resolve signed repository/network/policy availability with the administrator; rerun `./launch.sh --yes install-deps`. Do not substitute a floating version, global pip or whole-OS upgrade. |
| Approval refused / missing sudo | Stop. Authorize required fixed supervision/package operations interactively with the host administrator, not by running the whole launcher as root. |
| `REBOOT_REQUIRED` / driver maintenance | Coordinate maintenance with the GPU owner. Routine `--yes` does not authorize driver changes or reboot; the launcher never reboots. |
| Dependency supervision unavailable / drain pending | Require systemd 255+, cgroup v2 and reachable system/login user managers. If a transaction was admitted, recover its exact owned unit with the administrator; retain ownership until verified drain. Do not force-kill the controller to bypass it. |
| `MODEL_DOWNLOAD_APPROVAL_REQUIRED` | For a single-node managed cache, review pinned size/destination, then use `--accept-model-download` after setup/start. This consent is separate from `--yes`; it does not provision TP ranks or repair external paths. |
| `MODEL_DOWNLOAD_FAILED` / offline access | Restore access and rerun single-node setup at the same pin to resume managed missing files. Do not change the model repository/quantization or expose the HF token. |
| `MODEL_INTEGRITY` / fingerprint/index mismatch | Stop. Inspect only the named model data locally. Preserve corrupt/extra content for diagnosis; no automatic overwrite or arbitrary alternate model is allowed. External single-node and per-rank `model_dir` sources remain read-only. |
| `EXISTING_MODEL_IDENTITY_FAILED: <CODE>` | An external `model_dir` failed read-only admission and `<CODE>` says which check (`UNLISTED_MODEL_DIRECTORY`, `UNLISTED_MODEL_PAYLOAD`, `PAYLOAD_HASH_MISMATCH`, `NONREGULAR_OR_LINKED_FILE`, `UNSAFE_*`). The launcher never edits that directory. Remove the unlisted entry yourself, or point `model_dir` at an exact copy of the locked revision. Only `.cache/huggingface/` (download bookkeeping) is tolerated. |
| `INSUFFICIENT_MODEL_DISK` | Free space deliberately or choose a reviewed matching model binding. Required free download space is remaining locked bytes plus reserve; partial fragments already consume disk. |
| `PORT_OCCUPIED_OR_UNAVAILABLE`, `GPU_LOCK_BUSY`, `GPU_BUSY_OR_IDLE_PROBE_FAILED` | Coordinate with the workload owner. A healthy existing listener cannot satisfy your invocation's readiness. Never broad-`pkill`, evict a device user, or unlink a lock to proceed. |
| `UNAPPROVED_SITE_FIELD`, `UNSUPPORTED_SITE_OVERRIDE`, cluster identity refusal | Use only the selected profile's typed, implemented site fields and reviewed normalized cluster schema. Older launchers cannot be upgraded by editing profile JSON or bypassing an adapter. |
| `EXPLICIT_TWO_NODE_SITE_REQUIRED` | Attested TP requires `--site` with both `nodes.leader` and `nodes.follower`, including their required read-only `model_dir` and `binary_dir` paths. An omitted or empty site exits 2; there is no implicit managed-cache placement. See SETUP. |
| `SITE_FABRIC_INTERFACE_REQUIRED`, `SITE_HCA_REQUIRED`, `SITE_TP_PORT_REQUIRED`, `SITE_FABRIC_INTERFACE_PLACEHOLDER`, `SITE_HCA_PLACEHOLDER` | The two-node site omitted a network field or still holds the `REPLACE_WITH_...` example names. Set `fabric_interface`, `hca` (one or more `device:port` rails) and `tp_port` for your real network in the site file; the sealed example network is not a default. Exit 2, nothing was started. |
| `RANK_IDENTITY_FAILED`, `RANK_IDENTITY_RESPONSE_INVALID` | Stop before rank launch. Each rank needs the exact matching native binaries, complete locked payload and dependencies pre-staged, plus trusted SSH access. There is no automatic remote download/install/repair, even with `--accept-model-download`. Resolve the named rank's identity/trust failure with its operator; do not chmod external data as an automatic fix. A trailing code such as `RANK_IDENTITY_FAILED: 1 (UNLISTED_MODEL_DIRECTORY)` is the rank's own refusal reason (an extra directory or file in that rank's `model_dir`); a `.cache/huggingface/` left by `hf download --local-dir` is tolerated, anything else must be removed by the operator. |
| `READINESS_TIMEOUT`, smoke/model identity failure | Inspect local owned logs and sanitized diagnostics; compare expected served model/effective context and the profile's launch deadline. Let owned cleanup finish. Do not expand sealed timeouts blindly or declare success from `/health` alone. |
| `RUNTIME_BUSY` / `ORPHANED_CLUSTER_REFUSED` after the launcher was killed or its session dropped without cleanup | Run `stop` (then `start` again) with the original bundle/profile/site: when the launcher process is gone, `stop` and `reset-state` terminate the recorded server (or run the native cluster `down`) only if its PID and start identity still match the record. |
| `STALE_OR_REUSED_PID_REFUSED`, `INSTANCE_PROFILE_OR_RELEASE_MISMATCH` | Do not signal a recorded PID. Use the original bundle/profile/site to inspect/stop its owner. Only after no owned processes remain may guarded `reset-state` clear instance/stop-request records. |
| `STOP_TIMEOUT`, cluster cleanup failure | Inspect exact owned state/controller and ask the administrator; a refused cleanup is not stopped. No unsafe external escalation is attempted by the wrapper. |
| HTTP 400 / unsupported field / over-window | Inspect JSON error and `/v1/models`. Keep within the effective context including completion budget; remove unsupported audio/video/constrained-JSON or forced-tool fields. |
| Client connects to the wrong host/port | Export the same `RUNTIME_ROOT`, `PROFILE` and `SITE` as the launcher. The README client derives its URL from real `print-command` resolution (TP uses the leader address). Unset a stale `API_BASE`; keep it only for a deliberate route to that same instance. Do not treat configuration resolution as owned readiness. |
| `finish_reason: length`, empty content or incomplete SSE | Treat as incomplete, not a pass. Reasoning can consume budget; inspect `reasoning_content`, token usage and `[DONE]`. Adjust the request within the declared context, not the sealed serving profile. |
| Tool turn returns no calls | `auto` permits a normal answer. For the documented tool smoke this is a failure to exercise the tool, not permission to synthesize a call/result. Never execute arbitrary model-supplied code. |

Full model verification can take significant time. The readiness deadline begins
at native launch, not at the start of a dependency install/download. A longer
operator waiting budget does not alter the profile's sealed native timeout or
qualify a different context/concurrency. No fixed cold-load or performance promise
is made here.

## Logs, diagnostics and reporting

```bash
./launch.sh status --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh logs --profile "$PROFILE" "${SITE_ARGS[@]}" --lines 80 --no-follow
./launch.sh diagnostics --profile "$PROFILE" "${SITE_ARGS[@]}"
./launch.sh stop --profile "$PROFILE" "${SITE_ARGS[@]}"
# Only after proving every previous owned process is absent:
./launch.sh reset-state --profile "$PROFILE" "${SITE_ARGS[@]}"
```

Logs default to a bounded tail (80 lines), not follow. Single-node `--follow` is
an explicit local option. Cluster logs support `--rank N` and bounded
`--no-follow`; do not assume cluster follow is supported. `--lines` accepts 1–1000.
Raw server/installer output can include private filesystem paths or application
content. Never post raw logs, prompts, responses, environment dumps, credentials,
SSH keys, sudo input or model payloads automatically.

`diagnostics` writes/prints an **allowlisted JSON** report with runtime/profile,
profile/model/release/site digests, phase, state-match flag, verified bundle
identity, model revision, production commit and optional server exit code. It
omits raw logs and sensitive environment/request data. Its `bundle_integrity`
field is not a fresh model-payload or inference qualification claim, and its phase
is recorded state rather than a replacement for an owned `status` probe. Even a
sanitized report deserves operator review before sharing.

Include the exact release tag/archive SHA-256, selected profile and sanitized
site-field names, exit code/error token, steps to reproduce and the reviewed
diagnostics JSON. State which checks actually ran and which were blocked; do not
paste historical metrics as results from this host. Consult `KNOWN_ISSUES.md` and
that release's sibling approval for qualification limits. See SETUP for tested
update/rollback sequencing; preserve old archives and model data.
