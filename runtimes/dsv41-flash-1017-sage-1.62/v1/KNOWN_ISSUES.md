# Known issues — DSV4.1-Flash SAGE EXL3 1.62 bpw (single node)

## Not advertised in this candidate

- **Two-node tensor parallel is refused by this bundle.** `runtimes/template-v1` refuses remote
  ranks until a qualified adapter attests complete remote payload/sidecar identity and scrubs
  the remote environment (`CLUSTER_REMOTE_IDENTITY_UNSUPPORTED`). The 2x GB10 profile measured
  on the development cluster (RoCE all-gather, NCCL 2.30.7, 15,360/15,360 experts resident per
  rank) is **not** part of this candidate. Do not add cluster JSON to enable it; that is a
  launcher adapter and qualification task, not a configuration change.
- **Images, video and audio are refused** (`images`, `video`, `audio` = false). The profile
  boots with `--no-vision`, so the vision tower is never bound; the TP=2 development profile is
  where image input was qualified.
- **`json_schema` / `json_object` response formats are refused** by the server
  (`response_format supports only {"type":"text"}`). Structured prompting is supported;
  constrained-decoding response formats are not.
- **Single-node serving is loopback only** (`127.0.0.1`). Binding a non-loopback address in a
  single-node choice is rejected by the launcher; remote access needs a reviewed adapter.

## Behaviour testers should expect

- **Concurrency adds capacity, not throughput.** Measured on the development box: aggregate
  34.8 -> 42.6 -> 43.1 tok/s at C=1/2/4 while per-stream falls 34.8 -> 21.3 -> 10.8. The routed
  block pays each touched expert's weight once per step (6-of-384 routing), so a batched row set
  disperses across experts instead of sharing reads. Use one stream per box for throughput.
- **A `--slots 1` boot queues concurrent requests**; do not read its C>1 numbers as batching.
- **Reasoning is on by default.** DSV4.1-Flash is trained with a thinking budget; a very small
  `max_tokens` can land inside the reasoning span and return empty content. Give the reply room
  or lower the reasoning effort explicitly.
- **Prefix-cache snapshots need free memory.** With less than ~2 GiB available the boot runs with
  the prefix cache off rather than failing.

## Not qualified by this candidate

Dependency installation from a clean host, systemd/cgroup supervision, the unpacked-artifact run
without a compiler present, and clean-host rehearsal are hardware-acceptance steps recorded
separately. A successful build is not a qualification verdict.
