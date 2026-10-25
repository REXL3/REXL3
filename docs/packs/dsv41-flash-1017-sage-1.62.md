# Pack sheet: `dsv41-flash-1017-sage-1.62`

Checked against release `prod/dsv41-flash-1017-sage-1.62/v1` (source commit `37bb034b9995`, archive
`dsv41-flash-1017-sage-1.62-1.tar.gz`), the contents of `runtimes/dsv41-flash-1017-sage-1.62/v1/`.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `dsv41-flash-1017-sage-1.62` | `profile.json` |
| Versions | `v1` (latest) | `runtimes/dsv41-flash-1017-sage-1.62/` |
| Release tag | `prod/dsv41-flash-1017-sage-1.62/v1` | `approval.json` |
| What it serves | DeepSeek-V4.1-Flash-1017 in EXL3 format at 1.62 bits per weight (the pack named `DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw`) | `model.lock.json` |
| Model repository and revision | `vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw` at `c852150e06f5e5ee8c4c14f9cd1a9afc6bb21960` | `model.lock.json` |
| Draft or second model | none | `model.lock.json` |
| Model licence | See the upstream model card and the `LICENSE` in the pack (the lock does not restate it) | `model.lock.json` |
| Download size | 23 files, 304.16 GiB; free space needed: the remaining bytes plus a reserve | `model.lock.json` |

This is a **legacy-format** tree (the first release). It has no source archive attached; ask the
maintainers for the Corresponding Source if you need it ([user guide](../USER_GUIDE.md) section 10).

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `dsv41-single` (the only one) | single node | 32,768 tokens (`--max-model-len 32768`) | 1 | 8940 | `127.0.0.1` | 300 s after model verification |

Source: `profile.json`. The profile runs a 4-bit compressed KV cache (`--kv-dtype q4`), speculative
decoding (`--spec auto`) and the prefix cache (`--prefix-cache on`), with the vision tower off
(`--no-vision`). There is no two-node profile in this release: the bundle refuses remote ranks.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | On by default for this release (`KNOWN_ISSUES.md`). Switch it off with `"chat_template_kwargs": {"enable_thinking": false}`, or lower it with `reasoning_effort` |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | **no** | The vision tower is not loaded; image parts are refused |
| Video and audio | no | Refused |
| Constrained JSON (`response_format`) | no | Refused; structured prompting works |
| Prefix cache | yes | Automatic. With under about 2 GiB of free memory the server boots with the cache off instead of failing |
| Speculative decoding | yes | Automatic |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`.

## 4. Sampling and output defaults

The pack ships no `generation_config.json`, so a request that sets nothing gets the server defaults:
temperature 1.0, top_p 0.95, top_k 20. Because the model thinks by default, give every request room: a
very small `max_tokens` can end inside the thinking and return empty `content` with `finish_reason:
length`. Use a few thousand tokens for general questions.

## 5. Behaviour to expect

All from `KNOWN_ISSUES.md` of this release:

- **One stream gives the best speed.** Running requests together adds a little total throughput but each
  stream gets slower. This profile has one slot, so concurrent requests queue.
- **Single-node serving is loopback only.** A non-loopback address is rejected by the launcher.
- **Reasoning is on by default**, as above.
- **The prefix cache needs free memory**, as in section 3.

## 6. Measured behaviour

| Quantity | Value | Hardware and release | Recorded in |
| --- | --- | --- | --- |
| Generation speed, one stream | 34.8 tokens per second | The development GB10 box, this release's configuration | `KNOWN_ISSUES.md` |
| Total throughput with 2 and 4 concurrent streams | 42.6 and 43.1 tokens per second (each stream 21.3 and 10.8) | The same | `KNOWN_ISSUES.md` |
| Prompt speed | not published for this release | | |

These are the release's own measurements on its test machine. Yours can differ with memory, other load
and the prompt.

## 7. Limits of this release

Not advertised: two-node tensor parallel, images, video, audio and constrained-JSON response formats.
Not qualified by this release: dependency installation on a clean host, systemd and cgroup supervision,
running the unpacked bundle without a compiler present, and clean-host rehearsal (`KNOWN_ISSUES.md`,
last section). Read that file in full before relying on the pack.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/dsv41-flash-1017-sage-1.62/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
