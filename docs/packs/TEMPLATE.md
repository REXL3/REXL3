# Pack sheet: `<family>/<model>`

> Template. Copy to `docs/packs/<model>.md`, replace every `<...>`, and keep every section (write
> "not stated" where the sealed files do not say). Delete this note. The rules are in
> [README](README.md).

Checked against model version `v<N>` in release `v<X.Y.Z>` (source commit `<12 hex>`, bundle asset
`<runtime-id>-v<N>-bundle.tar.gz`).

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `<runtime-id>` | `profile.json` |
| Versions | `<vN>` (latest), `<vM>` ... | `runtimes/<family>/<model>/LATEST` |
| Release tag | `v<X.Y.Z>` (the product release that contains `v<N>`) | `releases/v<X.Y.Z>/manifest.json` |
| Release assets | `<runtime-id>-v<N>-approval.json`, `<runtime-id>-v<N>-bundle.tar.gz` | the release page |
| What it serves | `<model family and size, quantization format and bit rate as published>` | `model.lock.json` |
| Model repository and revision | `<owner/repo>` at `<full revision>` | `model.lock.json` |
| Draft or second model | `<repository and revision, or "none">` | `model.lock.json` |
| Model licence | `<as stated in the lock, or "see the pack's own licence">` | `model.lock.json` |
| Download size | `<N files, X GiB>`; free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `<name>` (default) | `<single node / two nodes>` | `<N tokens>` | `<n>` | `<port>` | `<127.0.0.1, or the leader's address for two nodes>` | `<seconds>` after model verification |

Source: `profile.json`. A two-node profile also needs a site file (user guide section 4.6); say so here.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | `<yes/no>` | |
| Thinking (reasoning) | `<yes/no>` | Default when a request names no setting: `<on/off>`. Switch: `<enable_thinking; effort levels in chat_template_kwargs.reasoning_effort; whether top-level reasoning_effort is accepted>` |
| Tools | `<yes/no>` | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | `<yes/no>` | Inline data URI only |
| Prefix cache | `<yes/no>` | Automatic |
| Speculative decoding | `<yes/no>` | Automatic |

Source: `profile.json` `capabilities` and `KNOWN_ISSUES.md`. Everything not listed as supported is refused
(user guide section 5.10).

## 4. Sampling and output defaults

Defaults the server applies when a request does not set them: `<temperature, top_p, top_k, from the
pack's generation_config.json or the server defaults>`. Advice for this model: `<for example, minimum
max_tokens for a thinking model>`.

## 5. Behaviour to expect

`<Copy the points from KNOWN_ISSUES.md that a tester will notice, each with its source line.>`

## 6. Measured behaviour

| Quantity | Value | Hardware and release | Recorded in |
| --- | --- | --- | --- |
| `<generation speed, one stream>` | `<value>` | `<GB10, version>` | `<file>` |
| `<prompt speed>` | `<value or "not published">` | | |

## 7. Limits of this release

`<What the release does not do and what was not qualified, from KNOWN_ISSUES.md.>`

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/<family>/<model>/v<N>/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
