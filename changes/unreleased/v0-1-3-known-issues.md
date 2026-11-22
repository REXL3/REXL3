## Known issues

Each runtime's `KNOWN_ISSUES.md` gives the full list, with what happens, who is affected and what to do. In summary:

- **Every runtime: NVIDIA driver.** The 580 branch at build 580.142 or later. Builds after 580.178.04 are accepted with `DRIVER_NOT_YET_QUALIFIED` but are not covered by this release's qualification evidence. Both driver packages must be the same build, on every node of a two-node profile. Another branch (570, 590), or a build below 580.142, is refused. If `install-deps` stops with `DRIVER_PACKAGE_UNAVAILABLE`, install any 580 build at or above 580.142 with the host package manager, reboot, and rerun `./launch.sh doctor`.
- **Two-node runtimes: NCCL.** `libnccl2` 2.30.7 or later on major version 2, the same version on both nodes. Upgrade both together, or hold the package on both.
- **Two-node runtimes: the HTTP endpoint is on the network and has no authentication.** The leader listens on its site-file address, and anyone who can reach the port can use it. Run it on a private network, or behind a firewall or an SSH tunnel.
- **Images are refused** under temporary exceptions:
  - DeepSeek-V4.1-Flash-1017 single-node and two-node, until 2026-12-31: the quantized pack answers a solid-colour probe wrongly.
  - Qwen3.8-Flash-Next, until 2027-01-31: the engine does not build its vision tower.
  - Ling-3.1-Flash-1019 is a text-only model.
  - CYBER-FROST-3.8 runs `--no-vision`.
- **Thinking off:**
  - GLM-5.3-Flash (single-node and two-node) always thinks; `enable_thinking: false` has no effect, and the response says so in the `X-Rexl3-Note` header. Lower it with `chat_template_kwargs: {"reasoning_effort": "low"}`.
  - Ling-3.1-Flash-1019 with thinking turned off puts reasoning text and a stray `</think>` into the answer. The default (thinking on) is not affected; this exception expires on 2026-12-31.
- **No thinking unless asked:** MiniCPM-V-4.7, Nemotron-3.5-Super-VL and Qwen3.8-Flash-Next do not think on a request that names no thinking setting, although their chat templates would. Send `chat_template_kwargs` to turn it on; each runtime's file gives the exact field.
- **Qwen3.8-Flash-Next, default choice:** after a prefix-cache hit that ends inside a 256-token page, MTP can accept fewer drafted tokens, so decoding can be slower. Replies are not affected. Continuing an earlier chat turn is such a hit, and the size of the slowdown is not measured. Send `{"prefix_cache": false}` to avoid it for one request (#177, open).
- **GLM-5.3-Flash single-node:** the prefix cache holds one long conversation at a time; a different full-window prompt evicts it.
- **Ling-3.1-Flash-1019 refuses to start at 262,144 tokens on a host short of free memory.** Start on an idle host (see the upgrade notes).
- **DeepSeek-V4.1:** at temperature 0 the model can repeat a phrase until `max_tokens`; keep the sampling defaults for long reasoning. `--kv-dtype int8` silently runs the `q4` cache.
- **Local model copies:** Qwen3.8-27B, MiniCPM-V-4.7 and the Qwen3.8-Flash-Next drafting choice have two packs (target and draft) and cannot use an existing local copy named in a site file; let the launcher download them.
- **Port 8889** is the default of the Qwen3.8-27B, MiniCPM-V-4.7 and Ling-3.1 runtimes. Run one at a time on a host, or set `http_port` in a site file.
- **Nemotron-3.5-Super-VL:** video-frame input is accepted but not part of the release contract (`capabilities.video` is `false`).
- **RED-SNOW-5.3-Flash single-node is not part of v0.1.3.** Its qualification found that the profile's prefix-cache size does not cover the full 262,144-token window. It is planned for a later release with a corrected profile. The two-node RED-SNOW-5.3-Flash profile is not affected.
