# MiniCPM-V-4.7-35B-A3B EXL3 6.08 bpw REXL3-SAGE with the DSpark draft: frozen configuration, not qualification

One normalized choice, `minicpmv47-single-262k`: one GB10, 262,144 context, 4 slots, speculative decoding with the DSpark draft pack
(`--spec auto` with `--draft`), prefix cache on (`--prefix-cache-mib 32768`), vision on. Capabilities: text, SSE, reasoning, tools, images,
prefix cache. Audio and constrained JSON response formats (`json_object`, `json_schema`) are refused by the server; video is not a claimed
capability (see below). There is no tensor-parallel choice. `qualification.state` is `unverified` and no measurement is transferred from the
earlier 6.08 bpw pack of this model.

## State: frozen, not qualified

`handoff.json` is `final` / `frozen`; `qualification.state` stays `unverified` until the release chain's receipts exist. The pack
`r0b0tlab/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw-REXL3-SAGE` is pinned at revision `25b7aa3361dd8e319c7503ebc002947ceb1fd290` (4 shards,
fingerprint `sha256:96eb01e7fdf81af2c1e248f81114dea2706c6383e63704774bf51fa85025458a`), the same values as the release requirements and the
engine's validated list.

`profile.json` `source.commit` names a commit on `main` that predates this entry, and `source.sha256` is the digest of
`docs/examples/cluster.minicpmv47-rexl3-sage-single.json` as this change adds it. The file exists on `main` only once this entry merges, so a
follow-up change points `source.commit` at that merge commit before the entry is sealed (the sealer reads the file at `source.commit`), as was
done for the Ling-3.1, Flash-Next and Nemotron entries.

## Where the arguments come from

The argv is normalized from `docs/examples/cluster.minicpmv47-rexl3-sage-single.json`, which is the serve line of the published MiniCPM-V-4.7
runtime (`--slots 4 --max-model-len 262144 --draft ... --prefix-cache on`, vision on). Changes made while normalizing:

- `--backend cuda`, `--queue-depth 64`, `--queue-timeout 300` and `--shutdown-grace 8` are dropped: each equals the server default and none is
  a serving-contract option (CONTRIBUTING section 6).
- `--spec auto` and `--prefix-cache on` are pinned as contract options (section 6.1). The checkpoint has no MTP head (`--spec mtp` is refused
  by name), so `auto` with `--draft` runs the DSpark drafter.
- `--prefix-cache-mib 32768` replaces the automatic cache. The automatic cache takes at most 8 GiB; with the draft loaded a pool page is
  7.5 MiB (target and draft planes) and an entry unit is one 61.4 MiB recurrent-state snapshot plus 8 pages, 121.4 MiB, so 8 GiB caches at most
  about 137,000 tokens and the qualifier's full-window warm repeat would miss. One full-window entry needs 15,540 MiB; 32,768 MiB holds two.
  The plan with the pin is about 72 GiB on a 121 GiB GB10 (the 6.08 bpw pack's plan was 40.4 GiB without the cache). To be confirmed by the
  first boot on the pack (`minicpm_prefix_plan`, `MemAvailable` at ready).
- The memory floor is the engine default (8 GiB); `REXL3_MEMORY_FLOOR_GIB` is not set, and no other `REXL3_` variable is.
- `--kv-dtype` is not passed (the flag selects only the DeepSeek and Nemotron caches); `--no-video` and `--no-vision` are not passed.
- `--queue-timeout` and `--request-timeout` stay at their defaults (300 s and 1,800 s). The 4 slots share one 262,144-token KV pool and
  admission is FIFO, so a request queued behind cold full-window prefills can wait: with the 6.08 bpw pack's measurements (cold 261,000-token
  prefill 137.5 s, 21.5 tok/s decode at that depth without the draft) one full-window request with an 8,192-token answer takes about 520 s, and
  three ahead about 1,560 s. The gate and the qualifier send their full-window requests one at a time; if a concurrent full-window workload is
  part of the contract, both timeouts need a pin derived from the measured rates of this pack, as the Flash-Next and Nemotron recipes do.

## Prefix cache: where the cuts are

The gated delta net layers' recurrent state cannot be cut back to an earlier position, so a request resumes only from a snapshot taken
inside the prefix it shares with an earlier prompt. The family runs the interactive cut policy (`CutPolicy::interactive()`, as the Qwen3.8,
GLM and Nemotron engines do): no cut below 128 tokens, the end cut 64 tokens clear of the end of the prompt, a new cut once a prompt has grown
128 tokens past the state the prefill already holds, and a shared cut (a prefix two prompts have in common) when it saves 256 more tokens.
So a second, different question about the same long document resumes inside the document, and a tool loop is cached from its second turn.

- An exact repeat of a prompt is served from the prompt's length less 64 tokens (`cached_tokens` = prompt − 64); prompts of 128 tokens or
  fewer are never cached. Live checks pass `--tail 64` to `tools/check_prefix_cache.py`.
- A cut never falls inside an image or a video frame group: it moves to the picture's start or is dropped.
- A growing conversation takes a snapshot slot (one entry unit, 121.4 MiB with the draft) per 128 tokens it adds instead of per 256.
- `REXL3_PREFIX_MIN_CUT`, `REXL3_PREFIX_TAIL`, `REXL3_PREFIX_MIN_GAP` and `REXL3_PREFIX_GHOST_GAIN` still override each value; the profile
  sets none of them.

## Launching: managed models only, no `--site model_dir`

The lock has two roles, `target` and `draft`. As for the Qwen3.8-27B entry, `runtimes/template-v1/scripts/runtime.py` accepts an existing
model copy only for a single-role lock, so the profile runs with the packs the launcher downloads itself (`@MODEL_DIR@` and `@DRAFT_DIR@`) and
cannot yet be pointed at existing local copies through `--site model_dir`. The qualification tools take both directories
(`lifecycle_driver.py --model-dir --draft-dir`, `gb10_baseline.py --model-dir --draft-dir`).

## Sampling defaults and thinking

- The pack ships `generation_config.json` with temperature 1.0, top_p 0.95 and top_k 20 (a copy of the base checkpoint's), and the server
  reads all three from it.
- The chat template opens a `<think>` block unless `enable_thinking` is false and has no effort variable. The profile passes neither
  `--default-thinking` nor `--default-reasoning-effort`, so the server default applies: a request that names nothing is served with
  `enable_thinking` false (no reasoning). A client that wants thinking sends `chat_template_kwargs: {"enable_thinking": true}`; the Q200 lane does
  so on every request (`evals/q200_minicpm`). Neither the pack nor the base model card recommends a default; `--default-thinking template`
  (the template's own default, thinking on, as the earlier MiniCPM-V-4.7 runtime served) is the alternative, and it would raise the gate's
  reply budgets.

## Vision and video limits

- Images: `image_url` parts with `data:` URIs (PNG or JPEG), up to 9 slices of 448 pixels plus a thumbnail, at most 4,096 tokens an image
  (`--max-image-tokens` default). Generated solid-colour images must be answered correctly and one image+text request must give a correct tool
  call (the vision gate).
- Video: frames only. A container (`data:video/mp4`) is refused by name; at most 32 frames (`--max-video-frames`) and 4,096 video tokens. Video
  is not a claimed capability (`capabilities.video` is `false`): the qualifier has only a refusal probe for video, which the container refusal
  answers.
- Audio: refused (the pack has no sound tower).

## Model identity

The lock pins the target pack at its Hugging Face revision (above) and the DSpark draft
`r0b0tlab/MiniCPM-V-4.7-35B-A3B-DSpark-EXL3-6.00bpw` at revision `e1ea1a3045277ea69a108b4d477983affae0f3c1`: its `model.safetensors` digest is
the LFS object id of that revision and the engine's pin for the draft fingerprint
`sha256:7731abc818a30cd7636af7d72de016d21371c3f5c5c46e5ab38121e602bdf5ab`; `config.json` and `quantization_config.json` were hashed from the
files of the same revision. The draft lists only the three files the engine reads; the repository's other documents are not downloaded. Only
`README.md`, `LICENSE` and `.gitattributes` may sit beside the locked files unlisted. The pair must be on the engine's validated list
(`VALIDATED_MINICPMV4_7`), so `--allow-unvalidated-pack` is never needed.

## Dependencies

`dependencies.lock.json` and `requirements-runtime.lock` are those of the Qwen3.8-27B entry (the DeepSeek runtime's set without `libnccl2`,
which the single-node engine does not load). They have not been rehearsed on a clean host for this runtime.
