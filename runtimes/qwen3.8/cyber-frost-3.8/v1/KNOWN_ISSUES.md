# CYBER-FROST 3.8 SAGE REXL3 3.35 bpw: frozen configuration, not qualification

One profile, `cyberfrost-3.8-single-262k`: one GB10, a 262,144-token context,
2 slots, MTP speculative decoding, prefix cache on with a 12,288 MiB budget,
and vision off under the temporary exception recorded in the gate recipe.
Text, SSE, reasoning, tools and prefix caching are advertised. Images, audio,
video and constrained JSON response formats are refused.

The entry is final as configuration. `qualification.state` remains `unverified`;
configuration freeze is not a throughput, long-context or quality claim. Gate
results belong to the exact engine binary and release evidence, not this file.

## Pack identity

- Repository: `vcruz305/CYBER-FROST-3.8-SAGE-REXL3-3.35bpw`
- Revision: `33be4227ff69c85cbb94f940abe7f22ad3b504e1`
- Architecture: `Qwen4ExpForConditionalGeneration` (`qwen4_exp`)
- Fingerprint: `sha256:656af44a88db8434ab2ca36798097cde1416931da7ed7d3c73485937395ee450`
- Payload: nine indexed weight shards, the MTP mixer sidecar, the 6-bit n-gram
  table, tokenizer/configuration files and the published model documents/assets.
- License: Qwen Community License 1.0, supplied as the pack's `LICENSE`.

The fingerprint is registered as `CF335` / `CF335_FP` in
`crates/model/src/fingerprint.rs`, with all nine shard SHA-256/CRC-32 pins.
Every published LFS file was matched to the GB10 copy by direct-I/O SHA-256
before registration. The runtime boots without `--allow-unvalidated-pack`.
Unlike the older CYBER-FROST export, this published revision already contains
`mtp_hyper_connection_mixer_patch.safetensors`; no overlay is needed.

SAGE is our own mixed-precision quantization method for EXL3. The method is not published.

## Serving and memory contract

The effective launch is:

```
rexl3-serve --model PACK --slots 2 --max-model-len 262144 --spec mtp \
  --default-thinking template --default-reasoning-effort low \
  --prefix-cache on --prefix-cache-mib 12288 --queue-timeout 1500 \
  --served-model-name CYBER-FROST-3.8-SAGE-REXL3-3.35bpw --no-vision
```

with `REXL3_NGRAM_WARM=0`. The served name is the same in the fingerprint entry,
source cluster document, profile smoke request and gate recipe.

The pack template enables thinking. The profile follows it with
`--default-thinking template --default-reasoning-effort low`: an A/B of the
12-turn lookup loop found that the thinking-off line called `lookup(step=12)`
again after the 12th result (unchanged with cache or speculation disabled),
whereas template thinking with low effort returned all twelve tokens with
`finish_reason=stop`. This is a serving-contract requirement, not a quality or
oracle claim.

The n-gram table is 36.36 GiB. Cold operation leaves its mmap pages reclaimable
instead of warming the whole table while full-window KV and prefix-cache storage
are reserved. The family default for grouped-MoE prefill and the unconditional
compensated-fp16 HCG prefill contract are used; no prefill environment switch is
added. The boot log must name `prefill_contract=hcg-compensated-fp16`.

`CacheBudget::fit_pages_first` gives the 12,288 MiB pin, at two live slots,
24 snapshot slots and 1,428 pages of 256 tokens: **365,568 tokens of planned
reach**, above the 262,144-token window. This is planner arithmetic to be
confirmed by the boot's `flash_next_prefix_plan` and a real full-window replay.
It is not a claim that two full-window entries coexist.

## Shared KV pool and queueing

Flash-Next uses one KV pool capped at 262,144 tokens, shared by both slots. One
request may use the whole window. Two concurrent requests must fit their prompt
and output reservations within that pool; a second full-window request waits in
strict FIFO order. The flags do not provide 262,144 tokens per slot.

`--queue-timeout 1500` replaces the 300 s default, which can expire behind one
cold full-window prefill. It covers three full-window requests ahead and stays
below the 1,800 s request timeout so queue expiry retains its clean 503 and
`rexl3_queue_timeouts_total` behavior.

## Vision

The pack ships vision tensors, but this engine does not build the Flash-Next
vision tower (`unbuilt_vision()`). `--no-vision` is used under the maintainer's
temporary exception granted 2026-10-23 and expiring 2027-01-31, recorded in
`scripts/gates/recipes/cyberfrost-3.8-sage-rexl3-3.35.json`. Past that date the
runtime needs either a tested vision implementation or a same-contract
vision-on `does-not-fit` record.

## Validation boundaries

The full supported context is 262,144 tokens. No reduced-context result is a
substitute. A valid hardware receipt must cover boot without a pack-validation
override, drafted generation, prefix-cache replay, cached-versus-cold exactness,
chunked-versus-whole exactness, the baseline gate and its speed floors. Any
unrun or failed item remains unresolved rather than being inferred from another
pack or engine build.
