# Nemotron 3.5 Super VL: layer-1 latent-MoE fixture

`layer1_fixture.safetensors` (349 KB) is torch's reference for the latent-MoE block of text layer 1, the first `E` block of the
pattern. The Rust reference (`crates/model/src/families/nemotron_h_omni/moe_ref.rs`) and the test
`crates/model/tests/nemotron_omni_moe.rs` check against it.

## How it was made

    ~/exllamav3-dflash2/.venv/bin/python -I tools/gen_moe_fixture.py \
        --pack ~/projects/rexl3-nemotron3.5-super-vl-exl3-mtp --out layer1_fixture.safetensors

Run on node 4 with torch 2.14.0 (CPU). Input: `torch.randn` with seed 20261010, 4 tokens. Weights were read through
`model.safetensors.index.json` only, one tensor at a time, never globbed. The file's metadata records the pack bytes read.

## The repaired pack's fixture

`layer1_fixture_repaired.safetensors` (349 KB) is the same reference made from the repaired pack
(`rexl3-nemotron3.5-super-vl-exl3-mtp-repaired-20261010`, index sha256 `5c4938c5...0776`; `docs/NEMOTRON_REPAIR.md`), with the same
generator, seed and input (the generator reads only through the index, so it takes the overlay shard's copies):

    ~/exllamav3-dflash2/.venv/bin/python -I tools/gen_moe_fixture.py \
        --pack ~/projects/rexl3-nemotron3.5-super-vl-exl3-mtp-repaired-20261010 --out layer1_fixture_repaired.safetensors

Run on node 1 (torch CPU, under a minute). Its metadata carries `pack_index_sha256`. The repair re-encoded layer 1's shared-expert up
projection and routed experts 11 and 429. Compared with `layer1_fixture.safetensors`: `shared_out` differs by 7.7e-2 and `out` by
3.7e-2 (both relative to their max), which is the repair and is far outside the 1e-3 block gate; every other tensor is equal (the
input, the router's logits, indices and weights, `x_latent`, `routed_latent`, `routed_out`, the spot expert, the bf16 expert's
input and output, the four digests), because the router, fc1, fc2 and the 82 experts these four tokens pick are not among the
re-encoded matrices. `crates/cuda/tests/nemotron_moe.rs` chooses the fixture by the FNV-1a of the pack's index and compares the
repaired pack with this one under the same tolerances; any other pack is compared with the published pack's fixture.

## What is in it (all fp32 unless noted)

| tensor | shape | meaning |
| --- | --- | --- |
| `h` | [4, 4096] | the block input |
| `router_logits` | [4, 512] | `h @ gate.weight^T`, pre-sigmoid |
| `router_bias` | [512] | `e_score_correction_bias` |
| `route_indices` | [4, 22] int64 | torch's top-22 on `sigmoid + bias`, lower index first on a tie |
| `route_weights` | [4, 22] | unbiased sigmoid of the selected experts, normalized, times 5.0 |
| `x_latent` | [4, 1024] | fc1's output |
| `routed_latent` | [4, 1024] | the weighted routed expert outputs, summed per token (before fc2) |
| `routed_out` | [4, 4096] | fc2 applied to `routed_latent` |
| `shared_out` | [4, 4096] | the shared expert on `h` |
| `out` | [4, 4096] | `routed_out + shared_out`, the block's output |
| `spot_exl3_expert` | [1] int64 | the EXL3 expert that layer 1 selects for token 0 (82 distinct experts are selected for the 4 tokens) |
| `spot_exl3_input`, `spot_exl3_output` | [1, 1024] | that expert on token 0's latent |
| `spot_bf16_input`, `spot_bf16_output` | [4, 1024] | the bf16 expert (layer 26, expert 111) on layer 1's latents |
| `digest_*` | [1] uint64 | fnv1a of the fp16 bits of a decoded `W_q` (little-endian), for fc1, fc2 and the spot expert's up and down |

The router, the selection and the weights are recorded as torch computed them. The minimum gap between the 22nd and 23rd
selection values is 9.6e-4 (in the metadata), far above fp32 rounding, so the selection is stable across implementations.

## Deviations from the brief

- **The EXL3 expert is not stored dequantized to fp16.** The brief asked for that under a 2 MB cap, and the cap cannot hold any real
  projection. In MiB: one routed projection at fp16 (`[2688, 1024]`) is 5.25; one expert's raw 4-bit trellis (up and down) is 2.6;
  the router (bf16) is 4; fc1 and fc2 (fp16) are 8 each; one shared-expert projection (fp16) is 42. The fixture therefore holds
  activations and torch's expectations only. The weights stay in the pack.
- **The pack-backed test needs the pack.** `crates/model/tests/nemotron_omni_moe.rs` reads the weights through the pack's index when
  `REXL3_NEMO_OMNI_DIR` names the pack directory (the same variable as `nemotron_omni_pack.rs`). Without it, only the router check on
  torch's logits runs. The route check does not need the pack.
- **The bf16 expert is not on layer 1.** Layer 1 has no bf16 expert. The measured bf16 pairs are (26, 111), (39, 304) and (81, 461), so
  the bf16 case uses (26, 111) on layer 1's latents. This tests the expert's arithmetic, not its routing.
- **The torch side decodes EXL3 with a numpy port of `gb_exl3::unpack`.** The fork's own EXL3 reconstruct is CUDA-only, and the brief
  asked for CPU jobs. The port follows `crates/exl3/src/lib.rs` (the mul1 codebook, `stream_ends`, `tile_positions`, `states`). The
  digests check that the Rust decoder (`gb_exl3::unpack`) and the port produce the same bits for four projections.

## Precision

Everything in the reference is fp32, including the router matmul and the sigmoid. The fork's production path keeps fp16 at several
points: the router logits and weights, the latent projection's output, and the expert intermediates. This fixture does not model
those, so it measures the arithmetic, not the production rounding.
