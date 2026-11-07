# Layer-0 Mamba2 SSM fixture

`layer0_fixture.safetensors` (753,352 bytes, sha256 `06d125bcb114041d7bcb969335e4dfc80df2f912662632f63f2c3f7f46e02891`) is the torch
oracle for `crates/model/src/families/nemotron_h_omni/ssm_ref.rs`, checked by `crates/model/tests/nemotron_omni_ssm.rs`.

Contents:
- Six layer-0 parameters from the pack, bf16 and bit-exact: `conv1d.weight` [10240, 1, 4], `conv1d.bias`, `A_log`, `D`, `dt_bias`
  [128], `norm.weight` [8192]. Read through `model.safetensors.index.json` only, from the shard the index names (`model-00001-of-00019`).
- `gated_ref` F32 [16, 8192]: the torch output of the conv, the recurrence and the gated RMSNorm for 16 tokens. It stops before
  `out_proj`.
- `state_ref_heads` F32 [2, 128, 64]: the final recurrent state of heads 0 and 1.
- `conv_last` F32 [10240], `dt_last` and `g_last` F32 [128]: the last token's conv, dt and g.
- `input_checksum` I64 [3]: the sum of the generator's 24-bit integers for each input stream. The Rust test fails if its generator drifts.

The inputs are not stored. Both sides regenerate them: element `t * cols + j` of stream `s` (z=0, xBC=1, dt_raw=2) is
`k / 2^23 - 1`, where `k` is the top 24 bits of `splitmix64(20261010 + s << 48 + t * cols + j)`. Shapes are z [16, 8192], xBC
[16, 10240], dt_raw [16, 128].

Not in this fixture: `in_proj` and `out_proj`. Both are EXL3 (trellis) in the pack, and the fork dequantizes EXL3 only through CUDA
kernels. The input is the post-`in_proj` stream, and the output stops before `out_proj`.

The reference math is the fork's own, run as CPU torch (`exllamav3-dflash2` commit `44636f060774df81cad39c9928ffb3ee3f545be1`, also
recorded in the file metadata). `tools/gen_ssm_fixture.py` pulls these functions from the fork's source by AST, without importing
the CUDA extension: `mamba2.py:torch_recurrent_mamba2`, `gated_delta_net_fn/conv1d.py:causal_conv1d_update_function_torch`, and
`gated_rmsnorm.py:GatedRMSNorm.forward_torch`. The dt rule follows `mamba2_dt_op` in `gdn.cu`. Everything is fp32.

## Example regeneration procedure (CPU only)

The historical generator expects `~/projects/rexl3-nemotron3.5-super-vl-exl3-mtp` on its execution host. The generator reads only
the six tensors above, through the index, and never copies a shard. The host, address and user below are portable placeholders,
not original measured infrastructure; supply your own endpoint before running. No tensor fixture was changed in this privacy edit.
Original README SHA-256: `903886f877579187a24e29135135709e8c69ee5ed4d15b34e17ad022e6a4fc4b`.

```
ssh rexl3@192.0.2.4 'mkdir -p ~/ssm_fixture_work'
scp tools/gen_ssm_fixture.py rexl3@192.0.2.4:~/ssm_fixture_work/gen_ssm_fixture.py
ssh rexl3@192.0.2.4 'cd ~/ssm_fixture_work && CUDA_VISIBLE_DEVICES= ~/exllamav3-dflash2/.venv/bin/python -I gen_ssm_fixture.py --out ~/ssm_fixture_work/layer0_fixture.safetensors'
scp rexl3@192.0.2.4:~/ssm_fixture_work/layer0_fixture.safetensors testdata/nemotron_omni/ssm/layer0_fixture.safetensors
```

## Check (local)

```
REXL3_NO_CUDA=1 cargo test -p gb-model --lib nemotron_h_omni
REXL3_NO_CUDA=1 cargo test -p gb-model --test nemotron_omni_ssm -- --nocapture
```

The gate is max |rust - torch| / max |torch| <= 1e-4 for each tensor. `prefill_scan` must match the step loop to 1e-6 absolute.
