# Nemotron 3.5 Super VL fixtures

- `config.json`: the published pack's `config.json`, unchanged.
- `chat_template.jinja`: the published pack's chat template, unchanged.
- `model.safetensors.index.json`: a **trimmed** index. It keeps only the expert `down_proj.trellis` key of each MoE layer, which is
  what the family's probe checks to prove the EXL3 format (`families/nemotron_h_omni/mod.rs`). The full index has 166,757 entries and
  is not committed. The full pack is checked by the env-gated test `crates/model/tests/nemotron_omni_pack.rs` (`REXL3_NEMO_OMNI_DIR`).
