# REXL3 v0.1.2 release notes

## Highlights

- Deduplicated packaging over the nine unchanged `v0.1.1` runtime seals on engine `75031112`, reducing release assets from 47 to 39.
- Corrected third-party license attributions and notice files across runtime templates (#164, #165, #166).
- Unified single-PR release promotion and `uploads.github.com` asset uploads (#154, #163, #167).
- Hardened `cut_release.py` immutable-releases enforcement (#162).

## Supported models

| Model | Architecture | Pack | Context | Speculative decoding | Prefix cache | Vision | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.8/qwen3.8-27b | Qwen3_5ForConditionalGeneration | EXL3 4.00 bpw + DFlash2 4.00 bpw | 262,144 (single-node) | DFlash2 (k=16, auto) | on | on (image) | qwen38-27b-dflash2-exl3-4.00-v4-evidence.tar.gz |
| qwen3.8/qwen3.8-flash-next | Qwen4ExpForConditionalGeneration | EXL3 3.05 bpw (+ DFlash2 4.00 bpw draft, opt-in profile) | 262,144 (single-node) | MTP (auto); DFlash2 draft in qwen38-flash-next-single-262k-draft | on | off (owner exception through 2027-01-31) | qwen38-flash-next-exl3-3.05-v3-evidence.tar.gz |
| glm-5.3/glm-5.3-flash | Glm5NextForConditionalGeneration | EXL3 2.25 bpw | 262,144 (single-node) | MTP (auto) | on | on (image) | glm53-flash-sage-2.25-v2-evidence.tar.gz |
| glm-5.3/glm-5.3-flash-tp2 | Glm5NextForConditionalGeneration | EXL3 2.25 bpw | 1,048,576 (two-node) | MTP (auto) | on | on (image) | glm53-flash-tp2-sage-2.25-v1-evidence.tar.gz |
| glm-5.3/red-snow-5.3-flash-tp2 | Glm5NextForConditionalGeneration | EXL3 4.91 bpw | 524,288 (two-node) | MTP (auto) | on (pool-page) | on (image) | redsnow-5.3-flash-tp2-sage-4.91-v1-evidence.tar.gz |
| nemotron-3.5/nemotron-3.5-super-vl | NemotronH_Omni_Reasoning_V3 | EXL3 3.85 bpw | 1,048,576 (single-node) | MTP (auto) | on | on (image) | nemotron35-super-vl-sage-3.85-v2-evidence.tar.gz |
| minicpm-v-4.7/minicpm-v-4.7-35b-a3b | MiniCPMV4_7ForConditionalGeneration | EXL3 6.08 bpw + DSpark 6.00 bpw | 262,144 (single-node) | DSpark draft (auto) | on | on (image) | minicpmv47-35b-a3b-rexl3-sage-6.08-v1-evidence.tar.gz |
| deepseek-v4.1/deepseek-v4.1-flash-1017-tp2 | DeepseekV41ForCausalLM | EXL3 1.55 bpw | 1,048,576 (two-node, dsv41-tp2-1m) | MTP (auto) | on | off (owner exception through 2026-12-31) | dsv41-flash-1017-sage-rexl3-1.55-v1-evidence.tar.gz |
| ling-3.1/ling-3.1-flash-1019 | BailingMoeV3ForCausalLM | EXL3 1.51 bpw | 262,144 (single-node) | MTP (mtp) | on (disk-backed) | off (text-only model; owner exception through 2027-03-31) | ling31-flash-1019-sage-1.51-v1-evidence.tar.gz |

## New features and improvements

- Publish shared engine corresponding source once while retaining per-seal digest and size validation.
- Support co-author trailers on promotion commits and document single-PR promotion and local-validation rules (#163).
- Accept new approval anchors only from release PRs with matching manifest entries (#163).
- Correct third-party license attributions and notice files across runtime templates (#164, #165, #166).

## Bug fixes

- Require immutable releases to report `enabled` as boolean `true` before cutting a release (#162).
- Allow initial promotion of new registry versions with an authenticated `--new-anchor` and load modular registries correctly (#163).
- Bind registry modules to their expected paths and prevent approval-only PRs from changing the verification workflow (#163).
- Correct multi-model promotion commands to reuse the protected `main` base (#163).
- Upload release assets through `uploads.github.com` (#167).

## Breaking changes and upgrade notes

None.

## Known issues

None.

## Assets and checksums

- `rexl3-cluster-0.1.1-75031112-linux-aarch64-sm121` sha256 `623d4a65d75d3da53f68dcb637587a494ecdc7d0cefc3d4c2d5586364a291f71`
- `rexl3-serve-0.1.1-75031112-linux-aarch64-sm121` sha256 `4fc2cb440b9e668ba07618faabd23c213feb2a28968370950fa98f0b0d165fe5`
- `rexl3-engine-source-0.1.1-75031112.tar.gz` sha256 `f6280178d6229a72fceb89c298658cb12e9738de919c356cfc1c0b1be28826b4` (12313591 bytes): engine corresponding source, published once for every model whose sealed approval names this digest
- qwen3.8/qwen3.8-27b v4: approval sha256 `d17dda427fbf9ba6e2b9b03706b4f8a366c627a894cefdbdb8efa664b629167f`
- qwen3.8/qwen3.8-flash-next v3: approval sha256 `bc2371eb5b2e8021b281b6082f89986ea25285b3750ea375b5a3012cd981a8b7`
- glm-5.3/glm-5.3-flash v2: approval sha256 `7dd917e144bd94a8be40d7d19017ddb191833e506545fbf84f4afba1bb52a1cf`
- glm-5.3/glm-5.3-flash-tp2 v1: approval sha256 `32185a603fbc745862854394ae668a089be7d914c19f6aeb4bfe25a7ef32d8ef`
- glm-5.3/red-snow-5.3-flash-tp2 v1: approval sha256 `ec30e53e48079899bbf2047c81213f6f3c444d99cfd00f3789cb6defa1bdeb68`
- nemotron-3.5/nemotron-3.5-super-vl v2: approval sha256 `bdc5d82c9ca25c447edf428dd6c216312f8f9df740474213f568625dfaf310a3`
- minicpm-v-4.7/minicpm-v-4.7-35b-a3b v1: approval sha256 `6c5726edbc5cd6b3debef627758aa68fbf3c177d3d70e338d9afad3c6ac0c1a5`
- deepseek-v4.1/deepseek-v4.1-flash-1017-tp2 v1: approval sha256 `35758d241246f14456abc9c357da68213b72e9048b3f03979def5e5733956adb`
- ling-3.1/ling-3.1-flash-1019 v1: approval sha256 `0842a29237d60ed80b20260b8ec8dca8a6f18753c5aad9a0c155e2f68958ef3a`

## Provenance

- Engine dev commit `75031112ba1279b6da97d5604be5160648cc9f2c` tree `519678470f283dd5d39281d7d90df8bbc31999e8`; dev engine release tag `engine-v0.1.1-75031112`.
- All models are sealed from the same dev engine release commit.
