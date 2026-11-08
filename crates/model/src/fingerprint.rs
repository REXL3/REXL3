//! Pack fingerprints: which exact quantization this server was validated on.
//!
//! A fingerprint is SHA-256 over a canonical description of a pack: its `config.json` with object keys sorted, then every
//! EXL3 group (prefix, bits, codebook, K, N, scale kinds, bias) and every plain tensor (name, dtype, shape), in name order.
//! It reads headers only. Two packs with the same fingerprint present the same shapes, bit widths and codebooks to the
//! kernels; a re-download, a different shard split or a changed README does not change it, while a different quant, a
//! changed layer width or an edited model config does.
//!
//! This release serves the packs on its validated list only: the (target, draft) pairs of [`VALIDATED`] (Qwen) and
//! [`VALIDATED_MINICPMV4_7`], and the packs validated without a draft of [`VALIDATED_TEXT`] (GLM, MiMo, and the three Nemotron 3.5 Super VL
//! packs of v0.1.2), the shards of the pinned packs in [`VALIDATED_SHARDS`]; anything else is refused unless the operator passes
//! `--allow-unvalidated-pack`, and is then reported as unvalidated. A fingerprint names a pack's form, not its bytes (a matrix re-encoded at
//! the same width keeps it): `--verify-pack` checks the bytes against the pinned CRC-32s.

use gb_exl3::checkpoint::Checkpoint;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Version of the canonical form; bump it if the description below changes.
const FORM: &str = "rexl3-pack-fingerprint-v1";

/// A validated (target, draft) pair: (name, target fingerprint, draft fingerprint).
pub type ValidatedPair = (&'static str, &'static str, &'static str);
/// A pack validated without a draft: (name, target fingerprint).
pub type ValidatedText = (&'static str, &'static str);

/// Packs this release was validated on, as (target, draft) pairs: the Qwen pair (the MiniCPM pairs are [`VALIDATED_MINICPMV4_7`]). The
/// Nemotron packs carry their MTP head inside the pack and are served without `--draft`, so they are entries of [`VALIDATED_TEXT`]. An entry
/// is one tuple here plus its shards in [`VALIDATED_SHARDS`] (`rexl3-serve --print-fingerprints` and `--print-crc32` print both).
pub const VALIDATED: &[ValidatedPair] = &[(
    "Qwen3.8-27B-EXL3-4.00bpw + DFlash2",
    "sha256:967c3be941ce25284d4ddaaaed02cd3cda5e940ffa622400ed792ffc7a367940",
    "sha256:3b47e03a4d1e15615e5340baf0fa5e5ffbb0ab15f2b25b069dc25132a18c687d",
)];

/// MiniCPM-V-4.7-35B-A3B packs this release was validated on: (name, target fingerprint, DSpark draft fingerprint). The draft column is the
/// draft PACK's fingerprint, as for every family (`serve::check_packs`); `SpecRig::identity` is the prefix cache's namespace and never an
/// entry here. One entry a shipped pair, each pack's shards pinned in [`VALIDATED_SHARDS`]: `scripts/minicpmv47_fingerprint.sh PACK DRAFT`
/// prints both blocks to paste (section 16 of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`). A pack with no entry is refused unless the operator
/// passes `--allow-unvalidated-pack`, and `--print-fingerprints` says `validated no`.
pub const VALIDATED_MINICPMV4_7: &[ValidatedPair] = &[(
    "MiniCPM-V-4.7-35B-A3B-EXL3 q3-6.00bpw (6.08 bpw) + DSpark-EXL3-6.00bpw",
    "sha256:9bc579ceb1ae00bd016922d354e340bb395e9da52438cc226d1c898d5e63d03f",
    "sha256:7731abc818a30cd7636af7d72de016d21371c3f5c5c46e5ab38121e602bdf5ab",
)];

/// Our Nemotron 3.5 Super VL pack with the 4-bit MTP head (v0.1.1's qualified pack): Hugging Face weights revision e3857098, index sha256
/// eb0f1ada... (32 shards).
pub const NEMOTRON_H4: &str = "r0b0tlab/rexl3-nemotron3.5-super-vl-exl3-4.22bpw-mtp @e3857098 (4-bit MTP head)";
/// The same repository's v0.1.0 revision, the bf16 MTP head (the release gate's pack): revision b546ce51, index sha256 d2c82b14...
/// (33 shards).
pub const NEMOTRON_BF16_HEAD: &str = "r0b0tlab/rexl3-nemotron3.5-super-vl-exl3-4.22bpw-mtp @b546ce51 (bf16 MTP head, v0.1.0)";
/// The mixed-bit-rate (MixedK) pack of v0.1.2: revision 3963815246eda892a4c3b88e18d0d1c39288c8e0, index sha256 75291cea... (8 shards;
/// experts at K 2 to 6, the other trunk linears K 4 to 8, an EXL3 lm_head and MTP head, F16 router gates and latent projections).
pub const NEMOTRON_MIXEDK: &str = "vcruz305/Nemotron-Super-VL-EXL3-3.85bpw @39638152 (MixedK)";

const FP_NEMOTRON_H4: &str = "sha256:9eef39bb948deda87a99e96787d9663fd0722534dbee82b243aeb7cc37a98ae3";
const FP_NEMOTRON_BF16_HEAD: &str = "sha256:4aca1d25c0d982c408cc0310ca271ee1c80e23c46d362391095f8c920d09c774";
const FP_NEMOTRON_MIXEDK: &str = "sha256:b17cec7004215df7f3b303bca1af552ca71b026dbb301a2fd94183eb766a1826";
/// The DSV4.1-Flash SAGE pack (decision T1.2, 2026-10-20): vcruz305's own 1.62bpw EXL3 quant of DeepSeek-V4.1-Flash-1017 with
/// the engram tables unquantized in shards 16-17 and the DSpark stages in-pack. Fingerprint computed by this code on the anonymized host-b
/// local copy; every shard's bytes matched its published Hugging Face LFS object id before its CRC-32 was taken.
pub const DSV41_SAGE: &str = "vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw";
pub const DSV41_SAGE_FP: &str = "sha256:23d204ab2c22293805f4c3f09bcfcff3c7e09ea1055b4a071bc314bdd76b08ec";
/// The DSV4.1-Flash SAGE 1.55bpw pack (2026-10-29): vcruz305's 1.55bpw EXL3 quant of
/// DeepSeek-V4.1-Flash-1017, the single-node release default (vision fits with full
/// 15,360-expert residency at 262K; report reports/dsv155-fit-test.md). Fingerprint
/// computed by this code (`rexl3-serve --print-fingerprints`) on the 5b84 local copy;
/// every shard's bytes matched its published Hugging Face LFS object id before its
/// CRC-32 was taken.
pub const DSV41_SAGE_155: &str = "vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.55bpw";
pub const DSV41_SAGE_155_FP: &str = "sha256:5aaff81ca01e970f9372bcf4c5f0fd578296b9dd879cfc543673a321882e4e26";
pub const DSV41_SAGE_REXL3_155: &str = "vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw";
pub const DSV41_SAGE_REXL3_155_FP: &str = "sha256:c34045423c126e40d051943ff46bd57a67c07350d8b1743ed5185415db61351c";
/// The DSV4.1-Flash SAGE r0b0t 1.60bpw max-performance pack (2026-10-21): r0b0tlab's extension of vcruz305's SAGE 1.62bpw EXL3 quant
/// of DeepSeek-V4.1-Flash-1017 (4-bit EXL3 head, trunk wq_b/wo_b/shared_experts, and 3-bit EXL3 routed expert outlier replacement;
/// 1.60 bpw routed experts, 100% resident experts on single-rank).
pub const DSV41_R0B0T: &str = "r0b0tlab/DSV4.1-Flash-1017-SAGE-EXL3-r0b0t-1.60bpw";
pub const DSV41_R0B0T_FP: &str = "sha256:ea479b4b6f7365d019805abd084ef7a154782a8aa9e79ec9c0f3d8a9b6c6ef14";

/// The Qwen3.8-Flash-Next pack at 3.05 bits/parameter average (2026-10-23): the local
/// ExLlamaV3 export of Qwen3.8-Flash-Next (ten weight shards, the n-gram table and the MTP
/// mixer sidecar; 256K context, MTP depth 6). Fingerprint computed by this code
/// (`rexl3-serve --print-fingerprints`) on the local copy; validated by serving it on one
/// NVIDIA GB10 (`docs/QWEN38_FLASH_NEXT_3.05_SERVE.md`). The pack ships the vision tower's
/// weights but this release does not build it: serve with `--no-vision`.
pub const QFN305: &str = "Qwen3.8-Flash-Next-EXL3-3.05bpw";
pub const QFN305_FP: &str = "sha256:9e31f4ee36ba86dbe5eb5b263492216e6231e1808a220de2d42483e349b30824";

/// Packs validated without a draft: (name, target fingerprint). GLM-5.3-Flash's serial text forward is validated against the ExLlamaV3
/// oracle (`docs/GLM53.md`, "Whole-model oracle and tolerances") and against itself (a window equals the serial steps, bit for bit). The
/// Nemotron packs are here although they are not text-only: the serve path asks this list for every pack booted without `--draft`
/// ([`validated_pack`]). Their fingerprints were computed by this code (`rexl3-serve --print-fingerprints`) on the local copies of the three
/// packs, 2026-10-17 (decision D6 of the v0.1.2 rules).
pub const VALIDATED_TEXT: &[ValidatedText] = &[
    ("GLM-5.3-Flash-EXL3-2.25bpw-sm121 (text)", "sha256:96a51ffaaddfad22d1ab6f862f405254fcd26473aec059fb5ad93ca9603f52c5"),
    ("MiMo-V2.6-Flash-RL-EXL3-2.20bpw (text)", "sha256:f6653f639e44cbabd67d29a6d93abf27ddb4b3e90f748c361552b179114893fd"),
    (NEMOTRON_H4, FP_NEMOTRON_H4),
    (NEMOTRON_BF16_HEAD, FP_NEMOTRON_BF16_HEAD),
    (NEMOTRON_MIXEDK, FP_NEMOTRON_MIXEDK),
    (DSV41_SAGE, DSV41_SAGE_FP),
    (DSV41_R0B0T, DSV41_R0B0T_FP),
    (DSV41_SAGE_155, DSV41_SAGE_155_FP),
    (DSV41_SAGE_REXL3_155, DSV41_SAGE_REXL3_155_FP),
    ("Ling-3.1-flash-1019-EXL3-SAGE-1.51bpw (text)", "sha256:84c8b957a7268a0a721ea81280dadf96fa803d3ce2231991c2f0f6f1df6f2547"),
    ("vcruz305/Ornith-1.5-122B-EXL3-SAGE-2.58bpw", "sha256:e1c56961f37d8d4a7ade0f24368dd1f2950b8dffdd93b792746350f0828f62e3"),
    // RED-SNOW-5.3-FLASH 2.49bpw (2026-10-25): a GLM-5.3-Flash derivative (`glm5_next`, one MTP layer in the pack, mixed-width experts),
    // so it is served without `--draft`. What "validated" rests on for THIS pack: (1) the P2 prefill tests of PR #95, the per-subbank
    // results equal the uniform-bank ones bit for bit and the real-layer determinism tests; (2) a coherent whole-model boot and
    // generation with MTP drafting on one GB10, up to a 250,210-token prompt; (3) every file equal to the published revision
    // 071adefe6c1c8de379d76977985811b7d4385454 (SHA-256, below). There is NO ExLlamaV3 oracle comparison for this pack, unlike the
    // GLM-5.3-Flash 2.25bpw entry above; do not read "validated" here as oracle-validated.
    (
        "vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-2.49bpw (text)",
        "sha256:56555220e0e7c455f2a2bb20d1d324f1c79d9b85c08a9e8658cb5b84b9b34a42",
    ),
    (QFN305, QFN305_FP),
];

/// The name of the text-only pack with this fingerprint, if this release was validated on it.
pub fn validated_text(target: &str) -> Option<&'static str> {
    VALIDATED_TEXT.iter().find(|(_, t)| *t == target).map(|(n, _)| *n)
}

/// One weight shard of a validated pack: file name, the SHA-256 it is published under (the Hugging Face LFS object id), and
/// its CRC-32. The local copies were checked against the SHA-256 on 2026-10-01 (`sha256sum`), and the CRC-32 taken from those
/// same bytes: `--verify-pack` compares the CRC-32, which runs at memory speed, and the SHA-256 records where it came from.
pub type PinnedShard = (&'static str, &'static str, u32);

/// The weight shards of each validated pack, keyed by the pack's fingerprint. The packs' own `crc32.txt` lists no shard.
///
/// The three Nemotron blocks list every shard the pack's index references, and no other. Taken 2026-10-17 from the local copies: the
/// SHA-256 column is each shard's published SHA-256 (the Hugging Face LFS object id: the r0b0tlab revisions' published `CHECKSUMS.sha256`,
/// the MixedK revision's Hub file listing), which every local shard matched (`sha256sum` of its bytes) before its CRC-32 was taken with
/// `rexl3-serve --print-crc32`. The packs ship no `crc32.txt`. A shard of the two r0b0tlab revisions with the same name is the same file:
/// e3857098 dropped b546ce51's bf16 head experts (`model-00005` and `model-00008`) and added `model-head4-00001-of-00001.safetensors`.
pub const VALIDATED_SHARDS: &[(&str, &[PinnedShard])] = &[
    // DSV4.1-Flash SAGE (T1.2): the 17 shards of vcruz305's pack. SHA-256 column = the Hugging Face LFS object ids of the
    // published files (repo tree API, 2026-10-20), matched by sha256sum of the anonymized host-b local bytes before the CRC-32 was taken.
    (
        "sha256:23d204ab2c22293805f4c3f09bcfcff3c7e09ea1055b4a071bc314bdd76b08ec",
        &[
            ("model-00001-of-00017.safetensors", "ac51ca7efe935e1566d86be2ffabf2ab6df0083c38116315908e3ec8a9f34e8a", 0x3a389ae8),
            ("model-00002-of-00017.safetensors", "272fc820c126affafe9dcc8a681038883df95d961a8a2171943327583691e084", 0xd6319a48),
            ("model-00003-of-00017.safetensors", "94f9f3822ebcdc36d52fe1f857ab8ab20f1974226d8d4e7c8694b7f45bee55a4", 0x54c876e3),
            ("model-00004-of-00017.safetensors", "01b3dc66064724819df25cd178abb25dcfa22dd00c567df26ce289fd5dfc3daa", 0xa1149394),
            ("model-00005-of-00017.safetensors", "dfe1970998f68ca9fcd39cf649d403a1e918f48a87c7d13087ccb01a041f6afd", 0xe4907879),
            ("model-00006-of-00017.safetensors", "e215bfb496c9f73c3d83f19054c22b2bff14f54a33c8bfb8174c4738f89e2801", 0x3641bb25),
            ("model-00007-of-00017.safetensors", "e08d5d61cffd5f0bcdfd7327f77ded6200aaaa924470fd93fba831a83c6adaaa", 0x3237118e),
            ("model-00008-of-00017.safetensors", "13d9071eaedd9ebf3224c40bd9730ed7e1a76abaacd61289a60c9847dbab9a29", 0x14c8a697),
            ("model-00009-of-00017.safetensors", "9d766562049a4d2990a5d72439403db00d87f983c05e9e395deee98a4ff525e0", 0xb01ed903),
            ("model-00010-of-00017.safetensors", "cd48078a47acfc8c4501be4cf5dd2200addccd3f5cc3a30788df1c20f06427a0", 0xcf8a1c7d),
            ("model-00011-of-00017.safetensors", "512060b23fe2f1142a8c097fb3dab548bdf81dc3214dc64666796028f7bc3b75", 0x6d3ed78c),
            ("model-00012-of-00017.safetensors", "45ab31bd41e4c6bb9fe774825688beec791a197e977040ced56cce9913da5d11", 0x7ec0d694),
            ("model-00013-of-00017.safetensors", "8e21d07bd569f8fa3bbb566cd5034b0a8bda0b53bb168c93691334bc4d7465f2", 0x0b372741),
            ("model-00014-of-00017.safetensors", "dda9dbaab1a1de9179d7cc4f5fde6933f613718568492bccc649e82d2901c998", 0x0af10cb4),
            ("model-00015-of-00017.safetensors", "e1ec37b0da1605cc304c8039fa47a24155e3adbdb37da060659488d7bade9bf2", 0x8964868b),
            ("model-00016-of-00017.safetensors", "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5", 0x2e2186cf),
            ("model-00017-of-00017.safetensors", "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd", 0x5b4c19d2),
        ],
    ),
    // DSV4.1-Flash SAGE 1.55bpw (2026-10-29): the 17 shards of vcruz305's pack (the
    // single-node release default). SHA-256 column = the Hugging Face LFS object ids of
    // the published files (repo tree API + local sha256sum on 5b84, 2026-10-29);
    // CRC-32 taken from those same local bytes with zlib.crc32.
    (
        DSV41_SAGE_155_FP,
        &[
            ("model-00001-of-00017.safetensors", "6e92b9cbac9e12de55265ee53d0206624e2e72474b8261609e3faef20accda8c", 0xaacedb44),
            ("model-00002-of-00017.safetensors", "93fcaac64bfdf2a6e771b1db1b10c5ffe0d266719f9f4ae0d0394193542a510d", 0xdd462538),
            ("model-00003-of-00017.safetensors", "985046cc64d8e51f332f2aa428858fda644610f32467c972f60f0604736d5f4e", 0xd3886e20),
            ("model-00004-of-00017.safetensors", "62d1f38976415f3bdc9935bb6d408a39508e5e02e307dbaf6d54c8f6053b75e6", 0x4de862f2),
            ("model-00005-of-00017.safetensors", "409d5ed8b1ce51838b75bef39894b15a9b0b8f6033ebeffdae5448e13f6c586f", 0xbea2b78d),
            ("model-00006-of-00017.safetensors", "dde04f078fb907f921637d4821c53c7f46d674ff9ff96e9fcfd2539aaa376d5c", 0xa9197a35),
            ("model-00007-of-00017.safetensors", "c4f2f91a10d28e0b26bd211f32ff39efc12a0640edb77017da82f5655f781f62", 0x92e63fe3),
            ("model-00008-of-00017.safetensors", "b5bc247741dc7f2260e22db8f930b5760d2eea5381292f4f2acea7cf6ca42e4c", 0x2d6a9fe5),
            ("model-00009-of-00017.safetensors", "108a3c03dfb94113bd01a9a1488f6bb0f53eda291c248d28b902bdd742c114f0", 0xc59d1889),
            ("model-00010-of-00017.safetensors", "a3bbd25601e8f68c2bcf8a2bf41e839142de5f855513dffda6d519ef983e559c", 0x03002b78),
            ("model-00011-of-00017.safetensors", "29468a08e3249e37bceab36f23782ec9a49fb293c065320048a9dacae01616f7", 0xf4586171),
            ("model-00012-of-00017.safetensors", "19cc970a83eb8d250cf4b882ec25c528825dadfb53cd632444c48c21776c036f", 0x8692d8ec),
            ("model-00013-of-00017.safetensors", "1c3b88979361893bf26e4d13be22d367feee6f1b2f80e86f835ac18044d442a4", 0x36f9e80a),
            ("model-00014-of-00017.safetensors", "776a92b1d1ef5f4fc8412c27dcf64b2ffe1ce1324ed94aa9253c10a05efad9af", 0xf3e826c3),
            ("model-00015-of-00017.safetensors", "5bbc73888b28bf16ad8dc428c863defda34d2002635477bdd8e421e33a60ec95", 0x8162973d),
            ("model-00016-of-00017.safetensors", "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5", 0x2e2186cf),
            ("model-00017-of-00017.safetensors", "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd", 0x5b4c19d2),
        ],
    ),
    (
        "sha256:ea479b4b6f7365d019805abd084ef7a154782a8aa9e79ec9c0f3d8a9b6c6ef14",
        &[
            ("model-00001-of-00017.safetensors", "e39402c26aba1627cd39672936ef9e6fab433dab6b516f52e8994c020a8a4ba8", 0xf3fd43d4),
            ("model-00002-of-00017.safetensors", "163db713b6e2936d1307337ed2859252c6e3d78dc0b1b155896694c6173fcf80", 0xb3851e6a),
            ("model-00003-of-00017.safetensors", "94f9f3822ebcdc36d52fe1f857ab8ab20f1974226d8d4e7c8694b7f45bee55a4", 0x54c876e3),
            ("model-00004-of-00017.safetensors", "8db7e5f565b4b742ed3499b1e85b2ef88d2fc1554ad1b084ab16149be42a1de5", 0x1fdb4157),
            ("model-00005-of-00017.safetensors", "d59a8423de951279e31ec66f4f9cdaf80ca4f34b8d6cf02d0d17301a8b05762e", 0x1d678435),
            ("model-00006-of-00017.safetensors", "2147ee28ce5d8f17482892492b0aa095ae5b7a394779b8ef9fd13ddbe1aed909", 0x44180eff),
            ("model-00007-of-00017.safetensors", "ba3d7c1954926cc57339f511244191acb24edfd45108b2d5bfa1fb917e147308", 0x6bc86922),
            ("model-00008-of-00017.safetensors", "fd570221dbe0c201537300e52aebd38758c30606cb4c4123708dd85a0f01ec1d", 0xfb63e7a3),
            ("model-00009-of-00017.safetensors", "44e5a946b6bf35550fc0c50a7510d390b572f4928f0638856b674cd392d1b2ee", 0x7ed54361),
            ("model-00010-of-00017.safetensors", "4b56dce5ec2399543c5bf9756d1e507e75fe1617262cd82cf384bbf364fe36a4", 0x3b62fe5c),
            ("model-00011-of-00017.safetensors", "0540edde607a445c81d4a40cccc0978112cfd73ed1055eda3241012c15fa05f2", 0x8d6ec6ca),
            ("model-00012-of-00017.safetensors", "877eb57b70859f9e0a9db2766a80bff8a83b1112454925165e2b5ed279331795", 0xfd2b61a1),
            ("model-00013-of-00017.safetensors", "435ff19828f712302644b3fb4af52f5c7f66333e073fda376e39e0ca9acd2c5f", 0xf117021e),
            ("model-00014-of-00017.safetensors", "30bc65a9f1d9b3d353e28d36ac4c2b1a48e203cd9864a3503e3617c042464784", 0xcee0a0b5),
            ("model-00015-of-00017.safetensors", "1a160c0b56d368f9c5c303c18b771832d6eda6f86c5bb0a85f86da26173eb298", 0xe43e59d7),
            ("model-00016-of-00017.safetensors", "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5", 0x2e2186cf),
            ("model-00017-of-00017.safetensors", "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd", 0x5b4c19d2),
        ],
    ),
    (
        "sha256:967c3be941ce25284d4ddaaaed02cd3cda5e940ffa622400ed792ffc7a367940",
        &[
            ("model-00001-of-00002.safetensors", "30fa5c8f7ff4c83cb293fe814459e8088838af9b9740ceb0623b6f97112a8475", 0x43ff08c7),
            ("model-00002-of-00002.safetensors", "a54f860c6e502a5fbdc6aba66b27a53799b33879f72f8a8285c97d0024e28042", 0x48ee3b31),
        ],
    ),
    (
        "sha256:3b47e03a4d1e15615e5340baf0fa5e5ffbb0ab15f2b25b069dc25132a18c687d",
        &[("model.safetensors", "e278f218318565562af07fc333045e411ffa0d83523f8224fc8c2d24d5a68223", 0x38cbb33b)],
    ),
    // MiniCPM-V-4.7-35B-A3B (E7). Taken from the local shards of `packs/q3-6.00bpw` and `draft-packs/dspark-6.00bpw` with
    // `scripts/minicpmv47_fingerprint.sh` on 2026-10-13, before any upload: the release verification confirms the SHA-256 column against
    // the Hugging Face LFS object ids of the published files (the CRC-32 is what `--verify-pack` compares).
    (
        "sha256:9bc579ceb1ae00bd016922d354e340bb395e9da52438cc226d1c898d5e63d03f",
        &[
            ("model-00001-of-00008.safetensors", "e35037215430c76bbd841a5f39ebe6f5759b421909f29faa90f4dd3119a964b7", 0xcada3fbb),
            ("model-00002-of-00008.safetensors", "165c85ab89c74950842728ca8122894301ffbc90d8155b1b3ebc915e93836674", 0xe166a75e),
            ("model-00003-of-00008.safetensors", "a1ba7fca2d96d71c4c82cd1db909caaf52eb9b9840eec9ab779e5f236e8459bc", 0xb4458d60),
            ("model-00004-of-00008.safetensors", "0c6743519751398d1088f7f501147760f04a9789162db017aa5429e4d77d867f", 0x1c80597a),
            ("model-00005-of-00008.safetensors", "eb197338fb6838cd8740174f804ccdd669cc68ddd2ba8d9e4dc878cd84c6d3b1", 0x650b94ba),
            ("model-00006-of-00008.safetensors", "aff890e36675821474595ad28b7e9d9556d2cf7f56e66746f2637fedf61b6767", 0xf9a39cea),
            ("model-00007-of-00008.safetensors", "8bb73fb2bd4ce673695cb4bc889c6d2c948d0e69f5ff88c327295d7003fb880f", 0xc19a386e),
            ("model-00008-of-00008.safetensors", "8ef5f1cd7b36bbc4d428c7e700843fabbe1bfa72d724dd08427cfaa2e4926ce9", 0x3b01d7e8),
        ],
    ),
    (
        "sha256:7731abc818a30cd7636af7d72de016d21371c3f5c5c46e5ab38121e602bdf5ab",
        &[("model.safetensors", "728cdd789e20c142f1085e16f8381973f3a7018d4fabcbc776c13b6e77eff964", 0x342ca3a2)],
    ),
    // Nemotron 3.5 Super VL (v0.1.2, decision D6): the three packs of VALIDATED_TEXT, every shard of each index.
    (
        FP_NEMOTRON_H4,
        &[
            ("model-00001-of-00032.safetensors", "ed2ba875472c106bde54b195a3d9057db25c1b378a3fe6cc623d0b6f27c6446b", 0xfa28f5ae),
            ("model-00002-of-00032.safetensors", "c006d5fdd8410878019e1e125c9ac0bd940dd54d558b8c48be7cb666a39e7d0f", 0x42e1ea3b),
            ("model-00003-of-00032.safetensors", "1cc458b4aaebe6d1a20a74cd515df5b965be2b8b0ee272492e8c41a885b307c1", 0x46b9ff6c),
            ("model-00004-of-00032.safetensors", "33a734fe9f4bdd2c2e33a8294205c737d196d23846c3a7e0be464e89f2ff13d4", 0x0fb7d3f2),
            ("model-00006-of-00032.safetensors", "86e94da3b6de91da435829cb5d408f270ffb7545657cd092c9dc7cf425f8ef7d", 0x468a9523),
            ("model-00007-of-00032.safetensors", "ea9e3b12c8a6c92523bf57da0d3f092a80991973a35ecbed8fb51e3f495a496c", 0x420163da),
            ("model-00009-of-00032.safetensors", "330e5d54c6c100e7ae80fbe767e0869dd5395263543fc7ddd1dc0af5e3d8277b", 0xaacd6bb0),
            ("model-00010-of-00032.safetensors", "164f84660e4eee216bbc5bb544dbd57d69e0b4a512feddded649ca3f4f046662", 0x054a80a9),
            ("model-00011-of-00032.safetensors", "b39bd5016f6708d4e8c70044b763bc3a8a8218ef423a40d3a06a33abaa5f7e7b", 0x942bfdb0),
            ("model-00012-of-00032.safetensors", "e89390aa847e75f96b337109f8b8f25324d41d146411e32b79b518373d9a28c0", 0x5e2589e8),
            ("model-00013-of-00032.safetensors", "ebbcad454b211f6fcd12bdb9a207f5a7006f486e4a66c7233b1b3cc8b8c0e802", 0x0c9a0952),
            ("model-00014-of-00032.safetensors", "bac9ea55251cbd0fcb5d41a798e3c0a54111ec8b34a807ad47cd3c50141576e8", 0x354c96b7),
            ("model-00015-of-00032.safetensors", "d24c9a3fe4cf069d00024879ddb0849a71f03d2b7a76431f3083755a0ffc399d", 0xd0b471fd),
            ("model-00016-of-00032.safetensors", "c9ed9fc3292271da6bce5f25e052930930367d1d0039123a0a4f4179a23ef549", 0x97af3c29),
            ("model-00017-of-00032.safetensors", "0d45b8d114e66e2a4a19bf80c6e33928dd38e03b0e41d22590f7834291c3d619", 0x1e1f85bd),
            ("model-00018-of-00032.safetensors", "2d8ece782a2fc9315f1748282b78ee5a201a40ed71016b2e1ffe9466910e6f9d", 0xd9f6043b),
            ("model-00019-of-00032.safetensors", "fefda580e287bea525cfd1a254433fbdc62ae94a8bd69b1468ce2930a9f4a05d", 0xbb7b57a9),
            ("model-00020-of-00032.safetensors", "1d95478ad46eea160419543c9a962c949a852269b939dc96c917bd7e54950c0a", 0x0f73bb4d),
            ("model-00021-of-00032.safetensors", "0fed3fc212a3c8e1e2c3d712dc4a20cd7ea1bc8fe1a0bb292ef42878930bb71a", 0xd62ade67),
            ("model-00022-of-00032.safetensors", "75bdf8134d24c9f32d422e8fc3cbbf401723a0f5df361c4e9bb15915555d321a", 0xd5ca3b14),
            ("model-00023-of-00032.safetensors", "8cdfa19f1d840fead932a3bff7b294205caca65bd1664057f3d0ab2b46cf16f7", 0xc73241ac),
            ("model-00024-of-00032.safetensors", "b078edc1f5824bafd91bf60898deda11810afbaad78bcbb2abd4431a5b7c0b7d", 0x44a3f321),
            ("model-00025-of-00032.safetensors", "2d56d8dd95fc56c4f9becfe8093faaa84fa529afab242d84aa70d278ed0a7bb7", 0x3eadaa75),
            ("model-00026-of-00032.safetensors", "2f3ad66d616f948b4fbc0410335640818815106de8f3f61c5aa4be994dafe742", 0xf20cfff7),
            ("model-00027-of-00032.safetensors", "b79ef7ca1eb713893ee8f4ab290142a7a9727c130cc79a9c9e33982c07eeae47", 0xd8a203a9),
            ("model-00028-of-00032.safetensors", "8a3bfe1670e043748e595adc3ce8587abf41933ebb598f3277093fcc4fd92e5d", 0x83481ab6),
            ("model-00029-of-00032.safetensors", "45ee23beb0a733c30eb5317d3c7f5137d1111c2f7919e6c8398480ebc6a23cf1", 0x6e650f81),
            ("model-00030-of-00032.safetensors", "6ca2ee5809975c2cdec99492dd8978eba25ba72021cec106dea688ce8f5b966b", 0xa5fb1dca),
            ("model-00031-of-00032.safetensors", "13ee47adc2b583f979a105b749bc7439bf67dcdc6ca93ea93f80ae17ec5e7f82", 0x4305220c),
            ("model-00032-of-00032.safetensors", "45d0e3009d8ecf626992f808ca143ad66d004df4c51a5748a220b4bb85fde879", 0x7cc3ac6c),
            (
                "model-head4-00001-of-00001.safetensors",
                "56f1292b57dd84c2f4e132dac48a661178d3381b03e124bf7c4dc847e676f16b",
                0x7808cfb4,
            ),
            (
                "model-overlay-00001-of-00001.safetensors",
                "db72d175d6ae44443584a5c560bfd411a3cfbef9984287ed8b83326b9b092fc1",
                0x022ef8f5,
            ),
        ],
    ),
    (
        FP_NEMOTRON_BF16_HEAD,
        &[
            ("model-00001-of-00032.safetensors", "ed2ba875472c106bde54b195a3d9057db25c1b378a3fe6cc623d0b6f27c6446b", 0xfa28f5ae),
            ("model-00002-of-00032.safetensors", "c006d5fdd8410878019e1e125c9ac0bd940dd54d558b8c48be7cb666a39e7d0f", 0x42e1ea3b),
            ("model-00003-of-00032.safetensors", "1cc458b4aaebe6d1a20a74cd515df5b965be2b8b0ee272492e8c41a885b307c1", 0x46b9ff6c),
            ("model-00004-of-00032.safetensors", "33a734fe9f4bdd2c2e33a8294205c737d196d23846c3a7e0be464e89f2ff13d4", 0x0fb7d3f2),
            ("model-00005-of-00032.safetensors", "5ea8e57bc60e56721d0a2c45b5cffb66256def0748bfe3e6866db6d0e6b0bc4a", 0x3f6003ca),
            ("model-00006-of-00032.safetensors", "86e94da3b6de91da435829cb5d408f270ffb7545657cd092c9dc7cf425f8ef7d", 0x468a9523),
            ("model-00007-of-00032.safetensors", "ea9e3b12c8a6c92523bf57da0d3f092a80991973a35ecbed8fb51e3f495a496c", 0x420163da),
            ("model-00008-of-00032.safetensors", "736e36e9974deb22c2eea5447d87c9060448a6213382e67a10607c5dcb8641ac", 0x552b3f82),
            ("model-00009-of-00032.safetensors", "330e5d54c6c100e7ae80fbe767e0869dd5395263543fc7ddd1dc0af5e3d8277b", 0xaacd6bb0),
            ("model-00010-of-00032.safetensors", "164f84660e4eee216bbc5bb544dbd57d69e0b4a512feddded649ca3f4f046662", 0x054a80a9),
            ("model-00011-of-00032.safetensors", "b39bd5016f6708d4e8c70044b763bc3a8a8218ef423a40d3a06a33abaa5f7e7b", 0x942bfdb0),
            ("model-00012-of-00032.safetensors", "e89390aa847e75f96b337109f8b8f25324d41d146411e32b79b518373d9a28c0", 0x5e2589e8),
            ("model-00013-of-00032.safetensors", "ebbcad454b211f6fcd12bdb9a207f5a7006f486e4a66c7233b1b3cc8b8c0e802", 0x0c9a0952),
            ("model-00014-of-00032.safetensors", "bac9ea55251cbd0fcb5d41a798e3c0a54111ec8b34a807ad47cd3c50141576e8", 0x354c96b7),
            ("model-00015-of-00032.safetensors", "d24c9a3fe4cf069d00024879ddb0849a71f03d2b7a76431f3083755a0ffc399d", 0xd0b471fd),
            ("model-00016-of-00032.safetensors", "c9ed9fc3292271da6bce5f25e052930930367d1d0039123a0a4f4179a23ef549", 0x97af3c29),
            ("model-00017-of-00032.safetensors", "0d45b8d114e66e2a4a19bf80c6e33928dd38e03b0e41d22590f7834291c3d619", 0x1e1f85bd),
            ("model-00018-of-00032.safetensors", "2d8ece782a2fc9315f1748282b78ee5a201a40ed71016b2e1ffe9466910e6f9d", 0xd9f6043b),
            ("model-00019-of-00032.safetensors", "fefda580e287bea525cfd1a254433fbdc62ae94a8bd69b1468ce2930a9f4a05d", 0xbb7b57a9),
            ("model-00020-of-00032.safetensors", "1d95478ad46eea160419543c9a962c949a852269b939dc96c917bd7e54950c0a", 0x0f73bb4d),
            ("model-00021-of-00032.safetensors", "0fed3fc212a3c8e1e2c3d712dc4a20cd7ea1bc8fe1a0bb292ef42878930bb71a", 0xd62ade67),
            ("model-00022-of-00032.safetensors", "75bdf8134d24c9f32d422e8fc3cbbf401723a0f5df361c4e9bb15915555d321a", 0xd5ca3b14),
            ("model-00023-of-00032.safetensors", "8cdfa19f1d840fead932a3bff7b294205caca65bd1664057f3d0ab2b46cf16f7", 0xc73241ac),
            ("model-00024-of-00032.safetensors", "b078edc1f5824bafd91bf60898deda11810afbaad78bcbb2abd4431a5b7c0b7d", 0x44a3f321),
            ("model-00025-of-00032.safetensors", "2d56d8dd95fc56c4f9becfe8093faaa84fa529afab242d84aa70d278ed0a7bb7", 0x3eadaa75),
            ("model-00026-of-00032.safetensors", "2f3ad66d616f948b4fbc0410335640818815106de8f3f61c5aa4be994dafe742", 0xf20cfff7),
            ("model-00027-of-00032.safetensors", "b79ef7ca1eb713893ee8f4ab290142a7a9727c130cc79a9c9e33982c07eeae47", 0xd8a203a9),
            ("model-00028-of-00032.safetensors", "8a3bfe1670e043748e595adc3ce8587abf41933ebb598f3277093fcc4fd92e5d", 0x83481ab6),
            ("model-00029-of-00032.safetensors", "45ee23beb0a733c30eb5317d3c7f5137d1111c2f7919e6c8398480ebc6a23cf1", 0x6e650f81),
            ("model-00030-of-00032.safetensors", "6ca2ee5809975c2cdec99492dd8978eba25ba72021cec106dea688ce8f5b966b", 0xa5fb1dca),
            ("model-00031-of-00032.safetensors", "13ee47adc2b583f979a105b749bc7439bf67dcdc6ca93ea93f80ae17ec5e7f82", 0x4305220c),
            ("model-00032-of-00032.safetensors", "45d0e3009d8ecf626992f808ca143ad66d004df4c51a5748a220b4bb85fde879", 0x7cc3ac6c),
            (
                "model-overlay-00001-of-00001.safetensors",
                "db72d175d6ae44443584a5c560bfd411a3cfbef9984287ed8b83326b9b092fc1",
                0x022ef8f5,
            ),
        ],
    ),
    (
        FP_NEMOTRON_MIXEDK,
        &[
            ("model-00001-of-00008.safetensors", "bdc266908fcd20ba383ec7e7d846229a0d9e8f1e509b5e4c90937b747939cef1", 0x8728320b),
            ("model-00002-of-00008.safetensors", "5725acc7f0cc5995da2f20377af9fe22f3196a9ff653913a64f4231e391736e9", 0xb4574d8d),
            ("model-00003-of-00008.safetensors", "fabcc2c5dd572d6f1948063c0034047a04c8451bd91a987715e0252b7871985f", 0x97982823),
            ("model-00004-of-00008.safetensors", "629d161e34c4b465969ffd25f76991e0a39cb6b8761621c07354f4838e5f9fb0", 0x71a20897),
            ("model-00005-of-00008.safetensors", "b6a660b6d4916d6697097c07df91f7c2f24238e4e6858f4b062eab8923e91f36", 0xc04984e3),
            ("model-00006-of-00008.safetensors", "f6f363054185cffc820e218fa4058b7b39f05dd8489bcdfb5c015e3c0da58bf6", 0x96a644e1),
            ("model-00007-of-00008.safetensors", "ea8f9aad9e940674bda3eab7560753a8b9fb16d2de97758c6da34209d441d03f", 0x1940473f),
            ("model-00008-of-00008.safetensors", "42de8cb03ff6e0371cd5f2ad5bbb9a8c00c058ecc9f5e1cd16fc984d8ffc77c3", 0x17f96cf5),
        ],
    ),
    // RED-SNOW-5.3-FLASH 2.49bpw: the 15 shards of vcruz305's pack at revision 071adefe6c1c8de379d76977985811b7d4385454. SHA-256 column = the
    // Hugging Face LFS object ids of the published files (repo tree API, 2026-10-25), matched by sha256sum of the bytes of the local copy
    // on 2026-10-25 before the CRC-32 was taken with `rexl3-serve --print-crc32` (all 22 locked files matched).
    (
        "sha256:56555220e0e7c455f2a2bb20d1d324f1c79d9b85c08a9e8658cb5b84b9b34a42",
        &[
            ("model-00001-of-00015.safetensors", "6084383ff9a7a4ab2f66fa8898ccd0d79b63c8a434439f43290226fba88447e2", 0xc7987551),
            ("model-00002-of-00015.safetensors", "5a55f9fe19831094aac5d08c4496add09b329eb954c863e316cbb1dc5a96671e", 0x95f65d99),
            ("model-00003-of-00015.safetensors", "2f57e0942eeb429e1b86d019bdf186560cb5c06db5e3b109dd365b5758b4f8f3", 0x43c189dc),
            ("model-00004-of-00015.safetensors", "4b4b328b120729530ef1822445731af49463a1f1f850b983f1aa340bb320f1b7", 0x65710182),
            ("model-00005-of-00015.safetensors", "469f0b3a6d3d04ec9e2b6e355c9ca80095a7580d60510ea7eac4abba7c11ed49", 0x307d3995),
            ("model-00006-of-00015.safetensors", "e3f3939506ce2e14a915e267cbb4bc1fc258b69787e511ee9f403bd879917e87", 0x561a82bb),
            ("model-00007-of-00015.safetensors", "2d53a75a22de0ba5720f21d78797b957a9e68b78af3facfd658586052abd85b3", 0x55999733),
            ("model-00008-of-00015.safetensors", "6437ec0224cff19924effe2f94fb44a4dea2c81352fcbf878066d99fbeb7b900", 0x8fb6c870),
            ("model-00009-of-00015.safetensors", "6979bee0d4d868375f23f9e54cc10ba8778ab43711cd30b29f8225d20b7bd0f3", 0xa009215f),
            ("model-00010-of-00015.safetensors", "e6e5b58f723adf0645377782bce10c1d44ff2ae3a8fb7814d4c3eefcb2357bfa", 0x477ee098),
            ("model-00011-of-00015.safetensors", "254bc1c5fcff7245a13c13900c4122ccb6ca5ee59ddc09dc9e95e7bfdab0847d", 0xba429fce),
            ("model-00012-of-00015.safetensors", "b19daaff8037271cac817052ba37fe89733764a7522f171ccf89a14b273b2f0e", 0x2017ae9c),
            ("model-00013-of-00015.safetensors", "23b1b6cecb855b54e1a318c72a6c0a20ff08ff7edfaa0b6b61652c8db369dafc", 0x568d83c9),
            ("model-00014-of-00015.safetensors", "7c79ea6fce9bef3ba730f1a724cc54b689431ac91bc1b940ec70fb0428aa5d74", 0x8b1f2ae5),
            ("model-00015-of-00015.safetensors", "ef2a16fe5dfb4a2a9dc4e23f1ccda476733fed78f4f4c026af336f479ecd6eec", 0xf0e56bf2),
        ],
    ),
];

/// The pinned shards of the pack with this fingerprint (empty for a pack that is not validated).
pub fn pinned_shards(fingerprint: &str) -> &'static [PinnedShard] {
    VALIDATED_SHARDS.iter().find(|(f, _)| *f == fingerprint).map_or(&[], |(_, s)| *s)
}

fn canonical(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*k).clone()).to_string());
                out.push(':');
                canonical(&m[*k], out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonical(x, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// The canonical text a fingerprint hashes (exposed for tests and for diffing two packs).
pub fn description(ck: &Checkpoint) -> String {
    let mut s = String::new();
    s.push_str(FORM);
    s.push_str("\nconfig ");
    canonical(&ck.config, &mut s);
    s.push('\n');
    for (name, g) in &ck.groups {
        s.push_str(&format!(
            "group {name} bits={} codebook={} k={} n={} in={} out={} bias={}\n",
            g.bits.as_f64(),
            g.codebook.name(),
            g.k,
            g.n,
            g.in_scales,
            g.out_scales,
            g.bias
        ));
    }
    for (name, e) in &ck.plain {
        s.push_str(&format!("tensor {name} {} {:?}\n", e.dtype, e.shape));
    }
    s
}

/// `sha256:<64 hex digits>` of [`description`].
pub fn fingerprint(ck: &Checkpoint) -> String {
    let d = Sha256::digest(description(ck).as_bytes());
    let mut s = String::from("sha256:");
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The validated pair a target and draft fingerprint belong to. A target served without a draft matches on the target.
pub fn validated(target: &str, draft: Option<&str>) -> Option<&'static str> {
    VALIDATED
        .iter()
        .chain(VALIDATED_MINICPMV4_7)
        .find(|(_, t, d)| *t == target && draft.is_none_or(|x| x == *d))
        .map(|(n, _, _)| *n)
}

/// What this release was validated on, for any family: a (target, draft) pair of [`VALIDATED`], or, with no draft, a text-only pack of
/// [`VALIDATED_TEXT`]. The ONE validation: the Qwen serve path (`serve::check_packs`), the GLM one (`glm::engine::check_pack`) and
/// `rexl3-serve --print-fingerprints` all ask it, so the report can never say `no` for a pack the server serves as validated.
pub fn validated_pack(target: &str, draft: Option<&str>) -> Option<&'static str> {
    validated(target, draft).or_else(|| draft.is_none().then(|| validated_text(target)).flatten())
}

/// The `validated ...` line of `rexl3-serve --print-fingerprints`.
pub fn validation_line(target: &str, draft: Option<&str>) -> String {
    match validated_pack(target, draft) {
        Some(name) => format!("validated {name}"),
        None => "validated no (serving needs --allow-unvalidated-pack)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_sage_pack_is_validated_with_pinned_shards() {
        let sage = DSV41_SAGE_FP;
        assert_eq!(validated_text(sage), Some(DSV41_SAGE));
        // every pinned shard row names a file the pack's own index references, and the shard count is the SAGE pack's 17
        let (_, shards) = VALIDATED_SHARDS.iter().find(|(fp, _)| *fp == sage).expect("SAGE shards pinned");
        assert_eq!(shards.len(), 17);
        assert!(shards.iter().all(|(n, _, _)| n.starts_with("model-") && n.ends_with(".safetensors")));
    }

    #[test]
    fn the_sage_155_pack_is_validated_with_pinned_shards() {
        let sage = DSV41_SAGE_155_FP;
        assert_eq!(validated_text(sage), Some(DSV41_SAGE_155));
        let (_, shards) = VALIDATED_SHARDS.iter().find(|(fp, _)| *fp == sage).expect("SAGE 1.55 shards pinned");
        assert_eq!(shards.len(), 17);
        assert!(shards.iter().all(|(n, _, _)| n.starts_with("model-") && n.ends_with(".safetensors")));
    }

    #[test]
    fn the_redsnow_pack_is_validated_with_its_fifteen_pinned_shards() {
        let fp = "sha256:56555220e0e7c455f2a2bb20d1d324f1c79d9b85c08a9e8658cb5b84b9b34a42";
        let (name, _) = VALIDATED_TEXT.iter().find(|(_, t)| *t == fp).expect("RED-SNOW is in VALIDATED_TEXT");
        assert_eq!(*name, "vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-2.49bpw (text)");
        assert_eq!(validated_pack(fp, None), Some(*name));
        assert_eq!(validation_line(fp, None), format!("validated {name}"));
        // the pack carries its own MTP head: it is a text entry only, never half of a (target, draft) pair
        assert!(VALIDATED.iter().all(|(_, t, d)| *t != fp && *d != fp));
        // one pin for each of the pack's 15 index-referenced shards, no shard twice
        let shards = pinned_shards(fp);
        assert_eq!(shards.len(), 15);
        for (i, (file, sha, _)) in shards.iter().enumerate() {
            assert_eq!(*file, format!("model-{:05}-of-00015.safetensors", i + 1));
            assert!(sha.len() == 64 && sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')), "{file} {sha}");
        }
    }

    #[test]
    fn the_glm_text_pack_is_validated_and_nothing_else_is() {
        let glm = VALIDATED_TEXT[0].1;
        assert_eq!(validated_text(glm), Some("GLM-5.3-Flash-EXL3-2.25bpw-sm121 (text)"));
        assert_eq!(validated_text("sha256:0"), None);
        // a text pack is never also a (target, draft) pair, and the Qwen target is not a text-only entry
        assert!(VALIDATED.iter().all(|(_, t, d)| *t != glm && *d != glm));
        assert_eq!(validated_text(VALIDATED[0].1), None);
    }

    // Fingerprints no table lists (the release's own tables are what the tests below read).
    const T: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";
    const D: &str = "sha256:2222222222222222222222222222222222222222222222222222222222222222";

    fn is_hex(s: &str, len: usize) -> bool {
        s.len() == len && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    /// v0.1.2 (D6): the three Nemotron packs are validated without a draft (their MTP head is in the pack), each with every shard of its
    /// index pinned; none of them is a pair's target or draft, and no other family's row has one of their fingerprints; any other pack
    /// still needs the flag. The real packs are checked against these rows in `tests/nemotron_omni_pins.rs` (fingerprint, shard list,
    /// `--verify-pack`).
    #[test]
    fn this_release_pins_the_three_nemotron_packs_and_every_other_pack_needs_the_flag() {
        let nemotron = [NEMOTRON_H4, NEMOTRON_BF16_HEAD, NEMOTRON_MIXEDK];
        let rows: Vec<ValidatedText> = VALIDATED_TEXT.iter().copied().filter(|(n, _)| nemotron.contains(n)).collect();
        let names: Vec<&str> = rows.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, nemotron, "each Nemotron pack is listed once");
        let fps: std::collections::BTreeSet<&str> = rows.iter().map(|(_, f)| *f).collect();
        assert_eq!(fps.len(), 3, "three distinct fingerprints");
        // the other families' rows stay theirs: no pair names a Nemotron fingerprint, no other text row has one, one pin block each
        assert!(VALIDATED.iter().chain(VALIDATED_MINICPMV4_7).all(|(_, t, d)| !fps.contains(t) && !fps.contains(d)));
        assert_eq!(
            VALIDATED_TEXT.iter().filter(|(_, f)| fps.contains(f)).count(),
            3,
            "another text row has a Nemotron fingerprint"
        );
        assert_eq!(VALIDATED_SHARDS.iter().filter(|(f, _)| fps.contains(f)).count(), 3, "one pin block for each Nemotron pack");
        for (name, fp) in &rows {
            assert!(fp.starts_with("sha256:") && is_hex(&fp[7..], 64), "{name}: {fp}");
            assert_eq!(validated_pack(fp, None), Some(*name));
            assert_eq!(validation_line(fp, None), format!("validated {name}"));
            assert_eq!(validated_pack(fp, Some(D)), None, "{name}: validated without a draft, not with one");
            let shards = pinned_shards(fp);
            let want =
                [(NEMOTRON_H4, 32), (NEMOTRON_BF16_HEAD, 33), (NEMOTRON_MIXEDK, 8)].iter().find(|(n, _)| n == name).unwrap().1;
            assert_eq!(shards.len(), want, "{name}: every shard of its index");
            let mut seen = std::collections::BTreeSet::new();
            for (file, sha, _) in shards {
                assert!(file.ends_with(".safetensors") && !file.contains('/') && seen.insert(*file), "{name}: {file}");
                assert!(is_hex(sha, 64), "{name}: {file}: {sha}");
            }
        }
        // the two r0b0tlab revisions share their trunk: a shard of the same name is the same file, the head's shards are their own
        let (h4, bf16) = (pinned_shards(FP_NEMOTRON_H4), pinned_shards(FP_NEMOTRON_BF16_HEAD));
        let shared: Vec<&PinnedShard> = h4.iter().filter(|s| bf16.iter().any(|b| b.0 == s.0)).collect();
        assert_eq!(shared.len(), 31, "30 trunk shards and the overlay");
        assert!(shared.iter().all(|s| bf16.contains(s)), "a shared shard pinned differently");
        let own = |a: &[PinnedShard], b: &[PinnedShard]| {
            a.iter().filter(|s| !b.iter().any(|x| x.0 == s.0)).map(|s| s.0).collect::<Vec<_>>()
        };
        assert_eq!(own(h4, bf16), ["model-head4-00001-of-00001.safetensors"]);
        assert_eq!(own(bf16, h4), ["model-00005-of-00032.safetensors", "model-00008-of-00032.safetensors"]);
        for draft in [None, Some(D)] {
            assert_eq!(validated_pack(T, draft), None);
            assert_eq!(validation_line(T, draft), "validated no (serving needs --allow-unvalidated-pack)");
        }
        assert!(pinned_shards(T).is_empty());
    }

    #[test]
    fn one_validation_answers_for_every_family_and_the_fingerprint_report_uses_it() {
        // `--print-fingerprints` printed `validated no` for the GLM pack that the server serves as validated: it asked the Qwen pair list only. The
        // serve paths of both families and the report now ask `validated_pack`
        let (qwen_name, qwen, draft) = VALIDATED[0];
        let (glm_name, glm) = VALIDATED_TEXT[0];
        let (mimo_name, mimo) = VALIDATED_TEXT[1];
        assert_eq!(validated_pack(qwen, Some(draft)), Some(qwen_name));
        assert_eq!(validated_pack(qwen, None), Some(qwen_name), "a Qwen target served without its draft is the pair's target");
        assert_eq!(validated_pack(glm, None), Some(glm_name));
        assert_eq!(validated_pack(mimo, None), Some(mimo_name));
        assert_eq!(validation_line(mimo, None), format!("validated {mimo_name}"));
        // a text-only pack with a draft is not a pair, a draft that is not the pair's draft is not validated, and the unknown never is
        assert_eq!(validated_pack(glm, Some(draft)), None);
        assert_eq!(validated_pack(qwen, Some("sha256:other")), None);
        assert_eq!(validated_pack("sha256:other", None), None);
        // the report says what the serve path will say
        assert_eq!(validation_line(qwen, Some(draft)), format!("validated {qwen_name}"));
        assert_eq!(validation_line(glm, None), format!("validated {glm_name}"));
        assert_eq!(validation_line("sha256:other", None), "validated no (serving needs --allow-unvalidated-pack)");
    }

    fn is_fingerprint(s: &str) -> bool {
        s.strip_prefix("sha256:").is_some_and(|h| h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
    }

    #[test]
    #[allow(clippy::const_is_empty)] // the guard is the point: it fails when the shipped entry is removed
    fn a_listed_minicpm_pair_is_validated_and_every_other_one_stays_refused() {
        // E7: the shipped pair is listed (the E2 boot test pins the real pack's fingerprint to the same value)
        assert!(!VALIDATED_MINICPMV4_7.is_empty(), "the shipping pair has an entry");
        for (name, t, d) in VALIDATED_MINICPMV4_7 {
            // a listed pair is validated as a pair and by its target alone (a text-only serve), and the report says so
            assert_eq!(validated_pack(t, Some(d)), Some(*name));
            assert_eq!(validated_pack(t, None), Some(*name));
            assert_eq!(validation_line(t, Some(d)), format!("validated {name}"));
            assert_eq!(validation_line(t, None), format!("validated {name}"));
            // the pair is the pair: another draft, another target, a draft where the target should be, and the draft alone are not
            assert_eq!(validated_pack(t, Some("sha256:other")), None);
            assert_eq!(validated_pack("sha256:other", Some(d)), None);
            assert_eq!(validated_pack(d, Some(t)), None);
            assert_eq!(validated_pack(d, None), None, "a draft pack is not a servable target");
            assert_eq!(validation_line(t, Some("sha256:other")), "validated no (serving needs --allow-unvalidated-pack)");
        }
        // an unknown target, with or without a draft, is not validated and the report says what to pass
        for draft in [None, Some("sha256:draft")] {
            assert_eq!(validated_pack("sha256:minicpm", draft), None);
            assert_eq!(validation_line("sha256:minicpm", draft), "validated no (serving needs --allow-unvalidated-pack)");
        }
        // the Qwen pair is still its own entry, whatever the MiniCPM slot holds
        let (name, t, d) = VALIDATED[0];
        assert_eq!(validated(t, Some(d)), Some(name));
    }

    #[test]
    fn every_minicpm_entry_is_well_formed_and_pinned() {
        // adding an entry is one tuple here and the pack's block in VALIDATED_SHARDS: both are checked, so a half-added pack fails this
        for (i, (name, t, d)) in VALIDATED_MINICPMV4_7.iter().enumerate() {
            assert!(!name.trim().is_empty(), "entry {i} has no name");
            assert!(is_fingerprint(t) && is_fingerprint(d), "{name}: {t} {d} are not sha256:<64 hex digits>");
            assert_ne!(t, d, "{name}: the target and the draft are one pack");
            for fp in [t, d] {
                let pins = pinned_shards(fp);
                assert!(!pins.is_empty(), "{name}: {fp} has no pinned shard digests (VALIDATED_SHARDS)");
                for (file, sha, _) in pins {
                    assert!(
                        file.ends_with(".safetensors")
                            && !file.contains('/')
                            && sha.len() == 64
                            && sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                        "{name}: pin {file} {sha}"
                    );
                }
            }
            // no pack is listed twice with different drafts under one name, and no name repeats
            assert_eq!(VALIDATED_MINICPMV4_7.iter().filter(|(n, _, _)| n == name).count(), 1, "{name} is listed twice");
        }
        // the Qwen pair and the MiniCPM pairs share no fingerprint, so one family's pack never validates another's
        for (_, t, d) in VALIDATED_MINICPMV4_7 {
            for (_, qt, qd) in VALIDATED {
                assert!(t != qt && t != qd && d != qt && d != qd);
            }
            assert!(VALIDATED_TEXT.iter().all(|(_, x)| x != t && x != d));
        }
        // every pin block belongs to a listed pack: an orphan block is a stale paste
        for (fp, _) in VALIDATED_SHARDS {
            let listed = VALIDATED.iter().chain(VALIDATED_MINICPMV4_7).any(|(_, t, d)| t == fp || d == fp)
                || VALIDATED_TEXT.iter().any(|(_, t)| t == fp);
            assert!(listed, "{fp} is pinned but no entry lists it");
        }
    }

    #[test]
    fn canonical_json_ignores_key_order_and_whitespace() {
        let (mut a, mut b) = (String::new(), String::new());
        canonical(&json!({"b": [1, {"y": 2, "x": 1}], "a": "s"}), &mut a);
        canonical(&serde_json::from_str::<Value>(r#"{ "a" : "s", "b":[1,{"x":1,"y":2}] }"#).unwrap(), &mut b);
        assert_eq!(a, b);
        assert_eq!(a, r#"{"a":"s","b":[1,{"x":1,"y":2}]}"#);
    }

    #[test]
    fn a_changed_config_value_changes_the_fingerprint() {
        let mut ck = Checkpoint::default();
        ck.config = json!({"hidden_size": 5120});
        let a = fingerprint(&ck);
        ck.config = json!({"hidden_size": 5121});
        assert_ne!(a, fingerprint(&ck));
        assert!(a.starts_with("sha256:") && a.len() == 7 + 64);
    }

    #[test]
    fn every_validated_pack_has_pinned_shards() {
        for (_, t, d) in VALIDATED {
            for fp in [t, d] {
                let pins = pinned_shards(fp);
                assert!(!pins.is_empty(), "{fp} has no pinned shard digests");
                for (name, sha, _) in pins {
                    assert!(
                        name.ends_with(".safetensors") && sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
                        "{name} {sha}"
                    );
                }
            }
        }
        assert!(pinned_shards("sha256:other").is_empty());
    }

    #[test]
    fn only_listed_pairs_are_validated() {
        let (_, t, d) = VALIDATED[0];
        assert!(validated(t, Some(d)).is_some());
        assert!(validated(t, None).is_some());
        assert!(validated(t, Some("sha256:other")).is_none());
        assert!(validated("sha256:other", Some(d)).is_none());
    }

    #[test]
    fn the_qwen_flash_next_pack_is_validated_and_other_packs_are_refused() {
        // X10: the shipped Qwen3.8-Flash-Next 3.05bpw pack (fingerprint computed by this code on the
        // local copy, `rexl3-serve --print-fingerprints`, and cross-checked on the GB10 box) is a
        // VALIDATED_TEXT entry, so `serve::check_packs` accepts it without `--allow-unvalidated-pack`.
        assert!(is_fingerprint(QFN305_FP), "QFN305_FP is not a sha256 fingerprint");
        assert_eq!(validated_text(QFN305_FP), Some(QFN305));
        assert_eq!(validated_pack(QFN305_FP, None), Some(QFN305));
        assert_eq!(validation_line(QFN305_FP, None), format!("validated {QFN305}"));
        // a text pack is not a (target, draft) pair: a draft is refused
        assert_eq!(validated_pack(QFN305_FP, Some("sha256:draft")), None);
        // the QFN fingerprint is not any other listed pack's, and no other entry claims it
        assert!(VALIDATED.iter().all(|(_, t, d)| *t != QFN305_FP && *d != QFN305_FP));
        assert_eq!(VALIDATED_TEXT.iter().filter(|(_, t)| *t == QFN305_FP).count(), 1);
        // wrong packs are refused: an unknown target is not validated and the report says what to pass
        assert_eq!(validated_pack("sha256:other", None), None);
        assert_eq!(validation_line("sha256:other", None), "validated no (serving needs --allow-unvalidated-pack)");
    }
}
