#!/usr/bin/env python3
"""Shared, side-effect-free runtime contracts. Parsed provenance is not qualification."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

DSV_REPO = "vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw"
DSV_REPO_162 = "vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw"
# Every validated DSV pack shares the same rules: no external draft role and a
# complete 17-shard roster with the index/tokenizer/config metadata.
DSV_REPOS = frozenset({DSV_REPO, DSV_REPO_162})
EXCLUDED_REPO = "r0b0tlab/DSV4.1-Flash-1017-SAGE-EXL3-r0b0t-1.60bpw"
DSV_DIRECTORY = "dsv41-flash-1017-sage-1.55"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SLUG = re.compile(r"[a-z0-9][a-z0-9.-]{0,63}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
TOKEN = re.compile(r"@[A-Z][A-Z0-9_]*@")
FORBIDDEN_ARGS = {"--allow-unvalidated-pack"}
FORBIDDEN_ENV = {"REXL3_NO_CUDA", "GB_NO_CUDA", "EXTRA_ARGS", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME"}
SECRET_KEY = re.compile(r"TOKEN|SECRET|PASSWORD|PRIVATE_KEY|API_KEY|AUTHORIZATION|CREDENTIAL", re.I)
CAPABILITY_KEYS = {"text", "sse", "reasoning", "tools", "images", "video", "audio", "json_schema", "prefix_cache"}
# site.example.json ships unusable network names with this prefix so an unedited copy is refused.
SITE_PLACEHOLDER_PREFIX = "REPLACE_WITH_"
SITE_FIELDS = {"http_host", "http_port", "nodes", "ssh_user", "model_dir", "binary_dir", "fabric_interface", "hca", "tp_port"}
PATH_TOKENS = {"@MODEL_DIR@", "@DRAFT_DIR@", "@BUNDLE_DIR@", "@STATE_DIR@"}
CORE_SOURCE_FILES = {
    "launch.sh", "profile.json", "model.lock.json", "dependencies.lock.json", "requirements-runtime.lock",
    "scripts/contract.py", "scripts/bootstrap.sh", "scripts/install_dependencies.py", "scripts/doctor.py",
    "scripts/fetch_models.py", "scripts/runtime.py", "scripts/remote_identity.py", "README.md", "docs/SETUP.md", "docs/TROUBLESHOOTING.md",
    "site.example.json", "KNOWN_ISSUES.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
}
NATIVE_VERSION_KEYS = {"version", "commit", "system_fingerprint", "nvcc", "cuda_arch", "kernel_hash", "cudart", "cuda_driver", "cublas"}


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _finite_json(value):
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite JSON value")
    elif isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "JSON keys must be strings")
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)
    else:
        require(value is None or type(value) in (str, int, bool), "non-JSON value")


def load_json(path):
    def nonfinite(value):
        raise ContractError("non-finite JSON value: " + value)
    try:
        with open(path, encoding="utf-8") as stream:
            result = json.load(stream, object_pairs_hook=_pairs, parse_constant=nonfinite)
        _finite_json(result)  # parse_constant alone does not catch 1e999 overflow.
        return result
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("cannot read JSON: " + str(path)) from error


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(data):
    _finite_json(data)
    return sha256_bytes(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())


def relative_file(value):
    require(isinstance(value, str) and bool(value), "empty or non-string path")
    p = PurePosixPath(value)
    require(bool(p.parts) and not p.is_absolute() and str(p) == value, "noncanonical relative path")
    require(not any(ord(c) < 32 or ord(c) == 127 for c in value) and "\\" not in value and ":" not in value,
            "control or platform-dependent path")
    require(all(x not in ("..", ".git") for x in p.parts), "unsafe relative path")
    return value


def safe_file(root, relative):
    """An existing regular file beneath root; reject links in root AND descendants."""
    relative_file(relative)
    root = Path(os.path.abspath(root))
    path = root / relative
    cursor = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            cursor /= component
            mode = cursor.lstat().st_mode
            require(not stat.S_ISLNK(mode), "SYMLINK_REFUSED: " + str(cursor))
            require(stat.S_ISREG(mode) if cursor == path else stat.S_ISDIR(mode), "UNSAFE_BUNDLE_PATH: " + str(cursor))
        require(root in path.resolve().parents, "PATH_ESCAPES_ROOT")
    except OSError as error:
        raise ContractError("UNSAFE_OR_MISSING_FILE: " + relative) from error
    return path


def full_hash(value, regex, name):
    require(isinstance(value, str) and regex.fullmatch(value) is not None, "invalid " + name)


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and not any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _tokens(choice, value):
    allowed = PATH_TOKENS | {"@" + key.upper() + "@" for key in choice["site_fields"]}
    if choice.get("cluster_adapter") == "attested-v1": allowed |= {"@HTTP_HOST@"}
    require(set(TOKEN.findall(value)) <= allowed, "unapproved runtime path/site token")
    require(re.search(r"@[A-Z][A-Z0-9_]*", TOKEN.sub("", value)) is None, "malformed runtime token")


def _smoke(request):
    require(isinstance(request, dict) and _text(request.get("model")), "smoke_request model required")
    messages = request.get("messages")
    require(isinstance(messages, list) and bool(messages), "smoke_request messages required")
    for message in messages:
        require(isinstance(message, dict) and message.get("role") in ("system", "user", "assistant", "developer")
                and _text(message.get("content")), "smoke_request requires literal text messages")
    require(request.get("stream", False) is False, "smoke_request must be non-streaming")
    for key in ("max_tokens", "max_completion_tokens"):
        if key in request:
            require(type(request[key]) is int and request[key] > 0, "invalid smoke token limit")
    def no_secrets(value):
        if isinstance(value, dict):
            for key, item in value.items():
                # Token limits are request policy, not authentication secrets.
                require(key in ("max_tokens", "max_completion_tokens") or not SECRET_KEY.search(key), "secret smoke field refused")
                no_secrets(item)
        elif isinstance(value, list):
            for item in value:
                no_secrets(item)
    no_secrets(request)


def validate(profile, models, required_repo=None):
    require(isinstance(profile, dict) and isinstance(models, dict), "contracts must be objects")
    _finite_json(profile)
    _finite_json(models)
    require(profile.get("schema") == "rexl3.runtime-profile.v1", "unknown profile schema")
    require(profile.get("state") == "final", "PROFILE_NOT_READY")
    full_hash(profile.get("runtime_id"), SLUG, "runtime ID")
    source = profile.get("source")
    require(isinstance(source, dict), "missing profile source")
    full_hash(source.get("commit"), HEX40, "profile source commit")
    full_hash(source.get("sha256"), HEX64, "profile source digest")
    relative_file(source.get("path"))
    full_hash(source.get("repository"), REPOSITORY, "source repository")
    require(models.get("schema") == "rexl3.model-lock.v1", "unknown model-lock schema")
    packs = models.get("packs")
    require(isinstance(packs, dict) and "target" in packs and bool(set(packs) - {"draft"}),
            "invalid pack roles: target and at least one non-draft role required")
    for role, pack in packs.items():
        require(bool(SLUG.fullmatch(role)), "invalid pack role name")
        require(isinstance(pack, dict), "invalid pack record")
        repo = pack.get("repo")
        full_hash(repo, REPOSITORY, "model repository")
        require(repo != EXCLUDED_REPO, "DEPRECATED_MODEL_REFUSED")
        full_hash(pack.get("revision"), HEX40, role + " revision")
        fp = pack.get("fingerprint", "")
        require(isinstance(fp, str) and fp.startswith("sha256:"), "missing pack fingerprint")
        full_hash(fp[7:], HEX64, role + " fingerprint")
        files = pack.get("files")
        require(isinstance(files, dict) and bool(files), "empty pack file manifest")
        for path, item in files.items():
            relative_file(path)
            require(isinstance(item, dict), "invalid file record")
            full_hash(item.get("sha256"), HEX64, "payload digest")
            require(type(item.get("bytes")) is int and item["bytes"] > 0, "invalid payload size")
        require("config.json" in files, "config must be pinned")
        require(_text(pack.get("license")), "license reference required")
    target = packs["target"]
    require(profile.get("target_repo") == target["repo"], "profile/model identity mismatch")
    if required_repo is not None:
        require(target["repo"] == required_repo, "model is not the approved beta target")
    # DSV packs share one rule set regardless of which role carries them: no
    # external draft role, complete metadata and an exact 17-shard roster.
    dsv_roles = [role for role, pack in packs.items() if pack["repo"] in DSV_REPOS]
    if dsv_roles:
        require("draft" not in packs, "DSV uses the embedded draft; external draft refused")
        for role in dsv_roles:
            files = packs[role]["files"]
            require({"model.safetensors.index.json", "tokenizer.json", "config.json"} <= set(files),
                    "DSV metadata incomplete")
            require(sum(name.endswith(".safetensors") for name in files) == 17,
                    "DSV shard roster must contain 17 files")
    choices = profile.get("profiles")
    require(isinstance(choices, dict) and bool(choices), "no serve profiles")
    default = profile.get("default_profile")
    require(isinstance(default, str) and default in choices, "missing explicit default profile")
    for name, choice in choices.items():
        full_hash(name, SLUG, "profile ID")
        require(isinstance(choice, dict), "invalid profile entry")
        require(choice.get("kind") in ("single", "cluster"), "unsupported runtime kind")
        if "cluster_adapter" in choice:
            require(choice["kind"] == "cluster" and choice["cluster_adapter"] == "attested-v1", "unsupported cluster adapter")
        # A choice may pin the lock role it serves (default: the target pack) and
        # override the entry runtime_id so receipts and recipe lookup are pack-true.
        pack_role = choice.get("pack", "target")
        require(isinstance(pack_role, str) and pack_role in packs and pack_role != "draft",
                "profile names an unknown pack role")
        runtime_override = choice.get("runtime_id")
        require(runtime_override is None or bool(SLUG.fullmatch(runtime_override)),
                "choice runtime ID override must be a slug")
        site = choice.get("site_fields")
        require(isinstance(site, list) and all(isinstance(x, str) and x in SITE_FIELDS for x in site), "invalid site field allowlist")
        require(len(site) == len(set(site)), "duplicate site field")
        argv = choice.get("argv")
        require(isinstance(argv, list) and bool(argv) and all(_text(v) for v in argv), "argv must be a nonempty string array")
        require(argv[0].startswith("--"), "argv contains server arguments only; the wrapper selects the bundled executable")
        require(not any(v.split("=", 1)[0] in FORBIDDEN_ARGS for v in argv), "unsafe validation override")
        if packs[pack_role]["repo"] in DSV_REPOS:
            require(not any(v.split("=", 1)[0] == "--draft" or "@DRAFT_DIR@" in v for v in argv), "unexpected external DSV draft")
        for word in argv:
            _tokens(choice, word)
        env = choice.get("env")
        require(isinstance(env, dict), "effective environment required")
        for key, value in env.items():
            require(isinstance(key, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", key), "invalid environment key")
            require(key not in FORBIDDEN_ENV and not SECRET_KEY.search(key), "forbidden environment variable")
            require(isinstance(value, str) and "\0" not in value and "\n" not in value, "invalid environment value")
            _tokens(choice, value)
        require(type(choice.get("readiness_timeout_s")) is int and choice["readiness_timeout_s"] > 0, "readiness timeout required")
        require(type(choice.get("http_port")) is int and 1 <= choice["http_port"] <= 65535, "http_port required in 1..65535")
        host = choice.get("http_host", "127.0.0.1")
        require(isinstance(host, str) and re.fullmatch(r"[A-Za-z0-9_.:-]+", host), "invalid http_host")
        _smoke(choice.get("smoke_request"))
        if "smoke_expect" in choice:
            require(_text(choice["smoke_expect"]), "smoke_expect must be nonempty literal content")
        capabilities = choice.get("capabilities")
        require(isinstance(capabilities, dict) and set(capabilities) == CAPABILITY_KEYS, "complete capability contract required")
        require(all(type(value) is bool for value in capabilities.values()), "capability values must be booleans")
        require(isinstance(choice.get("files"), dict), "immutable config inventory required")
        for path, digest in choice["files"].items():
            relative_file(path)
            full_hash(digest, HEX64, "configuration digest")
        for key, value in env.items():
            if key.endswith("_KNOBS_FILE"):
                require(value.startswith("@BUNDLE_DIR@/"), "knobs file must be immutable bundle content")
                require(value[len("@BUNDLE_DIR@/"):] in choice["files"], "knobs file digest missing")
        if choice["kind"] == "cluster":
            relative_file(choice.get("cluster_file"))
            require(choice["cluster_file"] in choice["files"], "cluster configuration digest missing")
            require(_text(choice.get("group")), "cluster group required")
    return profile, models


def render_argv(profile, name, substitutions):
    require(isinstance(name, str) and name in profile["profiles"], "unknown profile")
    choice = profile["profiles"][name]
    allowed = PATH_TOKENS | {"@" + key.upper() + "@" for key in choice["site_fields"]}
    if choice.get("cluster_adapter") == "attested-v1": allowed |= {"@HTTP_HOST@"}
    require(isinstance(substitutions, dict) and set(substitutions) <= allowed, "unapproved site/path substitution")
    require(all(isinstance(v, str) and "\0" not in v and TOKEN.search(v) is None for v in substitutions.values()), "invalid substitution value")
    def replace(match):
        require(match[0] in substitutions, "unresolved runtime path/site token")
        return substitutions[match[0]]
    return [TOKEN.sub(replace, word) for word in choice["argv"]]


def validate_site(site, kind):
    """Only placement is mutable. No arbitrary env, argv, SSH options or knobs."""
    from remote_identity import absolute_path
    allowed = {'http_host', 'http_port', 'model_dir', 'binary_dir', 'pack'} if kind == 'single' else {
        'nodes', 'http_port', 'tp_port', 'fabric_interface', 'hca', 'pack'}
    require(isinstance(site, dict) and set(site) <= allowed, 'UNAPPROVED_SITE_FIELD')
    def address(value):
        require(isinstance(value, str), 'INVALID_SITE_ADDRESS')
        try: parsed = ipaddress.IPv4Address(value)
        except ValueError: raise ContractError('INVALID_SITE_ADDRESS') from None
        require(not parsed.is_unspecified and not parsed.is_multicast, 'INVALID_SITE_ADDRESS')
    for key, value in site.items():
        if key == 'pack':
            require(isinstance(value, str) and bool(SLUG.fullmatch(value)), 'INVALID_SITE_PACK')
        elif key in ('http_port', 'tp_port'):
            require(type(value) is int and 1 <= value <= 65535, 'INVALID_SITE_PORT')
        elif key in ('model_dir', 'binary_dir'):
            absolute_path(value)
        elif key == 'http_host':
            try: ipaddress.ip_address(value)
            except (ValueError, TypeError): raise ContractError('INVALID_SITE_ADDRESS') from None
        elif key == 'fabric_interface':
            require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,32}', value), 'INVALID_SITE_INTERFACE')
        elif key == 'hca':
            require(isinstance(value, list) and bool(value) and
                    all(isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_]+:[1-9][0-9]*', v) for v in value) and len(set(value)) == len(value), 'INVALID_SITE_HCA')
        elif key == 'nodes':
            require(isinstance(value, dict) and set(value) == {'leader', 'follower'}, 'EXACT_TWO_SITE_NODES_REQUIRED')
            for node in value.values():
                require(isinstance(node, dict) and set(node) == {'address', 'fabric_address', 'user', 'model_dir', 'binary_dir'},
                        'INVALID_SITE_NODE_FIELDS')
                address(node['address']); address(node['fabric_address'])
                require(isinstance(node['user'], str) and re.fullmatch(r'[a-z_][a-z0-9_-]{0,31}', node['user']), 'INVALID_SSH_USER')
                absolute_path(node['model_dir']); absolute_path(node['binary_dir'])
            require(value['leader']['address'] != value['follower']['address'], 'DUPLICATE_SITE_NODE')
    if kind == 'cluster' and 'nodes' in site:
        # Attested multi-node (TP2): the sealed cluster file carries only example network
        # values. Falling back to them would point NCCL at a device that may not exist on
        # this host, so the tester must state the network explicitly. One rail stays valid.
        for key, code in (('fabric_interface', 'SITE_FABRIC_INTERFACE_REQUIRED'), ('hca', 'SITE_HCA_REQUIRED'),
                          ('tp_port', 'SITE_TP_PORT_REQUIRED')):
            require(key in site, code)
        require(not site['fabric_interface'].startswith(SITE_PLACEHOLDER_PREFIX),
                'SITE_FABRIC_INTERFACE_PLACEHOLDER: replace the example value in site.example.json')
        require(not any(rail.startswith(SITE_PLACEHOLDER_PREFIX) for rail in site['hca']),
                'SITE_HCA_PLACEHOLDER: replace the example values in site.example.json')
    return site


def expected_mode(relative):
    return 0o755 if relative == "launch.sh" or relative.startswith("bin/") or (relative.startswith("scripts/") and relative.endswith((".py", ".sh"))) else 0o644


def validate_platform(platform):
    require(isinstance(platform, dict), "release platform required")
    for key in ("system", "machine", "gpu_compute_capability", "gpu_name_contains"):
        require(_text(platform.get(key)), "invalid platform " + key)
    version = platform.get("minimum_python")
    require(isinstance(version, list) and len(version) == 2 and all(type(n) is int and n >= 0 for n in version), "minimum_python must be two integers")
    require(version[0] >= 3, "Python 3 required")
    require(re.fullmatch(r"\d+\.\d+", platform["gpu_compute_capability"]), "invalid GPU compute capability")
    for key in ("commands", "libraries"):
        items = platform.get(key)
        require(isinstance(items, list) and all(_text(x) and not x.startswith("-") for x in items), "invalid platform " + key)
        require(len(items) == len(set(items)), "duplicate platform " + key)


def validate_native_identity(fields, commit, platform):
    require(isinstance(fields, dict) and set(fields) == NATIVE_VERSION_KEYS, "complete measured native_version_fields required")
    require(all(_text(value) and value.lower() not in ("none", "unknown", "unavailable", "pending", "no_cuda")
                and "dirty" not in value.lower() and "no_cuda" not in value.lower() for value in fields.values()), "CPU_STUB_OR_DIRTY_NATIVE_IDENTITY_REFUSED")
    require(fields["commit"] == commit, "NATIVE_COMMIT_MISMATCH")
    full_hash(fields["commit"], HEX40, "native commit")
    # crates/cuda/build.rs emits a 64-bit FNV-1a identity, NOT a SHA-256.
    full_hash(fields["kernel_hash"], re.compile(r"[0-9a-f]{16}\Z"), "native kernel hash")
    arch = platform["gpu_compute_capability"].replace(".", "")
    require(fields["cuda_arch"] in (arch, "sm_" + arch, "compute_" + arch), "NATIVE_GPU_ARCH_MISMATCH")


def required_inventory(profile):
    paths = CORE_SOURCE_FILES | {"bin/rexl3-serve"}
    for choice in profile["profiles"].values():
        paths |= set(choice["files"])
        if choice["kind"] == "cluster":
            paths.add("bin/rexl3-cluster")
    return paths


def verify_release(root):
    """Verify every immutable byte/mode before callers install or execute anything."""
    release_path = safe_file(root, "release.lock.json")
    require(stat.S_IMODE(release_path.stat().st_mode) == 0o644, "RELEASE_LOCK_MODE_MISMATCH")
    release = load_json(release_path)
    require(isinstance(release, dict) and release.get("schema") == "rexl3.release.v1", "RELEASE_LOCK_INVALID")
    profile = load_json(safe_file(root, "profile.json"))
    models = load_json(safe_file(root, "model.lock.json"))
    validate(profile, models, required_repo=release.get("target_repo"))
    require(release.get("runtime_id") == profile["runtime_id"] and release.get("target_repo") == profile["target_repo"], "RELEASE_IDENTITY_MISMATCH")
    for key in ("production_commit", "production_tree"):
        full_hash(release.get(key), HEX40, key)
    require(_text(release.get("version")) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", release["version"])
            and release["version"].lower() not in ("latest", "pending", "unknown"), "immutable release version required")
    validate_platform(release.get("platform"))
    validate_native_identity(release.get("native_version_fields"), release["production_commit"], release["platform"])
    inventory = release.get("files")
    require(isinstance(inventory, dict) and required_inventory(profile) <= set(inventory), "RELEASE_INVENTORY_INCOMPLETE")
    require(any(name.startswith("sbom/") for name in inventory), "RELEASE_SBOM_MISSING")
    require("release.lock.json" not in inventory, "self-referential release inventory")
    for alias, name in (("profile_sha256", "profile.json"), ("model_lock_sha256", "model.lock.json")):
        if alias in release:
            require(isinstance(inventory[name], dict) and release[alias] == inventory[name].get("sha256"), "RELEASE_DIGEST_ALIAS_MISMATCH: " + alias)
    for relative, item in inventory.items():
        path = safe_file(root, relative)
        require(isinstance(item, dict) and set(item) == {"sha256", "bytes", "mode"}, "invalid release file record")
        full_hash(item.get("sha256"), HEX64, "release file digest")
        require(type(item.get("bytes")) is int and item["bytes"] >= 0, "invalid release file size")
        require(type(item.get("mode")) is int and item["mode"] in (0o644, 0o755), "invalid release file mode")
        require(path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"], "BUNDLE_INTEGRITY_FAILED: " + relative)
        require(stat.S_IMODE(path.stat().st_mode) == item["mode"] == expected_mode(relative), "BUNDLE_MODE_MISMATCH: " + relative)
    # Mutable site/cache/state are not release inventory; immutable code/config roots are closed.
    managed = {"bin", "scripts", "config", "docs", "sbom"} | {name.split("/")[0] for name in inventory if "/" in name}
    for directory in managed:
        base = Path(root) / directory
        if base.exists():
            for path in base.rglob("*"):
                require(not path.is_symlink(), "SYMLINK_REFUSED: " + str(path))
                if not path.is_dir():
                    require(path.relative_to(root).as_posix() in inventory, "EXTRA_MANAGED_FILE: " + path.relative_to(root).as_posix())
    for choice in profile["profiles"].values():
        for name, digest in choice["files"].items():
            require(inventory[name]["sha256"] == digest, "CONFIGURATION_HASH_MISMATCH: " + name)
    return profile, models, release
