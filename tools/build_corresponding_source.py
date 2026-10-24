#!/usr/bin/env python3
"""Deterministic, fail-closed corresponding-source projection of a Git commit.

Only Git objects are source authority. No checkout, filters, Cargo, compiler,
network, source rewriting, binary qualification, or publication is performed.
Python >=3.11 and Git are required. Exit 2: contract/I/O error; 3: privacy blocked.
"""
from __future__ import annotations

import argparse
import ctypes
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import posixpath
import re
import stat
import subprocess
import sys
import tarfile
import tomllib

SCHEMA = "rexl3.corresponding-source.v1"
MAX_BLOB_BYTES = 32 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_ENTRIES = 20000
TEMPLATE = "tools/templates/corresponding-source-README.md"
REQUIRED = ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
ROOTS = (".cargo/", "crates/", "third_party/", "testdata/")
INPUTS = "tools/corresponding_source_inputs.json"
ALLOWANCES = "tools/corresponding_source_test_allowances.json"
TOOLS = ("tools/build_corresponding_source.py", "tools/test_build_corresponding_source.py", TEMPLATE, INPUTS, ALLOWANCES)
EXTRA_INPUTS = ("tools/dashboard_check.py", "dashboard/panels.json")
# Do not open/hash tensor payloads, even when their filename calls them a fixture.
# Opaque containers are excluded too: this exporter does not unpack model archives.
MODEL_SUFFIXES = (".safetensors", ".pt", ".pth", ".ckpt", ".gguf", ".ggml", ".bin",
                  ".onnx", ".h5", ".hdf5", ".npy", ".npz", ".pkl", ".pickle",
                  ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".zst", ".7z")
INCLUDE_START = re.compile(rb"\binclude(?:_str|_bytes)?!\s*[({\[]")
INCLUDE_LITERAL = re.compile(rb'\s*"([^"\\]+)"\s*,?\s*[)}\]]')
RUST_PATH_START = re.compile(rb'#\s*\[\s*path\b')
RUST_PATH_LITERAL = re.compile(rb'\s*=\s*"([^"\\]+)"\s*\]')
C_INCLUDE = re.compile(rb'(?m)^[ \t]*#[ \t]*(include(?:_next)?|import)\b([^\r\n]*)')
C_SUFFIXES = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cu", ".cuh", ".S", ".s", ".inc")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

# Deliberately conservative. Even test fixtures require committed remediation
# or explicit human review; there is no ignore/allow-privacy/export-anyway flag.
# Pattern spellings avoid embedding the actual private identifiers in this tool.
PRIVACY = (
    ("home_path", rb"(?:/(?:home|Users)/[A-Za-z0-9_.-]+|/root/(?:projects|\.ssh|\.aws|\.config)|[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9_.-]+)"),
    ("private_ipv4", rb"(?<![0-9.])(?:10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})(?![0-9.])"),
    ("internal_host", rb"(?i)\b(?:r0b0t(?:dgx)[0-9]?|(?:spark|gn100)-[a-f0-9]{4}|node[1-4])\b"),
    ("hf_token", rb"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,}"),
    ("github_token", rb"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ("api_key", rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{24,}"),
    ("aws_key", rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ("slack_token", rb"\bxox[abeprs]-[A-Za-z0-9-]{10,}"),
    ("bearer", rb"(?i)\bbearer[ \t]+[A-Za-z0-9._~+/=-]{20,}"),
    ("url_credential", rb"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@\"'<>]+:[^\s/@\"'<>]+@"),
    ("private_key", rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    ("credential_assignment", rb'''(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret[_-]?key)["']?\s*[:=]\s*["'][A-Za-z0-9_./+=-]{12,}["']'''),
)
PRIVACY_RX = tuple((category, re.compile(pattern)) for category, pattern in PRIVACY)


class SourceError(Exception):
    """Only fixed, content-free error codes cross the CLI boundary."""


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_digest(kind, data):
    return hashlib.sha1(kind.encode() + b" " + str(len(data)).encode() + b"\0" + data).hexdigest()


def validate_path(path):
    """Portable relative archive names only; never echo an untrusted filename."""
    if not isinstance(path, str) or not path or len(path) > 240:
        raise SourceError("unsafe_path")
    parts = path.split("/")
    if any(not re.fullmatch(r"[A-Za-z0-9_.+][A-Za-z0-9_.+-]*", p)
           or p in (".", "..") or p.lower() == ".git" for p in parts):
        raise SourceError("unsafe_path")
    if any(p.endswith(".") or re.fullmatch(r"(?i)(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", p) for p in parts):
        raise SourceError("unsafe_path")
    return path


def git(repo, *args):
    # Ignore ambient alternate-index/repo/replace configuration. Do not fetch a
    # missing promisor object, invoke filters, or print Git's private-path errors.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1",
               GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        result = subprocess.run(["git", "--no-replace-objects", "-C", os.fspath(repo), *args],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise SourceError("git_unavailable") from None
    if result.returncode:
        raise SourceError("git_object_unavailable")
    return result.stdout


def read_blob(repo, entry):
    if entry["bytes"] > MAX_BLOB_BYTES:
        raise SourceError("blob_limit")
    data = git(repo, "cat-file", "blob", entry["git_blob"])
    if len(data) != entry["bytes"] or git_digest("blob", data) != entry["git_blob"]:
        raise SourceError("blob_integrity")
    return data


def toml(data):
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise SourceError("invalid_build_toml") from None


def relative_input(owner, value):
    if not isinstance(value, str) or value.startswith("/") or "\\" in value:
        raise SourceError("unsafe_build_reference")
    return validate_path(posixpath.normpath(posixpath.join(posixpath.dirname(owner), value)))


def environment_identifier_spans(line):
    """Recognize shell identifier positions, never host values or whole lines.

    Uppercase alone is not sufficient: assignments, parameter expansions,
    export lists and the CLI's shell-assignment help list have distinct syntax.
    This changes classification only, not the original source or public CLI.
    """
    identifier = rb"[A-Z][A-Z0-9_]*"
    patterns = (
        rb"(?<![A-Za-z0-9_./:@-])(" + identifier + rb")=(?!=)",
        rb"\$\{\{?(" + identifier + rb")(?=[:}])",
        rb'''(?:^|["';])[ \t]*export[ \t]+(''' + identifier + rb"(?:[ \t]+" + identifier + rb''')*)(?=["';]|$)''',
        rb"^[ \t]*env[ \t]+shell assignments[^()]*\((" + identifier + rb"(?:,[ \t]*" + identifier + rb")+)\)",
    )
    return {(m.start(1) + item.start(), m.start(1) + item.end())
            for pattern in patterns for m in re.finditer(pattern, line)
            for item in re.finditer(identifier, m[1])}


def scan_bytes(data):
    """Return locations/categories, never match values (including binary bytes).

    Decode ASCII JSON/Rust unicode/hex escapes for scanning only; original
    source bytes are never modified. Lines retain their original byte offsets.
    """
    found = set()
    for number, line in enumerate(data.split(b"\n"), 1):
        decoded = re.sub(rb"\\(?:u00|x)([0-9a-fA-F]{2})", lambda m: bytes([int(m[1], 16)]), line)
        for variant in {line, decoded}:
            identifiers = environment_identifier_spans(variant)
            for category, rx in PRIVACY_RX:
                if any(category != "internal_host" or m.span() not in identifiers for m in rx.finditer(variant)):
                    found.add((number, category))
    return sorted(found)


def scan_file(path, data):
    name_hits = scan_bytes(path.encode())
    safe_name = path if not name_hits else "[redacted-path]"
    findings = [{"path": safe_name, "path_sha256": digest(path.encode()), "line": 0, "category": c}
                for _, c in name_hits]
    findings.extend({"path": safe_name, "path_sha256": digest(path.encode()), "line": line, "category": c}
                    for line, c in scan_bytes(data))
    if re.search(r"(?:^|/)(?:\.env(?:\..*)?|[^/]+\.env)$", path):
        findings.append({"path": safe_name, "path_sha256": digest(path.encode()), "line": 0, "category": "env_file"})
    return findings


def strict_json(data, error):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError()
            result[key] = value
        return result
    def invalid(_):
        raise ValueError()
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError):
        raise SourceError(error) from None


def privacy_result(files):
    """Partition raw findings using only exact-content committed allowances.

    The policy itself cannot be allowed. Missing policy grants nothing. Review
    records are immutable inputs, not authority to publish; independent release
    review remains necessary even with zero unreviewed findings.
    """
    findings = [f for path, data in sorted(files.items()) for f in scan_file(path, data)]
    allowed = set()
    if ALLOWANCES in files:
        policy = strict_json(files[ALLOWANCES], "invalid_test_allowances")
        if (not isinstance(policy, dict) or set(policy) != {"schema", "entries"}
                or policy["schema"] != "rexl3.source-test-allowances.v1"
                or not isinstance(policy["entries"], list) or len(policy["entries"]) > 4096):
            raise SourceError("invalid_test_allowances")
        seen = set()
        categories = {category for category, _ in PRIVACY}
        actual = {(f["path_sha256"], f["line"], f["category"]) for f in findings}
        for entry in policy["entries"]:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "findings", "reason"}:
                raise SourceError("invalid_test_allowance_entry")
            path = validate_path(entry["path"])
            if path == ALLOWANCES or path not in files or path in seen:
                raise SourceError("stale_or_duplicate_test_allowance_path")
            seen.add(path)
            if (not isinstance(entry["sha256"], str) or not HEX64.fullmatch(entry["sha256"])
                    or entry["sha256"] != digest(files[path])):
                raise SourceError("test_allowance_hash_mismatch")
            if not isinstance(entry["reason"], str) or not entry["reason"].strip():
                raise SourceError("invalid_test_allowance_reason")
            locations = entry["findings"]
            if not isinstance(locations, list) or not locations or len(locations) > 4096:
                raise SourceError("invalid_test_allowance_findings")
            for location in locations:
                if (not isinstance(location, dict) or set(location) != {"line", "category"}
                        or type(location["line"]) is not int or location["line"] <= 0
                        or not isinstance(location["category"], str) or location["category"] not in categories):
                    raise SourceError("invalid_test_allowance_location")
                key = digest(path.encode()), location["line"], location["category"]
                if key in allowed:
                    raise SourceError("duplicate_test_allowance_finding")
                if key not in actual:
                    raise SourceError("stale_test_allowance")
                allowed.add(key)
    reviewed = []
    unreviewed = []
    for finding in findings:
        key = finding["path_sha256"], finding["line"], finding["category"]
        (reviewed if key in allowed else unreviewed).append(finding)
    status = "BLOCKED" if unreviewed else "PASS_REVIEWED_TEST_FIXTURES" if reviewed else "PASS"
    return {"findings": findings, "reviewed_findings": reviewed, "unreviewed_findings": unreviewed,
            "privacy_status": status}


def excluded_payload(path):
    return path.lower().endswith(MODEL_SUFFIXES)


def allowed_input(path):
    return path.startswith(ROOTS + ("docs/examples/",)) or path in EXTRA_INPUTS


def path_list(value):
    if not isinstance(value, list) or len(value) > 4096:
        raise SourceError("invalid_input_manifest")
    result = [validate_path(p) for p in value]
    if len(set(result)) != len(result):
        raise SourceError("duplicate_input_manifest_entry")
    return result


def input_policy(entries, selected, load):
    """Explicit finite runtime-test inputs and reviewed dynamic build closure.

    Arbitrary build scripts are not statically decidable. Every enabled Cargo
    build script therefore requires an exact-content, committed review listing
    its repository inputs and external prerequisites. This is review evidence,
    not proof that the script is honest, and never authorizes publication.
    """
    policy = strict_json(load(INPUTS), "invalid_input_manifest") if INPUTS in entries else {
        "schema": "rexl3.source-inputs.v1", "files": [], "build_scripts": []}
    if (not isinstance(policy, dict) or set(policy) not in (
            {"schema", "files", "build_scripts"},
            {"schema", "files", "build_scripts", "external_test_dependencies"})
            or policy["schema"] != "rexl3.source-inputs.v1"):
        raise SourceError("invalid_input_manifest")
    def require(path):
        if excluded_payload(path):
            raise SourceError("excluded_model_build_input")
        if not allowed_input(path):
            raise SourceError("input_outside_projection")
        if path not in entries:
            raise SourceError("missing_build_reference")
        selected.add(path)
    for path in path_list(policy["files"]):
        require(path)
    scripts = {}
    for owner in sorted(p for p in selected if p.endswith("Cargo.toml")):
        package = toml(load(owner)).get("package", {})
        if not isinstance(package, dict):
            raise SourceError("invalid_package_table")
        build = package.get("build", True)
        if build is False:
            continue
        if not isinstance(build, str) and build is not True:
            raise SourceError("invalid_cargo_build_script")
        script = relative_input(owner, "build.rs" if build is True else build)
        if build is True and script not in entries:
            continue
        if not script.startswith(ROOTS):
            raise SourceError("build_script_outside_projection")
        require(script)
        scripts[script] = posixpath.dirname(owner) + "/"
    reviews = policy["build_scripts"]
    if not isinstance(reviews, list) or len(reviews) > 256:
        raise SourceError("invalid_input_manifest")
    reviewed = set()
    include_dirs = {}
    for review in reviews:
        if (not isinstance(review, dict) or set(review) != {
                "path", "sha256", "inputs", "include_dirs", "external_inputs", "reason"}):
            raise SourceError("invalid_build_script_review")
        path = validate_path(review["path"])
        if path not in scripts or path in reviewed:
            raise SourceError("stale_or_duplicate_build_script_review")
        if (not isinstance(review["sha256"], str) or not HEX64.fullmatch(review["sha256"])
                or digest(load(path)) != review["sha256"]):
            raise SourceError("build_script_review_hash_mismatch")
        if not isinstance(review["reason"], str) or not review["reason"].strip():
            raise SourceError("invalid_build_script_review")
        external = review["external_inputs"]
        if (not isinstance(external, list) or len(external) > 64
                or any(not isinstance(x, str) or not x.strip() for x in external)):
            raise SourceError("invalid_build_script_review")
        inputs = path_list(review["inputs"])
        for item in inputs:
            require(item)
        dirs = path_list(review["include_dirs"])
        for directory in dirs:
            if not directory.startswith(ROOTS) or not any(p.startswith(directory + "/") for p in inputs):
                raise SourceError("unreviewed_include_directory")
        include_dirs[scripts[path]] = dirs
        reviewed.add(path)
    if reviewed != set(scripts):
        raise SourceError("unreviewed_dynamic_build_script")
    return include_dirs, policy


def base_selection(entries):
    return {p for p in entries if not excluded_payload(p)
            and (p in REQUIRED or p in TOOLS or p.startswith(ROOTS))}


def c_preprocessor_text(data):
    # C translation phases 2/3 for dependency scanning only; keep source bytes
    # untouched and preserve quoted strings while removing comments. This makes
    # comments/line splices unable to hide a quoted out-of-projection header.
    data = data.replace(b"\\\r\n", b"").replace(b"\\\n", b"")
    tokens = rb'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|/\*[\s\S]*?\*/|//[^\r\n]*'''
    return re.sub(tokens, lambda m: b" " if m[0].startswith((b"/*", b"//")) else m[0], data)


def rust_structure(data):
    """Small fail-closed lexer for reviewed test scope, not a Rust parser.

    Strings/chars/raw strings and nested comments cannot supply delimiters or
    cfg markers. Keep original byte offsets; never transform shipping source.
    Only balanced token trees and plain top-level cfg(test) modules are admitted.
    """
    raw = re.compile(rb'(?:br|cr|r)(#{0,255})"')
    quoted = re.compile(rb'(?:b|c)?"(?:\\[\s\S]|[^"\\])*"')
    char = re.compile(rb"b?'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|[^\r\n])|[\xc2-\xf4][\x80-\xbf]+|[^\\'\r\n])'")
    identifier = re.compile(rb"(?:r#)?[A-Za-z_][A-Za-z0-9_]*")
    tokens, depths, pairs, stack = [], [], {}, []
    pos = 0
    while pos < len(data):
        start = pos
        if data[pos:pos + 1] in b" \t\r\n":
            pos += 1
            continue
        if data.startswith(b"//", pos):
            end = data.find(b"\n", pos)
            pos = len(data) if end < 0 else end
            continue
        if data.startswith(b"/*", pos):
            nesting = 1
            pos += 2
            while nesting and pos < len(data):
                if data.startswith(b"/*", pos):
                    nesting += 1
                    pos += 2
                elif data.startswith(b"*/", pos):
                    nesting -= 1
                    pos += 2
                else:
                    pos += 1
            if nesting:
                raise SourceError("external_test_scope_unverified")
            continue
        match = raw.match(data, pos)
        if match:
            end = data.find(b'"' + match[1], match.end())
            if end < 0:
                raise SourceError("external_test_scope_unverified")
            pos = end + 1 + len(match[1])
        else:
            match = quoted.match(data, pos) or char.match(data, pos) or identifier.match(data, pos)
            if match:
                pos = match.end()
            else:
                if data[pos] == 34 or data[pos] >= 128:
                    raise SourceError("external_test_scope_unverified")
                pos += 1
        value = data[start:pos]
        index = len(tokens)
        depths.append(len(stack))
        tokens.append((value, start, pos))
        if value in (b"(", b"[", b"{"):
            stack.append(index)
        elif value in (b")", b"]", b"}"):
            if not stack or tokens[stack[-1]][0] != {b")": b"(", b"]": b"[", b"}": b"{"}[value]:
                raise SourceError("external_test_scope_unverified")
            pairs[stack.pop()] = index
    if stack:
        raise SourceError("external_test_scope_unverified")
    return tokens, depths, pairs


def verify_test_scope(review, data, offset, selected, load, policy):
    tokens, depths, pairs = rust_structure(data)
    if not any(value == b"include_bytes" and start == offset for value, start, _ in tokens):
        raise SourceError("external_test_scope_unverified")
    scope = review["scope"]
    if not isinstance(scope, dict):
        raise SourceError("external_test_scope_unverified")
    if scope.get("kind") == "cfg_test_module":
        if (set(scope) != {"kind", "attribute_line", "module_line"}
                or any(type(scope[k]) is not int or scope[k] <= 0 for k in ("attribute_line", "module_line"))):
            raise SourceError("external_test_scope_unverified")
        marker = [b"#", b"[", b"cfg", b"(", b"test", b")", b"]", b"mod", b"tests", b"{"]
        for i in range(len(tokens) - len(marker) + 1):
            if depths[i] or [v for v, _, _ in tokens[i:i + len(marker)]] != marker:
                continue
            if i and tokens[i - 1][0] not in (b";", b"}"):
                continue  # no preceding attribute, macro arguments or partial item
            opening = i + len(marker) - 1
            if (data.count(b"\n", 0, tokens[i][1]) + 1 == scope["attribute_line"]
                    and data.count(b"\n", 0, tokens[i + 7][1]) + 1 == scope["module_line"]
                    and tokens[opening][1] < offset < tokens[pairs[opening]][1]):
                return
        raise SourceError("external_test_scope_unverified")
    if scope.get("kind") != "cargo_integration_test" or set(scope) != {"kind", "manifest", "manifest_sha256"}:
        raise SourceError("external_test_scope_unverified")
    # Only Cargo's immediate, auto-discovered tests/*.rs form is supported.
    match = re.fullmatch(r"(crates/[^/]+)/tests/[^/]+\.rs", review["source"])
    if not match or scope["manifest"] != match[1] + "/Cargo.toml" or scope["manifest"] not in selected:
        raise SourceError("external_test_scope_unverified")
    manifest = load(scope["manifest"])
    if scope["manifest_sha256"] != digest(manifest):
        raise SourceError("external_test_manifest_hash_mismatch")
    if toml(manifest).get("package", {}).get("autotests", True) is not True:
        raise SourceError("external_test_scope_unverified")
    # A tests/ filename does not suffice if any Cargo target/build script reuses
    # it. Rust include!/path reuse is rejected independently during closure.
    def reused(value, owner):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("path", "build") and isinstance(item, str):
                    if relative_input(owner, item) == review["source"]:
                        return True
                elif reused(item, owner):
                    return True
        elif isinstance(value, list):
            return any(reused(item, owner) for item in value)
        return False
    if (any(reused(toml(load(p)), p) for p in selected if p.endswith("Cargo.toml"))
            or any(review["source"] in r["inputs"] for r in policy["build_scripts"])):
        raise SourceError("external_test_scope_unverified")


def external_test_reviews(entries, selected, load, policy):
    """Exact reference reviews; target provenance uses tree metadata ONLY."""
    reviews = policy.get("external_test_dependencies", [])
    if not isinstance(reviews, list) or len(reviews) > 256:
        raise SourceError("invalid_external_test_dependencies")
    result = {}
    for review in reviews:
        if not isinstance(review, dict) or set(review) != {
                "source", "source_sha256", "line", "column", "reference", "target",
                "target_git_blob", "target_bytes", "scope", "reason"}:
            raise SourceError("invalid_external_test_review")
        path, target = validate_path(review["source"]), validate_path(review["target"])
        if (path not in selected or not path.endswith(".rs") or not path.startswith("crates/")
                or not excluded_payload(target) or not target.startswith("testdata/")):
            raise SourceError("invalid_external_test_review")
        data = load(path)
        if not isinstance(review["source_sha256"], str) or review["source_sha256"] != digest(data):
            raise SourceError("external_test_source_hash_mismatch")
        if (type(review["target_bytes"]) is not int or review["target_bytes"] < 0
                or target not in entries or review["target_git_blob"] != entries[target]["git_blob"]
                or review["target_bytes"] != entries[target]["bytes"]):
            raise SourceError("external_test_provenance_mismatch")
        if (any(type(review[k]) is not int or review[k] <= 0 for k in ("line", "column"))
                or not isinstance(review["reference"], str) or not review["reference"].isascii()
                or relative_input(path, review["reference"]) != target
                or not isinstance(review["reason"], str) or not review["reason"].strip()):
            raise SourceError("invalid_external_test_review")
        lines = data.split(b"\n")
        if review["line"] > len(lines) or review["column"] > len(lines[review["line"] - 1]):
            raise SourceError("stale_external_test_reference")
        offset = sum(len(line) + 1 for line in lines[:review["line"] - 1]) + review["column"] - 1
        start = INCLUDE_START.match(data, offset)
        literal = INCLUDE_LITERAL.match(data, start.end()) if start else None
        if (not start or not start[0].startswith(b"include_bytes!") or not literal
                or literal[1] != review["reference"].encode("ascii")):
            raise SourceError("stale_external_test_reference")
        verify_test_scope(review, data, offset, selected, load, policy)
        key = path, offset
        if key in result:
            raise SourceError("duplicate_external_test_review")
        result[key] = review
    return result


def complete_inputs(entries, selected, load, external_test_dependencies=None):
    blockers = []
    include_dirs, policy = input_policy(entries, selected, load)
    reviews = external_test_reviews(entries, selected, load, policy)
    integration_sources = {r["source"] for r in reviews.values() if r["scope"]["kind"] == "cargo_integration_test"}
    consumed = set()
    # Conservative static closure, not a Rust/C preprocessor. Unsupported
    # spellings fail closed; no build scripts or compiler are run to discover it.
    pending = sorted(selected)
    visited = set()
    def include(target):
        if target in integration_sources:
            raise SourceError("external_test_source_reused")
        if excluded_payload(target):
            if "excluded_model_build_input" not in blockers:
                blockers.append("excluded_model_build_input")
            return  # keep diagnostic/source text, never load the payload
        if not allowed_input(target):
            raise SourceError("include_outside_projection")
        if target not in entries:
            raise SourceError("missing_build_reference")
        selected.add(target)
        if target not in visited:
            pending.append(target)
    def reference(path, raw):
        try:
            return relative_input(path, raw.decode("ascii"))
        except UnicodeError:
            raise SourceError("unsafe_build_reference") from None
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        data = load(path)
        if path.endswith(".rs"):
            for start in INCLUDE_START.finditer(data):
                literal = INCLUDE_LITERAL.match(data, start.end())
                if not literal:
                    raise SourceError("unsupported_dynamic_include")
                key = path, start.start()
                if key in reviews:
                    consumed.add(key)  # verified test-only external requirement, never read/shipped
                else:
                    include(reference(path, literal[1]))
            if re.search(rb'#\s*\[\s*cfg_attr\b[^\]]*\bpath\b', data):
                raise SourceError("unsupported_conditional_rust_path")
            for start in RUST_PATH_START.finditer(data):
                literal = RUST_PATH_LITERAL.match(data, start.end())
                if not literal:
                    raise SourceError("unsupported_rust_path")
                include(reference(path, literal[1]))
        if path.endswith(C_SUFFIXES):
            # Honor only reviewed include search directories of this crate.
            directories = next((dirs for owner, dirs in include_dirs.items() if path.startswith(owner)), [])
            for match in C_INCLUDE.finditer(c_preprocessor_text(data)):
                if match[1] != b"include":
                    raise SourceError("unsupported_c_include_directive")
                value = match[2].strip()
                if re.match(rb'<[^<>\r\n]+>', value):
                    continue  # explicitly external system/toolchain prerequisite
                quoted = re.match(rb'"([^"\\]+)"', value)
                if not quoted:
                    raise SourceError("unsupported_dynamic_c_include")
                candidates = [reference(path, quoted[1])]
                candidates.extend(reference(d + "/_", quoted[1]) for d in directories)
                target = next((p for p in candidates if p in entries), candidates[0])
                include(target)
    if consumed != set(reviews):
        raise SourceError("stale_external_test_reference")
    if external_test_dependencies is not None:
        external_test_dependencies.extend(reviews[key] for key in sorted(reviews))
    return blockers


def inspect_source(repo, commit):
    """Inspect a literal real commit; this API never writes a file or runs code."""
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise SourceError("commit_must_be_40_lowercase_hex")
    if git(repo, "cat-file", "-t", commit) != b"commit\n":
        raise SourceError("object_is_not_commit")
    commit_bytes = git(repo, "cat-file", "commit", commit)
    if git_digest("commit", commit_bytes) != commit:
        raise SourceError("commit_integrity")
    first = commit_bytes.split(b"\n", 1)[0]
    if not re.fullmatch(rb"tree [0-9a-f]{40}", first):
        raise SourceError("invalid_commit_tree")
    tree = first[5:].decode()
    rows = git(repo, "ls-tree", "-rlz", "--full-tree", tree).split(b"\0")
    if len(rows) - 1 > MAX_ENTRIES:
        raise SourceError("tree_entry_limit")
    entries = {}
    folded = set()
    for row in rows:
        if not row:
            continue
        try:
            meta, name = row.split(b"\t", 1)
            mode, kind, oid, size = meta.split()
            path = name.decode("utf-8", "surrogateescape")
        except (ValueError, UnicodeError):
            raise SourceError("invalid_tree_entry") from None
        # Excluded private inventory may contain spaces/non-ASCII. Only its
        # canonical digest is published, never its names or contents. Selected
        # archive names are validated below and at every include boundary.
        if path.startswith("/") or any(p in (".", "..", ".git") for p in path.split("/")):
            raise SourceError("unsafe_path")
        if mode not in (b"100644", b"100755") or kind != b"blob":
            raise SourceError("symlink_submodule_or_special_entry")
        if path.casefold() in folded:
            raise SourceError("duplicate_or_case_colliding_path")
        if not HEX40.fullmatch(oid.decode()) or not size.isdigit():
            raise SourceError("invalid_tree_entry")
        folded.add(path.casefold())
        entries[path] = {"git_mode": mode.decode(), "git_blob": oid.decode(), "bytes": int(size)}
    if any(p not in entries for p in REQUIRED) or not any(p.startswith("crates/") for p in entries):
        raise SourceError("missing_required_build_input")
    selected = base_selection(entries)
    blockers = [] if TEMPLATE in selected else ["missing_build_template"]
    for path in selected:
        validate_path(path)
    payload = {}
    total = 0

    def load(path):
        nonlocal total
        if excluded_payload(path):
            raise SourceError("excluded_model_build_input")
        if path not in payload:
            if path not in entries:
                raise SourceError("missing_build_reference")
            total += entries[path]["bytes"]
            if total > MAX_SOURCE_BYTES:
                raise SourceError("source_byte_limit")
            payload[path] = read_blob(repo, entries[path])
        return payload[path]

    external_tests = []
    blockers.extend(complete_inputs(entries, selected, load, external_tests))
    workspace = toml(load("Cargo.toml")).get("workspace", {})
    if not isinstance(workspace, dict) or not isinstance(workspace.get("package"), dict):
        raise SourceError("invalid_workspace_table")
    if workspace["package"].get("license") != "AGPL-3.0-only":
        raise SourceError("unexpected_engine_license")
    members = workspace.get("members", [])
    if not members or not isinstance(members, list):
        raise SourceError("missing_workspace_members")
    for member in members:
        if not isinstance(member, str) or not member.startswith("crates/") or member + "/Cargo.toml" not in selected:
            raise SourceError("workspace_member_outside_projection")
    # Check local Cargo paths without executing cargo metadata or build scripts.
    def check_paths(value, owner):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "path":
                    target = relative_input(owner, item)
                    if target not in selected and target + "/Cargo.toml" not in selected:
                        raise SourceError("cargo_path_outside_projection")
                else:
                    check_paths(item, owner)
        elif isinstance(value, list):
            for item in value:
                check_paths(item, owner)
    for p in selected:
        if p.endswith("Cargo.toml"):
            check_paths(toml(load(p)), p)
    toolchain = toml(load("rust-toolchain.toml")).get("toolchain", {})
    if not isinstance(toolchain, dict):
        raise SourceError("invalid_toolchain_table")
    channel = toolchain.get("channel")
    if not isinstance(channel, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", channel):
        raise SourceError("unpinned_rust_toolchain")
    for p in sorted(selected):
        if load(p).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise SourceError("unresolved_lfs_build_input")
    excluded = [{"path": p, **entries[p]} for p in sorted(entries) if p not in selected]
    return {"commit": commit, "tree": tree, "toolchain": channel, "entries": entries,
            "selected": sorted(selected), "payload": payload, **privacy_result(payload),
            "excluded_entries": len(excluded), "excluded_inventory_sha256": digest(canonical(excluded)),
            "blockers": sorted(blockers), "external_test_dependencies": external_tests}


def archive_bytes(source):
    if source["blockers"]:
        raise SourceError("source_not_exportable")
    files = dict(source["payload"])
    if set(files) != set(source["selected"]):
        raise SourceError("projection_integrity")
    for path, data in files.items():
        validate_path(path)
        if excluded_payload(path):
            raise SourceError("excluded_model_build_input")
        entry = source["entries"][path]
        if (len(data) != entry["bytes"] or git_digest("blob", data) != entry["git_blob"]
                or entry["git_mode"] not in ("100644", "100755")):
            raise SourceError("blob_integrity")
    # Recompute selection/closure and privacy from verified original bytes,
    # never caller-edited diagnostics or a selectively pruned payload mapping.
    expected = base_selection(source["entries"])
    def load(path):
        if path not in files:
            raise SourceError("projection_integrity")
        return files[path]
    external_tests = []
    if complete_inputs(source["entries"], expected, load, external_tests) or expected != set(files):
        raise SourceError("projection_integrity")
    privacy = privacy_result(files)
    if privacy["unreviewed_findings"]:
        raise SourceError("source_not_exportable")
    template = files[TEMPLATE]
    if external_tests and b"{{EXTERNAL_TEST_DEPENDENCIES}}" not in template:
        raise SourceError("missing_external_test_build_instructions")
    for key, value in (("COMMIT", source["commit"]), ("TREE", source["tree"]), ("TOOLCHAIN", source["toolchain"])):
        template = template.replace(("{{" + key + "}}").encode(), value.encode())
    external_text = "\n".join(
        f'- `{r["source"]}:{r["line"]}:{r["column"]}` -> `{r["target"]}` '
        f'({r["scope"]["kind"]}; excluded Git blob `{r["target_git_blob"]}`, {r["target_bytes"]} bytes).'
        for r in external_tests) or "No reviewed compile-time external test dependencies in this commit."
    template = template.replace(b"{{EXTERNAL_TEST_DEPENDENCIES}}", external_text.encode())
    if b"{{" in template:
        raise SourceError("unresolved_build_template")
    files["SOURCE_BUILD.md"] = template
    files[".rexl3_commit"] = (source["commit"] + "\n").encode()
    # Scan generated content too, without source edits or ignore lists.
    if scan_file("SOURCE_BUILD.md", template):
        raise SourceError("unsafe_build_template")
    records = {}
    for p, data in sorted(files.items()):
        original = source["entries"].get(p) if p in source["selected"] else None
        mode = "0755" if original and original["git_mode"] == "100755" else "0644"
        records[p] = {"bytes": len(data), "sha256": digest(data), "archive_mode": mode,
                      "git_blob": original["git_blob"] if original else None,
                      "git_mode": original["git_mode"] if original else None,
                      "origin": "git_blob" if original else "generated"}
    records["SOURCE_BUILD.md"]["template_git_blob"] = source["entries"][TEMPLATE]["git_blob"]
    identity = {"schema": SCHEMA, "source_commit": source["commit"], "source_tree": source["tree"],
                "license": "AGPL-3.0-only", "rust_toolchain": source["toolchain"],
                "qualification_claimed": False,
                "privacy": {"status": privacy["privacy_status"], "policy": "conservative-v1",
                            "source_rewritten": False, "release_authorized": False,
                            "findings": privacy["findings"], "reviewed_findings": privacy["reviewed_findings"],
                            "unreviewed_findings": privacy["unreviewed_findings"],
                            "allowance_policy": ALLOWANCES if ALLOWANCES in files else None,
                            "allowance_policy_sha256": digest(files[ALLOWANCES]) if ALLOWANCES in files else None},
                "projection": {"required_files": list(REQUIRED), "source_subtrees": list(ROOTS),
                               "excluded_payload_suffixes": list(MODEL_SUFFIXES),
                               "external_asset_tests": "test source retained; excluded asset-dependent tests require separately licensed assets",
                               "external_test_dependencies": external_tests,
                               "native_build_source_complete": True,
                               "input_manifest": INPUTS if INPUTS in files else None,
                               "export_tools": list(TOOLS),
                               "include_closure": "literal Rust includes/path attributes, quoted C headers, reviewed Cargo build scripts",
                               "excluded_entries": source["excluded_entries"],
                               "excluded_inventory_sha256": source["excluded_inventory_sha256"],
                               "symlinks_and_submodules": "rejected throughout commit",
                               "cargo_registry_dependencies": "locked, not vendored",
                               "git_history": "not exported"},
                "serialization": {"tar": "USTAR", "mtime": 0, "uid": 0, "gid": 0,
                                  "gzip": "mtime=0; no filename; level=9", "order": "ASCII path"},
                "manifest_hash_scope": "all archive files except SOURCE_IDENTITY.json itself",
                "files": records}
    files["SOURCE_IDENTITY.json"] = canonical(identity)
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
            for p, data in sorted(files.items()):
                info = tarfile.TarInfo(p)
                info.size = len(data)
                info.mode = int(records[p]["archive_mode"], 8) if p in records else 0o644
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
    return output.getvalue()


@contextmanager
def output_parent(path):
    """Walk parent descriptors without following symlinks (Linux/POSIX).

    No parent directories are created. Exclusive open/check happens against the
    same anchored parent descriptor, not against a second pathname resolution.
    """
    p = Path(path)
    if ".." in p.parts or p.name in ("", ".", ".."):
        raise SourceError("unsafe_output_path")
    p = Path.cwd() / p if not p.is_absolute() else p
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in p.parts[1:-1]:
            new_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new_fd
        yield fd, p.name
    finally:
        os.close(fd)


def linux_primitive(name):
    """No check/replace or named-stage fallback on unsupported platforms."""
    if sys.platform != "linux" or not hasattr(os, "O_TMPFILE"):
        raise SourceError("atomic_output_unsupported")
    try:
        fn = getattr(ctypes.CDLL(None, use_errno=True), name)
    except AttributeError:
        raise SourceError("atomic_output_unsupported") from None
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def install_fd(fd, parent, name):
    # AT_EMPTY_PATH links the anonymous inode, not a substitutable staging name.
    # linkat is intrinsically no-replacement, including symlinks at the target.
    if linux_primitive("linkat")(fd, b"", parent, os.fsencode(name), 0x1000):
        error = ctypes.get_errno()
        raise OSError(error, "atomic_install_failed")


def rename_noreplace(src_parent, src, dst_parent, dst):
    if linux_primitive("renameat2")(src_parent, os.fsencode(src), dst_parent, os.fsencode(dst), 1):
        error = ctypes.get_errno()
        raise OSError(error, "atomic_capture_failed")


def inode(info):
    return info.st_dev, info.st_ino


def snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_output(fd, data, links=1):
    before = os.fstat(fd)
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != links or before.st_size != len(data)):
        raise SourceError("unsafe_existing_archive")
    offset = 0
    while offset < len(data):
        chunk = os.pread(fd, min(1024 * 1024, len(data) - offset), offset)
        if not chunk or chunk != data[offset:offset + len(chunk)]:
            raise SourceError("archive_byte_mismatch")
        offset += len(chunk)
    if snapshot(before) != snapshot(os.fstat(fd)):
        raise SourceError("output_changed_during_read")
    return before


def verify_parent(path, parent):
    try:
        with output_parent(path) as (current, _):
            if inode(os.fstat(current)) != inode(os.fstat(parent)):
                raise SourceError("output_parent_moved")
    except OSError:
        raise SourceError("output_parent_moved") from None


def remove_owned(parent, name, owned):
    """Capture before unlink; never unlink a publicly substitutable name.

    The private recovery directory (0700, random name) is exclusive to this
    operation. Other processes must not tamper with this recovery namespace.
    One foreign capture at most is retained if no-replace restoration collides;
    its presence is a hard, explicit cleanup error, never reported as success.
    """
    try:
        if inode(os.stat(name, dir_fd=parent, follow_symlinks=False)) != owned:
            return
    except FileNotFoundError:
        return
    recovery = ".source-recovery-" + os.urandom(16).hex()
    os.mkdir(recovery, 0o700, dir_fd=parent)
    recovery_fd = os.open(recovery, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    recovery_id = inode(os.fstat(recovery_fd))
    retained = False
    try:
        try:
            rename_noreplace(parent, name, recovery_fd, "captured")
        except FileNotFoundError:
            return
        captured = os.stat("captured", dir_fd=recovery_fd, follow_symlinks=False)
        if inode(captured) == owned:
            os.unlink("captured", dir_fd=recovery_fd)
        else:
            # A foreign replacement raced the observation above. Put it back
            # only if the public name is still absent; otherwise preserve it.
            try:
                rename_noreplace(recovery_fd, "captured", parent, name)
            except OSError:
                retained = True
                raise SourceError("output_cleanup_quarantined") from None
    finally:
        durability_error = False
        try:
            try:
                os.fsync(recovery_fd)
            except OSError:
                durability_error = True
            # Even persistent fsync failure must not prevent bounded owned
            # namespace cleanup. It still prevents any durability/success claim.
            try:
                if not retained:
                    if inode(os.stat(recovery, dir_fd=parent, follow_symlinks=False)) != recovery_id:
                        raise SourceError("output_cleanup_directory_changed")
                    os.rmdir(recovery, dir_fd=parent)
            finally:
                try:
                    os.fsync(parent)
                except OSError:
                    durability_error = True
        finally:
            os.close(recovery_fd)
        if durability_error and not retained:
            raise SourceError("output_cleanup_durability_failed") from None


def store_or_check(path, data, check=False):
    """Point-in-time admission, not post-return immutability against the owner.

    Linux O_TMPFILE + linkat(AT_EMPTY_PATH) are required for creation. The file
    has no pathname until all bytes and its inode have been fsynced. Parent
    movement, descriptor/name drift and metadata mutation fail closed. This is
    one artifact, not a transaction with stdout or other release files.
    """
    path = Path.cwd() / path if not Path(path).is_absolute() else Path(path)
    with output_parent(path) as (parent, name):
        if check:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        else:
            linux_primitive("renameat2")  # require safe rollback before writing
            linux_primitive("linkat")
            fd = os.open(".", os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC, 0o600, dir_fd=parent)
        owned = inode(os.fstat(fd))  # establish ownership before the first write
        try:
            if not check:
                os.fchmod(fd, 0o600)
                offset = 0
                while offset < len(data):
                    count = os.write(fd, memoryview(data)[offset:])
                    if count <= 0:
                        raise SourceError("short_output_write")
                    offset += count
                os.fsync(fd)
                verify_parent(path, parent)
                install_fd(fd, parent, name)
                os.fsync(parent)
            admitted = read_output(fd, data)
            verify_parent(path, parent)
            # This is deliberately AFTER the last read and parent check. Bind
            # the admitted bytes to both the held descriptor and public name.
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if snapshot(current) != snapshot(admitted) or snapshot(os.fstat(fd)) != snapshot(admitted):
                raise SourceError("output_changed_after_read")
        except BaseException:
            if not check:
                # Also handles an install that completed but then raised.
                remove_owned(parent, name, owned)
            raise
        finally:
            os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True, help="literal 40-character lowercase Git commit ID")
    parser.add_argument("--output", help="new .tar.gz path (or existing path with --check)")
    parser.add_argument("--check", action="store_true", help="recreate and compare every byte; never overwrite")
    parser.add_argument("--diagnose", action="store_true", help="redacted report only; NEVER create an archive")
    args = parser.parse_args(argv)
    try:
        if args.diagnose and (args.output or args.check) or not args.diagnose and not args.output:
            raise SourceError("choose_diagnostic_or_output")
        source = inspect_source(args.repo, args.commit)
        report = {"schema": SCHEMA, "source_commit": source["commit"], "source_tree": source["tree"],
                  "publishable": not source["unreviewed_findings"] and not source["blockers"], "archive_created": False,
                  "selected_files": len(source["selected"]), "findings": source["findings"], "blockers": source["blockers"],
                  "reviewed_findings": source["reviewed_findings"], "unreviewed_findings": source["unreviewed_findings"],
                  "privacy_status": source["privacy_status"], "release_authorized": False,
                  "external_test_dependencies": source["external_test_dependencies"]}
        if args.diagnose or source["unreviewed_findings"] or source["blockers"]:
            print(canonical(report).decode(), end="")
            return 3 if source["unreviewed_findings"] else 2 if source["blockers"] else 0
        data = archive_bytes(source)
        store_or_check(args.output, data, args.check)
        print(json.dumps({"status": "identical" if args.check else "created", "source_commit": args.commit,
                          "sha256": digest(data), "bytes": len(data), "source_files": len(source["selected"])}))
        return 0
    except SourceError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    except (OSError, ValueError, tarfile.TarError):
        # Do not leak host paths, filenames, credentials, or subprocess output.
        print('{"error": "archive_io_or_format_error"}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
