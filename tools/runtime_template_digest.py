#!/usr/bin/env python3
"""Content digest of a runtime template directory, and the parser for its approval file.

The digest is sha256 over the sorted records ``path\\0sha256(content)\\n`` for every regular
file in the template. It is what a generated runtime tree records in
``source.manifest.json`` and what production compares against its approved list, so a
template edit that nobody approved cannot ride along inside a bundle.

CLI: python3 tools/runtime_template_digest.py runtimes/template-v1
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCHEMA = "rexl3.template-approval.v1"
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}
# the approval is not part of what it approves, so it must not feed its own digest
SKIP_FILES = {"TEMPLATE.json"}
REQUIRED_KEYS = ("schema", "version", "digest", "approved_commit", "approved_at", "approved_by")


def read_tree(path):
    """Every regular file under ``path``, as {posix relative path: bytes}. Refuses symlinks."""
    out = {}
    if os.path.islink(path) or not os.path.isdir(path):
        raise SystemExit(f"template directory required without symlink: {path}")
    for base, dirs, files in os.walk(path):
        for name in dirs:
            if os.path.islink(os.path.join(base, name)):
                raise SystemExit(f"template contains symlink directory: {name}")
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            full = os.path.join(base, name)
            if os.path.islink(full) or not os.path.isfile(full):
                raise SystemExit(f"template contains a non-regular file: {full}")
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            if rel in SKIP_FILES:
                continue
            with open(full, "rb") as handle:
                out[rel] = handle.read()
    if not out:
        raise SystemExit(f"template directory is empty: {path}")
    return out


def digest_tree(tree):
    """``sha256:<hex>`` over sorted ``path\\0sha256(bytes)\\n`` records."""
    hasher = hashlib.sha256()
    for rel in sorted(tree):
        hasher.update(rel.encode() + b"\0" + hashlib.sha256(tree[rel]).hexdigest().encode() + b"\n")
    return "sha256:" + hasher.hexdigest()


def read_approval(path):
    """Load and validate a template approval file; exits 2 with a named reason when it is unusable."""
    if not os.path.isfile(path):
        raise SystemExit(f"template approval missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        approval = json.load(handle)
    missing = [key for key in REQUIRED_KEYS if key not in approval]
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")
    if approval["schema"] != SCHEMA:
        raise SystemExit(f"{path}: schema is {approval['schema']}, expected {SCHEMA}")
    digest = approval["digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise SystemExit(f"{path}: digest is not sha256:<64 hex>")
    if not isinstance(approval['approved_commit'], str) or not re.fullmatch(r'[0-9a-f]{40}', approval['approved_commit']):
        raise SystemExit(f"{path}: full approved_commit required")
    return approval


def verified_identity(template_dir, *, repo, commit):
    """Resolve real Git objects and recompute current AND approved template bytes."""
    try:
        from . import release_evidence as e
    except ImportError:
        import release_evidence as e
    root = e.checked_path(repo)
    directory = e.checked_path(template_dir)
    e.require(directory.is_relative_to(root), 'template must be in source repo')
    rel = directory.relative_to(root).as_posix()
    e.relative_path(rel)
    approval_path = e.safe_file(directory, 'TEMPLATE.json')
    approval = e.load_json(approval_path)
    e.require(approval.get('schema') == SCHEMA, 'template approval schema')
    e.git_identity(root, commit)
    e.git_identity(root, approval.get('approved_commit'))
    actual = read_tree(directory)
    e.require(digest_tree(actual) == approval.get('digest'), 'template digest drift')
    e.require(e.git_file(root, commit, rel + '/TEMPLATE.json') == approval_path.read_bytes(), 'uncommitted template approval')
    for revision in (commit, approval['approved_commit']):
        rows = e.git(root, 'ls-tree', '-rz', revision, '--', rel + '/').split(b'\0')
        snapshot = {}
        for row in filter(None, rows):
            header, name = row.split(b'\t', 1)
            mode, kind, oid = header.split()
            e.require(mode in (b'100644', b'100755') and kind == b'blob', 'nonregular template Git member')
            local = name.decode()[len(rel) + 1:]
            if local in SKIP_FILES or any(part in SKIP_DIRS for part in Path(local).parts):
                continue
            snapshot[local] = e.git(root, 'cat-file', 'blob', oid.decode())
        e.require(snapshot == actual, 'template commit content mismatch')
    return {key: approval[key] for key in ('version', 'digest', 'approved_commit')} | {'path': rel}


def main(argv):
    if len(argv) != 2:
        print("usage: runtime_template_digest.py <template-dir>", file=sys.stderr)
        return 2
    print(digest_tree(read_tree(argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
