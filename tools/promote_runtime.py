#!/usr/bin/env python3
"""Fail-closed promotion using protected verifier code and pinned approval.

Prepares the model branch `model/<model>` carrying `runtimes/<family>/<model>/v<N>/`
(the sealed tree), MODEL.json (schema rexl3.prod-model.v1) and LATEST. Dev release
tags derive from the registry entry's runtime_id (approve-<runtime_id>-<N>), never
from folder names; MODEL.json and the sealed approval must agree. Per-runtime
production releases (prod/<runtime-id>/v<N>) are retired: this tool never creates a
release or a tag, never edits rulesets or settings, and requires the exact merged
target commit.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from urllib.parse import quote
try:
    from . import release_evidence as e
    from . import model_layout as m
    from .verify_runtime_tree import verify_tree, tree_inventory
except ImportError:
    import release_evidence as e
    import model_layout as m
    from verify_runtime_tree import verify_tree, tree_inventory

DEV_REPO = 'REXL3/REXL3-dev'
PROD_REPO = 'REXL3/REXL3'
validate_request = e.validate_request


def _run(command, cwd=None, *, not_found=False):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        if not_found and 'HTTP 404' in result.stderr:
            return None
        raise ValueError('command failed: ' + ' '.join(command) + ': ' + result.stderr.strip())
    return result


def gh(*args, cwd=None, not_found=False):
    return _run(['gh', *map(str, args)], cwd=cwd, not_found=not_found)


class ReleaseAssetCache:
    """Thread-safe transport cache; every consumer still authenticates copied bytes."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._guard = threading.Lock()
        self._locks = {}

    def _lock(self, key):
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def fetch(self, tag, repo, name, dest):
        e.require(e.is_engine_tag(tag) and repo == DEV_REPO, 'shared cache is only for unified dev releases')
        e.relative_path(name)
        e.require('/' not in name, 'cached asset must be a basename')
        key = (repo, tag, name)
        with self._lock(key):
            folder = self.root / __import__('hashlib').sha256((repo + '/' + tag).encode()).hexdigest()
            folder.mkdir(exist_ok=True)
            cached = e.checked_path(folder / name)
            if not cached.exists():
                with tempfile.TemporaryDirectory(prefix='.shared-download-', dir=self.root) as tmp:
                    gh('release', 'download', tag, '-R', repo, '-p', name, '-D', tmp)
                    source = e.safe_file(Path(tmp), name)
                    with source.open('rb') as incoming, cached.open('xb') as output:
                        shutil.copyfileobj(incoming, output)
                        output.flush()
                        os.fsync(output.fileno())
            target = e.checked_path(Path(dest) / name)
            if target.exists():
                e.require(e.sha256_file(target) == e.sha256_file(cached), 'shared asset destination conflict')
            else:
                with cached.open('rb') as incoming, target.open('xb') as output:
                    shutil.copyfileobj(incoming, output)
            return target


def git(*args, cwd):
    return _run(['git', '--no-replace-objects', *map(str, args)], cwd=cwd)


def release_metadata(tag, repo, *, missing_ok=False):
    result = gh('api', f'repos/{repo}/releases/tags/{quote(tag, safe="")}', not_found=missing_ok)
    if result is None:
        return None
    doc = e.loads(result.stdout)
    e.require(type(doc.get('id')) is int and doc['id'] > 0 and doc.get('tag_name') == tag, 'release id/tag mismatch')
    e.require(doc.get('draft') is False and doc.get('prerelease') is False, 'usable published release required')
    e.require(isinstance(doc.get('assets'), list) and doc['assets'], 'release assets missing')
    seen, ids = set(), set()
    for asset in doc['assets']:
        name = e.relative_path(asset.get('name'))
        e.require('/' not in name and name not in seen, 'unsafe/duplicate release asset name')
        seen.add(name)
        e.require(type(asset.get('id')) is int and asset['id'] > 0 and asset['id'] not in ids
                  and type(asset.get('size')) is int and asset['size'] > 0,
                  'release asset id/size required')
        ids.add(asset['id'])
    return stable_release(doc)


def stable_release(doc):
    return {
        'id': doc['id'], 'tag_name': doc['tag_name'], 'target_commitish': doc.get('target_commitish'),
        'draft': doc['draft'], 'prerelease': doc['prerelease'],
        'assets': [{'id': a['id'], 'name': a['name'], 'size': a['size'], 'digest': a.get('digest'),
                    'state': a.get('state')} for a in doc['assets']],
    }


def tag_commit(tag, repo):
    data = e.loads(gh('api', f'repos/{repo}/git/ref/tags/{quote(tag, safe="")}').stdout)['object']
    for _ in range(5):
        e.full_hash(data.get('sha'), e.HEX40)
        if data.get('type') == 'commit':
            return data['sha']
        e.require(data.get('type') == 'tag', 'tag does not resolve to commit')
        data = e.loads(gh('api', f'repos/{repo}/git/tags/{data["sha"]}').stdout)['object']
    raise ValueError('annotated tag chain limit')


def verify_approval(approval_path, approved_templates, registry, engine_dir=None, key=None, precomputed=None,
                    trusted_source_registry=None):
    return e.verify_assets(approval_path, registry=registry, approved_templates=approved_templates,
                           engine_dir=engine_dir, key=key, precomputed=precomputed,
                           trusted_source_registry=trusted_source_registry)


def precomputed_for_assets(approval, assets_dir):
    """Phase-entry inventory pass-through: one cheap streaming digest per archive.

    Streams SHA-256 and size of every compressed transport archive and requires
    equality with the approval record before returning the approved member
    inventories. Downstream phases may then skip decompression/re-derivation;
    the fresh digest remains mandatory at every verification boundary.
    """
    root = e.checked_path(assets_dir)
    precomputed = {}
    for asset in approval_assets(approval):
        path = e.safe_file(root, asset['archive'])
        e.require(path.stat().st_size == asset['bytes'], 'approved asset size differ: ' + asset['archive'])
        e.require(e.sha256_file(path) == asset['sha256'], 'approved asset bytes differ: ' + asset['archive'])
        precomputed[asset['archive']] = asset['members']
    return precomputed


def engine_release_tag(approval):
    engine = approval.get('engine_binaries')
    if engine is None:
        return None
    return next(iter(e.validate_engine_binaries(engine).values()))['release_tag']


def fetch_engine(approval, stage, *, repo, cache=None):
    """Download the engine release assets an approval references into ``stage`` and check them byte for byte."""
    tag = engine_release_tag(approval)
    if tag is None:
        return None
    engine = approval['engine_binaries']
    before = release_metadata(tag, repo)
    remote = {asset['name']: asset for asset in before['assets']}
    for item in engine.values():
        asset = remote.get(item['asset'])
        e.require(asset is not None and asset['size'] == item['bytes'], 'engine release asset missing or size differs')
        e.require(asset.get('digest') in (None, 'sha256:' + item['sha256']), 'engine release asset digest differs')
    stage.mkdir(parents=True)
    for item in engine.values():
        if cache is None:
            gh('release', 'download', tag, '-R', repo, '-p', item['asset'], '-D', stage)
        else:
            cache.fetch(tag, repo, item['asset'], stage)
    e.require({f.name for f in stage.iterdir()} == {item['asset'] for item in engine.values()}, 'engine download inventory mismatch')
    for item in engine.values():
        path = e.safe_file(stage, item['asset'])
        e.require(path.stat().st_size == item['bytes'] and e.sha256_file(path) == item['sha256'], 'engine asset bytes differ from the approval')
    if repo == DEV_REPO:
        e.require(tag_commit(tag, repo) == approval['source_commit'], 'engine release tag does not name the approved engine commit')
    e.require(release_metadata(tag, repo) == before, 'engine release changed while downloading')
    return before


def approval_assets(approval):
    return e.approval_assets(approval)


DEV_DOWNLOAD_RECEIPT = '.dev-download-identity.json'
MAX_DOWNLOAD_IDENTITY_BYTES = 64 * 1024
MAX_DOWNLOAD_ASSETS = 64


def _validate_download_identity(doc):
    e.require(isinstance(doc, dict) and set(doc) == {'schema', 'repo', 'release_id', 'tag',
              'target_commit', 'target_commitish', 'assets'}, 'invalid download identity')
    e.require(doc['schema'] == 'rexl3.dev-download.v1' and doc['repo'] == DEV_REPO, 'invalid download identity')
    e.require(type(doc['release_id']) is int and doc['release_id'] > 0, 'invalid download release id')
    e.require(isinstance(doc['tag'], str) and 0 < len(doc['tag']) <= 256, 'invalid download tag')
    e.full_hash(doc['target_commit'], e.HEX40)
    target = doc['target_commitish']
    e.require(isinstance(target, str) and 0 < len(target) <= 256 and
              all(32 < ord(c) < 127 for c in target), 'invalid download target')
    assets = doc['assets']
    e.require(isinstance(assets, dict) and 0 < len(assets) <= MAX_DOWNLOAD_ASSETS, 'invalid download asset count')
    ids = set()
    for name, asset in assets.items():
        e.relative_path(name)
        e.require('/' not in name and name != DEV_DOWNLOAD_RECEIPT, 'invalid download asset name')
        e.require(isinstance(asset, dict) and set(asset) == {'id', 'size', 'digest', 'sha256'}, 'invalid download asset identity')
        e.require(type(asset['id']) is int and asset['id'] > 0 and asset['id'] not in ids, 'invalid download asset id')
        ids.add(asset['id'])
        e.require(type(asset['size']) is int and 0 < asset['size'] <= e.MAX_BYTES, 'invalid download asset size')
        e.full_hash(asset['sha256'])
        e.require(asset['digest'] is None or asset['digest'] == 'sha256:' + asset['sha256'], 'invalid download asset digest')
    e.require(len(e.json_bytes(doc)) <= MAX_DOWNLOAD_IDENTITY_BYTES, 'download identity size limit')
    return doc


def _load_download_identity(path):
    path = e.safe_file(path.parent, path.name)
    e.require(path.stat().st_size <= MAX_DOWNLOAD_IDENTITY_BYTES and path.stat().st_mode & 0o777 == 0o600,
              'invalid download identity file')
    with path.open('rb') as stream:
        data = stream.read(MAX_DOWNLOAD_IDENTITY_BYTES + 1)
    e.require(len(data) <= MAX_DOWNLOAD_IDENTITY_BYTES, 'download identity size limit')
    return _validate_download_identity(e.loads(data))


def _store_download_identity(path, receipt):
    data = e.json_bytes(_validate_download_identity(receipt))
    if path.exists():
        e.require(_load_download_identity(path) == receipt, 'download retry identity drift')
        return
    with tempfile.TemporaryDirectory(prefix='.download-identity-', dir=path.parent) as work:
        pending = Path(work) / 'receipt.json'
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
        pending.unlink()
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    e.require(_load_download_identity(path) == receipt, 'download identity persistence failed')


def download_release(tag, dest, *, repo=DEV_REPO, registry, approved_templates, target_commit=None, key=None,
                     cache=None, trusted_source_registry=None):
    """Download with content-free failures; CLI/metadata errors may contain secrets."""
    try:
        return _download_release(tag, dest, repo=repo, registry=registry,
                                 approved_templates=approved_templates, target_commit=target_commit, key=key,
                                 cache=cache, trusted_source_registry=trusted_source_registry)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError):
        raise ValueError('release download verification refused') from None


def registry_key_for_tag(tag, repo):
    """The registry key a dev promotion tag can only ever belong to: approve-<id>-<N>."""
    pattern = r'approve-(.+)-([1-9][0-9]*)'
    match = re.fullmatch(pattern, tag) if isinstance(tag, str) else None
    e.require(match is not None and e.ID.fullmatch(match[1]), 'release tag is not a promotion tag')
    e.require(repo == DEV_REPO, 'per-runtime production releases are retired')
    return f'{match[1]}/v{match[2]}'


def _download_release(tag, dest, *, repo, registry, approved_templates, target_commit, key=None, cache=None,
                      trusted_source_registry=None):
    registry_key = key
    approvals = registry.get('approvals') if isinstance(registry, dict) else None
    e.require(isinstance(approvals, dict), 'independently trusted approval registry required')
    if registry_key is None and e.is_engine_tag(tag):
        matches = [name for name, entry in approvals.items()
                   if isinstance(entry, dict) and entry.get('approve_tag') == tag]
        e.require(len(matches) == 1, 'unified release needs an explicit family/model registry key')
        registry_key = matches[0]
    if registry_key is None:
        registry_key = registry_key_for_tag(tag, repo)
    e.require(registry_key in approvals, 'approval not in independently trusted registry: ' + registry_key)
    entry = approvals[registry_key]
    trust_key = registry_key if registry_key.count('/') == 2 else None
    e.validate_registry_entry(registry_key, entry, legacy=(registry_key.count('/') == 1))
    version = int(registry_key.rsplit('/v', 1)[1])
    runtime_id = entry.get('runtime_id')
    if registry_key.count('/') == 1:
        runtime_id = runtime_id or registry_key.rsplit('/v', 1)[0]
    e.validate_request(runtime_id, tag, version)
    if entry.get('approve_tag') is not None:
        e.require(entry['approve_tag'] == tag, 'release tag differs from the trusted registry')
    dest = e.checked_path(dest)
    receipt_path = e.checked_path(dest / DEV_DOWNLOAD_RECEIPT)
    previous = _load_download_identity(receipt_path) if repo == DEV_REPO and receipt_path.exists() else None
    before = release_metadata(tag, repo)
    e.require(before is not None, 'release not found')
    e.require(0 < len(before['assets']) <= MAX_DOWNLOAD_ASSETS, 'release asset count limit')
    for asset in before['assets']:
        e.require(asset['size'] <= e.MAX_BYTES, 'release asset size limit')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-', dir=dest.parent) as tmp:
        stage = Path(tmp) / 'release'
        stage.mkdir()
        unified = e.is_engine_tag(tag)
        approval = None
        remote_names = {asset['name'] for asset in before['assets']}
        if unified:
            # Fetch only this model's sealed approval first. Its pinned bytes, not a
            # candidate-controlled tag or URL, determine the remaining asset names.
            approval_name = f'{runtime_id}-v{version}-approval.json'
            e.require(approval_name in remote_names, 'unified release approval missing')
            if cache is None:
                gh('release', 'download', tag, '-R', repo, '-p', approval_name, '-D', stage)
            else:
                cache.fetch(tag, repo, approval_name, stage)
            prefixed_approval = e.safe_file(stage, approval_name)
            approval = e.trusted_entry(prefixed_approval, registry, key=trust_key)
            e.require(approval.get('approve_tag') == tag, 'unified approval tag mismatch')
            assets = approval_assets(approval)
            archive_names = {asset['archive'] for asset in assets}
            e.require(archive_names <= remote_names, 'unified release assets missing')
            if cache is None:
                command = ['release', 'download', tag, '-R', repo]
                for name in sorted(archive_names):
                    command.extend(('-p', name))
                gh(*command, '-D', stage)
            else:
                for name in sorted(archive_names):
                    cache.fetch(tag, repo, name, stage)
            os.replace(prefixed_approval, stage / 'approval.json')
            selected_assets = [asset for asset in before['assets']
                               if asset['name'] == approval_name or asset['name'] in archive_names]
        else:
            gh('release', 'download', tag, '-R', repo, '-D', stage)
            assets = None
            selected_assets = before['assets']
        approval_path = e.safe_file(stage, 'approval.json')
        if approval is None:
            approval = e.trusted_entry(approval_path, registry, key=trust_key)
        e.validate_request(approval.get('runtime_id'), approval.get('approve_tag'), approval.get('version'))
        engine_stage = Path(tmp) / 'engine'
        engine_release = fetch_engine(approval, engine_stage, repo=repo, cache=cache)
        if assets is None:
            assets = approval_assets(approval)
        names = {'approval.json'} | {asset['archive'] for asset in assets}
        if unified:
            e.require(names <= ({'approval.json'} | remote_names), 'unexpected/missing release assets')
        else:
            e.require(remote_names == names, 'unexpected/missing release assets')
        e.require({f.name for f in stage.iterdir()} == names, 'download inventory mismatch')
        measured = {}
        for asset in selected_assets:
            local_name = 'approval.json' if unified and asset['name'].endswith('-approval.json') else asset['name']
            path = e.safe_file(stage, local_name)
            e.require(path.stat().st_size == asset['size'], 'download size mismatch')
            measured[asset['name']] = e.sha256_file(path)
            if asset.get('digest') is not None:
                e.require(asset['digest'] == 'sha256:' + measured[asset['name']], 'GitHub asset digest mismatch')
        for asset in assets:
            e.require(measured[asset['archive']] == asset['sha256'] and
                      (stage / asset['archive']).stat().st_size == asset['bytes'], 'approved asset bytes differ')
        evidence = approval['evidence_asset']
        transport = e.safe_file(stage, evidence['archive'])
        e.require(e.asset_record(transport) == evidence, 'evidence archive mismatch')
        e.require(not set(evidence['members']) & (names | {DEV_DOWNLOAD_RECEIPT}), 'evidence collides with release assets/state')
        e.require(not any(name.startswith('engine/') for name in evidence['members']), 'evidence collides with the engine directory')
        e.extract_files(transport, stage, evidence['members'])
        verify_approval(approval_path, approved_templates, registry,
                        engine_dir=engine_stage if engine_release else None, key=trust_key,
                        trusted_source_registry=trusted_source_registry)
        expected_commit = approval['source_commit']
        e.full_hash(expected_commit, e.HEX40)
        expected_tag = approval['approve_tag']
        e.require(tag == expected_tag and tag_commit(tag, repo) == expected_commit, 'release tag/commit identity drift')
        after = release_metadata(tag, repo)
        e.require(after == before, 'release changed while downloading')
        receipt = _validate_download_identity(dict(schema='rexl3.dev-download.v1', repo=repo,
            release_id=before['id'], tag=tag, target_commit=expected_commit,
            target_commitish=before.get('target_commitish'), assets={asset['name']: dict(
                id=asset['id'], size=asset['size'], digest=asset.get('digest'), sha256=measured[asset['name']])
                for asset in selected_assets}))
        e.require(previous is None or previous == receipt, 'download retry identity drift')
        files = {p.relative_to(stage).as_posix(): p for p in stage.rglob('*') if p.is_file()}
        if engine_release:
            files.update({'engine/' + p.name: p for p in engine_stage.iterdir()})
        for rel, source in files.items():
            target = e.checked_path(dest / rel)
            if target.exists():
                e.require(e.sha256_file(target) == e.sha256_file(source), 'cached download conflict: ' + rel)
        dest.mkdir(parents=True, exist_ok=True)
        _store_download_identity(receipt_path, receipt)
        for rel, source in files.items():
            target = dest / rel
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as incoming, target.open('xb') as output:
                shutil.copyfileobj(incoming, output)
                output.flush()
                os.fsync(output.fileno())
            target.chmod(0o644)
        for rel, source in files.items():
            target = e.safe_file(dest, rel)
            e.require(target.stat().st_size == source.stat().st_size and
                      e.sha256_file(target) == e.sha256_file(source), 'cached download bytes differ')
        verify_approval(dest / 'approval.json', approved_templates, registry,
                        engine_dir=dest / 'engine' if engine_release else None, key=trust_key,
                        precomputed=precomputed_for_assets(approval, dest),
                        trusted_source_registry=trusted_source_registry)
        e.require(_load_download_identity(receipt_path) == receipt, 'download identity persistence failed')
    return dest / 'approval.json'


def extract_tree(archive, dest, approval, *, approval_path=None, preverified=False):
    """Preflight ALL headers, even excluded native entries, before creating dest.

    ``preverified`` skips only full re-decompression/re-derivation: the fresh
    compressed-transport digest/size still must equal the approval bundle, and
    every per-file destination check still runs.
    """
    archive = e.checked_path(archive)
    e.require(approval.get('schema') == e.SCHEMA, 'v2 approval required')
    if preverified:
        e.require(e.sha256_file(archive) == approval['bundle']['sha256'] and
                  Path(archive).stat().st_size == approval['bundle']['bytes'], 'approved bundle mismatch')
        members = approval['bundle']['members']
    else:
        members = e.inspect_archive(archive)
        e.require(e.asset_record(archive, members) == approval['bundle'], 'approved bundle mismatch')
    selected = {name: item for name, item in members.items() if not name.startswith('bin/') and name != 'release.lock.json'}
    e.require(selected == approval['source_asset']['members'], 'source projection mismatch')
    e.require(approval_path is not None, 'exact approval bytes required')
    approval_bytes = e.safe_file(Path(approval_path).parent, Path(approval_path).name).read_bytes()
    e.require(e.loads(approval_bytes) == approval, 'approval bytes/document mismatch')
    dest = e.checked_path(dest)
    expected = dict(selected, **{'approval.json': {'sha256': __import__('hashlib').sha256(approval_bytes).hexdigest(),
                                                  'bytes': len(approval_bytes), 'mode': 0o644}})
    if dest.exists():
        e.require(tree_inventory(dest) == expected, 'existing runtime tree differs')
        return
    e.extract_files(archive, dest, members, selected=selected, preverified=preverified)
    with (dest / 'approval.json').open('xb') as stream:
        stream.write(approval_bytes)
    (dest / 'approval.json').chmod(0o644)
    e.require(tree_inventory(dest) == expected, 'extracted tree differs')


def write_latest(latest, version):
    """Point LATEST at v<version>: create it, keep it when it already names v<version>, or move it forward
    from an earlier v<M> when a later version of a model already on main is promoted. A malformed LATEST
    or one naming a later version is refused, never overwritten."""
    data = f'v{version}\n'.encode()
    if latest.exists():
        e.require(latest.is_file() and not latest.is_symlink(), 'existing LATEST is not a regular file')
        match = re.fullmatch(rb'v([1-9][0-9]*)\n', latest.read_bytes())
        e.require(match is not None, 'existing LATEST malformed')
        current = int(match[1])
        e.require(current <= version, 'existing LATEST names a later version')
        if current == version:
            return
        latest.write_bytes(data)
    else:
        with latest.open('xb') as stream:
            stream.write(data)
    latest.chmod(0o644)
    e.require(latest.read_bytes() == data, 'LATEST write drift')


def _prepare_model_branch(root, model, target_commit):
    """Local `model/<model>` branch anchored on the remote when present, then target merged forward.

    Never force-pushes or resets remote history: the local branch starts at the remote tip (fetched)
    or at the target commit when the branch is new, and the target merge is a normal forward merge.
    """
    branch = f'model/{model}'
    remote_exists = False
    try:
        git('fetch', 'origin', branch, cwd=root)
        remote_exists = True
    except ValueError:
        remote_exists = False
    base = ('origin/' + branch) if remote_exists else target_commit
    git('switch', '-C', branch, base, cwd=root)
    git('merge', '--no-edit', target_commit, cwd=root)
    return branch


def _prepare_release_branch(root, branch, target_commit):
    """Create or reuse one local release branch without resetting earlier models."""
    e.require(isinstance(branch, str) and re.fullmatch(r'release/v[0-9]+\.[0-9]+\.[0-9]+', branch),
              'invalid release branch')
    current = git('branch', '--show-current', cwd=root).stdout.strip()
    if current == branch:
        git('merge-base', '--is-ancestor', target_commit, 'HEAD', cwd=root)
        return branch
    exists = subprocess.run(['git', '--no-replace-objects', '-C', str(root), 'show-ref', '--verify', '--quiet',
                             'refs/heads/' + branch], capture_output=True, check=False).returncode == 0
    if exists:
        git('switch', branch, cwd=root)
        git('merge-base', '--is-ancestor', target_commit, 'HEAD', cwd=root)
    else:
        git('switch', '-c', branch, target_commit, cwd=root)
    return branch


def _write_release_registry_entry(root, family, model, version, entry):
    """Persist this model's anchor without rewriting or clobbering sibling modules."""
    path = root / 'runtimes/templates/approvals' / family / f'{model}.json'
    key = m.registry_key(family, model, version)
    merged = e.load_registry(root / 'runtimes/templates/APPROVALS.json')
    if key in merged.get('approvals', {}):
        e.require(merged['approvals'][key] == entry, 'existing approval registry entry differs')
        return None
    if path.exists():
        doc = e.load_json(path)
        e.require(isinstance(doc, dict) and doc.get('schema') == 'rexl3.approval-registry.v1',
                  'invalid existing model approval module')
        entries = doc.setdefault('approvals', {})
        e.require(isinstance(entries, dict), 'invalid existing model approvals')
        if key in entries:
            e.require(entries[key] == entry, 'existing model approval module entry differs')
            return None
        entries[key] = entry
    else:
        doc = {'schema': 'rexl3.approval-registry.v1', 'approvals': {key: entry}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e.json_bytes(doc))
    path.chmod(0o644)
    e.require(e.load_registry(root / 'runtimes/templates/APPROVALS.json').get('approvals', {}).get(key) == entry,
              'written model approval module did not merge cleanly')
    return path


def model_assets_dir(assets_dir, family, model, version):
    """<assets_dir>/<family>/<model>/v<N>: one model version's verified dev release assets in the cache."""
    return Path(assets_dir) / family / model / f'v{version}'


CO_AUTHOR_RE = re.compile(r'[^<>\x00-\x1f\x7f]*[^<>\s\x00-\x1f\x7f] '
                          r'<(?:[0-9]+\+)?[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?@users\.noreply\.github\.com>')


def co_author_trailers(co_authors):
    """`Co-authored-by:` trailer lines for tester credit, one per `Name <noreply>` value.

    Only single-line GitHub noreply identities are accepted (CONTRIBUTING.md section 8: no personal
    email addresses in commits); duplicates are refused.
    """
    lines = []
    for value in co_authors or ():
        e.require(isinstance(value, str) and CO_AUTHOR_RE.fullmatch(value),
                  'co-author must be one line of the form "Name <id+login@users.noreply.github.com>"')
        line = 'Co-authored-by: ' + value
        e.require(line not in lines, 'duplicate co-author: ' + value)
        lines.append(line)
    return lines


def load_new_anchor(path, key, approve_tag):
    """The maintainer-written registry anchor for a version the trusted registry does not hold yet.

    The file has the form of the approvals/<family>/<model>.json module the promotion writes:
    schema plus exactly one approvals entry, keyed by the promoted <family>/<model>/v<N>. It is the
    maintainer's independent statement of which sealed approval to trust; the downloaded approval is
    then authenticated against it exactly as against any other registry entry.
    """
    doc = e.load_json(path)
    e.require(isinstance(doc, dict) and doc.get('schema') == 'rexl3.approval-registry.v1'
              and set(doc) <= {'schema', 'approvals'}, 'new anchor file must be one approval registry module')
    entries = doc.get('approvals')
    e.require(isinstance(entries, dict) and list(entries) == [key],
              'new anchor file must hold exactly the promoted registry key: ' + key)
    entry = e.validate_registry_entry(key, entries[key])
    if e.is_engine_tag(approve_tag):
        e.require(entry.get('approve_tag') == approve_tag,
                  'new anchor for a unified engine release must name its approve_tag')
    return entry


def prepare_promotion(family, model, version, *, approve_tag, target_commit, prod_checkout, assets_dir,
                      registry, approved_templates, meta, dry_run=False, release_branch=None,
                      trusted_source_registry=None, co_authors=()):
    key = m.registry_key(family, model, version)
    entry = registry.get('approvals', {}).get(key)
    e.require(isinstance(entry, dict), 'approval not in independently trusted registry: ' + key)
    runtime_id = entry.get('runtime_id')
    validate_request(runtime_id, approve_tag, version)
    e.full_hash(target_commit, e.HEX40)
    trailers = co_author_trailers(co_authors)
    # `assets_dir` is the cache root. This model's verified dev assets live in <family>/<model>/v<N>/ beneath it:
    # where --download writes them (main) and where verify_tree reads them, as in CI's fetch_dev_assets layout.
    release_dir = model_assets_dir(assets_dir, family, model, version)
    approval_path = e.safe_file(release_dir, 'approval.json')
    precomputed = precomputed_for_assets(e.load_json(approval_path), release_dir)
    approval, archive = verify_approval(approval_path, approved_templates, registry,
                                        engine_dir=release_dir / 'engine', key=key,
                                        precomputed=precomputed,
                                        trusted_source_registry=trusted_source_registry)
    e.require((approval['runtime_id'], approval['version'], approval['approve_tag']) ==
              (runtime_id, version, approve_tag), 'promotion identity mismatch')
    root = e.checked_path(prod_checkout)
    current_head = git('rev-parse', 'HEAD', cwd=root).stdout.strip()
    if release_branch is None:
        e.require(current_head == target_commit, 'explicit base commit required')
    else:
        e.require(re.fullmatch(r'release/v[0-9]+\.[0-9]+\.[0-9]+', release_branch), 'invalid release branch')
        current_branch = git('branch', '--show-current', cwd=root).stdout.strip()
        if current_head != target_commit:
            e.require(current_branch == release_branch, 'release promotion must start at the target or its release branch')
            git('merge-base', '--is-ancestor', target_commit, 'HEAD', cwd=root)
    e.require(not git('status', '--porcelain', '--untracked-files=all', cwd=root).stdout.strip(), 'dirty production checkout')
    git('merge-base', '--is-ancestor', target_commit, 'origin/main', cwd=root)
    family_dir = root / 'runtimes' / family
    model_dir = family_dir / model
    tree = model_dir / f'v{version}'
    if tree.exists():
        problems = verify_tree(tree, approved_templates, trusted_approvals=registry,
                               assets_dir=assets_dir, family=family, model=model,
                               precomputed=precomputed, trusted_source_registry=trusted_source_registry)
        e.require(not problems, 'existing version mismatch: ' + '; '.join(problems))
        print('VERIFIED_TREE_NOOP; model branch unchanged')
        return 0
    if dry_run:
        with tempfile.TemporaryDirectory(prefix='.promotion-', dir=Path(assets_dir).parent) as tmp:
            extract_tree(archive, Path(tmp) / 'tree', approval, approval_path=approval_path,
                         preverified=True)
        print(f'VERIFIED_DRY_RUN would prepare model/{model} at {target_commit}; MODEL.json + LATEST included')
        return 0
    if meta is None:
        meta = {}
    meta = dict(meta, runtime_id=runtime_id)
    doc = m.model_json_doc(family=family, model=model, **meta)
    if release_branch is None:
        branch = _prepare_model_branch(root, model, target_commit)
    else:
        branch = _prepare_release_branch(root, release_branch, target_commit)
    m.write_or_validate_model_json(model_dir / 'MODEL.json', family=family, model=model, doc=doc)
    extract_tree(archive, tree, approval, approval_path=approval_path, preverified=True)
    write_latest(model_dir / 'LATEST', version)
    problems = verify_tree(tree, approved_templates, trusted_approvals=registry,
                           assets_dir=assets_dir, family=family, model=model,
                           precomputed=precomputed, trusted_source_registry=trusted_source_registry)
    e.require(not problems, '; '.join(problems))
    registry_path = None
    if release_branch is not None:
        registry_path = _write_release_registry_entry(root, family, model, version, entry)
    paths = [str(model_dir.relative_to(root))]
    if registry_path is not None:
        paths.append(str(registry_path.relative_to(root)))
    git('add', '--', *paths, cwd=root)
    message = f'FEAT: Promoted {model} v{version}'
    if trailers:
        message += '\n\n' + '\n'.join(trailers)
    git('commit', '-m', message, cwd=root)
    if release_branch is not None:
        print(f'VERIFIED_RELEASE_BRANCH {branch}; promoted {family}/{model}/v{version} locally')
        return 0
    git('push', '--set-upstream', 'origin', branch, cwd=root)
    gh('pr', 'create', '--repo', PROD_REPO, '--base', 'main', '--head', branch,
       '--title', f'FEAT: Promoted {model} v{version}',
       '--body', f'Sealed v2 approval {e.sha256_file(approval_path)}; dev release {approve_tag}. '
                 'MODEL.json and the sealed approval agree on runtime_id.')
    remote = git('ls-remote', 'origin', 'refs/heads/' + branch, cwd=root).stdout.split()
    e.require(remote and remote[0] == git('rev-parse', 'HEAD', cwd=root).stdout.strip(), 'remote branch mismatch')
    pr = e.loads(gh('pr', 'view', branch, '--repo', PROD_REPO, '--json', 'headRefOid,state,url').stdout)
    e.require(pr.get('headRefOid') == remote[0] and pr.get('state') == 'OPEN', 'promotion PR readback mismatch')
    print('VERIFIED_PR ' + pr['url'])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--approve-tag', required=True)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--target-commit', required=True, help='exact production base commit')
    parser.add_argument('--prod-checkout', required=True, help='owned clean checkout; fetch origin/main before use')
    parser.add_argument('--assets-dir', required=True,
                        help='durable verified dev asset cache root; a model version lives in <family>/<model>/v<N>/')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--approved', required=True, help='protected template allow-list JSON')
    parser.add_argument('--trusted-approvals', required=True)
    parser.add_argument('--trusted-approvals-sha256')
    parser.add_argument('--release-branch', help='local release/v<X.Y.Z> branch; never pushes or creates a PR')
    parser.add_argument('--new-anchor', metavar='FILE',
                        help='with --release-branch: maintainer-written approvals module (outside the checkout) '
                             'holding exactly the new <family>/<model>/v<N> anchor; the promotion commits it')
    parser.add_argument('--co-author', action='append', default=[], metavar='"NAME <NOREPLY>"',
                        help='credit a tester: adds a Co-authored-by trailer to the promotion commit (repeatable; '
                             'GitHub noreply addresses only)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--display-name', help='human display name (required when MODEL.json is new)')
    parser.add_argument('--engine-family', help='dev engine module name (required when MODEL.json is new)')
    parser.add_argument('--pack-format', help='pack format, e.g. exl3 (required when MODEL.json is new)')
    parser.add_argument('--architecture', action='append', default=[], metavar='CLASS',
                        help='HF architecture class (repeatable; required when MODEL.json is new)')
    parser.add_argument('--pack', action='append', default=[], metavar='KEY=VALUE',
                        help='descriptive pack metadata, e.g. bits=4.00 or draft=...')
    args = parser.parse_args(argv)
    try:
        m.registry_key(args.family, args.model, args.version)
        # validate_request already ran below with the entry runtime_id; here only shape checks.
        meta = None
        if args.display_name or args.engine_family or args.pack_format or args.architecture or args.pack:
            pack = {'format': args.pack_format}
            for kv in args.pack:
                key, sep, value = kv.partition('=')
                e.require(sep and key and value, '--pack requires KEY=VALUE')
                pack[key] = value
            meta = dict(display_name=args.display_name, engine_family=args.engine_family,
                        architectures=args.architecture, pack=pack)
        registry = e.load_registry(args.trusted_approvals, args.trusted_approvals_sha256)
        approved = e.load_json(args.approved)['templates']
        key = m.registry_key(args.family, args.model, args.version)
        co_author_trailers(args.co_author)
        # Carried-gate sources resolve only against the registry the checkout already holds, never
        # against the new anchor (as verify resolves them against the protected base only).
        source_registry = registry
        if args.new_anchor is not None:
            e.require(args.release_branch is not None, '--new-anchor requires --release-branch')
            e.require(key not in registry.get('approvals', {}),
                      'registry already anchors ' + key + '; drop --new-anchor')
            new_entry = load_new_anchor(args.new_anchor, key, args.approve_tag)
            registry = dict(registry, approvals=dict(registry.get('approvals', {}), **{key: new_entry}))
        entry = registry.get('approvals', {}).get(key)
        e.require(isinstance(entry, dict), 'approval not in independently trusted registry: ' + key)
        runtime_id = entry.get('runtime_id')
        if meta is not None:
            # Shape check before any download. MODEL.json carries the registry entry's runtime_id (prepare_promotion
            # derives it the same way); checking without it refused every promotion that passed metadata flags.
            m.model_json_doc(family=args.family, model=args.model, runtime_id=runtime_id, **meta)
        if args.download:
            validate_request(runtime_id, args.approve_tag, args.version)
            download_release(args.approve_tag, model_assets_dir(args.assets_dir, args.family, args.model, args.version),
                             registry=registry, approved_templates=approved, key=key,
                             trusted_source_registry=source_registry)
        return prepare_promotion(args.family, args.model, args.version, approve_tag=args.approve_tag,
            target_commit=args.target_commit, prod_checkout=args.prod_checkout, assets_dir=args.assets_dir,
            registry=registry, approved_templates=approved, meta=meta, dry_run=args.dry_run,
            release_branch=args.release_branch, trusted_source_registry=source_registry,
            co_authors=args.co_author)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        print('PROMOTION_REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
