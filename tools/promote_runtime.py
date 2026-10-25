#!/usr/bin/env python3
"""Two-phase, fail-closed promotion using protected verifier code and pinned approval.

Prepare a PR without LATEST; after merge, --publish-only requires its exact
--target-commit. Publication is verified byte-for-byte before a separate LATEST PR.
No blind workflow dispatch or unverified existing-release success path exists.
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
from urllib.parse import quote
try:
    from . import release_evidence as e
    from .verify_runtime_tree import verify_tree, tree_inventory
except ImportError:
    import release_evidence as e
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
    """A release's identity without the fields GitHub moves on its own.

    The raw API document carries per-asset download_count (it rises with every download, including ours and every CI
    run's) and updated_at. Comparing it before and after a download therefore always read as "release changed".
    Only what identifies the release and its bytes is kept: id, tag, target, state, and each asset's id, name, size,
    digest and state.
    """
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


def verify_approval(approval_path, approved_templates, registry):
    return e.verify_assets(approval_path, registry=registry, approved_templates=approved_templates)


def approval_assets(approval):
    """Iterable transport records; new *_asset fields need no identity-code switch."""
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
    """Install a complete, fsynced receipt atomically without replacing a prior pin.

    Fully verified staging bytes pin the remote identity before cache copying.
    This is NOT a cache-completion marker. An interrupted temporary receipt can
    never be read as the pin; retries always re-download and reverify all bytes.
    """
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
        os.link(pending, path)  # exclusive atomic commit, never --clobber
        pending.unlink()
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    e.require(_load_download_identity(path) == receipt, 'download identity persistence failed')


def download_release(tag, dest, *, repo=DEV_REPO, registry, approved_templates, target_commit=None):
    """Download with content-free failures; CLI/metadata errors may contain secrets."""
    try:
        return _download_release(tag, dest, repo=repo, registry=registry,
                                 approved_templates=approved_templates, target_commit=target_commit)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError):
        raise ValueError('release download verification refused') from None


def registry_key_for_tag(tag, repo):
    """The registry key a promotion tag can only ever belong to: approve-<id>-<N> in dev, prod/<id>/v<N> in prod."""
    pattern = r'approve-(.+)-([1-9][0-9]*)' if repo == DEV_REPO else r'prod/([^/]+)/v([1-9][0-9]*)'
    match = re.fullmatch(pattern, tag) if isinstance(tag, str) else None
    e.require(match is not None and e.ID.fullmatch(match[1]), 'release tag is not a promotion tag')
    return f'{match[1]}/v{match[2]}'


def _download_release(tag, dest, *, repo, registry, approved_templates, target_commit):
    """Fresh verification every attempt plus durable dev release/asset identity pins.

    A receipt authenticates neither partial cache files nor future downloads by
    itself. It only narrows which independently verified remote identity may be
    reused. Keep this local state outside the publication/evidence asset set.
    """
    # The registry is the only authority, so it is asked before any network call: a pull request must not be able to
    # make the privileged step probe or pull releases that nobody registered.
    key = registry_key_for_tag(tag, repo)
    approvals = registry.get('approvals') if isinstance(registry, dict) else None
    e.require(isinstance(approvals, dict) and key in approvals, 'approval not in independently trusted registry: ' + key)
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
        stage = Path(tmp)
        gh('release', 'download', tag, '-R', repo, '-D', stage)
        approval_path = e.safe_file(stage, 'approval.json')
        approval = e.trusted_entry(approval_path, registry)
        assets = approval_assets(approval)
        names = {'approval.json'} | {asset['archive'] for asset in assets}
        e.require({a['name'] for a in before['assets']} == names, 'unexpected/missing release assets')
        e.require({f.name for f in stage.iterdir()} == names, 'download inventory mismatch')
        measured = {}
        for asset in before['assets']:
            path = e.safe_file(stage, asset['name'])
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
        e.extract_files(transport, stage, evidence['members'])
        verify_approval(approval_path, approved_templates, registry)
        expected_commit = approval['source_commit'] if repo == DEV_REPO else target_commit
        e.full_hash(expected_commit, e.HEX40)
        expected_tag = approval['approve_tag'] if repo == DEV_REPO else f"prod/{approval['runtime_id']}/v{approval['version']}"
        e.require(tag == expected_tag and tag_commit(tag, repo) == expected_commit, 'release tag/commit identity drift')
        after = release_metadata(tag, repo)
        e.require(after == before, 'release changed while downloading')
        if repo == DEV_REPO:
            receipt = _validate_download_identity(dict(schema='rexl3.dev-download.v1', repo=repo,
                release_id=before['id'], tag=tag, target_commit=expected_commit,
                target_commitish=before.get('target_commitish'), assets={asset['name']: dict(
                    id=asset['id'], size=asset['size'], digest=asset.get('digest'), sha256=measured[asset['name']])
                    for asset in before['assets']}))
            e.require(previous is None or previous == receipt, 'download retry identity drift')
        # Only after full validation may bytes enter/reuse the durable cache.
        files = {p.relative_to(stage).as_posix(): p for p in stage.rglob('*') if p.is_file()}
        for rel, source in files.items():
            target = e.checked_path(dest / rel)
            if target.exists():
                e.require(e.sha256_file(target) == e.sha256_file(source), 'cached download conflict: ' + rel)
        dest.mkdir(parents=True, exist_ok=True)
        if repo == DEV_REPO:
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
        # A durable remote identity never authorizes a partial cache. Recheck all
        # installed bytes, including generic future assets, before returning.
        for rel, source in files.items():
            target = e.safe_file(dest, rel)
            e.require(target.stat().st_size == source.stat().st_size and
                      e.sha256_file(target) == e.sha256_file(source), 'cached download bytes differ')
        verify_approval(dest / 'approval.json', approved_templates, registry)
        if repo == DEV_REPO:
            e.require(_load_download_identity(receipt_path) == receipt, 'download identity persistence failed')
    return dest / 'approval.json'


def extract_tree(archive, dest, approval, *, approval_path=None):
    """Preflight ALL headers, even excluded native entries, before creating dest."""
    members = e.inspect_archive(archive)
    e.require(approval.get('schema') == e.SCHEMA, 'v2 approval required')
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
    e.extract_files(archive, dest, members, selected=selected)
    with (dest / 'approval.json').open('xb') as stream:
        stream.write(approval_bytes)
    (dest / 'approval.json').chmod(0o644)
    e.require(tree_inventory(dest) == expected, 'extracted tree differs')


def verify_merged_tree(runtime_id, version, target_commit, prod_checkout, approval_path, registry, approved_templates):
    """Require exact checked-out, clean, main-reachable target and identical approval/tree."""
    e.full_hash(target_commit, e.HEX40)
    root = e.checked_path(prod_checkout)
    e.require(git('rev-parse', '--show-toplevel', cwd=root).stdout.strip() == str(root), 'checkout root mismatch')
    e.git_identity(root, target_commit)
    e.require(git('rev-parse', 'HEAD', cwd=root).stdout.strip() == target_commit, 'checkout is not explicit target commit')
    e.require(not git('status', '--porcelain', '--untracked-files=all', cwd=root).stdout.strip(), 'production checkout is dirty')
    # Caller fetches origin/main before invoking; failures do not become success.
    git('merge-base', '--is-ancestor', target_commit, 'origin/main', cwd=root)
    tree = root / 'runtimes' / runtime_id / f'v{version}'
    problems = verify_tree(tree, approved_templates, trusted_approvals=registry, assets_dir=Path(approval_path).parent)
    e.require(not problems, 'merged tree verification failed: ' + '; '.join(problems))
    return tree


def verify_published_release(tag, target_commit, approval_path, registry, approved_templates, dest):
    before = release_metadata(tag, PROD_REPO)
    remote = download_release(tag, dest, repo=PROD_REPO, registry=registry,
                              approved_templates=approved_templates, target_commit=target_commit)
    e.require(Path(remote).read_bytes() == Path(approval_path).read_bytes(), 'published approval bytes differ')
    expected = e.load_json(approval_path)
    for asset in approval_assets(expected):
        e.require(e.sha256_file(Path(remote).parent / asset['archive']) == asset['sha256'], 'published asset bytes differ')
    e.require(tag_commit(tag, PROD_REPO) == target_commit, 'published tag commit differs')
    after = release_metadata(tag, PROD_REPO)
    e.require(before == after, 'published release id/assets changed during verification')
    return after


def publish_release(runtime_id, approve_tag, version, work, *, target_commit, prod_checkout, registry,
                    approved_templates, dry_run=False):
    validate_request(runtime_id, approve_tag, version)
    e.full_hash(target_commit, e.HEX40)
    approval_path = e.safe_file(work, 'approval.json')
    approval, _ = verify_approval(approval_path, approved_templates, registry)
    e.require((approval['runtime_id'], approval['version'], approval['approve_tag']) ==
              (runtime_id, version, approve_tag), 'publication request identity mismatch')
    verify_merged_tree(runtime_id, version, target_commit, prod_checkout, approval_path, registry, approved_templates)
    tag = f'prod/{runtime_id}/v{version}'
    if dry_run:
        print(f'VERIFIED_DRY_RUN would publish {tag} at {target_commit}; LATEST unchanged')
        return 0
    existing = release_metadata(tag, PROD_REPO, missing_ok=True)
    publication_path = e.checked_path(Path(work) / f'publication-{runtime_id}-v{version}.json')
    previous = e.load_json(publication_path) if publication_path.exists() else None
    if previous is not None:
        e.require(existing is not None and existing['id'] == previous.get('release_id'), 'previously verified release disappeared or changed id')
    if existing is None:
        assets = [approval_path] + [Path(work) / asset['archive'] for asset in approval_assets(approval)]
        gh('release', 'create', tag, *assets, '-R', PROD_REPO, '--target', target_commit,
           '--title', f'{runtime_id} v{version}', '--notes', f'Sealed by {approve_tag}; exact v2 evidence attached.', '--latest=false')
    with tempfile.TemporaryDirectory(prefix='.published-', dir=Path(work).parent) as tmp:
        verified = verify_published_release(tag, target_commit, approval_path, registry, approved_templates, Path(tmp) / 'assets')
    receipt = dict(schema='rexl3.publication.v1', runtime_id=runtime_id, version=version, tag=tag,
        release_id=verified['id'], target_commit=target_commit, approval_sha256=e.sha256_file(approval_path),
        assets={asset['name']: dict({key: asset.get(key) for key in ('id', 'size', 'digest')},
            sha256=e.sha256_file(e.safe_file(work, asset['name']))) for asset in verified['assets']})
    if previous is not None:
        e.require(previous == receipt, 'previously verified publication identity changed')
    else:
        with publication_path.open('xb') as stream:
            stream.write(e.json_bytes(receipt))
        publication_path.chmod(0o644)
    e.require(e.load_json(publication_path) == receipt, 'publication receipt persistence failed')
    print(f'VERIFIED_RELEASE id={verified["id"]} tag={tag} commit={target_commit}; LATEST not changed')
    return 0


def prepare_promotion(runtime_id, approve_tag, version, *, target_commit, prod_checkout, assets_dir,
                      registry, approved_templates, dry_run=False):
    validate_request(runtime_id, approve_tag, version)
    e.full_hash(target_commit, e.HEX40)
    approval_path = e.safe_file(assets_dir, 'approval.json')
    approval, archive = verify_approval(approval_path, approved_templates, registry)
    e.require((approval['runtime_id'], approval['version'], approval['approve_tag']) == (runtime_id, version, approve_tag), 'promotion identity mismatch')
    root = e.checked_path(prod_checkout)
    e.require(git('rev-parse', 'HEAD', cwd=root).stdout.strip() == target_commit, 'explicit base commit required')
    e.require(not git('status', '--porcelain', '--untracked-files=all', cwd=root).stdout.strip(), 'dirty production checkout')
    git('merge-base', '--is-ancestor', target_commit, 'origin/main', cwd=root)
    tree = root / 'runtimes' / runtime_id / f'v{version}'
    if tree.exists():
        problems = verify_tree(tree, approved_templates, trusted_approvals=registry, assets_dir=assets_dir)
        e.require(not problems, 'existing version mismatch: ' + '; '.join(problems))
        print('VERIFIED_TREE_NOOP; publish with --publish-only and exact merged target SHA')
        return 0
    branch = f'promote/{runtime_id}-v{version}'
    if dry_run:
        with tempfile.TemporaryDirectory(prefix='.promotion-', dir=Path(assets_dir).parent) as tmp:
            extract_tree(archive, Path(tmp) / 'tree', approval, approval_path=approval_path)
        print(f'VERIFIED_DRY_RUN would prepare {branch}; no LATEST update')
        return 0
    # Fail if a local branch exists: do not reset or overwrite someone else's work.
    git('checkout', '-b', branch, cwd=root)
    extract_tree(archive, tree, approval, approval_path=approval_path)
    problems = verify_tree(tree, approved_templates, trusted_approvals=registry, assets_dir=assets_dir)
    e.require(not problems, '; '.join(problems))
    git('add', '--', str(tree.relative_to(root)), cwd=root)
    git('-c', 'user.name=r0b0tlab', '-c', 'user.email=r0b0tlab@users.noreply.github.com',
        'commit', '-m', f'promote {runtime_id} v{version} from {approve_tag}', cwd=root)
    git('push', '--set-upstream', 'origin', branch, cwd=root)
    gh('pr', 'create', '--repo', PROD_REPO, '--base', 'main', '--head', branch,
       '--title', f'promote {runtime_id} v{version}', '--body', f'V2 approval {e.sha256_file(approval_path)}. No LATEST update before verified publication.')
    # Read back exact remote branch and PR rather than treating writes as success.
    remote = git('ls-remote', 'origin', 'refs/heads/' + branch, cwd=root).stdout.split()
    e.require(remote and remote[0] == git('rev-parse', 'HEAD', cwd=root).stdout.strip(), 'remote branch mismatch')
    pr = e.loads(gh('pr', 'view', branch, '--repo', PROD_REPO, '--json', 'headRefOid,state,url').stdout)
    e.require(pr.get('headRefOid') == remote[0] and pr.get('state') == 'OPEN', 'promotion PR readback mismatch')
    print('VERIFIED_PR ' + pr['url'] + '; after protected merge run --publish-only with exact merged SHA')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-id', required=True)
    parser.add_argument('--approve-tag', required=True)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--target-commit', required=True, help='exact production base (prepare) or merged commit (publish)')
    parser.add_argument('--prod-checkout', required=True, help='owned clean checkout; fetch origin/main before use')
    parser.add_argument('--assets-dir', required=True, help='durable verified dev asset cache')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--approved', required=True, help='protected template allow-list JSON')
    parser.add_argument('--trusted-approvals', required=True)
    parser.add_argument('--trusted-approvals-sha256', required=True)
    parser.add_argument('--publish-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        validate_request(args.runtime_id, args.approve_tag, args.version)
        e.full_hash(args.target_commit, e.HEX40)
        registry = e.load_registry(args.trusted_approvals, args.trusted_approvals_sha256)
        approved = e.load_json(args.approved)['templates']
        if args.download:
            download_release(args.approve_tag, args.assets_dir, registry=registry, approved_templates=approved)
        if args.publish_only:
            return publish_release(args.runtime_id, args.approve_tag, args.version, args.assets_dir,
                target_commit=args.target_commit, prod_checkout=args.prod_checkout, registry=registry,
                approved_templates=approved, dry_run=args.dry_run)
        return prepare_promotion(args.runtime_id, args.approve_tag, args.version, target_commit=args.target_commit,
            prod_checkout=args.prod_checkout, assets_dir=args.assets_dir, registry=registry,
            approved_templates=approved, dry_run=args.dry_run)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        print('PROMOTION_REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
