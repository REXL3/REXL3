#!/usr/bin/env python3
"""Cut a product release (contract section 7): maintainer ceremony, never an agent action.

    python3 tools/cut_release.py --version v0.1.0            # dry run verification (default)
    python3 tools/cut_release.py --version v0.1.0 --publish  # create tag + draft release, verify, publish
    python3 tools/cut_release.py --version v0.1.0 --verify    # confirm the published release matches main

Gates (all run in every mode): the release PR is merged with an approving review by a non-author
maintainer; the manifest is valid and consistent with main; NOTES.md and the CHANGELOG entry exist;
the tag is absent; immutable releases are enabled. --publish additionally requires the tag ruleset to
allow creating exactly this tag (a maintainer temporarily allows it; this tool never edits rulesets or
settings). All API access is REST over urllib with a token from the environment (GH_TOKEN).
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

try:
    from . import release_evidence as e
    from . import model_layout as m
    from .release_manifest import validate_manifest, shared_engine_sources, shared_source_name, SOURCE_ARCHIVES
except ImportError:
    import release_evidence as e
    import model_layout as m
    from release_manifest import validate_manifest, shared_engine_sources, shared_source_name, SOURCE_ARCHIVES

DEV_REPO = 'REXL3/REXL3-dev'
API = 'https://api.github.com'
UPLOAD_API = 'https://uploads.github.com'
MAX_ASSET_BYTES = 8 * 1024 ** 3


class Api:
    def __init__(self, token, repo, opener=None):
        self.token = token
        self.repo = repo
        self.opener = opener or urllib.request

    def _request(self, method, path, data=None, accept=None, raw=False):
        headers = {'Authorization': 'Bearer ' + self.token, 'User-Agent': 'rexl3-cut-release',
                   'Accept': accept or 'application/vnd.github+json'}
        body = json.dumps(data).encode() if data is not None else None
        request = urllib.request.Request(API + path, data=body, headers=headers, method=method)
        try:
            with self.opener.urlopen(request, timeout=120) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404 and method == 'GET':
                return None
            detail = ''
            try:
                detail = error.read().decode('utf-8', 'replace')[:200]
            except Exception:
                pass
            raise ValueError(f'GitHub API {error.code} for {method} {path}: {detail}') from None
        if raw:
            return payload
        return json.loads(payload.decode('utf-8')) if payload else None

    def get(self, path, **kwargs):
        return self._request('GET', path, **kwargs)

    def post(self, path, data, **kwargs):
        return self._request('POST', path, data=data, **kwargs)

    def patch(self, path, data, **kwargs):
        return self._request('PATCH', path, data=data, **kwargs)

    def download(self, path):
        return self._request('GET', path, accept='application/octet-stream', raw=True)

    def upload(self, release_id, name, data):
        headers = {'Authorization': 'Bearer ' + self.token, 'User-Agent': 'rexl3-cut-release',
                   'Accept': 'application/vnd.github+json', 'Content-Type': 'application/octet-stream'}
        request = urllib.request.Request(
            f'{UPLOAD_API}/repos/{self.repo}/releases/{release_id}/assets?name={urllib.parse.quote(name, safe="")}',
            data=data, headers=headers, method='POST')
        try:
            with self.opener.urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            raise ValueError(f'asset upload {name} failed: HTTP {error.code}') from None


def maintainers(path):
    owners = set()
    for line in path.read_text(errors='replace').splitlines():
        line = line.strip()
        if line.startswith('*') and not line.startswith('*.'):
            owners.update(f.lstrip('@') for f in line.split()[1:] if f.startswith('@'))
    return owners


def release_pr(api, repo, version, owners):
    """The merged release PR for `version`: exactly one, merged, with a non-author maintainer approval."""
    head = f'{repo.split("/", 1)[0]}:release/{version}'
    pulls = api.get(f'/repos/{repo}/pulls?state=closed&base=main&head={urllib.parse.quote(head, safe="")}&per_page=100')
    if not isinstance(pulls, list) or not pulls:
        raise ValueError(f'no release PR for release/{version}')
    if len(pulls) > 1:
        raise ValueError(f'more than one release PR for release/{version}')
    pull = pulls[0]
    if pull.get('merged_at') is None:
        raise ValueError('release PR is not merged')
    author = pull.get('user', {}).get('login')
    number = pull.get('number')
    pr_head_sha = pull.get('head', {}).get('sha')
    e.full_hash(pr_head_sha, e.HEX40)
    reviews = []
    page = 1
    while True:
        batch = api.get(f'/repos/{repo}/pulls/{number}/reviews?per_page=100&page={page}')
        e.require(isinstance(batch, list), 'release PR reviews response is malformed')
        reviews.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    approved = any(r.get('state') == 'APPROVED'
                   and r.get('user', {}).get('login') in owners
                   and r.get('user', {}).get('login') != author
                   and r.get('commit_id') == pr_head_sha
                   for r in reviews)
    if not approved:
        raise ValueError('release PR has no approving review by a non-author maintainer at the PR head')
    merge_commit = pull.get('merge_commit_sha')
    e.full_hash(merge_commit, e.HEX40)
    return pull, merge_commit


def tag_exists(api, repo, version):
    ref = api.get(f'/repos/{repo}/git/ref/tags/{version}')
    return ref is not None


def immutable_releases(api, repo):
    """Require the repo-level immutable-releases setting (curl /repos/{owner}/{repo}/immutable-releases)."""
    raw = api.get(f'/repos/{repo}/immutable-releases', raw=True)
    if raw is None:
        raise ValueError('immutable releases setting not readable; cannot confirm it is enabled')
    try:
        doc = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeError):
        raise ValueError('immutable releases setting returned an unreadable response') from None
    if not isinstance(doc, dict) or doc.get('enabled') is False:
        raise ValueError('immutable releases are disabled')
    return True


def check_local_main(repo_root, version, token=None, api=None):
    """Validate manifest + notes + registry + trees on the checked-out main."""
    root = Path(repo_root)
    manifest_path = root / 'releases' / version / 'manifest.json'
    manifest = validate_manifest(e.load_json(manifest_path), version)
    notes = root / 'releases' / version / 'NOTES.md'
    if not notes.is_file() or not notes.read_text(encoding='utf-8').strip():
        raise ValueError('NOTES.md missing or empty')
    changelog = root / 'CHANGELOG.md'
    if not changelog.is_file() or version not in changelog.read_text(encoding='utf-8'):
        raise ValueError('CHANGELOG.md has no entry for ' + version)
    registry = e.load_registry(root / 'runtimes/templates/APPROVALS.json')
    for item in manifest['models']:
        folder = root / 'runtimes' / item['family'] / item['model']
        model_doc = m.parse_model_json(folder / 'MODEL.json')
        m.validate_model_doc(model_doc, item['family'], item['model'])
        if model_doc.get('runtime_id') != item['runtime_id']:
            raise ValueError(f"MODEL.json runtime_id differs from manifest for {item['family']}/{item['model']}")
        tree = folder / item['version']
        if not (tree / 'approval.json').is_file():
            raise ValueError(f'sealed tree missing: {folder / item["version"]}')
        key = m.registry_key(item['family'], item['model'], int(item['version'][1:]))
        entry = registry.get('approvals', {}).get(key)
        if not isinstance(entry, dict) or entry.get('approval_sha256') != item['approval_sha256']:
            raise ValueError(f'registry does not anchor {key} with the manifest approval hash')
        if entry.get('runtime_id') != item['runtime_id']:
            raise ValueError(f'registry runtime_id differs for {key}')
        if item.get('carried_forward'):
            # Carry-forward means "the family/kernel scope is untouched by the engine commits SINCE
            # the approval was sealed" — so the qualified engine release must be an EARLIER release
            # than the release's primary engine tag. Carrying forward from the SAME engine release
            # this release ships is nonsensical (it would mark a model as untouched while the engine
            # tag advanced), so refuse it before any trust check.
            e.require(item['qualified_engine_release_tag'] != manifest['engine']['engine_release_tag'],
                      'carried-forward model must name an earlier engine release than the release '
                      'primary engine tag (self carry-forward is not a carry-forward)')
            # T0 is the only prod carry-forward: the prior sealed approval travels as-is. T1/T2
            # re-run gates on the new engine binary and re-seal a new approval at the new engine
            # commit, so they are not manifest carry-forwards.
            e.require(item['qualification_tier'] == 'T0',
                      'carried-forward model must be qualification tier T0 (T1/T2 re-seal a new '
                      'approval on the new engine build)')
            # Trust the pinned registry bytes before reading the local approval (a tampered local
            # approval.json must never feed the carried-forward checks).
            e.full_hash(entry['approval_sha256'])
            e.require(e.sha256_file(tree / 'approval.json') == entry['approval_sha256'],
                      'local approval bytes differ from the pinned registry (untrusted local approval)')
            approval = e.load_json(tree / 'approval.json')
            e.full_hash(approval['source_commit'], e.HEX40)
            # The scope diff base must be the exact commit the carried approval was sealed at.
            e.require(item['scope_diff_base'] == approval['source_commit'],
                      'scope_diff_base must equal the carried approval source commit')
            engine = approval.get('engine_binaries')
            if engine is not None:
                e.validate_engine_binaries(engine)
                e.require({record['release_tag'] for record in engine.values()} == {item['qualified_engine_release_tag']},
                          'carried-forward approval names another engine release than '
                          + item['qualified_engine_release_tag'])
            # The qualified tag must name the approval's source commit locally (fast, offline) and,
            # when the dev API is available, must resolve on the dev repo to that exact commit (a
            # made-up tag that happens to carry the right suffix is still refused).
            e.require(item['qualified_engine_release_tag'].endswith('-' + approval['source_commit'][:8]),
                      'carried-forward qualified engine release tag does not name the approval commit')
            if api is not None:
                try:
                    tag_commit = dev_tag_commit(api, item['qualified_engine_release_tag'])
                except ValueError as err:
                    raise ValueError('carried-forward qualified engine release tag does not name '
                                     'the approval commit: ' + str(err)) from None
                e.require(tag_commit == approval['source_commit'],
                          'carried-forward qualified engine release tag does not name the approval commit')
                _require_engine_ahead(api, approval['source_commit'], manifest['engine']['dev_commit'])
    return manifest


def _require_engine_ahead(api, base, head):
    """The primary engine dev commit must be ahead of the carried approval's source commit."""
    e.full_hash(base, e.HEX40)
    e.full_hash(head, e.HEX40)
    compare = api.get(f'/repos/{DEV_REPO}/compare/{base}...{head}')
    e.require(isinstance(compare, dict) and compare.get('status') == 'ahead',
              'engine dev commit is not ahead of the carried approval source commit')


def asset_prefix(item):
    """`<runtime_id>-v<N>`: the release asset prefix of one manifest model (its `version` is already `v<N>`)."""
    version = item['version']
    e.require(isinstance(version, str) and m.VERSION_RE.fullmatch(version), 'invalid model version: ' + repr(version))
    return f"{item['runtime_id']}-{version}"


def model_assets(manifest, repo_root=None):
    """Expected release asset names: per-model approval and archives plus engine artifacts.

    Every carried-forward model's engine binaries (the assets its sealed approval names in
    ``engine_binaries``) are also release assets: they are staged from that model's qualified
    engine release, so a mixed-engine release publishes each model's own engine build. The
    carried engine assets are read from the local prod tree when ``repo_root`` is given.

    A manifest with ``engine.source_archives`` publishes each named engine corresponding-source
    archive once: every model maps to one of them by its sealed digest and size, and no model
    keeps a per-model engine-source asset. A manifest without the key is unchanged.
    """
    names = set(manifest['engine']['artifacts'])
    shared_sources = dict(shared_engine_sources(manifest, repo_root))
    names.update(shared_sources.values())
    if repo_root is not None:
        root = Path(repo_root)
        for item in manifest['models']:
            tree = root / 'runtimes' / item['family'] / item['model'] / item['version']
            approval = e.load_json(tree / 'approval.json')
            e.approval_assets(approval)
            engine_source = approval['engine_source_asset']['archive']
            if engine_source == e.shared_engine_source_name(approval.get('approve_tag')):
                shared_sources[(item['family'], item['model'], item['version'])] = engine_source
                names.add(engine_source)
            if item.get('carried_forward') is not True:
                continue
            engine = approval.get('engine_binaries')
            if engine is not None:
                e.validate_engine_binaries(engine)
                names.update(record['asset'] for record in engine.values())
    for item in manifest['models']:
        prefix = asset_prefix(item)
        names.add(f'{prefix}-approval.json')
        for kind in ('bundle', 'source', 'engine-source', 'evidence'):
            if kind == 'engine-source' and (item['family'], item['model'], item['version']) in shared_sources:
                continue
            names.add(f'{prefix}-{kind}.tar.gz')
    return names


def dev_tag_commit(api, tag):
    """The commit a dev tag resolves to: a lightweight tag, or a chain of at most five annotated tags."""
    data = api.get(f'/repos/{DEV_REPO}/git/ref/tags/{urllib.parse.quote(tag, safe="")}')
    for _ in range(5):
        target = data.get('object') if isinstance(data, dict) else None
        e.require(isinstance(target, dict), 'dev tag missing: ' + tag)
        e.full_hash(target.get('sha'), e.HEX40)
        if target.get('type') == 'commit':
            return target['sha']
        e.require(target.get('type') == 'tag', 'dev tag does not resolve to a commit: ' + tag)
        data = api.get(f'/repos/{DEV_REPO}/git/tags/{target["sha"]}')
    raise ValueError('annotated tag chain limit: ' + tag)


def stable_engine_release(release):
    return (release.get('id'), release.get('tag_name'), release.get('draft'), release.get('prerelease'),
            [(a.get('id'), a.get('name'), a.get('size'), a.get('digest')) for a in release.get('assets', [])])


def stage_engine_binaries(api, approval, dest, engine_tag):
    """Download the native executables an approval leaves to the dev engine release into ``dest``.

    The same checks as promote_runtime.fetch_engine, over this tool's REST client: the release is the one
    the manifest names and is published, each asset exists under its exact name with the approved size,
    GitHub digest and bytes, the tag resolves to the approval's source commit, and the release does not
    change while it is read. Returns None for an approval whose bundle carries its own executables.
    """
    engine = approval.get('engine_binaries')
    if engine is None:
        return None
    records = e.validate_engine_binaries(engine)
    tag = next(iter(records.values()))['release_tag']
    e.require(tag == engine_tag, f'approval takes its engine from {tag}, the manifest names {engine_tag}')
    path = f'/repos/{DEV_REPO}/releases/tags/{urllib.parse.quote(tag, safe="")}'
    release = api.get(path)
    e.require(isinstance(release, dict) and release.get('tag_name') == tag, 'dev engine release missing: ' + tag)
    e.require(release.get('draft') is False and release.get('prerelease') is False, 'dev engine release is not published: ' + tag)
    remote = {asset.get('name'): asset for asset in release.get('assets', [])}
    dest.mkdir()
    for item in records.values():
        asset = remote.get(item['asset'])
        e.require(isinstance(asset, dict) and type(asset.get('id')) is int and asset.get('size') == item['bytes'],
                  'engine release asset missing or size differs: ' + item['asset'])
        e.require(asset.get('digest') in (None, 'sha256:' + item['sha256']), 'engine release asset digest differs: ' + item['asset'])
        data = api.download(f'/repos/{DEV_REPO}/releases/assets/{asset["id"]}')
        e.require(isinstance(data, bytes) and len(data) == item['bytes'] and hashlib_hex(data) == item['sha256'],
                  'engine asset bytes differ from the approval: ' + item['asset'])
        target = dest / item['asset']
        with target.open('xb') as stream:
            stream.write(data)
        target.chmod(0o755)
    e.require(dev_tag_commit(api, tag) == approval['source_commit'], 'engine release tag does not name the approved engine commit')
    after = api.get(path)
    e.require(isinstance(after, dict) and stable_engine_release(after) == stable_engine_release(release),
              'dev engine release changed while downloading: ' + tag)
    return dest


def download_dev_assets(api, manifest, repo_root):
    """Download and verify each model, caching unified release metadata and shared bytes.

    Legacy approve-* releases retain their exact inventory checks. A unified release
    supplies one prefixed approval per model; only that approval's bound archives are
    selected. Separate evidence stages and per-model verification remain mandatory.
    """
    root = Path(repo_root)
    registry = e.load_registry(root / 'runtimes/templates/APPROVALS.json')
    approved = e.load_json(root / 'runtimes/templates/APPROVED.json')['templates']
    work = Path(tempfile.mkdtemp(prefix='.cut-release-'))
    release_cache = {}
    byte_cache = {}
    source_archives = manifest['engine'].get(SOURCE_ARCHIVES)
    original_api = api

    class DownloadCache:
        def get(self, *args, **kwargs):
            return original_api.get(*args, **kwargs)

        def download(self, path):
            if path not in byte_cache:
                byte_cache[path] = original_api.download(path)
            return byte_cache[path]

    api = DownloadCache()

    def release_for(tag):
        if tag not in release_cache:
            release = original_api.get(f'/repos/{DEV_REPO}/releases/tags/{urllib.parse.quote(tag, safe="")}')
            if release is None:
                raise ValueError('dev release missing: ' + tag)
            e.require(isinstance(release, dict) and release.get('tag_name') == tag
                      and release.get('draft') is False and release.get('prerelease') is False
                      and isinstance(release.get('assets'), list), 'unpublished or malformed dev release: ' + tag)
            release_cache[tag] = release
        return release_cache[tag]

    def asset_bytes(asset):
        name = asset.get('name')
        if not name or '/' in name:
            raise ValueError('unsafe dev asset name')
        data = api.download(f'/repos/{DEV_REPO}/releases/assets/{asset["id"]}')
        if asset.get('size') not in (None, len(data)):
            raise ValueError('dev asset size mismatch: ' + name)
        if asset.get('digest') not in (None, 'sha256:' + hashlib_hex(data)):
            raise ValueError('dev asset digest mismatch: ' + name)
        return data

    for item in manifest['models']:
        engine_tag = manifest['engine']['engine_release_tag']
        if item.get('carried_forward'):
            engine_tag = item['qualified_engine_release_tag']
        version = int(item['version'][1:])
        key = m.registry_key(item['family'], item['model'], version)
        entry = registry.get('approvals', {}).get(key)
        if not isinstance(entry, dict):
            raise ValueError('registry entry missing: ' + key)
        e.validate_registry_entry(key, entry)
        tag = entry.get('approve_tag') or f"approve-{item['runtime_id']}-{version}"
        e.validate_request(item['runtime_id'], tag, version)
        release = release_for(tag)
        remote = {asset.get('name'): asset for asset in release.get('assets', [])}
        e.require(len(remote) == len(release['assets']), 'duplicate dev release asset name')
        dest = work / '.stage' / item['family'] / item['model'] / item['version']
        dest.mkdir(parents=True)
        approval = dest / 'approval.json'
        if e.is_engine_tag(tag):
            approval_name = f"{item['runtime_id']}-v{version}-approval.json"
            if approval_name not in remote:
                raise ValueError('unified dev release has no model approval: ' + approval_name)
            approval.write_bytes(asset_bytes(remote[approval_name]))
            e.full_hash(entry['approval_sha256'])
            if e.sha256_file(approval) != entry['approval_sha256']:
                raise ValueError('dev approval bytes differ from the pinned registry')
            doc = e.load_json(approval)
            e.require(doc.get('approve_tag') == tag, 'unified dev approval tag mismatch')
            for asset in e.approval_assets(doc):
                name = asset['archive']
                if name not in remote:
                    raise ValueError('unified dev asset missing: ' + name)
                (dest / name).write_bytes(asset_bytes(remote[name]))
        else:
            for asset in release.get('assets', []):
                (dest / asset['name']).write_bytes(asset_bytes(asset))
            if not approval.exists():
                raise ValueError('dev release has no approval.json')
            e.full_hash(entry['approval_sha256'])
            if e.sha256_file(approval) != entry['approval_sha256']:
                raise ValueError('dev approval bytes differ from the pinned registry')
            doc = e.load_json(approval)
        e.validate_request(doc.get('runtime_id'), doc.get('approve_tag'), doc.get('version'))
        if e.is_engine_tag(tag):
            e.require(dev_tag_commit(original_api, tag) == doc['source_commit'],
                      'unified dev tag does not name the sealed source commit')
        # The bound evidence (receipts, observations, raw files) travels inside evidence.tar.gz.
        evidence = doc['evidence_asset']
        e.require(not any(name == 'engine' or name.startswith('engine/') for name in evidence['members']),
                  'evidence collides with the engine directory')
        e.extract_files(dest / evidence['archive'], dest, evidence['members'])
        engine_dir = stage_engine_binaries(api, doc, dest / 'engine', engine_tag)
        e.verify_assets(approval, registry=registry, approved_templates=approved, engine_dir=engine_dir, key=key)
        if engine_dir is not None:
            for path in sorted(engine_dir.iterdir()):
                if not path.is_file():
                    continue
                target = work / path.name
                if target.exists():
                    e.require(e.sha256_file(target) == e.sha256_file(path),
                              'engine asset collision in the release work dir: ' + path.name)
                else:
                    target.write_bytes(path.read_bytes())
                    target.chmod(0o755)
        prefix = asset_prefix(item)
        (work / (prefix + '-approval.json')).write_bytes(approval.read_bytes())
        for kind, field in (('bundle', 'bundle'), ('source', 'source_asset'), ('engine-source', 'engine_source_asset'),
                            ('evidence', 'evidence_asset')):
            archive = doc[field]['archive']
            shared = archive == e.shared_engine_source_name(tag)
            target = work / (archive if shared else f'{prefix}-{kind}.tar.gz')
            data = (dest / archive).read_bytes()
            if kind == 'engine-source' and source_archives is not None:
                # One shared archive per sealed digest: the approval still names its own digest and
                # size, and the bytes just verified against it must be the manifest's bytes.
                shared = True
                target = work / shared_source_name(doc, source_archives)
                record = source_archives[target.name]
                e.require(len(data) == record['bytes'] and hashlib_hex(data) == record['sha256'],
                          'engine source bytes differ from the manifest: ' + target.name)
            if target.exists():
                e.require(shared and e.sha256_file(target) == e.sha256_file(dest / archive),
                          'release asset collision: ' + target.name)
            else:
                target.write_bytes(data)
    for name in source_archives or {}:
        e.require((work / name).is_file(), 'manifest names an engine source archive no model was sealed from: ' + name)
    # Engine artifacts named by the manifest come from the dev engine release.
    engine_tag = manifest['engine']['engine_release_tag']
    engine_release = original_api.get(f'/repos/{DEV_REPO}/releases/tags/{urllib.parse.quote(engine_tag, safe="")}')
    if engine_release is None:
        raise ValueError('dev engine release missing: ' + engine_tag)
    engine_assets = {a['name']: a for a in engine_release.get('assets', [])}
    for name, digest in manifest['engine']['artifacts'].items():
        asset = engine_assets.get(name)
        if asset is None:
            raise ValueError('engine artifact missing from dev engine release: ' + name)
        data = api.download(f'/repos/{DEV_REPO}/releases/assets/{asset["id"]}')
        e.full_hash(digest)
        if hashlib_hex(data) != digest:
            raise ValueError('engine artifact digest mismatch: ' + name)
        target = work / name
        if target.exists():
            e.require(e.sha256_file(target) == digest, 'engine artifact collision: ' + name)
        else:
            target.write_bytes(data)
    return work


def hashlib_hex(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def upload_assets(api, repo, release_id, work, manifest, repo_root=None):
    expected = model_assets(manifest, repo_root)
    uploaded = set()
    for path in sorted(work.iterdir()):
        if not path.is_file() or path.name == 'approval.json':
            continue
        if path.name not in expected:
            raise ValueError('unexpected upload file: ' + path.name)
        data = path.read_bytes()
        asset = api.upload(release_id, path.name, data)
        if asset.get('size') != len(data):
            raise ValueError('uploaded asset size mismatch: ' + path.name)
        uploaded.add(path.name)
    missing = expected - uploaded
    if missing:
        raise ValueError('release assets missing after upload: ' + ', '.join(sorted(missing)))


def verify_published(api, repo, version, merge_commit, manifest, *, draft_ok=False, repo_root=None):
    release = api.get(f'/repos/{repo}/releases/tags/{urllib.parse.quote(version, safe="")}')
    if release is None:
        raise ValueError('published release missing')
    if not draft_ok and (release.get('draft') or release.get('prerelease')):
        raise ValueError('release is draft/prerelease')
    ref = api.get(f'/repos/{repo}/git/ref/tags/{version}')
    if ref is None or ref.get('object', {}).get('sha') != merge_commit:
        raise ValueError('tag does not point at the release PR merge commit')
    assets = {a['name']: a for a in release.get('assets', [])}
    expected = model_assets(manifest, repo_root)
    if set(assets) != expected:
        raise ValueError('published assets differ from the manifest: ' + ', '.join(sorted(set(assets) ^ expected)))
    for name, asset in assets.items():
        if asset.get('state') != 'uploaded' or asset.get('size', 0) <= 0:
            raise ValueError('asset not uploaded: ' + name)
        if asset.get('digest') and not re.fullmatch(r'sha256:[0-9a-f]{64}', asset['digest']):
            raise ValueError('unexpected asset digest format: ' + name)
    for name, record in (manifest['engine'].get(SOURCE_ARCHIVES) or {}).items():
        asset = assets[name]
        if asset.get('size') != record['bytes'] or asset.get('digest') not in (None, 'sha256:' + record['sha256']):
            raise ValueError('published engine source differs from the manifest: ' + name)
    return release


def cut(api, repo, version, mode, repo_root='.'):
    m.product_version(version)
    if mode == 'verify':
        # --verify confirms the published release matches the manifest and the lock is restored.
        release = api.get(f'/repos/{repo}/releases/tags/{urllib.parse.quote(version, safe="")}')
        if release is None:
            raise ValueError('no published release for tag ' + version)
        pull, merge_commit = release_pr(api, repo, version,
                                        maintainers(Path(repo_root) / '.github/CODEOWNERS'))
        manifest = check_local_main(repo_root, version, api=api)
        published = verify_published(api, repo, version, merge_commit, manifest, repo_root=repo_root)
        print(f'VERIFIED {version} id={published["id"]} tag={version} commit={merge_commit}')
        return 0
    if tag_exists(api, repo, version):
        raise ValueError('tag already exists: ' + version)
    pull, merge_commit = release_pr(api, repo, version,
                                    maintainers(Path(repo_root) / '.github/CODEOWNERS'))
    immutable_releases(api, repo)
    manifest = check_local_main(repo_root, version, api=api)
    if mode == 'dry':
        print(f'DRY_RUN_OK would tag {version} at {merge_commit} and publish the release with the verified dev assets')
        return 0
    # --publish: create the tag (ruleset must currently allow exactly this tag), draft release, upload, verify, publish.
    work = download_dev_assets(api, manifest, repo_root)
    created = api.post(f'/repos/{repo}/git/refs',
                       {'ref': 'refs/tags/' + version, 'sha': merge_commit})
    if created is None:
        raise ValueError('tag creation failed')
    notes = (Path(repo_root) / 'releases' / version / 'NOTES.md').read_text(encoding='utf-8')
    release = api.post(f'/repos/{repo}/releases',
                       {'tag_name': version, 'name': version, 'body': notes, 'draft': True, 'prerelease': False})
    if not isinstance(release, dict) or not release.get('id'):
        raise ValueError('draft release creation failed')
    upload_assets(api, repo, release['id'], work, manifest, repo_root=repo_root)
    published = verify_published(api, repo, version, merge_commit, manifest, draft_ok=True, repo_root=repo_root)
    api.patch(f'/repos/{repo}/releases/{release["id"]}', {'draft': False})
    print(f'PUBLISHED {version} id={published["id"]} at {merge_commit}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--repo', default='REXL3/REXL3')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--token')
    args = parser.parse_args(argv)
    token = args.token or os.environ.get('GH_TOKEN')
    if not token:
        print('cut_release.py requires GH_TOKEN in the environment', file=sys.stderr)
        return 2
    if args.publish and args.verify:
        print('--publish and --verify are mutually exclusive', file=sys.stderr)
        return 2
    mode = 'verify' if args.verify else 'publish' if args.publish else 'dry'
    try:
        api = Api(token, args.repo)
        return cut(api, args.repo, args.version, mode, repo_root=args.repo_root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('CUT_REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
