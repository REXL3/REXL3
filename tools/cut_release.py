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
import uuid
import urllib.error
import urllib.parse
import urllib.request

try:
    from . import release_evidence as e
    from . import model_layout as m
    from .release_manifest import validate_manifest
except ImportError:
    import release_evidence as e
    import model_layout as m
    from release_manifest import validate_manifest

DEV_REPO = 'REXL3/REXL3-dev'
API = 'https://api.github.com'
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
        boundary = uuid.uuid4().hex
        head = (f'--{boundary}\r\nContent-Disposition: form-data; name="attachment"; filename="{name}"\r\n'
                f'Content-Type: application/octet-stream\r\n\r\n').encode()
        tail = f'\r\n--{boundary}--\r\n'.encode()
        body = head + data + tail
        headers = {'Authorization': 'Bearer ' + self.token, 'User-Agent': 'rexl3-cut-release',
                   'Accept': 'application/vnd.github+json',
                   'Content-Type': 'multipart/form-data; boundary=' + boundary}
        request = urllib.request.Request(f'{API}/repos/{self.repo}/releases/{release_id}/assets?name={urllib.parse.quote(name, safe="")}',
                                         data=body, headers=headers, method='POST')
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


def release_pr(api, repo, version):
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
    reviews = api.get(f'/repos/{repo}/pulls/{number}/reviews?per_page=100')
    approved = any(r.get('state') == 'APPROVED' and r.get('user', {}).get('login') not in (None, author)
                   for r in reviews)
    if not approved:
        raise ValueError('release PR has no approving review by a non-author maintainer')
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


def check_local_main(repo_root, version, token=None):
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
    registry = e.load_json(root / 'runtimes/templates/APPROVALS.json')
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
    return manifest


def asset_prefix(item):
    """`<runtime_id>-v<N>`: the release asset prefix of one manifest model (its `version` is already `v<N>`)."""
    version = item['version']
    e.require(isinstance(version, str) and m.VERSION_RE.fullmatch(version), 'invalid model version: ' + repr(version))
    return f"{item['runtime_id']}-{version}"


def model_assets(manifest):
    """Expected release asset names: five per model (approval + four archives) plus engine artifacts."""
    names = set(manifest['engine']['artifacts'])
    for item in manifest['models']:
        prefix = asset_prefix(item)
        names.add(f'{prefix}-approval.json')
        for kind in ('bundle', 'source', 'engine-source', 'evidence'):
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
    """Download the dev release for every model and verify it against the prod registry (local files).

    Each model is staged in its own directory, as fetch_dev_assets gives each tree its own asset
    directory: receipts of different runtimes share member paths with different bytes. An approval
    whose bundle leaves the native executables to the dev engine release is verified with those
    executables, downloaded into the stage's ``engine/`` directory.

    Returns a flat directory with the release asset files named exactly as they will be uploaded.
    """
    root = Path(repo_root)
    registry = e.load_json(root / 'runtimes/templates/APPROVALS.json')
    approved = e.load_json(root / 'runtimes/templates/APPROVED.json')['templates']
    work = Path(tempfile.mkdtemp(prefix='.cut-release-'))
    engine_tag = manifest['engine']['engine_release_tag']
    for item in manifest['models']:
        tag = f"approve-{item['runtime_id']}-{int(item['version'][1:])}"
        key = m.registry_key(item['family'], item['model'], int(item['version'][1:]))
        entry = registry.get('approvals', {}).get(key)
        if not isinstance(entry, dict):
            raise ValueError('registry entry missing: ' + key)
        release = api.get(f'/repos/{DEV_REPO}/releases/tags/{urllib.parse.quote(tag, safe="")}')
        if release is None:
            raise ValueError('dev release missing: ' + tag)
        dest = work / '.stage' / item['family'] / item['model'] / item['version']
        dest.mkdir(parents=True)
        for asset in release.get('assets', []):
            name = asset.get('name')
            if not name or '/' in name:
                raise ValueError('unsafe dev asset name')
            data = api.download(f'/repos/{DEV_REPO}/releases/assets/{asset["id"]}')
            path = dest / name
            path.write_bytes(data)
            if asset.get('size') not in (None, len(data)):
                raise ValueError('dev asset size mismatch: ' + name)
        approval = dest / 'approval.json'
        if not approval.exists():
            raise ValueError('dev release has no approval.json')
        e.full_hash(entry['approval_sha256'])
        if e.sha256_file(approval) != entry['approval_sha256']:
            raise ValueError('dev approval bytes differ from the pinned registry')
        doc = e.load_json(approval)
        # The bound evidence (receipts, observations, raw files) travels inside evidence.tar.gz.
        evidence = doc['evidence_asset']
        e.require(not any(name == 'engine' or name.startswith('engine/') for name in evidence['members']),
                  'evidence collides with the engine directory')
        e.extract_files(dest / evidence['archive'], dest, evidence['members'])
        engine_dir = stage_engine_binaries(api, doc, dest / 'engine', engine_tag)
        e.verify_assets(approval, registry=registry, approved_templates=approved, engine_dir=engine_dir, key=key)
        prefix = asset_prefix(item)
        (work / (prefix + '-approval.json')).write_bytes(approval.read_bytes())
        for kind, field in (('bundle', 'bundle'), ('source', 'source_asset'), ('engine-source', 'engine_source_asset'),
                            ('evidence', 'evidence_asset')):
            archive = doc[field]['archive']
            (work / f'{prefix}-{kind}.tar.gz').write_bytes((dest / archive).read_bytes())
    # Engine artifacts named by the manifest come from the dev engine release.
    engine_tag = manifest['engine']['engine_release_tag']
    engine_release = api.get(f'/repos/{DEV_REPO}/releases/tags/{urllib.parse.quote(engine_tag, safe="")}')
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
        (work / name).write_bytes(data)
    return work


def hashlib_hex(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def upload_assets(api, repo, release_id, work, manifest):
    expected = model_assets(manifest)
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


def verify_published(api, repo, version, merge_commit, manifest, *, draft_ok=False):
    release = api.get(f'/repos/{repo}/releases/tags/{urllib.parse.quote(version, safe="")}')
    if release is None:
        raise ValueError('published release missing')
    if not draft_ok and (release.get('draft') or release.get('prerelease')):
        raise ValueError('release is draft/prerelease')
    ref = api.get(f'/repos/{repo}/git/ref/tags/{version}')
    if ref is None or ref.get('object', {}).get('sha') != merge_commit:
        raise ValueError('tag does not point at the release PR merge commit')
    assets = {a['name']: a for a in release.get('assets', [])}
    expected = model_assets(manifest)
    if set(assets) != expected:
        raise ValueError('published assets differ from the manifest: ' + ', '.join(sorted(set(assets) ^ expected)))
    for name, asset in assets.items():
        if asset.get('state') != 'uploaded' or asset.get('size', 0) <= 0:
            raise ValueError('asset not uploaded: ' + name)
        if asset.get('digest') and not re.fullmatch(r'sha256:[0-9a-f]{64}', asset['digest']):
            raise ValueError('unexpected asset digest format: ' + name)
    return release


def cut(api, repo, version, mode, repo_root='.'):
    m.product_version(version)
    if mode == 'verify':
        # --verify confirms the published release matches the manifest and the lock is restored.
        release = api.get(f'/repos/{repo}/releases/tags/{urllib.parse.quote(version, safe="")}')
        if release is None:
            raise ValueError('no published release for tag ' + version)
        pull, merge_commit = release_pr(api, repo, version)
        manifest = check_local_main(repo_root, version)
        published = verify_published(api, repo, version, merge_commit, manifest)
        print(f'VERIFIED {version} id={published["id"]} tag={version} commit={merge_commit}')
        return 0
    if tag_exists(api, repo, version):
        raise ValueError('tag already exists: ' + version)
    pull, merge_commit = release_pr(api, repo, version)
    immutable_releases(api, repo)
    manifest = check_local_main(repo_root, version)
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
    upload_assets(api, repo, release['id'], work, manifest)
    published = verify_published(api, repo, version, merge_commit, manifest, draft_ok=True)
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
