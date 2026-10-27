"""cut_release.py: dry-run gates and publish/verify ceremony, fully mocked at the API boundary."""
import json
import tempfile
import unittest
from pathlib import Path

from tools import cut_release as c
from tools import release_evidence as e
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import encoded, sha, write

REPO = 'REXL3/REXL3'
DEV = 'REXL3/REXL3-dev'
VERSION = 'v0.1.0'
FAMILY, MODEL = 'qwen3.8', 'qwen3.8-27b'
MERGE = 'c' * 40
HEX64 = 'd' * 64


def manifest(approval_hash):
    return {'schema': 'rexl3.release-manifest.v1', 'version': VERSION,
            'engine': {'dev_commit': 'a' * 40, 'dev_tree': 'b' * 40,
                       'engine_release_tag': 'engine-v0.1.2-4208d848',
                       'artifacts': {'rexl3-serve.tar.gz': HEX64}},
            'models': [{'family': FAMILY, 'model': MODEL, 'version': 'v1',
                        'runtime_id': 'synthetic-demo', 'approval_sha256': approval_hash}],
            'notes': 'first release'}


class FakeApi(c.Api):
    def __init__(self, fixtures, reviews, merged=True, immutable=True, existing_tags=()):
        self.fixtures = fixtures
        self.reviews = reviews
        self.merged = merged
        self.immutable = immutable
        self.existing_tags = set(existing_tags)
        self.created_refs = {}
        self.releases = {}
        self.uploads = []

    def get(self, path, raw=False, **kwargs):
        if '/pulls?' in path:
            if not self.merged:
                return [{'number': 9, 'merged_at': None, 'user': {'login': 'vcruz305'}}]
            return [{'number': 9, 'merged_at': '2026-10-26T00:00:00Z', 'merge_commit_sha': MERGE,
                     'user': {'login': 'vcruz305'}}]
        if '/reviews' in path:
            return self.reviews
        if '/git/ref/tags/' in path:
            tag = path.rsplit('/', 1)[1]
            if tag in self.existing_tags or 'refs/tags/' + tag in self.created_refs:
                return {'object': {'sha': self.created_refs.get('refs/tags/' + tag, MERGE), 'type': 'commit'}}
            return None
        if '/releases/tags/' in path:
            tag = path.rsplit('/', 1)[1]
            if DEV in path:
                if tag == 'engine-v0.1.2-4208d848':
                    return self.fixtures['engine_release']
                return self.fixtures['dev_release']
            return self.releases.get(tag)
        if '/releases/assets/' in path:
            asset_id = int(path.rsplit('/', 1)[1])
            return self.fixtures['assets'][asset_id]
        if '/immutable-releases' in path:
            if not self.immutable:
                raise ValueError('GitHub API 403')
            return b'{"enabled": true}'
        raise AssertionError('unexpected GET ' + path)

    def post(self, path, data, **kwargs):
        if path.endswith('/git/refs'):
            self.created_refs[data['ref']] = data['sha']
            return {'ref': data['ref'], 'object': {'sha': data['sha']}}
        if '/releases/assets' in path:
            return self._upload(path, data)
        if path.endswith('/releases'):
            release = dict(data, id=200, draft=True, prerelease=False, assets=[])
            self.releases[data['tag_name']] = release
            return release
        raise AssertionError('unexpected POST ' + path)

    def _upload(self, path, data):
        name = path.split('name=', 1)[1].split('&', 1)[0]
        self.uploads.append((name, data))
        asset = {'id': 300 + len(self.uploads), 'name': name, 'size': len(data),
                 'digest': 'sha256:' + sha(data), 'state': 'uploaded'}
        tag = next(iter(self.releases))
        self.releases[tag]['assets'].append(asset)
        return asset

    def patch(self, path, data, **kwargs):
        tag = next(iter(self.releases))
        self.releases[tag]['draft'] = data.get('draft', False)
        return self.releases[tag]

    def download(self, path):
        return self.get(path, raw=True)

    def upload(self, release_id, name, data):
        return self._upload('/assets?name=' + name, data)


class CutFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        self.approval, self.doc, self.registry = self.f.published_fixture_model(FAMILY, MODEL)
        self.root = Path(self.tmp.name) / 'prod'
        write(self.root, 'runtimes/templates/APPROVALS.json', encoded(self.registry))
        write(self.root, 'runtimes/templates/APPROVED.json',
             encoded({'schema': 'rexl3.approved-templates.v1', 'templates': self.f.approved}))
        tree = self.root / 'runtimes' / FAMILY / MODEL / 'v1'
        write(tree, 'approval.json', self.approval.read_bytes())
        write(tree.parent, 'MODEL.json', e.json_bytes(self.f.model_doc(FAMILY, MODEL)))
        write(tree.parent, 'LATEST', b'v1\n')
        approval_hash = sha(self.approval.read_bytes())
        write(self.root, f'releases/{VERSION}/manifest.json', encoded(manifest(approval_hash)))
        write(self.root, f'releases/{VERSION}/NOTES.md', b'# notes\n')
        write(self.root, 'CHANGELOG.md', f'# Changelog\n\n## [{VERSION}] - 2026-10-26\n\nx\n'.encode())
        write(self.root, '.github/CODEOWNERS', b'* @am423 @vcruz305\n')
        names = ['approval.json'] + [a['archive'] for a in e.approval_assets(self.doc)]
        dev_assets = [{'id': i + 1, 'name': name, 'size': (self.f.assets / name).stat().st_size,
                        'digest': 'sha256:' + e.sha256_file(self.f.assets / name)}
                      for i, name in enumerate(names)]
        engine_bytes = b'SYNTHETIC ENGINE ARTIFACT'
        self.engine_digest = sha(engine_bytes)
        assets = {i + 1: (self.f.assets / name).read_bytes() for i, name in enumerate(names)}
        assets[99] = engine_bytes
        self.fixtures = {'dev_release': {'id': 50, 'tag_name': 'approve-synthetic-demo-1',
                                         'draft': False, 'prerelease': False, 'assets': dev_assets},
                         'assets': assets,
                         'engine_release': {'id': 60, 'tag_name': 'engine-v0.1.2-4208d848',
                                            'draft': False, 'prerelease': False,
                                            'assets': [{'id': 99, 'name': 'rexl3-serve.tar.gz',
                                                        'size': len(engine_bytes),
                                                        'digest': 'sha256:' + self.engine_digest}]}}
        # manifest digest must match the fixture engine artifact bytes
        self.manifest = manifest(approval_hash)
        self.manifest['engine']['artifacts'] = {'rexl3-serve.tar.gz': self.engine_digest}
        write(self.root, f'releases/{VERSION}/manifest.json', encoded(self.manifest))

    def api(self, **kwargs):
        return FakeApi(self.fixtures, [{'state': 'APPROVED', 'user': {'login': 'am423'}}], **kwargs)

    def cut(self, api, mode='dry'):
        return c.cut(api, REPO, VERSION, mode, repo_root=str(self.root))


class DryRunTest(CutFixture):
    def test_dry_run_passes_all_gates(self):
        api = self.api()
        self.assertEqual(self.cut(api), 0)
        self.assertEqual(api.created_refs, {})

    def test_unmerged_release_pr_refuses(self):
        with self.assertRaisesRegex(ValueError, 'not merged'):
            self.cut(self.api(merged=False))

    def test_no_non_author_approval_refuses(self):
        api = FakeApi(self.fixtures, [{'state': 'APPROVED', 'user': {'login': 'vcruz305'}}])
        with self.assertRaisesRegex(ValueError, 'non-author maintainer'):
            self.cut(api)

    def test_existing_tag_refuses(self):
        with self.assertRaisesRegex(ValueError, 'tag already exists'):
            self.cut(self.api(existing_tags=(VERSION,)))

    def test_immutable_disabled_refuses(self):
        with self.assertRaises(ValueError):
            self.cut(self.api(immutable=False))

    def test_manifest_missing_refuses(self):
        (self.root / f'releases/{VERSION}/manifest.json').unlink()
        with self.assertRaises(ValueError):
            self.cut(self.api())


class PublishTest(CutFixture):
    def test_publish_creates_tag_and_release_uploads_and_publishes(self):
        api = self.api()
        self.assertEqual(self.cut(api, 'publish'), 0)
        self.assertEqual(api.created_refs['refs/tags/' + VERSION], MERGE)
        release = api.releases[VERSION]
        self.assertIs(release['draft'], False)
        expected = c.model_assets(self.manifest)
        self.assertEqual({a['name'] for a in release['assets']}, expected)
        self.assertEqual(len(api.uploads), len(expected))

    def test_publish_refuses_when_the_registry_anchor_is_missing(self):
        registry = e.load_json(self.root / 'runtimes/templates/APPROVALS.json')
        del registry['approvals'][f'{FAMILY}/{MODEL}/v1']
        write(self.root, 'runtimes/templates/APPROVALS.json', encoded(registry))
        with self.assertRaises(ValueError):
            self.cut(self.api(), 'publish')


class VerifyTest(CutFixture):
    def test_verify_confirms_published_release(self):
        api = self.api()
        self.cut(api, 'publish')
        # simulate the lock restored: tag still present, release published
        self.assertEqual(self.cut(api, 'verify'), 0)

    def test_verify_refuses_when_release_assets_differ(self):
        api = self.api()
        self.cut(api, 'publish')
        api.releases[VERSION]['assets'] = api.releases[VERSION]['assets'][:-1]
        with self.assertRaises(ValueError):
            self.cut(api, 'verify')


if __name__ == '__main__':
    unittest.main()
