"""cut_release.py: dry-run gates and publish/verify ceremony, fully mocked at the API boundary."""
import json
import shutil
import tempfile
import types
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

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

# Release asset names spelled out literally: the model version in a manifest is already `v<N>`, so a
# check derived from the code's own name builder cannot see a doubled `v` (`...-vv3-...`).
LITERAL_RUNTIME = 'qwen38-27b-dflash2-exl3-4.00'
LITERAL_NAMES = {
    'qwen38-27b-dflash2-exl3-4.00-v3-approval.json',
    'qwen38-27b-dflash2-exl3-4.00-v3-bundle.tar.gz',
    'qwen38-27b-dflash2-exl3-4.00-v3-source.tar.gz',
    'qwen38-27b-dflash2-exl3-4.00-v3-engine-source.tar.gz',
    'qwen38-27b-dflash2-exl3-4.00-v3-evidence.tar.gz',
    'rexl3-serve.tar.gz',
}
PUBLISHED_NAMES = {
    'synthetic-demo-v1-approval.json',
    'synthetic-demo-v1-bundle.tar.gz',
    'synthetic-demo-v1-source.tar.gz',
    'synthetic-demo-v1-engine-source.tar.gz',
    'synthetic-demo-v1-evidence.tar.gz',
    'rexl3-serve.tar.gz',
}
# The doubled-v bug, reintroduced: the mutation the literal checks below must catch.
DOUBLE_V = ('''return f"{item['runtime_id']}-{version}"''', '''return f"{item['runtime_id']}-v{version}"''')


def literal_manifest():
    return {'engine': {'artifacts': {'rexl3-serve.tar.gz': HEX64}},
            'models': [{'family': FAMILY, 'model': MODEL, 'version': 'v3',
                        'runtime_id': LITERAL_RUNTIME, 'approval_sha256': HEX64}]}


def mutant(old, new):
    """cut_release.py with one source edit, loaded as a separate module (the real module is untouched)."""
    source = Path(c.__file__).read_text()
    if source.count(old) != 1:
        raise AssertionError('mutation site not found exactly once: ' + old)
    module = types.ModuleType('cut_release_mutant')
    module.__file__ = c.__file__
    module.__package__ = c.__package__
    exec(compile(source.replace(old, new), c.__file__, 'exec'), module.__dict__)
    return module


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


class AssetNameTest(CutFixture):
    def test_model_assets_are_the_literal_release_names(self):
        self.assertEqual(c.model_assets(literal_manifest()), LITERAL_NAMES)

    def test_publish_uploads_the_literal_release_names(self):
        api = self.api()
        self.assertEqual(self.cut(api, 'publish'), 0)
        self.assertEqual({a['name'] for a in api.releases[VERSION]['assets']}, PUBLISHED_NAMES)
        self.assertEqual({name for name, _ in api.uploads}, PUBLISHED_NAMES)

    def test_a_version_without_its_v_prefix_is_refused(self):
        for version in ('3', 'vv3', 3):
            doc = literal_manifest()
            doc['models'][0]['version'] = version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'invalid model version'):
                c.model_assets(doc)

    def test_the_double_v_mutation_fails_both_literal_checks(self):
        bad = mutant(*DOUBLE_V)
        names = bad.model_assets(literal_manifest())
        self.assertIn('qwen38-27b-dflash2-exl3-4.00-vv3-approval.json', names)
        with self.assertRaises(AssertionError):
            self.assertEqual(names, LITERAL_NAMES)
        # The mutant stays self-consistent, so its own publish and verify gates still pass: only the
        # literal names catch it.
        api = self.api()
        self.assertEqual(bad.cut(api, REPO, VERSION, 'publish', repo_root=str(self.root)), 0)
        published = {a['name'] for a in api.releases[VERSION]['assets']}
        self.assertIn('synthetic-demo-vv1-bundle.tar.gz', published)
        with self.assertRaises(AssertionError):
            self.assertEqual(published, PUBLISHED_NAMES)


class DevReleaseApi(FakeApi):
    """FakeApi whose dev releases are routed by tag and whose dev tags resolve to commits."""
    def __init__(self, fixtures, reviews, dev_releases, dev_tags, **kwargs):
        super().__init__(fixtures, reviews, **kwargs)
        self.dev_releases = dev_releases
        self.dev_tags = dev_tags
        self.downloads = []

    def get(self, path, raw=False, **kwargs):
        if f'/repos/{DEV}/releases/tags/' in path:
            return self.dev_releases.get(urllib.parse.unquote(path.rsplit('/', 1)[1]))
        if f'/repos/{DEV}/git/ref/tags/' in path:
            commit = self.dev_tags.get(urllib.parse.unquote(path.rsplit('/', 1)[1]))
            return None if commit is None else {'object': {'sha': commit, 'type': 'commit'}}
        return super().get(path, raw=raw, **kwargs)

    def download(self, path):
        self.downloads.append(path)
        return super().download(path)


def publish_model(seal, first_id, **seal_changes):
    """Seal ``seal`` (optionally by reference to an engine release), write its five dev assets and return
    (approval doc, approval path, dev release metadata, {asset id: bytes})."""
    doc = seal.seal(**seal_changes)
    if 'archive' in seal_changes:
        write(seal.assets, doc['bundle']['archive'], Path(seal_changes['archive']).read_bytes())
    e.write_evidence_archive(seal.assets, doc['evidence_asset']['members'], seal.assets / 'evidence.tar.gz')
    path = write(seal.assets, 'approval.json', encoded(doc))
    names = ['approval.json'] + [a['archive'] for a in e.approval_assets(doc)]
    assets = [{'id': first_id + i, 'name': name, 'size': (seal.assets / name).stat().st_size,
               'digest': 'sha256:' + e.sha256_file(seal.assets / name)} for i, name in enumerate(names)]
    release = {'id': first_id * 10, 'tag_name': doc['approve_tag'], 'draft': False, 'prerelease': False, 'assets': assets}
    return doc, path, release, {first_id + i: (seal.assets / name).read_bytes() for i, name in enumerate(names)}


def prod_root(root, registry, approved, models, engine_tag, artifacts):
    """A prod checkout as cut_release reads it: registry, allow-list, one sealed tree per model, manifest, notes."""
    write(root, 'runtimes/templates/APPROVALS.json', encoded(registry))
    write(root, 'runtimes/templates/APPROVED.json', encoded({'schema': 'rexl3.approved-templates.v1', 'templates': approved}))
    entries = []
    for family, model, version, path, seal in models:
        tree = root / 'runtimes' / family / model / version
        write(tree, 'approval.json', path.read_bytes())
        write(tree.parent, 'MODEL.json', e.json_bytes(seal.model_doc(family, model)))
        write(tree.parent, 'LATEST', (version + '\n').encode())
        entries.append({'family': family, 'model': model, 'version': version, 'runtime_id': 'synthetic-demo',
                        'approval_sha256': sha(path.read_bytes())})
    doc = {'schema': 'rexl3.release-manifest.v1', 'version': VERSION,
           'engine': {'dev_commit': 'a' * 40, 'dev_tree': 'b' * 40, 'engine_release_tag': engine_tag, 'artifacts': artifacts},
           'models': entries, 'notes': 'first release'}
    write(root, f'releases/{VERSION}/manifest.json', encoded(doc))
    write(root, f'releases/{VERSION}/NOTES.md', b'# notes\n')
    write(root, 'CHANGELOG.md', f'# Changelog\n\n## [{VERSION}] - 2026-11-03\n\nx\n'.encode())
    write(root, '.github/CODEOWNERS', b'* @am423 @vcruz305\n')
    return doc


class EngineReleaseCutTest(unittest.TestCase):
    """A bundle without bin/ whose executables come from the dev engine release (the v0.1.0 route)."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        # cut_release's work directories (mkdtemp) land in this test's directory and go with it
        stage_root = patch.object(tempfile, 'tempdir', self.tmp.name)
        stage_root.start()
        self.addCleanup(stage_root.stop)
        self.f = SyntheticSeal(self.tmp.name)
        self.tag = 'engine-v0.1.2-' + self.f.commit[:8]
        self.asset = 'rexl3-serve-0.1.2-%s-linux-aarch64-sm121' % self.f.commit[:8]
        self.binary = self.f.bundle['bin/rexl3-serve']
        remote = base / 'engine-release'
        remote.mkdir()
        (remote / self.asset).write_bytes(self.binary)
        (remote / self.asset).chmod(0o755)
        stripped = base / 'stripped' / self.f.archive.name
        stripped.parent.mkdir()
        self.f.tar(stripped, {k: v for k, v in self.f.bundle.items() if not k.startswith('bin/')})
        self.doc, path, release, assets = publish_model(
            self.f, 1, archive=stripped, engine_binary_files={'bin/rexl3-serve': remote / self.asset}, engine_release_tag=self.tag)
        self.assertNotIn('bin/rexl3-serve', self.doc['bundle']['members'])
        registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {f'{FAMILY}/{MODEL}/v1': {
            'approval_sha256': sha(path.read_bytes()), 'source_commit': self.f.commit, 'source_tree': self.f.tree,
            'runtime_id': 'synthetic-demo'}}}
        self.root = base / 'prod'
        self.manifest = prod_root(self.root, registry, self.f.approved, [(FAMILY, MODEL, 'v1', path, self.f)],
                                  self.tag, {self.asset: sha(self.binary)})
        assets[99] = self.binary
        self.fixtures = {'assets': assets}
        self.engine_release = {'id': 60, 'tag_name': self.tag, 'draft': False, 'prerelease': False,
                               'assets': [{'id': 99, 'name': self.asset, 'size': len(self.binary),
                                           'digest': 'sha256:' + sha(self.binary)}]}
        self.dev_releases = {release['tag_name']: release, self.tag: self.engine_release}
        self.dev_tags = {self.tag: self.f.commit}

    def api(self):
        return DevReleaseApi(self.fixtures, [{'state': 'APPROVED', 'user': {'login': 'am423'}}],
                             self.dev_releases, self.dev_tags)

    def test_publish_verifies_the_approval_with_the_engine_release_binaries(self):
        api = self.api()
        self.assertEqual(c.cut(api, REPO, VERSION, 'publish', repo_root=str(self.root)), 0)
        release = api.releases[VERSION]
        self.assertIs(release['draft'], False)
        self.assertEqual({a['name'] for a in release['assets']}, c.model_assets(self.manifest))
        self.assertIn(self.asset, {a['name'] for a in release['assets']})
        self.assertEqual(dict(api.uploads)[self.asset], self.binary)

    def test_the_engine_binary_is_staged_beside_the_model_it_belongs_to(self):
        api = self.api()
        work = c.download_dev_assets(api, self.manifest, self.root)
        self.addCleanup(shutil.rmtree, work, True)
        staged = work / '.stage' / FAMILY / MODEL / 'v1' / 'engine' / self.asset
        self.assertEqual(staged.read_bytes(), self.binary)
        self.assertEqual(staged.stat().st_mode & 0o777, 0o755)
        self.assertEqual({p.name for p in work.iterdir() if p.is_file()}, c.model_assets(self.manifest))

    def test_without_the_engine_binaries_the_approval_does_not_verify(self):
        # the defect this guards: verifying a by-reference approval with no engine directory
        bad = mutant('engine_dir = stage_engine_binaries(api, doc, dest / \'engine\', engine_tag)', 'engine_dir = None')
        with self.assertRaisesRegex(ValueError, 'engine binary files are required'):
            bad.download_dev_assets(self.api(), self.manifest, self.root)

    def test_a_missing_or_resized_engine_asset_is_refused(self):
        good = self.engine_release['assets'][0]
        for change in ({'name': 'other'}, {'size': len(self.binary) + 1}):
            self.engine_release['assets'][0] = dict(good, **change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'engine release asset missing or size differs'):
                c.download_dev_assets(self.api(), self.manifest, self.root)

    def test_a_wrong_engine_digest_or_wrong_bytes_are_refused(self):
        self.engine_release['assets'][0]['digest'] = 'sha256:' + '0' * 64
        with self.assertRaisesRegex(ValueError, 'engine release asset digest differs'):
            c.download_dev_assets(self.api(), self.manifest, self.root)
        self.engine_release['assets'][0]['digest'] = None
        self.fixtures['assets'][99] = self.binary[:-1] + bytes([self.binary[-1] ^ 1])
        with self.assertRaisesRegex(ValueError, 'engine asset bytes differ from the approval'):
            c.download_dev_assets(self.api(), self.manifest, self.root)

    def test_an_engine_tag_on_another_commit_is_refused(self):
        self.dev_tags[self.tag] = '0' * 40
        with self.assertRaisesRegex(ValueError, 'does not name the approved engine commit'):
            c.download_dev_assets(self.api(), self.manifest, self.root)
        del self.dev_tags[self.tag]
        with self.assertRaisesRegex(ValueError, 'dev tag missing'):
            c.download_dev_assets(self.api(), self.manifest, self.root)

    def test_an_unpublished_or_absent_engine_release_is_refused(self):
        for key, value in (('draft', True), ('prerelease', True)):
            with self.subTest(key=key):
                self.engine_release[key] = value
                with self.assertRaisesRegex(ValueError, 'dev engine release is not published'):
                    c.download_dev_assets(self.api(), self.manifest, self.root)
                self.engine_release[key] = False
        del self.dev_releases[self.tag]
        with self.assertRaisesRegex(ValueError, 'dev engine release missing'):
            c.download_dev_assets(self.api(), self.manifest, self.root)

    def test_an_engine_release_that_changes_while_it_downloads_is_refused(self):
        api = self.api()
        answers = iter([self.engine_release, dict(self.engine_release, id=61)])
        dev_get = api.get

        def get(path, raw=False, **kwargs):
            if path.endswith('/releases/tags/' + self.tag):
                return next(answers)
            return dev_get(path, raw=raw, **kwargs)
        api.get = get
        with self.assertRaisesRegex(ValueError, 'changed while downloading'):
            c.download_dev_assets(api, self.manifest, self.root)

    def test_an_approval_naming_another_engine_release_than_the_manifest_is_refused(self):
        self.manifest['engine']['engine_release_tag'] = 'engine-v0.1.2-' + '0' * 8
        with self.assertRaisesRegex(ValueError, 'the manifest names engine-v0.1.2-00000000'):
            c.download_dev_assets(self.api(), self.manifest, self.root)

    def test_a_bundle_with_its_own_binary_takes_nothing_from_the_engine_release(self):
        plain = {k: v for k, v in self.doc.items() if k != 'engine_binaries'}
        target = Path(self.tmp.name) / 'unused-engine'
        self.assertIsNone(c.stage_engine_binaries(self.api(), plain, target, self.tag))
        self.assertFalse(target.exists())


class SeparateStageTest(unittest.TestCase):
    """Two models whose evidence shares member paths with different bytes, as every runtime's receipts do."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        # cut_release's work directories (mkdtemp) land in this test's directory and go with it
        stage_root = patch.object(tempfile, 'tempdir', self.tmp.name)
        stage_root.start()
        self.addCleanup(stage_root.stop)
        (base / 'a').mkdir()
        (base / 'b').mkdir()
        # a different pack revision gives b different receipt bytes regardless of the commit clock
        self.a = SyntheticSeal(base / 'a')
        self.b = SyntheticSeal(base / 'b', edit_lock=lambda packs: packs['target'].update(revision='e' * 40))
        doc_a, path_a, release_a, assets_a = publish_model(self.a, 1)
        doc_b, path_b, release_b, assets_b = publish_model(self.b, 11, version=2)
        members_a, members_b = doc_a['evidence_asset']['members'], doc_b['evidence_asset']['members']
        self.shared = sorted(k for k in members_a if k in members_b and members_a[k] != members_b[k])
        self.second = ('qwen3.8', 'qwen3.8-flash-next')
        registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {
            f'{FAMILY}/{MODEL}/v1': {'approval_sha256': sha(path_a.read_bytes()), 'source_commit': self.a.commit,
                                     'source_tree': self.a.tree, 'runtime_id': 'synthetic-demo'},
            f'{self.second[0]}/{self.second[1]}/v2': {'approval_sha256': sha(path_b.read_bytes()), 'source_commit': self.b.commit,
                                                      'source_tree': self.b.tree, 'runtime_id': 'synthetic-demo'}}}
        self.engine = b'SYNTHETIC ENGINE ARTIFACT'
        approved = dict(self.a.approved)
        self.assertEqual(approved, self.b.approved)
        self.root = base / 'prod'
        self.manifest = prod_root(self.root, registry, approved,
                                  [(FAMILY, MODEL, 'v1', path_a, self.a), (*self.second, 'v2', path_b, self.b)],
                                  'engine-v0.1.2-4208d848', {'rexl3-serve.tar.gz': sha(self.engine)})
        engine_release = {'id': 60, 'tag_name': 'engine-v0.1.2-4208d848', 'draft': False, 'prerelease': False,
                          'assets': [{'id': 99, 'name': 'rexl3-serve.tar.gz', 'size': len(self.engine),
                                      'digest': 'sha256:' + sha(self.engine)}]}
        self.fixtures = {'assets': {**assets_a, **assets_b, 99: self.engine}}
        self.dev_releases = {release_a['tag_name']: release_a, release_b['tag_name']: release_b,
                             'engine-v0.1.2-4208d848': engine_release}

    def api(self):
        return DevReleaseApi(self.fixtures, [{'state': 'APPROVED', 'user': {'login': 'am423'}}], self.dev_releases, {})

    def test_the_fixture_evidence_really_collides(self):
        self.assertTrue(self.shared)

    def test_each_model_is_verified_in_its_own_stage(self):
        work = c.download_dev_assets(self.api(), self.manifest, self.root)
        self.addCleanup(shutil.rmtree, work, True)
        self.assertEqual((work / '.stage' / FAMILY / MODEL / 'v1' / 'approval.json').read_bytes(),
                         (self.a.assets / 'approval.json').read_bytes())
        self.assertEqual((work / '.stage' / self.second[0] / self.second[1] / 'v2' / 'approval.json').read_bytes(),
                         (self.b.assets / 'approval.json').read_bytes())
        self.assertEqual({p.name for p in work.iterdir() if p.is_file()}, c.model_assets(self.manifest))

    def test_publish_with_two_models_uploads_both(self):
        api = self.api()
        self.assertEqual(c.cut(api, REPO, VERSION, 'publish', repo_root=str(self.root)), 0)
        self.assertEqual({a['name'] for a in api.releases[VERSION]['assets']}, c.model_assets(self.manifest))

    def test_one_shared_stage_hits_the_evidence_collision(self):
        # the defect this guards: every model extracted into one shared stage directory
        bad = mutant("dest = work / '.stage' / item['family'] / item['model'] / item['version']\n        dest.mkdir(parents=True)",
                     "dest = work / '.stage'\n        dest.mkdir(parents=True, exist_ok=True)")
        with self.assertRaisesRegex(ValueError, 'destination conflict'):
            bad.download_dev_assets(self.api(), self.manifest, self.root)


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
