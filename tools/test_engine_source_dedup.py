"""Shared engine corresponding-source assets: `engine.source_archives` in the release manifest.

A manifest with the key publishes one engine-source archive per distinct sealed digest; a manifest
without it keeps the per-model assets, so the published v0.1.0 and v0.1.1 asset sets do not change.
The real manifests and sealed approvals on this checkout are read offline; everything else is synthetic.
"""
import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import assemble_release_notes as notes
from tools import cut_release as c
from tools import release_evidence as e
from tools import release_manifest as rm
from tools import test_cut_release as t
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import encoded, sha, write

ROOT = Path(__file__).resolve().parents[1]
V011_SOURCE = 'rexl3-engine-source-0.1.1-75031112.tar.gz'
V011_SOURCE_SHA256 = 'f6280178d6229a72fceb89c298658cb12e9738de919c356cfc1c0b1be28826b4'
V011_SOURCE_BYTES = 12313591
V011_BINARIES = {'rexl3-serve-0.1.1-75031112-linux-aarch64-sm121', 'rexl3-cluster-0.1.1-75031112-linux-aarch64-sm121'}


def real_manifest(version):
    return e.load_json(ROOT / 'releases' / version / 'manifest.json')


def engine_sources(names):
    return {name for name in names if 'engine-source' in name}


class PublishedReleasesUnchangedTest(unittest.TestCase):
    """The asset sets of the two published releases, from their real manifests and sealed approvals."""

    def test_v0_1_0_keeps_its_thirty_assets(self):
        manifest = rm.validate_manifest(real_manifest('v0.1.0'), 'v0.1.0')
        self.assertNotIn(rm.SOURCE_ARCHIVES, manifest['engine'])
        names = c.model_assets(manifest, ROOT)
        self.assertEqual(len(names), 30)
        self.assertEqual(len(engine_sources(names)), 5)
        # v0.1.0 also published this archive, with the digest two of its per-model copies carry:
        # a rule keyed on digest alone would have dropped those two published assets.
        self.assertIn('rexl3-engine-0.1.0-afc9b5df-source.tar.gz', names)
        self.assertIn('dsv41-flash-1017-sage-1.55-v1-engine-source.tar.gz', names)

    def test_v0_1_1_keeps_its_forty_seven_assets(self):
        manifest = rm.validate_manifest(real_manifest('v0.1.1'), 'v0.1.1')
        self.assertNotIn(rm.SOURCE_ARCHIVES, manifest['engine'])
        names = c.model_assets(manifest, ROOT)
        self.assertEqual(len(names), 9 * 5 + 2)
        self.assertEqual(len(engine_sources(names)), 9)
        self.assertEqual({name for name in names if name.startswith('rexl3-')}, V011_BINARIES)

    def test_v0_1_1_notes_assets_section_is_unchanged(self):
        manifest = real_manifest('v0.1.1')
        section = notes.assets_section(manifest)
        self.assertNotIn('engine corresponding source', section)
        self.assertIn(section, (ROOT / 'releases/v0.1.1/NOTES.md').read_text(encoding='utf-8'))


class NineSealsOneSourceTest(unittest.TestCase):
    """The nine v0.1.1 seals shipped again with `engine.source_archives`: 9 x 4 + 1 source + 2 binaries."""

    def setUp(self):
        self.manifest = real_manifest('v0.1.1')
        self.manifest['version'] = 'v0.1.2'
        self.block = rm.derive_source_archives(self.manifest, ROOT)
        self.manifest['engine'][rm.SOURCE_ARCHIVES] = self.block
        rm.validate_manifest(self.manifest, 'v0.1.2')

    def test_one_archive_is_derived_from_nine_approvals(self):
        self.assertEqual(self.block, {V011_SOURCE: {'sha256': V011_SOURCE_SHA256, 'bytes': V011_SOURCE_BYTES}})

    def test_the_asset_set_is_thirty_nine(self):
        names = c.model_assets(self.manifest, ROOT)
        self.assertEqual(len(names), 9 * 4 + 1 + 2)
        self.assertEqual(engine_sources(names), {V011_SOURCE})
        self.assertEqual({name for name in names if name.startswith('rexl3-')}, V011_BINARIES | {V011_SOURCE})
        for item in self.manifest['models']:
            prefix = f"{item['runtime_id']}-{item['version']}"
            for suffix in ('-approval.json', '-bundle.tar.gz', '-source.tar.gz', '-evidence.tar.gz'):
                self.assertIn(prefix + suffix, names)

    def test_every_approval_still_names_its_own_digest(self):
        mapped = rm.shared_engine_sources(self.manifest, ROOT)
        self.assertEqual(len(mapped), 9)
        self.assertEqual(set(mapped.values()), {V011_SOURCE})
        for item in self.manifest['models']:
            approval = e.load_json(ROOT / 'runtimes' / item['family'] / item['model'] / item['version'] / 'approval.json')
            self.assertEqual(approval['engine_source_asset']['sha256'], V011_SOURCE_SHA256)
            self.assertNotEqual(approval['engine_source_asset']['archive'], V011_SOURCE)

    def test_a_wrong_manifest_digest_or_size_is_refused(self):
        for change in ({'sha256': '0' * 64}, {'bytes': V011_SOURCE_BYTES + 1}):
            manifest = copy.deepcopy(self.manifest)
            manifest['engine'][rm.SOURCE_ARCHIVES][V011_SOURCE].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'engine source differs from the shared archive'):
                c.model_assets(manifest, ROOT)

    def test_an_archive_under_another_engine_release_name_is_refused(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['engine'][rm.SOURCE_ARCHIVES] = {'rexl3-engine-source-0.1.1-00000000.tar.gz': self.block[V011_SOURCE]}
        with self.assertRaisesRegex(ValueError, 'engine source differs from the shared archive'):
            c.model_assets(manifest, ROOT)

    def test_an_archive_no_model_was_sealed_from_is_refused(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['engine'][rm.SOURCE_ARCHIVES]['rexl3-engine-source-0.1.1-00000000.tar.gz'] = {'sha256': '1' * 64, 'bytes': 1}
        with self.assertRaisesRegex(ValueError, 'no model was sealed from'):
            c.model_assets(manifest, ROOT)

    def test_shared_archives_need_the_production_tree(self):
        with self.assertRaisesRegex(ValueError, 'need the production tree'):
            c.model_assets(self.manifest)

    def test_manifest_shape_is_strict(self):
        good = self.block[V011_SOURCE]
        bad_blocks = (
            {}, [], {'qwen-engine-source.tar.gz': good}, {'rexl3-engine-0.1.1-75031112-source.tar.gz': good},
            {V011_SOURCE: V011_SOURCE_SHA256}, {V011_SOURCE: dict(good, extra=1)}, {V011_SOURCE: {'sha256': 'f' * 63, 'bytes': 1}},
            {V011_SOURCE: {'sha256': V011_SOURCE_SHA256, 'bytes': 0}}, {V011_SOURCE: {'sha256': V011_SOURCE_SHA256, 'bytes': True}},
            {V011_SOURCE: good, 'rexl3-engine-source-0.1.1-00000000.tar.gz': good})
        for block in bad_blocks:
            manifest = copy.deepcopy(self.manifest)
            manifest['engine'][rm.SOURCE_ARCHIVES] = block
            with self.subTest(block=block), self.assertRaises(ValueError):
                rm.validate_manifest(manifest, 'v0.1.2')

    def test_notes_list_the_shared_archive_once(self):
        section = notes.assets_section(self.manifest)
        self.assertEqual(section.count(V011_SOURCE), 1)
        self.assertIn(f'`{V011_SOURCE}` sha256 `{V011_SOURCE_SHA256}` ({V011_SOURCE_BYTES} bytes)', section)
        self.assertEqual(section.count('approval sha256'), 9)

    def test_the_command_prints_the_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain = real_manifest('v0.1.1')
            plain['version'] = 'v0.1.2'
            for item in plain['models']:
                tree = Path('runtimes') / item['family'] / item['model'] / item['version']
                write(root / tree, 'approval.json', (ROOT / tree / 'approval.json').read_bytes())
            write(root, 'releases/v0.1.2/manifest.json', encoded(plain))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(rm.main(['--repo-root', tmp, '--version', 'v0.1.2', '--source-archives']), 0)
                self.assertEqual(rm.main(['--repo-root', tmp, '--version', 'v0.1.3', '--source-archives']), 2)
            self.assertEqual(json.loads(out.getvalue()), {rm.SOURCE_ARCHIVES: self.block})
            self.assertIn('MANIFEST_REFUSED', err.getvalue())


class SameDigestCutTest(unittest.TestCase):
    """Two models sealed from one engine release with the same engine-source bytes: one published archive."""

    SECOND = ('qwen3.8', 'qwen3.8-flash-next')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
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
        by_reference = dict(archive=stripped, engine_binary_files={'bin/rexl3-serve': remote / self.asset},
                            engine_release_tag=self.tag)
        self.doc_a, path, release_a, assets_a = t.publish_model(self.f, 1, **by_reference)
        path_a = write(base, 'approval-a.json', path.read_bytes())
        (self.f.assets / 'evidence.tar.gz').unlink()  # the second seal writes its own evidence archive
        self.doc_b, path, release_b, assets_b = t.publish_model(self.f, 11, version=2, **by_reference)
        path_b = write(base, 'approval-b.json', path.read_bytes())
        self.source = self.f.engine_source_asset.read_bytes()
        self.shared = e.shared_engine_source_name(self.tag)
        self.assertEqual(self.doc_a['engine_source_asset']['sha256'], self.doc_b['engine_source_asset']['sha256'])
        self.registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {
            f'{t.FAMILY}/{t.MODEL}/v1': {'approval_sha256': sha(path_a.read_bytes()), 'source_commit': self.f.commit,
                                         'source_tree': self.f.tree, 'runtime_id': 'synthetic-demo'},
            f'{self.SECOND[0]}/{self.SECOND[1]}/v2': {'approval_sha256': sha(path_b.read_bytes()), 'source_commit': self.f.commit,
                                                      'source_tree': self.f.tree, 'runtime_id': 'synthetic-demo'}}}
        self.root = base / 'prod'
        self.manifest = t.prod_root(self.root, self.registry, self.f.approved,
                                    [(t.FAMILY, t.MODEL, 'v1', path_a, self.f), (*self.SECOND, 'v2', path_b, self.f)],
                                    self.tag, {self.asset: sha(self.binary)})
        self.manifest['engine'][rm.SOURCE_ARCHIVES] = rm.derive_source_archives(self.manifest, self.root)
        self.write_manifest()
        self.fixtures = {'assets': {**assets_a, **assets_b, 99: self.binary}}
        engine_release = {'id': 60, 'tag_name': self.tag, 'draft': False, 'prerelease': False,
                          'assets': [{'id': 99, 'name': self.asset, 'size': len(self.binary), 'digest': 'sha256:' + sha(self.binary)}]}
        self.dev_releases = {release_a['tag_name']: release_a, release_b['tag_name']: release_b, self.tag: engine_release}
        self.source_ids = [asset['id'] for release in (release_a, release_b) for asset in release['assets']
                           if asset['name'] == self.doc_a['engine_source_asset']['archive']]

    def write_manifest(self):
        write(self.root, f'releases/{t.VERSION}/manifest.json', encoded(self.manifest))

    def api(self):
        return t.DevReleaseApi(self.fixtures, [{'state': 'APPROVED', 'commit_id': t.PR_HEAD, 'user': {'login': 'am423'}}],
                               self.dev_releases, {self.tag: self.f.commit})

    def expected(self):
        return {self.asset, self.shared,
                'synthetic-demo-v1-approval.json', 'synthetic-demo-v1-bundle.tar.gz', 'synthetic-demo-v1-source.tar.gz',
                'synthetic-demo-v1-evidence.tar.gz',
                'synthetic-demo-v2-approval.json', 'synthetic-demo-v2-bundle.tar.gz', 'synthetic-demo-v2-source.tar.gz',
                'synthetic-demo-v2-evidence.tar.gz'}

    def test_the_block_names_one_archive(self):
        self.assertEqual(self.manifest['engine'][rm.SOURCE_ARCHIVES],
                         {self.shared: {'sha256': sha(self.source), 'bytes': len(self.source)}})
        self.assertEqual(len(self.source_ids), 2)
        self.assertEqual(c.model_assets(self.manifest, self.root), self.expected())

    def test_without_the_block_each_model_keeps_its_own_copy(self):
        del self.manifest['engine'][rm.SOURCE_ARCHIVES]
        names = c.model_assets(self.manifest, self.root)
        self.assertEqual(engine_sources(names), {'synthetic-demo-v1-engine-source.tar.gz', 'synthetic-demo-v2-engine-source.tar.gz'})
        work = c.download_dev_assets(self.api(), self.manifest, self.root)
        self.addCleanup(shutil.rmtree, work, True)
        self.assertEqual({p.name for p in work.iterdir() if p.is_file()}, names)

    def test_download_stages_the_archive_once(self):
        work = c.download_dev_assets(self.api(), self.manifest, self.root)
        self.addCleanup(shutil.rmtree, work, True)
        self.assertEqual({p.name for p in work.iterdir() if p.is_file()}, self.expected())
        self.assertEqual((work / self.shared).read_bytes(), self.source)

    def test_publish_uploads_the_archive_once_and_verify_confirms_it(self):
        api = self.api()
        self.assertEqual(c.cut(api, t.REPO, t.VERSION, 'publish', repo_root=str(self.root)), 0)
        uploaded = [name for name, _ in api.uploads]
        self.assertEqual(sorted(uploaded), sorted(self.expected()))
        self.assertEqual(dict(api.uploads)[self.shared], self.source)
        api.existing_tags.add(t.VERSION)
        self.assertEqual(c.cut(api, t.REPO, t.VERSION, 'verify', repo_root=str(self.root)), 0)

    def test_verify_refuses_a_published_archive_with_other_bytes(self):
        api = self.api()
        self.assertEqual(c.cut(api, t.REPO, t.VERSION, 'publish', repo_root=str(self.root)), 0)
        asset = next(a for a in api.releases[t.VERSION]['assets'] if a['name'] == self.shared)
        for change in ({'digest': 'sha256:' + '0' * 64}, {'size': asset['size'] + 1}):
            saved = dict(asset)
            asset.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'published engine source differs from the manifest'):
                c.cut(api, t.REPO, t.VERSION, 'verify', repo_root=str(self.root))
            asset.clear()
            asset.update(saved)

    def test_download_refuses_a_manifest_digest_the_approvals_do_not_name(self):
        self.manifest['engine'][rm.SOURCE_ARCHIVES][self.shared]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'engine source differs from the shared archive'):
            c.download_dev_assets(self.api(), self.manifest, self.root)

    def test_download_refuses_tampered_dev_bytes(self):
        tampered = self.source[:-1] + bytes([self.source[-1] ^ 1])
        self.fixtures['assets'][self.source_ids[1]] = tampered
        for release in self.dev_releases.values():
            for asset in release['assets']:
                if asset['id'] == self.source_ids[1]:
                    asset['digest'] = 'sha256:' + sha(tampered)
        with self.assertRaisesRegex(ValueError, 'asset hash/bytes/inventory mismatch'):
            c.download_dev_assets(self.api(), self.manifest, self.root)

    def test_a_staged_archive_with_other_bytes_is_refused(self):
        # the defect this guards: writing the shared name without comparing it with the manifest's bytes
        real = c.hashlib_hex
        calls = []

        def second_copy_differs(data):
            if data == self.source:
                calls.append(1)
                if len(calls) > 1:
                    return '0' * 64
            return real(data)
        with patch.object(c, 'hashlib_hex', second_copy_differs), \
                self.assertRaisesRegex(ValueError, 'engine source bytes differ from the manifest'):
            c.download_dev_assets(self.api(), self.manifest, self.root)

    def test_one_name_with_two_digests_is_refused(self):
        tree = self.root / 'runtimes' / self.SECOND[0] / self.SECOND[1] / 'v2'
        doc = e.load_json(tree / 'approval.json')
        doc['engine_source_asset']['sha256'] = '0' * 64
        write(tree, 'approval.json', encoded(doc))
        self.manifest['models'][1]['approval_sha256'] = sha(encoded(doc))
        with self.assertRaisesRegex(ValueError, 'same engine source archive with different bytes'):
            rm.derive_source_archives(self.manifest, self.root)
        with self.assertRaisesRegex(ValueError, 'engine source differs from the shared archive'):
            c.model_assets(self.manifest, self.root)

    def test_local_approval_bytes_must_be_the_manifests(self):
        tree = self.root / 'runtimes' / t.FAMILY / t.MODEL / 'v1'
        write(tree, 'approval.json', (tree / 'approval.json').read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'local approval bytes differ from the manifest'):
            c.model_assets(self.manifest, self.root)

    def test_a_bundle_with_its_own_binaries_cannot_share(self):
        plain = {k: v for k, v in self.doc_a.items() if k != 'engine_binaries'}
        with self.assertRaisesRegex(ValueError, 'names no engine release'):
            rm.shared_source_record(plain)


class MixedDigestCutTest(unittest.TestCase):
    """Two models sealed from two engine releases: one published archive per distinct digest."""

    SECOND = t.CarryForwardCutTest.SECOND
    setUp = t.CarryForwardCutTest.setUp
    api = t.CarryForwardCutTest.api

    def declare(self):
        self.manifest['engine'][rm.SOURCE_ARCHIVES] = rm.derive_source_archives(self.manifest, self.root)
        write(self.root, f'releases/{t.VERSION}/manifest.json', encoded(self.manifest))
        return {e.shared_engine_source_name(self.old_tag), e.shared_engine_source_name(self.new_tag)}

    def test_two_digests_give_two_archives(self):
        shared = self.declare()
        block = self.manifest['engine'][rm.SOURCE_ARCHIVES]
        self.assertEqual(set(block), shared)
        self.assertEqual(len({record['sha256'] for record in block.values()}), 2)
        names = c.model_assets(self.manifest, self.root)
        self.assertEqual(engine_sources(names), shared)
        self.assertEqual(len(names), 2 * 4 + 2 + 2)

    def test_publish_uploads_one_archive_per_digest(self):
        shared = self.declare()
        api = self.api()
        self.assertEqual(c.cut(api, t.REPO, t.VERSION, 'publish', repo_root=str(self.root)), 0)
        uploads = dict(api.uploads)
        self.assertEqual(set(uploads), c.model_assets(self.manifest, self.root))
        self.assertEqual(engine_sources(uploads), shared)
        self.assertEqual(uploads[e.shared_engine_source_name(self.old_tag)], self.carried.engine_source_asset.read_bytes())
        self.assertEqual(uploads[e.shared_engine_source_name(self.new_tag)], self.fresh.engine_source_asset.read_bytes())

    def test_declaring_only_one_of_two_digests_is_refused(self):
        self.declare()
        del self.manifest['engine'][rm.SOURCE_ARCHIVES][e.shared_engine_source_name(self.old_tag)]
        with self.assertRaisesRegex(ValueError, 'engine source differs from the shared archive'):
            c.model_assets(self.manifest, self.root)
        with self.assertRaisesRegex(ValueError, 'engine source differs from the shared archive'):
            c.download_dev_assets(self.api(), self.manifest, self.root)


if __name__ == '__main__':
    unittest.main()
