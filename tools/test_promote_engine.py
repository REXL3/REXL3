"""Mock-only tests for promoting a runtime whose bundle references an engine release: no network, credentials or writes."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools import promote_runtime as p
from tools import release_evidence as e
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import encoded, sha, write


class PromoteEngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.f = SyntheticSeal(self.tmp.name)
        self.tag = 'engine-v0.1.2-' + self.f.commit[:8]
        self.remote = root / 'remote'
        self.remote.mkdir()
        self.asset = 'rexl3-serve-0.1.2-%s-linux-aarch64-sm121' % self.f.commit[:8]
        self.binary = self.remote / self.asset
        self.binary.write_bytes(self.f.bundle['bin/rexl3-serve'])
        self.binary.chmod(0o755)
        stripped = root / 'stripped' / self.f.archive.name
        stripped.parent.mkdir()
        self.f.tar(stripped, {k: v for k, v in self.f.bundle.items() if not k.startswith('bin/')})
        self.approval = self.f.seal(archive=stripped, engine_binary_files={'bin/rexl3-serve': self.binary},
                                    engine_release_tag=self.tag)
        self.meta = {'id': 77, 'tag_name': self.tag, 'target_commitish': self.f.commit, 'draft': False, 'prerelease': False,
                     'assets': [{'id': 5, 'name': self.asset, 'size': self.binary.stat().st_size,
                                 'digest': 'sha256:' + sha(self.binary.read_bytes()), 'state': 'uploaded'}]}

    def gh(self, calls=None, serve=None):
        serve = self.remote if serve is None else serve

        def fake(*args, **kwargs):
            if calls is not None:
                calls.append([str(a) for a in args])
            if args[:2] == ('release', 'download'):
                target = Path(args[args.index('-D') + 1])
                name = args[args.index('-p') + 1]
                shutil.copy(serve / name, target / name)
            return subprocess.CompletedProcess([], 0, '', '')
        return fake

    def fetch(self, repo=p.DEV_REPO, meta=None, commit=None, serve=None):
        stage = Path(self.tmp.name) / 'stage'
        shutil.rmtree(stage, ignore_errors=True)
        with patch.object(p, 'release_metadata', side_effect=meta or (lambda *a, **k: self.meta)), \
             patch.object(p, 'tag_commit', return_value=commit or self.f.commit), \
             patch.object(p, 'gh', side_effect=self.gh(serve=serve)):
            return p.fetch_engine(self.approval, stage, repo=repo), stage

    def test_a_bundle_with_its_own_binary_needs_no_engine_release(self):
        plain = {k: v for k, v in self.approval.items() if k != 'engine_binaries'}
        self.assertIsNone(p.engine_release_tag(plain))
        self.assertIsNone(p.fetch_engine(plain, Path(self.tmp.name) / 'none', repo=p.DEV_REPO))
        self.assertEqual(p.engine_release_tag(self.approval), self.tag)

    def test_fetch_downloads_exactly_the_approved_asset_and_checks_the_tag_commit_in_dev(self):
        meta, stage = self.fetch()
        self.assertEqual(meta, self.meta)
        self.assertEqual([f.name for f in stage.iterdir()], [self.asset])

    def test_a_dev_engine_tag_on_another_commit_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'does not name the approved engine commit'):
            self.fetch(commit='0' * 40)
        # production tags the merged promotion commit, so only the bytes decide there
        self.fetch(repo=p.PROD_REPO, commit='1' * 40)

    def test_replaced_bytes_a_wrong_size_or_a_wrong_digest_are_refused(self):
        other = Path(self.tmp.name) / 'other'
        other.mkdir()
        (other / self.asset).write_bytes(self.binary.read_bytes()[:-1] + b'\0')
        with self.assertRaisesRegex(ValueError, 'bytes differ from the approval'):
            self.fetch(serve=other)
        bad = json.loads(json.dumps(self.meta))
        bad['assets'][0]['size'] += 1
        with self.assertRaisesRegex(ValueError, 'missing or size differs'):
            self.fetch(meta=lambda *a, **k: bad)
        bad = json.loads(json.dumps(self.meta))
        bad['assets'][0]['digest'] = 'sha256:' + '0' * 64
        with self.assertRaisesRegex(ValueError, 'digest differs'):
            self.fetch(meta=lambda *a, **k: bad)
        bad = json.loads(json.dumps(self.meta))
        bad['assets'][0]['name'] = 'other-name'
        with self.assertRaisesRegex(ValueError, 'missing or size differs'):
            self.fetch(meta=lambda *a, **k: bad)

    def test_a_release_that_changes_while_it_downloads_is_refused(self):
        changed = json.loads(json.dumps(self.meta))
        changed['id'] = 78
        answers = iter([self.meta, changed])
        with self.assertRaisesRegex(ValueError, 'changed while downloading'):
            self.fetch(meta=lambda *a, **k: next(answers))

    def test_publish_creates_the_engine_release_at_the_merged_commit_and_verifies_it_by_download(self):
        calls = []
        answers = iter([None, self.meta, self.meta])
        with patch.object(p, 'release_metadata', side_effect=lambda *a, **k: next(answers)), \
             patch.object(p, 'tag_commit', return_value='2' * 40), \
             patch.object(p, 'gh', side_effect=self.gh(calls)):
            verified = p.publish_engine_release(self.approval, self.remote, '2' * 40, self.tmp.name)
        self.assertEqual(verified, self.meta)
        create = [c for c in calls if c[:2] == ['release', 'create']]
        self.assertEqual(len(create), 1)
        self.assertEqual(create[0][2], self.tag)
        self.assertIn(str(self.binary), create[0])
        self.assertEqual(create[0][create[0].index('--target') + 1], '2' * 40)
        self.assertIn('--latest=false', create[0])

    def test_publish_refuses_a_created_release_whose_tag_is_elsewhere(self):
        answers = iter([None, self.meta, self.meta])
        with patch.object(p, 'release_metadata', side_effect=lambda *a, **k: next(answers)), \
             patch.object(p, 'tag_commit', return_value='3' * 40), patch.object(p, 'gh', side_effect=self.gh()):
            with self.assertRaisesRegex(ValueError, 'published engine tag commit differs'):
                p.publish_engine_release(self.approval, self.remote, '2' * 40, self.tmp.name)

    def test_an_existing_engine_release_is_reused_without_creating_and_must_verify(self):
        calls = []
        with patch.object(p, 'release_metadata', return_value=self.meta), patch.object(p, 'tag_commit', return_value='4' * 40), \
             patch.object(p, 'git', return_value=subprocess.CompletedProcess([], 0, '', '')) as git, \
             patch.object(p, 'gh', side_effect=self.gh(calls)):
            p.publish_engine_release(self.approval, self.remote, '2' * 40, self.tmp.name)
        self.assertFalse([c for c in calls if c[:2] == ['release', 'create']])
        self.assertEqual(git.call_args.args[:3], ('merge-base', '--is-ancestor', '4' * 40))
        other = Path(self.tmp.name) / 'other'
        other.mkdir()
        (other / self.asset).write_bytes(b'x' * self.binary.stat().st_size)
        with patch.object(p, 'release_metadata', return_value=self.meta), patch.object(p, 'tag_commit', return_value='4' * 40), \
             patch.object(p, 'git', return_value=subprocess.CompletedProcess([], 0, '', '')), \
             patch.object(p, 'gh', side_effect=self.gh(serve=other)):
            with self.assertRaisesRegex(ValueError, 'bytes differ from the approval'):
                p.publish_engine_release(self.approval, self.remote, '2' * 40, self.tmp.name)

    def test_publish_does_nothing_for_a_bundle_that_carries_its_binary(self):
        plain = {k: v for k, v in self.approval.items() if k != 'engine_binaries'}
        with patch.object(p, 'gh', side_effect=AssertionError('no GitHub call expected')):
            self.assertIsNone(p.publish_engine_release(plain, self.remote, '2' * 40, self.tmp.name))

    def test_verify_approval_needs_the_engine_directory(self):
        path = write(self.f.assets, 'approval.json', encoded(self.approval))
        write(self.f.assets, self.approval['bundle']['archive'], (Path(self.tmp.name) / 'stripped' / self.f.archive.name).read_bytes())
        e.write_evidence_archive(self.f.assets, self.approval['evidence_asset']['members'], self.f.assets / 'evidence.tar.gz')
        registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {'synthetic-demo/v1': {
            'approval_sha256': sha(path.read_bytes()), 'source_commit': self.f.commit, 'source_tree': self.f.tree}}}
        with self.assertRaisesRegex(ValueError, 'engine binary files are required'):
            p.verify_approval(path, self.f.approved, registry)
        approval, _ = p.verify_approval(path, self.f.approved, registry, engine_dir=self.remote)
        self.assertEqual(approval['engine_binaries'], self.approval['engine_binaries'])

    def publish_assets(self):
        """Write the sealed release assets as a release would hold them; return (registry, model release metadata)."""
        stripped = Path(self.tmp.name) / 'stripped' / self.f.archive.name
        write(self.f.assets, self.approval['bundle']['archive'], stripped.read_bytes())
        e.write_evidence_archive(self.f.assets, self.approval['evidence_asset']['members'], self.f.assets / 'evidence.tar.gz')
        path = write(self.f.assets, 'approval.json', encoded(self.approval))
        registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {'synthetic-demo/v1': {
            'approval_sha256': sha(path.read_bytes()), 'source_commit': self.f.commit, 'source_tree': self.f.tree}}}
        names = ['approval.json'] + [a['archive'] for a in e.approval_assets(self.approval)]
        model = {'id': 123, 'tag_name': 'approve-synthetic-demo-1', 'target_commitish': self.f.commit, 'draft': False,
                 'prerelease': False, 'assets': [{'id': i + 1, 'name': n, 'size': (self.f.assets / n).stat().st_size,
                 'digest': 'sha256:' + e.sha256_file(self.f.assets / n)} for i, n in enumerate(names)]}
        return registry, model, names

    def test_a_development_download_fetches_verifies_and_caches_the_engine_with_the_release(self):
        registry, model, names = self.publish_assets()
        engine_calls = []

        def metadata(tag, repo, **kwargs):
            return model if tag == model['tag_name'] else self.meta

        def gh(*args, **kwargs):
            target = Path(args[args.index('-D') + 1])
            if '-p' in args:
                engine_calls.append(args)
                shutil.copy(self.remote / args[args.index('-p') + 1], target)
            else:
                for name in names:
                    shutil.copy(self.f.assets / name, target / name)
            return subprocess.CompletedProcess([], 0, '', '')

        dest = Path(self.tmp.name) / 'download'
        with patch.object(p, 'release_metadata', side_effect=metadata), patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=gh):
            got = p.download_release('approve-synthetic-demo-1', dest, registry=registry, approved_templates=self.f.approved)
            self.assertEqual(len(engine_calls), 1, 'exactly one engine download per release download')
        self.assertEqual((dest / 'engine' / self.asset).read_bytes(), self.binary.read_bytes())
        self.assertTrue(set(names) <= {f.name for f in dest.iterdir()})
        # a cache whose engine file was altered is refused rather than trusted
        (dest / 'engine' / self.asset).write_bytes(b'tampered')
        with patch.object(p, 'release_metadata', side_effect=metadata), patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=gh):
            with self.assertRaises(ValueError):
                p.download_release('approve-synthetic-demo-1', dest, registry=registry, approved_templates=self.f.approved)


if __name__ == '__main__':
    unittest.main()
