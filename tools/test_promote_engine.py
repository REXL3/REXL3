"""Mock-only tests for promoting a runtime whose bundle references an engine release: no network, credentials or writes."""
import copy
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
        (dest / 'engine' / self.asset).write_bytes(b'tampered')
        with patch.object(p, 'release_metadata', side_effect=metadata), patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=gh):
            with self.assertRaises(ValueError):
                p.download_release('approve-synthetic-demo-1', dest, registry=registry, approved_templates=self.f.approved)

    def test_a_route_b_download_then_promotion_read_the_cached_release_and_engine(self):
        # v0.1.0 runtimes leave their executables to the engine release: --download caches the approval release and
        # the engine binaries under <cache>/<family>/<model>/v<N>/{,engine/}, and the promotion must verify from there.
        family, model = 'qwen3.8', 'qwen3.8-27b'
        _, release_meta, names = self.publish_assets()
        approval = self.f.assets / 'approval.json'
        registry = {'schema': 'rexl3.approval-registry.v1', 'legacy': {}, 'approvals': {f'{family}/{model}/v1': {
            'approval_sha256': sha(approval.read_bytes()), 'runtime_id': self.approval['runtime_id'],
            'source_commit': self.f.commit, 'source_tree': self.f.tree}}}
        policy = Path(self.tmp.name) / 'policy'
        approved_path = write(policy, 'APPROVED.json', e.json_bytes({'templates': self.f.approved}))
        registry_path = write(policy, 'APPROVALS.json', e.json_bytes(registry))
        root = Path(self.tmp.name) / 'production'
        root.mkdir()
        for args in [('init', '-q'), ('-c', 'user.name=synthetic', '-c', 'user.email=synthetic@example.invalid',
                     'commit', '--allow-empty', '-qm', 'synthetic production base')]:
            subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True)
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(['git', '-C', str(root), 'update-ref', 'refs/remotes/origin/main', commit], check=True)
        head = lambda: subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                      check=True).stdout.strip()

        def metadata(tag, repo, **kwargs):
            return release_meta if tag == release_meta['tag_name'] else self.meta

        def gh(*args, **kwargs):
            if args[:2] == ('release', 'download'):
                target = Path(args[args.index('-D') + 1])
                if '-p' in args:
                    shutil.copy(self.remote / args[args.index('-p') + 1], target)
                else:
                    for name in names:
                        shutil.copy(self.f.assets / name, target / name)
                return subprocess.CompletedProcess([], 0, '', '')
            if args[:2] == ('pr', 'create'):
                return subprocess.CompletedProcess([], 0, '', '')
            if args[:2] == ('pr', 'view'):
                return subprocess.CompletedProcess([], 0, json.dumps({'headRefOid': head(), 'state': 'OPEN', 'url': 'u'}), '')
            raise AssertionError('unexpected gh call: ' + repr(args))

        original_git = p.git

        def git(*args, **kwargs):
            if args[0] == 'ls-remote':
                return subprocess.CompletedProcess([], 0, head() + '\trefs/heads/model/qwen3.8-27b\n', '')
            if args[0] == 'fetch':
                raise ValueError('fatal: could not fetch')
            if args[0] == 'push':
                return subprocess.CompletedProcess([], 0, '', '')
            return original_git(*args, **kwargs)

        cache = Path(self.tmp.name) / 'cache'
        argv = ['--family', family, '--model', model, '--version', '1', '--approve-tag', 'approve-synthetic-demo-1',
                '--target-commit', commit, '--prod-checkout', str(root), '--assets-dir', str(cache), '--download',
                '--approved', str(approved_path), '--trusted-approvals', str(registry_path),
                '--trusted-approvals-sha256', sha(registry_path.read_bytes()),
                '--display-name', 'Synthetic Demo', '--engine-family', 'synthetic', '--pack-format', 'exl3',
                '--architecture', 'SyntheticForConditionalGeneration']
        with patch.object(p, 'release_metadata', side_effect=metadata), patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=gh), patch.object(p, 'git', side_effect=git):
            self.assertEqual(p.main(argv + ['--dry-run']), 0)
            self.assertEqual(p.main(argv), 0)
        release = cache / family / model / 'v1'
        self.assertEqual((release / 'engine' / self.asset).read_bytes(), self.binary.read_bytes())
        self.assertFalse((cache / 'approval.json').exists() or (cache / 'engine').exists())
        tree = root / 'runtimes' / family / model / 'v1'
        self.assertEqual((tree / 'approval.json').read_bytes(), approval.read_bytes())
        self.assertFalse((tree / 'bin').exists(), 'the executables stay in the engine release')


if __name__ == '__main__':
    unittest.main()
