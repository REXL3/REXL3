"""Mock-only GitHub promotion tests: no network, credentials, or production writes.

Covers the family/model layout: registry keys <family>/<model>/v<N> carrying runtime_id, dev tags
derived from runtime_id, MODEL.json write/validate, and the model/<model> branch flow.
"""
import copy
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools import promote_runtime as p
from tools import release_evidence as e
from tools import model_layout as m
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import encoded, sha, write

FAMILY, MODEL = 'qwen3.8', 'qwen3.8-27b'

# Pin a deterministic git identity for every commit these tests make. Git falls back to the OS user
# name when no identity is configured; GitHub runner images may not resolve one, so the commit in
# prepare_promotion fails with "Author identity unknown" unless the environment pins it.
os.environ.setdefault('GIT_AUTHOR_NAME', 'synthetic')
os.environ.setdefault('GIT_AUTHOR_EMAIL', 'synthetic@example.invalid')
os.environ.setdefault('GIT_COMMITTER_NAME', 'synthetic')
os.environ.setdefault('GIT_COMMITTER_EMAIL', 'synthetic@example.invalid')


def result(value=None, rc=0, error=''):
    return subprocess.CompletedProcess([], rc, json.dumps(value) if value is not None else '', error)


class PromotionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        self.approval, self.doc, self.registry = self.f.published_fixture_model(FAMILY, MODEL)
        self.assets = self._model_assets_dir()

    def _model_assets_dir(self):
        root = Path(self.tmp.name) / 'assets'
        from tools.test_release_evidence import stage_assets
        stage_assets(root / FAMILY / MODEL / 'v1', self.f, self.doc)
        return str(root)

    def test_weak_schema_never_promotes(self):
        doc = dict(self.doc, schema='rexl3.profile-approval.v1')
        self.approval.write_bytes(e.json_bytes(doc))
        with self.assertRaises((ValueError, SystemExit)):
            p.verify_approval(self.approval, self.f.approved, self.registry)

    def test_extract_matches_exact_inventory_and_no_symlink_dest(self):
        dest = Path(self.tmp.name) / 'tree'
        p.extract_tree(self.f.archive, dest, self.doc, approval_path=self.approval)
        self.assertEqual((dest / 'approval.json').read_bytes(), self.approval.read_bytes())
        self.assertFalse((dest / 'bin').exists())
        self.assertFalse((dest / 'release.lock.json').exists())
        (dest / 'launch.sh').write_bytes(b'edited')
        with self.assertRaises(ValueError):
            p.extract_tree(self.f.archive, dest, self.doc, approval_path=self.approval)
        link = Path(self.tmp.name) / 'link'
        link.symlink_to(dest, target_is_directory=True)
        with self.assertRaises(ValueError):
            p.extract_tree(self.f.archive, link, self.doc, approval_path=self.approval)

    def test_tag_relation_and_versions_fail_before_network(self):
        with patch.object(p, 'gh', side_effect=AssertionError('network must not run')):
            for runtime, tag, version in [('x', 'approve-y-1', 1), ('../x', 'approve-../x-1', 1),
                                          ('x', 'approve-x-0', 0), ('x', 'approve-x-01', 1)]:
                with self.subTest(tag=tag), self.assertRaises(ValueError):
                    p.validate_request(runtime, tag, version)

    def test_nonzero_git_and_gh_fail_closed(self):
        with patch.object(p.subprocess, 'run', return_value=result(rc=1, error='synthetic failure')):
            with self.assertRaises(ValueError):
                p.git('status', cwd=self.f.repo)
            with self.assertRaises(ValueError):
                p.gh('release', 'view', 'tag')

    def test_prepare_refuses_an_unregistered_family_model_version(self):
        with self.assertRaises(ValueError):
            p.prepare_promotion(FAMILY, MODEL, 2, approve_tag='approve-synthetic-demo-2',
                                target_commit=self.f.commit, prod_checkout=self.f.repo,
                                assets_dir=self.assets, registry=self.registry,
                                approved_templates=self.f.approved, meta=None)

    def test_prepare_refuses_when_approval_runtime_id_differs_from_registry(self):
        with self.assertRaises(ValueError):
            p.prepare_promotion(FAMILY, MODEL, 1, approve_tag='approve-synthetic-demo-1',
                                target_commit=self.f.commit, prod_checkout=self.f.repo,
                                assets_dir=self.assets, registry=self.registry,
                                approved_templates=self.f.approved, meta=None)
        forged = copy.deepcopy(self.registry)
        forged['approvals'][f'{FAMILY}/{MODEL}/v1']['runtime_id'] = 'other-runtime'
        with self.assertRaises(ValueError):
            p.prepare_promotion(FAMILY, MODEL, 1, approve_tag='approve-synthetic-demo-1',
                                target_commit=self.f.commit, prod_checkout=self.f.repo,
                                assets_dir=self.assets, registry=forged,
                                approved_templates=self.f.approved, meta=None)

    def make_prod(self, base_commit=None):
        root = Path(self.tmp.name) / 'production'
        tree = root / 'runtimes' / FAMILY / MODEL / 'v1'
        p.extract_tree(self.f.archive, tree, self.doc, approval_path=self.approval)
        write(tree.parent, 'MODEL.json', e.json_bytes(self.f.model_doc(FAMILY, MODEL)))
        write(tree.parent, 'LATEST', b'v1\n')
        for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=synthetic', '-c', 'user.email=synthetic@example.invalid',
                     'commit', '-qm', 'synthetic promotion')]:
            subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True)
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(['git', '-C', str(root), 'update-ref', 'refs/remotes/origin/main', commit], check=True)
        return root, commit

    def make_prod_bare(self):
        """A clean production checkout with no model content yet (tree is extracted by prepare)."""
        root = Path(self.tmp.name) / 'production'
        root.mkdir(parents=True, exist_ok=True)
        for args in [('init', '-q'), ('-c', 'user.name=synthetic', '-c', 'user.email=synthetic@example.invalid',
                     'commit', '--allow-empty', '-qm', 'synthetic production base')]:
            subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True)
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(['git', '-C', str(root), 'update-ref', 'refs/remotes/origin/main', commit], check=True)
        return root, commit

    def test_prepare_dry_run_verifies_then_prints_without_writing(self):
        root, commit = self.make_prod_bare()
        with patch.object(p, 'gh', side_effect=AssertionError('no GitHub write expected in dry run')):
            code = p.prepare_promotion(FAMILY, MODEL, 1, approve_tag='approve-synthetic-demo-1',
                                       target_commit=commit, prod_checkout=str(root),
                                       assets_dir=self.assets, registry=self.registry,
                                       approved_templates=self.f.approved, meta=dict(
                                           display_name='Synthetic Demo', engine_family='synthetic',
                                           architectures=['SyntheticForConditionalGeneration'],
                                           pack={'format': 'exl3', 'bits': '4.00'}), dry_run=True)
        self.assertEqual(code, 0)
        self.assertFalse((root / 'runtimes' / FAMILY / MODEL / 'v1').exists())

    def test_prepare_creates_model_branch_with_tree_model_json_and_latest(self):
        root, commit = self.make_prod_bare()
        calls = []
        original_git = p.git

        def gh(*args, **kwargs):
            calls.append(args)
            if args[:2] == ('pr', 'create'):
                return result({'url': 'https://github.com/REXL3/REXL3/pull/99'})
            if args[:2] == ('pr', 'view'):
                head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                      capture_output=True, text=True, check=True).stdout.strip()
                return result({'headRefOid': head, 'state': 'OPEN', 'url': 'https://github.com/REXL3/REXL3/pull/99'})
            if args[0] == 'api':
                return result({'id': 1})
            raise AssertionError('unexpected gh call: ' + repr(args))

        with patch.object(p, 'gh', side_effect=gh), patch.object(p, 'git') as git_mock:
            def real_git(*args, **kwargs):
                if args[0] in ('push', 'ls-remote', 'fetch'):
                    if args[0] == 'ls-remote':
                        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                              capture_output=True, text=True, check=True).stdout.strip()
                        return subprocess.CompletedProcess([], 0, head + '\trefs/heads/model/qwen3.8-27b\n', '')
                    if args[0] == 'fetch':
                        raise ValueError('fatal: could not fetch')
                    return subprocess.CompletedProcess([], 0, '', '')
                return original_git(*args, **kwargs)
            git_mock.side_effect = real_git
            code = p.prepare_promotion(FAMILY, MODEL, 1, approve_tag='approve-synthetic-demo-1',
                                       target_commit=commit, prod_checkout=str(root),
                                       assets_dir=self.assets, registry=self.registry,
                                       approved_templates=self.f.approved, meta=dict(
                                           display_name='Synthetic Demo', engine_family='synthetic',
                                           architectures=['SyntheticForConditionalGeneration'],
                                           pack={'format': 'exl3', 'bits': '4.00'}))
        self.assertEqual(code, 0)
        branch = root / 'runtimes' / FAMILY / MODEL
        self.assertTrue((branch / 'v1/approval.json').is_file())
        self.assertEqual((branch / 'LATEST').read_text(), 'v1\n')
        mdl = e.load_json(branch / 'MODEL.json')
        self.assertEqual(mdl['schema'], m.MODEL_SCHEMA)
        self.assertEqual(mdl['runtime_id'], 'synthetic-demo')
        self.assertEqual(mdl['family'], FAMILY)
        self.assertEqual(mdl['model'], MODEL)
        current = subprocess.run(['git', '-C', str(root), 'rev-parse', '--abbrev-ref', 'HEAD'],
                                 capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(current, 'model/qwen3.8-27b')
        self.assertTrue(any(args[:2] == ('pr', 'create') for args in calls))

    def test_prepare_validates_an_existing_model_json(self):
        root, commit = self.make_prod()
        with patch.object(p, 'gh', side_effect=AssertionError('tree exists: no write expected')), \
             patch.object(p, 'verify_tree', return_value=[]) as vt:
            code = p.prepare_promotion(FAMILY, MODEL, 1, approve_tag='approve-synthetic-demo-1',
                                       target_commit=commit, prod_checkout=str(root),
                                       assets_dir=self.assets, registry=self.registry,
                                       approved_templates=self.f.approved, meta=None)
        self.assertEqual(code, 0)
        vt.assert_called_once()

    def test_prepare_refuses_model_json_drift(self):
        root, commit = self.make_prod()
        write(root / 'runtimes' / FAMILY / MODEL, 'MODEL.json',
             e.json_bytes(self.f.model_doc(FAMILY, MODEL, display_name='Drifted')))
        with self.assertRaises(ValueError):
            p.prepare_promotion(FAMILY, MODEL, 1, approve_tag='approve-synthetic-demo-1',
                                target_commit=commit, prod_checkout=str(root),
                                assets_dir=self.assets, registry=self.registry,
                                approved_templates=self.f.approved, meta=None)

    # ---- the CLI: --download into the cache, then the promotion from that same cache

    def cli_policy(self):
        """Write the allow-list and the registry the CLI reads; return their paths and the registry pin."""
        policy = Path(self.tmp.name) / 'policy'
        approved = write(policy, 'APPROVED.json', e.json_bytes({'templates': self.f.approved}))
        registry = write(policy, 'APPROVALS.json', e.json_bytes(self.registry))
        return approved, registry, e.sha256_file(registry)

    def cli_gh(self, root, tag):
        def gh(*args, **kwargs):
            if args[:2] == ('release', 'download'):
                return self.mock_download(*args, **kwargs)
            if args[:2] == ('pr', 'create'):
                return result({'url': 'https://github.com/REXL3/REXL3/pull/99'})
            if args[:2] == ('pr', 'view'):
                head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                      capture_output=True, text=True, check=True).stdout.strip()
                return result({'headRefOid': head, 'state': 'OPEN', 'url': 'https://github.com/REXL3/REXL3/pull/99'})
            raise AssertionError('unexpected gh call: ' + repr(args))
        return gh

    def cli_git(self, root):
        original_git = p.git

        def git(*args, **kwargs):
            if args[0] == 'ls-remote':
                head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                      capture_output=True, text=True, check=True).stdout.strip()
                return subprocess.CompletedProcess([], 0, head + '\trefs/heads/model/qwen3.8-27b\n', '')
            if args[0] == 'fetch':
                raise ValueError('fatal: could not fetch')       # a new model: no remote model branch yet
            if args[0] == 'push':
                return subprocess.CompletedProcess([], 0, '', '')
            return original_git(*args, **kwargs)
        return git

    def test_download_then_promotion_from_the_same_cache_directory(self):
        # The documented command: --download caches the dev release under <cache>/<family>/<model>/v<N>/ and the
        # promotion reads it there. The promotion used to read approval.json, the archives and engine/ from the
        # cache root and refused with "missing/nonregular/hardlinked file: approval.json".
        root, commit = self.make_prod_bare()
        cache = Path(self.tmp.name) / 'cache'
        approved, registry, pin = self.cli_policy()
        tag = 'approve-synthetic-demo-1'
        argv = ['--family', FAMILY, '--model', MODEL, '--version', '1', '--approve-tag', tag,
                '--target-commit', commit, '--prod-checkout', str(root), '--assets-dir', str(cache), '--download',
                '--approved', str(approved), '--trusted-approvals', str(registry), '--trusted-approvals-sha256', pin,
                '--display-name', 'Synthetic Demo', '--engine-family', 'synthetic', '--pack-format', 'exl3',
                '--architecture', 'SyntheticForConditionalGeneration', '--pack', 'bits=4.00']
        with patch.object(p, 'release_metadata', return_value=self.metadata(tag)), \
             patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=self.cli_gh(root, tag)), patch.object(p, 'git', side_effect=self.cli_git(root)):
            self.assertEqual(p.main(argv + ['--dry-run']), 0)
            self.assertFalse((root / 'runtimes').exists(), 'a dry run writes nothing in the checkout')
            self.assertEqual(p.main(argv), 0)
        release = cache / FAMILY / MODEL / 'v1'
        self.assertEqual((release / 'approval.json').read_bytes(), self.approval.read_bytes())
        self.assertFalse((cache / 'approval.json').exists(), 'nothing is read from or copied to the cache root')
        tree = root / 'runtimes' / FAMILY / MODEL
        self.assertEqual((tree / 'v1' / 'approval.json').read_bytes(), self.approval.read_bytes())
        self.assertEqual((tree / 'LATEST').read_text(), 'v1\n')
        self.assertEqual(e.load_json(tree / 'MODEL.json')['runtime_id'], 'synthetic-demo')
        current = subprocess.run(['git', '-C', str(root), 'rev-parse', '--abbrev-ref', 'HEAD'],
                                 capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(current, 'model/qwen3.8-27b')

    def test_metadata_flags_are_checked_with_the_registry_runtime_id_before_any_download(self):
        approved, registry, pin = self.cli_policy()
        base = ['--family', FAMILY, '--model', MODEL, '--version', '1', '--approve-tag', 'approve-synthetic-demo-1',
                '--target-commit', 'a' * 40, '--prod-checkout', self.tmp.name, '--assets-dir', self.assets, '--download',
                '--approved', str(approved), '--trusted-approvals', str(registry), '--trusted-approvals-sha256', pin,
                '--display-name', 'Synthetic Demo', '--pack-format', 'exl3', '--architecture', 'SyntheticForConditionalGeneration']
        with patch.object(p, 'download_release') as download, patch.object(p, 'prepare_promotion', return_value=0) as prepare:
            self.assertEqual(p.main(base + ['--engine-family', 'synthetic']), 0)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(prepare.call_args.kwargs['meta']['engine_family'], 'synthetic')
            download.reset_mock()
            self.assertEqual(p.main(base + ['--engine-family', 'Not An Id']), 2)      # bad metadata: refused first
            download.assert_not_called()

    # ---- download path unchanged (dev releases only)

    def metadata(self, tag):
        names = ['approval.json'] + [v['archive'] for k, v in self.doc.items() if k == 'bundle' or k.endswith('_asset')]
        return {'id': 123, 'tag_name': tag, 'target_commitish': self.f.commit, 'draft': False, 'prerelease': False, 'assets': [
            {'id': i + 1, 'name': name, 'size': (self.f.assets / name).stat().st_size,
             'digest': 'sha256:' + e.sha256_file(self.f.assets / name)} for i, name in enumerate(names)]}

    def mock_download(self, *args, **kwargs):
        self.assertEqual(args[:2], ('release', 'download'))
        destination = Path(args[args.index('-D') + 1])
        for name in ['approval.json'] + [v['archive'] for k, v in self.doc.items() if k == 'bundle' or k.endswith('_asset')]:
            shutil.copyfile(self.f.assets / name, destination / name)
        return result()

    def test_download_counters_that_move_during_a_download_are_not_a_change(self):
        tag = 'approve-synthetic-demo-1'
        calls = {'n': 0}

        def gh(*args, **kwargs):
            if args[0] == 'api':
                calls['n'] += 1
                doc = self.metadata(tag)
                doc['updated_at'] = '2026-10-24T00:00:%02dZ' % calls['n']
                for asset in doc['assets']:
                    asset.update(download_count=calls['n'] * 3, updated_at='2026-10-24T00:01:%02dZ' % calls['n'],
                                 state='uploaded')
                return result(doc)
            return self.mock_download(*args, **kwargs)

        with patch.object(p, 'tag_commit', return_value=self.f.commit), patch.object(p, 'gh', side_effect=gh):
            got = p.download_release(tag, Path(self.tmp.name) / 'download', registry=self.registry,
                                     approved_templates=self.f.approved, key=f'{FAMILY}/{MODEL}/v1')
        self.assertEqual(got.read_bytes(), self.approval.read_bytes())
        self.assertGreaterEqual(calls['n'], 2, 'the release is read before and after the download')

    def test_registry_key_for_tag(self):
        self.assertEqual(p.registry_key_for_tag('approve-synthetic-demo-1', p.DEV_REPO), 'synthetic-demo/v1')
        self.assertEqual(p.registry_key_for_tag('approve-a-b-c-12', p.DEV_REPO), 'a-b-c/v12')
        with self.assertRaises(ValueError):
            p.registry_key_for_tag('prod/synthetic-demo/v3', p.PROD_REPO)

    def test_unregistered_approval_is_refused_before_any_network_call(self):
        for tag in ('approve-synthetic-demo-2', 'approve-other-demo-1'):
            with self.subTest(tag=tag), \
                 patch.object(p, 'gh', side_effect=AssertionError('network call before the registry check')), \
                 patch.object(p, 'release_metadata', side_effect=AssertionError('metadata read before the registry check')):
                with self.assertRaises(ValueError):
                    p.download_release(tag, Path(self.tmp.name) / 'never', registry=self.registry,
                                       approved_templates=self.f.approved)
        self.assertFalse((Path(self.tmp.name) / 'never').exists())

    def test_a_tag_that_is_not_a_promotion_tag_is_refused_before_any_network_call(self):
        for tag in ('latest', 'approve-synthetic-demo', 'approve--1', '../approve-synthetic-demo-1',
                    'approve-Synthetic Demo-1', 'approve-synthetic-demo-01', 'prod/synthetic-demo/v1', None, 7):
            with self.subTest(tag=tag), patch.object(p, 'gh', side_effect=AssertionError('network call')):
                with self.assertRaises(ValueError):
                    p.download_release(tag, Path(self.tmp.name) / 'never', registry=self.registry,
                                       approved_templates=self.f.approved)

    def test_repeated_authenticated_download_into_populated_cache(self):
        tag = 'approve-synthetic-demo-1'
        dest = Path(self.tmp.name) / 'download'
        with patch.object(p, 'release_metadata', return_value=self.metadata(tag)), \
             patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=self.mock_download) as cli:
            for _ in range(2):
                got = p.download_release(tag, dest, registry=self.registry, approved_templates=self.f.approved, key=f'{FAMILY}/{MODEL}/v1')
                self.assertEqual(got.read_bytes(), self.approval.read_bytes())
            self.assertEqual(cli.call_count, 2)
            (dest / 'qualification.json').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                p.download_release(tag, dest, registry=self.registry, approved_templates=self.f.approved, key=f'{FAMILY}/{MODEL}/v1')
            self.assertEqual((dest / 'qualification.json').read_bytes(), b'tampered')

    def test_dev_retry_rejects_replacement_release_and_every_asset_id_with_identical_bytes(self):
        original = self.metadata('approve-synthetic-demo-1')
        cases = [('release', None), ('target', None), ('digest-removed', 0)] + [('asset', i) for i in range(len(original['assets']))]
        for number, (case, index) in enumerate(cases):
            dest = Path(self.tmp.name) / ('retry-' + str(number))
            with patch.object(p, 'release_metadata', return_value=original), \
                 patch.object(p, 'tag_commit', return_value=self.f.commit), \
                 patch.object(p, 'gh', side_effect=self.mock_download):
                p.download_release('approve-synthetic-demo-1', dest, registry=self.registry,
                                   approved_templates=self.f.approved, key=f'{FAMILY}/{MODEL}/v1')
            before = {f.relative_to(dest).as_posix(): f.read_bytes() for f in dest.rglob('*') if f.is_file()}
            changed = copy.deepcopy(original)
            if case == 'release':
                changed['id'] += 1
            elif case == 'target':
                changed['target_commitish'] = 'moved-branch'
            elif case == 'digest-removed':
                del changed['assets'][index]['digest']
            else:
                changed['assets'][index]['id'] += 100
            with self.subTest(case=case, index=index), self.assertRaises(ValueError):
                with patch.object(p, 'release_metadata', return_value=changed), \
                     patch.object(p, 'tag_commit', return_value=self.f.commit), \
                     patch.object(p, 'gh', side_effect=self.mock_download):
                    p.download_release('approve-synthetic-demo-1', dest, registry=self.registry,
                                       approved_templates=self.f.approved, key=f'{FAMILY}/{MODEL}/v1')
            self.assertEqual({f.relative_to(dest).as_posix(): f.read_bytes() for f in dest.rglob('*') if f.is_file()}, before)

    def test_dev_retry_corrupt_or_linked_receipt_never_bootstraps_a_new_identity(self):
        for case in ('truncated', 'oversized', 'symlink', 'hardlink', 'malformed-typed'):
            dest = Path(self.tmp.name) / case
            with patch.object(p, 'release_metadata', return_value=self.metadata('approve-synthetic-demo-1')), \
                 patch.object(p, 'tag_commit', return_value=self.f.commit), \
                 patch.object(p, 'gh', side_effect=self.mock_download):
                p.download_release('approve-synthetic-demo-1', dest, registry=self.registry,
                                   approved_templates=self.f.approved, key=f'{FAMILY}/{MODEL}/v1')
            receipt = dest / '.dev-download-identity.json'
            self.assertTrue(receipt.is_file())
            original = receipt.read_bytes()
            if case == 'truncated':
                receipt.write_bytes(b'{')
            elif case == 'oversized':
                receipt.write_bytes(b' ' * (64 * 1024 + 1))
            elif case in ('symlink', 'hardlink'):
                target = Path(self.tmp.name) / (case + '.json')
                target.write_bytes(original)
                receipt.unlink()
                if case == 'symlink':
                    receipt.symlink_to(target)
                else:
                    receipt.hardlink_to(target)
            else:
                doc = e.loads(original)
                doc['release_id'] = True
                receipt.write_bytes(e.json_bytes(doc))
            with self.subTest(case=case), self.assertRaises(ValueError):
                with patch.object(p, 'release_metadata', return_value=self.metadata('approve-synthetic-demo-1')), \
                     patch.object(p, 'tag_commit', return_value=self.f.commit), \
                     patch.object(p, 'gh', side_effect=self.mock_download):
                    p.download_release('approve-synthetic-demo-1', dest, registry=self.registry,
                                       approved_templates=self.f.approved, key=f'{FAMILY}/{MODEL}/v1')
            self.assertEqual((dest / 'approval.json').read_bytes(), self.approval.read_bytes())

    def test_verify_approval_threads_precomputed(self):
        precomputed = {'synthetic-demo-1.tar.gz': {'profile.json': {}}}
        seen = {}
        def fake_verify_assets(approval_path, **kwargs):
            seen.update(kwargs)
            return self.doc, self.f.archive
        with patch.object(e, 'verify_assets', side_effect=fake_verify_assets):
            doc, archive = p.verify_approval(self.approval, self.f.approved, self.registry,
                                              precomputed=precomputed)
        self.assertIs(seen['precomputed'], precomputed)
        self.assertEqual((doc, archive), (self.doc, self.f.archive))

    def test_precomputed_for_assets_requires_streaming_digest_equality(self):
        precomputed = p.precomputed_for_assets(self.doc, self.f.assets)
        self.assertEqual(set(precomputed), {a['archive'] for a in e.approval_assets(self.doc)})
        self.assertEqual(precomputed[self.doc['bundle']['archive']], self.doc['bundle']['members'])
        name = self.doc['bundle']['archive']
        path = self.f.assets / name
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
        with self.assertRaises(ValueError):
            p.precomputed_for_assets(self.doc, self.f.assets)

    def test_download_first_verification_is_full_and_second_threads_precomputed(self):
        tag = 'approve-synthetic-demo-1'
        dest = Path(self.tmp.name) / 'download'
        seen = []
        original = p.verify_approval
        def spy(*args, **kwargs):
            seen.append((args, kwargs))
            return original(*args, **kwargs)
        with patch.object(p, 'verify_approval', side_effect=spy), \
             patch.object(p, 'release_metadata', return_value=self.metadata(tag)), \
             patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=self.mock_download):
            got = p.download_release(tag, dest, registry=self.registry,
                                      approved_templates=self.f.approved,
                                      key=f'{FAMILY}/{MODEL}/v1')
        self.assertEqual(got.read_bytes(), self.approval.read_bytes())
        self.assertEqual(len(seen), 2)
        self.assertNotIn('precomputed', seen[0][1])
        precomputed = seen[1][1]['precomputed']
        self.assertEqual(set(precomputed), {a['archive'] for a in e.approval_assets(self.doc)})
        self.assertEqual(precomputed[self.doc['bundle']['archive']], self.doc['bundle']['members'])

    def test_verify_approval_precomputed_rejects_tampered_cached_archive(self):
        tag = 'approve-synthetic-demo-1'
        dest = Path(self.tmp.name) / 'download'
        with patch.object(p, 'release_metadata', return_value=self.metadata(tag)), \
             patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=self.mock_download):
            p.download_release(tag, dest, registry=self.registry,
                                approved_templates=self.f.approved,
                                key=f'{FAMILY}/{MODEL}/v1')
        name = self.doc['engine_source_asset']['archive']
        path = dest / name
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
        precomputed = {a['archive']: a['members'] for a in e.approval_assets(self.doc)}
        with self.assertRaises(ValueError):
            p.verify_approval(dest / 'approval.json', self.f.approved, self.registry,
                               key=f'{FAMILY}/{MODEL}/v1', precomputed=precomputed)

    def test_extract_tree_preverified_skips_redecompression_but_keeps_digest(self):
        dest = Path(self.tmp.name) / 'tree'
        with patch('tools.release_evidence.inspect_archive',
                   side_effect=AssertionError('must not run')):
            p.extract_tree(self.f.archive, dest, self.doc, approval_path=self.approval,
                           preverified=True)
        expected = dict(self.doc['source_asset']['members'])
        expected['approval.json'] = dict(e.record(dest / 'approval.json'), mode=0o644)
        self.assertEqual(p.tree_inventory(dest), expected)
        data = self.f.archive.read_bytes()
        self.f.archive.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
        with self.assertRaises(ValueError):
            p.extract_tree(self.f.archive, Path(self.tmp.name) / 'other', self.doc,
                           approval_path=self.approval, preverified=True)

    def test_prepare_dry_run_uses_preverified_extraction_no_decompression(self):
        root, commit = self.make_prod_bare()
        with patch('tools.release_evidence.inspect_archive',
                   side_effect=AssertionError('must not run')), \
             patch.object(p, 'gh', side_effect=AssertionError('no GitHub write expected in dry run')):
            code = p.prepare_promotion(FAMILY, MODEL, 1, approve_tag='approve-synthetic-demo-1',
                                       target_commit=commit, prod_checkout=str(root),
                                       assets_dir=self.assets, registry=self.registry,
                                       approved_templates=self.f.approved, meta=None, dry_run=True)
        self.assertEqual(code, 0)
        self.assertFalse((root / 'runtimes' / FAMILY / MODEL / 'v1').exists())

    def test_prepare_promotion_threads_precomputed_and_skips_redecompression(self):
        root, commit = self.make_prod_bare()
        calls = []
        original_git = p.git

        def gh(*args, **kwargs):
            calls.append(args)
            if args[:2] == ('pr', 'create'):
                return result({'url': 'https://github.com/REXL3/REXL3/pull/99'})
            if args[:2] == ('pr', 'view'):
                head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                      capture_output=True, text=True, check=True).stdout.strip()
                return result({'headRefOid': head, 'state': 'OPEN', 'url': 'https://github.com/REXL3/REXL3/pull/99'})
            if args[0] == 'api':
                return result({'id': 1})
            raise AssertionError('unexpected gh call: ' + repr(args))

        with patch.object(p, 'gh', side_effect=gh), patch.object(p, 'git') as git_mock, \
             patch('tools.release_evidence.inspect_archive',
                   side_effect=AssertionError('must not run')):
            def real_git(*args, **kwargs):
                if args[0] in ('push', 'ls-remote', 'fetch'):
                    if args[0] == 'ls-remote':
                        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                              capture_output=True, text=True, check=True).stdout.strip()
                        return subprocess.CompletedProcess([], 0, head + '\trefs/heads/model/qwen3.8-27b\n', '')
                    if args[0] == 'fetch':
                        raise ValueError('fatal: could not fetch')
                    return subprocess.CompletedProcess([], 0, '', '')
                return original_git(*args, **kwargs)
            git_mock.side_effect = real_git
            code = p.prepare_promotion(FAMILY, MODEL, 1, approve_tag='approve-synthetic-demo-1',
                                       target_commit=commit, prod_checkout=str(root),
                                       assets_dir=self.assets, registry=self.registry,
                                       approved_templates=self.f.approved, meta=dict(
                                           display_name='Synthetic Demo', engine_family='synthetic',
                                           architectures=['SyntheticForConditionalGeneration'],
                                           pack={'format': 'exl3', 'bits': '4.00'}))
        self.assertEqual(code, 0)
        self.assertTrue((root / 'runtimes' / FAMILY / MODEL / 'v1/approval.json').is_file())


if __name__ == '__main__':
    unittest.main()
