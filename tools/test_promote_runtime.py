"""Mock-only GitHub promotion tests: no network, credentials, or production writes."""
import copy
import json
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools import promote_runtime as p
from tools import release_evidence as e
from tools.test_approve_profile import SyntheticSeal


def result(value=None, rc=0, error=''):
    return subprocess.CompletedProcess([], rc, json.dumps(value) if value is not None else '', error)


class PromotionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        self.approval, self.doc, self.registry = self.f.published_fixture()

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

    def test_published_release_metadata_must_match_exact_id_commit_and_assets(self):
        good = {'id': 123, 'tag_name': 'prod/synthetic-demo/v1', 'draft': False, 'prerelease': False,
                'target_commitish': self.f.commit, 'assets': [{'id': 456, 'name': 'approval.json', 'size': self.approval.stat().st_size}]}
        # Metadata alone is deliberately insufficient: bytes and tag resolution are required.
        with patch.object(p, 'gh', return_value=result(good)), patch.object(p, 'download_release', side_effect=ValueError('bytes differ')):
            with self.assertRaises(ValueError):
                p.verify_published_release('prod/synthetic-demo/v1', self.f.commit, self.approval,
                                           self.registry, self.f.approved, Path(self.tmp.name) / 'remote')

    def test_publish_only_must_validate_merged_tree_before_create(self):
        with patch.object(p, 'verify_merged_tree', side_effect=ValueError('wrong merged version')), \
             patch.object(p, 'gh', side_effect=AssertionError('publication before merged validation')):
            with self.assertRaises(ValueError):
                p.publish_release('synthetic-demo', 'approve-synthetic-demo-1', 1, self.f.assets,
                    target_commit=self.f.commit, prod_checkout=self.f.repo,
                    registry=self.registry, approved_templates=self.f.approved)

    def make_prod(self):
        root = Path(self.tmp.name) / 'production'
        tree = root / 'runtimes/synthetic-demo/v1'
        p.extract_tree(self.f.archive, tree, self.doc, approval_path=self.approval)
        for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=synthetic', '-c', 'user.email=synthetic@example.invalid',
                     'commit', '-qm', 'synthetic promotion')]:
            subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True)
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(['git', '-C', str(root), 'update-ref', 'refs/remotes/origin/main', commit], check=True)
        return root, commit

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

    def api_doc_with_counters(self, tag, step):
        # What GitHub returns: download_count and updated_at move between two reads of the same release.
        doc = self.metadata(tag)
        doc['updated_at'] = '2026-10-24T00:00:%02dZ' % step
        for asset in doc['assets']:
            asset.update(download_count=step * 3, updated_at='2026-10-24T00:01:%02dZ' % step, state='uploaded')
        return doc

    def test_download_counters_that_move_during_a_download_are_not_a_change(self):
        tag = 'approve-synthetic-demo-1'
        calls = {'n': 0}

        def gh(*args, **kwargs):
            if args[0] == 'api':
                calls['n'] += 1
                return result(self.api_doc_with_counters(tag, calls['n']))
            return self.mock_download(*args, **kwargs)

        with patch.object(p, 'tag_commit', return_value=self.f.commit), patch.object(p, 'gh', side_effect=gh):
            got = p.download_release(tag, Path(self.tmp.name) / 'download', registry=self.registry,
                                     approved_templates=self.f.approved)
        self.assertEqual(got.read_bytes(), self.approval.read_bytes())
        self.assertGreaterEqual(calls['n'], 2, 'the release is read before and after the download')

    def test_a_real_change_during_a_download_is_still_refused_when_counters_also_move(self):
        tag = 'approve-synthetic-demo-1'
        calls = {'n': 0}

        def gh(*args, **kwargs):
            if args[0] == 'api':
                calls['n'] += 1
                doc = self.api_doc_with_counters(tag, calls['n'])
                if calls['n'] > 1:
                    doc['assets'][1]['size'] += 1
                return result(doc)
            return self.mock_download(*args, **kwargs)

        with patch.object(p, 'tag_commit', return_value=self.f.commit), patch.object(p, 'gh', side_effect=gh):
            with self.assertRaises(ValueError):
                p.download_release(tag, Path(self.tmp.name) / 'download', registry=self.registry,
                                   approved_templates=self.f.approved)

    def test_stable_release_drops_counters_and_keeps_identity(self):
        doc = self.api_doc_with_counters('approve-synthetic-demo-1', 7)
        stable = p.stable_release(doc)
        self.assertNotIn('updated_at', stable)
        for asset in stable['assets']:
            self.assertEqual(set(asset), {'id', 'name', 'size', 'digest', 'state'})
        other = self.api_doc_with_counters('approve-synthetic-demo-1', 99)
        self.assertEqual(stable, p.stable_release(other))
    def test_unregistered_approval_is_refused_before_any_network_call(self):
        for tag, repo in (('approve-synthetic-demo-2', p.DEV_REPO), ('approve-other-demo-1', p.DEV_REPO),
                          ('prod/synthetic-demo/v2', p.PROD_REPO)):
            with self.subTest(tag=tag), \
                 patch.object(p, 'gh', side_effect=AssertionError('network call before the registry check')), \
                 patch.object(p, 'release_metadata', side_effect=AssertionError('metadata read before the registry check')):
                with self.assertRaises(ValueError):
                    p.download_release(tag, Path(self.tmp.name) / 'never', repo=repo, registry=self.registry,
                                       approved_templates=self.f.approved)
        self.assertFalse((Path(self.tmp.name) / 'never').exists())

    def test_a_tag_that_is_not_a_promotion_tag_is_refused_before_any_network_call(self):
        for tag in ('latest', 'approve-synthetic-demo', 'approve--1', '../approve-synthetic-demo-1', 'approve-Synthetic Demo-1',
                    'approve-synthetic-demo-01', 'prod/synthetic-demo/v1/extra', None, 7):
            with self.subTest(tag=tag), patch.object(p, 'gh', side_effect=AssertionError('network call')):
                with self.assertRaises(ValueError):
                    p.download_release(tag, Path(self.tmp.name) / 'never', registry=self.registry,
                                       approved_templates=self.f.approved)

    def test_registry_key_for_tag(self):
        self.assertEqual(p.registry_key_for_tag('approve-synthetic-demo-1', p.DEV_REPO), 'synthetic-demo/v1')
        self.assertEqual(p.registry_key_for_tag('approve-a-b-c-12', p.DEV_REPO), 'a-b-c/v12')
        self.assertEqual(p.registry_key_for_tag('prod/synthetic-demo/v3', p.PROD_REPO), 'synthetic-demo/v3')

    def test_repeated_authenticated_download_into_populated_cache(self):
        tag = 'approve-synthetic-demo-1'
        dest = Path(self.tmp.name) / 'download'
        with patch.object(p, 'release_metadata', return_value=self.metadata(tag)), \
             patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=self.mock_download) as cli:
            for _ in range(2):
                got = p.download_release(tag, dest, registry=self.registry, approved_templates=self.f.approved)
                self.assertEqual(got.read_bytes(), self.approval.read_bytes())
            self.assertEqual(cli.call_count, 2)
            (dest / 'qualification.json').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                p.download_release(tag, dest, registry=self.registry, approved_templates=self.f.approved)
            self.assertEqual((dest / 'qualification.json').read_bytes(), b'tampered')

    def dev_download(self, dest, metadata=None, download=None):
        tag = 'approve-synthetic-demo-1'
        with patch.object(p, 'release_metadata', return_value=metadata or self.metadata(tag)), \
             patch.object(p, 'tag_commit', return_value=self.f.commit), \
             patch.object(p, 'gh', side_effect=download or self.mock_download):
            return p.download_release(tag, dest, registry=self.registry, approved_templates=self.f.approved)

    def test_dev_download_persists_bounded_exact_identity_and_retry_is_immutable(self):
        dest = Path(self.tmp.name) / 'download'
        tag = 'approve-synthetic-demo-1'
        self.dev_download(dest)
        receipt = dest / '.dev-download-identity.json'
        self.assertTrue(receipt.is_file(), 'verified dev download needs a durable identity receipt')
        original = receipt.read_bytes()
        self.assertLessEqual(len(original), 64 * 1024)
        doc = e.load_json(receipt)
        self.assertEqual(doc['schema'], 'rexl3.dev-download.v1')
        self.assertEqual((doc['repo'], doc['release_id'], doc['tag'], doc['target_commit'], doc['target_commitish']),
                         (p.DEV_REPO, 123, tag, self.f.commit, self.f.commit))
        expected = {a['name']: dict(id=a['id'], size=a['size'], digest=a['digest'],
                    sha256=e.sha256_file(self.f.assets / a['name'])) for a in self.metadata(tag)['assets']}
        self.assertEqual(doc['assets'], expected)
        self.dev_download(dest)
        self.assertEqual(receipt.read_bytes(), original)
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)

    def test_dev_retry_rejects_replacement_release_and_every_asset_id_with_identical_bytes(self):
        original = self.metadata('approve-synthetic-demo-1')
        cases = [('release', None), ('target', None), ('digest-removed', 0)] + [('asset', i) for i in range(len(original['assets']))]
        for number, (case, index) in enumerate(cases):
            dest = Path(self.tmp.name) / ('retry-' + str(number))
            self.dev_download(dest, original)
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
                self.dev_download(dest, changed)
            self.assertEqual({f.relative_to(dest).as_posix(): f.read_bytes() for f in dest.rglob('*') if f.is_file()}, before)

    def test_dev_receipt_optional_github_digest_still_pins_measured_bytes(self):
        dest = Path(self.tmp.name) / 'download'
        metadata = self.metadata('approve-synthetic-demo-1')
        for asset in metadata['assets']:
            asset.pop('digest')
        self.dev_download(dest, metadata)
        receipt = dest / '.dev-download-identity.json'
        self.assertTrue(receipt.is_file())
        stored = e.load_json(receipt)['assets']
        self.assertTrue(all(a['digest'] is None and e.HEX64.fullmatch(a['sha256']) for a in stored.values()))
        self.dev_download(dest, metadata)
        metadata['assets'][0]['id'] += 1
        with self.assertRaises(ValueError):
            self.dev_download(dest, metadata)

    def test_dev_retry_corrupt_or_linked_receipt_never_bootstraps_a_new_identity(self):
        for case in ('truncated', 'oversized', 'symlink', 'hardlink', 'malformed-typed'):
            dest = Path(self.tmp.name) / case
            self.dev_download(dest)
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
                self.dev_download(dest)
            self.assertEqual((dest / 'approval.json').read_bytes(), self.approval.read_bytes())

    def test_dev_incomplete_download_or_partial_cache_never_gets_a_verified_receipt(self):
        dest = Path(self.tmp.name) / 'download'
        def interrupted(*args, **kwargs):
            stage = Path(args[args.index('-D') + 1])
            (stage / 'approval.json').write_bytes(b'partial')
            raise ValueError('synthetic interruption')
        with self.assertRaises(ValueError):
            self.dev_download(dest, download=interrupted)
        self.assertFalse((dest / '.dev-download-identity.json').exists())
        self.dev_download(dest)
        # Even with a completed prior identity, every subsequent attempt must
        # download and verify again rather than trusting a marker/cache shortcut.
        with self.assertRaises(ValueError):
            self.dev_download(dest, download=interrupted)
        partial = Path(self.tmp.name) / 'partial'
        partial.mkdir()
        (partial / 'approval.json').write_bytes(b'partial')
        with self.assertRaises(ValueError):
            self.dev_download(partial)
        self.assertFalse((partial / '.dev-download-identity.json').exists())
        self.assertEqual((partial / 'approval.json').read_bytes(), b'partial')

    def test_first_verified_download_identity_survives_interrupted_cache_copy(self):
        dest = Path(self.tmp.name) / 'download'
        original_copy = shutil.copyfileobj
        def interrupted(incoming, output, *args, **kwargs):
            if Path(output.name).is_relative_to(dest):
                output.write(b'partial')
                raise OSError('synthetic cache copy interruption')
            return original_copy(incoming, output, *args, **kwargs)
        with patch.object(p.shutil, 'copyfileobj', side_effect=interrupted):
            with self.assertRaises(ValueError):
                self.dev_download(dest)
        self.assertTrue((dest / '.dev-download-identity.json').is_file(),
                        'a fully verified remote identity must survive later cache-copy interruption')
        original = (dest / '.dev-download-identity.json').read_bytes()
        metadata = self.metadata('approve-synthetic-demo-1')
        metadata['id'] += 1
        with self.assertRaises(ValueError):
            self.dev_download(dest, metadata)
        # Presence of the identity receipt never authorizes partial cache bytes.
        with self.assertRaises(ValueError):
            self.dev_download(dest)
        self.assertEqual((dest / '.dev-download-identity.json').read_bytes(), original)

    def test_atomic_identity_install_failure_never_trusts_a_partial_receipt(self):
        original_link = p.os.link
        for after in (False, True):
            dest = Path(self.tmp.name) / ('install-' + str(after))
            def failed_link(source, target, **kwargs):
                if after:
                    original_link(source, target, **kwargs)
                raise OSError('synthetic atomic identity install interruption')
            with patch.object(p.os, 'link', side_effect=failed_link):
                with self.assertRaises(ValueError):
                    self.dev_download(dest)
            receipt = dest / '.dev-download-identity.json'
            self.assertEqual(receipt.exists(), after)
            if after:
                self.assertEqual(receipt.stat().st_nlink, 1)
                self.assertEqual(e.load_json(receipt)['release_id'], 123)
            self.dev_download(dest)
            self.assertEqual((dest / 'approval.json').read_bytes(), self.approval.read_bytes())
            self.assertEqual([f.name for f in dest.iterdir() if f.name.startswith('.download-identity-')], [])

    def test_dev_download_errors_do_not_disclose_raw_command_or_metadata(self):
        canary = 'CANARY_PRIVATE /private/path token=synthetic-secret'
        def failed(*args, **kwargs):
            raise ValueError(canary)
        with self.assertRaises(ValueError) as caught:
            self.dev_download(Path(self.tmp.name) / 'download', download=failed)
        self.assertNotIn(canary, str(caught.exception))
        self.assertNotIn('/private/path', str(caught.exception))

    def test_dev_receipt_covers_additional_assets_without_a_three_asset_switch(self):
        # Generic transport coverage only; this defines no engine-source semantics.
        extra = self.f.assets / 'extra-fixture.tar.gz'
        self.f.tar(extra, {'extra-fixture.txt': b'SYNTHETIC OPTIONAL ASSET'})
        self.doc['extra_fixture_asset'] = e.asset_record(extra)
        self.approval.write_bytes(e.json_bytes(self.doc))
        self.registry['approvals']['synthetic-demo/v1']['approval_sha256'] = e.sha256_file(self.approval)
        dest = Path(self.tmp.name) / 'download'
        self.dev_download(dest)
        receipt = dest / '.dev-download-identity.json'
        self.assertTrue(receipt.is_file())
        self.assertIn(extra.name, e.load_json(receipt)['assets'])
        metadata = self.metadata('approve-synthetic-demo-1')
        next(a for a in metadata['assets'] if a['name'] == extra.name)['id'] += 100
        with self.assertRaises(ValueError):
            self.dev_download(dest, metadata)

    def test_existing_published_release_retry_is_verified_noop(self):
        root, target = self.make_prod()
        tag = 'prod/synthetic-demo/v1'
        metadata = self.metadata(tag)
        with patch.object(p, 'release_metadata', return_value=metadata), \
             patch.object(p, 'tag_commit', return_value=target), \
             patch.object(p, 'gh', side_effect=self.mock_download) as cli:
            for _ in range(2):
                self.assertEqual(p.publish_release('synthetic-demo', 'approve-synthetic-demo-1', 1, self.f.assets,
                    target_commit=target, prod_checkout=root, registry=self.registry, approved_templates=self.f.approved), 0)
            self.assertTrue(all(c.args[:2] == ('release', 'download') for c in cli.call_args_list))
            self.assertFalse((root / 'runtimes/synthetic-demo/LATEST').exists())
        receipt = e.load_json(self.f.assets / 'publication-synthetic-demo-v1.json')
        self.assertEqual(set(receipt['assets']), {a['name'] for a in metadata['assets']})
        self.assertEqual(len(receipt['assets']), 5)  # approval + four distinct archives
        for asset in metadata['assets']:
            self.assertEqual(receipt['assets'][asset['name']], dict(
                id=asset['id'], size=asset['size'], digest=asset['digest'],
                sha256=e.sha256_file(self.f.assets / asset['name'])))

    def test_new_publish_targets_exact_merged_commit_and_reads_back(self):
        root, target = self.make_prod()
        tag = 'prod/synthetic-demo/v1'
        created = []
        def metadata(*args, **kwargs):
            if kwargs.get('missing_ok') and not created:
                return None
            return self.metadata(tag)
        def cli(*args, **kwargs):
            if args[:2] == ('release', 'create'):
                self.assertEqual(args[args.index('--target') + 1], target)
                created.append(args)
                return result()
            return self.mock_download(*args, **kwargs)
        with patch.object(p, 'release_metadata', side_effect=metadata), \
             patch.object(p, 'tag_commit', return_value=target), patch.object(p, 'gh', side_effect=cli):
            self.assertEqual(p.publish_release('synthetic-demo', 'approve-synthetic-demo-1', 1, self.f.assets,
                target_commit=target, prod_checkout=root, registry=self.registry, approved_templates=self.f.approved), 0)
        self.assertEqual(len(created), 1)
        uploaded = {Path(value).name for value in created[0][3:created[0].index('-R')]}
        self.assertEqual(uploaded, {'approval.json'} | {a['archive'] for a in p.approval_assets(self.doc)})
        self.assertEqual(len(uploaded), 5)
        self.assertFalse((root / 'runtimes/synthetic-demo/LATEST').exists())

    def test_publication_retry_rejects_each_asset_identity_change_without_overwrite(self):
        root, target = self.make_prod()
        original = self.metadata('prod/synthetic-demo/v1')
        def publish(metadata):
            with patch.object(p, 'release_metadata', return_value=metadata), \
                 patch.object(p, 'tag_commit', return_value=target), patch.object(p, 'gh', side_effect=self.mock_download):
                return p.publish_release('synthetic-demo', 'approve-synthetic-demo-1', 1, self.f.assets,
                    target_commit=target, prod_checkout=root, registry=self.registry, approved_templates=self.f.approved)
        publish(original)
        receipt = self.f.assets / 'publication-synthetic-demo-v1.json'
        saved = receipt.read_bytes()
        for i, asset in enumerate(original['assets']):
            for field in ('id', 'size', 'digest'):
                changed = copy.deepcopy(original)
                changed['assets'][i][field] = ('sha256:' + 'f' * 64 if field == 'digest' else asset[field] + 100)
                with self.subTest(asset=asset['name'], field=field), self.assertRaises(ValueError):
                    publish(changed)
                self.assertEqual(receipt.read_bytes(), saved)
                self.assertFalse((root / 'runtimes/synthetic-demo/LATEST').exists())

    def test_missing_or_tampered_engine_download_never_changes_verified_cache(self):
        dest = Path(self.tmp.name) / 'download'
        self.dev_download(dest)
        before = {f.relative_to(dest).as_posix(): f.read_bytes() for f in dest.rglob('*') if f.is_file()}
        name = self.doc['engine_source_asset']['archive']
        for missing in (True, False):
            def corrupt(*args, **kwargs):
                result = self.mock_download(*args, **kwargs)
                engine = Path(args[args.index('-D') + 1]) / name
                if missing:
                    engine.unlink()
                else:
                    engine.write_bytes(b'SYNTHETIC TRANSPORT TAMPER')
                return result
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.dev_download(dest, download=corrupt)
            self.assertEqual({f.relative_to(dest).as_posix(): f.read_bytes() for f in dest.rglob('*') if f.is_file()}, before)
        self.dev_download(dest)
        self.assertEqual({f.relative_to(dest).as_posix(): f.read_bytes() for f in dest.rglob('*') if f.is_file()}, before)

    def test_wrong_merge_and_remote_commit_refused(self):
        root, target = self.make_prod()
        with patch.object(p, 'gh', side_effect=AssertionError('must fail before GitHub')):
            with self.assertRaises(ValueError):
                p.publish_release('synthetic-demo', 'approve-synthetic-demo-1', 1, self.f.assets,
                    target_commit=self.f.commit, prod_checkout=root, registry=self.registry, approved_templates=self.f.approved)
        with patch.object(p, 'release_metadata', return_value=self.metadata('prod/synthetic-demo/v1')), \
             patch.object(p, 'tag_commit', return_value='f' * 40), patch.object(p, 'gh', side_effect=self.mock_download):
            with self.assertRaises(ValueError):
                p.publish_release('synthetic-demo', 'approve-synthetic-demo-1', 1, self.f.assets,
                    target_commit=target, prod_checkout=root, registry=self.registry, approved_templates=self.f.approved)

    def test_release_id_change_between_verified_retries_refused(self):
        root, target = self.make_prod()
        tag = 'prod/synthetic-demo/v1'
        metadata = self.metadata(tag)
        with patch.object(p, 'release_metadata', return_value=metadata), \
             patch.object(p, 'tag_commit', return_value=target), patch.object(p, 'gh', side_effect=self.mock_download):
            p.publish_release('synthetic-demo', 'approve-synthetic-demo-1', 1, self.f.assets,
                target_commit=target, prod_checkout=root, registry=self.registry, approved_templates=self.f.approved)
            metadata['id'] = 999
            with self.assertRaises(ValueError):
                p.publish_release('synthetic-demo', 'approve-synthetic-demo-1', 1, self.f.assets,
                    target_commit=target, prod_checkout=root, registry=self.registry, approved_templates=self.f.approved)


if __name__ == '__main__':
    unittest.main()
