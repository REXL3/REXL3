"""CPU-only synthetic seals whose bundle leaves bin/ to an engine release; not native qualification."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from tools import approve_profile as a
from tools import release_evidence as e
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import encoded, sha, write


class EngineByReference(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        self.tag = 'engine-v0.1.2-' + self.f.commit[:8]
        self.engine_dir = Path(self.tmp.name) / 'engine'
        self.engine_dir.mkdir()
        self.asset = 'rexl3-serve-0.1.2-%s-linux-aarch64-sm121' % self.f.commit[:8]
        self.binary = self.engine_dir / self.asset
        self.binary.write_bytes(self.f.bundle['bin/rexl3-serve'])
        self.binary.chmod(0o755)
        self.full_archive = self.f.archive
        stripped = {k: v for k, v in self.f.bundle.items() if not k.startswith('bin/')}
        self.stripped = Path(self.tmp.name) / 'stripped' / self.full_archive.name
        self.stripped.parent.mkdir()
        self.f.tar(self.stripped, stripped)
        self.kwargs = dict(archive=self.stripped, engine_binary_files={'bin/rexl3-serve': self.binary},
                           engine_release_tag=self.tag)

    def seal(self, **changes):
        return self.f.seal(**(self.kwargs | changes))

    def publish(self, doc=None):
        """Write the approval as published and return (path, doc, registry)."""
        doc = doc or self.seal()
        path = write(self.f.assets, 'approval.json', encoded(doc))
        registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {'synthetic-demo/v1': {
            'approval_sha256': sha(path.read_bytes()), 'source_commit': self.f.commit, 'source_tree': self.f.tree}}}
        return path, doc, registry

    def verify(self, doc=None, **changes):
        # verify_assets reads the bundle archive named by the approval from the approval's directory
        path, doc, registry = self.publish(doc)
        write(self.f.assets, doc['bundle']['archive'], self.stripped.read_bytes())
        if not (self.f.assets / 'evidence.tar.gz').exists():
            e.write_evidence_archive(self.f.assets, doc['evidence_asset']['members'], self.f.assets / 'evidence.tar.gz')
        return e.verify_assets(path, registry=registry, approved_templates=self.f.approved, **({'engine_dir': self.engine_dir} | changes))

    def retag(self, mutate):
        doc = self.seal()
        mutate(doc['engine_binaries'])
        return doc

    # ---- acceptance

    def test_seal_records_the_engine_release_and_leaves_the_bundle_without_bin(self):
        doc = self.seal()
        record = doc['engine_binaries']['bin/rexl3-serve']
        self.assertEqual(record, dict(release_tag=self.tag, asset=self.asset, sha256=sha(self.binary.read_bytes()),
                                      bytes=self.binary.stat().st_size, mode=0o755))
        self.assertEqual(doc['bundle'], e.asset_record(self.stripped))
        self.assertFalse([n for n in doc['bundle']['members'] if n.startswith('bin/')])
        # the engine binary is not one of the release's own transport assets
        self.assertNotIn(self.asset, {asset['archive'] for asset in e.approval_assets(doc)})
        # the wrapper projection is exactly what it was with bin/ inside the bundle
        self.assertEqual(doc['source_asset'], e.asset_record(self.f.source_asset))
        # receipts bind the same identities: release lock and binary hash are those of the full bundle
        self.assertEqual(doc['identities']['binary_sha256'], sha(self.binary.read_bytes()))
        self.assertEqual(doc['identities']['release_lock_sha256'], sha(self.f.bundle['release.lock.json']))

    def test_sealed_approval_verifies_with_the_engine_files(self):
        verified, archive = self.verify()
        self.assertEqual(verified['engine_binaries']['bin/rexl3-serve']['asset'], self.asset)

    def test_identities_equal_those_of_the_full_bundle(self):
        legacy = self.f.seal(archive=self.full_archive)
        self.assertNotIn('engine_binaries', legacy)
        by_reference = self.seal()
        self.assertEqual(by_reference['identities'], legacy['identities'])
        self.assertEqual(by_reference['profiles'], legacy['profiles'])
        self.assertEqual(by_reference['source_asset'], legacy['source_asset'])

    # ---- the sealed v1 shape is unchanged

    def test_an_approval_without_engine_binaries_verifies_exactly_as_before(self):
        path, doc, registry = self.f.published_fixture()
        self.assertNotIn('engine_binaries', doc)
        self.assertEqual(e.verify_assets(path, registry=registry, approved_templates=self.f.approved)[0], doc)
        # an engine_dir offered to such an approval is not consulted
        self.assertEqual(e.verify_assets(path, registry=registry, approved_templates=self.f.approved, engine_dir=self.engine_dir)[0], doc)

    def test_a_legacy_seal_still_refuses_a_bundle_without_bin(self):
        with self.assertRaisesRegex(ValueError, 'incomplete bundle'):
            self.f.seal(archive=self.stripped)

    def test_dropping_engine_binaries_from_an_approval_cannot_pass(self):
        doc = self.seal()
        del doc['engine_binaries']
        with self.assertRaisesRegex(ValueError, 'incomplete bundle'):
            self.verify(doc)

    def test_a_null_engine_binaries_is_refused(self):
        doc = self.seal()
        doc['engine_binaries'] = None
        with self.assertRaisesRegex(ValueError, 'engine_binaries must be an object'):
            self.verify(doc)

    # ---- refusals

    def test_verifier_refuses_rather_than_skips_without_the_engine_files(self):
        with self.assertRaisesRegex(ValueError, 'engine binary files are required'):
            self.verify(engine_dir=None)

    def test_missing_engine_file_is_refused(self):
        self.binary.unlink()
        with self.assertRaises(ValueError):
            self.verify()

    def test_binary_bytes_that_differ_from_the_approval_are_refused(self):
        doc = self.seal()
        data = bytearray(self.binary.read_bytes())
        data[-1] ^= 1
        self.binary.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, 'engine binary hash/size mismatch'):
            self.verify(doc)
        # truncation and extension too
        self.binary.write_bytes(bytes(data) + b'x')
        with self.assertRaisesRegex(ValueError, 'engine binary hash/size mismatch'):
            self.verify(doc)

    def test_sealer_refuses_a_binary_whose_hash_is_not_the_release_locks(self):
        data = bytearray(self.binary.read_bytes())
        data[-1] ^= 1
        self.binary.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, 'release inventory mismatch'):
            self.seal()

    def test_a_bin_member_left_in_the_archive_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'must not carry native executables'):
            self.seal(archive=self.full_archive)

    def test_wrong_platform_elf_is_refused(self):
        header = bytearray(self.binary.read_bytes())
        struct.pack_into('<H', header, 18, 62)  # x86-64 on an aarch64 release
        self.binary.write_bytes(bytes(header))
        item = dict(release_tag=self.tag, asset=self.asset, sha256=sha(bytes(header)), bytes=len(header), mode=0o755)
        with self.assertRaisesRegex(ValueError, 'native ELF/platform mismatch'):
            e.engine_binary_files({'bin/rexl3-serve': item}, self.engine_dir, 'aarch64')
        e.engine_binary_files({'bin/rexl3-serve': item}, self.engine_dir, 'x86_64')
        for junk in (b'#!/bin/sh\nexit 0\n' + bytes(60), b'\x7fELF\x01\x01\x01' + bytes(57)):  # script, 32-bit ELF
            self.binary.write_bytes(junk)
            item = dict(item, sha256=sha(junk), bytes=len(junk))
            with self.assertRaisesRegex(ValueError, 'native ELF/platform mismatch'):
                e.engine_binary_files({'bin/rexl3-serve': item}, self.engine_dir, 'aarch64')

    def test_record_shape_is_strict(self):
        good = self.seal()['engine_binaries']
        e.validate_engine_binaries(good)
        record = good['bin/rexl3-serve']
        bad = {
            'mode 0644': dict(record, mode=0o644), 'mode bool': dict(record, mode=True), 'mode string': dict(record, mode='0755'),
            'bytes zero': dict(record, bytes=0), 'bytes bool': dict(record, bytes=True), 'bytes float': dict(record, bytes=1.0),
            'short hash': dict(record, sha256=record['sha256'][:63]), 'upper hash': dict(record, sha256=record['sha256'].upper()),
            'path asset': dict(record, asset='dir/' + self.asset), 'dotdot asset': dict(record, asset='../' + self.asset),
            'empty asset': dict(record, asset=''), 'extra key': dict(record, url='https://example.invalid/x'),
            'missing key': {k: v for k, v in record.items() if k != 'asset'},
        }
        for name, item in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                e.validate_engine_binaries({'bin/rexl3-serve': item})
        for name, engine in {'empty': {}, 'not a dict': [record], 'no serve': {'bin/rexl3-cluster': record},
                             'unknown path': {'bin/rexl3-serve': record, 'bin/other': record},
                             'absolute path': {'/bin/rexl3-serve': record}}.items():
            with self.subTest(name), self.assertRaises(ValueError):
                e.validate_engine_binaries(engine)
        with self.assertRaisesRegex(ValueError, 'one engine release'):
            e.validate_engine_binaries({'bin/rexl3-serve': record, 'bin/rexl3-cluster': dict(record, asset='x', release_tag='engine-v0.1.2-00000000')})
        with self.assertRaisesRegex(ValueError, 'distinct'):
            e.validate_engine_binaries({'bin/rexl3-serve': record, 'bin/rexl3-cluster': record})

    def test_release_tag_names_the_engine_not_a_model_and_matches_the_native_identity(self):
        for tag in ('qwen38-27b-dflash2-exl3-4.00', 'engine-v0.1.2', 'engine-v0.1.2-4208D848', 'engine-v0.1.2-4208d84',
                    'engine-v0.1.2-4208d848-qwen', 'prod/synthetic-demo/v1', 'Engine-v0.1.2-4208d848', ''):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'invalid engine release tag'):
                self.seal(engine_release_tag=tag)
        # well formed, but not this engine's version/commit
        for tag in ('engine-v0.1.3-' + self.f.commit[:8], 'engine-v0.1.2-' + '0' * 8):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'does not name the native engine version and commit'):
                self.seal(engine_release_tag=tag)

    def test_approval_mutations_after_the_seal_are_refused(self):
        def edit(path, value):
            return lambda engine: engine['bin/rexl3-serve'].__setitem__(path, value)
        record = self.seal()['engine_binaries']['bin/rexl3-serve']
        for name, mutate in {
            'sha256': edit('sha256', '0' * 64), 'bytes': edit('bytes', record['bytes'] + 1), 'mode': edit('mode', 0o644),
            'asset': edit('asset', 'other'), 'tag': edit('release_tag', 'engine-v0.1.2-' + '0' * 8),
            'extra key': edit('url', 'https://example.invalid/x'),
            'extra path': lambda engine: engine.__setitem__('bin/rexl3-cluster', dict(record, asset='second')),
        }.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.verify(self.retag(mutate))

    def test_release_inventory_must_equal_archive_plus_engine_records(self):
        # a release lock that lists no executable at all: the engine record has nothing to match
        release = copy.deepcopy(self.f.release)
        del release['files']['bin/rexl3-serve']
        bundle = {k: v for k, v in self.f.bundle.items() if not k.startswith('bin/')}
        bundle['release.lock.json'] = encoded(release)
        self.f.tar(self.stripped, bundle)
        with self.assertRaisesRegex(ValueError, 'release inventory mismatch'):
            self.seal()
        # a release lock that lists an extra file the archive does not hold
        release = copy.deepcopy(self.f.release)
        release['files']['docs/extra.md'] = dict(sha256='1' * 64, bytes=1, mode=0o644)
        bundle['release.lock.json'] = encoded(release)
        self.f.tar(self.stripped, bundle)
        with self.assertRaisesRegex(ValueError, 'release inventory mismatch'):
            self.seal()

    def test_files_in_different_directories_or_one_flag_alone_are_refused(self):
        other = Path(self.tmp.name) / 'other'
        other.mkdir()
        (other / 'x').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'share one directory'):
            a.engine_records({'bin/rexl3-serve': self.binary, 'bin/rexl3-cluster': other / 'x'}, self.tag)
        with self.assertRaisesRegex(ValueError, 'required together'):
            self.seal(engine_release_tag=None)
        with self.assertRaisesRegex(ValueError, 'required together'):
            self.seal(engine_binary_files=None)

    # ---- the sealer command

    def test_cli_seals_by_reference_and_names_the_engine_release(self):
        out = Path(self.tmp.name) / 'durable'
        command = [sys.executable, str(Path(a.__file__).resolve()), '--runtime-id', 'synthetic-demo',
            '--target-commit', self.f.commit, '--tree', self.f.tree, '--repo', str(self.f.repo), '--version', '1',
            '--template-dir', str(self.f.template), '--archive', str(self.stripped), '--source-asset', str(self.f.source_asset),
            '--engine-source-asset', str(self.f.engine_source_asset), '--receipt', str(self.f.receipt_path),
            '--evidence-root', str(self.f.assets), '--approved-by', 'synthetic', '--out-dir', str(out), '--no-publish',
            '--engine-binary', 'bin/rexl3-serve=' + str(self.binary), '--engine-release-tag', self.tag]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('engine release ' + self.tag, run.stdout)
        doc = json.loads((out / 'approval.json').read_bytes())
        self.assertEqual(doc['engine_binaries']['bin/rexl3-serve']['asset'], self.asset)
        self.assertFalse((out / self.asset).exists())  # not copied into the release's own assets
        self.assertEqual(subprocess.run(command, capture_output=True, text=True).returncode, 0)  # exact retry
        for drop in (['--engine-release-tag', self.tag], ['--engine-binary', 'bin/rexl3-serve=' + str(self.binary)]):
            partial = list(command)
            index = partial.index(drop[0])
            del partial[index:index + 2]
            refused = subprocess.run(partial, capture_output=True, text=True)
            self.assertEqual(refused.returncode, 2, refused.stdout)
            self.assertIn('required together', refused.stderr)
        for bad in ('bin/rexl3-serve', '=x', 'bin/rexl3-serve='):
            partial = list(command)
            partial[partial.index('--engine-binary') + 1] = bad
            self.assertEqual(subprocess.run(partial, capture_output=True, text=True).returncode, 2)


if __name__ == '__main__':
    unittest.main()
