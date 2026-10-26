"""CPU synthetic producer -> seal -> consumer -> promotion contracts.

Real disposable Git objects and protected exporter; ELF-shaped fixture bytes are
NEVER executed. Mock transports are not publication or hardware evidence.
"""
import copy
import gzip
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools import approve_profile as a
from tools import build_corresponding_source as exporter
from tools import corresponding_source_identity as source_identity
from tools import promote_runtime as p
from tools import release_evidence as e
from tools import verify_runtime_tree as v
from tools.test_approve_profile import SyntheticSeal


def members(path):
    with tarfile.open(path, 'r:*') as tar:
        return {m.name: (m.mode, tar.extractfile(m).read()) for m in tar}


def repack(path, files):
    with tarfile.open(path, 'w:gz', format=tarfile.USTAR_FORMAT) as tar:
        for name, (mode, data) in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.mode, info.size = mode, len(data)
            tar.addfile(info, io.BytesIO(data))


def rebind_manifest(files):
    """Attacker can update every internal digest, but cannot create Git objects."""
    manifest = e.loads(files['SOURCE_IDENTITY.json'][1])
    for name, item in list(manifest['files'].items()):
        if name not in files:
            del manifest['files'][name]
            continue
        mode, data = files[name]
        item.update(sha256=exporter.digest(data), bytes=len(data), archive_mode=format(mode, '04o'))
        if item['origin'] == 'git_blob':
            item.update(git_blob=exporter.git_digest('blob', data), git_mode='100755' if mode == 0o755 else '100644')
    files['SOURCE_IDENTITY.json'] = (0o644, exporter.canonical(manifest))


class CorrespondingSourceSealTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='engine-source-seal-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.f = SyntheticSeal(self.root)
        self.original = members(self.f.engine_source_asset)

    def verify(self, path, registry):
        return e.verify_assets(path, registry=registry, approved_templates=self.f.approved)

    def test_missing_argument_and_wrapper_only_cannot_seal(self):
        args = dict(self.f.kwargs)
        args.pop('engine_source_asset')
        with self.assertRaises(TypeError):
            a.build_approval(**args)
        for asset in (self.f.source_asset, self.f.archive):
            with self.subTest(asset=asset.name), self.assertRaises(ValueError):
                self.f.seal(engine_source_asset=asset)

    def test_consumer_requires_engine_asset_even_if_old_approval_is_trusted(self):
        path, doc, registry = self.f.published_fixture()
        del doc['engine_source_asset']
        path.write_bytes(e.json_bytes(doc))
        registry['approvals']['synthetic-demo/v1']['approval_sha256'] = e.sha256_file(path)
        with self.assertRaises(ValueError):
            self.verify(path, registry)
        with self.assertRaises(ValueError):
            p.approval_assets(doc)

    def test_every_original_file_matches_git_objects_and_modes(self):
        doc = self.f.seal()
        identity = e.loads(self.original['SOURCE_IDENTITY.json'][1])
        self.assertEqual((identity['source_commit'], identity['source_tree']), (self.f.commit, self.f.tree))
        for name, item in identity['files'].items():
            if item['origin'] == 'git_blob':
                self.assertEqual(e.git_file(self.f.repo, self.f.commit, name), self.original[name][1])
                self.assertEqual(item['git_blob'], exporter.git_digest('blob', self.original[name][1]))
        self.assertEqual(doc['engine_source_asset'], e.asset_record(self.f.engine_source_asset))
        self.assertNotEqual(doc['engine_source_asset'], doc['source_asset'])

    def test_recompressed_equal_members_seal_without_equal_gzip(self):
        original = self.f.engine_source_asset.read_bytes()
        self.f.engine_source_asset.write_bytes(gzip.compress(gzip.decompress(original), compresslevel=1, mtime=123))
        self.assertNotEqual(original, self.f.engine_source_asset.read_bytes())
        self.assertEqual(self.original, members(self.f.engine_source_asset))
        doc = self.f.seal()
        self.assertEqual(doc['engine_source_asset']['sha256'], e.sha256_file(self.f.engine_source_asset))

    def test_hidden_tail_and_concatenated_archives_are_not_source_members(self):
        original = self.f.engine_source_asset.read_bytes()
        raw = gzip.decompress(original)
        for data in (gzip.compress(raw + b'SYNTHETIC UNLISTED PAYLOAD'),
                     original + gzip.compress(raw)):
            self.f.engine_source_asset.write_bytes(data)
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                self.f.seal()

    def test_rehashed_original_bytes_and_mode_forgery_cannot_seal(self):
        target = 'crates/server/src/main.rs'
        for mode, content in ((0o644, b'fn forged() {}\n'), (0o755, self.original[target][1])):
            files = copy.deepcopy(self.original)
            files[target] = (mode, content)
            rebind_manifest(files)
            repack(self.f.engine_source_asset, files)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.f.seal()

    def test_missing_build_license_vendor_and_include_inputs_cannot_seal(self):
        paths = ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'LICENSE', 'NOTICE',
                 'THIRD_PARTY_NOTICES.md', 'crates/server/Cargo.toml', 'crates/cuda/csrc/kernel.cu',
                 'third_party/toks/tokenizer.c', 'docs/examples/compile.json']
        for name in paths:
            files = copy.deepcopy(self.original)
            del files[name]
            rebind_manifest(files)
            repack(self.f.engine_source_asset, files)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.f.seal()

    def test_rehashed_identity_projection_and_generated_instruction_forgery(self):
        for field in ('source_commit', 'source_tree', 'projection', 'git_blob', 'generated', 'template'):
            files = copy.deepcopy(self.original)
            manifest = e.loads(files['SOURCE_IDENTITY.json'][1])
            if field in ('source_commit', 'source_tree'):
                manifest[field] = 'f' * 40
            elif field == 'projection':
                manifest['projection']['excluded_entries'] += 1
            elif field == 'git_blob':
                manifest['files']['Cargo.lock']['git_blob'] = 'a' * 40
            else:
                target = 'SOURCE_BUILD.md' if field == 'generated' else exporter.TEMPLATE
                files[target] = (0o644, b'forged instructions\n')
            files['SOURCE_IDENTITY.json'] = (0o644, exporter.canonical(manifest))
            if field in ('generated', 'template'):
                rebind_manifest(files)
            repack(self.f.engine_source_asset, files)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.f.seal()

    def test_unlisted_and_self_consistently_listed_extra_members(self):
        for listed in (False, True):
            files = copy.deepcopy(self.original)
            data = b'NOT A COMMITTED INPUT\n'
            files['crates/server/extra.rs'] = (0o644, data)
            if listed:
                manifest = e.loads(files['SOURCE_IDENTITY.json'][1])
                manifest['files']['crates/server/extra.rs'] = dict(bytes=len(data), sha256=exporter.digest(data),
                    archive_mode='0644', git_blob=exporter.git_digest('blob', data), git_mode='100644', origin='git_blob')
                files['SOURCE_IDENTITY.json'] = (0o644, exporter.canonical(manifest))
            repack(self.f.engine_source_asset, files)
            with self.subTest(listed=listed), self.assertRaises(ValueError):
                self.f.seal()

    def test_git_privacy_block_cannot_be_waived_by_a_clean_caller_manifest(self):
        target = self.f.repo / 'crates/server/src/main.rs'
        target.write_bytes(b'// SYNTHETIC scanner canary: ' + b'hf_' + b'A' * 32 + b'\n')
        self.f.git('add', '.')
        self.f.git('commit', '-qm', 'synthetic blocked source')
        commit, tree = self.f.git('rev-parse', 'HEAD'), self.f.git('rev-parse', 'HEAD^{tree}')
        self.assertTrue(exporter.inspect_source(self.f.repo, commit)['findings'])
        with self.assertRaisesRegex(ValueError, 'source_not_exportable'):
            source_identity.verify_git_projection(self.f.engine_source_asset, repo=self.f.repo, commit=commit, tree=tree)

    def test_bound_archive_validation_counts_reviewed_external_test_targets(self):
        # The consumer re-validation rebuilds its file list from the archive members; a reviewed external test target is
        # excluded from the archive and must still satisfy the exporter's review check.
        repo, target = self.f.repo, 'testdata/external.safetensors'
        marker = b'SYNTHETIC-PROVENANCE-NOT-A-TENSOR'
        (repo / target).write_bytes(marker)
        reference = '../../../' + target
        directive = b'const B: &[u8] = include_bytes!("' + reference.encode() + b'");\n'
        body = b'fn main() {}\n#[cfg(test)]\nmod tests {\n' + directive + b'}\n'
        (repo / 'crates/server/src/main.rs').write_bytes(body)
        offset = body.index(b'include_bytes!')
        review = {'source': 'crates/server/src/main.rs', 'source_sha256': exporter.digest(body),
                  'line': body.count(b'\n', 0, offset) + 1, 'column': offset - body.rfind(b'\n', 0, offset),
                  'reference': reference, 'target': target, 'target_git_blob': exporter.git_digest('blob', marker),
                  'target_bytes': len(marker), 'scope': {'kind': 'cfg_test_module', 'attribute_line': 2, 'module_line': 3},
                  'reason': 'Reviewed synthetic external test dependency; metadata only, never shipped.'}
        policy = e.loads((repo / exporter.INPUTS).read_bytes())
        policy['external_test_dependencies'] = [review]
        (repo / exporter.INPUTS).write_bytes(exporter.canonical(policy))
        self.f.git('add', '.')
        self.f.git('commit', '-qm', 'synthetic reviewed external test target')
        commit, tree = self.f.git('rev-parse', 'HEAD'), self.f.git('rev-parse', 'HEAD^{tree}')
        source = exporter.inspect_source(repo, commit)
        self.assertEqual(source['blockers'], [])
        self.f.engine_source_asset.write_bytes(exporter.archive_bytes(source))
        self.assertNotIn(target, members(self.f.engine_source_asset))
        asset = e.asset_record(self.f.engine_source_asset, e.inspect_archive(self.f.engine_source_asset))
        identity = source_identity.validate_bound_archive(self.f.engine_source_asset, commit=commit, tree=tree, asset=asset)
        self.assertEqual(identity['projection']['external_test_dependencies'], [review])

    def test_committed_instruction_forgery_is_not_a_trusted_template(self):
        (self.f.repo / exporter.TEMPLATE).write_bytes(b'SYNTHETIC untrusted build instructions\n')
        self.f.git('add', '.')
        self.f.git('commit', '-qm', 'synthetic untrusted template')
        commit, tree = self.f.git('rev-parse', 'HEAD'), self.f.git('rev-parse', 'HEAD^{tree}')
        self.f.engine_source_asset.write_bytes(exporter.archive_bytes(exporter.inspect_source(self.f.repo, commit)))
        with self.assertRaisesRegex(ValueError, 'protected template'):
            source_identity.verify_git_projection(self.f.engine_source_asset, repo=self.f.repo, commit=commit, tree=tree)

    def test_unrelated_real_engine_commit_does_not_match_native_release(self):
        (self.f.repo / 'crates/server/src/main.rs').write_bytes(b'fn changed() {}\n')
        self.f.git('add', '.')
        self.f.git('commit', '-qm', 'synthetic different native source')
        commit = self.f.git('rev-parse', 'HEAD')
        self.f.engine_source_asset.write_bytes(exporter.archive_bytes(exporter.inspect_source(self.f.repo, commit)))
        with self.assertRaises(ValueError):
            self.f.seal()

    def test_asset_names_are_distinct_and_receipts_cannot_alias_engine_transport(self):
        doc = self.f.seal()
        for role in ('bundle', 'source_asset', 'evidence_asset'):
            changed = copy.deepcopy(doc)
            changed['engine_source_asset']['archive'] = doc[role]['archive']
            with self.subTest(role=role), self.assertRaises(ValueError):
                e.approval_assets(changed)
        changed = copy.deepcopy(doc)
        changed['engine_source_asset']['files'] += 1
        with self.assertRaises(ValueError):
            e.approval_assets(changed)
        receipt = copy.deepcopy(self.f.receipt)
        receipt['gates']['doctor']['evidence'].append({'path': self.f.engine_source_asset.name,
                                                    'sha256': e.sha256_file(self.f.engine_source_asset)})
        self.f.receipt_path.write_bytes(e.json_bytes(receipt))
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.f.seal()

    def test_unsafe_engine_archive_fails_before_seal_output(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE):
            with tarfile.open(self.f.engine_source_asset, 'w:gz') as tar:
                member = tarfile.TarInfo('Cargo.toml')
                member.type, member.linkname, member.mode = kind, '../outside', 0o644
                tar.addfile(member)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.f.seal()
        for name in ('../outside', '/outside', './Cargo.toml'):
            repack(self.f.engine_source_asset, {name: (0o644, b'invalid')})
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.f.seal()

    def test_consumer_does_not_need_private_git_but_requires_independent_root(self):
        path, doc, registry = self.f.published_fixture()
        shutil.rmtree(self.f.repo)
        with patch.object(exporter, 'git', side_effect=AssertionError('consumer used Git')):
            self.assertEqual(self.verify(path, registry)[0], doc)
        files = copy.deepcopy(self.original)
        files['crates/server/src/main.rs'] = (0o644, b'fn counterfeit() {}\n')
        rebind_manifest(files)
        repack(self.f.engine_source_asset, files)
        doc['engine_source_asset'] = e.asset_record(self.f.engine_source_asset)
        path.write_bytes(e.json_bytes(doc))
        with self.assertRaisesRegex(ValueError, 'untrusted approval bytes'):
            self.verify(path, registry)

    def test_consumer_revalidates_internal_blob_and_identity_after_outer_hashes(self):
        path, original_doc, registry = self.f.published_fixture()
        for field in ('git_blob', 'source_tree', 'source_commit', 'generated', 'inventory'):
            files = copy.deepcopy(self.original)
            manifest = e.loads(files['SOURCE_IDENTITY.json'][1])
            if field == 'git_blob':
                manifest['files']['Cargo.lock']['git_blob'] = 'a' * 40
            elif field in ('source_tree', 'source_commit'):
                manifest[field] = 'f' * 40
            elif field == 'inventory':
                del manifest['files']['Cargo.lock']
            else:
                files['SOURCE_BUILD.md'] = (0o644, b'forged instructions\n')
            files['SOURCE_IDENTITY.json'] = (0o644, exporter.canonical(manifest))
            if field == 'generated':
                rebind_manifest(files)
            repack(self.f.engine_source_asset, files)
            doc = copy.deepcopy(original_doc)
            doc['engine_source_asset'] = e.asset_record(self.f.engine_source_asset)
            path.write_bytes(e.json_bytes(doc))
            # Simulate an independently pinned but malformed seal to exercise
            # semantic validation, not merely the unchanged approval hash guard.
            registry['approvals']['synthetic-demo/v1']['approval_sha256'] = e.sha256_file(path)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(path, registry)

    def test_exact_seal_retry_and_collisions_preserve_engine_bytes(self):
        doc = self.f.seal()
        out = self.root / 'durable'
        args = dict(archive=self.f.archive, source_asset=self.f.source_asset,
                    engine_source_asset=self.f.engine_source_asset, evidence_root=self.f.assets, out_dir=out)
        path = a.seal_to_directory(doc, **args)
        before = {f.relative_to(out).as_posix(): f.read_bytes() for f in out.rglob('*') if f.is_file()}
        a.seal_to_directory(self.f.seal(), **args)
        self.assertEqual(before, {f.relative_to(out).as_posix(): f.read_bytes() for f in out.rglob('*') if f.is_file()})
        self.f.engine_source_asset.write_bytes(b'tampered after validation')
        with self.assertRaises(ValueError):
            a.seal_to_directory(doc, **(args | {'out_dir': self.root / 'other'}))
        self.assertFalse((self.root / 'other').exists())
        self.assertEqual(path.read_bytes(), before['approval.json'])

    def test_full_producer_seal_consumer_and_promotion_flow(self):
        doc = self.f.seal()
        durable = self.root / 'durable'
        path = a.seal_to_directory(doc, archive=self.f.archive, source_asset=self.f.source_asset,
            engine_source_asset=self.f.engine_source_asset, evidence_root=self.f.assets, out_dir=durable)
        registry = {'schema': 'rexl3.approval-registry.v1', 'approvals': {'synthetic-demo/v1': {
            'approval_sha256': e.sha256_file(path), 'source_commit': self.f.commit, 'source_tree': self.f.tree}}}
        self.verify(path, registry)
        prod = self.root / 'production'
        prod.mkdir()
        for args in (('init', '-q'), ('-c', 'user.name=synthetic', '-c', 'user.email=synthetic@example.invalid',
                    'commit', '--allow-empty', '-qm', 'synthetic production base')):
            subprocess.run(['git', '-C', str(prod), *args], check=True, capture_output=True)
        base = e.git(prod, 'rev-parse', 'HEAD').decode().strip()
        e.git(prod, 'update-ref', 'refs/remotes/origin/main', base)
        with patch.object(p, 'gh', side_effect=AssertionError('no publication/network')):
            self.assertEqual(p.prepare_promotion('synthetic-demo', 'approve-synthetic-demo-1', 1,
                target_commit=base, prod_checkout=prod, assets_dir=durable,
                registry=registry, approved_templates=self.f.approved, dry_run=True), 0)
        tree = prod / 'runtimes/synthetic-demo/v1'
        p.extract_tree(durable / doc['bundle']['archive'], tree, doc, approval_path=path)
        self.assertEqual(v.verify_tree(tree, self.f.approved, trusted_approvals=registry, assets_dir=durable), [])
        self.assertEqual({asset['archive'] for asset in p.approval_assets(doc)},
            {doc[k]['archive'] for k in ('bundle', 'source_asset', 'engine_source_asset', 'evidence_asset')})
        self.assertEqual(v.tree_inventory(tree).keys(), doc['source_asset']['members'].keys() | {'approval.json'})
        (durable / doc['engine_source_asset']['archive']).unlink()
        self.assertTrue(v.verify_tree(tree, self.f.approved, trusted_approvals=registry, assets_dir=durable))


if __name__ == '__main__':
    unittest.main()
