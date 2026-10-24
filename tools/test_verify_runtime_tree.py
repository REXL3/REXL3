"""Offline production verifier contracts; fixtures are synthetic, never qualification."""
import copy
import tempfile
import unittest
from pathlib import Path
from tools import verify_runtime_tree as v
from tools import release_evidence as e
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import encoded, write, sha


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        self.approval, self.doc, self.registry = self.f.published_fixture()
        self.tree = Path(self.tmp.name) / 'prod/runtimes/synthetic-demo/v1'
        for name, data in self.f.source.items():
            p = write(self.tree, name, data)
            p.chmod(self.doc['source_asset']['members'][name]['mode'])
        write(self.tree, 'approval.json', self.approval.read_bytes())

    def verify(self, **kwargs):
        return v.verify_tree(self.tree, self.f.approved, trusted_approvals=self.registry,
                             assets_dir=self.f.assets, **kwargs)

    def test_clean_authenticated_tree_passes(self):
        self.assertEqual(self.verify(), [])

    def test_no_registry_is_not_authorized(self):
        self.assertTrue(v.verify_tree(self.tree, self.f.approved))

    def test_edited_manifest_and_rehashed_file_is_not_self_authorized(self):
        write(self.tree, 'launch.sh', b'forged')
        manifest = e.load_json(self.tree / 'source.manifest.json')
        manifest['files']['launch.sh']['sha256'] = sha(b'forged')
        write(self.tree, 'source.manifest.json', encoded(manifest))
        self.assertTrue(self.verify())

    def test_modes_and_extra_missing_symlink_files_rejected(self):
        p = self.tree / 'launch.sh'
        p.chmod(0o644)
        self.assertTrue(self.verify())
        p.chmod(0o755)
        extra = write(self.tree, 'extra.txt', b'extra')
        self.assertTrue(self.verify())
        extra.unlink()
        p.unlink()
        self.assertTrue(self.verify())
        p.symlink_to(self.f.template / 'launch.sh')
        self.assertTrue(self.verify())

    def test_tampered_receipt_file_fails(self):
        self.f.receipt_path.write_bytes(b'{}')
        self.assertTrue(self.verify())

    def test_runtime_path_version_identity(self):
        other = self.tree.with_name('v2')
        self.tree.rename(other)
        self.tree = other
        self.assertTrue(self.verify())

    def test_v1_history_is_explicit_and_exact_hash_anchored(self):
        legacy = copy.deepcopy(self.doc)
        legacy.update(schema='rexl3.profile-approval.v1', gates={'doctor': 'PASS'})
        legacy.pop('version')
        write(self.tree, 'approval.json', encoded(legacy))
        files = {p.relative_to(self.tree).as_posix(): e.record(p) for p in self.tree.rglob('*') if p.is_file()}
        self.registry['legacy'] = {'synthetic-demo/v1': {'approval_sha256': sha(encoded(legacy)),
            'source_commit': self.f.commit, 'source_tree': self.f.tree, 'files': files}}
        self.assertTrue(self.verify())
        self.assertEqual(self.verify(allow_legacy=True), [])
        write(self.tree, 'source.manifest.json', b'{}')
        self.assertTrue(self.verify(allow_legacy=True))


if __name__ == '__main__':
    unittest.main()
