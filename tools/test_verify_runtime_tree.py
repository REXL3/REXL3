"""Offline production verifier contracts for the family/model layout; fixtures are synthetic, never qualification."""
import copy
import tempfile
import unittest
from pathlib import Path
from tools import verify_runtime_tree as v
from tools import release_evidence as e
from tools import model_layout as m
from tools.test_approve_profile import SyntheticSeal
from tools.test_release_evidence import encoded, stage_assets, write, sha

FAMILY = 'qwen3.8'
MODEL = 'qwen3.8-27b'


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        self.approval, self.doc, self.registry = self.f.published_fixture_model(FAMILY, MODEL)
        self.tree = Path(self.tmp.name) / 'prod/runtimes' / FAMILY / MODEL / 'v1'
        for name, data in self.f.source.items():
            p = write(self.tree, name, data)
            p.chmod(self.doc['source_asset']['members'][name]['mode'])
        write(self.tree, 'approval.json', self.approval.read_bytes())
        write(self.tree.parent, 'MODEL.json', e.json_bytes(self.f.model_doc(FAMILY, MODEL)))
        write(self.tree.parent, 'LATEST', b'v1\n')
        stage_assets(Path(self.tmp.name) / 'assets' / FAMILY / MODEL / 'v1', self.f, self.doc)
        self.assets = Path(self.tmp.name) / 'assets'

    def verify(self, **kwargs):
        kwargs.setdefault('assets_dir', self.assets)
        return v.verify_tree(self.tree, kwargs.pop('approved', self.f.approved),
                             trusted_approvals=kwargs.pop('registry', self.registry),
                             family=kwargs.pop('family', FAMILY), model=kwargs.pop('model', MODEL), **kwargs)

    def test_clean_authenticated_tree_passes(self):
        self.assertEqual(self.verify(), [])

    def test_no_registry_is_not_authorized(self):
        problems = v.verify_tree(self.tree, self.f.approved, family=FAMILY, model=MODEL)
        self.assertTrue(problems)

    def test_registry_key_is_family_model_version(self):
        self.assertIn(f'{FAMILY}/{MODEL}/v1', self.registry['approvals'])
        self.assertEqual(self.registry['approvals'][f'{FAMILY}/{MODEL}/v1']['runtime_id'], 'synthetic-demo')

    def test_registry_runtime_id_mismatch_is_refused(self):
        self.registry['approvals'][f'{FAMILY}/{MODEL}/v1']['runtime_id'] = 'other-runtime'
        self.assertTrue(self.verify())

    def test_model_json_runtime_id_mismatch_is_refused(self):
        doc = self.f.model_doc(FAMILY, MODEL, runtime_id='other-runtime')
        write(self.tree.parent, 'MODEL.json', e.json_bytes(doc))
        self.assertTrue(self.verify())

    def test_model_json_folder_mismatch_is_refused(self):
        doc = self.f.model_doc(FAMILY, 'other-model')
        write(self.tree.parent, 'MODEL.json', e.json_bytes(doc))
        self.assertTrue(self.verify())

    def test_model_json_missing_is_refused(self):
        (self.tree.parent / 'MODEL.json').unlink()
        self.assertTrue(self.verify())

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
        stage = self.assets / FAMILY / MODEL / 'v1' / 'qualification.json'
        stage.write_bytes(b'{}')
        self.assertTrue(self.verify())

    def test_runtime_path_version_identity(self):
        other = self.tree.with_name('v2')
        self.tree.rename(other)
        self.tree = other
        self.assertTrue(self.verify())

    def test_family_model_folder_identity(self):
        self.assertTrue(v.verify_tree(self.tree, self.f.approved, trusted_approvals=self.registry,
                                      assets_dir=self.f.assets, family='other', model=MODEL))

    def test_flat_runtime_layout_is_rejected_by_discover(self):
        flat = Path(self.tmp.name) / 'flat/runtimes/synthetic-demo/v1'
        for name, data in self.f.source.items():
            write(flat, name, data)
        write(flat, 'approval.json', self.approval.read_bytes())
        with self.assertRaisesRegex(ValueError, 'old flat runtime layout rejected'):
            v.discover(Path(self.tmp.name) / 'flat')

    def test_legacy_v1_approval_is_refused(self):
        legacy = copy.deepcopy(self.doc)
        legacy.update(schema='rexl3.profile-approval.v1')
        write(self.tree, 'approval.json', encoded(legacy))
        problems = self.verify()
        self.assertTrue(any('v2 approval required' in p for p in problems))

    def test_discover_returns_family_model_triples(self):
        found = v.discover(Path(self.tmp.name) / 'prod')
        self.assertEqual([(f, mo, p) for f, mo, p in found], [(FAMILY, MODEL, self.tree)])


if __name__ == '__main__':
    unittest.main()
