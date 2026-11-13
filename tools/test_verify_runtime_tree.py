"""Offline production verifier contracts for the family/model layout; fixtures are synthetic, never qualification."""
import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
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

    def test_tree_under_the_renamed_runtime_id_passes_with_the_reviewed_alias(self):
        # MODEL.json names the runtime's current id; the v1 tree was sealed under its reviewed earlier id.
        doc = self.f.model_doc(FAMILY, MODEL, runtime_id='synthetic-renamed')
        write(self.tree.parent, 'MODEL.json', e.json_bytes(doc))
        self.assertTrue(self.verify())
        with patch.dict(e.RUNTIME_ID_ALIASES, {('synthetic-renamed', 'single'): 'synthetic-demo'}):
            self.assertEqual(self.verify(), [])

    def test_renamed_runtime_alias_is_exact(self):
        doc = self.f.model_doc(FAMILY, MODEL, runtime_id='synthetic-renamed')
        write(self.tree.parent, 'MODEL.json', e.json_bytes(doc))
        for aliases in ({('synthetic-renamed', 'other-profile'): 'synthetic-demo'},   # another profile choice
                        {('synthetic-renamed', 'single'): 'synthetic-other'},         # another earlier id
                        {('synthetic-other', 'single'): 'synthetic-demo'}):           # another current id
            with self.subTest(aliases=aliases), patch.dict(e.RUNTIME_ID_ALIASES, aliases):
                self.assertTrue(self.verify())

    def test_renamed_runtime_alias_is_one_way(self):
        # A MODEL.json naming the earlier id does not admit a tree sealed under the current id.
        doc = self.f.model_doc(FAMILY, MODEL, runtime_id='synthetic-earlier')
        write(self.tree.parent, 'MODEL.json', e.json_bytes(doc))
        with patch.dict(e.RUNTIME_ID_ALIASES, {('synthetic-demo', 'single'): 'synthetic-earlier'}):
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

    def precomputed(self, **exclude):
        skip = {exclude.get('skip', '')}
        return {asset['archive']: asset['members'] for asset in e.approval_assets(self.doc)
                if asset['archive'] not in skip}

    def test_verify_tree_precomputed_skips_decompression(self):
        with patch('tools.release_evidence.inspect_archive',
                   side_effect=AssertionError('must not run')):
            self.assertEqual(self.verify(precomputed=self.precomputed()), [])

    def test_verify_tree_partial_precomputed_falls_back_to_full_inspection(self):
        precomputed = self.precomputed(skip=self.doc['bundle']['archive'])
        self.assertEqual(self.verify(precomputed=precomputed), [])

    def test_verify_tree_tampered_archive_fails_with_and_without_precomputed(self):
        name = self.doc['bundle']['archive']
        path = self.assets / FAMILY / MODEL / 'v1' / name
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
        self.assertTrue(self.verify())
        self.assertTrue(self.verify(precomputed=self.precomputed()))
    def test_unified_tag_and_prefixed_asset_names_are_accepted(self):
        tag = 'engine-v0.1.2-' + self.f.commit[:8]
        e.validate_request('synthetic-demo', tag, 1)
        unified = copy.deepcopy(self.doc)
        unified['approve_tag'] = tag
        prefix = 'synthetic-demo-v1'
        for field, kind in (('bundle', 'bundle'), ('source_asset', 'source'),
                            ('evidence_asset', 'evidence')):
            unified[field]['archive'] = f'{prefix}-{kind}.tar.gz'
        unified['engine_source_asset']['archive'] = e.shared_engine_source_name(tag)
        self.assertEqual(len(e.approval_assets(unified)), 4)
        unified['evidence_asset']['archive'] = 'evidence.tar.gz'
        with self.assertRaisesRegex(ValueError, 'unified release asset name mismatch'):
            e.approval_assets(unified)

    def test_unchanged_tree_skips_assets_and_uses_base_anchor(self):
        candidate = Path(self.tmp.name) / 'prod'
        base = Path(self.tmp.name) / 'base'
        shutil.copytree(candidate, base)
        for root in (candidate, base):
            write(root, 'runtimes/templates/APPROVALS.json', encoded(self.registry))
            write(root, 'runtimes/templates/APPROVED.json', encoded({'templates': self.f.approved}))
        code = v.main(['--root', str(candidate), '--base', str(base),
                       '--approved', str(base / 'runtimes/templates/APPROVED.json'),
                       '--trusted-approvals', str(base / 'runtimes/templates/APPROVALS.json'),
                       '--assets-dir', str(Path(self.tmp.name) / 'no-downloads')])
        self.assertEqual(code, 0)

    def diff_fixture(self, name):
        original = Path(self.tmp.name) / 'prod'
        candidate = Path(self.tmp.name) / name / 'prod'
        base = Path(self.tmp.name) / name / 'base'
        candidate.parent.mkdir(parents=True)
        shutil.copytree(original, candidate)
        shutil.copytree(original, base)
        for root in (candidate, base):
            write(root, 'runtimes/templates/APPROVALS.json', encoded(self.registry))
            write(root, 'runtimes/templates/APPROVED.json', encoded({'templates': self.f.approved}))
        return candidate, base

    def run_diff(self, candidate, base, assets=None):
        return v.main(['--root', str(candidate), '--base', str(base),
                       '--approved', str(base / 'runtimes/templates/APPROVED.json'),
                       '--trusted-approvals', str(base / 'runtimes/templates/APPROVALS.json'),
                       '--assets-dir', str(assets or Path(self.tmp.name) / 'no-downloads')])

    def test_unchanged_tree_still_refuses_registry_and_local_metadata_tampering(self):
        key = f'{FAMILY}/{MODEL}/v1'
        mutations = ('delete-registry-anchor', 'alter-registry-anchor', 'malformed-MODEL',
                     'wrong-MODEL-runtime', 'delete-MODEL', 'alter-LATEST', 'delete-candidate-APPROVED')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                candidate, base = self.diff_fixture(mutation)
                registry_path = candidate / 'runtimes/templates/APPROVALS.json'
                if mutation == 'delete-registry-anchor':
                    doc = copy.deepcopy(self.registry)
                    del doc['approvals'][key]
                    write(registry_path.parent, registry_path.name, encoded(doc))
                elif mutation == 'alter-registry-anchor':
                    doc = copy.deepcopy(self.registry)
                    doc['approvals'][key]['runtime_id'] = 'attacker-controlled'
                    write(registry_path.parent, registry_path.name, encoded(doc))
                elif mutation == 'malformed-MODEL':
                    write(candidate / f'runtimes/{FAMILY}/{MODEL}', 'MODEL.json', b'{ definitely not json')
                elif mutation == 'wrong-MODEL-runtime':
                    doc = self.f.model_doc(FAMILY, MODEL, runtime_id='attacker-controlled')
                    write(candidate / f'runtimes/{FAMILY}/{MODEL}', 'MODEL.json', encoded(doc))
                elif mutation == 'delete-MODEL':
                    (candidate / f'runtimes/{FAMILY}/{MODEL}/MODEL.json').unlink()
                elif mutation == 'alter-LATEST':
                    write(candidate / f'runtimes/{FAMILY}/{MODEL}', 'LATEST', b'v999\n')
                else:
                    (candidate / 'runtimes/templates/APPROVED.json').unlink()
                self.assertNotEqual(self.run_diff(candidate, base), 0)

    def test_orphan_candidate_anchor_is_refused_before_the_skip(self):
        candidate, base = self.diff_fixture('orphan')
        doc = copy.deepcopy(self.registry)
        doc['approvals']['attacker/orphan/v999'] = {
            'runtime_id': 'attacker', 'approval_sha256': 'a' * 64,
            'source_commit': 'b' * 40, 'source_tree': 'c' * 40,
            'binary_sha256': 'd' * 64,
            'profiles': {'single': {'receipt': {'path': 'x', 'sha256': 'e' * 64}}}}
        path = candidate / 'runtimes/templates/APPROVALS.json'
        write(path.parent, path.name, encoded(doc))
        self.assertNotEqual(self.run_diff(candidate, base), 0)

    def test_new_anchor_for_an_existing_tree_is_fully_verified_not_skipped(self):
        candidate, base = self.diff_fixture('new-anchor')
        empty = {'schema': 'rexl3.approval-registry.v1', 'approvals': {}, 'legacy': {}}
        path = base / 'runtimes/templates/APPROVALS.json'
        write(path.parent, path.name, encoded(empty))
        self.assertNotEqual(self.run_diff(candidate, base), 0)
        self.assertEqual(self.run_diff(candidate, base, self.assets), 0)


class ModularRegistryTest(unittest.TestCase):
    def entry(self, runtime):
        return {'runtime_id': runtime, 'approval_sha256': 'a' * 64,
                'source_commit': 'b' * 40, 'source_tree': 'c' * 40}

    def test_modules_merge_without_a_pin_and_duplicates_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main = write(root, 'APPROVALS.json', encoded({
                'schema': 'rexl3.approval-registry.v1',
                'approvals': {'qwen3.8/one/v1': self.entry('one')}}))
            write(root, 'approvals/qwen3.8/two.json', encoded({
                'schema': 'rexl3.approval-registry.v1',
                'approvals': {'qwen3.8/two/v2': self.entry('two')}}))
            merged = e.load_registry(main)
            self.assertEqual(set(merged['approvals']), {'qwen3.8/one/v1', 'qwen3.8/two/v2'})
            write(root, 'approvals/qwen3.8/duplicate.json', encoded({
                'schema': 'rexl3.approval-registry.v1',
                'approvals': {'qwen3.8/one/v1': self.entry('one')}}))
            with self.assertRaisesRegex(ValueError, 'duplicate approval registry key'):
                e.load_registry(main)

    def test_legacy_anchors_are_immutable_and_new_legacy_anchors_are_refused(self):
        base = {'schema': 'rexl3.approval-registry.v1', 'approvals': {},
                'legacy': {'old-runtime/v1': self.entry('old-runtime')}}
        for label, candidate in (
                ('deleted', {'schema': base['schema'], 'approvals': {}, 'legacy': {}}),
                ('altered', copy.deepcopy(base)),
                ('new', copy.deepcopy(base))):
            if label == 'altered':
                candidate['legacy']['old-runtime/v1']['approval_sha256'] = 'b' * 64
            elif label == 'new':
                candidate['legacy']['other/v2'] = self.entry('other')
            with self.subTest(label=label), self.assertRaises(ValueError):
                e.validate_registry_transition(base, candidate, set())


if __name__ == '__main__':
    unittest.main()
