"""Model layout contract tests: naming regexes, MODEL.json schema and the capability contract."""
import tempfile
import unittest
from pathlib import Path

from tools import model_layout as m
from tools import release_evidence as e
from tools.test_release_evidence import encoded
from tools.test_approve_profile import SyntheticSeal


class ModelNameTest(unittest.TestCase):
    def test_valid_names(self):
        for name in ('qwen3.8', 'qwen3.8-27b', 'deepseek-v4.1-flash-1017', 'glm-5.3',
                     'nemotron-3.5', 'a', 'a1.b2-c3'):
            m.model_name(name)
            m.model_name(name, 'family')

    def test_invalid_names(self):
        for name in ('Qwen3.8', 'qwen3.8_27b', 'qwen 3', '-bad', 'bad-', 'a/b', 'a..b',
                     'a--b', 'a.', 'a-', 'a_', '.a', 'a.b.'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.model_name(name)

    def test_registry_key(self):
        self.assertEqual(m.registry_key('qwen3.8', 'qwen3.8-27b', 1), 'qwen3.8/qwen3.8-27b/v1')
        with self.assertRaises(ValueError):
            m.registry_key('Qwen', 'qwen', 1)
        with self.assertRaises(ValueError):
            m.registry_key('qwen', 'qwen', 0)


class ModelDocTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        self.doc = self.f.model_doc('qwen3.8', 'qwen3.8-27b')

    def test_valid_doc_passes(self):
        m.validate_model_doc(self.doc, 'qwen3.8', 'qwen3.8-27b')

    def test_folder_agreement(self):
        with self.assertRaisesRegex(ValueError, 'family differs'):
            m.validate_model_doc(self.doc, 'other', 'qwen3.8-27b')
        with self.assertRaisesRegex(ValueError, 'model differs'):
            m.validate_model_doc(self.doc, 'qwen3.8', 'other')

    def test_required_keys(self):
        for key in ('schema', 'family', 'model', 'display_name', 'architectures', 'engine_family',
                    'runtime_id', 'pack'):
            import copy
            doc = copy.deepcopy(self.doc)
            del doc[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_model_doc(doc, 'qwen3.8', 'qwen3.8-27b')

    def test_schema(self):
        self.doc['schema'] = 'other'
        with self.assertRaises(ValueError):
            m.validate_model_doc(self.doc, 'qwen3.8', 'qwen3.8-27b')

    def test_pack_format_required(self):
        self.doc['pack'] = {'bits': '4.00'}
        with self.assertRaises(ValueError):
            m.validate_model_doc(self.doc, 'qwen3.8', 'qwen3.8-27b')

    def test_architectures_must_be_nonempty_list(self):
        for value in ([], ['A', 1], 'A'):
            self.doc['architectures'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.validate_model_doc(self.doc, 'qwen3.8', 'qwen3.8-27b')

    def test_runtime_id_matches_dev_identity(self):
        self.doc['runtime_id'] = 'Bad Id'
        with self.assertRaises(ValueError):
            m.validate_model_doc(self.doc, 'qwen3.8', 'qwen3.8-27b')

    def test_write_or_validate_exclusive(self):
        root = Path(self.tmp.name) / 'write'
        path = m.write_or_validate_model_json(root / 'MODEL.json', family='qwen3.8', model='qwen3.8-27b',
                                              doc=self.doc)
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes(), e.json_bytes(self.doc))
        # Same derived metadata is accepted; any drift is refused; never overwritten.
        m.write_or_validate_model_json(path, family='qwen3.8', model='qwen3.8-27b', doc=self.doc)
        changed = dict(self.doc, display_name='Changed')
        with self.assertRaises(ValueError):
            m.write_or_validate_model_json(path, family='qwen3.8', model='qwen3.8-27b', doc=changed)
        self.assertEqual(path.read_bytes(), e.json_bytes(self.doc))


class RenamedRuntimeTest(unittest.TestCase):
    """A folder whose runtime was renamed: v1 sealed under the reviewed earlier id, later versions under the
    current id, MODEL.json naming the current id (release_evidence.RUNTIME_ID_ALIASES, exact pairs only)."""
    OLD, NEW, TP2 = 'glm53-flash-2.25bpw-sm121', 'glm53-flash-sage-2.25', 'glm53-flash-tp2-sage-2.25'
    OLD_REPO, NEW_REPO = 'r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121', 'r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / 'runtimes' / 'glm-5.3' / 'glm-5.3-flash'
        self.v1 = dict(runtime_id=self.OLD, profiles={'glm53-flash-single': {}, 'glm53-flash-tp2-1m': {}})
        self.write_tree(1, self.v1)

    def write_tree(self, version, approval):
        tree = self.folder / f'v{version}'
        tree.mkdir(parents=True, exist_ok=True)
        (tree / 'approval.json').write_bytes(e.json_bytes(approval))

    def doc(self, runtime_id, repo, **changes):
        doc = m.model_json_doc(family='glm-5.3', model='glm-5.3-flash', display_name='GLM-5.3-Flash',
                               architectures=['Glm5NextForConditionalGeneration'], engine_family='glm',
                               runtime_id=runtime_id, pack={'format': 'exl3', 'bits': '2.25', 'target_repo': repo,
                                                            'target_revision': 'b' * 40})
        return dict(doc, **changes)

    def write_model_json(self, doc):
        path = self.folder / 'MODEL.json'
        path.write_bytes(e.json_bytes(doc))
        return path

    def promote(self, doc):
        return m.write_or_validate_model_json(self.folder / 'MODEL.json', family='glm-5.3', model='glm-5.3-flash',
                                              doc=doc)

    def test_matches_the_current_id_and_its_reviewed_earlier_id(self):
        v2 = dict(runtime_id=self.NEW, profiles={'glm53-flash-single': {}})
        self.assertTrue(m.model_runtime_id_matches(self.NEW, v2))          # current id: exact
        self.assertTrue(m.model_runtime_id_matches(self.NEW, self.v1))     # earlier id via the single-node pair
        self.assertTrue(m.model_runtime_id_matches(self.OLD, self.v1))     # unchanged folders still match
        tp2_only = dict(runtime_id=self.OLD, profiles={'glm53-flash-tp2-1m': {}})
        self.assertTrue(m.model_runtime_id_matches(self.TP2, tp2_only))

    def test_refuses_every_other_pairing(self):
        v2 = dict(runtime_id=self.NEW, profiles={'glm53-flash-single': {}})
        self.assertFalse(m.model_runtime_id_matches(self.OLD, v2))          # the alias is one way
        tp2_only = dict(runtime_id=self.OLD, profiles={'glm53-flash-tp2-1m': {}})
        self.assertFalse(m.model_runtime_id_matches(self.NEW, tp2_only))   # pair is for another profile choice
        self.assertFalse(m.model_runtime_id_matches(self.NEW, dict(self.v1, runtime_id='glm53-flash-other')))
        self.assertFalse(m.model_runtime_id_matches('glm53-flash-sage', self.v1))   # no prefix or fuzzy match
        self.assertFalse(m.model_runtime_id_matches(self.NEW, dict(runtime_id=self.OLD)))
        self.assertFalse(m.model_runtime_id_matches(self.NEW, None))

    def test_promotion_moves_model_json_to_the_current_id(self):
        self.write_model_json(self.doc(self.OLD, self.OLD_REPO))
        new = self.doc(self.NEW, self.NEW_REPO)
        path = self.promote(new)
        self.assertEqual(path.read_bytes(), e.json_bytes(new))
        self.assertEqual(path.stat().st_mode & 0o777, 0o644)
        self.promote(new)                                               # idempotent once moved
        with self.assertRaises(ValueError):
            self.promote(self.doc(self.OLD, self.OLD_REPO))             # never moved back

    def test_promotion_keeps_the_pack_repo_when_it_was_not_renamed(self):
        self.write_model_json(self.doc(self.OLD, self.OLD_REPO))
        new = self.doc(self.NEW, self.OLD_REPO)
        self.assertEqual(self.promote(new).read_bytes(), e.json_bytes(new))

    def test_promotion_refuses_a_rename_that_changes_anything_else(self):
        old = self.doc(self.OLD, self.OLD_REPO)
        path = self.write_model_json(old)
        for doc in (self.doc(self.NEW, self.NEW_REPO, display_name='Changed'),
                    self.doc(self.NEW, 'r0b0tlab/elsewhere'),
                    dict(self.doc(self.NEW, self.NEW_REPO), pack=dict(self.doc(self.NEW, self.NEW_REPO)['pack'],
                                                                     bits='3.00')),
                    self.doc('glm53-flash-other', self.NEW_REPO)):
            with self.subTest(doc=doc):
                with self.assertRaisesRegex(ValueError, 'differs'):
                    self.promote(doc)
                self.assertEqual(path.read_bytes(), e.json_bytes(old))

    def test_promotion_refuses_when_a_sealed_tree_would_no_longer_match(self):
        old = self.doc(self.OLD, self.OLD_REPO)
        path = self.write_model_json(old)
        self.write_tree(2, dict(runtime_id='glm53-flash-other', profiles={'glm53-flash-single': {}}))
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.promote(self.doc(self.NEW, self.NEW_REPO))
        self.assertEqual(path.read_bytes(), e.json_bytes(old))

    def test_promotion_refuses_a_rename_whose_pair_is_for_another_profile_choice(self):
        # (glm53-flash-tp2-sage-2.25, glm53-flash-tp2-1m) is a reviewed pair, but a folder whose only tree seals
        # glm53-flash-single under the earlier id cannot move to the two-node id.
        old = self.doc(self.OLD, self.OLD_REPO)
        path = self.write_model_json(old)
        self.write_tree(1, dict(runtime_id=self.OLD, profiles={'glm53-flash-single': {}}))
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.promote(self.doc(self.TP2, self.NEW_REPO))
        self.assertEqual(path.read_bytes(), e.json_bytes(old))

    def test_promotion_refuses_a_rename_without_a_sealed_tree(self):
        old = self.doc(self.OLD, self.OLD_REPO)
        empty = Path(self.tmp.name) / 'runtimes' / 'glm-5.3' / 'empty'
        empty.mkdir(parents=True)
        path = empty / 'MODEL.json'
        path.write_bytes(e.json_bytes(dict(old, model='empty')))
        with self.assertRaisesRegex(ValueError, 'differs'):
            m.write_or_validate_model_json(path, family='glm-5.3', model='empty',
                                           doc=dict(self.doc(self.NEW, self.NEW_REPO), model='empty'))


class CapabilityContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = SyntheticSeal(self.tmp.name)
        _, self.doc, _ = self.f.published_fixture()
        self.model_doc = self.f.model_doc('qwen3.8', 'qwen3.8-27b')

    def test_synthetic_fixture_has_prefix_cache_and_vision(self):
        self.assertEqual(m.capability_contract(self.doc, self.model_doc), [])

    def test_prefix_cache_off_fails(self):
        self.doc['profiles']['single']['identity']['capabilities']['prefix_cache'] = False
        problems = m.capability_contract(self.doc, self.model_doc)
        self.assertTrue(any('prefix cache' in p for p in problems))

    def test_vision_off_requires_justification(self):
        self.doc['profiles']['single']['identity']['capabilities']['images'] = False
        problems = m.capability_contract(self.doc, self.model_doc)
        self.assertTrue(any('vision is off' in p for p in problems))
        justified = dict(self.model_doc, pack={'format': 'exl3', 'vision_does_not_fit': 'no vision-capable GPU fit'})
        self.assertEqual(m.capability_contract(self.doc, justified), [])
        # a bare boolean or empty reason is not a justification
        for bad in (True, '', '  '):
            with self.subTest(bad=bad):
                self.assertTrue(any('vision is off' in p for p in
                                    m.capability_contract(self.doc, dict(self.model_doc,
                                                                        pack={'format': 'exl3',
                                                                              'vision_does_not_fit': bad}))))


if __name__ == '__main__':
    unittest.main()
