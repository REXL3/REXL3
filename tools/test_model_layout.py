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
