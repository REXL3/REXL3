"""release_scope.py: classify a release manifest's model entries by qualification scope."""
import tempfile
import unittest
from pathlib import Path

from tools import release_scope as s
from tools.test_release_evidence import write, encoded

VERSION = 'v0.1.1'
HEX40 = '0' * 40
HEX64 = 'd' * 64
ENGINE_TAG = 'engine-v0.1.1-11111111'
DEV_COMMIT = '11111111' * 5


def manifest():
    return {'schema': 'rexl3.release-manifest.v1', 'version': VERSION,
            'engine': {'dev_commit': DEV_COMMIT, 'dev_tree': 'b' * 40,
                       'engine_release_tag': ENGINE_TAG,
                       'artifacts': {'rexl3-serve.tar.gz': HEX64}},
            'models': [{'family': 'qwen3.8', 'model': 'qwen3.8-27b', 'version': 'v1',
                        'runtime_id': 'synthetic-demo', 'approval_sha256': HEX64},
                       {'family': 'qwen3.8', 'model': 'qwen3.8-flash-next', 'version': 'v1',
                        'runtime_id': 'synthetic-demo', 'approval_sha256': HEX64,
                        'carried_forward': True,
                        'qualified_engine_release_tag': 'engine-v0.1.0-22222222',
                        'qualification_tier': 'T0', 'scope_diff_base': HEX40}],
            'notes': 'scope classification'}


class ClassifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write(self.root, f'releases/{VERSION}/manifest.json', encoded(manifest()))
        write(self.root, 'runtimes/qwen3.8/qwen3.8-27b/v1/approval.json',
              encoded({'source_commit': HEX40}))

    def test_classify_reports_t0_and_fresh_entries(self):
        out = s.classify(str(self.root), VERSION)
        self.assertIn(f'engine {ENGINE_TAG} dev_commit {DEV_COMMIT}', out)
        self.assertIn(f'qwen3.8/qwen3.8-27b v1: fresh scope_diff_base {HEX40} '
                      f'engine_release_tag {ENGINE_TAG}', out)
        self.assertIn('qwen3.8/qwen3.8-flash-next v1: T0 scope_diff_base '
                      f'{HEX40} engine_release_tag engine-v0.1.0-22222222', out)

    def test_classify_refuses_a_missing_sealed_tree(self):
        import shutil
        shutil.rmtree(self.root / 'runtimes/qwen3.8/qwen3.8-27b/v1')
        with self.assertRaisesRegex(ValueError, 'sealed tree missing'):
            s.classify(str(self.root), VERSION)

    def test_classify_refuses_an_invalid_manifest(self):
        doc = manifest()
        doc['engine']['dev_commit'] = 'a' * 40
        write(self.root, f'releases/{VERSION}/manifest.json', encoded(doc))
        with self.assertRaises(Exception):
            s.classify(str(self.root), VERSION)


if __name__ == '__main__':
    unittest.main()
