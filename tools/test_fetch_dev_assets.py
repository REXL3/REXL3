"""fetch_dev_assets: which trees need assets and which tags it asks for. No network, no credentials."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import fetch_dev_assets as f
from tools import model_layout as m
from tools.test_release_evidence import write, encoded, sha
from tools import release_evidence as e

FAMILY, MODEL = 'qwen3.8', 'qwen3.8-27b'


def tree(root, family, model, version, schema='rexl3.profile-approval.v2', runtime_id='synthetic-demo'):
    d = Path(root) / 'runtimes' / family / model / f'v{version}'
    d.mkdir(parents=True)
    (d / 'approval.json').write_bytes(encoded({'schema': schema, 'runtime_id': runtime_id}))
    return d


def registry(version=1):
    return {'schema': 'rexl3.approval-registry.v1', 'approvals': {
        f'{FAMILY}/{MODEL}/v{version}': {'approval_sha256': 'a' * 64, 'source_commit': 'b' * 40,
                                         'source_tree': 'c' * 40, 'runtime_id': 'synthetic-demo'}}}


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_v2_tree_asks_for_the_tag_derived_from_the_registry_runtime_id(self):
        tree(self.root, FAMILY, MODEL, 2)
        with patch.object(f.p, 'download_release') as dl:
            self.assertEqual(f.fetch(self.root, Path(self.root) / 'a', registry(2), {'t': 2}),
                             ['approve-synthetic-demo-2'])
        args, kwargs = dl.call_args
        self.assertEqual(args[0], 'approve-synthetic-demo-2')
        self.assertEqual(args[1], Path(self.root) / 'a' / FAMILY / MODEL / 'v2')
        self.assertEqual(kwargs, {'registry': registry(2), 'approved_templates': {'t': 2},
                                  'key': f'{FAMILY}/{MODEL}/v2'})

    def test_the_candidate_cannot_name_its_own_tag(self):
        d = tree(self.root, FAMILY, MODEL, 2)
        (d / 'approval.json').write_bytes(encoded({'schema': 'rexl3.profile-approval.v2',
                                                   'approve_tag': 'approve-other-9', 'runtime_id': 'synthetic-demo'}))
        with patch.object(f.p, 'download_release') as dl:
            f.fetch(self.root, Path(self.root) / 'a', registry(2), {})
        self.assertEqual(dl.call_args[0][0], 'approve-synthetic-demo-2')

    def test_candidate_runtime_id_that_differs_from_the_registry_is_refused(self):
        tree(self.root, FAMILY, MODEL, 2, runtime_id='some-other-runtime')
        with patch.object(f.p, 'download_release') as dl:
            with self.assertRaises(ValueError):
                f.fetch(self.root, Path(self.root) / 'a', registry(2), {})
        dl.assert_not_called()

    def test_unsafe_family_or_model_is_refused_before_any_download(self):
        tree(self.root, 'Bad Family', MODEL, 1)
        with patch.object(f.p, 'download_release') as dl:
            with self.assertRaises(ValueError):
                f.fetch(self.root, Path(self.root) / 'a', registry(1), {})
        dl.assert_not_called()

    def test_missing_registry_entry_is_refused(self):
        tree(self.root, FAMILY, MODEL, 3)
        with self.assertRaises(ValueError):
            f.fetch(self.root, Path(self.root) / 'a', registry(1), {})

    def test_flat_layout_is_rejected(self):
        Path(self.root, 'runtimes/synthetic-demo/v1').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'old flat runtime layout rejected'):
            f.v2_trees(self.root)

    def test_download_failure_exits_content_free(self):
        tree(self.root, FAMILY, MODEL, 2)
        with patch.object(f, 'e') as e_mod:
            e_mod.load_registry.return_value = registry(2)
            e_mod.load_json.return_value = {'templates': {}}
            with patch.object(f, 'fetch', side_effect=ValueError('release download verification refused')):
                self.assertEqual(f.main(['--root', self.root, '--approved', 'x', '--trusted-approvals', 'y',
                                         '--trusted-approvals-sha256', 'z', '--assets-dir', self.root]), 2)


if __name__ == '__main__':
    unittest.main()
