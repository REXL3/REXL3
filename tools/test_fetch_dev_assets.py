"""fetch_dev_assets: which trees need assets and which tags it asks for. No network, no credentials."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import fetch_dev_assets as f


def tree(root, runtime_id, version, schema):
    d = Path(root) / 'runtimes' / runtime_id / f'v{version}'
    d.mkdir(parents=True)
    (d / 'approval.json').write_text(json.dumps({'schema': schema, 'runtime_id': runtime_id}))
    return d


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_legacy_v1_needs_no_assets(self):
        tree(self.root, 'demo-runtime', 1, 'rexl3.profile-approval.v1')
        with patch.object(f.p, 'download_release') as dl:
            self.assertEqual(f.fetch(self.root, Path(self.root) / 'a', {}, {}), [])
        dl.assert_not_called()

    def test_v2_tree_asks_for_the_derived_tag_only(self):
        tree(self.root, 'demo-runtime', 2, 'rexl3.profile-approval.v2')
        with patch.object(f.p, 'download_release') as dl:
            self.assertEqual(f.fetch(self.root, Path(self.root) / 'a', {'r': 1}, {'t': 2}), ['approve-demo-runtime-2'])
        args, kwargs = dl.call_args
        self.assertEqual(args[0], 'approve-demo-runtime-2')
        self.assertEqual(args[1], Path(self.root) / 'a' / 'demo-runtime' / 'v2')
        self.assertEqual(kwargs, {'registry': {'r': 1}, 'approved_templates': {'t': 2}})

    def test_a_candidate_cannot_name_its_own_tag(self):
        d = tree(self.root, 'demo-runtime', 2, 'rexl3.profile-approval.v2')
        (d / 'approval.json').write_text(json.dumps({'schema': 'rexl3.profile-approval.v2', 'approve_tag': 'approve-other-9'}))
        with patch.object(f.p, 'download_release') as dl:
            f.fetch(self.root, Path(self.root) / 'a', {}, {})
        self.assertEqual(dl.call_args[0][0], 'approve-demo-runtime-2')

    def test_unsafe_runtime_id_is_refused_before_any_download(self):
        tree(self.root, 'Bad Id', 1, 'rexl3.profile-approval.v2')
        with patch.object(f.p, 'download_release') as dl:
            with self.assertRaises(ValueError):
                f.fetch(self.root, Path(self.root) / 'a', {}, {})
        dl.assert_not_called()

    def test_download_failure_exits_content_free(self):
        tree(self.root, 'demo-runtime', 2, 'rexl3.profile-approval.v2')
        with patch.object(f, 'e') as e_mod:
            e_mod.load_registry.return_value = {}
            e_mod.load_json.return_value = {'templates': {}}
            with patch.object(f, 'fetch', side_effect=ValueError('release download verification refused')):
                self.assertEqual(f.main(['--root', self.root, '--approved', 'x', '--trusted-approvals', 'y',
                                         '--trusted-approvals-sha256', 'z', '--assets-dir', self.root]), 2)


if __name__ == '__main__':
    unittest.main()
