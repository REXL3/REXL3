"""pr-rules: title, branch, scope and fragment mutations. No network: changed files are injected."""
import unittest
from unittest import mock

from tools import check_pr_rules as r

TOKEN = 'x' * 40
REPO = 'REXL3/REXL3'


def f(name, status='modified', patch=None):
    item = {'filename': name, 'status': status}
    if patch is not None:
        item['patch'] = patch
    return item


class TitleTest(unittest.TestCase):
    def check(self, title):
        return r.title_problems(title)

    def test_good_titles(self):
        for title in ('FEAT: Promoted qwen3.8-27b v1', 'CI: Made verify approve receipts from the protected base',
                      'DOCS: Rewrote the promotion guide', 'FIX: Fixed a crash', 'BUMP: Bumped checkout to v7'):
            self.assertEqual(self.check(title), [], title)

    def test_bad_titles(self):
        cases = {
            'fix: added check': 'not TYPE',
            'FEAT: adds a check': 'not past tense',
            'DOCS: Updated x.': 'trailing period',
            'DOC: Added x': 'not a type',
            'FEAT: ' + 'A' * 91: 'exceeds 90',
            'FEAT: ' + 'A' * 90: 'length',
            '': 'missing',
        }
        for title, reason in cases.items():
            with self.subTest(title=title):
                self.assertTrue(self.check(title), (title, reason))


class BranchTest(unittest.TestCase):
    def test_model_branches(self):
        self.assertEqual(r.classify_branch('model/qwen3.8-27b'), ('model', 'qwen3.8-27b'))
        self.assertEqual(r.classify_branch('model/qwen3.8-27b-approval'), ('model-approval', 'qwen3.8-27b'))
        self.assertEqual(r.classify_branch('model/qwen-approval'), ('model-approval', 'qwen'))
        self.assertIsNone(r.classify_branch('model/Qwen3.8')[0])
        self.assertIsNone(r.classify_branch('model/a/b')[0])
        self.assertIsNone(r.classify_branch('model/-qwen')[0])

    def test_engine_and_release_branches(self):
        self.assertEqual(r.classify_branch('engine/topic'), ('engine', 'topic'))
        self.assertEqual(r.classify_branch('engine/a-b-c'), ('engine', 'a-b-c'))
        self.assertIsNone(r.classify_branch('engine/Topic')[0])
        self.assertIsNone(r.classify_branch('engine/a_b')[0])
        self.assertEqual(r.classify_branch('release/v0.1.0'), ('release', 'v0.1.0'))
        self.assertIsNone(r.classify_branch('release/v0.1')[0])
        self.assertIsNone(r.classify_branch('docs/foo')[0])


class ScopeTest(unittest.TestCase):
    def check(self, head_ref, files, title='CI: Made checks report on every PR'):
        return r.check_pr_rules(REPO, 1, title, head_ref, TOKEN, files=files)

    def names(self, results, check):
        return [reason for name, ok, reason in results if name == check]

    def test_model_approval_scope_and_pin(self):
        files = [f('runtimes/templates/APPROVALS.json'),
                 f('.github/workflows/verify.yml', patch='@@ -30 +30 @@\n-  TRUSTED_APPROVALS_SHA256: 111\n+  TRUSTED_APPROVALS_SHA256: 222\n')]
        results = self.check('model/qwen3.8-27b-approval', files)
        self.assertEqual(self.names(results, 'scope'), [''])
        self.assertEqual(self.names(results, 'verify pin only'), [''])
        bad = [f('runtimes/templates/APPROVALS.json'), f('tools/verify_runtime_tree.py')]
        results = self.check('model/qwen3.8-27b-approval', bad)
        self.assertTrue(any('out of scope' in reason for reason in self.names(results, 'scope')))
        pin_bad = [f('.github/workflows/verify.yml', patch='@@ -30 +30 @@\n-  TRUSTED_APPROVALS_SHA256: 111\n+  jobs:\n+    verify:\n')]
        results = self.check('model/qwen3.8-27b-approval', pin_bad)
        self.assertTrue(any('pin line' in reason for reason in self.names(results, 'verify pin only')))

    def test_model_scope(self):
        files = [f('runtimes/qwen3.8/qwen3.8-27b/v1/approval.json'),
                 f('runtimes/qwen3.8/qwen3.8-27b/MODEL.json'),
                 f('docs/packs/qwen3.8-27b.md'),
                 f('changes/unreleased/qwen3.8-27b.md')]
        results = self.check('model/qwen3.8-27b', files, title='FEAT: Promoted qwen3.8-27b v1')
        self.assertEqual(self.names(results, 'scope'), [''])
        self.assertEqual(self.names(results, 'fragment'), [''])
        bad = files + [f('tools/check.py')]
        results = self.check('model/qwen3.8-27b', bad, title='FEAT: Promoted qwen3.8-27b v1')
        self.assertTrue(any('out of scope' in reason for reason in self.names(results, 'scope')))
        no_frag = [f('runtimes/qwen3.8/qwen3.8-27b/v1/approval.json')]
        results = self.check('model/qwen3.8-27b', no_frag, title='FEAT: Promoted qwen3.8-27b v1')
        self.assertTrue(any('must be present' in reason for reason in self.names(results, 'fragment')))

    def test_engine_scope_and_fragment(self):
        files = [f('tools/verify_runtime_tree.py'), f('changes/unreleased/engine.md')]
        results = self.check('engine/topic', files)
        self.assertEqual(self.names(results, 'scope'), [''])
        self.assertEqual(self.names(results, 'fragment'), [''])
        sealed = [f('runtimes/qwen3.8/qwen3.8-27b/v1/approval.json')]
        results = self.check('engine/topic', sealed)
        self.assertTrue(any('forbidden paths' in reason for reason in self.names(results, 'scope')))
        no_frag = [f('tools/verify_runtime_tree.py')]
        results = self.check('engine/topic', no_frag)
        self.assertTrue(any('fragment' in reason for reason in self.names(results, 'fragment')))
        docs = [f('docs/README.md')]
        results = self.check('engine/topic', docs, title='DOCS: Updated the readme')
        self.assertEqual(self.names(results, 'fragment'), ['not applicable (DOCS-only or no behavior change)'])

    def test_release_scope_and_fragment_deletions(self):
        files = [f('releases/v0.1.0/manifest.json'),
                 f('releases/v0.1.0/NOTES.md'),
                 f('CHANGELOG.md'),
                 f('changes/unreleased/qwen3.8-27b.md', status='removed')]
        results = self.check('release/v0.1.0', files)
        self.assertEqual(self.names(results, 'scope'), [''])
        add_frag = files[:-1] + [f('changes/unreleased/new.md')]
        results = self.check('release/v0.1.0', add_frag)
        self.assertTrue(any('must be a deletion' in reason for reason in self.names(results, 'scope')))
        out = files + [f('tools/check.py')]
        results = self.check('release/v0.1.0', out)
        self.assertTrue(any('out of scope' in reason for reason in self.names(results, 'scope')))


class VerifyYmlPinTest(unittest.TestCase):
    """The no-patch fallback must compare the PR base against the branch head (never the same ref)."""

    BASE = '  TRUSTED_APPROVALS_SHA256: 111' + '1' * 64 + '\n'
    HEAD_PIN = '  TRUSTED_APPROVALS_SHA256: 222' + '2' * 64 + '\n'

    def _run(self, head_content, base_content=None):
        files = [{'filename': '.github/workflows/verify.yml', 'status': 'modified', 'sha': 'x' * 40}]
        pr_doc = {'base': {'sha': 'b' * 40}, 'head': {'ref': 'model/m-approval'}}

        def fake_api_json(path, token, **kwargs):
            if path.startswith('/repos/REXL3/REXL3/pulls/'):
                return pr_doc
            raise AssertionError('unexpected API call: ' + path)

        def fake_file_at(repo, path, ref, token):
            if ref == 'b' * 40:
                return base_content if base_content is not None else self.BASE
            if ref == 'model/m-approval':
                return head_content
            raise AssertionError('unexpected file ref: ' + str(ref))

        with mock.patch.object(r, 'api_json', side_effect=fake_api_json), \
             mock.patch.object(r, 'file_at', side_effect=fake_file_at):
            return r.verify_yml_diff_ok(files, REPO, 1, TOKEN)

    def test_pin_only_change_passes(self):
        ok, reason = self._run(self.HEAD_PIN)
        self.assertTrue(ok, reason)

    def test_non_pin_change_fails(self):
        head = self.HEAD_PIN + 'jobs:\n  verify:\n'
        ok, reason = self._run(head)
        self.assertFalse(ok)
        self.assertIn('beyond the TRUSTED_APPROVALS_SHA256 pin line', reason)

    def test_head_unreadable_fails(self):
        ok, reason = self._run(None)
        self.assertFalse(ok)
        self.assertIn('unreadable', reason)


if __name__ == '__main__':
    unittest.main()
